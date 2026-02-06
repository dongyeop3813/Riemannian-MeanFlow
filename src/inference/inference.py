import torch
import numpy as np
from pathlib import Path
import argparse

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx
from omegaconf import OmegaConf

from hydra.utils import instantiate
from tqdm import tqdm

import rootutils

rootutils.setup_root(Path.cwd(), indicator=".project-root", pythonpath=True, cwd=True)

from src.model import *
from src.utils import *
from src.data import *
from src.eval import *
from src.factory import *
from src.probability_path import *
from src.manifold import *

sei_eval = SeiEval()
sei_eval.to_device("gpu")  # Move Sei to GPU for faster inference
# reward_fn = sei_eval.create_reward_fn(target_signal=None)


def make_reward_fn_without_x1(reward_fn):
    """Create a reward function that does not use x_1."""

    def calc_reward(x):
        reward = reward_fn(x)
        return reward.sum()

    return calc_reward


def main(
    ckpt_dir: str,
    use_x1: bool = False,
    max_batches: int = None,
    seed: int = 0,
    guidance_scale: float = 1.0,
    n_steps: int = 1,
    setting: str = None,
    scale_by_t: bool = False,
    ckpt_path: str = "ckpts/best_dna_ckpt/default",
    config_path: str = ".hydra/config.yaml",
):
    """
    Run inference over the validation dataloader with reward-guided sampling.

    Args:
        use_x1: If True, use a reward function that incorporates x_1 (target data).
                If False, use a reward function without x_1.
        max_batches: Maximum number of batches to process. If None, process all.
        seed: Random seed for reproducibility.
        guidance_scale: Scale factor for reward guidance.
        setting: Optional setting flag to pass to sampling method.
        scale_by_t: If True, multiply grad_reward by t (scales guidance with timestep).
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    key = jr.PRNGKey(seed)

    # Paths
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / ckpt_path
    config_path = ckpt_dir / config_path

    # Load config
    cfg = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(cfg))

    manifold = instantiate(cfg.manifold)
    key, prior_key = jr.split(key)
    prior = UniformPrior(manifold, prior_key)
    x_shape = tuple(cfg.x_shape)

    cfg.data.val.split = "test"

    val_dataset = instantiate(cfg.data.val)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    print(f"Manifold: {manifold}")
    print(f"x_shape: {x_shape}")

    # Create model and optimizer
    model = init_flow_map_model(cfg.model, manifold, x_shape, seed=cfg.seed)
    optimizer = init_optim(cfg.optim, model, 1000)

    print(f"Model created: {type(model).__name__}")
    print(f"Optimizer created: {type(optimizer).__name__}")

    # Load checkpoint
    model, optimizer, epoch = restore_ckpt(ckpt_path, model, optimizer)
    print(f"Loaded checkpoint from epoch {epoch}")

    # JIT compile the model
    model.model = nnx.jit(model.model)

    all_outputs = []
    all_rewards = []  # Track rewards for each batch (sum)
    all_rewards_normalized = []  # Track rewards normalized by batch size
    batch_sizes = []  # Track batch sizes
    total_batches = (
        len(val_dataloader)
        if max_batches is None
        else min(max_batches, len(val_dataloader))
    )
    init_rewards = []  # Track rewards for each batch (sum)
    init_rewards_normalized = []  # Track rewards normalized by batch size
    final_rewards = []  # Track rewards for each batch (sum)
    final_rewards_normalized = []  # Track rewards normalized by batch size
    all_preds = []
    all_tgts = []

    for batch_idx, data in enumerate(
        tqdm(val_dataloader, desc="Processing batches", total=total_batches)
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_1, signal = data
        x_1, signal = prepare_promoter_data(x_1, signal)

        # Sample from prior with deterministic key for reproducibility
        key, sample_key = jr.split(key)
        batch_size = x_1.shape[0]
        x_0 = prior.sample(batch_size, *x_shape)

        # Create reward function based on whether we use x_1
        if use_x1:
            sei_reward_fn = sei_eval.create_reward_fn(target=x_1, signal=signal)
        # reward_fn = make_reward_fn_with_x1(x_1)
        else:
            sei_reward_fn = sei_eval.create_reward_fn(target=None)
        reward_fn = make_reward_fn_without_x1(sei_reward_fn)

        x_0_onehot = jnp.argmax(x_0, axis=-1)
        x_0_onehot = jnp.eye(4)[x_0_onehot]
        init_reward = reward_fn(x_0_onehot)
        init_rewards.append(init_reward)
        init_rewards_normalized.append(init_reward / batch_size)

        # Reward-guided sampling
        output, reward_history = model.sample_x1_with_reward_guidance(
            x_0,
            n_steps=n_steps,
            reward_fn=reward_fn,
            guidance_scale=guidance_scale,
            return_rewards=True,
            signal=signal,
            setting=setting,
            scale_by_t=scale_by_t,
        )
        projected_output = sphere_to_simplex(manifold.projx(output))
        x_pred_onehot = jnp.argmax(projected_output, axis=-1)
        x_pred_onehot = jnp.eye(4)[x_pred_onehot]
        final_reward = reward_fn(x_pred_onehot)

        final_rewards.append(final_reward)
        final_rewards_normalized.append(final_reward / batch_size)

        all_outputs.append(output)
        all_preds.append(x_pred_onehot)
        all_tgts.append(x_1)
        all_rewards.append(reward_history)
        all_rewards_normalized.append(reward_history / batch_size)
        batch_sizes.append(batch_size)

        print(
            f"Batch {batch_idx}: input shape {x_1.shape}, output shape {output.shape}"
        )
        for i in range(len(reward_history)):
            print(f"      Rewards step {i}: {reward_history[i] / batch_size:.4f}")
        print(
            f"  Rewards summary: initial={init_reward / batch_size:.4f}, final={final_reward / batch_size:.4f}"
        )

    # Stack all outputs
    all_outputs = jnp.concatenate(all_outputs, axis=0)
    all_preds = jnp.concatenate(all_preds, axis=0)
    all_tgts = jnp.concatenate(all_tgts, axis=0)

    all_rewards = jnp.stack(all_rewards, axis=0)  # (num_batches, n_steps + 1)
    all_rewards_normalized = jnp.stack(
        all_rewards_normalized, axis=0
    )  # (num_batches, n_steps + 1)
    batch_sizes = jnp.array(batch_sizes)
    init_rewards = jnp.array(init_rewards)
    init_rewards_normalized = jnp.array(init_rewards_normalized)
    final_rewards = jnp.array(final_rewards)
    final_rewards_normalized = jnp.array(final_rewards_normalized)
    print("\n===============================")
    print(f"Final reward across all batches: {final_rewards_normalized.mean()}")

    # Save results to file
    output_dir = Path("/network/scratch/m/marta.skreta/meanflows/dna_outputs")
    output_dir.mkdir(exist_ok=True)
    setting_suffix = f"_setting{setting}" if setting else ""
    scale_by_t_suffix = "_scalebyt" if scale_by_t else ""
    output_file = (
        output_dir
        / f"inference_seed{seed}_nsteps{n_steps}_guidance{guidance_scale}_x1{use_x1}{setting_suffix}{scale_by_t_suffix}.npz"
    )
    np.savez(
        output_file,
        outputs=np.array(all_outputs),
        rewards=np.array(all_rewards),
        rewards_normalized=np.array(all_rewards_normalized),
        batch_sizes=np.array(batch_sizes),
        seed=seed,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        use_x1=use_x1,
        setting=setting,
        scale_by_t=scale_by_t,
        init_rewards=init_rewards,
        init_rewards_normalized=init_rewards_normalized,
        final_rewards=final_rewards,
        final_rewards_normalized=final_rewards_normalized,
        all_preds=all_preds,
        all_tgts=all_tgts,
    )
    print(f"Saved results to {output_file}")

    return all_outputs, all_rewards, all_rewards_normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference over validation data")
    parser.add_argument(
        "--use-x1",
        action="store_true",
        help="Use a reward function that incorporates x_1 (target data)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="Guidance scale for reward-guided sampling",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="rX1",
        choices=["rX1", "rXt"],
        help="Optional setting flag to pass to sampling method",
    )
    parser.add_argument(
        "--scale-by-t",
        action="store_true",
        help="Multiply grad_reward by t (scales guidance with timestep)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Base directory for checkpoints and config (REQUIRED)",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="ckpts/best_dna_ckpt/default",
        help="Checkpoint path relative to ckpt_dir",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=".hydra/config.yaml",
        help="Config path relative to ckpt_dir",
    )
    args = parser.parse_args()

    main(
        use_x1=args.use_x1,
        max_batches=args.max_batches,
        seed=args.seed,
        guidance_scale=args.guidance,
        n_steps=args.n_steps,
        setting=args.setting,
        scale_by_t=args.scale_by_t,
        ckpt_dir=args.ckpt_dir,
        ckpt_path=args.ckpt_path,
        config_path=args.config_path,
    )
