import os
from pathlib import Path
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import torch
import numpy as np

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"

import jax.numpy as jnp
import flax.nnx as nnx
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.model import *
from src.utils import *
from src.data import *
from src.eval import *
from src.factory import *
from src.probability_path import *
from src.manifold import *

import argparse


def do_sample(args):
    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = sample_dir.resolve()

    print(f"Saving samples to {sample_dir}")

    # Paths
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = ckpt_dir / Path(f"ckpts/{args.epoch:04d}/default")
    print(f"Checkpoint path: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} not found")

    config_path = ckpt_dir / Path(".hydra/config.yaml")

    # Load config
    cfg = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(cfg))

    manifold = instantiate(cfg.manifold)
    prior = UniformPrior(manifold, nnx.Rngs(args.seed)())
    x_shape = tuple(cfg.x_shape)

    print(f"Manifold: {manifold}")
    print(f"x_shape: {x_shape}")

    # Load model and optimizer from checkpoint
    model = init_flow_map_model(cfg.model, manifold, x_shape, seed=cfg.seed)
    optimizer = init_optim(cfg.optim, model, 1000)

    print(f"Model created: {type(model).__name__}")
    print(f"Optimizer created: {type(optimizer).__name__}")
    model, optimizer, epoch = restore_ckpt(ckpt_path, model, optimizer)
    print(f"Loaded checkpoint from epoch {epoch}")

    # Load dataset and dataloader
    cfg.data.val.split = args.split
    val_dataset = instantiate(cfg.data.val)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    def prepare_batch(data):
        x_1, signal = data
        x_1, signal = prepare_promoter_data(x_1, signal)
        cond = {"signal": signal}
        x_0 = prior.sample(*x_1.shape)

        return (x_0, x_1, cond)

    model.model = nnx.jit(model.model)

    samples = {
        1: [],
        2: [],
        4: [],
        8: [],
        16: [],
        32: [],
        64: [],
        100: [],
    }
    gt_samples = []

    for data in val_dataloader:
        (x_0, x_1, cond) = prepare_batch(data)
        gt_samples.append(x_1)

        for n in [1, 2, 4, 8, 16, 32, 64, 100]:
            if args.use_integration:
                x1 = manifold.projx(model.sample_x1_with_integration(x_0, n, **cond))
                x1 = sphere_to_simplex(x1)
            else:
                x1 = manifold.projx(model.sample_x1_with_n_steps(x_0, n, **cond))
                x1 = sphere_to_simplex(x1)
            samples[n].append(x1)

    # Concatenate samples
    samples = {k: jnp.concatenate(v, axis=0) for k, v in samples.items()}
    gt_samples = jnp.concatenate(gt_samples, axis=0)

    for n in samples:
        np.savez(
            sample_dir / f"n_steps_{n}.npz",
            model_sample=samples[n],
            test_data=gt_samples,
        )

    print(f"Saved samples to {sample_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument(
        "--use_integration",
        action="store_true",
        help="Use integration for sampling",
    )

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    do_sample(args)
