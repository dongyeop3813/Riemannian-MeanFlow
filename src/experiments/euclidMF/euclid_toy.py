import jax
import jax.numpy as jnp
from jax import jvp
import flax.nnx as nnx
import optax

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import wandb
from pathlib import Path
from torch.utils.data import DataLoader

import absl.logging as absl_logging

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.data import ToyThreeTokenDataset
from src.probability_path import LinearInterpolantProbPath
from src.model import MLP
from src.utils import (
    EuclidGaussianPrior,
    UniformTimeSamplerWithBoundary,
    save_ckpt,
    restore_ckpt,
)
from src.evaluation import *


@hydra.main(config_path="../config", config_name="euclid_toy.yaml", version_base="1.3")
def train(cfg):
    absl_logging.set_verbosity(absl_logging.ERROR)

    dataset = ToyThreeTokenDataset(
        cfg.data.dataset_size,
        probs=cfg.data.probs,
        seed=cfg.seed,
        transform_to_onehot=True,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )

    prior = EuclidGaussianPrior(cfg.seed)
    prob_path = LinearInterpolantProbPath()
    time_sampler = UniformTimeSamplerWithBoundary(cfg.seed, boundary_ratio=0.1)

    model = MLP(
        x_dim=cfg.model.x_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        rngs=nnx.Rngs(cfg.seed),
    )
    optimizer = nnx.Optimizer(model, optax.adam(cfg.optim.lr), wrt=nnx.Param)

    loss_metric = nnx.metrics.Average("loss")
    l2_metric = nnx.metrics.Average("l2")

    eval_data = dataset[: cfg.eval.num_samples]

    if cfg.get("resume", False):
        path = Path(cfg.ckpt_path)
        model, optimizer, start_epoch = restore_ckpt(path, model, optimizer)
    else:
        start_epoch = 0

    wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    for epoch in range(start_epoch, start_epoch + cfg.num_epochs):
        for x_1 in train_dataloader:
            bsz = x_1.shape[0]
            x_0 = prior.sample(bsz, 1, 3).squeeze()
            t, s = time_sampler.sample(bsz)
            path = prob_path.sample(x_0, x_1, t)

            train_step(model, optimizer, loss_metric, path, s)

        gen_data = gen_sample_from_model(model, prior, cfg.eval.num_samples)

        eval_step(gen_data, eval_data, l2_metric)

        loss_val = loss_metric.compute()
        l2_val = l2_metric.compute()

        log_step(epoch, loss_val, l2_val, gen_data)

        if epoch % cfg.save_interval == 0:
            save_ckpt(HydraConfig.get().runtime.output_dir, model, optimizer, epoch)

    wandb.finish()

    return l2_val


@nnx.jit
def train_step(model, optimizer, loss_metric, path, s):
    loss, grads = nnx.value_and_grad(loss_fn)(model, path, s)
    optimizer.update(grads)
    loss_metric.update(loss=loss)


def loss_fn(model, path, s):
    x_t = path.x_t
    v_t = path.dx_t
    t = path.t

    u_t, du_dt = jvp(model, (x_t, t, s), (v_t, jnp.ones_like(t), jnp.zeros_like(s)))
    u_tgt = (s - t)[:, None] * du_dt + v_t

    error = u_t - jax.lax.stop_gradient(u_tgt)
    loss = jnp.sum(error**2, axis=-1).mean()

    return loss


@nnx.jit
def eval_step(gen_data, eval_data, metrics):
    model_sample = jnp.argmax(gen_data, axis=-1)
    eval_sample = jnp.argmax(eval_data, axis=-1)
    metrics.update(l2=compute_histogram_l2(model_sample, eval_sample))


def gen_sample_from_model(model, prior, bsz):
    x0 = prior.sample(bsz, 1, 3).squeeze()
    model_x1 = model.forward_flow(x0, jnp.zeros(bsz), jnp.ones(bsz))
    return model_x1


def log_step(
    epoch,
    loss_val,
    l2_val,
    gen_data,
):
    print(f"Epoch {epoch}: Loss = {loss_val:.6f}, L2 = {l2_val:.6f}")
    wandb.log({"loss": loss_val, "l2_distance": l2_val}, step=epoch)

    hist_fig = plot_token_histogram(jnp.argmax(gen_data, axis=-1))
    sample_3d_fig = plot_3d_samples(gen_data)

    wandb.log(
        {
            "token_histogram": wandb.Image(hist_fig),
            "sample_3d": wandb.Image(sample_3d_fig),
        },
        step=epoch,
    )
    plt.close("all")


if __name__ == "__main__":
    train()
