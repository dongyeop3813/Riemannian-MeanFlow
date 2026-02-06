import jax
from jax import Array
from jax.sharding import SingleDeviceSharding
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint.args import StandardSave

import logging

from pathlib import Path


def _to_numpy(pytree):
    """Convert all JAX arrays to numpy arrays to remove sharding info."""

    def _convert(x):
        if isinstance(x, (jnp.ndarray, Array)):
            return np.asarray(x)
        return x

    return jax.tree_util.tree_map(_convert, pytree)


def _to_jax_on_device(pytree):
    """Convert numpy arrays back to JAX arrays on current device."""
    default_device = jax.local_devices()[0]
    sharding = SingleDeviceSharding(default_device)

    def _convert(x):
        if isinstance(x, np.ndarray):
            return jax.device_put(jnp.asarray(x), sharding)
        return x

    return jax.tree_util.tree_map(_convert, pytree)


def save_ckpt(
    path: Path | str,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    epoch: int,
):
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    _, model_state = nnx.split(model)

    payload = {
        "model_state": model_state,
        "opt_state": optimizer.opt_state,
        "step": optimizer.step,
        "epoch": epoch,
    }

    ocp.StandardCheckpointer().save(path / f"epoch={epoch}", payload)


def restore_ckpt(path: Path | str, model: nnx.Module, optimizer: nnx.Optimizer):
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    graphdef, model_state = nnx.split(model)

    # Convert to numpy to remove sharding info, allowing device-agnostic restore
    target = {
        "model_state": _to_numpy(model_state),
        "opt_state": _to_numpy(optimizer.opt_state),
        "step": _to_numpy(optimizer.step),
        "epoch": 0,
    }

    # Restore checkpoint (will return numpy arrays)
    restored = ocp.StandardCheckpointer().restore(path, target)

    # Convert back to JAX arrays on current device
    restored = _to_jax_on_device(restored)

    model = nnx.merge(graphdef, restored["model_state"])
    optimizer.opt_state = restored["opt_state"]
    optimizer.step = restored["step"]
    epoch = int(restored["epoch"])

    logging.info(f"Checkpoint loaded from {path}")
    logging.info(f"Resuming from epoch {epoch}")

    return model, optimizer, epoch


class Checkpoint:
    pass


class CheckpointEveryNEpochs(Checkpoint):
    def __init__(self, run_dir: Path | str, save_interval: int, max_to_keep: int):
        self.sample_dir = Path(run_dir) / "eval_samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = Path(run_dir) / "ckpts"
        self.checkpoint_manager = CheckpointManager(
            self.ckpt_dir,
            options=CheckpointManagerOptions(
                save_interval_steps=save_interval,
                max_to_keep=max_to_keep,
                step_format_fixed_length=4,
            ),
        )

    def save(self, epoch: int, model: nnx.Module, optimizer: nnx.Optimizer):
        _, model_state = nnx.split(model)
        payload = {
            "model_state": model_state,
            "opt_state": optimizer.opt_state,
            "step": optimizer.step,
            "epoch": epoch,
        }
        try:
            self.checkpoint_manager.save(epoch, args=StandardSave(payload))
        except Exception as e:
            # Log error but don't fail training if checkpoint save fails
            # This can happen due to race conditions when multiple jobs run simultaneously
            logging.warning(f"Failed to save checkpoint at epoch {epoch}: {e}")
            logging.warning("Training will continue, but checkpoint may not be saved.")

    def save_array(self, epoch: int, pred: Array, prefix: str = ""):
        jnp.save(self.sample_dir / f"{prefix}{epoch}.npy", pred)


class CheckpointOnBestValLoss(Checkpoint):
    def __init__(
        self,
        run_dir: Path | str,
        metric_to_monitor: nnx.metrics.Metric,
        max_to_keep: int,
    ):
        self.sample_dir = Path(run_dir) / "eval_samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = Path(run_dir) / "ckpts"

        self.metric_to_monitor = metric_to_monitor

        self.checkpoint_manager = CheckpointManager(
            self.ckpt_dir,
            options=CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                step_format_fixed_length=4,
                best_fn=lambda x: x.compute(),
                best_mode="min",
            ),
        )

    def save(self, epoch: int, model: nnx.Module, optimizer: nnx.Optimizer):
        _, model_state = nnx.split(model)
        payload = {
            "model_state": model_state,
            "opt_state": optimizer.opt_state,
            "step": optimizer.step,
            "epoch": epoch,
        }
        try:
            self.checkpoint_manager.save(
                epoch,
                args=StandardSave(payload),
                metrics=self.metric_to_monitor,
            )
        except Exception as e:
            # Log error but don't fail training if checkpoint save fails
            # This can happen due to race conditions when multiple jobs run simultaneously
            logging.warning(f"Failed to save checkpoint at epoch {epoch}: {e}")
            logging.warning("Training will continue, but checkpoint may not be saved.")

    def save_array(self, epoch: int, pred: Array):
        jnp.save(self.sample_dir / f"{epoch}.npy", pred)
