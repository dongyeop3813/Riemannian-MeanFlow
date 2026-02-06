from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from .etc import add_prefix_to_keys


def t_stratified_loss(t: Array, loss: Array, prefix: str = "") -> dict[str, Array]:
    edges = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=t.dtype)
    B = edges.shape[0] - 1

    # bin index: right-closed on the last bin
    # searchsorted gives indices in [0..B], we map to [0..B-1]
    idx = jnp.searchsorted(edges, t, side="right") - 1
    idx = jnp.clip(idx, 0, B - 1)

    # sums and counts per bin
    num = jnp.bincount(idx, weights=loss, length=B)
    cnt = jnp.bincount(idx, length=B)
    means = jnp.where(cnt > 0, num / jnp.maximum(cnt, 1.0), 0.0)

    return {f"{prefix}loss_bin{i}": means[i] for i in range(B)}


def t_stratified_loss_v2(t: Array, loss: Array, prefix: str = "") -> dict[str, Array]:
    edges: Array = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # (B,)  B = number of bins
    lows = edges[:-1]
    highs = edges[1:]
    B = lows.shape[0]

    # (B, N) broadcasting to calculate bin membership
    t_row = t[None, :]  # (1, N)
    in_low = t_row >= lows[:, None]  # (B, N)

    # last bin includes right side
    is_last = (jnp.arange(B) == (B - 1))[:, None]  # (B, 1) -> (B, N) broadcasting
    in_high = (t_row < highs[:, None]) | (is_last & (t_row <= highs[:, None]))
    mask = in_low & in_high  # (B, N)

    w = mask.astype(loss.dtype)  # (B, N)
    # binwise (weighted) sum and count
    num = (w * loss[None, :]).sum(axis=1)  # (B,)
    count = w.sum(axis=1)  # (B,)

    # if count==0, return 0
    means = jnp.where(count > 0, num / jnp.maximum(count, 1.0), 0.0)

    return {f"{prefix}loss_bin{i}": means[i] for i in range(B)}


def make_train_step(loss_fn, has_aux=False):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=has_aux)

    if has_aux:

        def train_step(model, optimizer, loss_metric, batch):
            (loss, aux), grads = grad_fn(model, batch)
            loss_metric.update(loss=loss, **aux)
            optimizer.update(grads)
            return loss, optax.global_norm(grads)

    else:

        def train_step(model, optimizer, loss_metric, batch):
            loss, grads = grad_fn(model, batch)
            loss_metric.update(loss=loss)
            optimizer.update(grads)
            return loss, optax.global_norm(grads)

    return train_step


def weighted_loss(loss, p=0.0):
    if p == 0.0:
        return loss.mean()

    weight = 1 / ((loss + 1e-3) ** p)
    weight = jax.lax.stop_gradient(weight)
    return (weight * loss).mean()


def compute_loss_and_reset(metrics: dict[str, nnx.metrics.Metric] = None, **kwargs):
    if metrics is None:
        metrics = kwargs

    loss_val_dict = {}

    for name, metric in metrics.items():
        if isinstance(metric, nnx.metrics.MultiMetric):
            mode = "train" if name.startswith("train") else "val"
            metric_val_dict = add_prefix_to_keys(metric.compute(), mode)
            loss_val_dict.update(metric_val_dict)
            metric.reset()
        else:
            name = name.replace("train_", "train/").replace("val_", "val/")
            metric_val = metric.compute()
            loss_val_dict[name] = metric_val
            metric.reset()

    return loss_val_dict


def compute_metric_and_reset(metric: nnx.metrics.Metric):
    val = metric.compute()
    metric.reset()
    return val


def put_model(model, device):
    state = nnx.state(model)
    new_state = jax.device_put(state, device)
    nnx.update(model, new_state)


def put_device(model: nnx.Module, device: str):
    if device not in ("cpu", "gpu"):
        raise ValueError(f"Invalid device: {device}")

    devs = jax.devices(device)
    if not devs:
        raise RuntimeError(f"No {device.upper()} device available.")

    put_model(model, devs[0])
