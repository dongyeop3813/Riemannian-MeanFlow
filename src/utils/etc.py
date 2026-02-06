import jax
import jax.numpy as jnp
from jax import tree_util

import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from .checkpoint import restore_ckpt


def expand_to(t, n_dims):
    return t.reshape((-1,) + (1,) * (n_dims - 1))


def resume_run_if_needed(cfg, model, optimizer):
    if cfg.get("resume", False):
        path = Path(cfg.ckpt_path)
        model, optimizer, current_epoch = restore_ckpt(path, model, optimizer)
    else:
        current_epoch = 0
    return model, optimizer, current_epoch


def add_prefix_to_keys(dict, prefix: str):
    return {f"{prefix}/{key}": value for key, value in dict.items()}


def hydra_run_dir():
    return Path(HydraConfig.get().runtime.output_dir)


def should_eval(epoch: int, every_n_epochs: int) -> bool:
    return (epoch + 1) % every_n_epochs == 0


def split_pytree(pytree, split_idx: int):
    """
    Split the batch dimension of the leaves in a JAX pytree,
    and return two pytrees with the same structure.

    Args:
        pytree: JAX pytree (each leaf must be an array)
        split_idx: Index at which to split. If None, splits at half the batch size.

    Returns:
        tuple: (first_half_pytree, second_half_pytree)
    """

    # Function to get the first half of a leaf
    def get_first_half(leaf):
        return leaf[:split_idx]

    # Function to get the second half of a leaf
    def get_second_half(leaf):
        return leaf[split_idx:]

    # Create the two pytrees by mapping the split functions
    first_pytree = tree_util.tree_map(get_first_half, pytree)
    second_pytree = tree_util.tree_map(get_second_half, pytree)

    return first_pytree, second_pytree


def norm_clip(x, clip):
    """
    Applies norm clipping to the input array x along the last axis.

    Args:
        x: input array (jax.numpy array)
        clip: maximum allowed norm

    Returns:
        x scaled so that its norm does not exceed clip
    """
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    # Calculate scaling factor: 1.0 if norm <= clip, otherwise clip / norm
    scale = jnp.minimum(1.0, clip / (norm + 1e-8))
    return x * scale


def plot_loss_dist(t_values, losses):
    # Create time bins
    n_bins = 10
    t_min, t_max = t_values.min(), t_values.max()
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)

    # Bin the data
    bin_indices = np.digitize(t_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Group losses by time bins
    binned_losses = []
    bin_centers = []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            binned_losses.append(losses[mask])
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.violinplot(
        binned_losses,
        positions=bin_centers,
        widths=0.1,
        showmeans=True,
        showmedians=True,
    )
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Distribution Over Time (Violin Plot)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def concat(arrs):
    return jnp.concatenate(arrs, axis=0)
