import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
from typing import Sequence

from src.manifold import simplex_to_sphere


def extract_first_channel(signal):
    signal = signal[..., 0:1]
    return signal


def prepare_promoter_data(x_1: Array, signal: Array) -> Array:
    """
    Convert x_1 to a sphere representation and extract the first channel from signal.

    Args:
        x_1 (Array): Input array to be converted using simplex_to_sphere.
        signal (Array): Input signal array from which the first channel will be extracted.

    Returns:
        Tuple[Array, Array]: Tuple of (processed x_1, processed signal)
    """
    x_1 = jnp.asarray(x_1)
    x_1 = simplex_to_sphere(x_1)

    signal = extract_first_channel(signal)
    signal = jnp.asarray(signal)

    return x_1, signal


def compute_ppl(
    model: nnx.Module,
    path,
    prior,
    eval_ppl: nnx.metrics.Metric,
    x_shape: Sequence[int],
    rngs: nnx.Rngs,
    **kwargs,
):
    log_prob = lambda x: prior.log_prob(x, *x_shape)
    _, ll = model.log_likelihood(
        path.x_1,
        inference_steps=100,
        prior_log_prob=log_prob,
        tangent=False,
        div_mode="hutchinson",
        key=rngs(),
        x_ndim=len(x_shape),
        **kwargs,
    )

    eval_ppl.update(ppl=ll)
