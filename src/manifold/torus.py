from .manifold_type import Manifold

import jax.numpy as jnp
from jax import Array, random as jr
from jax.scipy.special import gammaln
from typing import Type


class Torus(Manifold):
    """
    A flat torus manifold.
    """

    def projx(self, x: Array) -> Array:
        return x % (2.0 * jnp.pi)

    def proju(self, x: Array, u: Array) -> Array:
        return u

    def exp_map(self, p: Array, v: Array) -> Array:
        return (p + v) % (2.0 * jnp.pi)

    def log_map(self, p: Array, q: Array) -> Array:
        return jnp.arctan2(
            jnp.sin(q - p),
            jnp.cos(q - p),
        )

    def geodesic_distance(self, p: Array, q: Array) -> Array:
        diff = jnp.abs(q - p)
        return jnp.minimum(diff, 2.0 * jnp.pi - diff).sum(axis=-1)

    def all_belong(self, x: Array) -> bool:
        return bool((x >= 0).all()) and bool((x < 2.0 * jnp.pi).all())

    def all_belong_tangent(self, p: Array, v: Array) -> bool:
        return True

    def uniform_prior(self, batch_size: int, d: int, key: Array) -> Array:
        return jr.uniform(key, (batch_size, d)) * 2.0 * jnp.pi

    def metric(self, p: Array, u: Array, v: Array) -> Array:
        return (u * v).sum(axis=-1)

    def parallel_transport(self, p: Array, q: Array, v: Array) -> Array:
        return v
