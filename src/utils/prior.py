from typing import Type, Sequence
import jax.random as jr
import jax.numpy as jnp

from jax import Array
import flax.nnx as nnx

from src.manifold import Manifold


class UniformPrior:
    def __init__(self, manifold: Type[Manifold], seed: int):
        """
        Uniform prior on the manifold (esp. k-product of d-dimensional manifold).

        Args:
            manifold: Manifold to sample from.
            k: Number of products.
            d: Dimension of manifold.
            seed: Random seed.
        """

        self.manifold = manifold
        self.rng = nnx.Rngs(seed)

    def sample(self, batch_size: int, *xshape: Sequence[int]) -> Array:
        return self.manifold.uniform_prior(batch_size, *xshape, key=self.rng())

    def log_prob(self, x: Array, *xshape: Sequence[int]) -> Array:
        return self.manifold.log_prob(x, *xshape)


class EuclidGaussianPrior:
    """
    Gaussian prior on the Euclidean space.

    Args:
        d: Dimension of Euclidean space.
        seed: Random seed.
    """

    def __init__(self, seed: int):
        self.rng = nnx.Rngs(seed)

    def sample(self, batch_size: int, d: int) -> Array:
        return self.rng.normal((batch_size, d))
