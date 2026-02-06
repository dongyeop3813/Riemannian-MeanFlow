from .manifold_type import Manifold

import numpy as np

import jax.numpy as jnp
from jax import Array
from functools import partial


class ProductManifold(Manifold):
    """
    Defines a product of homogeneous manifolds.

    This class implements a product manifold of a manifold using JAX,
    providing methods for exponential/logarithmic maps, geodesic distances, etc.

    Attributes:
    -----------
    manifold : Manifold
        Manifold to be producted
    """

    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    def projx(self, x: Array) -> Array:
        """
        Project points x to the product manifold.
        """
        return self.manifold.projx(x)

    def proju(self, x: Array, u: Array) -> Array:
        """
        Project tangent vectors u to the product manifold.
        """
        return self.manifold.proju(x, u)

    def exp_map(self, p: Array, v: Array) -> Array:
        """
        Exponential map on the product manifold.

        Parameters:
        -----------
        p : Array
            Point on the product manifold, shape (B, ..., D)
        v : Array
            Tangent vector, shape (B, ..., D)

        Returns:
        --------
        Array
            Exponential map result
        """
        return self.manifold.exp_map(p, v)

    def log_map(self, p: Array, q: Array) -> Array:
        return self.manifold.log_map(p, q)

    def geodesic_distance(self, p: Array, q: Array) -> Array:
        """
        Compute geodesic distance between two points on the product manifold.

        Parameters:
        -----------
        p, q : Array
            Points on the product manifold
        keepdims : bool
            Whether to keep dimensions

        Returns:
        --------
        Array
            Geodesic distance
        """
        # Compute distance for each component and sum
        distances = self.manifold.geodesic_distance(p, q)
        return jnp.sqrt(jnp.sum(distances**2, axis=-1))

    def metric(self, x: Array, u: Array, v: Array) -> Array:
        """
        Riemannian metric at point x between u and v.

        Parameters:
        -----------
        x : Array
            Point on the product manifold
        u, v : Array
            Tangent vectors

        Returns:
        --------
        Array
            Metric tensor evaluation
        """
        return self.manifold.metric(x, u, v).sum(axis=-1)

    def parallel_transport(self, p: Array, q: Array, v: Array) -> Array:
        """
        Parallel transport of vector v from p to q.

        Parameters:
        -----------
        p : Array
            Starting point
        q : Array
            End point
        v : Array
            Vector to transport

        Returns:
        --------
        Array
            Transported vector
        """
        return self.manifold.parallel_transport(p, q, v)

    def uniform_prior(self, batch_size: int, length: int, d: int, key: Array) -> Array:
        """
        Generate uniform random points on the product manifold.

        Parameters:
        -----------
        batch_size : int
            Batch size
        length : int
            Number of products

        Returns:
        --------
        Array
            Uniform random points on product manifold
        """
        prior_sample = self.manifold.uniform_prior(batch_size * length, d, key)
        return prior_sample.reshape(batch_size, length, d)

    def log_prob(self, x: Array, length: int, d: int) -> Array:
        return self.manifold.log_prob(x, d).sum(axis=-1)
