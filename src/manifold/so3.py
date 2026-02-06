from .manifold_type import Manifold

import numpy as np

import jax.numpy as jnp
from jax import Array
from functools import partial


class SO3(Manifold):
    """
    Defines a SO(3) manifold.

    This class implements a SO(3) manifold using JAX,
    providing methods for exponential/logarithmic maps, geodesic distances, and other manifold operations.

    A point on the SO(3) manifold is expressed as a rotation matrix.
    A tangent vector on the SO(3) manifold is expressed as a skew-symmetric matrix.
    """

    def lie_log(self, r: Array) -> Array:
        pass

    def lie_exp(self, v: Array) -> Array:
        pass

    def projx(self, x: Array) -> Array:
        """
        Project points x to the SO(3) manifold.
        Here, x is a 3x3 orthogonal matrix with determinant 1.
        """
        return x

    def proju(self, x: Array, u: Array) -> Array:
        """
        Project tangent vectors u to the SO(3) manifold.
        Here, u is a 3x3 skew-symmetric matrix such that u + u^T = 0.
        """
        return (u + u.T) / 2

    def exp_map(self, p: Array, v: Array) -> Array:
        """
        Exponential map on the SO(3) manifold.

        Parameters:
        -----------
        p : Array
            Point on the SO(3) manifold, shape (B..., 3, 3)
        v : Array
            Tangent vector, shape (B..., 3, 3)

        Returns:
        --------
        Array
            Exponential map result, shape (B..., 3, 3)
        """
        pass

    def log_map(self, p: Array, q: Array) -> Array:
        pass

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
        pass

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
        pass

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
