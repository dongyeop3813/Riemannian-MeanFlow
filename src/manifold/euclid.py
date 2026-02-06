import jax.numpy as jnp
from jax import Array
import einops
from abc import ABC, abstractmethod
from typing import Type, Callable
from .manifold_type import Manifold


class EuclideanSpace(Manifold):

    EPS = {jnp.dtype(jnp.float32): 1e-7, jnp.dtype(jnp.float64): 1e-7}

    def exp_map(self, p: Array, v: Array) -> Array:
        """
        Defines the exponential map at `p` in the direction `v`.

        Parameters:
            - `p`: the point on the manifold at which the map should be taken,
                of dimensions `(B, ..., D)`.
            - `v`: the direction of the map, same dimensions as `p`.

        Returns:
            The exponential map.
        """
        return p + v

    def log_map(self, p: Array, q: Array) -> Array:
        """
        Defines the logarithmic map from `p` to `q`.

        Parameters:
            - `p`, `q`: two points on the manifold of dimensions
                `(B, ..., D)`.

        Returns:
            The logarithmic map.
        """
        return q - p

    def projx(self, x: Array) -> Array:
        """
        Projects the point `x` to the manifold.
        """
        return x

    def proju(self, x: Array, u: Array) -> Array:
        """
        Projects the vector `u` to the tangent space of `x`.
        """
        return u

    def geodesic_distance(self, p: Array, q: Array) -> Array:
        """
        Returns the geodesic distance of points `p` and `q` on the manifold.

        Parameters:
            - `p`, `q`: two points on the manifold of dimensions
                `(B, ..., D)`.

        Returns:
            The geodesic distance.
        """
        return jnp.linalg.norm(q - p, axis=-1)

    def geodesic_interpolant(self, x_0: Array, x_1: Array, t: Array) -> Array:
        """
        Returns the geodesic interpolant at time `t`, i.e.,
        `exp_{x_0}(t log_{x_0}(x_1))`.

        Parameters:
            - `x_0`, `x_1`: two points on the manifold of dimensions
                `(B, ..., D)`.
            - `t`: the time tensor of dimensions `(B, 1)`.

        Returns:
            The geodesic interpolant at time `t`.
        """
        t = t[:, None]
        return x_0 + t * (x_1 - x_0)

    def metric(self, x: Array, u: Array, v: Array) -> Array:
        """
        Calculates the Riemannian metric at point `x` between
        `u` and `v`.
        """
        return jnp.sum(u * v, axis=-1)

    def parallel_transport(self, p: Array, q: Array, v: Array) -> Array:
        """
        Calculates the parallel transport of `v` in the tangent plane of `p`
        to that of `q`.

        Parameters:
            - `p`: starting point;
            - `q`: end point;
            - `v`: the vector to transport.
        """
        return v
