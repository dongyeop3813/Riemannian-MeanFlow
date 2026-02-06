import jax.numpy as jnp
from jax import Array
import einops
from abc import ABC, abstractmethod
from typing import Type, Callable


class Manifold(ABC):

    EPS = {jnp.dtype(jnp.float32): 1e-4, jnp.dtype(jnp.float64): 1e-7}

    @abstractmethod
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

    @abstractmethod
    def log_map(self, p: Array, q: Array) -> Array:
        """
        Defines the logarithmic map from `p` to `q`.

        Parameters:
            - `p`, `q`: two points on the manifold of dimensions
                `(B, ..., D)`.

        Returns:
            The logarithmic map.
        """

    @abstractmethod
    def projx(self, x: Array) -> Array:
        """
        Projects the point `x` to the manifold.
        """

    @abstractmethod
    def proju(self, x: Array, u: Array) -> Array:
        """
        Projects the vector `u` to the tangent space of `x`.
        """

    @abstractmethod
    def geodesic_distance(self, p: Array, q: Array) -> Array:
        """
        Returns the geodesic distance of points `p` and `q` on the manifold.

        Parameters:
            - `p`, `q`: two points on the manifold of dimensions
                `(B, ..., D)`.

        Returns:
            The geodesic distance.
        """

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
        t = t.expand_dims(-1)
        x_t = self.exp_map(x_0, t * self.log_map(x_0, x_1))
        return self.projx(x_t)

    def integrate(
        self,
        x_0: Array,
        model: Callable,
        steps: int,
        tangent: bool = True,
        stochastic: bool = False,
    ) -> Array:
        """
        Applies Euler integration on the manifold for the field defined
        by `model`.

        Parameters:
            - `x_0`: the starting point;
            - `model`: the field;
            - `steps`: the number of steps;
            - `tangent`: if `True`, performs tangent Euler integration;
                otherwise performs classical Euler integration.
        """
        assert not stochastic, "not implemented"

        dt = jnp.array(1.0 / steps)
        x = x_0
        t = jnp.zeros((x.shape[0], 1))
        for _ in range(steps):
            if tangent:
                x = self.exp_map(x, model(x=x, t=t) * dt)
            else:
                x = x + model(x=x, t=t) * dt
                x = self.projx(x)
            t += dt
        return x

    def pairwise_geodesic_distance(
        self,
        x_0: Array,
        x_1: Array,
    ) -> Array:
        n, prods, d = x_0.shape

        x_0 = einops.rearrange(x_0, "b c d -> b (c d)", c=prods, d=d)
        x_1 = einops.rearrange(x_1, "b c d -> b (c d)", c=prods, d=d)

        x_0 = x_0.repeat_interleave(n, dim=0)
        x_1 = x_1.repeat(n, 1)

        x_0 = einops.rearrange(x_0, "b (c d) -> b c d", c=prods, d=d)
        x_1 = einops.rearrange(x_1, "b (c d) -> b c d", c=prods, d=d)

        distances = self.geodesic_distance(x_0, x_1) ** 2
        return distances.reshape(n, n)

    @abstractmethod
    def metric(self, x: Array, u: Array, v: Array) -> Array:
        """
        Calculates the Riemannian metric at point `x` between
        `u` and `v`.
        """

    def square_norm_at(self, x: Array, v: Array) -> Array:
        """
        Calculates the square of the norm of `v` at the tangent space of `x`.
        """
        return self.metric(x, v, v)

    @abstractmethod
    def parallel_transport(self, p: Array, q: Array, v: Array) -> Array:
        """
        Calculates the parallel transport of `v` in the tangent plane of `p`
        to that of `q`.

        Parameters:
            - `p`: starting point;
            - `q`: end point;
            - `v`: the vector to transport.
        """

    def uniform_prior(self, batch_size: int, d: int, key: Array) -> Array:
        """
        Returns samples from a uniform prior on the manifold.
        """
        pass
