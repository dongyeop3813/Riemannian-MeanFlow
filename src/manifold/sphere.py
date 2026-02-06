from .manifold_type import Manifold

import jax.numpy as jnp
from jax import Array, random as jr
from jax.scipy.special import gammaln
from typing import Type


class Sphere(Manifold):
    """
    Sphere manifold implementation compatible with Manifold abstract class.

    This class implements a orthant of sphere manifold in n-dimensional space,
    providing methods for exponential/logarithmic maps, geodesic distances,
    and other manifold operations using JAX.
    """

    def __init__(self, pos_orthant_only: bool = True):
        self.pos_orthant_only = pos_orthant_only

    def projx(self, x: Array) -> Array:
        """
        Project points x to the sphere.

        Parameters:
        -----------
        x : Array
            Points to project

        Returns:
        --------
        Array
            Projected points on sphere
        """
        if self.pos_orthant_only:
            x = jnp.maximum(x, 1e-6)
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

    def proju(self, x: Array, u: Array) -> Array:
        return u - (x * u).sum(axis=-1, keepdims=True) * x

    def unsafe_exp(self, p, v):
        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        return p * jnp.cos(norm_v) + v * jnp.sin(norm_v) / norm_v

    def exp_map(self, p: Array, v: Array) -> Array:
        """
        Exponential map on the sphere.

        Parameters:
        -----------
        p : Array
            Point on the sphere, shape (B, ..., D)
        v : Array
            Tangent vector, shape (B, ..., D)

        Returns:
        --------
        Array
            Exponential map result
        """
        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        cond = norm_v > self.EPS[p.dtype]

        # Arbitrary safe value
        safe_v = jnp.ones_like(v)

        # Double where trick to avoid NaN gradient issues
        exp = self.projx(self.unsafe_exp(p, jnp.where(cond, v, safe_v)))
        retr = self.projx(p + v)
        return jnp.where(cond, exp, retr)

    def log_map(self, p: Array, q: Array) -> Array:
        """
        Logarithmic map on the sphere.

        Parameters:
        -----------
        p : Array
            Starting point on the sphere
        q : Array
            End point on the sphere

        Returns:
        --------
        Array
            Logarithmic map result
        """
        eps = self.EPS[p.dtype]
        v = self.proju(p, q - p)
        dist = self.geodesic_distance(p, q)[..., None]

        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        cond = norm_v > eps

        return jnp.where(cond, v * dist / jnp.clip(norm_v, min=eps), v)

    def geodesic_distance(self, p: Array, q: Array) -> Array:
        """
        Compute geodesic distance between two points on the sphere.

        Parameters:
        -----------
        p, q : Array
            Points on the sphere

        Returns:
        --------
        Array
            Geodesic distance
        """

        cos = jnp.sum(p * q, axis=-1)
        perp = p - cos[..., None] * q
        sin = jnp.sqrt(jnp.sum(perp**2, axis=-1) + 1e-12)

        theta = jnp.arctan2(sin, cos)
        return theta

    def metric(self, x: Array, u: Array, v: Array) -> Array:
        """
        Riemannian metric at point x between u and v.

        Parameters:
        -----------
        x : Array
            Point on the sphere
        u, v : Array
            Tangent vectors

        Returns:
        --------
        Array
            Metric tensor evaluation
        """
        return jnp.sum(u * v, axis=-1)

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
        v = self.proju(p, v)
        return self.proju(q, v)

    def uniform_prior(self, batch_size: int, d: int, key: Array) -> Array:
        """
        Generate uniform random points on the sphere.

        Parameters:
        -----------
        batch_size : int
            Batch size
        d: int
            Dimension of sphere

        Returns:
        --------
        Array
            Uniform random points on sphere
        """
        x = jr.normal(key, (batch_size, d))
        x = jnp.abs(x)
        return self.projx(x)

    def log_prob(self, x: Array, d: int) -> Array:
        # Area of n-sphere
        neg_log_area = gammaln(d / 2)
        neg_log_area -= jnp.log(2) + (d / 2) * jnp.log(jnp.pi)

        # Divide by (2 ** d) to get area of one orthant
        neg_log_area += d * jnp.log(2)

        return jnp.full_like(x[..., 0], neg_log_area)

    def smooth_labels(self, labels: Array, mx: float = 0.98) -> Array:
        """
        Smooth labels on the sphere.

        Parameters:
        -----------
        labels : Array
            Input labels
        mx : float
            Smoothing parameter

        Returns:
        --------
        Array
            Smoothed labels
        """
        from .simplex import Simplex

        smooth_labels_on_simplex = Simplex().smooth_labels(labels, mx)
        return Simplex.send_to(smooth_labels_on_simplex, Sphere)

    @classmethod
    def send_to(cls, x: Array, m: Type["Manifold"]) -> Array:
        """
        Send points x to manifold m.

        Parameters:
        -----------
        x : Array
            Points to send
        m : Manifold
            Target manifold

        Returns:
        --------
        Array
            Points on target manifold
        """
        from .simplex import Simplex

        if m is Simplex:
            squared = jnp.square(x)
            return squared / jnp.sum(squared, axis=-1, keepdims=True)
        elif m is Sphere:
            return x
        else:
            raise ValueError(f"Cannot send points to {m}")

    def all_belong(self, x: Array) -> bool:
        """
        Check if all points belong to the sphere.

        Parameters:
        -----------
        x : Array
            Points to check

        Returns:
        --------
        bool
            True if all points belong to sphere
        """
        # Check if all points have unit norm
        norms = jnp.linalg.norm(x, axis=-1)
        return jnp.allclose(norms, 1.0, atol=1e-5, rtol=1e-5)

    def all_belong_tangent(self, x: Array, v: Array) -> bool:
        """
        Check if all tangent vectors belong to tangent space.

        Parameters:
        -----------
        x : Array
            Points on sphere
        v : Array
            Tangent vectors to check

        Returns:
        --------
        bool
            True if all vectors are tangent
        """
        # Check if all vectors are orthogonal to their base points
        inner_products = jnp.sum(x * v, axis=-1)
        return jnp.allclose(inner_products, 0.0, atol=1e-5, rtol=1e-5)

    def _safe_arccos(self, x: Array) -> Array:
        """
        Safe arccos with clipping to avoid numerical issues.

        Parameters:
        -----------
        x : Array
            Input tensor

        Returns:
        --------
        Array
            Safe arccos result
        """
        # Clip to valid range for arccos
        x = jnp.clip(x, -1.0 + 1e-8, 1.0 - 1e-8)
        return jnp.arccos(x)

    # def log_map(self, p: Array, q: Array) -> Array:
    #     """
    #     Logarithmic map on the sphere.

    #     Parameters:
    #     -----------
    #     p : Array
    #         Starting point on the sphere
    #     q : Array
    #         End point on the sphere

    #     Returns:
    #     --------
    #     Array
    #         Logarithmic map result
    #     """
    #     eps = self.EPS[p.dtype]
    #     v = self.proju(p, q - p)
    #     dist = self.geodesic_distance(p, q, keepdims=True)

    #     cond = dist > eps
    #     norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)

    #     return jnp.where(cond, v * dist / jnp.clip(norm_v, min=eps), v)

    # def geodesic_distance(self, p: Array, q: Array, keepdims: bool = False) -> Array:
    #     """
    #     Compute geodesic distance between two points on the sphere.

    #     Parameters:
    #     -----------
    #     p, q : Array
    #         Points on the sphere
    #     keepdims : bool
    #         Whether to keep dimensions

    #     Returns:
    #     --------
    #     Array
    #         Geodesic distance
    #     """
    #     inner = jnp.sum(p * q, axis=-1, keepdims=keepdims)
    #     inner = jnp.clip(inner, -1.0 + 1e-8, 1.0 - 1e-8)
    #     return jnp.arccos(inner)


class NSphere(Manifold):
    """
    Sphere manifold implementation compatible with Manifold abstract class.

    This class implements a full sphere manifold in n-dimensional space,
    providing methods for exponential/logarithmic maps, geodesic distances,
    and other manifold operations using JAX.
    """

    def projx(self, x: Array) -> Array:
        """
        Project points x to the sphere.

        Parameters:
        -----------
        x : Array
            Points to project

        Returns:
        --------
        Array
            Projected points on sphere
        """
        norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-12)
        return x / norm

    def proju(self, x: Array, u: Array) -> Array:
        return u - (x * u).sum(axis=-1, keepdims=True) * x

    def unsafe_exp(self, p, v):
        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        return p * jnp.cos(norm_v) + v * jnp.sin(norm_v) / norm_v

    def exp_map(self, p: Array, v: Array) -> Array:
        """
        Exponential map on the sphere.

        Parameters:
        -----------
        p : Array
            Point on the sphere, shape (B, ..., D)
        v : Array
            Tangent vector, shape (B, ..., D)

        Returns:
        --------
        Array
            Exponential map result
        """
        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        cond = norm_v > self.EPS[p.dtype]

        # Arbitrary safe value
        safe_v = jnp.ones_like(v)

        # Double where trick to avoid NaN gradient issues
        exp = self.unsafe_exp(p, jnp.where(cond, v, safe_v))
        retr = self.projx(p + v)
        return jnp.where(cond, exp, retr)

    def log_map(self, p: Array, q: Array) -> Array:
        """
        Logarithmic map on the sphere.

        Parameters:
        -----------
        p : Array
            Starting point on the sphere
        q : Array
            End point on the sphere

        Returns:
        --------
        Array
            Logarithmic map result
        """
        eps = self.EPS[p.dtype]
        v = self.proju(p, q - p)
        dist = self.geodesic_distance(p, q)[..., None]

        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        cond = norm_v > eps

        return jnp.where(cond, v * dist / jnp.clip(norm_v, min=eps), v)

    def geodesic_distance(self, p: Array, q: Array) -> Array:
        """
        Compute geodesic distance between two points on the sphere.

        Parameters:
        -----------
        p, q : Array
            Points on the sphere

        Returns:
        --------
        Array
            Geodesic distance
        """

        cos = jnp.sum(p * q, axis=-1)
        perp = p - cos[..., None] * q
        sin = jnp.sqrt(jnp.sum(perp**2, axis=-1) + 1e-12)

        theta = jnp.arctan2(sin, cos)
        return theta

    def metric(self, x: Array, u: Array, v: Array) -> Array:
        """
        Riemannian metric at point x between u and v.

        Parameters:
        -----------
        x : Array
            Point on the sphere
        u, v : Array
            Tangent vectors

        Returns:
        --------
        Array
            Metric tensor evaluation
        """
        return jnp.sum(u * v, axis=-1)

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
        v = self.proju(p, v)
        return self.proju(q, v)

    def uniform_prior(self, batch_size: int, d: int, key: Array) -> Array:
        """
        Generate uniform random points on the sphere.

        Parameters:
        -----------
        batch_size : int
            Batch size
        d: int
            Dimension of sphere

        Returns:
        --------
        Array
            Uniform random points on sphere
        """
        x = jr.normal(key, (batch_size, d))
        return self.projx(x)

    def log_prob(self, x: Array, d: int) -> Array:
        # Area of n-sphere
        neg_log_area = gammaln(d / 2)
        neg_log_area -= jnp.log(2) + (d / 2) * jnp.log(jnp.pi)

        # Divide by (2 ** d) to get area of one orthant
        neg_log_area += d * jnp.log(2)

        return jnp.full_like(x[..., 0], neg_log_area)

    def all_belong(self, x: Array) -> bool:
        """
        Check if all points belong to the sphere.

        Parameters:
        -----------
        x : Array
            Points to check

        Returns:
        --------
        bool
            True if all points belong to sphere
        """
        # Check if all points have unit norm
        norms = jnp.linalg.norm(x, axis=-1)
        return jnp.allclose(norms, 1.0, atol=1e-5, rtol=1e-5)

    def all_belong_tangent(self, x: Array, v: Array) -> bool:
        """
        Check if all tangent vectors belong to tangent space.

        Parameters:
        -----------
        x : Array
            Points on sphere
        v : Array
            Tangent vectors to check

        Returns:
        --------
        bool
            True if all vectors are tangent
        """
        # Check if all vectors are orthogonal to their base points
        inner_products = jnp.sum(x * v, axis=-1)
        return jnp.allclose(inner_products, 0.0, atol=1e-5, rtol=1e-5)

    def _safe_arccos(self, x: Array) -> Array:
        """
        Safe arccos with clipping to avoid numerical issues.

        Parameters:
        -----------
        x : Array
            Input tensor

        Returns:
        --------
        Array
            Safe arccos result
        """
        # Clip to valid range for arccos
        x = jnp.clip(x, -1.0 + 1e-8, 1.0 - 1e-8)
        return jnp.arccos(x)
