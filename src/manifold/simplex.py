from .manifold_type import Manifold

import jax.numpy as jnp
from jax import Array, random as jr
from typing import Type


class Simplex(Manifold):
    """
    Defines an n-simplex (representable in n - 1 dimensions).

    This class implements an n-simplex manifold using JAX,
    providing methods for exponential/logarithmic maps, geodesic distances,
    and other manifold operations.
    """

    def projx(self, x: Array) -> Array:
        x = jnp.maximum(x, 0.0)
        return x / jnp.sum(x, axis=-1, keepdims=True)

    def proju(self, x: Array, u: Array) -> Array:
        return u - u.mean(axis=-1, keepdims=True)

    def exp_map(self, p: Array, v: Array) -> Array:
        """
        Exponential map on the simplex.

        Parameters:
        -----------
        p : Array
            Point on the simplex, shape (B, ..., D)
        v : Array
            Tangent vector, shape (B, ..., D)

        Returns:
        --------
        Array
            Exponential map result
        """
        s = jnp.sqrt(p)
        xs = v / (2.0 * s)
        theta = jnp.linalg.norm(xs, axis=-1, keepdims=True)

        def exp_nonzero(s, xs, theta):
            return (jnp.cos(theta) * s + self._usinc(theta) * xs) ** 2

        def exp_zero(s, xs, theta):
            return s**2

        result = jnp.where(
            theta > 1e-8, exp_nonzero(s, xs, theta), exp_zero(s, xs, theta)
        )

        return self.projx(result)

    def log_map(self, p: Array, q: Array) -> Array:
        """
        Logarithmic map on the simplex.

        Parameters:
        -----------
        p : Array
            Starting point on the simplex
        q : Array
            End point on the simplex

        Returns:
        --------
        Array
            Logarithmic map result
        """
        z = jnp.sqrt(p * q)
        s = jnp.sum(z, axis=-1, keepdims=True)

        s = jnp.clip(s, 1e-8, 1.0 - 1e-8)

        log_result = 2.0 * self._safe_arccos(s) / jnp.sqrt(1.0 - s**2) * (z - s * p)

        return self.proju(p, log_result)

    def geodesic_distance(self, p: Array, q: Array) -> Array:
        """
        Compute geodesic distance between two points on the simplex.

        Parameters:
        -----------
        p, q : Array
            Points on the simplex
        keepdims : bool
            Whether to keep dimensions

        Returns:
        --------
        Array
            Geodesic distance
        """
        # Ensure both points are on the simplex
        p = self.projx(p)
        q = self.projx(q)

        # Compute d = sum(sqrt(p * q))
        d = jnp.sum(jnp.sqrt(p * q), axis=-1)

        # Clamp for numerical stability
        d = jnp.clip(d, 1e-8, 1.0 - 1e-8)

        # Compute geodesic distance
        arccos_d = 2.0 * self._safe_arccos(d)
        if arccos_d.ndim > 0:
            return jnp.sqrt(jnp.sum(arccos_d**2, axis=-1))
        else:
            return jnp.sqrt(arccos_d**2)

    def metric(self, x: Array, u: Array, v: Array) -> Array:
        """
        Riemannian metric at point x between u and v.

        Parameters:
        -----------
        x : Array
            Point on the simplex
        u, v : Array
            Tangent vectors

        Returns:
        --------
        Array
            Metric tensor evaluation
        """
        x_safe = jnp.maximum(x, 1e-8)
        return jnp.sum((u * v) / x_safe, axis=-1)

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
        Generate uniform random points on the simplex.

        Parameters:
        -----------
        batch_size : int
            Batch size
        d : int
            Dimension of simplex

        Returns:
        --------
        Array
            Uniform random points on simplex
        """
        alpha = jnp.ones((d,))
        samples = jr.dirichlet(key, alpha, shape=(batch_size,))

        return self.projx(samples)

    def smooth_labels(self, labels: Array, mx: float = 0.98) -> Array:
        """
        Smooth labels on the simplex.

        Parameters:
        -----------
        labels : Array
            Input labels (one-hot encoded)
        mx : float
            Smoothing parameter

        Returns:
        --------
        Array
            Smoothed labels
        """
        num_classes = labels.shape[-1]

        increase = (1.0 - mx) / (num_classes - 1)

        smooth_labels = jnp.full_like(labels, increase, dtype=jnp.float32)

        smooth_labels = jnp.where(labels == 1, mx, smooth_labels)

        return self.projx(smooth_labels)

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
        from .sphere import Sphere

        if m is Simplex:
            return x
        elif m is Sphere:
            return jnp.sqrt(x)
        else:
            raise ValueError(f"Cannot send points to {m}")

    def _usinc(self, theta: Array) -> Array:
        """
        Compute usinc(theta) = sin(theta) / theta for theta != 0, 1 for theta = 0.

        Parameters:
        -----------
        theta : Array
            Input tensor

        Returns:
        --------
        Array
            usinc(theta)
        """

        def usinc_nonzero(theta):
            return jnp.sin(theta) / theta

        def usinc_zero(theta):
            return jnp.ones_like(theta)

        return jnp.where(jnp.abs(theta) > 1e-8, usinc_nonzero(theta), usinc_zero(theta))

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
