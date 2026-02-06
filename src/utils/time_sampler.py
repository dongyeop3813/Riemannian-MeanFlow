import jax.numpy as jnp
from jax import Array, random as jr

import flax.nnx as nnx

from abc import ABC, abstractmethod


class TimeSampler(ABC):
    @abstractmethod
    def sample(self, batch_size: int) -> Array:
        pass


class UniformTimeSampler(TimeSampler):
    def __init__(self, seed: int, max_t: float = 1.0):
        self.rng = nnx.Rngs(seed)
        self.max_t = max_t

    def sample(self, batch_size: int) -> Array:
        return jr.uniform(self.rng(), (batch_size,)) * self.max_t


class LogitNormalTimeSampler(TimeSampler):
    def __init__(
        self, seed: int, mu: float = -0.4, sigma: float = 1.0, max_t: float = 1.0
    ):
        self.rng = nnx.Rngs(seed)
        self.mu = mu
        self.sigma = sigma
        self.max_t = max_t

    def sample(self, batch_size: int) -> Array:
        normal_samples = self.mu + self.sigma * jr.normal(self.rng(), (batch_size,))
        t = 1 / (1 + jnp.exp(-normal_samples))
        return t * self.max_t


class UnorderedIntervalSampler(TimeSampler):
    def __init__(
        self,
        time_sampler: TimeSampler,
        boundary_ratio: float = 0.0,
        static_layout: bool = False,
    ):
        self.time_sampler = time_sampler
        self.boundary_ratio = boundary_ratio
        self.static_layout = static_layout

        # Share the same rng with the time sampler
        self.rng = time_sampler.rng

    def sample_boundary_mask(self, batch_size: int) -> Array:
        if self.static_layout:
            num_boundary = int(batch_size * self.boundary_ratio)
            is_boundary = jnp.concatenate(
                [jnp.ones((num_boundary,)), jnp.zeros((batch_size - num_boundary,))]
            )
        else:
            is_boundary = jr.bernoulli(self.rng(), self.boundary_ratio, (batch_size,))
        return is_boundary

    def sample(self, batch_size: int) -> tuple[Array, Array]:
        is_boundary = self.sample_boundary_mask(batch_size)

        t = self.time_sampler.sample(batch_size)
        s = self.time_sampler.sample(batch_size)

        s = jnp.where(is_boundary, t, s)

        return t, s


class OrderedIntervalSampler(TimeSampler):
    def __init__(
        self,
        time_sampler: TimeSampler,
        boundary_ratio: float = 0.75,
        static_layout: bool = False,
    ):
        self.time_sampler = time_sampler
        self.boundary_ratio = boundary_ratio
        self.static_layout = static_layout

        # Share the same rng with the time sampler
        self.rng = time_sampler.rng

    def sample_boundary_mask(self, batch_size: int) -> Array:
        if self.static_layout:
            num_boundary = int(batch_size * self.boundary_ratio)
            is_boundary = jnp.concatenate(
                [jnp.ones((num_boundary,)), jnp.zeros((batch_size - num_boundary,))]
            )
        else:
            is_boundary = jr.bernoulli(self.rng(), self.boundary_ratio, (batch_size,))
        return is_boundary

    def sample(self, batch_size: int) -> tuple[Array, Array]:
        is_boundary = self.sample_boundary_mask(batch_size)

        t_raw = self.time_sampler.sample(batch_size)
        s_raw = self.time_sampler.sample(batch_size)

        t = jnp.minimum(t_raw, s_raw)
        s = jnp.maximum(t_raw, s_raw)

        t = jnp.where(is_boundary, t_raw, t)
        s = jnp.where(is_boundary, t_raw, s)

        return t, s


class OrderedThreePointSampler(TimeSampler):
    def __init__(
        self,
        time_sampler: TimeSampler,
        boundary_ratio: float = 0.75,
        static_layout: bool = False,
        mid_point: bool = False,
        min_length: float = 0.0,
    ):
        self.time_sampler = time_sampler
        self.rng = time_sampler.rng
        self.mid_point = mid_point
        self.min_length = min_length

        self.interval_sampler = OrderedIntervalSampler(
            time_sampler,
            boundary_ratio,
            static_layout,
        )

    def sample(self, batch_size: int) -> tuple[Array, Array, Array]:
        t, r = self.interval_sampler.sample(batch_size)

        if self.min_length > 0.0:
            too_short = r - t < self.min_length
            r_adjusted = t + self.min_length

            exceeds_max = r_adjusted > 1.00
            t_adjusted = 1.00 - self.min_length

            t = jnp.where(too_short & exceeds_max, t_adjusted, t)
            r = jnp.where(
                too_short & exceeds_max, 1.00, jnp.where(too_short, r_adjusted, r)
            )
            still_too_short = r - t < self.min_length
            t = jnp.where(still_too_short, 1.00 - self.min_length, t)
            r = jnp.maximum(r, t + self.min_length)

        if self.mid_point:
            w = 0.5
        else:
            w = jr.uniform(self.rng(), (batch_size,))

        s = w * r + (1 - w) * t

        return t, s, r


class UnorderedThreePointSampler(TimeSampler):
    def __init__(
        self,
        time_sampler: TimeSampler,
    ):
        self.time_sampler = time_sampler
        self.rng = time_sampler.rng

    def sample(self, batch_size: int) -> tuple[Array, Array, Array]:
        t = self.time_sampler.sample(batch_size)
        r = self.time_sampler.sample(batch_size)
        s = self.time_sampler.sample(batch_size)

        return t, s, r
