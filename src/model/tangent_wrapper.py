import jax
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx
from jax import Array
from typing import Callable, Optional
from functools import partial

from ..manifold import Manifold, ProductManifold, simplex_to_sphere, sphere_to_simplex, Sphere
from ..utils import expand_to

from jax import random



class TangentWrapper(nnx.Module):
    def __init__(self, model: nnx.Module, manifold: Manifold):
        self.model = model
        self.manifold = manifold

    def __call__(self, x: Array, t: Array, *args, **kwargs) -> Array:
        v = self.model(x=x, t=t, *args, **kwargs)
        v = self.manifold.proju(x, v)
        return v

    def sample_x1(self, x_0: Array, *args, **kwargs) -> Array:
        return self.integrate(x_0, inference_steps=100, tangent=False, *args, **kwargs)

    def instant_velocity(self, x: Array, t: Array, *args, **kwargs) -> Array:
        return self(x, t, *args, **kwargs)

    def integrate(
        self,
        x_0: Array,
        inference_steps: int,
        tangent: bool = True,
        *args,
        **kwargs,
    ) -> Array:

        dt = jnp.array(1.0 / inference_steps)

        def body_fn(carry, _):
            x, t = carry
            v = self.instant_velocity(x=x, t=t, *args, **kwargs)
            if tangent:
                x_new = self.manifold.exp_map(x, v * dt)
            else:
                x_new = x + v * dt
                x_new = self.manifold.projx(x_new)
            t_new = t + dt
            return (x_new, t_new), None

        x_init = x_0
        t_init = jnp.zeros((x_0.shape[0],))
        (x_final, _), _ = jax.lax.scan(
            body_fn, (x_init, t_init), None, length=inference_steps
        )
        x = x_final
        return x

    def log_likelihood(
        self,
        x_1: Array,
        inference_steps: int,
        prior_log_prob: Callable,
        tangent: bool = True,
        div_mode: str = "exact",
        key: jr.PRNGKey = None,
        x_ndim: int = 1,
        *args,
        **kwargs,
    ) -> Array:

        dt = jnp.array(1.0 / inference_steps)
        t = jnp.ones((x_1.shape[0],))
        ll = jnp.zeros((x_1.shape[0],))

        eps = None
        if div_mode == "hutchinson":
            eps = (jr.randint(key, x_1.shape, 0, 2) * 2 - 1).astype(jnp.float32)

        def body_fn(carry, _):
            x, t, ll = carry
            vecfield = lambda x: self.instant_velocity(x=x, t=t, *args, **kwargs)
            v, div = output_and_div(
                vecfield, x, v=eps, div_mode=div_mode, x_ndim=x_ndim
            )

            if tangent:
                prev_x = self.manifold.exp_map(x, -v * dt)
            else:
                prev_x = self.manifold.projx(x - v * dt)

            return (prev_x, t - dt, ll - div * dt), None

        (x_0, _, ll), _ = jax.lax.scan(
            body_fn, (x_1, t, ll), None, length=inference_steps
        )
        ll = prior_log_prob(x_0) + ll
        return x_0, ll


class AverageVelocityWrapper(TangentWrapper):
    def __init__(
        self,
        model: nnx.Module,
        manifold: Manifold,
        clip_mode: bool = False,
        max_norm: float = jnp.pi / 2,
    ):
        self.model = model
        self.manifold = manifold
        self.clip_mode = clip_mode
        self.max_norm = max_norm

    def __call__(self, x: Array, t: Array, s: Array, *args, **kwargs) -> Array:
        v = self.model(x=x, t=t, s=s, *args, **kwargs)
        v = self.manifold.proju(x, v)
        if self.clip_mode:
            v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
            clipped_norm = jnp.clip(v_norm, a_max=self.max_norm)
            v = v * (clipped_norm / (v_norm + 1e-8))
        return v

    def forward_flow(self, x: Array, t: Array, s: Array, u_to_add: Optional[Array] = None, *args, **kwargs) -> Array:
        u = self(x, t, s, *args, **kwargs) ## v
        if u_to_add is not None:
            u = u + u_to_add

        t = expand_to(t, x.ndim)
        s = expand_to(s, x.ndim)

        return self.manifold.exp_map(x, (s - t) * u)

    def sample_x1(self, x_0: Array, *args, **kwargs) -> Array:
        return self.forward_flow(
            x_0, jnp.zeros(x_0.shape[0]), jnp.ones(x_0.shape[0]), *args, **kwargs
        )

    def sample_x1_with_n_steps(
        self, x_0: Array, n_steps: int, *args, **kwargs
    ) -> Array:
        ts = jnp.linspace(0, 1, n_steps + 1)
        ones = jnp.ones(x_0.shape[0])
        x = x_0
        for t, s in zip(ts[:-1], ts[1:]):
            x = self.forward_flow(x, ones * t, ones * s, *args, **kwargs)
        return x

    def sample_x1_with_reward_guidance(
        self,
        x_0: Array,
        n_steps: int,
        reward_fn: Callable[[Array], Array],
        guidance_scale: float = 1.0,
        return_rewards: bool = False,
        setting: str = None,
        scale_by_t: bool = False,
        *args,
        **kwargs,
    ) -> Array:
        """
        Sample x1 with reward guidance applied at each step.

        Args:
            x_0: Initial samples on the manifold (batch_size, seq_len, dim)
            n_steps: Number of integration steps
            reward_fn: Function that takes sphere representation and returns
                per-sample rewards (batch_size,). Should be differentiable.
            guidance_scale: Scale factor for the reward gradient guidance
            return_rewards: If True, return (samples, reward_history) tuple
            setting: Optional setting flag for different sampling configurations
            scale_by_t: If True, multiply grad_reward by t (scales guidance with timestep)
            *args, **kwargs: Additional arguments passed to forward_flow

        Returns:
            If return_rewards is False: Guided samples at t=1
            If return_rewards is True: (samples, reward_history) where reward_history
                is an array of shape (n_steps + 1, batch_size) containing rewards at each step
        """
        ts = jnp.linspace(0, 1, n_steps + 1)
        ones = jnp.ones(x_0.shape[0])
        x = x_0

        # Track rewards at each step
        reward_history = []


        for t, s in zip(ts[:-1], ts[1:]):
            def reward_x1(x):
                x_flowed = self.forward_flow(x, ones * t, ones * 1, *args, **kwargs)
                x_flowed = sphere_to_simplex(self.manifold.projx(x_flowed))
                rewards = reward_fn(x_flowed)
                return rewards, rewards
            def reward_xt(x):
                x = sphere_to_simplex(self.manifold.projx(x))
                rewards = reward_fn(x)
                return rewards, rewards

            if setting == "rX1":
                get_reward = reward_x1
                grad_reward, _ = jax.grad(get_reward, has_aux=True)(x)
                _, step_reward = reward_xt(x)

            elif setting == "rXt":
                get_reward = reward_xt
                grad_reward, step_reward = jax.grad(get_reward, has_aux=True)(x)

            if jnp.any(jnp.isnan(grad_reward)):
                nan_idx = jnp.where(jnp.isnan(grad_reward))[0]
                print("nan_idx", nan_idx)
                print("grad_reward", grad_reward[nan_idx])
                exit()
            
            reward_history.append(step_reward)

            # Project gradient to tangent space of manifold
            grad_reward = self.manifold.proju(x, grad_reward) * guidance_scale
            if scale_by_t:
                grad_reward = grad_reward * t
            x = self.forward_flow(x, ones * t, ones * s, u_to_add=grad_reward, *args, **kwargs)


        if return_rewards:
            reward_history = jnp.stack(reward_history, axis=0)  # (n_steps + 1, batch_size)
            return x, reward_history
        return x

    def sample_x1_with_integration(
        self, x_0: Array, steps: int = 100, *args, **kwargs
    ) -> Array:
        return self.integrate(x_0, inference_steps=steps, tangent=True, *args, **kwargs)

    def instant_velocity(self, x: Array, t: Array, *args, **kwargs) -> Array:
        return self(x, t, t, *args, **kwargs)


class CondVelocityParameterizedWrapper(AverageVelocityWrapper):
    def __init__(
        self,
        model: nnx.Module,
        manifold: Manifold,
        projection_mode: str = "default",
    ):
        self.model = model
        self.manifold = manifold
        self.projection_mode = projection_mode

    def __call__(self, x: Array, t: Array, s: Array, *args, **kwargs) -> Array:
        """
        Average velocity.
        """
        x1 = self.model(x=x, t=t, s=s, *args, **kwargs)

        if self.projection_mode == "default":
            x1 = self.manifold.projx(x1)
        elif self.projection_mode == "promoter":
            x1 = x1 / jnp.sqrt(jnp.sum(x1**2, axis=-1, keepdims=True) + 1e-6)
        elif self.projection_mode == "promoter_abs":
            x1 = jnp.abs(x1) / jnp.sqrt(jnp.sum(x1**2, axis=-1, keepdims=True) + 1e-6)
        elif self.projection_mode == "promoter_logit":
            x1 = nnx.softmax(x1, axis=-1)
            x1 = simplex_to_sphere(x1)
        else:
            raise ValueError(f"Invalid projection mode: {self.projection_mode}")

        v = self.manifold.log_map(x, x1) / jnp.clip(
            1.0 - expand_to(t, x.ndim), min=1e-6
        )
        return v


class FlowMapParameterizedWrapper(AverageVelocityWrapper):
    def __init__(
        self,
        model: nnx.Module,
        manifold: Manifold,
        clip_mode: bool = False,
        max_norm: float = jnp.pi / 2,
    ):
        self.model = model
        self.manifold = manifold
        self.clip_mode = clip_mode
        self.max_norm = max_norm

    def __call__(self, x: Array, t: Array, s: Array, *args, **kwargs) -> Array:
        x_s = self.forward_flow(x, t, s, *args, **kwargs)
        v = self.manifold.log_map(x, x_s) / jnp.clip(s - t, min=1e-6)[..., None]
        return v

    def instant_velocity(self, x: Array, t: Array, *args, **kwargs) -> Array:
        flow_fn = lambda s: self.forward_flow(x, t, s, *args, **kwargs)
        v = jax.jvp(flow_fn, (t,), (jnp.ones_like(t),))[1]
        v = self.manifold.proju(x, v)
        return v

    def forward_flow(self, x: Array, t: Array, s: Array, *args, **kwargs) -> Array:
        x_s = self.model(x, t, s, *args, **kwargs)
        x_s = self.manifold.projx(x_s)
        return x_s


def output_and_div(vecfield, x, v=None, div_mode="exact", x_ndim=1):
    # From: https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/model_pl.py#L45
    # Modified to use jax instead of torch
    def div_fn(u):
        """Accepts a function u:R^D -> R^D."""
        J = jax.jacrev(u)
        return lambda x: jnp.trace(J(x))

    if div_mode == "exact":
        dx = vecfield(x)
        div = jax.vmap(div_fn(vecfield))(x)
    elif div_mode == "hutchinson":
        dx, vjpfunc = jax.vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        # Sum over the last x_ndim axes to leave only the batch shape
        div = jnp.sum(vJ * v, axis=tuple(range(-x_ndim, 0)))
    else:
        raise ValueError(f"Invalid div_mode: {div_mode}")
    return dx, div


class SimplexFlowMapWrapper(TangentWrapper):
    """
    Wrapper for flow map on the simplex (instead of average velocity).
    The model output is interpreted as a point on the simplex,
    and we model the average velocity as log of the flow map.

    Model output is the logit of the simplex point, and we map it to the sphere.
    """

    def __init__(self, model: nnx.Module):
        self.model = model
        self.manifold = ProductManifold(Sphere())

    def forward_flow(self, x: Array, t: Array, s: Array, *args, **kwargs) -> Array:
        s_minus_t = expand_to(s - t, x.ndim)
        logit = jnp.log(x) + s_minus_t * self.model(x=x, t=t, s=s, *args, **kwargs)
        simplex_point = nnx.softmax(logit, axis=-1)
        sphere_point = simplex_to_sphere(simplex_point)
        return sphere_point

    def sample_x1(self, x_0: Array, *args, **kwargs) -> Array:
        return self.forward_flow(
            x_0, jnp.zeros(x_0.shape[0]), jnp.ones(x_0.shape[0]), *args, **kwargs
        )

    def sample_x1_with_n_steps(
        self, x_0: Array, n_steps: int, *args, **kwargs
    ) -> Array:
        time_bin = 1.0 / n_steps
        t = jnp.zeros((x_0.shape[0],))
        s = jnp.ones((x_0.shape[0],))
        x = x_0
        for _ in range(n_steps):
            x = self.forward_flow(x, t, s, *args, **kwargs)
            t = t + time_bin
            s = s + time_bin
        return x

    def average_velocity(self, x: Array, t: Array, s: Array, *args, **kwargs) -> Array:
        s_minus_t = expand_to(s - t, x.ndim)
        x_s = self.forward_flow(x, t, s, *args, **kwargs)
        return self.manifold.log_map(x, x_s) / (s_minus_t + 1e-8)

    def __call__(self, x: Array, t: Array, s: Array, *args, **kwargs) -> Array:
        # jax.debug.print("x: {}, t: {}, s: {}", x.shape, t.shape, s.shape)
        return jnp.where(
            expand_to(s == t, x.ndim),
            self.instant_velocity(x, t, *args, **kwargs),
            self.average_velocity(x, t, s, *args, **kwargs),
        )

    def instant_velocity(self, x: Array, t: Array, *args, **kwargs) -> Array:
        flow_fn = lambda s: self.forward_flow(x, t, s, *args, **kwargs)
        v = jax.jvp(flow_fn, (t,), (jnp.ones_like(t),))[1]
        return self.manifold.proju(x, v)
