import jax
import jax.numpy as jnp
from jax import Array


def sphere_to_label(x: Array) -> Array:
    return jnp.argmax(x, axis=-1)


def sphere_to_onehot(x: Array) -> Array:
    label = sphere_to_label(x)
    return jax.nn.one_hot(label, x.shape[-1])


def simplex_to_sphere(x: Array) -> Array:
    return jnp.sqrt(x)


def sphere_to_simplex(x: Array) -> Array:
    x_sq = jnp.square(x)
    sq_sum = jnp.sum(x_sq, axis=-1, keepdims=True) + 1e-12
    return x_sq / sq_sum
