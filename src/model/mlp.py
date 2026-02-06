import jax.numpy as jnp
import flax.nnx as nnx
from jax import Array


class MLP(nnx.Module):
    def __init__(self, x_dim, hidden_dim, n_layers, rngs: nnx.Rngs):
        self.layers = [nnx.Linear(x_dim + 2, hidden_dim, rngs=rngs)]
        for _ in range(n_layers - 2):
            self.layers.append(nnx.Linear(hidden_dim + 2, hidden_dim, rngs=rngs))
        self.read_out = nnx.Linear(hidden_dim, x_dim, rngs=rngs)

    def __call__(self, x: Array, t: Array, s: Array) -> Array:
        h = x
        for layer in self.layers:
            h = layer(jnp.concatenate([h, t[:, None], s[:, None]], axis=-1))
            h = nnx.silu(h)
        v = self.read_out(h)
        return v


class MLPwithShape(MLP):
    def __init__(self, x_dim, hidden_dim, n_layers, x_shape, rngs: nnx.Rngs):
        super().__init__(x_dim, hidden_dim, n_layers, rngs)
        self.x_shape = x_shape

        # Store as Python int to avoid tracing issues
        self.x_dim = int(jnp.prod(jnp.asarray(x_shape)))
        self.x_ndim = len(x_shape)

    def __call__(self, x: Array, t: Array, s: Array) -> Array:
        batch_shape = x.shape[: -self.x_ndim]
        x_flat = x.reshape(batch_shape + (self.x_dim,))
        v_flat = super().__call__(x_flat, t, s)
        return v_flat.reshape(batch_shape + self.x_shape)


class WeakMLP(nnx.Module):
    def __init__(self, x_dim, hidden_dim, n_layers, rngs: nnx.Rngs):
        self.layers = [nnx.Linear(x_dim + 2, hidden_dim, rngs=rngs)]
        for _ in range(n_layers - 2):
            self.layers.append(nnx.Linear(hidden_dim, hidden_dim, rngs=rngs))
        self.read_out = nnx.Linear(hidden_dim, x_dim, rngs=rngs)

    def __call__(self, x: Array, t: Array, s: Array) -> Array:
        h = jnp.concatenate([x, t[:, None], s[:, None]], axis=-1)
        for layer in self.layers:
            h = layer(h)
            h = nnx.relu(h)
        v = self.read_out(h)
        return v


class WeakMLPwithShape(WeakMLP):
    def __init__(self, x_dim, hidden_dim, n_layers, x_shape, rngs: nnx.Rngs):
        super().__init__(x_dim, hidden_dim, n_layers, rngs)
        self.x_shape = x_shape

        # Store as Python int to avoid tracing issues
        self.x_dim = int(jnp.prod(jnp.asarray(x_shape)))
        self.x_ndim = len(x_shape)

    def __call__(self, x: Array, t: Array, s: Array) -> Array:
        batch_shape = x.shape[: -self.x_ndim]
        x_flat = x.reshape(batch_shape + (self.x_dim,))
        v_flat = super().__call__(x_flat, t, s)
        return v_flat.reshape(batch_shape + self.x_shape)
