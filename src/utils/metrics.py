from jax import Array
import jax.numpy as jnp

import flax.nnx as nnx


class ArrayCheckPoint(nnx.metrics.Metric):
    def __init__(self, prefix: str, checkpoint):
        self.prefix = prefix
        self.checkpoint = checkpoint
        self.arrs = []

    def update(self, values: Array):
        self.arrs.append(values)

    def save(self, epoch: int):
        arr = jnp.concatenate(self.arrs, axis=0)
        self.checkpoint.save_array(epoch, arr, self.prefix)

    def compute(self):
        return jnp.concatenate(self.arrs, axis=0)

    def reset(self):
        self.arrs = []


class ArrayAppend(nnx.metrics.Metric):
    def __init__(self):
        self.arrs = []

    def update(self, values: Array):
        self.arrs.append(values)

    def compute(self):
        return jnp.concatenate(self.arrs, axis=0)

    def reset(self):
        self.arrs = []
