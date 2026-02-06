from jax import Array
from src.manifold import Manifold
from .prob_path import PathSample, ProbPath


class GeodesicProbPath(ProbPath):
    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    def sample(self, x_0: Array, x_1: Array, t: Array) -> PathSample:
        _t = t.reshape((-1,) + (1,) * (x_1.ndim - 1))

        x_t = self.manifold.exp_map(x_0, _t * self.manifold.log_map(x_0, x_1))

        return PathSample(
            x_1=x_1,
            x_0=x_0,
            t=t,
            x_t=x_t,
            dx_t=self.manifold.log_map(x_t, x_1) / (1 - _t),
        )
