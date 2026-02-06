from jax import Array
from src.manifold import Manifold
from .prob_path import PathSample, ProbPath


class LinearInterpolantProbPath(ProbPath):
    def sample(self, x_0: Array, x_1: Array, t: Array) -> PathSample:
        _t = t[:, None]

        return PathSample(
            x_1=x_1,
            x_0=x_0,
            t=t,
            x_t=x_0 * (1 - _t) + x_1 * _t,
            dx_t=x_1 - x_0,
        )
