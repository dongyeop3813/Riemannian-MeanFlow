import numpy as np

from src.manifold import Manifold


def compute_mmd(x1: np.ndarray, x2: np.ndarray, manifold: Manifold, gamma: float = 1.0):
    assert x1.shape[0] == x2.shape[0]

    def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ds = manifold.geodesic_distance(a, b) ** 2
        ds = np.exp(-gamma * ds)
        return ds

    # calculate all pairwise kernels using broadcasting to avoid .repeat errors
    # Shape: (N, D), x1 and x2
    N = x1.shape[0]
    x1a = x1[:, None, ...]  # (N, 1, D)
    x1b = x1[None, :, ...]  # (1, N, D)
    x2a = x2[:, None, ...]
    x2b = x2[None, :, ...]
    # (N,N,D) for all pairs

    k_xx = _kernel(x1a, x1b)
    k_yy = _kernel(x2a, x2b)
    k_xy = _kernel(x1a, x2b)

    # Take means, avoiding counting diagonal for unbiased estimate if desired
    mmd2 = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return np.sqrt(np.maximum(mmd2, 0.0))
