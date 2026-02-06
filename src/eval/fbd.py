import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
import flax.nnx as nnx


def get_wasserstein_dist(embeds1, embeds2):
    if (
        np.isnan(embeds2).any()
        or np.isnan(embeds1).any()
        or len(embeds1) == 0
        or len(embeds2) == 0
    ):
        return float("nan")
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    return fid_from_stats(mu1, sigma1, mu2, sigma2)


@jax.jit
def fid_from_stats(mu1, C1, mu2, C2, eps=1e-6):
    C1 = (C1 + C1.T) / 2 + eps * jnp.eye(C1.shape[0], dtype=C1.dtype)
    C2 = (C2 + C2.T) / 2 + eps * jnp.eye(C2.shape[0], dtype=C2.dtype)

    s1, U1 = jnp.linalg.eigh(C1)  # s1>=0
    s1 = jnp.clip(s1, 0.0, None)
    C1_sqrt = (U1 * jnp.sqrt(s1)) @ U1.T  # U1 @ diag(sqrt(s1)) @ U1^T

    # M = C1^{1/2} C2 C1^{1/2}
    M = C1_sqrt @ C2 @ C1_sqrt
    M = (M + M.T) / 2  # numerical symmetrization

    # Tr(sqrt(M)) = sum sqrt(eigvals(M))
    lm = jnp.linalg.eigvalsh(M)
    tr_sqrt = jnp.sum(jnp.sqrt(jnp.clip(lm, 0.0, None)))

    diff = mu1 - mu2
    fid = diff @ diff + jnp.trace(C1) + jnp.trace(C2) - 2.0 * tr_sqrt
    return fid


class FBD(nnx.metrics.Metric):
    """
    FBD metrics for SEI features.
    """

    feat_dim = 15360

    def __init__(self):
        self.embeds = []
        self.data_embeds = []

    def update(self, feat: Array, data_feat: Array):
        # Convert to numpy immediately to free GPU memory
        self.embeds.append(np.asarray(feat))
        self.data_embeds.append(np.asarray(data_feat))

    def compute(self):
        if len(self.embeds) == 0 or len(self.data_embeds) == 0:
            return float("nan")
        # Use numpy concatenate since arrays are stored as numpy (CPU)
        feats = np.concatenate(self.embeds, axis=0)
        data_feats = np.concatenate(self.data_embeds, axis=0)
        return get_wasserstein_dist(feats, data_feats) / self.feat_dim

    def reset(self):
        self.embeds = []
        self.data_embeds = []
