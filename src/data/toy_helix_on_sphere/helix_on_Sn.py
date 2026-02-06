import math
import numpy as np
from typing import Optional

import torch
from torch.utils.data import Dataset


# -------------------------
# Reuse / extend sphere utilities
# -------------------------


def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))


def normalize_sphere(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / _safe_norm(x, eps=eps)


def np_normalize_sphere(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def expmap_sphere(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Exponential map on S^n (unit sphere in R^{n+1}).
    x: (..., d) unit, v: (..., d) tangent (x·v=0)
    """
    theta = _safe_norm(v, eps=eps)  # (...,1)
    sin_over_theta = torch.where(
        theta > 1e-6, torch.sin(theta) / theta, 1.0 - (theta**2) / 6.0
    )
    y = torch.cos(theta) * x + sin_over_theta * v
    return normalize_sphere(y)


def gaussian_tangent_noise_at_sphere(
    x: torch.Tensor, sigma: float, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Isotropic Gaussian in tangent space T_x S^n, implemented by:
      z ~ N(0, I) in R^{n+1}
      v = (I - x x^T) z   (project to tangent)
      v *= sigma
    x: (..., d) unit
    """
    z = torch.randn(*x.shape, generator=generator, device=x.device, dtype=x.dtype)
    # project to tangent: v = z - <z,x> x
    dot = (z * x).sum(dim=-1, keepdim=True)
    v = z - dot * x
    v = v * sigma
    return v


def random_U_embed_S2_to_Sn(
    n: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample U in R^{(n+1) x 3} with orthonormal columns (U^T U = I_3).
    This is the manifold analog of column-orthogonal embedding.
    """
    A = torch.randn(n + 1, 3, generator=generator, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)  # Q: (n+1,3)
    # (optional) stabilize sign using R diag
    diag = torch.sign(torch.diag(R))
    Q = Q * diag
    return Q


def embed_S2_to_Sn(x_s2: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    x_s2: (...,3) on S^2
    U: (n+1,3) with orthonormal columns
    returns: (..., n+1) on S^n
    """
    return x_s2 @ U.T


def spherical_helix(t: torch.Tensor, a: float = 2.0, b: float = 0.6) -> torch.Tensor:
    """
    Same helix-like curve on S^2 as before.
    t: (...) in R, typically [0, 2pi]
    returns: (...,3) unit
    """
    theta = a * t
    phi = b * t
    cphi = torch.cos(phi)
    x = cphi * torch.cos(theta)
    y = cphi * torch.sin(theta)
    z = torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)


class SnEmbeddedHelixDataset(Dataset):
    """
    Samples a noisy 1D curve embedded in high-dimensional sphere S^n.

    Procedure:
      1) x_clean(t) on S^2 via spherical_helix
      2) y_clean(t) = U x_clean(t) in S^n (U has orthonormal columns)
      3) y = Exp_{y_clean}(v), v ~ N(0, sigma^2 I) in tangent space

    Output:
      dict with 'x': tensor shape [n+1]
    """

    def __init__(
        self,
        n: int = 512,  # sphere dimension: S^n in R^{n+1}
        n_samples: int = 100_000,
        t_min: float = -math.pi,
        t_max: float = math.pi,
        helix_a: float = 5.0,  # controls winding (longitude)
        helix_b: float = 0.5,  # controls latitude oscillation
        signal_sigma: float = 0.02,  # tangent noise around helix on the signal factor
        random_embedding: bool = True,  # if False, use canonical zero-padding embedding
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert n >= 2
        self.n = n
        self.n_samples = n_samples
        self.t_min = t_min
        self.t_max = t_max
        self.helix_a = helix_a
        self.helix_b = helix_b
        self.signal_sigma = signal_sigma
        self.random_embedding = random_embedding
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        if random_embedding:
            self.U = random_U_embed_S2_to_Sn(
                n, generator=self.gen, device=self.device, dtype=self.dtype
            )  # (n+1, 3)
        else:
            # canonical embedding: (x1,x2,x3) -> (x1,x2,x3,0,...,0)
            U = torch.zeros(n + 1, 3, device=self.device, dtype=self.dtype)
            U[0, 0] = 1.0
            U[1, 1] = 1.0
            U[2, 2] = 1.0
            self.U = U

        # -------------------------
        # Precompute / cache dataset
        # -------------------------

        # sample all t uniformly: [N]
        u = torch.rand(
            n_samples, generator=self.gen, device=self.device, dtype=self.dtype
        )
        t_all = t_min + (t_max - t_min) * u  # [N]

        # helix on S^2 for all t: [N,3]
        x_clean_s2_all = spherical_helix(t_all, a=helix_a, b=helix_b).to(self.dtype)

        # add tangent noise + expmap on S^2, vectorized over N
        v_all_s2 = gaussian_tangent_noise_at_sphere(
            x_clean_s2_all, sigma=signal_sigma, generator=self.gen
        )
        x_noisy_s2_all = expmap_sphere(x_clean_s2_all, v_all_s2)  # [N, 3]

        # embed noisy S^2 into S^n: [N, n+1]
        y_all = embed_S2_to_Sn(x_noisy_s2_all, self.U)

        # cache
        self._y = y_all

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        # read from cache
        return self._y[idx]  # [n+1]

    def project_to_S2(self, y: np.ndarray) -> np.ndarray:
        """
        Project ambient space point(s) in S^n back to S^2 using the
        embedding matrix U.

        y: tensor of shape [..., n+1] (points on S^n in ambient space)
        returns: tensor of shape [..., 3] on S^2
        """
        U = self.U.cpu().numpy()
        x = y @ U  # [..., 3]
        return np_normalize_sphere(x)

    def visualize_projection_to_S2(
        self,
        y_batch: np.ndarray,
        mode: str = "3d",
    ):
        """
        Project a batch of ambient points in S^n to S^2 and visualize.

        y_batch: tensor of shape [N, n+1]
        mode: "3d" for 3D scatter on S^2, "latlon" for (longitude, latitude) plot
        point_size: marker size for scatter plot
        """
        assert y_batch.ndim == 2, "y_batch must be of shape [N, n+1]"

        # Compute how far samples are from the subspace spanned by U.
        # Here we measure the average fraction of energy that lies in the
        # orthogonal complement of span(U).
        U = self.U.cpu().numpy()  # (n+1, 3), orthonormal columns
        # Projection of y onto span(U): y_parallel = (y U) U^T
        coeff = y_batch @ U  # [N, 3]
        y_parallel = coeff @ U.T  # [N, n+1]
        residual = y_batch - y_parallel  # orthogonal component

        # Energy fraction in orthogonal complement: ||residual||^2 / ||y||^2
        y_sq = np.sum(y_batch**2, axis=-1)  # [N]
        residual_sq = np.sum(residual**2, axis=-1)  # [N]
        energy_fraction = residual_sq / np.maximum(y_sq, 1e-12)
        mean_energy_fraction = float(energy_fraction.mean())

        # metric_text = f"Mean orthogonal energy fraction: {mean_energy_fraction:.4f}"

        # Project to S^2 and bring to CPU for plotting
        x_s2 = self.project_to_S2(y_batch)  # [N,3]

        from .viz_util import s2_scatter_plot

        return s2_scatter_plot(x_s2, mode=mode)


# -------------------------
# Quick usage example
# -------------------------
if __name__ == "__main__":
    ds = SnEmbeddedHelixDataset(n=128)
    sample = ds[0]
    print(sample.shape)  # torch.Size([n+1])
