import math
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# -------------------------
# Sphere utilities (S^2)
# -------------------------


def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))


def normalize_s2(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / _safe_norm(x, eps=eps)


def sample_uniform_s2(
    n: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform samples on S^2 via normalizing Gaussian."""
    x = torch.randn(n, 3, generator=generator, device=device, dtype=dtype)
    return normalize_s2(x)


def tangent_basis_at(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build an orthonormal basis (e1, e2) for T_x S^2.
    x: (..., 3), assumed unit.
    returns e1, e2: (..., 3)
    """
    # choose a vector not parallel to x
    z = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype).expand_as(x)
    alt = torch.tensor([1.0, 0.0, 0.0], device=x.device, dtype=x.dtype).expand_as(x)

    # if x too close to z, use alt
    close = x[..., 2:3].abs() > 0.9  # (...,1)
    a = torch.where(close, alt, z)

    e1 = torch.cross(a, x, dim=-1)
    e1 = normalize_s2(e1)
    e2 = torch.cross(x, e1, dim=-1)  # already orthonormal if x,e1 are
    return e1, e2


def expmap_s2(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Exponential map on S^2.
    x: (...,3) unit, v: (...,3) tangent (x·v=0)
    """
    theta = _safe_norm(v, eps=eps)  # (...,1)
    # sin(theta)/theta stable
    sin_over_theta = torch.where(
        theta > 1e-6, torch.sin(theta) / theta, 1.0 - (theta**2) / 6.0
    )
    y = torch.cos(theta) * x + sin_over_theta * v
    return normalize_s2(y)


def gaussian_tangent_noise_at(
    x: torch.Tensor, sigma: float, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Sample v ~ N(0, sigma^2 I) in tangent plane T_x S^2 (2D isotropic).
    """
    e1, e2 = tangent_basis_at(x)
    # coefficients in R^2
    a = (
        torch.randn(
            *x.shape[:-1], 1, generator=generator, device=x.device, dtype=x.dtype
        )
        * sigma
    )
    b = (
        torch.randn(
            *x.shape[:-1], 1, generator=generator, device=x.device, dtype=x.dtype
        )
        * sigma
    )
    v = a * e1 + b * e2
    return v


def random_rotation_matrix(
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a random rotation matrix in SO(3) using QR on Gaussian matrix.
    """
    A = torch.randn(3, 3, generator=generator, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # make it a proper rotation (det=+1)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# -------------------------
# Spherical helix on S^2
# -------------------------


def spherical_helix(t: torch.Tensor, a: float = 2.0, b: float = 0.6) -> torch.Tensor:
    """
    A simple "spherical helix-like" curve on S^2:
      theta(t) = a t (longitude)
      phi(t)   = b t (latitude modulation)
    Map:
      x = cos(phi) cos(theta)
      y = cos(phi) sin(theta)
      z = sin(phi)
    t: (...,) any real (we'll typically use t in [0, 2π]).
    returns: (...,3) unit
    """
    theta = a * t
    phi = b * t
    cphi = torch.cos(phi)
    x = cphi * torch.cos(theta)
    y = cphi * torch.sin(theta)
    z = torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)


# -------------------------
# Dataset (Product of Sphere Helix)
# -------------------------


class ProductSphereHelixDataset(Dataset):
    """
    Each sample is a point on (S^2)^K: tensor shape [K,3].

    - Factor 0: helix(t) on S^2 + tangent Gaussian noise via Exp map.

    Optional burying:
    - Fixed random rotation per factor (isometry).
    - Optional fixed permutation of factors.
    """

    def __init__(
        self,
        K: int = 16,  # number of S^2 factors
        n_samples: int = 100_000,  # dataset length
        t_min: float = -math.pi,
        t_max: float = math.pi,
        helix_a: float = 5.0,  # controls winding (longitude)
        helix_b: float = 0.5,  # controls latitude oscillation
        signal_sigma: float = 0.02,  # tangent noise around helix on the signal factor
        bury_with_rotations: bool = True,  # apply fixed random rotation per factor
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert K >= 1

        self.K = K
        self.n_samples = n_samples
        self.t_min = t_min
        self.t_max = t_max
        self.helix_a = helix_a
        self.helix_b = helix_b
        self.signal_sigma = signal_sigma
        self.bury_with_rotations = bury_with_rotations
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype

        # local generator for reproducibility
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        # -------------------------
        # Precompute / cache dataset
        # -------------------------
        n = n_samples

        # fixed rotations R_k
        if bury_with_rotations:
            self.R = torch.stack(
                [
                    random_rotation_matrix(
                        generator=self.gen, device=self.device, dtype=self.dtype
                    )
                    for _ in range(K)
                ],
                dim=0,
            )  # [K,3,3]

            # Fix the rotation matrix for the first factor manifold to be the identity matrix
            self.R[0] = torch.eye(3, device=self.device, dtype=self.dtype)

            R = self.R
        else:
            self.R = None
            R = None

        # a base point for "near_north"
        self.north = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)

        # sample t for all indices
        u = torch.rand(n, generator=self.gen, device=self.device, dtype=self.dtype)
        t_all = t_min + (t_max - t_min) * u  # [N]

        # clean helix points on factor 0 for all indices
        x_clean_all = spherical_helix(t_all, a=helix_a, b=helix_b).to(
            self.dtype
        )  # [N,3]

        # add tangent noise around the curve (signal factor) for all indices
        v_all = gaussian_tangent_noise_at(
            x_clean_all, sigma=signal_sigma, generator=self.gen
        )  # [N,3]
        x_signal_all = expmap_s2(x_clean_all, v_all)  # [N,3]

        # build product points: [N,K,3]
        X_all = x_signal_all.unsqueeze(1).expand(-1, K, 3)

        # bury with per-factor rotations (isometries)
        if R is not None:
            # apply: x -> R_k x for each factor k
            X_all = torch.einsum("kij,nkj->nki", R, X_all)

        # cache
        self._x = X_all  # [N,K,3]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self._x[idx]  # [K,3]

    def visualize_factor_s2(
        self, points: np.ndarray, factor_k: int = 0, mode: str = "3d"
    ):
        """
        points: [B, N, 3], unit vectors on S^2
        factor_k: which sphere factor to visualize (0 <= k < K)
        mode: "3d" or "latlon"
        """
        assert 0 <= factor_k < self.K

        points = points[:, factor_k, :]

        from .viz_util import s2_scatter_plot

        return s2_scatter_plot(points, mode=mode)

    def visualize_factor_s2_with_plotly(self, points: np.ndarray, factor_k: int = 0):
        """
        points: [B, N, 3], unit vectors on S^2
        factor_k: which sphere factor to visualize (0 <= k < K)
        """
        import plotly.graph_objects as go

        assert 0 <= factor_k < self.K
        points = points[:, factor_k, :]
        fig = go.Figure()

        # Draw unit sphere surface as a transparent background
        theta = np.linspace(0, 2 * np.pi, 60)
        phi = np.linspace(0, np.pi, 30)
        theta, phi = np.meshgrid(theta, phi)

        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)

        sphere = go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale="gray",
            opacity=0.15,
            showscale=False,
            name="sphere",
        )
        fig.add_trace(sphere)

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=2.0, color="blue", opacity=0.9),
            )
        )

        fig.update_layout(
            title=f"Factor k={factor_k} on S²",
            scene=dict(
                xaxis=dict(range=[-1.1, 1.1], title="x"),
                yaxis=dict(range=[-1.1, 1.1], title="y"),
                zaxis=dict(range=[-1.1, 1.1], title="z"),
                aspectmode="cube",  # equal aspect ratio
            ),
            width=700,
            height=700,
        )

        return fig


# -------------------------
# Quick usage example
# -------------------------
if __name__ == "__main__":
    ds = ProductSphereHelixDataset(K=32)
    sample = ds[0]
    print(sample.shape)  # torch.Size([K,3])
