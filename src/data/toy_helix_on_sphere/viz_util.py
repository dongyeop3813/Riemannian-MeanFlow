import math

import numpy as np
import matplotlib.pyplot as plt


def s2_scatter_plot(points: np.ndarray, mode: str = "3d", extra_text: str | None = None):
    """
    points: [B, 3], unit vectors on S^2
    mode: "3d" or "latlon"
    """
    assert points.ndim == 2, "points must be of shape [B, 3]"
    assert mode in ["3d", "latlon"], "mode must be '3d' or 'latlon'"

    if mode == "3d":
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Add a transparent unit sphere as background
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        u, v = np.meshgrid(u, v)
        xs = np.cos(u) * np.sin(v) * 0.98
        ys = np.sin(u) * np.sin(v) * 0.98
        zs = np.cos(v) * 0.98
        ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.1, linewidth=0, zorder=0)

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=1,
            alpha=0.8,
        )

        # Hide axes so that only the sphere surface and points are visible
        ax.set_axis_off()

        # Optionally annotate metric text on the figure
        if extra_text is not None:
            # Place text in axes coordinates (top-left)
            ax.text2D(
                0.02,
                0.98,
                extra_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )

        # Make axes roughly equal
        ax.set_box_aspect((1, 1, 1))
        plt.tight_layout()
        return fig

    elif mode == "latlon":
        # Convert to (lon, lat)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        lon = np.atan2(y, x)
        lat = np.arcsin(np.clip(z, -1.0, 1.0))

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(lon, lat, s=1, alpha=0.8)
        ax.set_xlabel("longitude (rad)")
        ax.set_ylabel("latitude (rad)")
        ax.set_xlim([-math.pi, math.pi])
        ax.set_ylim([-math.pi / 2.0, math.pi / 2.0])

        # Optionally annotate metric text on the figure
        if extra_text is not None:
            ax.text(
                0.02,
                0.98,
                extra_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )
        plt.tight_layout()
        return fig
    else:
        raise ValueError('mode must be "3d" or "latlon"')
