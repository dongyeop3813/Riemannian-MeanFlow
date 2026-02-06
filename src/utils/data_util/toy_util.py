import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
from src.manifold import sphere_to_simplex


def compute_histogram_l2(x_1_pred, x_1, num_classes=3):
    """
    Compute the L2 norm between the predicted and true histograms.

    Parameters:
    -----------
    x_1_pred : Array
        Predicted labels (not an one-hot vector)
    x_1 : Array
        True labels (not an one-hot vector)
    num_classes : int
        Number of classes

    Returns:
    --------
    float
        L2 norm between the predicted and true histograms
    """

    bins = jnp.arange(num_classes)

    def get_proportion(x):
        counts = jnp.array([(x == b).sum() for b in bins])
        return counts / counts.sum()

    hist_pred = get_proportion(x_1_pred)
    hist_true = get_proportion(x_1)
    l2_norm = jnp.linalg.norm(hist_pred - hist_true)
    return l2_norm


def plot_token_histogram(label, ax=None, num_classes=3):
    """
    Plot the token histogram of the model sample.
    We assume that the model sample is generated for ThreeToken dataset.

    Args:
        model_sample: The model sample. (label, not an one-hot vector)

    Returns:
        The figure and axis of the plot.
    """

    ax_is_not_given = ax is None
    if ax_is_not_given:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.set_title("Token histogram")
    ax.bar(
        range(num_classes),
        [(label == i).sum() for i in range(num_classes)],
        label="Model sample histogram",
    )
    ax.legend()

    if ax_is_not_given:
        return fig


def plot_on_simplex(coords, ax=None, color="C0", alpha=0.8):
    """
    Parameters:
    -----------
    coords : Array
        3D coordinates of the samples on the simplex
    ax : matplotlib axis
        The axis to plot on
    color : str
        The color of the samples
    """

    # Coordinates of the vertices of the equilateral triangle
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0.5, np.sqrt(3) / 2])
    vertices = np.stack([v1, v2, v3], axis=0)

    # Barycentric -> 2D coordinate transformation
    def barycentric_to_cartesian(p):
        # p: (..., 3)
        return p[..., 0:1] * v1 + p[..., 1:2] * v2 + p[..., 2:3] * v3

    coords = barycentric_to_cartesian(coords)

    ax_is_not_given = ax is None
    if ax_is_not_given:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Draw triangle
    triangle = np.vstack([vertices, vertices[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], "k-", lw=1)

    # Vertex labels
    ax.text(v1[0] - 0.05, v1[1] - 0.05, "0", fontsize=12)
    ax.text(v2[0] + 0.02, v2[1] - 0.05, "1", fontsize=12)
    ax.text(v3[0] + 0.03, v3[1], "2", fontsize=12)

    # Sample scatter plot
    ax.scatter(coords[:, 0], coords[:, 1], color=color, alpha=alpha, s=5)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Generated samples on probability simplex")

    if ax_is_not_given:
        return fig


def plot_3d_samples(coords):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color="C0", alpha=0.5, s=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Generated samples on Euclidean space")

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_zlim(-0.3, 1.3)

    return fig


def plot_xts_on_simplex(xts, prior, prob_path, data):

    num_eval_samples = data.shape[0]
    x_shape = data.shape[1:]

    model_fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    for t, xt, ax in zip(jnp.linspace(0, 1.0, 5), xts, axs):
        plot_on_simplex(sphere_to_simplex(xt), ax=ax)
        ax.set_title(f"t={t:.2f}")

    model_fig.suptitle("Model x_t on probability simplex")

    true_fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    for t, ax in zip(jnp.linspace(0, 1.0, 5), axs):
        x_0 = prior.sample(num_eval_samples, *x_shape)
        x_t = prob_path.sample(x_0, data, jnp.ones(num_eval_samples) * t).x_t
        plot_on_simplex(sphere_to_simplex(x_t), ax=ax)

        ax.set_title(f"t={t:.2f}")

    true_fig.suptitle("True x_t on probability simplex")

    return {
        "model_x_t": model_fig,
        "true_x_t": true_fig,
    }
