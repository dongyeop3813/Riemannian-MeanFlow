import os
from csv import reader
import numpy as np
from matplotlib import pyplot as plt
import geopandas
from shapely.geometry import Point

import torch
from torch.utils.data import Dataset, DataLoader, random_split


def cartesian_from_latlon(x):
    """Embedded 3D unit vector from spherical polar coordinates.

    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.

    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    assert x.shape[-1] == 2
    lat = x.select(-1, 0)
    lon = x.select(-1, 1)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)


def lonlat_from_cartesian(x):
    r = x.pow(2).sum(-1).sqrt()
    x, y, z = x[..., 0], x[..., 1], x[..., 2]
    lat = torch.asin(z / r)
    lon = torch.atan2(y, x)
    return torch.cat([lon.unsqueeze(-1), lat.unsqueeze(-1)], dim=-1)


def lonlat_from_cartesian_numpy(x):
    r = np.linalg.norm(x, axis=-1)
    x, y, z = x[..., 0], x[..., 1], x[..., 2]
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return np.concatenate([lon.reshape(-1, 1), lat.reshape(-1, 1)], axis=-1)


class ExpandDataset(Dataset):
    def __init__(self, dset, expand_factor=1):
        self.dset = dset
        self.expand_factor = expand_factor

    def __len__(self):
        return len(self.dset) * self.expand_factor

    def __getitem__(self, idx):
        return self.dset[idx % len(self.dset)]


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = np.array(list(lines)[1:]).astype(np.float64)
    return dataset


class EarthData(Dataset):
    dim = 3

    def __init__(self, dirname, filename):
        filename = os.path.join(dirname, filename)
        dataset = load_csv(filename)
        dataset = torch.Tensor(dataset)
        self.latlon = dataset
        self.data = cartesian_from_latlon(dataset / 180 * torch.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Volcano(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "volcano.csv")


class Earthquake(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "earthquake.csv")


class Fire(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "fire.csv")


class Flood(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "flood.csv")


class EarthDataset(Dataset):
    """Earth data split into train/validation/test splits using pure torch utilities."""

    expand_factors = {
        "volcano": 1550,
        "earthquake": 210,
        "fire": 100,
        "flood": 260,
    }

    def __init__(
        self,
        data_dir: str = "data/earth_data",
        dataset_file: str = "volcano",
        split: str = "train",
    ):
        """
        Initialize an `EarthDataset`.

        :param data_dir: Directory where CSV files are stored.
        :param dataset_file: Base name of the CSV file without extension, e.g. 'volcano'.
        :param split: One of 'train', 'val', 'test'.
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.data_dir = data_dir
        self.dataset_file = dataset_file
        self.split = split

        base_dataset = EarthData(
            dirname=self.data_dir, filename=f"{self.dataset_file}.csv"
        )

        n_val = int(0.1 * len(base_dataset))
        n_test = int(0.1 * len(base_dataset))
        n_train = len(base_dataset) - n_val - n_test

        self.data_train, self.data_val, self.data_test = random_split(
            base_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        expand_factor = self.expand_factors.get(self.dataset_file, 1)

        if split == "train":
            self.data = ExpandDataset(self.data_train, expand_factor=expand_factor)
        elif split == "val":
            self.data = self.data_val
        else:
            self.data = self.data_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def visualize(
        self,
        y_batch: np.ndarray,
    ):
        """
        Visualize a batch of ambient points in S^2 (Longitude/Latitude) on world map.

        y_batch: numpy array of shape [N, 3], Cartesian coordinates on the sphere.
        Returns a publication-quality matplotlib Figure.
        """
        assert y_batch.ndim == 2 and y_batch.shape[1] == 3, "y_batch must be of shape [N, 3]"

        # High-res, publication-ready figure with professional, minimal design
        fig, ax = plt.subplots(figsize=(10, 4.2), dpi=600)

        # Plot subtle, light gray world map background without axes
        world = geopandas.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
        world.plot(
            ax=ax,
            color="#f1f1f1",           # ultra light gray for background
            # edgecolor="#aaaaaa",       # slightly darker coastlines
            # linewidth=0.5,
            antialiased=True,
            zorder=0,
        )

        # Convert cartesian to lon/lat in degrees
        lonlat_batch = lonlat_from_cartesian_numpy(y_batch) / np.pi * 180

        # Main scatter: slightly larger, pure red tone, crisp no border
        ax.scatter(
            lonlat_batch[:, 0], lonlat_batch[:, 1],
            color="#D42020",        # paper-friendly vivid red, color-blind safe
            s=14,                   # increase size for print
            alpha=0.6,
            edgecolors="none",
            marker="o",
            zorder=10,
        )

        # Hide all spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Set custom ticks, no tick lines
        ax.tick_params(
            axis="both",          # Both axes
            which="both",
            bottom=False, top=False, left=False, right=False,
            labelbottom=True, labelleft=True,
            length=0, width=0, pad=8, labelsize=20,
        )

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks([-100, 0, 100])
        ax.set_yticks([-50, 0, 50])

        # Axis labels: clean serif, slightly higher up, no bold
        ax.set_xlabel("Longitude", fontsize=26, labelpad=16, fontname="serif")
        ax.set_ylabel("Latitude", fontsize=26, labelpad=12, fontname="serif")

        # Subtle grid for visual guidance (no lines on land)
        ax.grid(
            which="both",
            color="#bbbbbb",
            linewidth=0.6,
            linestyle="--",
            alpha=0.45,
            zorder=1,
        )

        # No legend, remove axis frame
        ax.set_frame_on(False)

        # Tight layout, a bit extra pad for axis labels
        fig.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.13)
        fig.patch.set_facecolor('white')

        return fig
