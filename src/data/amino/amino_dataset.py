import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ExpandDataset(Dataset):
    def __init__(self, dset, expand_factor=1):
        self.dset = dset
        self.expand_factor = expand_factor

    def __len__(self):
        return len(self.dset) * self.expand_factor

    def __getitem__(self, idx):
        return self.dset[idx % len(self.dset)]


class Top500(Dataset):
    dim = 2

    def __init__(
        self, root="data/top500", amino="General", split="train", expand_factor=1
    ):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=["source", "phi", "psi", "amino"],
        )

        amino_types = ["General", "Glycine", "Proline", "Pre-Pro"]
        assert amino in amino_types, f"amino type {amino} not implemented"

        data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")
        self.data = torch.tensor(data % 360 * torch.pi / 180)

        n_val = int(0.1 * len(self.data))
        n_test = int(0.1 * len(self.data))
        n_train = len(self.data) - n_val - n_test
        self.data_train, self.data_val, self.data_test = random_split(
            self.data,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        if split == "train":
            self.data = ExpandDataset(self.data_train, expand_factor)
        elif split == "val":
            self.data = self.data_val
        elif split == "test":
            self.data = self.data_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNA(Dataset):
    dim = 7
    expand_factor = 14

    def __init__(self, root="data/rna", split="train"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=[
                "source",
                "base",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "chi",
            ],
        )

        data = data[
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        ].values.astype("float32")
        self.data = torch.tensor(data % 360 * torch.pi / 180)

        n_val = int(0.1 * len(self.data))
        n_test = int(0.1 * len(self.data))
        n_train = len(self.data) - n_val - n_test
        self.data_train, self.data_val, self.data_test = random_split(
            self.data,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        if split == "train":
            self.data = ExpandDataset(self.data_train, self.expand_factor)
        elif split == "val":
            self.data = self.data_val
        elif split == "test":
            self.data = self.data_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
