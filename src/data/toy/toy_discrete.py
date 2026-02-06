import numpy as np
from torch.utils.data import Dataset
from typing import List


def np_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[labels]


def np_smooth_labels(labels: np.ndarray, mx: float = 0.98) -> np.ndarray:
    num_classes = labels.shape[-1]

    increase = (1.0 - mx) / (num_classes - 1)

    smooth_labels = np.full_like(labels, increase, dtype=np.float32)

    smooth_labels = np.where(labels == 1, mx, smooth_labels)

    return smooth_labels


def np_project_to_sphere(data: np.ndarray) -> np.ndarray:
    return data / np.linalg.norm(data, axis=-1, keepdims=True)


class ToyThreeTokenDataset(Dataset):
    """Length-1, 3-token dataset with proportions [0.5, 0.3, 0.2] (default).

    Returns numpy arrays from __getitem__.
    """

    def __init__(
        self,
        dataset_size: int,
        probs=(0.5, 0.3, 0.2),
        seed=None,
        transform_to_sphere=False,
        transform_to_onehot=False,
    ):
        super(ToyThreeTokenDataset, self).__init__()

        self.dataset_size = int(dataset_size)
        self.transform_to_sphere = transform_to_sphere
        self.transform_to_onehot = transform_to_onehot

        self._data = self._gen_data(probs, seed)
        self._data = self.transform_data(self._data)

    def _gen_data(self, probs, seed):
        p = np.asarray(probs, dtype=float)
        p = p / p.sum()
        rng = np.random.default_rng(seed)
        tokens = rng.choice(3, size=self.dataset_size, p=p)
        return tokens.astype(np.int32)

    def transform_data(self, data):
        if self.transform_to_sphere:
            data = np_one_hot(data, 3)
            data = np_smooth_labels(data, mx=0.95)
            data = np.sqrt(data)
            data = np_project_to_sphere(data)

        elif self.transform_to_onehot:
            # One-hot encoding of labels
            data = np_one_hot(data, 3)

        return data

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self._data[idx]


class ToyDiscreteDataset(Dataset):
    """Length-N, D-token dataset with independent token probabilities.

    Returns numpy arrays from __getitem__.
    """

    def __init__(
        self,
        dataset_size: int,
        seq_length: int,
        num_tokens: int,
        probs: List[float],
        seed=None,
        transform_to_sphere=False,
        transform_to_onehot=False,
    ):
        super(ToyDiscreteDataset, self).__init__()

        self.dataset_size = int(dataset_size)
        self.seq_length = int(seq_length)
        self.num_tokens = int(num_tokens)

        self.transform_to_sphere = transform_to_sphere
        self.transform_to_onehot = transform_to_onehot

        self._data = self._gen_data(probs, seed)
        self._data = self.transform_data(self._data)

    def _gen_data(self, probs, seed):

        p = np.asarray(probs, dtype=float)
        p = p / p.sum()

        rng = np.random.default_rng(seed)
        tokens = rng.choice(
            self.num_tokens, size=(self.dataset_size, self.seq_length), p=p
        )
        return tokens.astype(np.int32)

    def transform_data(self, data):
        if self.transform_to_sphere:
            data = np_one_hot(data, self.num_tokens)
            data = np_smooth_labels(data, mx=0.95)
            data = np.sqrt(data)
            data = np_project_to_sphere(data)

        elif self.transform_to_onehot:
            data = np_one_hot(data, self.num_tokens)

        return data

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self._data[idx]


if __name__ == "__main__":
    dataset = ToyThreeTokenDataset(
        10000, probs=[0.5, 0.3, 0.2], seed=42, transform_to_sphere=True
    )
    print(dataset[0])

    dataset = ToyDiscreteDataset(
        10000,
        seq_length=10,
        num_tokens=3,
        probs=[0.5, 0.3, 0.2],
        seed=42,
        transform_to_sphere=True,
    )
    print(dataset[0])
