from torch.utils.data import Dataset


from src.data import (
    PromoterDataset,
    ToyDiscreteDataset,
    ToyThreeTokenDataset,
    SnEmbeddedHelixDataset,
    EarthDataset,
    Top500,
    RNA,
)


def is_promoter_dataset(dataset: Dataset):
    return isinstance(dataset, PromoterDataset)


def is_toy_dataset(dataset: Dataset):
    return isinstance(dataset, ToyDiscreteDataset) or isinstance(
        dataset, ToyThreeTokenDataset
    )


def is_toy_helix_dataset(dataset: Dataset):
    return isinstance(dataset, SnEmbeddedHelixDataset)


def is_earth_dataset(dataset: Dataset):
    return isinstance(dataset, EarthDataset)


def is_amino_dataset(dataset: Dataset):
    return isinstance(dataset, Top500) or isinstance(dataset, RNA)
