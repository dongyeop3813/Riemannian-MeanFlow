from .dataset import PromoterDataset


class DebugValPromoterDataset(PromoterDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 128
