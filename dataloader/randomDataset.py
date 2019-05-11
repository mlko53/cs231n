import torch
from torch.utils import data


class RandomDataset(data.Dataset):
    """Dataset of random 224x224 images"""
    def __init__(self):
        super(RandomDataset, self).__init__()

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        x = torch.randn(3,224,224)
        y = torch.randn(1)

        return x, y
