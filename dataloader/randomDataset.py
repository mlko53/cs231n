import torch
from torch.utils import data


class RandomDataset(data.Dataset):
    """Dataset of random 224x224 images"""
    def __init__(self, split):
        super(RandomDataset, self).__init__()
        self.split = split
        self.data = torch.randn(1000, 3, 224, 244)

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        x = self.data[index]
        y = torch.randn(1)

        return x, y
