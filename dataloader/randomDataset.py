import torch
from torch.utils import data


class RandomDataset(data.Dataset):
    """Dataset of random 224x224 images"""
    def __init__(self, split):
        super(RandomDataset, self).__init__()
        self.split = split
        self.data = torch.empty(1000, 3, 32, 32, dtype=torch.long).random_(256)
        self.data = self.data.float() / 256.

    def __len__(self):
        return 10

    def __getitem__(self, index):
        x = self.data[index]
        y = torch.randn(1)

        return x, y
