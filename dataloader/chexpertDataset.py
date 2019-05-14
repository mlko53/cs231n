from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms


DATA_DIR = Path("/deep/group/CheXpert")
CHEXPERT_DIR = DATA_DIR / "CheXpert-v1.0-small"

class ChexpertDataset(data.Dataset):
    """Dataset of random 224x224 images"""
    def __init__(self, split, batch_size, size):
        super(ChexpertDataset, self).__init__()
        self.split = split
        self.size = size
        assert split == "train" or split == "val", "Invalid split"

        if self.split == "train":
            self.df = pd.read_csv(CHEXPERT_DIR / "train.csv")
        elif self.split ==  "val":
            self.df = pd.read_csv(CHEXPERT_DIR / "valid.csv")

        # Filter only frontal images
        self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        self.transforms = transforms.Compose([
            transforms.RandomCrop((320, 320)),
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        self.df = self.df[:-(len(self.df)%batch_size)]
        print(len(self.df))
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = self.transforms(Image.open(DATA_DIR / self.df.iloc[index]["Path"]).convert("RGB"))
        # TODO implement label indexing
        y = None

        return x
