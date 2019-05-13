from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils import data

DATA_DIR = Path("/home/drgnelement/project/cs231n/data")
CHEXPERT_DIR = DATA_DIR / "CheXpert-v1.0-small"

class ChexpertDataset(data.Dataset):
    """Dataset of random 224x224 images"""
    def __init__(self, split):
        super(ChexpertDataset, self).__init__()
        self.split = split
        assert split == "train" or split == "val", "Invalid split"

        if self.split == "train":
            self.df = pd.read_csv(CHEXPERT_DIR / "train.csv")
        elif self.split ==  "val":
            self.df = pd.read_csv(CHEXPERT_DIR / "valid.csv")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = Image.open(DATA_DIR / self.df.iloc[index]["Path"]).convert("RGB")
        # TODO implement label indexing
        y = None

        return x, y
