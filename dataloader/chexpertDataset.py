from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

DATA_DIR = Path("/deep/group/CheXpert")
CHEXPERT_DIR = DATA_DIR / "CheXpert-v1.0-small"
CHEXPERT_MEAN = [.5020, .5020, .5020]
CHEXPERT_STD = [.085585, .085585, .085585]

NO_FINDING = "No Finding"
MAIN_CATEGORIES = ["Atelectasis", "Caardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
ALL_CATEGORIES = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

class ChexpertDataset(data.Dataset):
    """Chexpert Dataset of specified size and input channel"""
    def __init__(self, split, batch_size, size, input_c):
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
            #transforms.Normalize(mean=CHEXPERT_MEAN, std=CHEXPERT_STD)
        ])

        self.df = self.df[:-(len(self.df)%batch_size)]
        print(len(self.df))
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = self.transforms(Image.open(DATA_DIR / self.df.iloc[index]["Path"])
        if input_c == 1:
            x = x.convert("L")
        elif input_c == 3:
            x = x.convert("RGB")
        else:
            raise ValueError("Input channel must be 1 or 3")

        return x
