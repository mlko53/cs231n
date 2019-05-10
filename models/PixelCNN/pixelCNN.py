import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelCNN(nn.Module):
    """PixelCNN Model

    Based on paper (https://arxiv.org/pdf/1601.06759.pdf)

    """

    def __init__(self, args):
        super(PixelCNN, self).__init__()


    def forward(self, x):
        return x
