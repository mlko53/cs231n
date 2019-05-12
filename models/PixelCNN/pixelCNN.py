import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import ConvMaskABlock, ConvMaskBBlock


class PixelCNN(nn.Module):
    """PixelCNN Model

    Based on paper (https://arxiv.org/pdf/1601.06759.pdf)

    """

    def __init__(self, args, device):
        super(PixelCNN, self).__init__()
        self.device = device
        self.conv_mask_A = ConvMaskABlock(3, args.num_channels, 7, 1, 3, self.device)

        self.conv_mask_Bs = []
        for i in range(args.num_levels):
            self.conv_mask_Bs.append(ConvMaskBBlock(args.num_channels // 2, 3, 1, 1, self.device))
        self.conv_mask_Bs = nn.Sequential(*self.conv_mask_Bs)
        
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(args.num_channels, 1024, 1, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 3, 1, 1, 0)
        )

    def forward(self, x):
        out = self.conv_mask_A(x)
        out = self.conv_mask_Bs(out)
        out = self.out(out)
        return out
