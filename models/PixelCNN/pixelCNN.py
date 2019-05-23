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
        self.conv_mask_A = ConvMaskABlock(args.input_c, args.num_channels, 7, 1, 3, self.device)

        self.conv_mask_Bs = []
        for i in range(args.num_levels):
            self.conv_mask_Bs.append(ConvMaskBBlock(args.num_channels // 2, 3, 1, 1, self.device))
        self.conv_mask_Bs = nn.Sequential(*self.conv_mask_Bs)
        
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(args.num_channels, 1024, 1, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, args.input_c * 256, 1, 1, 0)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        N, C, H, W = x.shape
        out = self.conv_mask_A(x)
        out = self.conv_mask_Bs(out)
        out = self.out(out)
        out = out.view(N, C, 256, H, W)
        out = out.permute(0, 1, 3, 4, 2)
        return out
