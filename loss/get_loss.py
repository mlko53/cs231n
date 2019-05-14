import torch
import torch.nn as nn
from torchvision import transforms

def get_loss(model):
    if model == "PixelCNN":
        loss_fn = PixelCNNLoss()
    else:
        raise ValueError()
    return loss_fn


CHEXPERT_MEAN = [.5020, .5020, .5020]
CHEXPERT_STD = [.085585, .085585, .085585]

class PixelCNNLoss(nn.Module):
    def __init__(self):
        super(PixelCNNLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.inv_norm = transforms.Normalize(mean=[-.5020/.085585, -.5020/.085585, -.5020/.085585], 
                                             std=[1/.085585, 1/.085585, 1/.085585])

    def forward(self, out, x):
        out = out.contiguous()
        logit = out.view(-1, 256)
        x = torch.stack([self.inv_norm(i) for i in x])
        target = x.view(-1)
        target = (target * 255.).long()
        loss = self.loss_fn(logit, target)
        return loss
