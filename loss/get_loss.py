import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

def get_loss(model):
    if model == "PixelCNN":
        loss_fn = PixelCNNLoss()
    elif model == "Glow":
        loss_fn = GlowLoss()
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
        #x = torch.stack([self.inv_norm(i) for i in x])
        target = x.view(-1)
        target = (target * 255.).long()
        loss = self.loss_fn(logit, target)
        return loss


class GlowLoss(nn.Module):
    def __init__(self):
        super(GlowLoss, self).__init__()
        

    def forward(self, output, x):
        out, sldj = output
        prior_ll = -0.5 * (out ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(256.) * np.prod(out.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean() / (out.shape[-1] ** 2)

        return nll
