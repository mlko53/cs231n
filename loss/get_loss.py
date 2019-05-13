import torch.nn as nn

def get_loss(model):
    if model == "PixelCNN":
        loss_fn = PixelCNNLoss()
    else:
        raise ValueError()
    return loss_fn


class PixelCNNLoss(nn.Module):
    def __init__(self):
        super(PixelCNNLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, out, x):
        out = out.contiguous()
        logit = out.view(-1, 256)
        target = x.view(-1)
        target = (target * 255.).long()
        loss = self.loss_fn(logit, target)
        return loss
