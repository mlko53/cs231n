import torch
import torch.nn as nn


class ConvMaskA(nn.Conv2d): 
    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(ConvMaskA, self).__init__(in_channel, out_channel,
                                        kernel, stride,
                                        pad, bias=False)

        out_c, in_c, HH, WW = self.weight.size()
        mask = torch.ones(out_c, in_c, HH, WW)
        # create mask here

    def forward(self, x):
        # update weight with mask here
        return super(ConvMaskA, self).forward(x)


class ConvMaskB(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(ConvMaskB, self).__init__(in_channel, out_channel,
                                        kernel, stride,
                                        pad, bias=False)

        out_c, in_c, HH, WW = self.weight.size()
        mask = torch.ones(out_c, in_c, HH, WW)
        # create mask here

    def forward(self, x):
        # update weight with mask here
        return super(ConvMaskB, self).forward(x)
