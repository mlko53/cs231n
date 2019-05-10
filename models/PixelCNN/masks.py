import torch
import torch.nn as nn


class ConvMaskA(nn.Conv2d): 
    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(ConvMaskA, self).__init__(in_channel, out_channel,
                                        kernel, stride,
                                        pad, bias=False)

        out_c, in_c, HH, WW = self.weight.size()
        mask = torch.ones(out_c, in_c, HH, WW)
        mask[:,:, HH // 2, WW // 2:] = 0
        mask[:,:, (HH // 2 + 1):, :] = 0

        self.mask =  mask

    def forward(self, x):
        self.weight.data *= self.mask
        return super(ConvMaskA, self).forward(x)


class ConvMaskB(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(ConvMaskB, self).__init__(in_channel, out_channel,
                                        kernel, stride,
                                        pad, bias=False)

        out_c, in_c, HH, WW = self.weight.size()
        mask = torch.ones(out_c, in_c, HH, WW)
        mask[:,:, HH // 2, (WW // 2 + 1):] = 0
        mask[:,:, (HH // 2 + 1):, :] = 0

        self.mask = mask

    def forward(self, x):
        self.weight.data *= self.mask
        return super(ConvMaskB, self).forward(x)


if __name__ == "__main__":
    print("Testing Conv Mask A")
    convMaskA = ConvMaskA(1, 1, 7, 1, 0)
    x = torch.ones(1, 1, 7, 7)
    print(convMaskA.weight.data[0,0,:,:])
    print(convMaskA.mask[0,0,:,:])
    print(convMaskA(x)[0,0,:,:])

    print("Testing Conv Mask B")
    convMaskB = ConvMaskB(1, 1, 7, 1, 0)
    x = torch.ones(1, 1, 7, 7)
    print(convMaskB.weight.data[0,0,:,:])
    print(convMaskB.mask[0,0,:,:])
    print(convMaskA(x)[0,0,:,:])
