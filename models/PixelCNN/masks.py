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


class ConvMaskABlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad):
        super(ConvMaskABlock, self).__init__()

        self.net = nn.Sequential(
            ConvMaskA(in_channel, out_channel, kernel, stride, pad),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.net(x)


class ConvMaskBBlock(nn.Module):
    def __init__(self, h, kernel, stride, pad):
        super(ConvMaskBBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2*h, h, 1),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            ConvMaskB(h, h, kernel, stride, pad),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            nn.Conv2d(h, 2*h, 1),
            nn.BatchNorm2d(2*h)
        )
        
    def forward(self, x):
        out = self.net(x) + x
        return out


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

    print("Testing Conv Mask A Block")
    convMaskABlock = ConvMaskABlock(1, 1, 7, 1, 0)
    x = torch.randn(10, 1, 7, 7)
    print(convMaskABlock(x))
    
    print("Testing Conv Mask B Block")
    convMaskBBlock = ConvMaskBBlock(2, 3, 1, 1)
    x = torch.randn(10, 4, 3, 3)
    print(convMaskBBlock(x).shape)
