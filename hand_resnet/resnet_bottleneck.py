# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import torch.nn as nn
from torchsummary import summary

"""
* 为什么 bottlenect 的 conv, BN, Relu 一组结构最后没有 Relu 
* 
"""


class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super (Bottleneck, self).__init__()
        # conv, BN, Relu | Conv, BN, Relu | Conv, BN
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.relu = nn.ReLU(inplace=True)
        # change chennel as same as bottleneck
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, stride),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


if __name__ == "__main__":

    model = Bottleneck(3, 3, 2)

    data = torch.ones(1, 3, 20, 20)

    b = model(data)

    print(b.shape)

    print(b)

    summary(model)












