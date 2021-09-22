# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

"""
* 为什么是 BN -> Relu -> Conv 这样的顺序？ 

"""

class Bottleneck(nn.Module):

    def __init__(self, n_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class Denseblock(nn.Module):

    def __init__(self, n_channels, growth_rate, n_dense_blocks):
        super(Denseblock, self).__init__()
        layers = []
        #
        for i in range(int(n_dense_blocks)):
            layers.append(Bottleneck(n_channels, growth_rate))
            n_channels += growth_rate
        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)



if __name__ == "__main__":

    model = Denseblock(64, 32, 6)

    summary(model)



