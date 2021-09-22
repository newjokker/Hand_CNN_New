# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch.nn as nn

class InverteResidual(nn.Module):

    def __init(self, inp, oup, stride, expend_ratio):
        super(InverteResidual, self).__init__()

        self.stride = stride
        hidden_dim = round(inp * expend_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            #
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            #
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


if __name__ == "__main__":

    pass


