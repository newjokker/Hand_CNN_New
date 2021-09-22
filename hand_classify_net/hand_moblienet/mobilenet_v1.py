# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import torch.nn as nn
from torchsummary import summary


class MobileNet(nn.Module):

    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            self.conv_bn(3, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1024)

    @staticmethod
    def conv_bn(dim_in, dim_out, stride):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def conv_dw(dim_in, dim_out, stride):
        return nn.Sequential(
            # 3 × 3 conv with group for less parameter
            nn.Conv2d(dim_in, dim_in, 3, stride, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(inplace=True),
            # 1 × 1 change channel number
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == "__main__":

    model = MobileNet()
    summary(model)




















