# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import torch
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary

# --------------------------------------------------
# todo 使用卷积进行 resize 也是一种不错的方式
# --------------------------------------------------

# Training settings
batch_size = 64


# MNIST Dataset
train_dataset = datasets.MNIST(root='../Data/MNist/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../Data/MNist/', train=False, transform=transforms.ToTensor())


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class LeNet(nn.Module):
    def __init__(self, in_channels, init_weights=True, num_classes=10):
        super(LeNet, self).__init__()

        self.num_classes = num_classes

        if init_weights:
            self._initialize_weights()

        # for change dataset's shape padding for
        self.resize = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=3)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.resize(x)                              # 6 x 28 x 28 --> 6 x 32 x 32
        z1 = self.conv1(x)                              # 6 x 28 x 28
        a1 = F.relu(z1)                                 # 6 x 28 x 28
        a1 = F.max_pool2d(a1, kernel_size=2, stride=2)  # 6 x 14 x 14
        z2 = self.conv2(a1)                             # 16 x 10 x 10
        a2 = F.relu(z2)                                 # 16 x 10 x 10
        a2 = F.max_pool2d(a2, kernel_size=2, stride=2)  # 16 x 5 x 5
        flatten_a2 = a2.view(a2.size(0), -1)
        z3 = self.fc1(flatten_a2)
        a3 = F.relu(z3)
        z4 = self.fc2(a3)
        a4 = F.relu(z4)
        z5 = self.fc3(a4)
        return F.log_softmax(z5)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def train(epoch):
    #
    for batch_idx, (data, target) in enumerate(train_loader):
        # 使用当前命名空间中的 grad，所以需要 optimizer 每次清空
        optimizer.zero_grad()
        output = model(data)
        # 计算 loss
        loss = F.nll_loss(output, target)
        # 计算反向梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 日志
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data.item()))


def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))




if __name__ == "__main__":


    model_path = r"./model/demo.pth"
    # 加载模型
    #model = torch.load(model_path)
    model = LeNet(1)
    # 优化器需要和 model 绑定，因为要执行 model 参数的更新
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # summary(model)

    for epoch in range(10):
        train(epoch)
        test()

    # 保存模型
    torch.save(model, model_path)
