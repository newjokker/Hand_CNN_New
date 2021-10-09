# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
sample = Variable(torch.ones(2,2))
a=torch.Tensor(2,2)
a[0,0]=5
a[0,1]=5
a[1,0]=5
a[1,1]=5
target = Variable (a)


# # sample = sample[:1, :1]
# # target = target[:1, :1]
#
# print(sample)
#
# print('-'*30)
#
# print(target)
#
# print('-'*30)
#
# criterion = nn.SmoothL1Loss()
# loss = criterion(sample, target)
#
# print(loss)



# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
#
# output = loss(input, target)
#
# print(input)
# print(target)
# print(output)
#
# output.backward()



m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(m(input), target)

print(m(input))
print(target)
print(output)


output.backward()




