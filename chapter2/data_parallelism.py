#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/19 11:37
# @File    : data_parallelism.py


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2
batch_size = 30
data_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 生成一个玩具数据。你只需要实现 getitem.
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)


# 为了做一个小 demo，我们的模型只是获得一个输入，执行一个线性操作，然后给一个输出。
# 尽管如此，你可以使用 DataParallel 在任何模型(CNN, RNN, Capsule Net 等等.)
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


# 创建模型并且数据并行处理
# 如果我们有多个GPU，我们可以用 nn.DataParallel 来包裹我们的模型。
# 然后我们使用 model.to(device) 把模型放到多GPU中。
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), 'GPUs!')
    model = nn.DataParallel(model)
model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())