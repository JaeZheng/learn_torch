#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/27 22:00
# @File    : torch_demo.py

from __future__ import print_function
import torch

# 构建一个5*3矩阵，不初始化
x = torch.empty(5, 3)
print(x)
# 构建随机初始化的矩阵
x = torch.rand(5, 3)
print(x)
# 构造一个矩阵全为 0，而且数据类型是 long.
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
# 构造一个张量，直接使用数据
x = torch.tensor([5.5, 3])
print(x)
# 创建一个 tensor 基于已经存在的 tensor。
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
# 获取张量的维度信息
print(x.size())
# 加法操作
y = torch.rand(5, 3)
print(x+y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
y.add_(x)
print(y)
# 注意: 任何使张量会发生变化的操作都有一个前缀'_'。例如：x.copy_(y), x.t_(), 将会改变 x.
# 改变大小：如果你想改变一个 tensor 的大小或者形状，你可以使用 torch.view:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
# 如果你有一个元素 tensor ，使用 .item() 来获得这个 value 。
x = torch.randn([2,2])
print(x)
for i in x:
    for j in i:
        print(j.item())
