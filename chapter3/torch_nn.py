#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/21 9:44
# @File    : torch_nn.py

import torch
import time

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # 取消注释以在GPU上运行

start = time.time()
# N是批量大小; D_in是输入维度;
# 49/5000 H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机输入和输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # forward
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算和打印损失
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop计算w1和w2相对于损耗的梯度
    grad_y_pred = 2.0*(y_pred-y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0]=0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

end = time.time()
print('Running Time: {}s'.format((end-start)))
# cpu time: 0.20948123931884766s
# gpu time: 3.6765434741973877s