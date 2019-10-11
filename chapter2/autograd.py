#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/10/11 21:53
# @File    : autograd.py

import torch

# 创建一个张量，设置requires_grad=True来跟踪与它相关的计算
x = torch.ones(2, 2, requires_grad=True)
print(x)

# y 作为操作的结果被创建，所以它有 grad_fn
y = x+2
print(y)
print(y.grad_fn)

z = y*y*3
out = z.mean()
print(z, out)

# .requires_grad_( ... ) 会改变张量的 requires_grad 标记。输入的标记默认为  False ，如果没有提供相应的参数。
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# 梯度。 我们现在后向传播，因为输出包含了一个标量，out.backward() 等同于out.backward(torch.tensor(1.))。
out.backward()
# 打印梯度  d(out)/dx
print(x.grad)

# 一个雅可比向量积的例子：
x = torch.randn(3, requires_grad=True)
y = x*2
while y.data.norm() < 1000:
    y = y*2
print(y)

# 可以通过将代码包裹在 with torch.no_grad()，来停止对从跟踪历史中 的 .requires_grad=True 的张量自动求导。
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)