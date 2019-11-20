#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/19 10:30
# @File    : loss.py

import torch.nn as nn
import torch.optim as optim


def get_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer

