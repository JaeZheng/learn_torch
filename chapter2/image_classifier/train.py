#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/19 10:31
# @File    : train.py

from data import get_train_data_loader, get_test_data_loader
from loss import get_criterion, get_optimizer
from model import Net
import datetime
import torch


def train():
    # 有GPU的话优先使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model = Net()  # 创建模型
    # model.to(device)
    train_data_loader = get_train_data_loader()
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    num_epoch = 5
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            # 得到输入数据
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            # 梯度置零
            optimizer.zero_grad()
            # 前传+后传+梯度更新
            outpus = model(inputs)
            loss = criterion(outpus, labels)
            loss.backward()
            optimizer.step()
            # 输出结果
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个mini-batch打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    save_path = "LeNet.pth"
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    start = datetime.datetime.now()
    train()
    end = datetime.datetime.now()
    print("Running Time: {}".format((end-start).seconds))