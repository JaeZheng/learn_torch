#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/19 10:31
# @File    : test.py

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from data import get_test_data_loader
from model import Net

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_test_sample():
    test_data_loader = get_test_data_loader()
    dataiter = iter(test_data_loader)
    images, labels = dataiter.next()

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1,2,0)))
        plt.show()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


def predict():
    # 加载模型
    load_path = "LeNet.pth"
    model = Net()
    model.load_state_dict(torch.load(load_path))
    # 在运行推理之前，务必调用model.eval() 设置 dropout 和 batch normalization 层为评估模式。
    # 如果不这么做，可能导致模型推断结果不一致。
    model.eval()
    # 加载测试数据
    test_data_loader = get_test_data_loader()
    correct = 0
    total = 0
    # 整体准确率
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    # 各个类别准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    # show_test_sample()
    predict()