import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np 

# 使用 torchvision ,用它来加载 CIFAR10 数据非常简单
# torchvision 数据集的输出是范围在[0,1]之间的 PILImage，
# 我们将他们转换成归一化范围为[-1,1]之间的张量 Tensors。


def get_train_data_loader():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    return trainloader


def get_test_data_loader():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False)
    return testloader


# 展示训练图片
def show_sample_data():
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1,2,0)))
        plt.show()

    # 随机拿到一些训练图片
    train_loader = get_train_data_loader()
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ == '__main__':
    show_sample_data()