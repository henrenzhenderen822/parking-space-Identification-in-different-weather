from torchvision import models
import torch
from torch import nn
from config import Parameters


ResNet = models.resnet18
num_ftrs = ResNet().fc.in_features    # 记录全连接层中的输入层的大小
num_classes = Parameters.classes
ResNet().fc = nn.Sequential(
    nn.Linear(num_ftrs, num_classes),    # 最后改成 num_classes 长的输出
)


if __name__ == '__main__':
    model = ResNet()
    x = torch.randn(16, 3, 64, 64)
    y = model(x)
    print(y.shape)



