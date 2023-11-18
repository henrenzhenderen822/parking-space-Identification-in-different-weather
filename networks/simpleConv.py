import torch
from torch import nn
from config import Parameters


num_classes = Parameters.classes


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.conv = nn.Sequential(self.conv1, self.conv2)

        self.fc = nn.Sequential(
            nn.Linear(784, 60),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(60, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.contiguous().view(x.shape[0], -1)   # 打平操作
        # print(x.shape)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(64, 3, 64, 64)
    model = MyNet()
    print(model)
    pred = model(x)
    print(pred.shape)

    print("模型的参数量为: {}  ".format(sum(x.numel() for x in model.parameters())))





