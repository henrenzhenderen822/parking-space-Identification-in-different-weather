''' 数据集之间相互训练测试 '''

import torch
from torch import nn, optim
import os
import time
from datasets.dataset import train_loader
from datasets.dataset import test_loader
from config import Parameters
from matplotlib import pyplot as plt
import numpy as np
from networks.resnet import ResNet
from networks.simpleConv import MyNet

# 设置随机数种子，确保每次的初始化相同
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print("当前脚本的参数如下：")
print(Parameters().show_args())    # 打印脚本的所有参数

# 训练参数
epochs = Parameters.epochs
leaning_rate = Parameters.learning_rate
img_size = Parameters.img_size

# 其他参数
weather = Parameters.weather
task = Parameters.task   # 0表示天气分类，1表示车位分类
define_model = Parameters.model

if task == "W":
    result_file = "weather_accuracy.txt"
    print("当前训练<天气分类器>")
else:
    if weather == "mix":
        result_file = "mix_accuracy.txt"
        print("当前训练<混合天气-车位分类器>")
    elif weather == "foggy":
        result_file = "foggy_accuracy.txt"
        print("当前训练<雾天-车位分类器>")
    elif weather == "rainy":
        result_file = "rainy_accuracy.txt"
        print("当前训练<雨天-车位分类器>")
    elif weather == "snowy":
        result_file = "snowy_accuracy.txt"
        print("当前训练<雪天-车位分类器>")

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
if define_model == "resnet18":
    model = ResNet().to(device)
else:
    model = MyNet().to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=leaning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)   # 学习率每4个epoch衰减成原来的1/2。


# 训练、验证
def main():

    since = time.time()
    now = time.strftime('%m-%d_%H%M')  # 结构化输出当前的时间
    with open("result/" + result_file, "a", encoding="utf-8") as f:
        f.write("\n\n时间:{}\n".format(now))
        f.write(str(Parameters().show_args()) + "\n")

    for epoch in range(epochs):
        # print('Epoch:{} 当前学习率为:{}'.format(epoch, scheduler.get_last_lr()))
        # 训练
        model.train()
        for batchidx, (x, labelP, labelW) in enumerate(train_loader):
            if task == "P":
                x, label = x.to(device), labelP.to(device)
            else:
                x, label = x.to(device), labelW.to(device)
            logits = model(x)
            loss = criteon(logits, label)  # 损失函数

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 训练集loss值
        print("EPOCH:{}\t\tLOSS:{}".format(epoch, loss.item()))
        # scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():  # 表示测试过程不需要计算梯度信息
            total_correct = 0
            total_num = 0
            for x, labelP, labelW in test_loader:
                if task == "P":
                    x, label = x.to(device), labelP.to(device)
                else:
                    x, label = x.to(device), labelW.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()  # 统计预测对的数量
                total_num += x.size(0)
        acc = total_correct / total_num  # 准确度
        print('Epoch:{}\t\tACCURACY:{:.2f}%'.format(epoch, acc*100))
        with open("result/" + result_file, "a", encoding="utf-8") as f:
            f.write('Epoch:{}\tACCURACY:{:.2f}%\n'.format(epoch, acc*100))

    # 保存最后一轮训练完成后的网络模型
    state = {
        'state_dict': model.state_dict(),       # 模型参数
        'optimizer': optimizer.state_dict(),    # 模型优化器
        'model_struct': model,                  # 模型结构
    }
    if task == "W":
        path = "checkpoints/weather"
    else:
        path = "checkpoints/" + weather
    torch.save(state, os.path.join(path, now + '.pth'))  # 以时间命名模型保存下来

    time_elapsed = time.time() - since
    print('time_elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':

    main()
