import torch
import os
from torch.utils.data import Dataset, DataLoader  # Dataset:自定义数据集的母类
from torchvision import transforms  # 图片的变换器
from PIL import Image  # PIL(Python Image Library)是python的第三方图像处理库
import pandas as pd
import random
from config import Parameters


# 路径
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)  # 获取上一层路径
data_path = parent_path + "/data"
# print(parent_path)


# 参数配置
size = Parameters.img_size
batchsz = Parameters.batch_size
input_size = Parameters.input_size
task = Parameters.task
weather = Parameters.weather
if weather == "mix" or task == "W":
    txtFile = "all_weather_labels.txt"
elif weather == "foggy":
    txtFile = "foggy_labels.txt"
elif weather == "rainy":
    txtFile = "rainy_labels.txt"
elif weather == "snowy":
    txtFile = "snowy_labels.txt"


class Patchs(Dataset):

    def __init__(self, root, tf, mode):   # mode:模式（训练 or 测试）   tf:transform对图像采取变换
        super(Patchs, self).__init__()

        self.root = root
        self.tf = tf

        # image_path, label   (路径+标签)
        self.images, self.labelsP, self.labelsW = self.load_txt(self.root + "/" + txtFile)

        # 从上面的全部图片信息中截取不同的数量用作不同用途
        if mode == 'train':
            self.images = self.images[:4000]
            self.labelsP = self.labelsP[:4000]
            self.labelsW = self.labelsW[:4000]
        elif mode == 'test':
            self.images = self.images[-2000:]
            self.labelsP = self.labelsP[-2000:]
            self.labelsW = self.labelsW[-2000:]

    def load_txt(self, filename):

        ''' 读取（加载）txt文件 '''
        images, labelsP, labelsW = [], [], []
        with open(filename, 'r', encoding='utf-8') as file:
            l = file.readlines()  # readlines 是一个列表，它会按行读取文件的所有内容

        for i in range(len(l)):
            # print(l[i])
            image, labelP, labelW = l[i].split('\t')
            images.append(self.root + '/' + image)
            labelsP.append(int(labelP))
            labelsW.append(int(labelW))

        assert len(images) == len(labelsP)  # 确保图片和标签的列表长度一致，不一致会报错
        return images, labelsP, labelsW

    def __len__(self):
        return len(self.images)    # 裁剪过后的长度

    def __getitem__(self, idx):
        # idx： [0 - len(images)]
        # self.images, self.labels
        # 图片信息目前不是想要的数据类型（需要转化为图片信息）
        # label: 0,1 标签信息已经是数据类型了
        img, labelP, labelW = self.images[idx], self.labelsP[idx], self.labelsW[idx]
        # print(img, label)

        img = self.tf(img)    # 变为数据
        labelP = torch.tensor(labelP)   # 把label也变为tensor类型
        labelW = torch.tensor(labelW)

        return img, labelP, labelW


# 对训练集数据进行0.5概率的水平翻转（左右镜像）
data_transforms ={
    'train': transforms.Compose([
        lambda x:Image.open(x).convert('RGB'),  # string path => image data (变为图像的数据类型)
        transforms.RandomHorizontalFlip(0.5),  # 水平角度翻转
        transforms.Resize([size, size]),   # 降维后图像的大小

        transforms.Resize([input_size, input_size]),
        transforms.ToTensor()]),

    'test': transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),  # string path => image data (变为图像的数据类型)
        transforms.Resize([size, size]),   # 降维后图像的大小

        transforms.Resize([input_size, input_size]),
        transforms.ToTensor()])
    }

train_datasets = Patchs(data_path, data_transforms['train'], mode='train')
test_datasets = Patchs(data_path, data_transforms['test'], mode='test')

train_loader = DataLoader(train_datasets, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=batchsz, shuffle=True)


if __name__ == '__main__':
    if weather == "mix" or task == "W":
        print("当前数据集为<混合天气>数据集")
    elif weather == "foggy":
        print("当前数据集为<雾天>数据集")
    elif weather == "rainy":
        print("当前数据集为<雨天>数据集")
    elif weather == "snowy":
        print("当前数据集为<雪天>数据集")
    print("----------------------------------------")
    a = len(train_loader.dataset)
    b = len(test_loader.dataset)
    c = a + b
    print('训练集的数量为：', a)
    print('测试集的数量为：', b)
    print('全部数据集数量为：', c)

    d = len(train_loader)
    e = len(test_loader)
    print("训练集的Batch数量为：", d)
    print("测试集的Batch数量为：", e)


    x, y1, y2 = next(iter(train_loader))
    print(x.shape, y1.shape, y2.shape)

    # count = 0
    # for i, x in enumerate(test_loader):
    #     count += 1
    #     print(count)


