'''配置'''

import argparse
import sys


config = {
    'batch_size': 64,
    'img_size': 64,
    'input_size': 64,
    'epochs': 30,
    'learning_rate': 0.001,
    "weather": "mix",    # mix, foggy, rainy, snowy
    "task": "W",     # P, W
    "model": "resnet18"
}

parser = argparse.ArgumentParser(description="demo of argparse")
# 通过对象的add_argument函数来增加参数。
parser.add_argument('-B', '--batch_size', default=config["batch_size"], type=int)
parser.add_argument('-I', '--img_size', default=config["img_size"], type=int)
parser.add_argument('--input_size', default=config["input_size"], type=int)
parser.add_argument('-E', '--epochs', default=config["epochs"], type=int)
parser.add_argument('-L', '--learning_rate', default=config["learning_rate"], type=float)
parser.add_argument("-W", "--weather", default=config["weather"], type=str)
parser.add_argument("-T", "--task", default=config["task"], type=str)
parser.add_argument("-M", "--model", default=config["model"], type=str)
args = parser.parse_args()


class Parameters():
    batch_size = args.batch_size         # 批大小
    img_size = args.img_size             # 图片大小
    input_size = args.input_size         # 输入大小
    epochs = args.epochs                 # 跑epochs轮数据集
    learning_rate = args.learning_rate   # 学习率大小
    weather = args.weather               # 天气
    task = args.task                     # 任务：训练天气识别或者车位识别
    if task == "W":
        classes = 3                      # calsses: 分类的类别数
    else:
        classes = 2
    model = args.model                   # 模型

    def show_args(self):
        return args


if __name__ == '__main__':
    print(args)
    # vars() 函数返回对象object的属性和属性值的字典对象。
    ap = vars(args)
    print(ap)
    print(ap['task'])
    print("Input argument is %s" %(sys.argv))