import random

from utils.foggy import add_fog
from utils.rainy import add_rain
from utils.snowy import add_snow
import os
import cv2
import numpy as np


def add_weather(weather: str):
    '''
    生成天气图像并记录标签数据
    :param weather:
    :return:
    '''
    image_folder = 'data/original'
    output_file = 'data/' + weather + '_labels.txt'

    # 获取图片文件夹中的所有文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    out_list = []

    for image_name in image_files:
        # 构建完整的图片文件路径
        image_path = os.path.join(image_folder, image_name)

        # 从文件加载图像
        image = cv2.imread(image_path)

        # 确保图像是 NumPy 数组并具有正确的数据类型
        image = image.astype(np.uint8)

        # 在这里执行你的处理操作，例如加雪、加雾等
        if weather == "foggy":
            dealed_img = add_fog(image)
            labelW = 0
        elif weather == "rainy":
            dealed_img = add_rain(image)
            labelW = 1
        elif weather == "snowy":
            dealed_img = add_snow(image)
            labelW = 2

        # 把图片信息写进txt
        label_info = weather + "/" + image_name
        if "busy" in label_info:
            labelP = 1
        else:
            labelP = 0

        out_list.append((label_info, labelP, labelW))

        output_img_info = "data/" + weather + "/" + image_name

        cv2.imwrite(output_img_info, dealed_img)

    count = len(out_list)
    random.shuffle(out_list)
    with open(output_file, 'w') as file:
        for label_info, labelP, labelW in out_list:
            file.write("{}\t{}\t{}\n".format(label_info, labelP, labelW))

    print("当前处理{}天气完毕，图像数量：{}".format(weather, count))

    return out_list


if __name__ == '__main__':
    mix_out_list = []
    for weather in ["foggy", "rainy", "snowy"]:
        mix_out_list.extend(add_weather(weather))
    # 将混合天气标签打乱并写入文件
    random.shuffle(mix_out_list)
    with open("data/all_weather_labels.txt", 'w') as file:
        for label_info, labelP, labelW in mix_out_list:
            file.write("{}\t{}\t{}\n".format(label_info, labelP, labelW))
