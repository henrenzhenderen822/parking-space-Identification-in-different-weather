import os
import cv2
import random
import numpy as np


def add_fog(img):
    fog = np.zeros_like(img, dtype='uint8')
    fog = np.random.randint(200, 256, size=img.shape, dtype=np.uint8)

    fog = cv2.GaussianBlur(fog, (101, 101), 0)
    fog = fog.astype(img.dtype)  # 将雾的数据类型与图像一致
    img_with_fog = cv2.addWeighted(img, 0.7, fog, 0.3, 0)

    return img_with_fog


def snow(snow_img):
    bg = np.zeros_like(snow_img, dtype='uint8')
    snow_list = []
    for i in range(150):
        w, h, _ = bg.shape
        x = random.randrange(0, h)
        y = random.randrange(0, w)
        sx = random.randint(-1, 1)
        sy = random.randint(1, 2)
        snow_list.append([x, y, sx, sy])

    # 雪花列表循环
    for i in range(len(snow_list)):
        # 绘制雪花，颜色、位置、大小
        xi = snow_list[i][0]
        yi = snow_list[i][1]
        cv2.circle(bg, (xi, yi), snow_list[i][3], thickness=-1, color=(255, 255, 255))

    return bg


def add_snow(image):
    img_with_fog = add_fog(image)
    bg = snow(img_with_fog)
    img_with_fog = cv2.add(img_with_fog, bg)

    return img_with_fog


if __name__ == '__main__':
    image_folder = 'data/original'
    # output_file = 'datasets2/labels/snowy_accuracy.txt'
    # 获取图片文件夹中的所有文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # with open(output_file, 'w') as file:
    for image_file in image_files:
        # 构建完整的图片文件路径
        image_path = os.path.join(image_folder, image_file)

        # 从文件加载图像
        image = cv2.imread(image_path)

        # 确保图像是 NumPy 数组并具有正确的数据类型
        image = image.astype(np.uint8)

        # 在这里执行你的处理操作，例如加雪、加雾等
        img_with_fog = add_fog(image)
        bg = snow(img_with_fog)
        img_with_fog = cv2.add(img_with_fog, bg)

        cv2.imshow("", img_with_fog)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

            # img_info = "snowy/snow_" + image_file
            # # print(img_info)
            # if "busy" in img_info:
            #     label = 1
            # else:
            #     label = 0
            #
            # file.write("{}\t{}\t{}\n".format(img_info, label, 2))
            #
            # # 保存处理后的图像到指定文件夹
            # output_folder = 'datasets2/img/db1/snowy'
            # output_path = os.path.join(output_folder, image_file)
            # cv2.imwrite(output_path, img_with_fog)


