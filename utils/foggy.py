import os
import cv2
import numpy as np


# def add_fog(img):
#     # 随机确定雾的强度
#     low_intensity = np.random.randint(180, 226)
#     high_intensity = np.random.randint(low_intensity + 1, 256)
#
#     fog = np.zeros_like(img, dtype='uint8')
#     fog = np.random.randint(low_intensity, high_intensity, size=img.shape, dtype=np.uint8)
#
#     # 随机确定高斯模糊的核大小（应该是奇数）
#     kernel_size = np.random.choice([51, 71, 91, 111, 131, 151])
#     fog = cv2.GaussianBlur(fog, (kernel_size, kernel_size), 0)
#     fog = fog.astype(img.dtype)  # 使雾的数据类型与图像匹配
#
#     # 随机确定混合权重
#     alpha = np.random.uniform(0.1, 0.6)
#     beta = 1 - alpha
#     img_with_fog = cv2.addWeighted(img, alpha, fog, beta, 0)
#
#     return img_with_fog


def fog(img):
    fog = np.zeros_like(img, dtype='uint8')
    fog = np.random.randint(200, 256, size=img.shape, dtype=np.uint8)

    fog = cv2.GaussianBlur(fog, (101, 101), 0)
    fog = fog.astype(img.dtype)  # 将雾的数据类型与图像一致
    img_with_fog = cv2.addWeighted(img, 0.1, fog, 0.9, 0)
    return img_with_fog


def add_fog(image):
    return fog(image)


if __name__ == '__main__':
    # 指定图片文件夹的路径
    image_folder = 'data/original'
    output_file = 'datasets2/labels/foggy.txt'
    # 获取图片文件夹中的所有文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    with open(output_file, 'w') as file:

        for image_file in image_files:
            # 构建完整的图片文件路径
            image_path = os.path.join(image_folder, image_file)
            # print(image_file)

            # 从文件加载图像
            image = cv2.imread(image_path)

            # 确保图像是 NumPy 数组并具有正确的数据类型
            image = image.astype(np.uint8)

            # 在这里执行你的处理操作，例如加雪、加雾等
            image_with_fog = add_fog(image)

            img_info = "foggy/foggy_" + image_file
            # print(img_info)
            if "busy" in img_info:
                label = 1
            else:
                label = 0

            file.write("{}\t{}\t{}\n".format(img_info, label, 0))

            # cv2.imshow("", image_with_fog)
            # cv2.waitKey(0)
            # cv2.destroyWindow()


            # 保存处理后的图像到指定文件夹
            output_folder = 'datasets2/img/db1/foggy'
            output_path = os.path.join(output_folder, image_file)
            # print(output_path)
            # break
            cv2.imwrite(output_path, image_with_fog)

    # img_show()


