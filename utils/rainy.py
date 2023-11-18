import os
import cv2
import random
import numpy as np


# 运动模糊
def motion_blur(img, degree=10, angle=90):
    M = cv2.getRotationMatrix2D((degree/2,degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))

    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel,M ,(degree,degree))
    motion_blur_kernel = motion_blur_kernel / degree

    blurred = cv2.filter2D(img, -1, motion_blur_kernel)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


def rain(rain_image):
    bg = np.zeros_like(rain_image, dtype='uint8')
    rain_list = []
    for i in range(150):
        w,h,_ = bg.shape
        x = random.randrange(0, h)
        y = random.randrange(0, w)
        sx = random.randint(-1, 1)
        sy = random.randint(1, 2)
        rain_list.append([x, y, sx, sy])

    for i in range(len(rain_list)):
        # 绘制，颜色、位置、大小
        xi = rain_list[i][0]
        yi = rain_list[i][1]
        cv2.circle(bg, (xi, yi), rain_list[i][3], thickness=-1, color=(150,150,150))

    return bg


def add_rain(image):
    rain_image = rain(image)
    rain_image = motion_blur(rain_image)
    image = cv2.add(image, rain_image)
    return image


if __name__ == '__main__':

    image_folder = 'data/original'
    output_file = 'datasets2/labels/rainy.txt'
    # 获取图片文件夹中的所有文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    with open(output_file, 'w') as file:

        for image_file in image_files:
            # 构建完整的图片文件路径
            image_path = os.path.join(image_folder, image_file)

            # 从文件加载图像
            image = cv2.imread(image_path)

            # 确保图像是 NumPy 数组并具有正确的数据类型
            image = image.astype(np.uint8)

            # 在这里执行你的处理操作，例如加雪、加雾等
            rain_image = rain(image)
            rain_image = motion_blur(rain_image)
            image = cv2.add(image, rain_image)

            img_info = "rainy/rain_" + image_file
            # print(img_info)
            if "busy" in img_info:
                label = 1
            else:
                label = 0

            file.write("{}\t{}\t{}\n".format(img_info, label, 1))

            # 保存处理后的图像到指定文件夹
            output_folder = 'datasets2/img/db1/rainy'
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, image)


