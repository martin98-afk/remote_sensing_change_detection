from math import *

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


## data augmentation
# 随机调节色调、饱和度值
def randomHueSaturationValue(image_A, hue_shift_limit=(-30, 30),
                             sat_shift_limit=(-5, 5),
                             val_shift_limit=(-15, 15), ratio=1.):
    if np.random.random() < ratio:
        image_A = cv2.cvtColor(image_A, cv2.COLOR_RGB2HSV)
        h_A, s_A, v_A = cv2.split(image_A)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h_A += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s_A = cv2.add(s_A, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v_A = cv2.add(v_A, val_shift)
        image_A = cv2.merge((h_A, s_A, v_A))
        image_A = cv2.cvtColor(image_A, cv2.COLOR_HSV2RGB)
    return image_A


# 随机移位旋转
def randomShiftScaleRotate(image_A, mask,
                           shift_limit=(-0.1, 0.1),
                           scale_limit=(-0.1, 0.1),
                           aspect_limit=(-0.1, 0.1),
                           rotate_limit=(-0, 0),
                           borderMode=cv2.BORDER_CONSTANT, ratio=0.5):
    if np.random.random() < ratio:
        height, width, channel = image_A.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image_A = cv2.warpPerspective(image_A, mat, (width, height), flags=cv2.INTER_LINEAR,
                                      borderMode=borderMode,
                                      borderValue=(
                                          0, 0,
                                          0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image_A, mask


def gauss_blur(img, ksize=(3, 3), sigma=0):
    '''
    高斯模糊
    :param img: 原始图片
    :param ksize: 高斯内核大小。 ksize.width和ksize.height可以不同，但​​它们都必须为正数和奇数，也可以为零
    :param sigma: 标准差，如果写0，则函数会自行计算
    :return:
    '''
    # 外部调用传入正整数即可,在这里转成奇数
    k_list = list(ksize)
    kw = (k_list[0] * 2) + 1
    kh = (k_list[1] * 2) + 1
    resultImg = cv2.GaussianBlur(img, (kw, kh), sigma)
    return resultImg


def blur(img, ksize=(5, 5)):
    '''
    均值模糊
    :param img: 原始图片
    :param ksize: 模糊内核大小
    :return:
    '''
    resultImg = cv2.blur(img, ksize)
    return resultImg


def median_blur(img, m=3):
    '''
    中值模糊
    :param img: 原始图片
    :param m: 孔径的尺寸，一个大于1的奇数
    :return:
    '''
    resultImg = cv2.medianBlur(img, m)
    return resultImg


def random_blur(image, mask, ratio=0.5):
    if np.random.random() < ratio / 3:
        image = gauss_blur(image)
    if np.random.random() < ratio / 3:
        image = median_blur(image)
    if np.random.random() < ratio / 3:
        image = blur(image)
    return image, mask


def random_crop(image, mask, ratio=0.5):
    if np.random.random() < ratio:
        (h, w) = image.shape[:2]
        scale = np.random.uniform(1.01, 1.2)
        image = cv2.resize(image, (int(scale * h), int(scale * w)))
        mask = cv2.resize(mask, (int(scale * h), int(scale * w)), interpolation=cv2.INTER_NEAREST)
        (nh, nw) = image.shape[:2]
        startx = np.random.randint(0, nh - h)
        starty = np.random.randint(0, nw - w)
        return image[startx:startx + h, starty:starty + w], mask[startx:startx + h,
                                                            starty:starty + w]
    return image, mask


def rotate_bound(image, mask, ratio=0.5):
    if np.random.random() < ratio:
        angle = np.random.randint(-45, 45)
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        rotate_center = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        # 计算图像新边界
        nH = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
        nW = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (nW - w) / 2
        M[1, 2] += (nH - h) / 2

        # perform the actual rotation and return the image'
        image, mask = cv2.warpAffine(image, M, (nW, nH)), cv2.warpAffine(mask, M, (nW, nH))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return image, mask


# def randomStyleTransfer(image_A, ratio):
#     if np.random.random() < ratio:
#         image = np.random.choice(image_list)
#         image_A = style_transfer(image_A, image)
#     return image_A


def randomHorizontalFlip(image_A, mask, ratio=0.5):
    if np.random.random() < ratio:
        image_A = cv2.flip(image_A, 1)
        mask = cv2.flip(mask, 1)

    return image_A, mask


def randomVerticleFlip(image_A, mask, ratio=0.5):
    if np.random.random() < ratio:
        image_A = cv2.flip(image_A, 0)
        mask = cv2.flip(mask, 0)

    return image_A, mask


def randomRotate90(image_A, mask, ratio=0.5):
    if np.random.random() < ratio:
        image_A = np.rot90(image_A).copy()
        mask = np.rot90(mask).copy()

    return image_A, mask


def data_agu_ss(image_A, label, ratio=0.5):
    """
    对图像数据集进行数据增强

    :param image_A:
    :param label:
    :param ratio:
    :return:
    """

    image_A = randomHueSaturationValue(image_A, ratio=ratio)

    # image_A = randomStyleTransfer(image_A, ratio=ratio)

    image_A, label = random_crop(image_A, label, ratio)

    image_A, label = random_blur(image_A, label, ratio)

    image_A, label = randomHorizontalFlip(image_A, label, ratio)

    image_A, label = randomVerticleFlip(image_A, label, ratio)

    image_A, label = randomRotate90(image_A, label, ratio)

    image_A, label = rotate_bound(image_A, label, ratio)

    return image_A, label


if __name__ == "__main__":
    tif = Image.open("../real_data/processed_data/2020_2_1_res_0.8.tif")
    tif = np.array(tif)
    mask = Image.open("../real_data/semantic_mask/2020_2_1_res_0.8.tif")
    mask = np.array(mask)
    tif, mask = rotate_bound(tif, mask, 15)
    fg, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(tif)
    ax[1].imshow(mask)
    plt.show()
