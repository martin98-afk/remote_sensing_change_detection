import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


#
class ResultVisualization(object):
    """
    对于地貌识别类别结果进行展示
    """

    def __init__(self, num_classes=8, class_names=None):
        self.num_classes = num_classes
        if os.path.exists(f'/home/xwtech/遥感识别专用/_COLORS_{num_classes}.csv'):
            _COLORS = pd.read_csv(f'/home/xwtech/遥感识别专用/_COLORS_{num_classes}.csv', header=None)
            ind2rgb = {}
            for i in range(num_classes):
                ind2rgb[i] = (
                    _COLORS.loc[i].values[0], _COLORS.loc[i].values[1], _COLORS.loc[i].values[2])
        else:
            ind2rgb = {}
            for i in range(num_classes):
                ind2rgb[i] = (
                    np.random.randint(255), np.random.randint(255), np.random.randint(255))

        self.rgb2ind = {v: k for k, v in ind2rgb.items()}
        self.ind2rgb = {k: v for k, v in ind2rgb.items()}
        self.class_names = class_names
        if class_names is not None:
            self.ind2label = {
                v: k for v, k in zip(np.arange(8), class_names)
            }
        else:
            self.ind2label = None

    def index2RGB(self, mask):
        new_image = np.zeros((mask.shape[0], mask.shape[1], 3))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                new_image[i, j, :] = self.ind2rgb[mask[i, j]]
        return new_image

    def show_label(self):
        assert self.class_names is not None, print("没有提供类别标签，无法显示！")
        sample_list = []
        for key in self.ind2rgb.keys():
            sample = np.zeros((10, 10, 3))
            sample[:, :, 0] = self.ind2rgb[key][0]
            sample[:, :, 1] = self.ind2rgb[key][1]
            sample[:, :, 2] = self.ind2rgb[key][2]
            sample_list.append(sample)
        fg, ax = plt.subplots(1, len(self.ind2rgb.keys()),
                              figsize=(len(self.ind2rgb.keys()) * 2, 1))
        for i, sample in enumerate(sample_list):
            ax[i].imshow(sample.astype('uint8'))
            ax[i].set_title(self.ind2label[i])
            ax[i].axis('off')
        plt.show()

    def visualize_change_detect_result(self, image1, image2, true_mask, predict_mask):
        image1 = np.transpose(image1, (0, 2, 3, 1))
        image2 = np.transpose(image2, (0, 2, 3, 1))
        fg, ax = plt.subplots(image1.shape[0], 4, figsize=(20, image1.shape[0] * 5))
        for i in range(image1.shape[0]):
            ax[i, 2].imshow(true_mask[i, ...])
            ax[i, 3].imshow(predict_mask[i, ...])
            ax[i, 2].set_title('ground_truth')
            ax[i, 3].set_title('predict mask')
            ax[i, 0].imshow((image1[i, ...] * 255).astype('uint8'))
            ax[i, 1].imshow((image2[i, ...] * 255).astype('uint8'))
            ax[i, 0].set_title('image1')
            ax[i, 1].set_title('image2')
        plt.show()

    def visualize_semantic_segment_result(self, image1, true_mask, predict_mask):
        image1 = np.transpose(image1, (0, 2, 3, 1))
        for i in range(image1.shape[0]):
            fg, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[1].imshow(self.index2RGB(true_mask[i, ...]).astype('uint8'))
            ax[2].imshow(self.index2RGB(predict_mask[i, ...]).astype('uint8'))
            ax[1].set_title('ground_truth')
            ax[2].set_title('predict mask')
            ax[0].imshow((image1[i, ...] * 255).astype('uint8'))
            ax[0].set_title('image')
            plt.show()
        if self.class_names is not None:
            self.show_label()

    def visualize_class_distribution(self, image):
        list = []
        for i in range(self.num_classes):
            list.append(np.sum(image == i))
        plt.bar(np.arange(self.num_classes), list)
        plt.title("the distribution of pixels from different classes")
        plt.show()


def visualize_image_bar(path, num_classes):
    """
    统计该图像各像素点的像素数量。

    :param path:
    :param num_classes:
    :return:
    """
    image = Image.open(path)
    image = np.array(image)
    count_list = [np.sum(image == i) for i in range(num_classes)]
    plt.bar(np.arange(num_classes), count_list)
    plt.title("sum pixels of each class")
    plt.show()


if __name__ == '__main__':
    # image1 = "../output/semantic_result/tif/2020_2_4_res_0.5_semantic_result.tif"
    # real_image = "../real_data/semantic_mask/2020_2_4_res_0.5.png"
    # vr = ResultVisualization(num_classes=9)
    # vr.visualize_class_distribution(np.asarray(Image.open(image1)))
    # vr.visualize_class_distribution(np.asarray(Image.open(real_image)))
    visualize_image_bar("../real_data/semantic_mask/2020_2_1_res_0.5.png", 6)
