# encoding:utf-8
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.augment_semantic_segment_image import data_agu_ss, add_featrue


class SSDataRandomCrop(Dataset):
    """
        根据大遥感地图随机裁剪的语义分割数据集，包含训练、验证、测试三部分数据集的构建。
    """

    def __init__(self, image_list, mask_list, mode, length, img_size=512, train_test_split=0.8):
        """

        :param image_list: 图像数据集
        :param mask_list: 标签数据集
        :param mode: 训练模型
        :param length: 随机切割图片的数据量
        :param img_size: 切割的图像大小
        """
        super(SSDataRandomCrop, self).__init__()
        self.mode = mode
        self.image_list = image_list
        self.mask_list = mask_list
        self.img_size = img_size
        self.length = length
        self.train_test_split = train_test_split
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        if self.mode == "test":
            image = add_featrue(self.image_list[index])
            # image = self.image_list[index]
            return self.image_list[index], self.normalize(image)
        else:
            select_index = np.random.randint(len(self.image_list))
            image = self.image_list[select_index]
            mask = self.mask_list[select_index]

            image_shape = image.shape
            size = self.img_size
            if self.mode == "train":
                x_rd = int(
                        np.random.random() * (int(image_shape[0] * self.train_test_split) - size))
                y_rd = int(np.random.random() * (image_shape[1] - size))
                image = image[x_rd:x_rd + size, y_rd:y_rd + size, :]
                mask = mask[x_rd:x_rd + size, y_rd:y_rd + size]
                image, mask = data_agu_ss(image, mask)
            elif self.mode == "val":
                x_rd = int(np.random.random() * (
                        int(image_shape[0] * (1 - self.train_test_split)) - size)) + int(
                        image_shape[0] * self.train_test_split)
                y_rd = int(np.random.random() * (image_shape[1] - size))
                image = image[x_rd:x_rd + size, y_rd:y_rd + size, :]
                mask = mask[x_rd:x_rd + size, y_rd:y_rd + size]
            elif self.mode == "noval":
                x_rd = int(np.random.random() * (image_shape[0] - size))
                y_rd = int(np.random.random() * (image_shape[1] - size))
                image = image[x_rd:x_rd + size, y_rd:y_rd + size, :]
                mask = mask[x_rd:x_rd + size, y_rd:y_rd + size]
                image, mask = data_agu_ss(image, mask)
            mask = torch.from_numpy(mask)
            image = add_featrue(image)
            image = self.normalize(image)
            return image, mask, index

    def __len__(self):
        return self.length


# data
class SSData(Dataset):
    """提供图像路径作为数据集"""

    def __init__(self, root, mode, use_pseudo_label=False):
        super(SSData, self).__init__()
        self.root = root

        self.mode = mode
        self.use_pseudo_label = use_pseudo_label

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, 'images'))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'images'))
            self.ids.sort()
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = Image.open(os.path.join(self.root, 'images', id))
        img1 = np.asarray(img1)
        if self.mode != 'test':
            mask_bin = Image.open(os.path.join(self.root, 'labels', id.replace("jpg", "png")))
            mask_bin = np.asarray(mask_bin)
            ## image augmentation
            if self.mode == 'train':
                img1, mask_bin = data_agu_ss(img1, mask_bin)
            mask_bin = torch.from_numpy(mask_bin)
            return self.normalize(img1), mask_bin, id
        else:
            return self.normalize(img1), id

    def __len__(self):
        return len(self.ids)
