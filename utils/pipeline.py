import gc
import os.path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from osgeo import gdal
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from models.unet import get_semantic_segment_model, edge_ce_dice_loss
from models import lovasz_losses as L
from models.hrnet import hrnet18
from utils.counter import AverageMeter
from utils.datasets import SSDataRandomCrop
from utils.polygon_utils import get_mask
from utils.visualize_result import ResultVisualization

os.makedirs("./real_data/semantic_mask", exist_ok=True)


class RSPipeline(object):

    def __init__(self, ind2label,
                 num_classes,
                 args):
        """
        初始化遥感识别系统管道流程。

        :param batch_size: 批量大小
        :param num_classes: 分类类别数
        :param epochs: 训练轮数
        :param image_size: default 512
        :param model_name: default efficientnet-b0
        :param model_name: 使用的模型类型
        :param num_workers: 数据读取使用的线程数
        :param ohem: 是否启用在线难例挖掘损失函数
        :param fp_16: 是否使用半精度训练
        :param device: default cpu
        """
        self.batch_size = args.batch_size
        self.ind2label = ind2label
        self.num_classes = num_classes
        self.ignore_background = args.ignore_background
        self.ign_lab = num_classes - 1 if self.ignore_background else None
        self.image_size = args.image_size[0]
        self.device = args.device
        self.model_name = args.model_name
        self.model_save_path = args.model_save_path
        self.fp16 = args.fp16
        self.epochs = args.epochs
        self.train_size = args.train_size
        self.val_size = args.val_size
        self.data_path = args.data
        self.num_workers = args.num_workers
        self.ohem = args.ohem
        self.swa_model, self.model, self.optimizer = self.init_model()
        self.output_model_info()
        self.train_loader, self.val_loader = self.build_dataset()

    def init_model(self):
        """
        初始化模型

        :return:
        """
        if self.model_name != "hrnet18":
            model = get_semantic_segment_model(num_classes=self.num_classes,
                                               model_name=self.model_name,
                                               device=self.device,
                                               pretrained_path=self.model_save_path)
        else:
            model = hrnet18(num_classes=self.num_classes, pretrained=False)
        model = model.to(self.device)
        # 优化器，使用adamw优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        swa_model = AveragedModel(model).to(self.device)
        RSPipeline.print_log("模型创建完毕")
        return swa_model, model, optimizer

    def change_resolution(self, image_size):
        """
        更换模型训练图像分辨率.

        :param image_size:
        :return:
        """
        self.image_size = image_size
        self.train_loader, self.val_loader = self.build_dataset()
        self.output_model_info()

    def build_dataset(self):
        """
        构建模型训练数据集，包括训练集，验证集

        :return:
        """
        mask_paths = glob(self.data_path + "/*.tif")
        mask_paths = [RSPipeline.check_path(path) for path in mask_paths]
        image_paths = [path.replace("semantic_mask", "processed_data")
                       for path in mask_paths]
        # 图片读取
        image_list = [Image.open(image_path) for image_path in image_paths]
        mask_list = [Image.open(mask_path) for mask_path in mask_paths]

        image_list = [np.asarray(image)[..., :3] for image in image_list]
        mask_list = [np.asarray(mask) for mask in mask_list]
        # 数据集构建
        trainset = SSDataRandomCrop(image_list, mask_list, mode="train",
                                    img_size=self.image_size, length=self.train_size)
        valset = SSDataRandomCrop(image_list, mask_list, mode="val",
                                  img_size=self.image_size, length=self.val_size)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                                 pin_memory=False, num_workers=self.num_workers, drop_last=True)
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False,
                               pin_memory=True, num_workers=self.num_workers, drop_last=False)

        for img1, mask_bin, id in trainloader:
            print('训练数据形状: ', img1.shape)
            print('标签数据形状: ', mask_bin.shape)
            break

        return trainloader, valloader

    def get_model_summary(self):
        """
        获得模型结构。

        :return:
        """
        RSPipeline.print_log("打印模型结构")
        summary(self.model, input_size=(2, 3, self.image_size, self.image_size))

    def freeze_model(self):
        """
        冻结模型参数。

        :return:
        """
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_model(self):
        """
        解冻模型参数。

        :return:
        """
        for p in self.model.parameters():
            p.requires_grad = True

    def train_fn(self, scaler):
        """
        训练函数

        :param epoch:
        :param scaler:
        :return:
        """
        self.model.train()

        summary_loss = AverageMeter()
        summary_acc = AverageMeter()
        tk0 = tqdm(self.train_loader)
        for step, (img1, mask_bin, id) in enumerate(tk0):
            self.optimizer.zero_grad()
            img1 = img1.float().to(self.device)
            labels = mask_bin.long().to(self.device)
            # 根据不同选择使用半精度和单精度训练
            if scaler is None:
                output = nn.Softmax(dim=1)(self.model(img1))
                if self.ohem:
                    losses = edge_ce_dice_loss(output, labels, self.ign_lab, self.ohem)
                else:
                    losses = L.lovasz_softmax(output, labels)
                losses.backward()
                self.optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    output = nn.Softmax(dim=1)(self.model(img1))
                    if self.ohem:
                        losses = edge_ce_dice_loss(output, labels, self.ign_lab, self.ohem)
                    else:
                        losses = L.lovasz_softmax(output, labels)
                scaler.scale(losses).backward()
                scaler.step(self.optimizer)
                scaler.update()

            output = np.argmax(output.cpu().data.numpy(), axis=1)
            acc = f1_score(output.reshape(-1, 1), labels.cpu().data.numpy().reshape(-1, 1),
                           average='macro')
            summary_acc.update(acc.item(), self.batch_size)
            summary_loss.update(losses.item(), self.batch_size)
            tk0.set_postfix(loss=summary_loss.avg, f1_score=summary_acc.avg)
        return summary_loss, summary_acc

    def eval_fn(self, visualize):
        """
        验证函数

        :param visualize:
        :return:
        """
        self.model.eval()
        summary_f1 = AverageMeter()
        summary_acc = AverageMeter()
        rs = ResultVisualization(num_classes=self.num_classes)
        # 验证集不需要梯度计算,加速和节省gpu空间
        with torch.no_grad():
            tk0 = tqdm(self.val_loader)
            for step, (img1, mask_bin, id) in enumerate(tk0):
                img1 = img1.float().to(self.device)
                labels = mask_bin.to(self.device)
                # 输入模型获得预测分割图像
                output = nn.Softmax(dim=1)(self.model(img1))
                output = np.argmax(output.cpu().data.numpy(), axis=1)
                labels = labels.cpu().data.numpy()
                # 对于需要忽略的标签位置直接置为忽略标签，避免影响准确率
                if self.ign_lab is not None:
                    output[labels == self.ign_lab] = self.ign_lab
                #
                f1 = f1_score(output.reshape(-1, 1), labels.reshape(-1, 1),
                              average='macro')
                acc = accuracy_score(output.reshape(-1, 1),
                                     labels.reshape(-1, 1))
                summary_acc.update(acc.item(), self.batch_size)
                summary_f1.update(f1.item(), self.batch_size)
                tk0.set_postfix(acc=summary_acc.avg, f1_sc=summary_f1.avg)
                if step % 20 == 0 and visualize:
                    RSPipeline.print_log("验证图像打印中")
                    rs.visualize_semantic_segment_result(img1.cpu().numpy(),
                                                         labels.cpu().data.numpy(),
                                                         output)

        return summary_acc, summary_f1

    def run(self, visualize=False):
        """
        代入模型进行训练，使用交叉熵损失+dice损失，可以解决类别不均衡问题，学习率使用的是余弦退火学习率，训练过程保存最优模型。

        :param visualize: 是否输出验证结果图像
        :return:
        """
        # 选择是否使用半精度训练
        if self.fp16:
            from torch.cuda.amp import GradScaler as GradScaler
            scaler = GradScaler()
        else:
            scaler = None

        # 余弦退火调整学习率
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=2,  # T_0就是初始restart的epoch数目
            T_mult=2,  # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
            eta_min=1e-5  # 最低学习率
        )

        best_acc = 0
        RSPipeline.print_log("模型开始训练")
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_fn(scaler)
            valid_acc, valid_f1 = self.eval_fn(visualize)
            RSPipeline.print_log(
                '|EPOCH {}| TRAIN_LOSS {}| TRAIN_F1_SCORE {}| VALID_ACC {}|  VALID_F1_SCORE {}|'.format(
                    epoch + 1, train_loss.avg, train_acc.avg, valid_acc.avg, valid_f1.avg))
            scheduler.step()
            if valid_f1.avg > best_acc and epoch > 10:
                best_acc = valid_f1.avg
                RSPipeline.print_log(f"在当前训练轮数中找到最佳模型，训练轮数为{epoch + 1}")
                self.swa_model.update_parameters(self.model)
                torch.save(self.model.state_dict(), self.model_save_path)
                RSPipeline.print_log("保存模型中")

        # Update bn statistics for the swa_model at the end
        torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model)
        torch.save(self.swa_model.state_dict(), self.model_save_path.replace(".pth", "last.pth"))
        RSPipeline.print_log("最终混合模型保存中")
        RSPipeline.print_log("训练过程结束")
        gc.collect()

    def output_model_info(self):
        """
        输出模型参数为yaml文件

        :return:
        """
        to_yaml = {"image_size": [self.image_size, self.image_size],
                   "ignore background": self.ignore_background,
                   "index to label": self.ind2label,
                   "device": self.device,
                   "epochs": self.epochs,
                   "model_name": self.model_name,
                   "save_path": self.model_save_path,
                   "num_classes": self.num_classes,
                   "batch_size": self.batch_size,
                   "OHEM loss": self.ohem,
                   "fp16training": self.fp16
                   }
        if os.path.exists(self.model_save_path.replace("pth", "yaml")):
            os.remove(self.model_save_path.replace("pth", "yaml"))
        with open(self.model_save_path.replace("pth", "yaml"), "w", encoding="utf-8") as f:
            yaml.dump(to_yaml, f, default_flow_style=False)

    @staticmethod
    def load_model(model_info_path, device='cuda'):
        """
        根据提供的yaml文件路径，解析模型参数，并导入模型。

        :param model_info_path:
        :return:
        """
        with open(model_info_path, 'r', encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        RSPipeline.print_log("根据yaml文件导入模型中")
        num_classes = data['num_classes']
        pretrained_path = data['save_path']
        model_name = data['model_name']

        if model_name != "hrnet18":
            model = get_semantic_segment_model(num_classes=num_classes,
                                               model_name=model_name,
                                               device=device,
                                               pretrained_path=pretrained_path)
        else:
            model = hrnet18(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(pretrained_path, map_location=device))

        model.to(device)
        model.eval()
        return data, model

    @staticmethod
    def check_file(path):
        """
        检查文件是否存在，如果存在则删除.

        :param path:
        :return:
        """
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def print_log(information):
        """
        打印日志

        :param information:
        :return:
        """
        print(66 * "-")
        print(int(33 - len(information) - 1) * "-", information,
              int(33 - len(information) - 1) * "-")

    @staticmethod
    def update_polygon(args, image_path, shp_path, num_classes, ind2num):
        """
        提供年份，地点，块标号以及矢量文件地址，将矢量文件批量转换为像素标签

        :param year:
        :param place:
        :param part:
        :param num_classes:
        :param shp_path:
        :param ind2num:
        :return:
        """
        RSPipeline.print_log("开始更新矢量标签数据为像素标签数据")
        # 获取所有填充后的标签的保存路径
        save_path = args.label + image_path.split('/')[-1]

        image = gdal.Open(image_path)
        image_shape = image.ReadAsArray().shape
        # 初始化标签
        mask = num_classes * np.ones((image_shape[1], image_shape[2]))
        # 根据对应的shp文件将标签进行填充
        mask = get_mask(image, mask, shp_path, ind2num)
        # 将标签转为Image形式，方便保存
        mask = Image.fromarray(mask.astype(np.uint8))
        RSPipeline.check_file(save_path)
        mask.save(save_path)
        RSPipeline.print_log("更新结果完毕，数据已保存")

    @staticmethod
    def check_path(path):
        return path.replace("\\", "/")
