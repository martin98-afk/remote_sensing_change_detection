# encoding:utf-8
import gc
import os.path
from glob import glob
from os import path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from models import lovasz_losses as L
from models.swin_unet import SwinTransformerSys
from models.unet import get_semantic_segment_model, edge_ce_dice_loss
from utils.counter import AverageMeter
from utils.datasets import SSDataRandomCrop
from utils.polygon_utils import read_shp, shp2tif
from utils.visualize_result import ResultVisualization
from remotesensing_alg.fast_glcm import *

os.makedirs("./real_data/semantic_mask", exist_ok=True)


class RSPipeline(object):

    def __init__(self, ind2label,
                 num_classes, in_channels,
                 args):
        """
        初始化遥感识别系统管道流程。

        :param num_classes: 分类类别数，
        :param in_channels: 输入通道数，
        :param args: 通过命令行传入的超参数,
        :param ind2label: 数值标签与中文标签的对应关系字典
        """
        self.batch_size = args.batch_size   # 批训练大小
        self.ind2label = ind2label    # 数值标签与中文标签对应字典
        self.num_classes = num_classes   # 待识别地貌类型数量
        self.ignore_background = args.ignore_background   #是否忽略背景类
        self.ign_lab = 0 if self.ignore_background else None  # 如果忽略背景类则定义背景值为0
        self.image_size = args.image_size[0]   # 输入的图像大小
        self.device = args.device    # 使用的训练机器
        self.model_name = args.model_name   # 使用的模型名称
        self.onnx_model_name = self.model_name.replace("pth", "onnx")  # 如果导出onnx模型，onnx模型的地址
        self.pretrained_model_path = args.pretrained_model_path  # 预训练模型地址
        self.model_save_path = args.model_save_path   # 模型保存地址
        self.fp16 = args.fp16    # 是否使用半精度训练
        self.epochs = args.epochs    # 训练的总轮数
        self.train_size = args.train_size   # 训练数据大小
        self.in_channels = in_channels    # 输入通道数
        self.val_size = args.val_size    # 验证集大小
        self.label_path = args.label    # 标签文件存储路径
        self.num_workers = args.num_workers  # 训练时使用几个worker并行读取数据
        self.pop_head = args.pop_head   # 是否剔除当前模型的分割头
        self.ohem = args.ohem     # 是否使用在线难例挖掘
        self.model, self.optimizer = self.init_model()   # 初始化模型
        self.output_model_info()     # 将模型超参数输出为yaml文件
        self.train_loader, self.val_loader = self.build_dataset()  # 构建数据集

    def init_model(self):
        """
        初始化模型

        :return:
        """
        if "efficientnet" in self.model_name:
            model = get_semantic_segment_model(num_classes=self.num_classes,
                                               in_channels=self.in_channels,
                                               model_name=self.model_name,
                                               device=self.device,
                                               pretrained_path=self.pretrained_model_path,
                                               pop_head=self.pop_head)
        elif "swin-unet" in self.model_name:
            model = SwinTransformerSys(img_size=512, num_classes=9)
            model = nn.DataParallel(model)
            if self.pretrained_model_path is not None and os.path.exists(
                    self.pretrained_model_path):
                try:
                    ckpt = torch.load(self.pretrained_model_path,
                                      map_location=torch.device(self.device))
                    model.load_state_dict(
                            ckpt,
                            strict=False
                    )
                    print('历史最佳模型已读取!')
                except:
                    print("参数发生变化，历史模型无效！")
        # 优化器，使用adamw优化器
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        RSPipeline.print_log("模型创建完毕")
        return model, optimizer

    def build_dataset(self):
        """
        构建模型训练数据集，包括训练集，验证集

        :return:
        """
        mask_paths = glob(self.label_path)
        print(mask_paths)
        mask_paths = [RSPipeline.check_path(path) for path in mask_paths]
        image_paths = [path.replace("semantic_mask", "processed_data")
                       for path in mask_paths]
        image = [np.array(Image.open(path))[..., :3] for path in image_paths]
        mask = [np.array(Image.open(path)) for path in mask_paths]
        # 数据集构建
        trainset = SSDataRandomCrop(image, mask, mode="train",
                                    img_size=self.image_size, length=self.train_size)
        valset = SSDataRandomCrop(image, mask, mode="val",
                                  img_size=self.image_size, length=self.val_size)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                                 pin_memory=False, num_workers=self.num_workers, drop_last=True)
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False,
                               pin_memory=False, num_workers=self.num_workers, drop_last=False)

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
        冻结除分类头以外的所有模型参数。

        :return:
        """
        for i, p in enumerate(self.model.parameters()):
            if i < len(self.model.state_dict()) - 2:
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
                    losses = edge_ce_dice_loss(output, labels, self.ign_lab, self.ohem)
                scaler.scale(losses).backward()
                scaler.step(self.optimizer)
                scaler.update()

            output = np.argmax(output.cpu().data.numpy(), axis=1)
            labels = labels.cpu().data.numpy()
            # 对于需要忽略的标签位置直接置为忽略标签，避免影响准确率
            if self.ign_lab is not None:
                output[labels == self.ign_lab] = self.ign_lab
            acc = f1_score(output.reshape(-1, 1),
                           labels.reshape(-1, 1),
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
                labels = mask_bin
                # 输入模型获得预测分割图像
                output = nn.Softmax(dim=1)(self.model(img1))
                output = np.argmax(output.cpu().data.numpy(), axis=1)
                labels = labels.data.numpy()
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
        # 如果分类头是随机参数首先冻结模型参数对分类头进行预热
        if self.pop_head:
            self.freeze_model()
        # TODO 使用预训练模型时加入微调，先固定除分类头外的所有参数，然后先对分类头的参数进行预热，然后在第五轮之后再对所有模型参数进行解冻参与训练
        for epoch in range(1, self.epochs):
            train_loss, train_acc = self.train_fn(scaler)
            valid_acc, valid_f1 = self.eval_fn(visualize)
            RSPipeline.print_log(
                    '|EPOCH {} / {}| TRAIN_LOSS {}| TRAIN_F1_SCORE {}| VALID_ACC {}|  '
                    'VALID_F1_SCORE {}|'.format(
                            epoch, self.epochs, train_loss.avg, train_acc.avg, valid_acc.avg,
                            valid_f1.avg))
            scheduler.step()
            if valid_f1.avg > best_acc and epoch > 3:
                best_acc = valid_f1.avg
                RSPipeline.print_log(f"在当前训练轮数中找到最佳模型，训练轮数为{epoch + 1}")
                torch.save(self.model.state_dict(), self.model_save_path)
                # # 对模型进行解冻
                # if
                # self.unfreeze_model()
                # torch2onnx(self.model, self.onnx_model_name)
                RSPipeline.print_log("保存模型中")
                self.update_model_score("f1-score", best_acc)

        RSPipeline.print_log("训练过程结束")
        gc.collect()

    def update_model_score(self, score_name, score):
        """
        输出模型参数为yaml文件

        :return:
        """
        with open(self.model_save_path.replace("pth", "yaml"), "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        data[score_name] = score
        with open(self.model_save_path.replace("pth", "yaml"), "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    def output_model_info(self):
        """
        输出模型参数为yaml文件

        :return:
        """
        to_yaml = {"image_size":        [self.image_size, self.image_size],
                   "ignore background": self.ignore_background,
                   "index to label":    self.ind2label,
                   "device":            self.device,
                   "epochs":            self.epochs,
                   "model_name":        self.model_name,
                   "in_channels":        self.in_channels,
                   "save_path":         self.model_save_path,
                   "num_classes":       self.num_classes,
                   "batch_size":        self.batch_size,
                   "OHEM loss":         self.ohem,
                   "fp16training":      self.fp16
                   }
        if os.path.exists(self.model_save_path.replace("pth", "yaml")):
            os.remove(self.model_save_path.replace("pth", "yaml"))
        with open(self.model_save_path.replace("pth", "yaml"), "w", encoding="utf-8") as f:
            yaml.dump(to_yaml, f, default_flow_style=False)

    @staticmethod
    def get_pytorch_model(model_name, num_classes, in_channels,
                          pretrained_path, image_size, device):
        if "efficientnet" in model_name:
            model = get_semantic_segment_model(num_classes=num_classes,
                                               in_channels=in_channels,
                                               model_name=model_name,
                                               device="cpu",
                                               pretrained_path=pretrained_path,
                                               pop_head=False)
        else:
            model = SwinTransformerSys(img_size=image_size, num_classes=num_classes)
            model = nn.DataParallel(model)
            if pretrained_path is not None and os.path.exists(pretrained_path):
                try:
                    ckpt = torch.load(pretrained_path, map_location=torch.device(device))
                    model.load_state_dict(
                            ckpt,
                            strict=False
                    )
                    print('历史最佳模型已读取!')
                except:
                    print("参数发生变化，历史模型无效！")
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def load_model(model_info_path, model_type, device="cuda", half=True):
        """
        根据提供的yaml文件路径，解析模型参数，并导入模型。

        :param model_type:
        :param device:
        :param model_info_path:
        :return:
        """
        with open(model_info_path, 'r', encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        RSPipeline.print_log("根据yaml文件导入模型中")
        num_classes = data['num_classes']
        in_channels = data['in_channels']
        pretrained_path = data['save_path']
        model_name = data['model_name']
        image_size = data['image_size'][0]
        onnx_path = pretrained_path.replace("pth", "onnx")
        if model_type == "pytorch":
            model = RSPipeline.get_pytorch_model(model_name, num_classes, in_channels,
                                                 pretrained_path, image_size,
                                                 device)
            model = model.half() if half else model
        else:
            raise "model type not support, only support onnx and pytorch!"
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

        :param args:
        :param image_path:
        :param num_classes:
        :param shp_path:
        :param ind2num:
        :return:
        """
        RSPipeline.print_log("开始更新矢量标签数据为像素标签数据")
        # 获取所有填充后的标签的保存路径
        save_path = path.join("/".join(args.label.split('/')[:-1]), image_path.split('/')[-1])
        print(save_path)
        shp_file = read_shp(shp_path)
        columns = shp_file.columns
        print(columns)
        # shp_file = shp_file[shp_file[columns[0]].notnull()]
        # shp_file = shp_file[shp_file[columns[0]].notna()]
        # shp_file.iloc[:, 0] = shp_file.iloc[:, 0].apply(lambda x: ind2num[str(x)[:2]])
        # shp_file.to_file(shp_path)
        shp2tif(shp_path=shp_path,
                refer_tif_path=image_path,
                target_tif_path=save_path,
                attribute_field=columns[0],
                nodata_value=0)
        RSPipeline.print_log("更新结果完毕，数据已保存")

    @staticmethod
    def check_path(path):
        return path.replace("\\", "/")
