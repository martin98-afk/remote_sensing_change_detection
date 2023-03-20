import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import CrossEntropyLoss
from torchinfo import summary


def get_semantic_segment_model(num_classes,
                               in_channels=3,
                               model_name='efficientnet-b0',
                               device='cuda',
                               pretrained_path='./output/label_2_best_perf.pth',
                               pop_head=False
                               ):
    """
    获取语义分割模型

    :param pop_head: 删除模型的分类头
    :param num_classes: 输出结果类别数
    :param model_name: 骨干网络类型
    :param device: 训练机器类型
    :param pretrained_path: 导入的预训练模型地址
    :return:
    """
    try:
        model = smp.UnetPlusPlus(
                encoder_name=model_name,
                encoder_weights=None,
                in_channels=in_channels,
                decoder_attention_type="scse",
                classes=num_classes,
                activation=None,
                aux_params=None
        )
    except:
        model = smp.Unet(
                encoder_name=model_name,
                encoder_weights=None,
                in_channels=in_channels,
                decoder_attention_type="scse",
                classes=num_classes,
                activation=None,
                aux_params=None
        )
    model.to(device)
    model = torch.nn.DataParallel(model)
    if pretrained_path is not None and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location=torch.device(device))
        try:
            if pop_head:
                ckpt.pop("module.segmentation_head.0.bias")
                ckpt.pop("module.segmentation_head.0.weight")
                print("已将模型头部去掉!")
            model.load_state_dict(
                    ckpt,
                    strict=False
            )
            print('历史最佳模型已读取!')
        except:
            print("参数发生变化，历史模型无效！")
            ckpt.pop("module.encoder._conv_stem.weight")
            model.load_state_dict(
                    ckpt,
                    strict=False
            )
    if device == "cpu":
        model = model.module.to(torch.device(device))
    return model


def edge_ce_dice_loss(pred, target, ignore_index, ohem, alpha=0.3):
    # diceloss在一定程度上可以缓解类别不平衡，但是训练容易
    dice_loss = DiceLoss(mode='multiclass', ignore_index=ignore_index)
    # 交叉熵
    ce_loss = CrossEntropyLoss(reduction="none")
    loss_ce = ce_loss(pred, target)
    if ignore_index is not None:
        a, b, c = np.where(target.cpu().data.numpy() == ignore_index)
        loss_ce[a, b, c] = 0
    loss_dice = dice_loss(pred, target)
    # OHEM
    if ohem:
        loss_ce_, ind = loss_ce.contiguous().view(-1).sort()
        min_value = loss_ce_[int(alpha * loss_ce.numel())]
        loss_ce = loss_ce[loss_ce > min_value]
    loss_ce = loss_ce.mean()
    loss = loss_dice + loss_ce
    return loss


if __name__ == "__main__":
    model = get_semantic_segment_model(12, model_name="efficientnet-b0",
                                       pretrained_path="../output/ss_eff_b0.pth",
                                       pop_head=True)
    print(summary(model, input_size=(1, 3, 512, 512)))
