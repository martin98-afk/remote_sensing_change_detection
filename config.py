# encoding:utf-8
import os
from numpy import random
import torch
import numpy as np
# import cupy as cp
import warnings
import matplotlib.pyplot as plt


# 忽略警告信息
warnings.filterwarnings('ignore')


# 设置随机种子
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


seed_it(1)

# 控制图像字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

font1 = {'family': 'SimHei',
         'weight': 'normal',
         'size': 20,
         }

# ind2label = {
#     "01": "耕地",
#     "02": "种植园地",
#     "03": "林地",
#     "04": "草地",
#     "05": "商业服务业用地",
#     "06": "工矿用地",
#     "07": "住宅用地",
#     "08": "公共管理与公共服务用地",
#     "09": "特殊用地",
#     "10": "交通运输用地",
#     "11": "水域及水利设施用地",
#     "12": "其他土地",
# }
# ind2num = {
#     "01": 1,
#     "02": 2,
#     "03": 3,
#     "04": 4,
#     "05": 5,
#     "06": 5,
#     "07": 5,
#     "08": 5,
#     "09": 8,
#     "10": 6,
#     "11": 7,
#     "12": 8,
# }