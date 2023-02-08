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
