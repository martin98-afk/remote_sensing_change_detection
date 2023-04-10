# encoding:utf-8
import os
from numpy import random
import torch
import numpy as np
# import cupy as cp
import warnings
import matplotlib.pyplot as plt


# 忽略警告信息
from utils.minio_store import MinioStore

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

# 数据库连接信息
# 连接minio文件数据库
minio_store = MinioStore(host="192.168.9.153:9000",
                         access_key="xw-admin",
                         secret_key="xw-admin",
                         bucket="ai-platform",
                         save_dirs="")

# mysql连接信息
MYSQL_CONFIG = {
    'host':    '192.168.9.153',
    'user':    'urbanlab_admin',
    'passwd':  'urbanlab_123',
    'db':      'db_landform_dev',
    'port':    3306,
    'charset': 'utf8'
}