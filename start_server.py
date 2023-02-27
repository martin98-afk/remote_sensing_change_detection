# encoding:utf-8
import gc
import os
from glob import glob

import requests
from flask import Flask, request

from detect_change import get_parser, change_block_detect, change_polygon_detect
from utils.conn_mysql import *
from utils.minio_store import MinioStore
from utils.pipeline import RSPipeline
from utils.polygon_utils import read_shp

app = Flask(__name__)
# 导入detect change中的超参数，同时导入模型
args = get_parser()
args = vars(args)
data, model = RSPipeline.load_model('output/ss_eff_b0.yaml',
                                    model_type=args["model_type"],
                                    device=args["device"],
                                    half=args["half"])
IMAGE_SIZE = data['image_size'][0]  # 划分图像块大小
num_classes = data['num_classes']  # 地貌分类数量


def save_file_from_url(save_path, url_path):
    """
    将url里面的文件保存到本地。

    :param save_path: 保存地址
    :param url_path: 接受到的url地址
    :return:
    """
    response = requests.get(url_path)

    # 保存图片到本地
    with open(save_path, 'wb') as f:  # 以二进制写入文件保存
        f.write(response.content)


@app.route("/detect_change", methods=['POST'])
def detect_change():
    """
    变化识别api接口程序，支持网格化变化识别、图斑变华识别，针对非农化土地变化识别。

    :return:
    """
    info_dict = {
        "id":                 request.json['id'],  # 保留字段，数据存库使用
        "modelType":          request.json['modelType'],  # 模型类型:0 非农化 1 非粮化
        "referImageUrl":      request.json['referImageUrl'],  # 参考影像地址
        "comparisonImageUrl": request.json['comparisonImageUrl'],  # 比对影像地址
        "resolution":         request.json['resolution'],  # 分辨率
        "occupyArea":         request.json['occupyArea'],  # 侵占面积
        "algorithmType":      request.json['algorithmType']  # 算法类型 0 卷积神经算法  1 融合算法  2灰度神经算法
    }
    args["id"] = info_dict["id"]
    os.makedirs("real_data/cache", exist_ok=True)
    # 导入模型参数
    minio_store = MinioStore(host="192.168.9.153:9000",
                             access_key="xw-admin",
                             secret_key="xw-admin",
                             bucket="ai-platform",
                             save_dirs="")
    save_file_from_url("real_data/cache/src_image.tif", info_dict["referImageUrl"])
    save_file_from_url("real_data/cache/target_image.tif", info_dict["comparisonImageUrl"])
    if info_dict["algorithmType"] == 0:
        save_path, tif_path, shp_path = change_block_detect(model,
                                                            src_image="real_data/cache/src_image.tif",
                                                            target_image="real_data/cache/target_image.tif",
                                                            IMAGE_SIZE=IMAGE_SIZE, args=args)
    elif info_dict["algorithmType"] == 1:
        save_path, tif_path, shp_path = change_polygon_detect(model,
                                                              target_image="real_data/cache/target_image.tif",
                                                              IMAGE_SIZE=IMAGE_SIZE, args=args)
    # 将识别结果上传minio服务器
    shp_path = shp_path.replace("shp", "*")
    shp_paths = glob(shp_path)
    file_name = save_path.split('/')[-1]
    minio_store.fput_object(f"change_result/{file_name}", save_path)
    save_url = f"http://221.226.175.85:9000/ai-platform/change_result/{file_name}"
    file_name = tif_path.split('/')[-1]
    minio_store.fput_object(f"change_result/{file_name}", tif_path)
    for path in shp_paths:
        file_name = path.split('/')[-1]
        minio_store.fput_object(f"change_result/{file_name}", path)
    # 获取当前变化识别出的变化图斑数量
    file = read_shp(shp_path.replace("*", "shp"))
    change_num = len(file)
    # 连接mysql数据库，更新数据处理进度
    # TODO 将存储的文件url以及任务id存储到数据库中。
    mysql_conn = MysqlConnectionTools(**MYSQL_CONFIG)
    mysql_conn.write_to_mysql_relation(info_dict["id"], save_url, change_num)
    mysql_conn.write_to_mysql_progress(args["id"], "100%")
    # 回收内存
    gc.collect()
    pass


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=8002)
    app.run(host='192.168.9.161', port=8002)
