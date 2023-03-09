# encoding:utf-8
import gc
import json
import os
import zipfile
from glob import glob

import requests
from flask import Flask, request
from osgeo import gdal

from detect_change import get_parser
from utils.conn_mysql import *
from utils.detect_change_server_func import DetectChangeServer as DCS
from utils.gdal_utils import preprocess_rs_image, transform_geoinfo_with_index
from utils.minio_store import MinioStore
from utils.pipeline import RSPipeline
from utils.polygon_utils import read_shp, json2shp

app = Flask(__name__)
# 导入detect change中的超参数，同时导入模型
args = get_parser()
args = vars(args)
data, model = RSPipeline.load_model('output/ss_eff_b0_new.yaml',
                                    model_type=args["model_type"],
                                    device=args["device"],
                                    half=args["half"])
IMAGE_SIZE = data['image_size'][0]  # 划分图像块大小
num_classes = data['num_classes']  # 地貌分类数量

os.makedirs("real_data/cache", exist_ok=True)
# 导入模型参数
minio_store = MinioStore(host="192.168.9.153:9000",
                         access_key="xw-admin",
                         secret_key="xw-admin",
                         bucket="ai-platform",
                         save_dirs="")


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


def file2zip(zip_file_name, file_names):
    """
    将多个文件压缩存储为zip

    :param zip_file_name:
    :param file_names:
    :return:
    """
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in file_names:
            parent_path, name = os.path.split(fn)
            zf.write(fn, arcname=name)


@app.route("/detect_change", methods=['POST'])
def detect_change():
    """
    变化识别api接口程序，支持网格化变化识别、图斑变华识别，针对非农化土地变化识别。

    :return:
    """
    info_dict = {
        "id":                 request.json['id'],  # 保留字段，数据存库使用
        "modelType":          request.json['modelType'],  # 模型类型:0 非农化 1 非粮化
        "topicType":          request.json['topicType'],  # 参考影像地址
        "referImageUrl":      request.json['referImageUrl'],  # 参考影像地址
        "comparisonImageUrl": request.json['comparisonImageUrl'],  # 比对影像地址
        "resolution":         request.json['resolution'],  # 分辨率
        "occupyArea":         request.json['occupyArea'],  # 侵占面积
        "algorithmType":      request.json['algorithmType']  # 算法类型 0 卷积神经算法  1 融合算法  2灰度神经算法
    }
    args["id"] = info_dict["id"]
    args["block_size"] = int(float(info_dict["occupyArea"]) / float(info_dict["resolution"]))

    if info_dict["topicType"] == 1:
        # 输入两张tif，分析两张遥感图像的网格变化区域
        tif1_path = "real_data/cache/src_image.tif"
        tif2_path = "real_data/cache/target_image.tif"
        save_file_from_url(tif1_path, info_dict["referImageUrl"])
        save_file_from_url(tif2_path, info_dict["comparisonImageUrl"])
        # TODO 图像预处理,分辨率统一为提供的分辨率
        for tif in [tif1_path, tif2_path]:
            transform_geoinfo_with_index(tif, 3857)
            gdal.Warp(tif, tif, format="GTiff",
                      xRes=float(info_dict["resolution"]),
                      yRes=float(info_dict["resolution"]))
        # 对影像进行预处理
        preprocess_rs_image("real_data/cache/src_image.tif",
                            "real_data/cache/target_image.tif",
                            info_dict["resolution"],
                            save_root="same")
        save_path, tif_path, shp_path = DCS.change_block_detect(model,
                                                                src_image=tif1_path,
                                                                target_image=tif2_path,
                                                                IMAGE_SIZE=IMAGE_SIZE, args=args)
    elif info_dict["topicType"] == 0:
        # 输入一张tif一张geojson，分析遥感图像上的变化图斑
        json_path = "real_data/cache/src_image.json"
        tif_path = "real_data/cache/target_image.tif"
        shp_path = "real_data/cache/src_image.shp"
        inp_shp_path = shp_path
        save_file_from_url(json_path, info_dict["referImageUrl"])
        save_file_from_url(tif_path, info_dict["comparisonImageUrl"])
        # TODO 图像预处理,分辨率统一为0.5米
        transform_geoinfo_with_index(tif_path, 3857)
        gdal.Warp(tif_path, tif_path, format="GTiff",
                  xRes=float(info_dict["resolution"]),
                  yRes=float(info_dict["resolution"]))

        # TODO 将geojson转换为shp文件
        json2shp(json_path, shp_path)
        save_path, mask_path, tif_path, shp_path = DCS.change_polygon_detect(model,
                                                                             src_shp=shp_path,
                                                                             target_image=tif_path,
                                                                             IMAGE_SIZE=IMAGE_SIZE,
                                                                             args=args)
        file_name = os.path.basename(mask_path)
        minio_store.fput_object(f"change_result/{file_name}", mask_path)

    # 将识别结果上传minio服务器
    zip_path = shp_path.replace("_spot.shp", "zip")
    shp_path = shp_path.replace("shp", "*")
    shp_paths = glob(shp_path)
    file_name = os.path.basename(save_path)
    minio_store.fput_object(f"change_result/{file_name}", save_path)
    save_url = f"http://221.226.175.85:9000/ai-platform/change_result/{file_name}"
    file_name = os.path.basename(tif_path)
    minio_store.fput_object(f"change_result/{file_name}", tif_path)
    file2zip(zip_path, shp_paths)
    file_name = os.path.basename(zip_path)
    minio_store.fput_object(f"change_result/{file_name}", zip_path)
    # 获取当前变化识别出的变化图斑数量
    file = read_shp(shp_path.replace("*", "shp"))
    change_num = len(file)
    # 连接mysql数据库，更新数据处理进度
    # TODO 将存储的文件url以及任务id存储到数据库中。
    mysql_conn = MysqlConnectionTools(**MYSQL_CONFIG)
    mysql_conn.write_to_mysql_relation(info_dict["id"], save_url, change_num - 1)
    mysql_conn.write_to_mysql_progress(args["id"], "100%")
    # 回收内存
    gc.collect()
    return json.loads(json.dumps({}))


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=8002)
    app.run(host='192.168.9.161', port=8002)
