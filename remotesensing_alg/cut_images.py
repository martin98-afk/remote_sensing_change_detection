# encoding:utf-8
from glob import glob

import numpy as np
from PIL import Image
from osgeo import gdal

from utils.gdal_utils import write_img
from utils.polygon_utils import read_shp, shp2tif

dict = {"耕地": [1], "建设用地": [3, 4, 5, 8], "林地园地草地": [2]}
class_id_list = np.arange(9)


def extract_target(class_id, name, res,
                   shp_path="real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_major_class.shp"):
    """
    根据类别id提取对应三调中标记的地区并存储为栅格图像。

    :param res:
    :param class_id: 类别标签
    :param name: 类别名称
    :return:
    """
    # 根据保存的一级类筛选对应的地貌类型转换为掩膜
    shp_file_ori = read_shp(shp_path)
    shp_file = shp_file_ori[shp_file_ori.iloc[:, 0] == class_id[0]]
    if len(class_id) > 1:
        for i in class_id[1:]:
            shp_file = shp_file.append(shp_file_ori[shp_file_ori.iloc[:, 0] == i])
    shp_file.to_file(shp_path.replace("major_calss.shp",f"{name}.shp"))
    refer_image_path = glob(f"real_data/processed_data/2021_1_*_res_{res}.tif")
    save_path = [path.replace("processed_data", "trad_alg")
                     .replace(".tif", f"_{name}_mask.tif") for path in refer_image_path]
    # 将耕地矢量图斑文件转为栅格数据
    for image_path, save in zip(refer_image_path, save_path):
        shp2tif(shp_path=f"real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_{name}.shp",
                refer_tif_path=image_path,
                target_tif_path=save,
                attribute_field="DLBM",
                nodata_value=0)

    # 使用栅格数据作为mask从2020年和2021年的图像之中提取对应图斑部分
    refer_image_path = glob(f"real_data/processed_data/202*_1_*_res_{res}.tif")
    refer_image_path = [path.replace('\\', '/') for path in refer_image_path]
    mask_path = [path.replace("processed_data", "trad_alg")
                     .replace(".tif", f"_{name}_mask.tif")
                     .replace("2020", "2021") for path in refer_image_path]
    save_path = [path.replace("processed_data", "trad_alg")
                     .replace(".tif", f"_{name}.tif") for path in refer_image_path]
    # 语义分割结果读取，用来对耕地图斑中的异常区域进行剔除
    ss_result_path = [path.replace("real_data", "output")
                          .replace("processed_data", "semantic_result/tif")
                          .replace(".tif", f"_semantic_result.tif") for path in refer_image_path]
    for path1, path2, path3, path4 in zip(refer_image_path, mask_path, save_path, ss_result_path):
        image = gdal.Open(path1)  # 待提取图片
        mask = Image.open(path2)  # 三调地貌区域蒙版
        mask = np.array(mask)

        # 添加第四个维度用来区分背景和前景，目前只在qgis中有用，在算法中并没有发挥到判别前景还有背景的作用。
        ones = 255 * np.ones_like(mask)
        ones[mask == 0] = 0

        # 对蒙版进行repeat从而使得 在判断条件以后可以直接对图像区域进行筛选。
        mask = np.repeat(mask[..., np.newaxis], 3, 2)
        # gdal读取图片后 channel在第一位
        image_array = np.transpose(image.ReadAsArray(), (1, 2, 0))
        image_array[mask == 0] = 0  # 直接将不需要部分取0
        # TODO 使用语义分割结果对栅格图像进行异常值剔除
        if "2021" in path1:
            ss_result = Image.open(path4)  # 根据语义分割对三调提取后的区域进行异常区域剔除
            ss_result = np.array(ss_result)
            ss_result[mask == 0] = 0
            for id in class_id_list:
                if id not in dict[name]:
                    image_array[ss_result == id] = 0
                    ones[ss_result == id] = 0
        # 添加第四个维度用来区分前景和背景值
        image_array = np.concatenate([image_array, ones.reshape((ones.shape[0], ones.shape[1], 1))],
                                     axis=-1)
        write_img(path3, image.GetProjection(),
                  image.GetGeoTransform(),
                  np.transpose(image_array, (2, 0, 1)))
