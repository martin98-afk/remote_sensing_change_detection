import os
from glob import glob

from osgeo import gdal


"""
Python+GDAL对两张遥感影像实现共同区域裁剪
"""

os.makedirs("../real_data/processed_data", exist_ok=True)


def change_resolution(image, resolution):
    """
    修改遥感影像分辨率, 对齐相同地区的遥感影像

    :param source_image_path:
    :param resolution:
    :return:
    """
    ds = gdal.Warp("",
                   image,
                   format="MEM",
                   xRes=resolution,
                   yRes=resolution)
    return ds


def compute_bounds(geo_transform, image_shape):
    """
    根据tif的地理位置信息算4个左上和右下的坐标

    :param geo_transform:
    :param image_shape:
    :return:
    """
    Xgeo = geo_transform[0]
    Ygeo = geo_transform[3]

    Xgeo2 = geo_transform[0] + image_shape[2] * geo_transform[1] + \
            image_shape[1] * geo_transform[2]
    Ygeo2 = geo_transform[3] + image_shape[2] * geo_transform[4] + \
            image_shape[1] * geo_transform[5]
    return Xgeo, Ygeo2, Xgeo2, Ygeo


def write_img(image_path, im_proj, im_geotrans, im_data):
    """
    根据指定图像文件路径，图像映射，图像地理位置，图像数据，对图像进行保存

    :param image_path:
    :param im_proj:
    :param im_geotrans:
    :param im_data:
    :return:
    """
    im_bands, im_height, im_width = im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(image_path, im_width, im_height, im_bands, gdal.GDT_Byte)
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def transform_geoinfo(source_image, target_image):
    """
    统一投影系统。

    :param geo_transform:
    :return:
    """
    new_image = gdal.Warp("", source_image, format="MEM", dstSRS=target_image.GetProjection())
    return new_image


def align_images(image1, image2):
    """
    对齐两个遥感图像

    :param image1:
    :param image2:
    :return:
    """
    image1_shape = image1.ReadAsArray().shape
    image2_shape = image2.ReadAsArray().shape

    x11, y11, x12, y12 = compute_bounds(image1.GetGeoTransform(), image1_shape)
    x21, y21, x22, y22 = compute_bounds(image2.GetGeoTransform(), image2_shape)
    x1, y1, x2, y2 = max(x11, x21), max(y11, y21), min(x12, x22), min(y12, y22)

    assert x1 < x2 and y1 < y2, print("两遥感图像不相交")
    image1 = gdal.Warp(
        "",
        image1,
        format="MEM",
        outputBounds=[x1, y1, x2, y2],
    )
    image2 = gdal.Warp(
        "",
        image2,
        format="MEM",
        outputBounds=[x1, y1, x2, y2],
    )
    return image1, image2


def preprocess_rs_image(image1_path, image2_path, resolution,
                        save_root="../real_data/processed_data/"):
    """
    对两个需要进行变化区域分析的遥感图像进行预处理，先统一分辨率，然后进行图像对齐,
    最后保存为带有地理信息的tif文件到 real_data/processed_data 文件夹下。
    同时会判断两个图像是否相交，如果不相交会提示报错。结果保存名为原名加 _res_ + 分辨率大小。

    :param image1_path: 图像1的路径
    :param image2_path: 图像2的路径
    :param resolution: 将2个图像预处理后的分辨率大小
    :param save_root: 保存路径根目录，default = ../real_data/processed_data/, 如果输入None则直接返回两张预处理好的图像
    :return:
    """
    image1 = gdal.Open(image1_path)
    image2 = gdal.Open(image2_path)
    image1 = transform_geoinfo(image1, image2)
    rs_name1 = image1_path.split('/')[-1].split('.')
    rs_name2 = image2_path.split('/')[-1].split('.')
    rs_name1[0] = rs_name1[0] + '_res_' + str(resolution)
    rs_name2[0] = rs_name2[0] + '_res_' + str(resolution)

    image1 = change_resolution(image1, resolution)
    image2 = change_resolution(image2, resolution)

    image1, image2 = align_images(image1, image2)

    if save_root is not None:
        save_name1 = save_root + '.'.join(rs_name1)
        save_name2 = save_root + '.'.join(rs_name2)
        if os.path.exists(save_name1):
            os.remove(save_name1)
        if os.path.exists(save_name2):
            os.remove(save_name2)
        write_img(save_name1, image1.GetProjection(), image1.GetGeoTransform(),
                  image1.ReadAsArray())
        write_img(save_name2, image2.GetProjection(), image2.GetGeoTransform(),
                  image2.ReadAsArray())
    else:
        return image1, image2


if __name__ == "__main__":
    res_list = [0.3, 0.5, 0.8]
    root_path = "../real_data/移交数据和文档/苏南/0.2米航片/"
    image_2020_files = glob(os.path.join(root_path, "2020*.tif"))
    image_2021_files = [path.replace("2020", "2021") for path in image_2020_files]
    for i, res in enumerate(res_list):
        for j, (path1, path2) in enumerate(zip(image_2020_files, image_2021_files)):
            preprocess_rs_image(path1, path2, res)
