import os
from glob import glob

from osgeo import gdal, osr

from utils.pipeline import RSPipeline

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
    RSPipeline.check_file(image_path)
    im_bands, im_height, im_width = im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(image_path, im_width, im_height, im_bands, gdal.GDT_Byte)
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def transform_geoinfo(source_image, spatial_reference):
    """
    使用给定坐标系进行投影系统转换。

    :param geo_transform:
    :return:
    """
    new_image = gdal.Warp("", source_image, format="MEM", dstSRS=spatial_reference)
    return new_image


def transform_geoinfo_with_index(source_image_path, index):
    """
    使用指定标签进行投影系统转换

    :param source_image_path:
    :param index:
    :return:
    """
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(index)
    gdal.Warp(source_image_path, source_image_path, format="GTiff",
              dstSRS=spatial_reference)


def align_images(image1, image2):
    """
    对齐两个遥感图像

    :param image1:
    :param image2:
    :return:
    """
    image1_shape = image1.ReadAsArray().shape
    image2_shape = image2.ReadAsArray().shape
    # 计算两个遥感图像相交区域
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
                        save_root="same"):
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
    # # 统一映射系统
    transform_geoinfo_with_index(image1_path, 3857)
    transform_geoinfo_with_index(image2_path, 3857)

    image1 = gdal.Open(image1_path)
    image2 = gdal.Open(image2_path)
    # if image1.GetProjection() != image2.GetProjection():
    #     spatial_reference = osr.SpatialReference()
    #     spatial_reference.ImportFromWkt(image2.GetProjection())
    #     image1 = transform_geoinfo(image1, spatial_reference)
    # # get the WGS84 spatial reference
    # spatial_reference = osr.SpatialReference()
    # spatial_reference.ImportFromEPSG(4326)
    # image1 = transform_geoinfo(image1, spatial_reference)
    # image2 = transform_geoinfo(image2, spatial_reference)
    # 统一图像分辨率
    image1 = change_resolution(image1, resolution)
    image2 = change_resolution(image2, resolution)
    # 遥感图像对齐
    image1, image2 = align_images(image1, image2)

    # 设定存储名称
    if save_root == "same":
        save_name1 = image1_path
        save_name2 = image2_path
    elif save_root is not None:
        rs_name1 = image1_path.split('/')[-1].split('.')
        rs_name2 = image2_path.split('/')[-1].split('.')
        rs_name1[0] = rs_name1[0] + '_res_' + str(resolution)
        rs_name2[0] = rs_name2[0] + '_res_' + str(resolution)
        save_name1 = save_root + '.'.join(rs_name1)
        save_name2 = save_root + '.'.join(rs_name2)
        if os.path.exists(save_name1):
            os.remove(save_name1)
        if os.path.exists(save_name2):
            os.remove(save_name2)
    else:
        return image1, image2
    # 将预处理好的图像写到内存之中，如果save_root为None则直接返回预处理好的图像
    write_img(save_name1, image1.GetProjection(), image1.GetGeoTransform(),
              image1.ReadAsArray())
    write_img(save_name2, image2.GetProjection(), image2.GetGeoTransform(),
              image2.ReadAsArray())


def cut_image(src_image_path, year, save_dir="real_data/sample_data"):
    """
    将大遥感图像切割成小的512*512大小的样本图像，方便在网页上跨苏展示。

    :param src_image_path:
    :param year:
    :param save_dir:
    :return:
    """
    # 建立存储文档地址
    os.makedirs(save_dir, exist_ok=True)
    # 读取要切的原图
    in_ds = gdal.Open(src_image_path)
    print("open tif file succeed")
    width = in_ds.RasterXSize  # 获取数据宽度
    height = in_ds.RasterYSize  # 获取数据高度
    outbandsize = in_ds.RasterCount  # 获取数据波段数
    im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = in_ds.GetProjection()  # 获取投影信息
    datatype = in_ds.GetRasterBand(1).DataType
    im_data = in_ds.ReadAsArray()  # 获取数据

    # 读取原图中的每个波段
    in_band1 = in_ds.GetRasterBand(1)
    in_band2 = in_ds.GetRasterBand(2)
    in_band3 = in_ds.GetRasterBand(3)

    # 定义切图的起始点坐标
    offset_x = 0
    offset_y = 0

    # offset_x = width/2  # 这里是随便选取的，可根据自己的实际需要设置
    # offset_y = height/2

    # 定义切图的大小（矩形框）
    block_xsize = 1024  # 行
    block_ysize = 1024  # 列

    k = 0
    for i in range(width // block_xsize):
        for j in range(height // block_xsize):
            out_band1 = in_band1.ReadAsArray(i * block_xsize, j * block_xsize, block_xsize,
                                             block_ysize)
            out_band2 = in_band2.ReadAsArray(i * block_xsize, j * block_xsize, block_xsize,
                                             block_ysize)
            out_band3 = in_band3.ReadAsArray(i * block_xsize, j * block_xsize, block_xsize,
                                             block_ysize)
            print(out_band3)
            k += 1

            ## 从每个波段中切需要的矩形框内的数据(注意读取的矩形框不能超过原图大小)
            # out_band1 = in_band1.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
            # out_band2 = in_band2.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
            # out_band3 = in_band3.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)

            # 获取Tif的驱动，为创建切出来的图文件做准备
            gtif_driver = gdal.GetDriverByName("GTiff")

            # 创建切出来的要存的文件（3代表3个不都按，最后一个参数为数据类型，跟原文件一致）
            out_ds = gtif_driver.Create(f"{save_dir}/{str(k)}_{year}.tif",
                                        block_xsize, block_ysize, outbandsize, datatype)
            # print("create new tif file succeed")

            # 获取原图的原点坐标信息，# 获取仿射矩阵信息
            ori_transform = in_ds.GetGeoTransform()
            if ori_transform:
                print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
                print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))

            # 读取原图仿射变换参数值
            top_left_x = ori_transform[0]  # 左上角x坐标
            w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
            top_left_y = ori_transform[3]  # 左上角y坐标
            n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率

            # 根据反射变换参数计算新图的原点坐标
            top_left_x = top_left_x + i * block_xsize * w_e_pixel_resolution
            top_left_y = top_left_y + j * block_xsize * n_s_pixel_resolution

            # 将计算后的值组装为一个元组，以方便设置
            dst_transform = (
                top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4],
                ori_transform[5])

            # 设置裁剪出来图的原点坐标
            out_ds.SetGeoTransform(dst_transform)

            # 设置SRS属性（投影信息）
            out_ds.SetProjection(in_ds.GetProjection())

            # 写入目标文件
            out_ds.GetRasterBand(1).WriteArray(out_band1)
            out_ds.GetRasterBand(2).WriteArray(out_band2)
            out_ds.GetRasterBand(3).WriteArray(out_band3)

            # 将缓存写入磁盘
            out_ds.FlushCache()
            print("FlushCache succeed")

            # 计算统计值
            # for i in range(1, 3):
            #     out_ds.GetRasterBand(i).ComputeStatistics(False)
            # print("ComputeStatistics succeed")

            del out_ds

            print("End!")


if __name__ == "__main__":
    os.makedirs("../real_data/sample_data3", exist_ok=True)
    cut_image("../real_data/processed_data/2021_1_3_res_0.5.tif", "2021",
              save_dir="../real_data/sample_data3")
    # path = "../real_data/test_bing.tif"
    # transform_geoinfo_with_index(path, 3857)
    # ds = gdal.Warp(path.replace(".tif", "_res_0.5.tif"),
    #                path, format="GTiff", xRes=0.5, yRes=0.5)
    # res_list = [0.3, 0.5, 0.8]
    # root_path = "../real_data/移交数据和文档/苏南/0.2米航片/"
    # image_2020_files = glob(os.path.join(root_path, "2020*.tif"))
    # image_2021_files = [path.replace("2020", "2021") for path in image_2020_files]
    # for i, res in enumerate(res_list):
    #     for j, (path1, path2) in enumerate(zip(image_2020_files, image_2021_files)):
    #         preprocess_rs_image(path1, path2, res, save_root="../real_data/processed_data/")
