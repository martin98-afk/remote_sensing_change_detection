import os

import cv2
import geopandas
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import osr, gdal, ogr

from config import *


def raster2vector(raster_path, vector_path, field_name='class', ignore_vales=None):
    """
    该代码文件包括所有任务检测结果的保存，主要为遥感图像分类结果栅格数据转换为矢量数据，并保存结果
    遥感图像像素级别分类（语义分割）结果是栅格图像，转成矢量shp更方便在arcgis中自定义展示，
    （比如只显示目标边框）以及进一步分析（比如缓冲区分析）。

    :param raster_path:
    :param vector_path:
    :param field_name:
    :param ignore_vales:
    :return:
    """
    raster = gdal.Open(raster_path)
    band = raster.GetRasterBand(1)
    # 读取栅格的投影信息， 为后面生成的矢量赋予相同的投影信息
    prj = osr.SpatialReference()
    prj.ImportFromWkt(raster.GetProjection())

    drv = ogr.GetDriverByName("ESRI Shapefile")
    # 若文件已经存在，删除
    if os.path.exists(vector_path):
        drv.DeleteDataSource(vector_path)

    # 创建目标文件
    polygon = drv.CreateDataSource(vector_path)
    # 创建面图层
    poly_layer = polygon.CreateLayer(vector_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)

    # 添加浮点型字段，用来存储栅格像素值
    field = ogr.FieldDefn(field_name, ogr.OFTReal)
    poly_layer.CreateField(field)

    # FPolygonize将每个襄垣转成一个矩形，然后将相似的像元进行合并
    gdal.FPolygonize(band, None, poly_layer, 0)

    if ignore_vales is not None:
        for feature in poly_layer:
            class_value = feature.GetField('class')
            for ignore_value in ignore_vales:
                if class_value == ignore_value:
                    poly_layer.DeleteFeature(feature.GetFID())
                    break
    polygon.SyncToDisk()
    polygon = None


def get_tif_meta(tif_path):
    """
    获得tif地址所指文件的信息，包括：长宽、投影信息

    :param tif_path:
    :return:
    """
    dataset = gdal.Open(tif_path)
    # 栅格矩阵的列数
    width = dataset.RasterXSize
    # 栅格矩阵的行数
    height = dataset.RasterYSize
    # 获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    # 获取投影信息
    proj = dataset.GetProjection()
    return width, height, geotrans, proj


def shp2tif(shp_path, refer_tif_path, target_tif_path, attribute_field="class", nodata_value=0):
    """
    使用gdal将矢量文件转换为栅格文件。

    :param shp_path:
    :param refer_tif_path:
    :param target_tif_path:
    :param attribute_field:
    :param nodata_value:
    :return:
    """
    width, height, geotrans, proj = get_tif_meta(refer_tif_path)
    # 读取shp文件
    shp_file = ogr.Open(shp_path)
    # 获取图层文件对象
    shp_layer = shp_file.GetLayer()
    # 创建栅格
    target_ds = gdal.GetDriverByName('GTiff').Create(
        utf8_path=target_tif_path,  # 栅格地址
        xsize=width,  # 栅格宽
        ysize=height,  # 栅格高
        bands=1,  # 栅格波段数
        eType=gdal.GDT_Byte  # 栅格数据类型
    )
    # 将参考栅格的仿射变换信息设置为结果栅格仿射变换信息
    target_ds.SetGeoTransform(geotrans)
    # 设置投影坐标信息
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    # 设置背景nodata数值
    band.SetNoDataValue(nodata_value)
    band.FlushCache()

    # 栅格化函数
    gdal.RasterizeLayer(
        dataset=target_ds,  # 输出的栅格数据集
        bands=[1],  # 输出波段
        layer=shp_layer,  # 输入待转换的矢量图层
        options=[f"ATTRIBUTE={attribute_field}"]  # 指定字段值为栅格值
    )

    del target_ds


def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系。

    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def lonlat2geo(prosrs, geosrs, lonlat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）。

    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''

    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = []
    for i in range(lonlat.shape[0]):
        coords.append(ct.TransformPoint(lonlat[i, 0], lonlat[i, 1])[:2])
    return np.array(coords)


def coord2pixel(coords, geotrans):
    """
    投影坐标转换为像素坐标。

    :param coords:
    :param geotrans:
    :return:
    """
    coords[:, 1] = (
            (coords[:, 1] - geotrans[3] - geotrans[4] / geotrans[1] * coords[:, 0] + geotrans[
                4] / geotrans[1] * geotrans[0]) / (
                    geotrans[5] - geotrans[4] / geotrans[1] * geotrans[2]))
    coords[:, 0] = ((coords[:, 0] - geotrans[0] - coords[:, 1] * geotrans[2]) / geotrans[1])
    return coords.astype(np.int)


def read_shp(path):
    """
    读取shp文件，对于不同分类体系进行不同的提取。

    :param path:
    :return:
    """
    file = geopandas.read_file(path, encoding='utf-8')
    if "DLBM" in file.columns:
        return file[['DLBM', 'geometry']]
    elif "CC" in file.columns:
        return file[['CC', 'geometry']]


def merge_shape_file(shp_file1_list, save_path):
    """
    融合多个shp文件，并输出到指定路径之下。
    :param shp_file1:
    :param shp_file2:
    :param save_path:
    :return:
    """
    shp1 = geopandas.read_file(shp_file1_list[0])
    shp1 = shp1.dropna()
    for i, path in enumerate(shp_file1_list[1:]):
        shp2 = geopandas.read_file(path)
        shp2 = shp2.dropna()
        shp1 = shp1.append(shp2)

    shp1.to_file(save_path)


def mask_for_polygons(mask, polygons, color, geo_transform, prosrs, geosrs):
    """
    将多边形转换为像素坐标后对mask中对应的区域填充相应的颜色。

    :param mask:
    :param polygons:
    :param color:
    :param geo_transform:
    :param prosrs:
    :param geosrs:
    :return:
    """
    # int_coords = lambda x: coord2pixel(np.array(x).astype(np.float32), geo_transform)
    try:
        exteriors = [np.array(polygons.exterior.coords)]
        interiors = [np.array(poli.coords) for poli in polygons.interiors]
        exteriors.extend(interiors)
    except:
        exteriors = [np.array(poly.exterior.coords) for poly in polygons]
        interiors = [np.array(poli.coords) for poly in polygons for poli in poly.interiors]
        exteriors.extend(interiors)

    point_sample = [str(point).split('.')[0] for point in exteriors[0][0, :]]
    exteriors_transformed = []
    for i in range(len(exteriors)):
        if len(exteriors[i].shape) != 2:
            continue
        if len(point_sample[0]) <= 3:
            exteriors_point = lonlat2geo(prosrs, geosrs, exteriors[i])
        else:
            exteriors_point = exteriors[i]
        exteriors_transformed.append(coord2pixel(exteriors_point, geo_transform))

    cv2.fillPoly(mask, exteriors_transformed, color=color)

    return mask


def get_mask(image, mask, shp_path, ind2num):
    """
    提供遥感影像，准备填充的mask矩阵，以及对应的shp文件路径，即可按照shp文件与遥感影像的重叠区域获得相应的语义分割结果。

    :param image:
    :param mask:
    :param shp_path:
    :return:
    """
    geo_transform = list(image.GetGeoTransform())
    print(geo_transform)
    prosrs, geosrs = getSRSPair(image)
    result_list = read_shp(shp_path)
    result_list.dropna(inplace=True)
    for i in range(len(result_list)):
        if result_list.iloc[i, 0] is not None and result_list.iloc[i, 0][:2] in ind2num.keys():
            try:
                mask = mask_for_polygons(mask, result_list.iloc[i, 1],
                                         ind2num[result_list.iloc[i, 0][:2]],
                                         geo_transform,
                                         prosrs,
                                         geosrs)
            except:
                ...
    return mask


if __name__ == "__main__":
    refer_image_path = "/home/xwtech/遥感识别专用/real_data/影像数据/2020_1.tif"
    save_path = refer_image_path.replace("影像数据", "semantic_mask")

    shp2tif(shp_path="../real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1.shp",
            refer_tif_path=refer_image_path,
            target_tif_path=save_path,
            attribute_field="DLBM",
            nodata_value=-1)
    # image = gdal.Open(image_paths)
    # image_shape = image.ReadAsArray().shape
    # mask = 7 * np.ones((image_shape[1], image_shape[2]))

    # mask = \
    #     get_mask(image, mask,
    #              "../real_data/移交数据和文档/苏北/0.2米航片对应矢量数据/LCRA_2020_2.shp")
    #
    # mask = \
    #     get_mask(image, mask,
    #              "../real_data/移交数据和文档/苏北/0.2米航片对应矢量数据/LCRA_2020_2_1.shp")

    # mask = Image.fromarray(mask.astype(np.uint8))
    # mask.save(save_paths)
    # plt.imshow(mask)
    # plt.show()
