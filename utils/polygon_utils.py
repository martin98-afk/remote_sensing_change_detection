import cv2
import geopandas
from PIL import Image
from osgeo import osr, gdal, ogr

from config import *


def transform_geoinfo(source_image, target_image, save_path=None):
    """
    统一投影系统。

    :param save_path:
    :param target_image: 要转入投影系统的目标图像，可以为路径。
    :param source_image: 待转换投影系统的目标图像，可以为路径。
    :return:
    """
    if type(source_image) == str:
        source_image = gdal.Open(source_image)
    if type(target_image) == str:
        target_image = gdal.Open(target_image)
    if save_path is not None:
        gdal.Warp(save_path, source_image, format="GTiff", dstSRS=target_image.GetProjection())
    else:
        new_image = gdal.Warp("", source_image, format="MEM", dstSRS=target_image.GetProjection())
        return new_image


def cut_raster(raster_path, vector_path,
               refer_path="../real_data/processed_data/2020_1_1_res_0.5.tif"):
    """
    统一投影系统。

    :param vector_path:
    :param raster_path:
    :param refer_path:
    :return:
    """
    image_new = transform_geoinfo(raster_path, refer_path)
    return gdal.Warp("", image_new, cutlineDSName=vector_path,
                     format="MEM", cropToCutline=False, copyMetadata=True, dstNodata=0)


def raster2vector(raster_path, vector_path, label=None,
                  field_name='class', ignore_vales=None, remove_tif=True):
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
    target_image = gdal.Open("real_data/移交数据和文档/苏北/0.2米航片/2020_2_1.tif")
    raster = transform_geoinfo(raster, target_image)
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
    gdal.Polygonize(band, None, poly_layer, 0)

    if ignore_vales is not None:
        for feature in poly_layer:
            class_value = feature.GetField('class')
            for ignore_value in ignore_vales:
                if class_value == ignore_value:
                    poly_layer.DeleteFeature(feature.GetFID())
                    break
    polygon.SyncToDisk()
    polygon = None
    # 将数字标签转为中文标签
    if label is not None:
        shp_file = read_shp(vector_path, encoding='gbk')
        label = ["其他"] + list(label.values())
        out_label = []
        for i in range(shp_file.shape[0]):
            out_label.append(label[int(shp_file['class'].iloc[i])])
        shp_file.insert(shp_file.shape[1], 'label', out_label)
        shp_file.to_file(vector_path, encoding='gb18030')

    if remove_tif:
        os.remove(raster_path)


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


def shp2tif(shp_path, refer_tif_path, target_tif_path,
            attribute_field="class", nodata_value=0,
            return_tif=False):
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
    if return_tif:
        return target_ds
    else:
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


def read_shp(path, encoding='utf-8'):
    """
    读取shp文件，对于不同分类体系进行不同的提取。

    :param encoding: 读取文件的编码方式
    :param path: 读取文件的路径
    :return:
    """
    file = geopandas.read_file(path, encoding=encoding)
    if "DLBM" in file.columns:
        return file[['DLBM', 'geometry']]
    elif "CC" in file.columns:
        return file[['CC', 'geometry']]
    else:
        return file[['class', 'geometry']]


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

    :param ind2num:
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


def reproject(inputfile, outputfile, layername, insrs, outsrs):
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(insrs, outsrs)

    # get the input layer
    inDataSet = driver.Open(inputfile)
    inLayer = inDataSet.GetLayer()

    # create the output layer
    outputShapefile = outputfile
    outDataSet = driver.CreateDataSource(outputShapefile)
    print(inLayer.GetGeomType())
    outLayer = outDataSet.CreateLayer(layername, geom_type=inLayer.GetGeomType())

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # destroy the features and get the next input feature
        outFeature.Destroy()
        inFeature.Destroy()
        inFeature = inLayer.GetNextFeature()

    # close the shapefiles
    inDataSet.Destroy()
    outDataSet.Destroy()


def joint_polygon(target_shp_file, con_shp_file):
    """
    计算2个不同shp文件中多边形的交集，同时计算相交面积，然后保留相交面积大于一定阈值的目标矢量文件中的多边形。

    :param target_shp_file: 目标矢量文件，
    :param con_shp_file: 识别出的变化区域矢量文件。
    :return:
4    """
    target = read_shp(target_shp_file)
    detect = read_shp(con_shp_file)
    detect_list = [3, 4, 5, 8]
    save_path = con_shp_file.replace(".shp", "_spot.shp")
    save_shp = None
    for ind in detect_list:
        detect_ind = detect[detect['class'] == ind]
        pop_list = []
        for i in range(len(detect_ind)):
            for j in range(len(target)):
                if detect_ind.iloc[i, 1].intersection(target.iloc[j, 1]).area / target.iloc[
                    j, 1].area > 0.1:
                    pop_list.append(j)
        pop_list = set(pop_list)
        if save_shp is None:
            select = target.iloc[list(pop_list), :]
            select["class"] = [ind] * len(pop_list)
            save_shp = select
        else:
            select = target.iloc[list(pop_list), :]
            select["class"] = [ind] * len(pop_list)
            save_shp = save_shp.append(select)
    save_shp.to_file(save_path)


if __name__ == "__main__":
    image = cut_raster("../real_data/test8.tif",
                       "../real_data/移交数据和文档/rice/rice.shp")

    # # 从大类标注文件中提取耕地图斑
    # shp_file = read_shp("../real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_major_class.shp")
    # shp_file = shp_file[shp_file.iloc[:, 0] == 1]
    # shp_file.to_file("../real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_farmland.shp")
    # refer_image_path = glob("../real_data/processed_data/2020_1_*_res_0.5.tif")
    # save_path = [path.replace("processed_data", "semantic_mask")
    #                  .replace(".tif", "_farmland.tif") for path in refer_image_path]
    # # 将耕地矢量图斑文件转为栅格数据
    # for image_path, save in zip(refer_image_path, save_path):
    #     shp2tif(shp_path="../real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_farmland.shp",
    #             refer_tif_path=image_path,
    #             target_tif_path=save,
    #             attribute_field="DLBM",
    #             nodata_value=0)
    # # 使用栅格数据作为mask从2020年和2021年的图像之中提取对应图斑部分
    # refer_image_path = glob("../real_data/processed_data/202*_1_*_res_0.5.tif")
    # save_path = [path.replace("processed_data", "semantic_mask")
    #                  .replace(".tif", "_farmland.tif")
    #                  .replace("2021", "2020") for path in refer_image_path]
    # for path1, path2 in zip(refer_image_path, save_path):
    #     image = gdal.Open(path1)
    #     mask = Image.open(path2)
    #     mask = np.array(mask)
    #     mask = np.repeat(mask[..., np.newaxis], 3, 2)
    #     image_array = np.transpose(image.ReadAsArray(), (1, 2, 0))
    #     image_array[mask == 0] = 0
    #     write_img(path1.replace(".tif", "_farmland.tif"), image.GetProjection(),
    #               image.GetGeoTransform(),
    #               np.transpose(image_array, (2, 0, 1)))

    # # TODO 测试将变化检测出的小方块映射到耕地图斑之上
    # shp_file = read_shp("../output/semantic_result/change_result/detect_change_block_1_4.shp")
    # tif = shp2tif(shp_path="../output/semantic_result/change_result/detect_change_block_1_4.shp",
    #               refer_tif_path="../real_data/processed_data/2020_1_4_res_0.5.tif",
    #               target_tif_path="../output/semantic_result/tif/detect_change_block_1_4.tif",
    #               attribute_field="class",
    #               nodata_value=0,
    #               return_tif=True)
    # tif_array = tif.ReadAsArray()
    #
    # mask = Image.open("../real_data/semantic_mask/2020_1_4_res_0.5_farmland.tif")
    # mask = np.array(mask)
    # tif_array[mask == 0] = 0
    # plt.imshow(tif_array)
    # plt.show()
