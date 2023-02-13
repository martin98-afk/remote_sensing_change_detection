import gc
from glob import glob

from osgeo import gdal

from utils.polygon_utils import raster2vector
from config import *

gdal.AllRegister()  # 先载入数据驱动，也就是初始化一个对象，让它“知道”某种数据结构，但是只能读，不能写

"""
将变化识别区域按照指定大小的方块进行切割，并将变化识别区域的最小单位转变为划分的这小小方块
"""


def load_tif(filepath, block_size):
    change_data = gdal.Open(filepath)  # 打开文件
    origin_proj = change_data.GetProjection()
    change_img_width, change_img_height = change_data.RasterXSize, change_data.RasterYSize  # 获取影像的宽高
    num_w = change_img_width // block_size
    num_h = change_img_height // block_size
    rm_w = change_img_width % block_size
    rm_h = change_img_height % block_size

    return change_data, num_w, num_h, rm_w, rm_h, origin_proj


def distribute_count(array):
    list = []
    for i in range(int(np.max(array)) + 1):
        list.append(np.sum(array == i))
    if len(list) > 1:
        return np.argmax(list[1:])
    else:
        return 0


def judge(array_in, s, block_size):
    """
    判断某一100*100的区域内变化像素数量，阈值为s，若大于s则该区域标位变化区域，小于s则该区域不标为变化区域

    :param array_in: 某区域的gdal.ReadAsArray结果
    :param s: 判断是否为变化区域的阈值
    :return: 变化区域标记mask_array【判定为变化则值全为1，未变化则值全为0】
    """
    count = np.count_nonzero(array_in)
    index = distribute_count(array_in)
    if (count / block_size / block_size) >= s:
        array1 = np.ones_like(array_in) * index
        # array1[0, :] = 0
        # array1[-1, :] = 0
        # array1[:, 0] = 0
        # array1[:, -1] = 0
    else:
        array1 = np.zeros_like(array_in)
        # array1[0, :] = 1
        # array1[-1, :] = 1
        # array1[:, 0] = 1
        # array1[:, -1] = 1
    return array1


def line_select(num_w, rm_w, h1, data, s, block_size):
    """
    图像某一行的判定的组合，对这一行的图像逐块判定，行宽为图像宽度，高为100，
    在按100分块后，右侧的剩余部分单独判定，并拼接到逐块判定的结果后

    :param num_w: 图像包含的列数（图像宽方向像素总数整除100）
    :param rm_w: 图像右边缘宽度（图像宽方向像素总数对100取余）
    :param h1: 该行图像左上角y坐标（x坐标一定为0）
    :param data: 图像数据
    :param s: 判断阈值
    :return: 该行的变化mask_array
    """
    line_array_out = judge(data.ReadAsArray(0, h1, block_size, block_size), s, block_size)
    for i in range(1, num_w):
        array_out = judge(data.ReadAsArray(i * block_size, h1, block_size, block_size), s, block_size)
        line_array_out = np.hstack((line_array_out, array_out))
    array_out1 = judge(data.ReadAsArray(num_w * block_size, h1, rm_w, block_size), s, block_size)
    line_array_out = np.hstack((line_array_out, array_out1))
    return line_array_out


def change_detect(num_w, num_h, rm_w, rm_h, data, s, block_size):
    """
    通过逐行，逐块扫描并判定，最终生成整个图像的判定mask_array，
    图像高方向在按100宽分行后剩余部分单独判定并拼接到逐行判定的结果后

    :param num_w: 图像包含的列数（图像宽方向像素总数整除100）
    :param num_h: 图像包含的行数（图像高方向像素总数整除100）
    :param rm_w: 图像右边缘宽度（图像宽方向像素总数对100取余）
    :param rm_h: 图像下边缘宽度（图像宽方向像素总数对100取余）
    :param data: 图像数据
    :param s: 变化阈值
    :return:
    """
    all_out_array = line_select(num_w, rm_w, 0, data, s, block_size)
    for i in range(1, num_h):
        out_array = line_select(num_w, rm_w, i * block_size, data, s, block_size)
        all_out_array = np.vstack((all_out_array, out_array))

    last_line_array_out = judge(data.ReadAsArray(0, num_h * block_size, block_size, rm_h), s, block_size)
    for i in range(1, num_w):
        array_out = judge(data.ReadAsArray(i * block_size, num_h * block_size, block_size, rm_h), s, block_size)
        last_line_array_out = np.hstack((last_line_array_out, array_out))
    array_out1 = judge(data.ReadAsArray(num_w * block_size, num_h * block_size, rm_w, rm_h), s, block_size)
    last_line_array_out = np.hstack((last_line_array_out, array_out1))
    all_out_array = np.vstack((all_out_array, last_line_array_out))
    return all_out_array


def ARR2TIF(data, origin_transform, origin_proj, out_filepath):
    """
    整个图像的mask_array转为一张新的tif

    :param data: 数组数据
    :param origin_dataset:gdal读取的原图像dataset
    :param filename: 存储的文件名
    :param origin_proj: 原图像投影信息
    :return:
    """
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    # 维度和numpy数组相反
    new_dataset = driver.Create(out_filepath, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)

    # 读取之前的tif信息，为新生成的tif添加地理信息
    # 如果不增加这一步，则生成的图片没有经纬度坐标、投影的信息
    # 获取投影信息
    new_dataset.SetProjection(origin_proj)
    # 仿射矩阵
    new_dataset.SetGeoTransform(origin_transform)
    band = new_dataset.GetRasterBand(1)
    band.WriteArray(data)
    gc.collect()


if __name__ == "__main__":
    s = 0.35  # 判定阈值
    block_size = 100
    in_filepath = glob(f"../output/semantic_result/tif/detect_change_2_2.tif")
    out_tif_filepath = [item.replace("/tif/", "/change_result/")
                            .replace("detect_change", f"detect_change_block_{str(block_size)}") for item in
                        in_filepath]

    out_shp_filepath = [item.replace(".tif", ".shp") for item in out_tif_filepath]
    for i, (in_file, out_tif, out_shp) in enumerate(zip(in_filepath, out_tif_filepath,
                                                        out_shp_filepath)):
        change_data, num_w, num_h, rm_w, rm_h, origin_proj = load_tif(in_file, block_size=block_size)
        arrayout = change_detect(num_w, num_h, rm_w, rm_h, change_data, s, block_size=block_size)
        ARR2TIF(arrayout, change_data, origin_proj, out_tif)
        raster2vector(arrayout, origin_proj, vector_path=out_shp)
