# encoding:utf-8
import time

from utils.detect_change_to_block import *
from utils.polygon_utils import joint_polygon, shp2tif, read_shp
from utils.segment_semantic import *


class DetectChangeServer:
    """
    服务器变化识别函数类
    """

    @staticmethod
    def detect_change(image0, image1, include_class=[1], identify_classes=[3, 4, 5, 6, 8]):
        """
        指定要检测的标签，检测image 1相比image 0中农田区域变化成指定标签地形的区域。

        :param image0: 原图像
        :param image1: 待识别变化的图像
        :param include_class: 待识别变化的地貌类型
        :param identify_classes: 变化识别方向的识别
        :return: 变化识别结果
        """
        image0_farm = np.zeros_like(image0)
        for i in include_class:
            image0_farm += image0 == i
        image1_identify_classes = [(image1 == clas) for clas in identify_classes]
        change_result = np.zeros((image0.shape[0], image0.shape[1]))
        for i, new_image in enumerate(image1_identify_classes):
            change = np.logical_and(image0_farm, new_image)
            change_result += identify_classes[i] * change
        return change_result

    @staticmethod
    def change_polygon_detect(model, src_shp, target_image, IMAGE_SIZE, args):
        """
        进行变化识别图斑变化检测，提供图斑数据，识别指定图像中发生变化的图斑区域。

        :param model: 进行语义分割的模型。
        :param target_image: 变化识别目标图像路径。
        :param IMAGE_SIZE: 图像大小
        :param args: 所有系统参数。
        :return:
        """
        RSPipeline.print_log('正在执行依据图斑变化检测模块')

        # 根据图像和对应的矢量路劲制作模板
        mask_path = target_image.replace(".tif", f"_mask.tif")
        # TODO 获取当前shp文件中的attribute field
        file = read_shp(src_shp)
        field = file.columns[0]
        # 将耕地矢量图斑文件转为栅格数据
        shp2tif(shp_path=src_shp,
                refer_tif_path=target_image,
                target_tif_path=mask_path,
                attribute_field=field,
                nodata_value=0)
        # 将相交的耕地图斑部分再重新转换为矢量
        # 将变化结果保存为矢量数据
        raster2vector(mask_path, vector_path=mask_path.replace("tif", "shp"),
                      remove_tif=args['remove_tif'])

        RSPipeline.print_log('根据图像和对应的矢量路劲制作模板完成')
        # 确定最后变化结果保存路径
        target_dir = os.path.dirname(target_image)
        tif_path = f'{target_dir}/detect_change.tif'
        # 获取当前时间戳
        current_time = round(time.time())
        result_shp_path = f'output/semantic_result/change_result/detect_change_spot_{current_time}.shp'
        result_tif_path = result_shp_path.replace(".shp", ".tif")
        result_png_path = result_shp_path.replace(".shp", ".png")
        # 打开目标图像获取图像的地理位置信息
        image = gdal.Open(target_image)
        # 进行地貌类型识别
        semantic_result = test_big_image(model, target_image, IMAGE_SIZE, args)
        # 打开当前影像的三调耕地图斑模板，将耕地以外的识别结果都删除，方便统计变化区域
        mask = Image.open(mask_path)
        mask = np.array(mask)
        semantic_result[mask == 0] = 0
        # 将变化结果保存为栅格数据
        ARR2TIF(semantic_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
        # 将变化结果保存为矢量数据
        raster2vector(tif_path, vector_path=result_shp_path, remove_tif=args['remove_tif'])
        # TODO 根据耕地mask裁剪出对应耕地识别结果。
        # 将检测出的变化区域转换为原始三调图斑，如果三调图斑中一个图斑中有0.1部分的面积被覆盖到，就算这个图斑为变化区域，并存储最终结果。
        RSPipeline.print_log('开始将变化识别结果图斑化')
        save_path = joint_polygon(mask_path.replace("tif", "shp"), result_shp_path)
        RSPipeline.print_log('变化识别结果图斑化完成')
        # 将识别结果存储为tif格式
        shp2tif(shp_path=save_path,
                refer_tif_path=target_image,
                target_tif_path=result_tif_path,
                attribute_field="class",
                nodata_value=0)
        # 将识别结果转存为png格式
        result_image = Image.open(result_tif_path)
        image = DetectChangeServer.index2RGB(np.array(result_image)).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(result_png_path)
        # 将所有耕地图斑转存为png格式
        result_image = Image.open(mask_path)
        image = DetectChangeServer.index2RGB(np.array(result_image)).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(result_tif_path.replace(".tif", "_mask.png"))
        return result_png_path, result_tif_path.replace(".tif", "_mask.png"), result_tif_path, \
               save_path

    @staticmethod
    def index2RGB(image):
        """
        将索引图像转为RGBA图像

        :param image:
        :return:
        """
        dict = {}
        dict[0] = np.array([0, 0, 0])
        transformed_image = np.zeros((image.shape[0], image.shape[1], 4))
        # image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=-1)
        for i in range(1, 9):
            dict[i] = np.round(np.random.rand(3) * 255)
            transformed_image[image == i] = np.array([255, 255, 255, 255])
        return transformed_image

    @staticmethod
    def change_block_detect(model, src_image, target_image, IMAGE_SIZE, args):
        """
        进行变化识别，识别结果同时进行网格化处理。

        :param model: 进行语义分割的模型。
        :param src_image: 源图像存储路径
        :param target_image: 变化识别目标图像路径。
        :param IMAGE_SIZE: 图像大小
        :param args: 所有系统参数。
        :return:
        """
        RSPipeline.print_log('正在执行依据图像变化检测模块')
        # 输入两年份的图片
        # 进行变化识别
        tif_path = 'output/semantic_result/tif/detect_change.tif'
        current_time = round(time.time())
        shp_path = f'output/semantic_result/change_result/detect_change_block_{current_time}.shp'
        tif_block_path = shp_path.replace(".shp", ".tif")
        png_block_path = shp_path.replace(".shp", ".png")

        # 对两幅图分别进行地貌类型识别
        semantic_result_src = test_big_image(model, src_image,
                                             IMAGE_SIZE, args, denominator=2, addon=0)
        semantic_result_target = test_big_image(model, target_image,
                                                IMAGE_SIZE, args, denominator=2, addon=50)
        # 检查两个图像大小是否一致
        assert semantic_result_src.shape == semantic_result_target.shape, "输入数据形状大小不一致"
        RSPipeline.print_log('两年份遥感数据语义分割已完成')
        # 进行变化识别，导入源图像的地理位置信息
        image = gdal.Open(src_image)
        change_result = DetectChangeServer.detect_change(semantic_result_src,
                                                         semantic_result_target)
        # 保存变化信息为栅格图像
        ARR2TIF(change_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
        RSPipeline.print_log('变化识别完成已保存结果')
        # 将变化区域转变为50*50的方块，以方便使用和统计准确率
        RSPipeline.print_log('开始将变化识别结果网格化')
        # 获取超参数设定的变化面积占比阈值以及网格大小
        threshold = args['change_threshold']  # 判定阈值
        block_size = args['block_size']
        # 对变化识别结果进行网格化
        change_data, num_w, num_h, rm_w, rm_h, origin_proj = load_tif(tif_path, block_size)
        arrayout = change_detect(num_w, num_h, rm_w, rm_h, change_data, threshold, block_size)
        RSPipeline.print_log('变化识别结果网格化完毕并存储')
        # 将变化网格结果存储为栅格图片
        ARR2TIF(arrayout, change_data.GetGeoTransform(), origin_proj, tif_block_path)
        # 同时存储为有rgba四通道的png图像进行前端展示，目前是把所有变化结果都归为一类，后续进行更改
        image = DetectChangeServer.index2RGB(arrayout).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(png_block_path)
        # 栅格图像转为矢量文件
        raster2vector(tif_block_path, vector_path=shp_path, remove_tif=False)
        # 向shp文件中添加面积字段
        return png_block_path, tif_block_path, shp_path
