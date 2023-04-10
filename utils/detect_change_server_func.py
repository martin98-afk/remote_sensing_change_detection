# encoding:utf-8
import os
import time

import requests

from utils.detect_change_to_block import *
from utils.gdal_utils import preprocess_rs_image, transform_geoinfo_with_index
from utils.polygon_utils import joint_polygon, shp2tif, read_shp
from utils.segment_semantic import *
from utils.zip_utils import zip2file, file2zip


class DetectTask:
    """
    地貌识别任务管理
    """

    def __init__(self, info_dict, minio_store, model, mysql_conn, args):
        self.info_dict = info_dict
        self.args = args
        self.model = model
        # 导入模型
        self.minio_store = minio_store
        self.mysql_conn = mysql_conn
        self.args["id"] = info_dict["id"]
        self.args["block_size"] = int(
                float(info_dict["occupyArea"]) / float(info_dict["resolution"]))
        self.args["occupyArea"] = float(info_dict["occupyArea"])
        self.topic_type = int(info_dict["topicType"])
        # 输入两张tif，分析两张遥感图像的网格变化区域
        # TODO 根据每个进程创建不同的缓存路径，使用完后删除
        # 构建存储中间结果的路径
        self.root = f"real_data/cache_{info_dict['id']}"
        os.makedirs(self.root, exist_ok=True)
        if self.topic_type != 2:
            # topic type 为2时不进行图片传输，直接使用本地图片进行处理
            self.target_path = self.root + "/target_image.tif"
            self.refer_path = self.root + "/src_image.tif" if self.topic_type == 1 else \
                self.root + "/src_image.zip"
            # 保存传输过来的目标图像和参照信息
            self.save_file_from_url(self.refer_path, info_dict["referImageUrl"])
            self.save_file_from_url(self.target_path, info_dict["comparisonImageUrl"])
        else:
            self.target_path = info_dict["comparisonImageUrl"]
            self.refer_path = info_dict["referImageUrl"]
        # 对数据进行预处理
        self.preprocess_data()

    def delete_file(self, root):
        for path in os.listdir(root):
            path = os.path.join(root, path)
            if os.path.isdir(path):
                for file in os.listdir(path):
                    os.remove(os.path.join(path, file))
                os.rmdir(path)
            else:
                os.remove(path)
        os.rmdir(root)

    def save_file_from_url(self, save_path, url_path):
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

    def preprocess_data(self):
        """
        对传输过来的图像进行预处理，转换为3857格式的影像，同时将分辨率转为指定分辨率。

        :return:
        """
        if self.topic_type != 0:
            # 对影像进行预处理
            preprocess_rs_image(self.refer_path,
                                self.target_path,
                                self.info_dict["resolution"],
                                save_root="same")
        else:
            transform_geoinfo_with_index(self.target_path, 3857)
            gdal.Warp(self.target_path, self.target_path,
                      format="GTiff",
                      xRes=float(self.info_dict["resolution"]),
                      yRes=float(self.info_dict["resolution"]))

    def process_task(self):
        if self.topic_type == 0:
            # 读取zip文件，将zip文件解压，并找到相应的shp路径
            if os.path.exists(self.root + "/src_image"):
                files = glob(self.root + "/src_image/*")
                [os.remove(file) for file in files]
            zip2file(self.refer_path, self.root)
            self.refer_path = glob(self.root + "/src_image/*.shp")[0]
            paths = \
                DetectChangeServer.change_polygon_detect(self.model,
                                                         src_shp=self.refer_path,
                                                         target_image=self.target_path,
                                                         IMAGE_SIZE=512,
                                                         args=self.args)
        else:
            paths = \
                DetectChangeServer.change_block_detect(self.model,
                                                       src_image=self.refer_path,
                                                       target_image=self.target_path,
                                                       IMAGE_SIZE=512, args=self.args)
            # 保存预处理后的遥感图像
            current_time = paths[0][:-4].split("_")[-1]
            file_name = f"refer_image_{current_time}.tif"
            self.minio_store.fput_object(f"change_result/{file_name}", self.refer_path)
            file_name = f"comparison_image_{current_time}.tif"
            self.minio_store.fput_object(f"change_result/{file_name}", self.target_path)

        self.save_result(paths)

    def save_result(self, paths):
        # 将tuple类转为list类，方便添加后续需要删除的路径
        paths = list(paths)
        for path in paths:
            # 将路径中的文件上传到minio服务器上
            file_name = os.path.basename(path)
            self.minio_store.fput_object(f"change_result/{file_name}", path)
            if file_name.endswith("shp"):
                self.minio_store.fput_object(f"change_result/{file_name.replace('shp', 'dbf')}",
                                             path.replace("shp", "dbf"))
                # 获取所有shp文件路径
                shp_path = path.replace("shp", "*")
                shp_paths = glob(shp_path)
                # 将所有shp文件压缩成zip文件
                zip_path = file_name.replace(".shp", ".zip")
                file2zip(zip_path, shp_paths)
                self.minio_store.fput_object(f"change_result/{os.path.basename(zip_path)}",
                                             zip_path)
                shp_paths.append(zip_path)
            elif file_name.endswith("png"):
                save_url = f"http://221.226.175.85:9000/ai-platform/change_result/{file_name}"

        # 获取当前变化识别出的变化图斑数量
        file = read_shp(paths[-1])
        change_num = len(file[file["class"] != 0])
        # 连接mysql数据库，更新数据处理进度
        # TODO 将存储的文件url以及任务id存储到数据库中。
        if self.topic_type == 2:
            self.mysql_conn.update_result(self.info_dict["id"], save_url, change_num)
        else:
            self.mysql_conn.write_to_mysql_relation(self.info_dict["id"], save_url, change_num)
        self.mysql_conn.update_status(self.info_dict["id"], status=5)
        self.mysql_conn.write_to_mysql_progress(self.info_dict["id"],
                                                str(100) + "%")
        # 回收内存
        gc.collect()
        # 删除所有缓存文件
        self.delete_file(self.root)


class DetectChangeServer:
    """
    服务器变化识别函数类
    """

    @staticmethod
    def detect_change(image0, image1, include_class=[1], identify_classes=[3, 4, 5, 8]):
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
        :return: png变化识别结果, png耕地图斑掩模, tif变化识别结果, shp变化识别结果
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
        save_path = joint_polygon(mask_path.replace("tif", "shp"), result_shp_path,
                                  args['occupyArea'])
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
        semantic_result_src = test_big_image(model, src_image, IMAGE_SIZE, args)
        semantic_result_target = test_big_image(model, target_image, IMAGE_SIZE, args)
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
