# encoding:utf-8
import argparse
import sys
import time

from utils.detect_change_to_block import *
from utils.gdal_utils import preprocess_rs_image
from utils.polygon_utils import joint_polygon, shp2tif
from utils.segment_semantic import *


def get_parser():
    """
    收集命令行提供的参数

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info-path', type=str, default='output/ss_eff_b0.yaml',
                        help='需要使用的模型信息文件存储路径')
    parser.add_argument('--model-type', type=str, default='pytorch',
                        help='需要使用的模型信息文件存储路径')
    parser.add_argument('--target-dir', type=str, default='./real_data',
                        help='需要进行预测的图像存储路径')
    parser.add_argument('--file-path', type=str, default='test9.tif',
                        help='语义分割、变化识别、图斑变化识别使用的图像')
    parser.add_argument('--file-path-extended', type=str, default='test9_2.tif',
                        help='仅变化识别使用的对比图像(过去)')
    parser.add_argument('--shp-path', type=str,
                        default='real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_耕地.shp',
                        help='变化识别中参考的矢量文件路径，在detect-change-shp模式下使用')
    parser.add_argument('--output-dir', type=str, default='output/semantic_result',
                        help='输出结果存储路径')
    parser.add_argument('--remove-tif', action='store_true',
                        help='是否要将结果储存为栅格形式')
    parser.add_argument('--mode', type=str, default='detect-change',
                        help='程序运行模式，包括detect-change、segment-semantic、detect-change-shp')
    parser.add_argument('--half', action='store_true',
                        help='模型计算时是否开启半精度运算')
    parser.add_argument('--slic', type=int, default=0,
                        help='是否对分割结果进行slic聚类算法后处理, 设置为0则不使用slic算法')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='进行验证时使用的批量大小')
    parser.add_argument('--device', type=str, default='cuda', help='确认训练模型使用的机器')
    parser.add_argument('--log-output', action='store_true', help='在控制台打印程序运行结果')
    parser.add_argument('--change-threshold', type=float, default=0.35,
                        help='最后网格化判断网格是否变化的阈值，1即如果网格中像素比例超过该阈值则判定该网格为变化区域。')
    parser.add_argument('--block-size', type=float, default=25,
                        help='变化区域网格化的大小。')
    opt = parser.parse_args()
    return opt


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


def change_polygon_detect(model, target_image, IMAGE_SIZE, args):
    """
    进行变化识别，识别结果同时进行网格化处理。

    :param model: 进行语义分割的模型。
    :param target_image: 变化识别目标图像路径。
    :param IMAGE_SIZE: 图像大小
    :param args: 所有系统参数。
    :return:
    """
    RSPipeline.print_log('正在执行依据图斑变化检测模块')
    image = gdal.Open(target_image)
    semantic_result = test_big_image(model, target_image,
                                     IMAGE_SIZE, args, denominator=2, addon=0)
    ARR2TIF(semantic_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
    raster2vector(tif_path, vector_path=shp_path, remove_tif=False)
    # TODO 根据耕地mask裁剪出对应耕地识别结果。
    # 将检测出的变化区域转换为原始三调图斑，如果三调图斑中一个图斑中有0.1部分的面积被覆盖到，就算这个图斑为变化区域，并存储最终结果。
    save_path = joint_polygon('real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_耕地.shp',
                              shp_path)
    save_tif_path = save_path.replace("shp", "tif")
    shp2tif(shp_path=save_path,
            refer_tif_path=target_image,
            target_tif_path=save_tif_path,
            attribute_field="DLBM",
            nodata_value=0)
    return save_tif_path


def index2RGB(image):
    dict = {}
    dict[0] = np.array([0, 0, 0])
    transformed_image = np.zeros((image.shape[0], image.shape[1], 3))
    # image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=-1)
    for i in range(1, 9):
        dict[i] = np.round(np.random.rand(3) * 255)
        transformed_image[image == i] = np.round(np.random.rand(3) * 255)
    return transformed_image


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
    shp_path = f'output/semantic_result/change_result/detect_change_block_{round(time.time())}.shp'
    tif_block_path = f'output/semantic_result/tif/detect_change_block_{round(time.time())}.tif'
    png_block_path = f'output/semantic_result/tif/detect_change_block_{round(time.time())}.png'
    semantic_result_src = test_big_image(model, src_image,
                                         IMAGE_SIZE, args, denominator=2, addon=0)
    semantic_result_target = test_big_image(model, target_image,
                                            IMAGE_SIZE, args, denominator=2, addon=50)
    assert semantic_result_src.shape == semantic_result_target.shape, "输入数据形状大小不一致"
    RSPipeline.print_log('两年份遥感数据语义分割已完成')
    image = gdal.Open(src_image)
    change_result = detect_change(semantic_result_src, semantic_result_target)
    ARR2TIF(change_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
    RSPipeline.print_log('变化识别完成已保存结果')

    # 将变化区域转变为50*50的方块，以方便使用和统计准确率
    RSPipeline.print_log('开始将变化识别结果网格化')
    threshold = args['change_threshold']  # 判定阈值
    block_size = args['block_size']

    change_data, num_w, num_h, rm_w, rm_h, origin_proj = load_tif(tif_path, block_size)
    arrayout = change_detect(num_w, num_h, rm_w, rm_h, change_data, threshold, block_size)
    RSPipeline.print_log('变化识别结果网格化完毕并存储')
    ARR2TIF(arrayout, change_data.GetGeoTransform(), origin_proj, tif_block_path)
    image = index2RGB(arrayout).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(png_block_path)
    raster2vector(tif_block_path, vector_path=shp_path, remove_tif=True)
    return png_block_path, tif_path, shp_path


if __name__ == '__main__':
    args = get_parser()
    # 判断是否要将输出打印到控制台
    if args.log_output:
        f_handler = open('./out_detect_change.log', 'w', encoding='utf-8')
        __console__ = sys.stdout
        sys.stdout = f_handler
        print('输出输入到日志中!')

    # 导入模型参数
    data, model = RSPipeline.load_model(args.model_info_path,
                                        args.model_type,
                                        args.device,
                                        args.half)
    IMAGE_SIZE = data['image_size'][0]
    num_classes = data['num_classes']
    ind2label = data['index to label']

    # 建立结果存放的文件夹
    os.makedirs(args.output_dir + '/change_result', exist_ok=True)
    os.makedirs(args.output_dir + '/tif', exist_ok=True)
    os.makedirs(args.output_dir + '/shp', exist_ok=True)
    # 判断是否参数符合规范
    assert args.mode in ['segment-semantic', 'detect-change', 'detect-change-shp'], \
        'Wrong process mode. Available options are: segment-semantic, detect-change,' \
        'detect-change-shp.'

    if args.mode == 'segment-semantic':
        RSPipeline.print_log('正在执行语义分割模块')
        # 直接进行语义分割
        test_semantic_segment_files(model,
                                    ind2label,
                                    IMAGE_SIZE,
                                    args)
    elif args.mode == 'detect-change-shp':
        RSPipeline.print_log('正在执行依据图斑变化检测模块')
        image_list = glob(os.path.join(args.target_dir, args.file_path))
        # 防止windows环境glob获取路径中包含\\
        image_list = [RSPipeline.check_path(path) for path in image_list]
        # 根据图像和对应的矢量路劲制作模板
        save_path = [path.replace("processed_data", "trad_alg")
                         .replace(".tif", f"_mask.tif") for path in image_list]
        # 将耕地矢量图斑文件转为栅格数据
        for image_path, save in zip(image_list, save_path):
            shp2tif(shp_path=args.shp_path,
                    refer_tif_path=image_path,
                    target_tif_path=save,
                    attribute_field="DLBM",
                    nodata_value=0)
        RSPipeline.print_log('根据图像和对应的矢量路劲制作模板完成')
        for i, (image_path, mask_path) in enumerate(zip(image_list, save_path)):
            file_name = image_path.split('/')[-1][:-4]
            # 确定最后变化结果保存路径
            tif_path = args.output_dir + f'/tif/detect_change_{file_name}.tif'
            result_shp_path = args.output_dir + f'/change_result/detect_change_{file_name}.shp'
            # 打开目标图像获取图像的地理位置信息
            image = gdal.Open(image_path)
            # 进行地貌类型识别
            semantic_result = test_big_image(model, image_path,
                                             IMAGE_SIZE, vars(args))
            # 打开当前影像的三调耕地图斑模板，将耕地以外的识别结果都删除，方便统计变化区域
            mask = Image.open(mask_path)
            mask = np.array(mask)
            semantic_result[mask == 0] = 0
            # 将变化结果保存为栅格数据
            ARR2TIF(semantic_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
            # 将变化结果保存为矢量数据
            raster2vector(tif_path, vector_path=result_shp_path, remove_tif=True)
            # TODO 根据耕地mask裁剪出对应耕地识别结果。
            # 将检测出的变化区域转换为原始三调图斑，如果三调图斑中一个图斑中有0.1部分的面积被覆盖到，就算这个图斑为变化区域，并存储最终结果。
            RSPipeline.print_log('开始将变化识别结果图斑化')
            joint_polygon(args.shp_path, result_shp_path)
            RSPipeline.print_log('变化识别结果图斑化完成')
        RSPipeline.print_log('变化结果板块提取完毕')
    elif args.mode == 'detect-change':
        RSPipeline.print_log('正在执行依据图像变化检测模块')
        # 输入两年份的图片
        image_src_list = glob(os.path.join(args.target_dir, args.file_path_extended))
        image_target_list = glob(os.path.join(args.target_dir, args.file_path))
        # 进行变化识别
        tif_paths = []
        for i, (image_src, image_target) in enumerate(zip(image_src_list, image_target_list)):
            # 如果两个图片没有对齐，先通过gdal工具包进行分辨率以及图像对齐
            preprocess_rs_image(image_src, image_target, 0.5, save_root="same")
            file_name = image_target.split("/")[-1][:-4]
            tif_path = args.output_dir + f'/tif/detect_change_{file_name}.tif'
            shp_path = args.output_dir + f'/change_result/detect_change_{file_name}.shp'
            semantic_result_src = test_big_image(model, image_src,
                                                  IMAGE_SIZE, vars(args))
            semantic_result_target = test_big_image(model, image_target,
                                                  IMAGE_SIZE, vars(args))
            RSPipeline.print_log('两年份遥感数据语义分割已完成')
            image = gdal.Open(image_src)
            change_result = detect_change(semantic_result_src, semantic_result_target)
            ARR2TIF(change_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
            RSPipeline.print_log('变化识别完成已保存结果')
            tif_paths.append(tif_path)

        # 将变化区域转变为50*50的方块，以方便使用和统计准确率
        RSPipeline.print_log('开始将变化识别结果网格化')
        threshold = args.change_threshold  # 判定阈值
        block_size = args.block_size
        out_tif_filepath = [item.replace('/tif/', '/change_result/')
                                .replace('detect_change', 'detect_change_block') for item in
                            tif_paths]

        out_shp_filepath = [item.replace('.tif', '.shp') for item in out_tif_filepath]
        for j, (in_file, out_tif, out_shp) in enumerate(zip(tif_paths, out_tif_filepath,
                                                            out_shp_filepath)):
            change_data, num_w, num_h, rm_w, rm_h, origin_proj = load_tif(in_file, block_size)
            arrayout = change_detect(num_w, num_h, rm_w, rm_h, change_data, threshold, block_size)
            ARR2TIF(arrayout, change_data.GetGeoTransform(), origin_proj, out_tif)
            raster2vector(out_tif, vector_path=out_shp)
        RSPipeline.print_log('变化识别结果网格化完毕')
    del model
    gc.collect()
