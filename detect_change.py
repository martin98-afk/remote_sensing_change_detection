# encoding:utf-8
"""
变化识别本地测试
"""
import argparse
import sys

from utils.detect_change_server_func import DetectChangeServer as DCS
from utils.detect_change_to_block import *
from utils.gdal_utils import preprocess_rs_image, transform_geoinfo_with_index
from utils.polygon_utils import joint_polygon, shp2tif, read_shp
from utils.segment_semantic import *


def get_parser():
    """
    收集命令行提供的参数

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info-path', type=str, default='output/ss_eff_b0_with_glcm.yaml',
                        help='需要使用的模型信息文件存储路径')
    parser.add_argument('--model-type', type=str, default='pytorch',
                        help='需要使用的模型信息文件存储路径')
    parser.add_argument('--target-dir', type=str, default='./real_data',
                        help='需要进行预测的图像存储路径')
    parser.add_argument('--file-path', type=str, default='test_google.tif',
                        help='语义分割、变化识别、图斑变化识别使用的图像')
    parser.add_argument('--file-path-extended', type=str, default='2021_1_1_res_0.5.tif',
                        help='仅变化识别使用的对比图像(过去)')
    parser.add_argument('--shp-path', type=str,
                        default='real_data/裁剪影像01.shp',
                        help='变化识别中参考的矢量文件路径，在detect-change-shp模式下使用')
    parser.add_argument('--output-dir', type=str, default='output/semantic_result',
                        help='输出结果存储路径')
    parser.add_argument('--remove-tif', action='store_true',
                        help='是否要将结果储存为栅格形式')
    parser.add_argument('--mode', type=str, default='segment-semantic',
                        help='程序运行模式，包括detect-change、segment-semantic、detect-change-shp')
    parser.add_argument('--half', action='store_true',
                        help='模型计算时是否开启半精度运算')
    parser.add_argument('--slic', type=int, default=0,
                        help='是否对分割结果进行slic聚类算法后处理, 设置为0则不使用slic算法')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='进行验证时使用的批量大小')
    parser.add_argument('--device', type=str, default='cuda', help='确认训练模型使用的机器')
    parser.add_argument('--log-output', action='store_true', help='在控制台打印程序运行结果')
    parser.add_argument('--change-threshold', type=float, default=0.25,
                        help='最后网格化判断网格是否变化的阈值，1即如果网格中像素比例超过该阈值则判定该网格为变化区域。')
    parser.add_argument('--block-size', type=float, default=25,
                        help='变化区域网格化的大小。')
    opt = parser.parse_args()
    return opt


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
        assert len(image_list) > 0, "输入的文件不存在"
        # 防止windows环境glob获取路径中包含\\
        image_list = [RSPipeline.check_path(path) for path in image_list]
        # TODO 图像预处理,分辨率统一为0.5米
        for path in image_list:
            # 转换为3857坐标系
            transform_geoinfo_with_index(path, 3857)
            ds = gdal.Warp(path.replace(".tif", "_res_0.5.tif"),
                           path.replace(".tif", "_3857.tif"),
                           format="GTiff", xRes=0.5, yRes=0.5)

        # 根据图像和对应的矢量路劲制作模板
        save_path = [path.replace("processed_data", "trad_alg")
                         .replace(".tif", f"_mask.tif") for path in image_list]
        # TODO 获取shp文件的attribute field
        file = read_shp(args.shp_path)
        field = file.columns[0]
        # 将耕地矢量图斑文件转为栅格数据
        for image_path, save in zip(image_list, save_path):
            shp2tif(shp_path=args.shp_path,
                    refer_tif_path=image_path.replace(".tif", "_res_0.5.tif"),
                    target_tif_path=save,
                    attribute_field=field,
                    nodata_value=0)
            # 将相交的耕地图斑部分再重新转换为矢量
            # 将变化结果保存为矢量数据
            raster2vector(save, vector_path=save.replace("tif", "shp"), remove_tif=args.remove_tif)

        RSPipeline.print_log('根据图像和对应的矢量路劲制作模板完成')
        for i, (image_path, mask_path) in enumerate(zip(image_list, save_path)):
            image_path = image_path.replace(".tif", "_res_0.5.tif")
            file_name = os.path.basename(image_path)[:-4]
            #####################
            # 确定最后变化结果保存路径
            #####################
            tif_path = args.output_dir + f'/tif/detect_change_{file_name}.tif'
            result_shp_path = args.output_dir + f'/change_result/detect_change_{file_name}.shp'
            #####################
            # 打开目标图像获取图像的地理位置信息
            image = gdal.Open(image_path)
            # 进行地貌类型识别
            semantic_result = test_big_image(model, image_path, IMAGE_SIZE, vars(args))
            # 打开当前影像的三调耕地图斑模板，将耕地以外的识别结果都删除，方便统计变化区域
            mask = Image.open(mask_path)
            mask = np.array(mask)
            semantic_result[mask == 0] = 0
            # 将变化结果保存为栅格数据
            ARR2TIF(semantic_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
            # 将变化结果保存为矢量数据
            raster2vector(tif_path, vector_path=result_shp_path, remove_tif=args.remove_tif)
            # TODO 根据耕地mask裁剪出对应耕地识别结果。
            # 将检测出的变化区域转换为原始三调图斑，如果三调图斑中一个图斑中有0.1部分的面积被覆盖到，就算这个图斑为变化区域，并存储最终结果。
            RSPipeline.print_log('开始将变化识别结果图斑化')
            joint_polygon(mask_path.replace("tif", "shp"), result_shp_path)
            RSPipeline.print_log('变化识别结果图斑化完成')
        RSPipeline.print_log('变化结果板块提取完毕')
    elif args.mode == 'detect-change':
        RSPipeline.print_log('正在执行依据图像变化检测模块')
        # 输入两年份的图片
        image_src_list = glob(os.path.join(args.target_dir, args.file_path_extended))
        image_target_list = glob(os.path.join(args.target_dir, args.file_path))
        assert len(image_target_list) == len(image_src_list), "变化识别前后比对图片数量不统一"
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
            change_result = DCS.detect_change(semantic_result_src,
                                              semantic_result_target)
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
            raster2vector(out_tif, vector_path=out_shp, remove_tif=args.remove_tif)
            print(f"结果存储地址：{out_shp}")

        RSPipeline.print_log('变化识别结果网格化完毕')
    del model
    gc.collect()
