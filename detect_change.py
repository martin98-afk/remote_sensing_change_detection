# encoding:utf-8
import argparse
import gc
import sys

from utils.detect_change_to_block import *
from utils.polygon_utils import joint_polygon
from utils.segment_semantic import *


def get_parser():
    '''
    收集命令行提供的参数

    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-info-path', type=str, default='./output/ss_eff_b0.yaml',
                        help='需要使用的模型信息文件存储路径')
    parser.add_argument('--target-dir', type=str, default='real_data/',
                        help='需要进行预测的图像存储路径')
    parser.add_argument('--file-path', type=str, default='test7.tif',
                        help='确认训练模型使用的机器')
    parser.add_argument('--output-dir', type=str, default='output/semantic_result',
                        help='输出结果存储路径')
    parser.add_argument('--save-tif', action='store_true',
                        help='是否要将结果储存为栅格形式')
    parser.add_argument('--mode', type=str, default='segment-semantic',
                        help='程序运行模式，包括detect-change、segment-semantic、detect-change-shp')
    parser.add_argument('--half', action='store_true',
                        help='模型计算时是否开启半精度运算')
    parser.add_argument('--slic', type=int, default=0,
                        help='是否对分割结果进行slic聚类算法后处理, 设置为0则不使用slic算法')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='进行验证时使用的批量大小')
    parser.add_argument('--device', type=str, default='cuda:1', help='确认训练模型使用的机器')
    parser.add_argument('--log-output', action='store_true', help='在控制台打印程序运行结果')
    parser.add_argument('--change-threshold', type=float, default=0.35,
                        help='最后网格化判断网格是否变化的阈值，1即如果网格中像素比例超过该阈值则判定该网格为变化区域。')
    parser.add_argument('--block-size', type=float, default=50,
                        help='变化区域网格化的大小。')
    opt = parser.parse_args()
    return opt


def detect_change(image0, image1, include_class=[1], identify_classes=[3, 4, 5, 6, 7, 8]):
    '''
    指定要检测的标签，检测image 1相比image 0中农田区域变化成指定标签地形的区域。

    :param image0: 原图像
    :param image1: 待识别变化的图像
    :param include_class: 待识别变化的地貌类型
    :param identify_classes: 变化识别方向的识别
    :return: 变化识别结果
    '''
    image0_farm = np.zeros_like(image0)
    for i in include_class:
        image0_farm += image0 == i
    image1_identify_classes = [(image1 == clas) for clas in identify_classes]
    change_result = np.zeros((image0.shape[0], image0.shape[1]))
    for i, new_image in enumerate(image1_identify_classes):
        change = np.logical_and(image0_farm, new_image)
        change_result += identify_classes[i] * change
    return change_result


if __name__ == '__main__':
    args = get_parser()

    # 判断是否要将输出打印到控制台
    if args.log_output:
        f_handler = open('./out_detect_change.log', 'w', encoding='utf-8')
        __console__ = sys.stdout
        sys.stdout = f_handler
        print('输出输入到日志中!')

    # 导入模型参数
    data, model = RSPipeline.load_model(args.model_info_path, device=args.device)
    model = model.half() if args.half else model
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
                                    num_classes,
                                    args)
    elif args.mode == 'detect-change-shp':
        RSPipeline.print_log('正在执行依据图斑变化检测模块')
        image_list = glob(os.path.join(args.target_dir, args.file_path))

        for i, image_path in enumerate(image_list):
            place, part = image_path.split('_')[-4], image_path.split('_')[-3]
            tif_path = args.output_dir + f'/tif/detect_change_{place}_{part}.tif'
            shp_path = args.output_dir + f'/change_result/detect_change_{place}_{part}.shp'
            image = gdal.Open(image_path)
            semantic_result = test_big_image(model, image_path,
                                             IMAGE_SIZE, num_classes,
                                             args)
            mask = Image.open(f'real_data/trad_alg/2021_{place}_{part}_res_0.5_耕地_mask.tif')
            mask = np.array(mask)
            semantic_result[mask == 0] = 0
            ARR2TIF(semantic_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
            raster2vector(tif_path, vector_path=shp_path, remove_tif=False)
            # TODO 根据耕地mask裁剪出对应耕地识别结果。
            # 将检测出的变化区域转换为原始三调图斑，如果三调图斑中一个图斑中有0.1部分的面积被覆盖到，就算这个图斑为变化区域，并存储最终结果。
            joint_polygon('real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_耕地.shp',
                          shp_path)


    elif args.mode == 'detect-change':
        RSPipeline.print_log('正在执行依据图像变化检测模块')
        # 输入两年份的图片
        image_2020_list = glob(os.path.join(args.target_dir, args.file_path))
        image_2021_list = [item.replace('2020', '2021') for item in image_2020_list]
        print(image_2021_list)
        # 进行变化识别
        tif_paths = []
        for i, (image_2020, image_2021) in enumerate(zip(image_2020_list, image_2021_list)):
            try:
                place, part = image_2020.split('_')[-4], image_2020.split('_')[-3]
                tif_path = args.output_dir + f'/tif/detect_change_{place}_{part}.tif'
                shp_path = args.output_dir + f'/change_result/detect_change_{place}_{part}.shp'
            except:
                tif_path = args.output_dir + f'/tif/detect_change_{i}.tif'
                shp_path = args.output_dir + f'/change_result/detect_change_{i}.shp'
            semantic_result_2020 = test_big_image(model, image_2020,
                                                  IMAGE_SIZE, num_classes,
                                                  args)
            semantic_result_2021 = test_big_image(model, image_2021,
                                                  IMAGE_SIZE, num_classes,
                                                  args)
            RSPipeline.print_log('两年份遥感数据语义分割已完成')
            image = gdal.Open(image_2020)
            change_result = detect_change(semantic_result_2020, semantic_result_2021)
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

        # # TODO 使用三调数据作为基础,筛选语义分割的结果，只保留耕地图斑的区域。
        # for i, image_2020 in enumerate(image_2020_list):
        #     place, part = image_2020.split('_')[-4], image_2020.split('_')[-3]
        #     mask_path = f'real_data/trad_alg/2021_{place}_{part}_res_0.5_耕地_mask.tif'
        #     tif_path = args.output_dir + f'/tif/detect_change_{place}_{part}.tif'
        #     shp_path = args.output_dir + f'/change_result/detect_change_{place}_{part}.shp'
        #     mask = Image.open(mask_path)
        #     mask = np.array(mask)
        #     image = gdal.Open(tif_path)
        #     band = image.GetRasterBand(1).ReadAsArray()
        #     band[mask == 0] = 0
        #     ARR2TIF(band, image.GetGeoTransform(), image.GetProjection(), tif_path)
        #     raster2vector(tif_path, vector_path=shp_path)
        # 将检测出的变化区域转换为原始三调图斑，如果三调图斑中一个图斑中有0.1部分的面积被覆盖到，就算这个图斑为变化区域，并存储最终结果。
        # joint_polygon('real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_耕地.shp',
        #               'output/semantic_result/change_result/detect_change_1_2.shp')
    del model
    gc.collect()
