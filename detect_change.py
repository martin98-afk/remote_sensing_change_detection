import argparse
import gc
import sys

from utils.segment_semantic import *
from utils.detect_change_to_block import *


def get_parser():
    """
    收集命令行提供的参数

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-info-path", type=str, default="./output/ss_eff_b0_new.yaml",
                        help="需要使用的模型信息文件存储路径")
    parser.add_argument("--target-dir", type=str, default="real_data/processed_data",
                        help="需要进行预测的图像存储路径")
    parser.add_argument("--file_path", type=str, default="2020_1_*_res_0.5.tif",
                        help="确认训练模型使用的机器")
    parser.add_argument("--output-dir", type=str, default="./output/semantic_result",
                        help="输出结果存储路径")
    parser.add_argument("--mode", type=str, default="detect-change",
                        help="程序运行模式，包括detect-change和segment-semantic")
    parser.add_argument("--device", type=str, default="cuda", help="确认训练模型使用的机器")
    parser.add_argument("--log-output", action='store_true', help="在控制台打印程序运行结果")
    parser.add_argument("--change-threshold", type=float, default=0.35,
                        help="最后网格化判断网格是否变化的阈值，1即如果网格中像素比例超过该阈值则判定该网格为变化区域。")
    parser.add_argument("--block-size", type=float, default=50,
                        help="变化区域网格化的大小。")
    opt = parser.parse_args()
    return opt


def detect_change(image0, image1, include_class=[1, 2], identify_classes=[3, 4, 5, 6, 7, 8]):
    """
    指定要检测的标签，检测image 1相比image 0中农田区域变化成指定标签地形的区域。

    :param image0:
    :param image1:
    :param identify_classes:
    :return:
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


if __name__ == "__main__":
    args = get_parser()

    # 判断是否要将输出打印到控制台
    if args.log_output:
        f_handler = open('./out_detect_change.log', 'w', encoding='utf-8')
        __console__ = sys.stdout
        sys.stdout = f_handler
        print('输出输入到日志中!')

    # 导入模型参数
    data, model = RSPipeline.load_model(args.model_info_path, device=args.device)
    model.half()
    IMAGE_SIZE = data['image_size'][0]
    num_classes = data['num_classes']
    ind2label = data['index to label']
    # 建立结果存放的文件夹
    os.makedirs(args.output_dir + "/change_result", exist_ok=True)
    os.makedirs(args.output_dir + "/tif", exist_ok=True)
    os.makedirs(args.output_dir + "/shp", exist_ok=True)

    assert args.mode in ["segment-semantic", "detect-change"], \
        "Wrong process mode. Available options are: ['segment-semantic', 'detect-change']"
    if args.mode == "segment-semantic":
        if "*" in args.file_path:
            test_semantic_segment_files(model,
                                        ind2label,
                                        args.target_dir + "/" + args.file_path,
                                        IMAGE_SIZE,
                                        num_classes,
                                        args.device)
        else:
            test_semantic_single_file(model,
                                      ind2label,
                                      args.target_dir + "/" + args.file_path,
                                      IMAGE_SIZE,
                                      num_classes,
                                      args.device)
    elif args.mode == "detect-change":
        # 输入两年份的图片
        image_2020_list = glob(os.path.join(args.target_dir, args.file_path))
        image_2021_list = [item.replace("2020", "2021") for item in image_2020_list]
        print(image_2021_list)
        # 进行变化识别
        tif_paths = []
        for i, (image_2020, image_2021) in enumerate(zip(image_2020_list, image_2021_list)):
            place, part = image_2020.split("_")[-4], image_2020.split("_")[-3]
            tif_path = args.output_dir + f"/tif/detect_change_{place}_{part}.tif"
            shp_path = args.output_dir + f"/change_result/detect_change_{place}_{part}.shp"
            semantic_result_2020 = test_big_image(model, image_2020,
                                                  IMAGE_SIZE, num_classes,
                                                  args.device,
                                                  batch_size=4)
            semantic_result_2021 = test_big_image(model, image_2021,
                                                  IMAGE_SIZE, num_classes,
                                                  args.device,
                                                  batch_size=4)
            RSPipeline.print_log("两年份遥感数据语义分割已完成")
            image = gdal.Open(image_2020)
            change_result = detect_change(semantic_result_2020, semantic_result_2021)
            write_img(tif_path,
                      image.GetProjection(),
                      image.GetGeoTransform(),
                      change_result.reshape((1, change_result.shape[0], change_result.shape[1])))
            raster2vector(raster_path=tif_path, vector_path=shp_path)
            RSPipeline.print_log("变化识别完成已保存结果")
            tif_paths.append(tif_path)

        # 将变化区域转变为50*50的方块，以方便使用和统计准确率
        threshold = args.change_threshold  # 判定阈值
        block_size = args.block_size
        out_tif_filepath = [item.replace("/tif/", "/change_result/")
                                .replace("detect_change", "detect_change_block") for item in
                            tif_paths]

        out_shp_filepath = [item.replace(".tif", ".shp") for item in out_tif_filepath]
        for i, (in_file, out_tif, out_shp) in enumerate(zip(tif_paths, out_tif_filepath,
                                                            out_shp_filepath)):
            change_data, num_w, num_h, rm_w, rm_h, origin_proj = load_tif(in_file, block_size)
            arrayout = change_detect(num_w, num_h, rm_w, rm_h, change_data, threshold, block_size)
            ARR2TIF(arrayout, change_data, origin_proj, out_tif)
            raster2vector(raster_path=out_tif, vector_path=out_shp)

    del model
    gc.collect()
