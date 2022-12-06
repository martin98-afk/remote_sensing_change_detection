import argparse
import sys
from glob import glob

from config import *
from utils.pipeline import RSPipeline

"""
训练模型
"""


def get_parser():
    """
    收集命令行提供的参数

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="确认训练模型使用的机器")
    parser.add_argument("--console-output", type=bool, default=True, help="在控制台打印程序运行结果")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data", type=str, default="./real_data/processed_data", help="数据存储路径")
    parser.add_argument("--label", type=str, default="./real_data/semantic_mask", help="标签存储路径")
    parser.add_argument("--image-size", nargs="+", type=int, default=[512, 512], help="[图片宽， 图片高]")
    parser.add_argument("--fp16", type=bool, default=True, help="是否使用半精度训练，默认为True")
    parser.add_argument("--model-name", type=str, default="efficientnet-b0", help="训练使用的骨干网络模型名称")
    parser.add_argument("--batch-size", type=int, default=20, help="训练一批的图片数量")
    parser.add_argument("--model-save-path", type=str, default="output/ss_eff_b0.pth",
                        help="模型保存路径，同时会在同目录生成一个相同名称的yaml文件保存模型各种参数变量。")
    parser.add_argument("--ohem", type=bool, default=True, help="是否使用在线难例挖掘")
    parser.add_argument("--update-polygon", action="store_true", help="是否更新训练标签")
    parser.add_argument("--ignore-background", type=bool, default=True, help="训练时是否忽视背景类")
    parser.add_argument("--train-size", type=int, default=10000, help="训练划分数据量")
    parser.add_argument("--val-size", type=int, default=1200, help="验证划分数据量")
    parser.add_argument("--num-workers", type=int, default=4, help="数据读取时使用的线程数量")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    ind2label = {
        "01": '种植植被',
        "03": '林草',
        "05": '房屋',
        "06": '铁路与道路',
        "07": '构筑物',
        "08": '人工堆掘地',
        "09": '裸土',
        "10": '水域'
    }
    label2ind = {v: k for k, v in ind2label.items()}
    ind2num = {item: i for i, item in enumerate(ind2label.keys())}
    num2ind = {i: item for i, item in enumerate(ind2label.keys())}
    num_classes = len(ind2label.keys())

    args = get_parser()
    # 判断是否要将输出打印到控制台
    if not args.console_output:
        f_handler = open('./out_train_and_validate.log', 'w', encoding='utf-8')
        __console__ = sys.stdout
        sys.stdout = f_handler
        print('输出输入到日志中!')

    # 更新标签数据
    if args.update_polygon:
        image_paths = glob(args.data + "/2020_2_1_res_*.tif")
        image_paths.extend(
            glob(args.data + "/2020_2_2_res_*.tif")
        )
        for path in image_paths:
            RSPipeline.update_polygon(args, image_path=path,
                                      shp_path="./real_data/移交数据和文档/苏北/0.2米航片对应矢量数据/LCRA_2020_2_merged_ma/"
                                               "LCRA_2020_2_merged.shp",
                                      num_classes=num_classes,
                                      ind2num=ind2num)
        image_paths = glob(args.data + "/2020_2_3_res_*.tif")
        image_paths.extend(
            glob(args.data + "/2020_2_4_res_*.tif")
        )
        for path in image_paths:
            RSPipeline.update_polygon(args, image_path=path,
                                      shp_path="./real_data/移交数据和文档/苏北/0.2米航片对应矢量数据/LCRA_2020_2_merged_fang/"
                                               "LCRA_2020_2_merged.shp",
                                      num_classes=num_classes,
                                      ind2num=ind2num)

    mm = RSPipeline(ind2label, num_classes + 1, args)

    # 打印模型结构
    mm.get_model_summary()
    mm.run(visualize=False)
