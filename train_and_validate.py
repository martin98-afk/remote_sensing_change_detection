# encoding:utf-8
import argparse
import sys
from glob import glob


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
    parser.add_argument("--device", type=str, default="cpu",
                        help="确认训练模型使用的机器")
    parser.add_argument("--log-output", action="store_true",
                        help="在控制台打印程序运行结果")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--label", type=str, default="./real_data/semantic_mask/*",
                        help="标签存储路径")
    parser.add_argument("--update-polygon", type=bool, default=False,
                        help="是否更新训练标签")
    parser.add_argument("--model-save-path", type=str, default="output/ss_eff_b0_with_glcm.pth",
                        help="模型保存路径，同时会在同目录生成一个相同名称的yaml文件保存模型各种参数变量。")
    parser.add_argument("--pretrained-model-path", type=str, default="output/ss_eff_b0_with_glcm.pth",
                        help="模型保存路径，同时会在同目录生成一个相同名称的yaml文件保存模型各种参数变量。")
    parser.add_argument("--model-name", type=str, default="efficientnet-b0",
                        help="训练使用的骨干网络模型名称")
    parser.add_argument("--pop-head", type=bool, default=False,
                        help="是否需要將模型的分類頭刪除")
    parser.add_argument("--image-size", nargs="+", type=int, default=[512, 512],
                        help="[图片宽， 图片高]")
    parser.add_argument("--fp16", type=bool, default=True,
                        help="是否使用半精度训练，默认为True")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="训练一批的图片数量")
    parser.add_argument("--ohem", type=bool, default=True,
                        help="是否使用在线难例挖掘")
    parser.add_argument("--ignore-background", type=bool, default=True,
                        help="训练时是否忽视背景类")
    parser.add_argument("--train-size", type=int, default=2000,
                        help="训练划分数据量")
    parser.add_argument("--val-size", type=int, default=300,
                        help="验证划分数据量")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="数据读取时使用的线程数量")
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
    ind2num = {
        "01": 1,
        "03": 2,
        "05": 3,
        "06": 4,
        "07": 5,
        "08": 6,
        "09": 7,
        "10": 8,
    }

    label2ind = {v: k for k, v in ind2label.items()}
    num2ind = {i: item for i, item in enumerate(ind2label.keys())}
    num_classes = len(set(ind2num.values()))

    args = get_parser()
    # 判断是否要将输出打印到控制台
    if args.log_output:
        f_handler = open('./out_train_and_validate.log', 'w', encoding='utf-8')
        __console__ = sys.stdout
        sys.stdout = f_handler
        print('输出输入到日志中!')

    # 更新标签数据，需要指定遥感影像以及对应的矢量文件
    # TODO 待更新添加新的训练数据的方式，可以设定一个prepared_data文件夹，如果文件夹中有数据则进行更新，更新完毕清空相应文件夹
    if args.update_polygon:
        # 从prepared_data中获取所有待更新文件, 注意shp文件需要和tif文件名相同
        shp_paths = glob("real_data/prepared_data/*.shp")
        tif_paths = [path.replace(".shp", ".tif") for path in shp_paths]

        for (shp_path, tif_path) in zip(shp_paths, tif_paths):
            RSPipeline.update_polygon(args,
                                      image_path=tif_path,
                                      shp_path=shp_path,
                                      num_classes=num_classes,
                                      ind2num=ind2num)
    # 构建遥感地貌识别模型训练pipeline
    mm = RSPipeline(ind2label, num_classes + 1, 9, args)

    # # 打印模型结构
    # mm.get_model_summary()
    # 开始进行模型训练
    mm.run(visualize=False)
