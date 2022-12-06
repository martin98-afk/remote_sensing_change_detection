from utils.calculate_acc import *

"""
   根据segment semantic.py 中跑出的结果统计准确率，
   包括acc，precision，recall，f1 score, iou, miou, fwiou
"""

if __name__ == "__main__":
    RSPipeline.print_log("读取对应模型训练的标签信息")
    info_path = "./output/ss_eff_b0.yaml"

    real_filelist_path = glob("./real_data/semantic_mask/*0.5.png")
    pred_filelist_path = [item.replace(".png", "_semantic_result.tif")
                              .replace("real_data/semantic_mask", "output/semantic_result/tif")
                          for item in real_filelist_path]
    model_measure = ModelMeasurement(info_path)
    RSPipeline.print_log("统计准确率中")
    iou, miou, fwiou = model_measure.show_acc(real_filelist_path, pred_filelist_path)
    RSPipeline.print_log("统计完毕，打印准确率结果")
    print("各类iou值为：", iou)
    print("平均iou值为：", miou)
    print("加权fwiou值为：", fwiou)
