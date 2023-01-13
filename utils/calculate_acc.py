import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from glob import glob
from osgeo import gdal
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.pipeline import RSPipeline

gdal.AllRegister()  # 先载入数据驱动，也就是初始化一个对象，让它“知道”某种数据结构，但是只能读，不能写
os.makedirs("../output/out_picture", exist_ok=True)


class ModelMeasurement(object):

    def __init__(self, model_info_path):
        with open(model_info_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        ind2label = data['index to label']
        ind2label['00'] = '其他'
        ind2label = sorted(ind2label.items(), key=lambda x: x[0])
        self.num2label = {i: v for i, (k, v) in enumerate(ind2label)}
        self.num_classes = data["num_classes"]

    # 保存图⽚完整
    def show_acc(self, real_filepath_list, pred_filepath_list):
        real_array, pred_array = self.get_all_array(real_filepath_list, pred_filepath_list)
        result_arr = self.concat_result(real_array, pred_array)
        conf_mat = self.cal_confusion_matrix(real_array, pred_array)
        iou = self.IntersectionOverUnion(conf_mat)
        iou[-1] = iou[-2]
        iou[-2] = 0
        iou = np.array(iou).reshape((-1, 1))
        result_arr = np.hstack([result_arr, iou])
        miou = self.MeanIntersectionOverUnion(conf_mat)
        fwiou = self.Frequency_Weighted_Intersection_over_Union(conf_mat)
        self.Results_visualization(result_arr)
        return iou, miou, fwiou

    def get_all_array(self, real_filepath_list, pred_filepath_list):
        a_real = np.array([], dtype=np.uint8)
        a_pred = np.array([], dtype=np.uint8)
        for i in range(len(real_filepath_list)):
            b_real, b_pred = self.openfile(real_filepath_list[i], pred_filepath_list[i])
            a_real = np.hstack((a_real, b_real))
            a_pred = np.hstack((a_pred, b_pred))
        return a_real, a_pred

    def openfile(self, real_file_path, pred_file_path):
        real_data = gdal.Open(real_file_path)  # 打开文件
        out_data = gdal.Open(pred_file_path)  # 打开文件
        real_data_array = real_data.ReadAsArray()
        array2 = real_data_array[int(0.8 * real_data_array.shape[1]):]
        real_data_array1 = np.ndarray.flatten(array2)
        out_data_array = out_data.ReadAsArray()
        array1 = out_data_array[int(0.8 * real_data_array.shape[1]):]
        out_data_array1 = np.ndarray.flatten(array1)

        return real_data_array1, out_data_array1

    def cal_confusion_matrix(self, array_real, array_out):
        return confusion_matrix(array_real, array_out)

    def IntersectionOverUnion(self, confusionMatrix):
        #  返回交并比IoU
        intersection = np.diag(confusionMatrix)
        union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(
                confusionMatrix)
        IoU = intersection / union
        return IoU

    def MeanIntersectionOverUnion(self, confusionMatrix):
        #  返回平均交并比mIoU
        intersection = np.diag(confusionMatrix)
        union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(
                confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self, confusionMatrix):
        #  返回频权交并比FWIoU
        freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
        iu = np.diag(confusionMatrix) / (
                np.sum(confusionMatrix, axis=1) +
                np.sum(confusionMatrix, axis=0) -
                np.diag(confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def cal_acc_score(self, array_real, array_out, type_id):
        xarr = np.where(array_real == type_id, 1, 0)
        yarr = np.where(array_out == type_id, 1, 0)
        acc = np.sum(xarr == yarr) / len(array_out)
        return acc

    def cal_pr_f1(self, real_array, out_array, id):
        reporter = classification_report(real_array, out_array, labels=[id], output_dict=True)
        dict1 = reporter[str(id)]
        p = dict1['precision']
        r = dict1['recall']
        f1 = dict1['f1-score']
        return p, r, f1

    def concat_result(self, real_array, out_array):
        out_list = []
        for i in range(self.num_classes):
            p, r, f = self.cal_pr_f1(real_array, out_array, i)
            a = self.cal_acc_score(real_array, out_array, i)
            type_list = [i, a, p, r, f]
            out_list.append(type_list)
        return np.array(out_list)

    def Results_visualization(self, arr):
        # 准备数据

        plt.rcParams["font.sans-serif"] = [u"SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        totalWidth = 2.4  # 一组柱状体的宽度
        labelNums = 5  # 一组有两种类别（例如：男生、女生）
        barWidth = totalWidth / labelNums  # 单个柱体的宽度

        plt.bar([3 * x for x in range(self.num_classes)], height=arr[:, 1], label="acc",
                width=barWidth)
        plt.bar([3 * x + barWidth for x in range(self.num_classes)], height=arr[:, 2],
                label="precision",
                width=barWidth)
        plt.bar([3 * x + 2 * barWidth for x in range(self.num_classes)], height=arr[:, 3],
                label="recall",
                width=barWidth)
        plt.bar([3 * x + 3 * barWidth for x in range(self.num_classes)], height=arr[:, 4],
                label="F1-score",
                width=barWidth)
        plt.bar([3 * x + 4 * barWidth for x in range(self.num_classes)], height=arr[:, 5],
                label="iou",
                width=barWidth)
        plt.xticks([3 * x + barWidth / 2 * (labelNums - 1) for x in range(self.num_classes)],
                   [self.num2label[i] for i in range(self.num_classes)])
        plt.xlabel("类别")
        plt.ylabel("数值")
        plt.title("评估参数")
        plt.legend(loc="lower right")
        plt.grid('on')
        plt.savefig("./output/out_picture/acc.png")
        plt.show()
