import gc
from glob import glob

import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from utils.crf import DenseCRF
from utils.datasets import SSDataRandomCrop
from utils.gdal_utils import write_img
from utils.pipeline import RSPipeline
from utils.polygon_utils import raster2vector

"""
根据训练好的模型对指定大幅遥感图像进行切割和预测。
"""

os.makedirs("./output/semantic_result/shp", exist_ok=True)
os.makedirs("./output/semantic_result/tif", exist_ok=True)
np.random.seed(0)
PAD_SIZE = 64  # 预测图像之间的重叠距离

post_processor = DenseCRF(
    iter_max=10,  # 10
    pos_xy_std=3,  # 3
    pos_w=3,  # 3
    bi_xy_std=140,  # 121, 140
    bi_rgb_std=5,  # 5, 5
    bi_w=5,  # 4, 5
)


def predict(model, image, ori_image, num_classes, device, threashold=0.2):
    image = image.to(device).float()
    with torch.no_grad():
        output = torch.nn.Softmax(dim=1)(model(image))
        output = output.detach().cpu().numpy()
    ori_image = ori_image.numpy().astype(np.uint8)
    # 使用dense crf进行后处理
    for i in range(output.shape[0]):
        output[i, ...] = post_processor(ori_image[i, ...], output[i, ...])
    # 根据阈值进行筛选，判断概率小于阈值的定为背景类。
    output_max = np.max(output, axis=1)
    output_result = np.argmax(output, axis=1)
    output_result[output_max < threashold] = num_classes - 1
    return output_result


def visualize_result(image1, predict_mask, crf_mask):
    """
    展示dense crf使用的效果。

    :param image1:
    :param predict_mask:
    :param crf_mask:
    :return:
    """
    image1 = np.transpose(image1, (0, 2, 3, 1))
    fg, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[2].imshow(crf_mask)
    ax[2].set_title('crf mask')
    ax[1].imshow(predict_mask)
    ax[1].set_title('predict mask')
    ax[0].imshow((image1[0, ...] * 255).astype('uint8'))
    ax[0].set_title('image')
    plt.show()


def get_dataloader(TifArray, batch_size, shuffle=False, num_workers=2):
    """
    根据遥感图像小块构造测试数据集。

    :param TifArray:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :return:
    """
    dataset = SSDataRandomCrop(image_list=TifArray, mask_list=None, mode="test",
                               length=len(TifArray))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=False)
    return dataloader


def TifCroppingArray(img, SideLength, IMAGE_SIZE):
    """
    裁剪大张遥感图像为一系列小图像

    :param img:
    :param SideLength: 裁剪边长
    :return:
    """
    TifArrayReturn = []
    ColumnNum = int((img.shape[0] - SideLength * 2) / (IMAGE_SIZE - SideLength * 2))
    RowNum = int((img.shape[1] - SideLength * 2) / (IMAGE_SIZE - SideLength * 2))

    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (IMAGE_SIZE - SideLength * 2): i * (
                    IMAGE_SIZE - SideLength * 2) + IMAGE_SIZE,
                      j * (IMAGE_SIZE - SideLength * 2): j * (
                              IMAGE_SIZE - SideLength * 2) + IMAGE_SIZE
                      ]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    # 向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[
                  i * (IMAGE_SIZE - SideLength * 2): i * (IMAGE_SIZE - SideLength * 2) + IMAGE_SIZE,
                  (img.shape[1] - IMAGE_SIZE):img.shape[1]]
        TifArrayReturn[i].append(cropped)
    # 向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[
                  (img.shape[0] - IMAGE_SIZE):img.shape[0],
                  j * (IMAGE_SIZE - SideLength * 2): j * (IMAGE_SIZE - SideLength * 2) + IMAGE_SIZE
                  ]
        TifArray.append(cropped)
    # 向前裁剪右下角
    cropped = img[(img.shape[0] - IMAGE_SIZE):img.shape[0], (img.shape[1] -
                                                             IMAGE_SIZE):img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (IMAGE_SIZE - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (IMAGE_SIZE - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


#  获得结果矩阵
def concat_result(shape, TifArrayShape, npyfile, RepetitiveLength, RowOver, ColumnOver, IMAGE_SIZE):
    """
    根据小块预测后的结果进行拼接，还原为原始遥感图像大小。

    :param shape:
    :param TifArrayShape:
    :param npyfile:
    :param RepetitiveLength:
    :param RowOver:
    :param ColumnOver:
    :return:
    """
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if i % TifArrayShape[1] == 0:
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if j == 0:
                result[0: IMAGE_SIZE - RepetitiveLength, 0: IMAGE_SIZE - RepetitiveLength] = img[
                                                                                             0: IMAGE_SIZE - RepetitiveLength,
                                                                                             0: IMAGE_SIZE - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif j == TifArrayShape[0] - 1:
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0],
                0: IMAGE_SIZE - RepetitiveLength] = img[
                                                    IMAGE_SIZE - ColumnOver - RepetitiveLength: IMAGE_SIZE,
                                                    0: IMAGE_SIZE - RepetitiveLength]
            else:
                result[j * (IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength,
                0:IMAGE_SIZE - RepetitiveLength] = img[
                                                   RepetitiveLength: IMAGE_SIZE - RepetitiveLength,
                                                   0: IMAGE_SIZE - RepetitiveLength]
                #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif i % TifArrayShape[1] == TifArrayShape[1] - 1:
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if j == 0:
                result[0: IMAGE_SIZE - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[
                                                                                         0: IMAGE_SIZE - RepetitiveLength,
                                                                                         IMAGE_SIZE - RowOver: IMAGE_SIZE]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif j == TifArrayShape[0] - 1:
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[
                                                                                        IMAGE_SIZE - ColumnOver: IMAGE_SIZE,
                                                                                        IMAGE_SIZE - RowOver: IMAGE_SIZE]
            else:
                result[j * (IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength,
                shape[1] - RowOver: shape[1]] = img[RepetitiveLength: IMAGE_SIZE - RepetitiveLength,
                                                IMAGE_SIZE - RowOver: IMAGE_SIZE]
                #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if j == 0:
                result[0: IMAGE_SIZE - RepetitiveLength,
                (i - j * TifArrayShape[1]) * (
                        IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength: (
                                                                                       i - j *
                                                                                       TifArrayShape[
                                                                                           1] + 1) * (
                                                                                       IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[0: IMAGE_SIZE - RepetitiveLength,
                    RepetitiveLength: IMAGE_SIZE - RepetitiveLength]
                #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if j == TifArrayShape[0] - 1:
                result[shape[0] - ColumnOver: shape[0],
                (i - j * TifArrayShape[1]) * (
                        IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength: (
                                                                                       i - j *
                                                                                       TifArrayShape[
                                                                                           1] + 1) * (
                                                                                       IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[IMAGE_SIZE - ColumnOver: IMAGE_SIZE,
                    RepetitiveLength: IMAGE_SIZE - RepetitiveLength]
            else:
                result[j * (IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * TifArrayShape[1]) * (
                        IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength: (
                                                                                       i - j *
                                                                                       TifArrayShape[
                                                                                           1] + 1) * (
                                                                                       IMAGE_SIZE - 2 * RepetitiveLength) + RepetitiveLength,
                ] = img[RepetitiveLength: IMAGE_SIZE - RepetitiveLength,
                    RepetitiveLength: IMAGE_SIZE - RepetitiveLength]
    return result


def test_big_image(model, image1_path, IMAGE_SIZE,
                   num_classes, device, batch_size=24,
                   cut_length=64):
    """


    :param model:
    :param image1_path:
    :param cut_length:
    :return:
    """
    # 历史图像1
    raw_image = Image.open(image1_path)
    raw_image = np.asarray(raw_image)[:, :, :3]
    RSPipeline.print_log("开始分割原始大遥感影像")
    TifArray, RowOver, ColumnOver = TifCroppingArray(raw_image, cut_length, IMAGE_SIZE)
    TifArray = np.array(TifArray)
    TifArray_shape = TifArray.shape
    TifArray = TifArray.reshape((-1, TifArray.shape[2], TifArray.shape[3], TifArray.shape[4]))
    print("遥感图像分割后大小", TifArray.shape)
    RSPipeline.print_log("根据划分图像构造数据集")
    predicts = None
    test_loader = get_dataloader(TifArray, batch_size=batch_size)
    RSPipeline.print_log("代入模型获得每小块验证结果")
    for ori_image, image in tqdm(test_loader):

        output = predict(model, image, ori_image, num_classes, device)
        output = np.uint8(output)
        if predicts is None:
            predicts = output
        else:
            predicts = np.vstack([predicts, output])

    RSPipeline.print_log("预测完毕，拼接结果中")
    # 保存结果
    result_shape = (raw_image.shape[0], raw_image.shape[1])
    result_data = concat_result(result_shape, TifArray_shape, predicts, cut_length, RowOver,
                                ColumnOver, IMAGE_SIZE)
    return result_data


def test_semantic_segment_files(model, file_path='./real_data/processed_data/2020_2_*_res_0.5.tif'):
    """


    :param model:
    :param file_path:
    :return:
    """
    image_2020_list = glob(file_path)
    loc_list = [path.split('/')[-1].split('_')[:5] for path in image_2020_list]
    for i, image_2020 in enumerate(image_2020_list):
        semantic_result = test_big_image(model, image_2020, IMAGE_SIZE, num_classes, device)
        RSPipeline.print_log("语义分割已完成")
        image = gdal.Open(image_2020)

        tif_path = f"./output/semantic_result/tif/{loc_list[i][0]}_" \
                   f"{loc_list[i][1]}_{loc_list[i][2]}_res_{loc_list[i][4]}_semantic_result.tif"
        shp_path = f"./output/semantic_result/shp/{loc_list[i][0]}_" \
                   f"{loc_list[i][1]}_{loc_list[i][2]}_res_{loc_list[i][4]}_semantic_result.shp"
        write_img(tif_path,
                  image.GetProjection(),
                  image.GetGeoTransform(),
                  semantic_result.reshape((1, semantic_result.shape[0], semantic_result.shape[1])))
        RSPipeline.print_log("分割结果栅格数据保存已完成")
        raster2vector(raster_path=tif_path, vector_path=shp_path)
        RSPipeline.print_log("分割结果矢量数据保存已完成")


def test_semantic_single_file(model, file_path):
    """


    :param model:
    :param file_path:
    :return:
    """
    semantic_result = test_big_image(model, file_path, IMAGE_SIZE, num_classes, device)
    RSPipeline.print_log("语义分割已完成")
    tif_path = f"./output/semantic_result/tif/test_semantic_result.tif"
    shp_path = f"./output/semantic_result/shp/test_semantic_result.shp"
    image = gdal.Open(file_path)
    write_img(tif_path,
              image.GetProjection(),
              image.GetGeoTransform(),
              semantic_result.reshape((1, semantic_result.shape[0], semantic_result.shape[1])))
    RSPipeline.print_log("分割结果栅格数据保存已完成")
    raster2vector(raster_path=tif_path, vector_path=shp_path)
    RSPipeline.print_log("分割结果矢量数据保存已完成")

if __name__ == "__main__":
    device = "cuda"
    data, model = RSPipeline.load_model("./output/ss_eff_b0.yaml", device=device)
    IMAGE_SIZE = data['image_size'][0]
    num_classes = data['num_classes']
    # 读取指定路径下的单个文件进行语义分割
    # test_semantic_single_file(model, "./test4.tif")
    # 读取glob格式的所有文件进行语义分割
    test_semantic_segment_files(model,
                                file_path='./real_data/processed_data/2020_2_*_res_0.5.tif')
    del model
    gc.collect()
