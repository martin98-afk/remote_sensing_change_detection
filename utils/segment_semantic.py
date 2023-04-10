import gc
from glob import glob

from PIL import Image
from osgeo import gdal
from skimage.segmentation import mark_boundaries, slic
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from utils.conn_mysql import *
from utils.crf import DenseCRF
from utils.datasets import SSDataRandomCrop
from utils.detect_change_to_block import ARR2TIF
from utils.pipeline import RSPipeline
from utils.polygon_utils import raster2vector
from utils.transfer_style import style_transfer

"""
根据训练好的模型对指定大幅遥感图像进行切割和预测。膨胀预测
"""

os.makedirs("./output/semantic_result/shp", exist_ok=True)
os.makedirs("./output/semantic_result/tif", exist_ok=True)
np.random.seed(0)
PAD_SIZE = 64  # 预测图像之间的重叠距离

post_processor = DenseCRF(
        iter_max=10,  # 10
        pos_xy_std=3,  # 3
        pos_w=3,  # 3
        bi_xy_std=(80, 80),  # 121, 140
        bi_rgb_std=13,  # 5, 5
        bi_w=10,  # 4, 5
)


def slic_segment(image, num_segments=700, mask=None, visualize=True):
    """
    使用slic算法对图像进行超像素划分。

    :param image:
    :param num_segments:
    :param visualize:
    :return:
    """
    segments = slic(image, n_segments=num_segments,
                    sigma=5, mask=mask)
    if visualize:
        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % (num_segments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments, color=(255, 0, 0)))
        plt.axis("off")

        # show the plots
        plt.show()
    return segments


def predict(model, image, ori_image, args, threashold=0.9):
    """
    针对不同的模型提供不同输入数据的途径，现支持onnx模型以及pytorch模型。

    :param model:
    :param image:
    :param ori_image:
    :param args:
    :param threashold:
    :return:
    """
    if args['model_type'] == "pytorch":
        image = image.to(args['device'])
        image = image.half() if args['half'] else image
        with torch.no_grad():
            output = torch.nn.Softmax(dim=1)(model.predict(image))
            output = output.detach().cpu().numpy()

    ori_image = ori_image.numpy().astype(np.uint8)
    # 使用dense crf进行后处理
    for i in range(output.shape[0]):
        output[i, ...] = post_processor(ori_image[i, ...], output[i, ...])
    # 根据阈值进行筛选，判断概率小于阈值的定为背景类。
    output_max = np.max(output, axis=1)
    output_result = np.argmax(output, axis=1)
    output_result[output_max < threashold] = 0
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


def get_dataloader(TifArray, batch_size, shuffle=False, num_workers=0):
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


@torch.no_grad()
def test_big_image(model, image_path, IMAGE_SIZE, args, denominator=1, addon=0):
    """
    对大型遥感图像进行切割以后，逐片进行语义分割，最后可以选择是否再进行超像素分割对语义分割结果进行后处理。

    :param model:
    :param image_path:
    :param IMAGE_SIZE:
    :param args:
    :param denominator:
    :param addon:
    :return:
    """
    # 如果是变化识别，目标图像会传入两个图像的地址用于进行风格变换
    if type(image_path) == str:
        # 历史图像
        raw_image = Image.open(image_path)
        raw_image = np.asarray(raw_image)[:, :, :3]
    else:
        raw_image1 = Image.open(image_path[0])
        raw_image1 = np.asarray(raw_image1)[:, :, :3]
        raw_image2 = Image.open(image_path[1])
        raw_image2 = np.asarray(raw_image2)[:, :, :3]
        raw_image = style_transfer(raw_image2, raw_image1)

    RSPipeline.print_log("开始分割原始大遥感影像")
    TifArray, RowOver, ColumnOver = TifCroppingArray(raw_image, 64, IMAGE_SIZE)
    TifArray = np.array(TifArray)
    TifArray_shape = TifArray.shape
    TifArray = TifArray.reshape((-1, TifArray.shape[2], TifArray.shape[3], TifArray.shape[4]))
    print("遥感图像分割后大小", TifArray.shape)
    RSPipeline.print_log("根据划分图像构造数据集")
    predicts = None
    # 将待检测地貌类型的图像加载成dataloader形式方便数据读取
    test_loader = get_dataloader(TifArray,
                                 batch_size=args['batch_size'],
                                 num_workers=args['num_workers'])
    # 连接mysql数据库，更新数据处理进度
    mysql_conn = MysqlConnectionTools(**MYSQL_CONFIG)
    RSPipeline.print_log("代入模型获得每小块验证结果")
    for i, (ori_image, image) in enumerate(tqdm(test_loader)):
        # TODO 获取当前进度并写入数据库之中
        output = predict(model, image, ori_image, args)
        output = np.uint8(output)
        if predicts is None:
            predicts = output
        else:
            predicts = np.vstack([predicts, output])
        progress = int(100 * i / len(test_loader) / denominator) + addon
        if "id" in args.keys():
            mysql_conn.write_to_mysql_progress(args["id"], str(progress) + "%")
    RSPipeline.print_log("预测完毕，拼接结果中")
    # 保存结果
    result_shape = (raw_image.shape[0], raw_image.shape[1])
    result_data = concat_result(result_shape, TifArray_shape, predicts, 64, RowOver,
                                ColumnOver, IMAGE_SIZE)
    # 回收内存，避免内存泄露
    del test_loader
    gc.collect()

    return result_data


def test_semantic_segment_files(model,
                                ind2label,
                                img_size,
                                args):
    """
    对多个遥感图像文件进行语义分割

    :param model: 用来语义分割的模型。
    :param ind2label: 数字标号转中文标号的字典。
    :param img_size: 图像大小。
    :param args: 参数集合
    :return:
    """
    output_dir = args.output_dir  # 输出文件夹
    file_path = args.target_dir + "/" + args.file_path  # 使用的文件路径
    assert args.device == 'cpu' or 'cuda' in args.device, '所使用的机器类型必须为cpu或者cuda'
    image_list = [RSPipeline.check_path(path) for path in glob(file_path)] \
        if "*" in file_path else [file_path]
    file_list = [path[:-4].split('/')[-1] for path in image_list]
    for i, image in enumerate(image_list):
        semantic_result = test_big_image(model, image,
                                         img_size, vars(args))
        RSPipeline.print_log("语义分割已完成")
        image = gdal.Open(image)

        tif_path = f"{output_dir}/tif/{file_list[i]}_semantic_result.tif"
        shp_path = f"{output_dir}/shp/{file_list[i]}_semantic_result.shp"
        ARR2TIF(semantic_result, image.GetGeoTransform(), image.GetProjection(), tif_path)
        RSPipeline.print_log("分割结果栅格数据保存已完成")
        raster2vector(tif_path, vector_path=shp_path, label=ind2label)
        RSPipeline.print_log("分割结果矢量数据保存已完成")
    return shp_path
