# encoding:utf-8
import time

import seaborn as sns
from osgeo import gdal
from skimage.segmentation import mark_boundaries, slic

from config import *
from remotesensing_alg.fast_glcm import *
from utils.gdal_utils import write_img

warnings.filterwarnings('ignore')

# TODO 根据三调矢量数据进行耕地、果园林地、建设用地等等区域进行提取
# 目前已完成。
# 从大类标注文件中提取耕地图斑
from glob import glob
import pandas as pd
import numpy as np

from PIL import Image
from utils.polygon_utils import read_shp, shp2tif

# 对应地理国情的编号
dict = {"耕地": [1], "建设用地": [3, 4, 5, 8], "林地园地草地": [2]}
class_id_list = np.arange(9)

# TODO 添加从原有一级类和二级类的标签中提取一级类，并保存
shp_file = read_shp("real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1.shp")
shp_file["DLBM"] = shp_file["DLBM"].apply(lambda x: int(x[:2]))
shp_file.to_file("real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_major_class.shp")

res = "0.5"

def extract_target(class_id, name):
    """
    根据类别id提取对应三调中标记的地区并存储为栅格图像。

    :param class_id: 类别标签
    :param name: 类别名称
    :return:
    """
    # 根据保存的一级类筛选对应的地貌类型转换为掩膜
    shp_file_ori = read_shp("real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_major_class.shp")
    shp_file = shp_file_ori[shp_file_ori.iloc[:, 0] == class_id[0]]
    if len(class_id) > 1:
        for i in class_id[1:]:
            shp_file = shp_file.append(shp_file_ori[shp_file_ori.iloc[:, 0] == i])
    shp_file.to_file(f"real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_{name}.shp")
    refer_image_path = glob(f"real_data/processed_data/2021_1_*_res_{res}.tif")
    save_path = [path.replace("processed_data", "trad_alg")
                     .replace(".tif", f"_{name}_mask.tif") for path in refer_image_path]
    # 将耕地矢量图斑文件转为栅格数据
    for image_path, save in zip(refer_image_path, save_path):
        shp2tif(shp_path=f"real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_{name}.shp",
                refer_tif_path=image_path,
                target_tif_path=save,
                attribute_field="DLBM",
                nodata_value=0)

    # 使用栅格数据作为mask从2020年和2021年的图像之中提取对应图斑部分
    refer_image_path = glob(f"real_data/processed_data/202*_1_*_res_{res}.tif")
    refer_image_path = [path.replace('\\', '/') for path in refer_image_path]
    mask_path = [path.replace("processed_data", "trad_alg")
                     .replace(".tif", f"_{name}_mask.tif")
                     .replace("2020", "2021") for path in refer_image_path]
    save_path = [path.replace("processed_data", "trad_alg")
                     .replace(".tif", f"_{name}.tif") for path in refer_image_path]
    # 语义分割结果读取，用来对耕地图斑中的异常区域进行剔除
    ss_result_path = [path.replace("real_data", "output")
                          .replace("processed_data", "semantic_result/tif")
                          .replace(".tif", f"_semantic_result.tif") for path in refer_image_path]
    for path1, path2, path3, path4 in zip(refer_image_path, mask_path, save_path, ss_result_path):
        image = gdal.Open(path1)  # 待提取图片
        mask = Image.open(path2)  # 三调地貌区域蒙版
        mask = np.array(mask)

        # 添加第四个维度用来区分背景和前景，目前只在qgis中有用，在算法中并没有发挥到判别前景还有背景的作用。
        ones = 255 * np.ones_like(mask)
        ones[mask == 0] = 0
        # gdal读取图片后 channel在第一位
        image_array = np.transpose(image.ReadAsArray(), (1, 2, 0))
        # 对蒙版进行repeat从而使得 在判断条件以后可以直接对图像区域进行筛选。
        image_array[np.repeat(mask[..., np.newaxis], 3, 2) == 0] = 0  # 直接将不需要部分取0
        # TODO 使用语义分割结果对栅格图像进行异常值剔除
        if "2021" in path1:
            ss_result = Image.open(path4)  # 根据语义分割对三调提取后的区域进行异常区域剔除
            ss_result = np.array(ss_result)
            ss_result[mask == 0] = 0
            for id in class_id_list:
                if id not in dict[name]:
                    image_array[ss_result == id] = 0
                    ones[ss_result == id] = 0
        # 添加第四个维度用来区分前景和背景值
        image_array = np.concatenate([image_array, ones.reshape((ones.shape[0], ones.shape[1], 1))],
                                     axis=-1)
        write_img(path3, image.GetProjection(),
                  image.GetGeoTransform(),
                  np.transpose(image_array, (2, 0, 1)))


# # 对应三调编号
# # 提取耕地
# extract_target([1], "耕地")
#
# # 提取林地
# extract_target([2, 3, 4], "林地园地草地")
#
# # 提取建设用地
# extract_target([5, 6, 7, 8], "建设用地")


# TODO 对提取出的栅格数据进行SLIC超像素分割
# loop over the number of segments
# apply SLIC and extract (approximately) the supplied number of segments
def slic_segment(image, num_segments=700, mask=None, visualize=True):
    """
    使用slic算法对图像进行超像素划分。

    :param image:
    :param num_segments:
    :param visualize:
    :return:
    """
    segments = slic(image, n_segments=num_segments, sigma=1, mask=mask)
    if visualize:
        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % num_segments)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments, color=(255, 0, 0)))
        plt.axis("off")

        # show the plots
        plt.show()
    return segments


# TODO 现在提取的信息特征均为图斑像素光谱、纹理特征的均值，后续可以考虑加入更复杂的特征来提高模型准确率。同时使用pca对特征进行筛选。
def feature_extractor(image, gray_image, image_segments, label):
    """
    对slic算法聚类出的图斑的光谱、纹理信息进行提取，同时标注类别标签，为后面的分类提供数据。

    :param image:
    :param label:
    :return:
    """
    glcm = fast_glcm(gray_image)
    homo = fast_glcm_homogeneity(gray_image, glcm)[..., np.newaxis]
    contrast = fast_glcm_contrast(gray_image, glcm)[..., np.newaxis]
    entropy = fast_glcm_entropy(gray_image, glcm)[..., np.newaxis]
    ASM = fast_glcm_ASM(gray_image, glcm)[0][..., np.newaxis]
    dissimilar = fast_glcm_dissimilarity(gray_image, glcm)[..., np.newaxis]

    matrix = np.concatenate([image[..., :3], homo, contrast, entropy, ASM, dissimilar], axis=-1)
    # 提取每个超像素的光谱、纹理信息，并将结果存储
    columns = ['r_mean', 'g_mean', 'b_mean', 'contrast',
               'homo', 'ASM', 'entropy', 'dissimilar',
               'r_var', 'g_var', 'b_var', 'contrast_var',
               'homo_var', 'ASM_var', 'entropy_var', 'dissimilar_var',
               'r_max', 'g_max', 'b_max', 'contrast_max',
               'homo_max', 'ASM_max', 'entropy_max', 'dissimilar_max',
               'label']
    result_df = np.zeros((np.max(image_segments), len(columns)))
    for i in range(1, np.max(image_segments) + 1):
        index = image_segments == i
        target = matrix[index, :]
        target_feature_mean = np.mean(target, axis=0)
        target_feature_var = np.var(target, axis=0)
        target_feature_max = np.max(target, axis=0)
        result_df[i - 1, :matrix.shape[-1]] = target_feature_mean
        result_df[i - 1, matrix.shape[-1]:2 * matrix.shape[-1]] = target_feature_var
        result_df[i - 1, 2 * matrix.shape[-1]:-1] = target_feature_max
        result_df[i - 1, -1] = label

    return pd.DataFrame(result_df, columns=columns)


# TODO 根据聚类后的图斑，对每个图斑进行特征提取
# 目前先对耕地和建设用地进行分类，从三调矢量文件中先筛选出对应耕地和建设用地的栅格数据，再使用slic进行图斑聚类划分，最后对每个小图斑进行特征提取，用于分类。
print("正在进行图斑特征提取.")
num_segments = 300
#
# feature_df = None
# for i in range(1, 5):
#     t = time.time()
#     image = Image.open(f"real_data/trad_alg/2021_1_{i}_res_0.5_耕地.tif")
#     gray_image = np.array(image.convert("L"))
#     image = np.array(image)
#     image_segments = slic_segment(image[..., :3], mask=image[..., 3],
#                                   visualize=False, num_segments=num_segments)
#     if feature_df is None:
#         feature_df = feature_extractor(image, gray_image, image_segments, label=0)
#     else:
#         feature_df = pd.concat([
#             feature_df, feature_extractor(image, gray_image, image_segments, label=0)
#         ])
#     print(f"农田特征提取完毕！消耗时间: {str(time.time() - t)}")
#     t = time.time()
#     image2 = Image.open(f"real_data/trad_alg/2021_1_{i}_res_0.5_建设用地.tif")
#     gray_image2 = np.array(image2.convert("L"))
#     image2 = np.array(image2)
#     image_segments = slic_segment(image2[..., :3], mask=image2[..., 3],
#                                   visualize=False, num_segments=num_segments)
#     feature_df = pd.concat([
#         feature_df, feature_extractor(image2, gray_image2, image_segments, label=1)
#     ])
#     print(f"建设用地特征提取完毕！消耗时间: {str(time.time() - t)}")
#     t = time.time()
#     image3 = Image.open(f"real_data/trad_alg/2021_1_{i}_res_0.5_林地园地草地.tif")
#     gray_image3 = np.array(image3.convert("L"))
#     image3 = np.array(image3)
#     image_segments = slic_segment(image3[..., :3], mask=image3[..., 3],
#                                   visualize=False, num_segments=num_segments)
#     feature_df = pd.concat([
#         feature_df, feature_extractor(image3, gray_image3, image_segments, label=2)
#     ])
#     print(f"林木用地特征提取完毕！消耗时间: {str(time.time() - t)}")
#
# feature_df.to_csv(
#         f"real_data/trad_alg/extract_features/extracted_features_{num_segments}_segments"
#         f"_{res}_res.csv")

# TODO 比较不同地貌类型提取图像的同质性分布差异
# 光谱部分完成，后续增加灰度共生矩阵所计算出的图像同质性，以及各种纹理特征。以及根据同质性特征对异常值的筛选。
# # 读取涂片1
# image = Image.open("real_data/trad_alg/2021_1_3_res_0.5_farmland.tif")
# image = np.array(image)
#
# image2 = Image.open("real_data/trad_alg/2021_1_3_res_0.5_forest.tif")
# image2 = np.array(image2)
#
#
# image4 = Image.open("real_data/trad_alg/2021_1_3_res_0.5_building.tif")
# image4 = np.array(image4)
#
# # 创建数据
# fg, ax = plt.subplots(1, 3, figsize=(15, 5))
# plt.suptitle("各种地貌类型的rgb三光谱分布箱型图")
# label = ['farmland', 'forest', 'building']
# sns.boxplot([image[image[..., 0] > 0, 0],
#              image2[image2[..., 0] > 0, 0],
#              image4[image4[..., 0] > 0, 0]
#              ], ax=ax[0])
# ax[0].set_xticklabels(label)
# sns.boxplot([image[image[..., 1] > 0, 1],
#              image2[image2[..., 1] > 0, 1],
#              image4[image4[..., 1] > 0, 1]
#              ], ax=ax[1])
# ax[1].set_xticklabels(label)
# sns.boxplot([image[image[..., 2] > 0, 2],
#              image2[image2[..., 2] > 0, 2],
#              image4[image4[..., 2] > 0, 2]
#              ], ax=ax[2])
# ax[2].set_xticklabels(label)
# plt.show()
#
# # 同质性分布
# image = fast_glcm_homogeneity(np.array(Image.fromarray(image).convert("L")))
# image2 = fast_glcm_homogeneity(np.array(Image.fromarray(image2).convert("L")))
# image4 = fast_glcm_homogeneity(np.array(Image.fromarray(image4).convert("L")))
# sns.boxplot([image[image < 25],
#              image2[image2 < 25],
#              image4[image4 < 25]
#              ])
# plt.show()
# TODO 对筛选以后的区域进行RGB、纹理的特征分析，并使用传统的机器学习算法对变化图斑进行分类
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix


def plotImp(model, X, num=20, fig_size=(40, 20)):
    """
    打印模型各个特征对预测影响的重要程度。

    :param model:
    :param X:
    :param num:
    :param fig_size:
    :return:
    """
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')
    plt.show()


data = pd.read_csv(
        f"real_data/trad_alg/extract_features/extracted_features_{num_segments}_segments_"
        f"{res}_res.csv",
        index_col="Unnamed: 0")
y = data['label'].values
x = data[data.columns[:-1]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf = LGBMClassifier()
clf.fit(x_train, y_train)
predict_train = clf.predict(x_train)
predict_test = clf.predict(x_test)
plotImp(clf, data[data.columns[:-1]])
print(f"模型得分：f1-score: {f1_score(predict_test, y_test, average='macro')}, accuracy-score: "
      f"{accuracy_score(predict_test, y_test)}, recall-score: {recall_score(predict_test, y_test, average='macro')}")
print(confusion_matrix(predict_test, y_test))

# TODO 对要进行变化识别的图片先进行超像素分割，再提取光谱、纹理特征。
for i in range(1, 5):
    # 使用当年三调图斑提取待检测变化的目标时域的遥感影像中的指定区域，并使用slic超像素分割
    target_image_path = f"real_data/trad_alg/2020_1_{i}_res_{res}_耕地.tif"
    image = Image.open(target_image_path)
    gray_image = np.array(image.convert("L")).astype(np.float32)
    image = np.array(image).astype(np.float32)
    image_segments = slic_segment(image[..., :3], mask=image[..., 3],
                                  visualize=False, num_segments=num_segments)
    # 然后提取每个超像素的光谱、纹理特征信息
    features = feature_extractor(image, gray_image, image_segments=image_segments, label=0)
    features = features[features.columns[:-1]].values
    # 输入到训练好的lightgbm模型中预测每块超像素块的类别信息
    # 使用指定阈值进行过滤
    predict = clf.predict_proba(features)
    detect_change = np.zeros_like(image_segments)
    for j in range(predict.shape[0]):
        for k in range(predict.shape[1]):
            if predict[j, k] > 0.6:
                detect_change[image_segments == j + 1] = k + 1
    plt.imshow(detect_change)
    plt.show()

    # TODO 将预测的变化结果保存为栅格以及矢量图像
    from osgeo import gdal
    from utils.gdal_utils import write_img
    from utils.polygon_utils import raster2vector

    image = gdal.Open(target_image_path)
    write_img(f"output/semantic_result/trad_alg/change_result/detect_change_1_{i}.tif",
              image.GetProjection(),
              image.GetGeoTransform(),
              detect_change.reshape((1, detect_change.shape[0], detect_change.shape[1])))
    raster2vector(f"output/semantic_result/trad_alg/change_result/detect_change_1_{i}.tif",
                  f"output/semantic_result/trad_alg/change_result/detect_change_1_{i}.shp")