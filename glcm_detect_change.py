# encoding:utf-8

import seaborn as sns
from skimage.segmentation import mark_boundaries, slic

from config import *
from remotesensing_alg.fast_glcm import *

warnings.filterwarnings('ignore')

# TODO 根据三调矢量数据进行耕地、果园林地、建设用地等等区域进行提取
# 目前已完成。
# 从大类标注文件中提取耕地图斑
import pandas as pd
import numpy as np

from PIL import Image


# 对应地理国情的编号
# TODO 添加从原有一级类和二级类的标签中提取一级类，并保存
# 在真实使用场景是直接使用提取好耕地图斑的数据进行识别
# shp_file = read_shp("real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1.shp")
# shp_file["DLBM"] = shp_file["DLBM"].apply(lambda x: int(x[:2]))
# shp_file.to_file("real_data/移交数据和文档/苏南/0.2米航片对应矢量数据/DLTB_2021_1_major_class.shp")
#
# res = "0.5"
#
# # 对应三调编号
# # 提取耕地
# extract_target([1], "耕地", res)
#
# # 提取林地
# extract_target([2], "林地园地草地", res)
#
# # 提取建设用地
# extract_target([5, 6, 7, 8], "建设用地", res)


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


num_segments = 500
data = pd.read_csv(
        f"real_data/trad_alg/extract_features/extracted_features_{num_segments}_segments.csv",
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
target_image_path = "/home/xwtech/遥感识别专用/real_data/trad_alg/2020_1_3_res_0.5_耕地.tif"

# image = cut_raster("real_data/test9.tif",
#                    "real_data/移交数据和文档/rice/rice.shp",
#                    refer_path="real_data/processed_data/2020_1_1_res_0.5.tif")
# image = image.ReadAsArray()
# image = np.transpose(image, (1, 2, 0))
# Image.fromarray(image).save("real_data/test9_rice.tif")
image = Image.open(target_image_path)
gray_image = np.array(image.convert("L")).astype(np.float32)
image = np.array(image).astype(np.float32)
image_segments = slic_segment(image[..., :3], mask=image[..., 3],
                              visualize=False, num_segments=num_segments)
features = feature_extractor(image, gray_image, image_segments=image_segments, label=0)
features = features[features.columns[:-1]].values
predict = clf.predict_proba(features)
detect_change = np.zeros_like(image_segments)
# for i in range(len(predict)):
#     detect_change[image_segments == i + 1] = predict[i] + 1
for j in range(predict.shape[0]):
    for k in range(predict.shape[1]):
        if predict[j, k] > 0.9:
            detect_change[image_segments == j + 1] = k + 1
        else:
            detect_change[image_segments == j + 1] = 1
plt.imshow(detect_change)
plt.show()

# TODO 将预测的变化结果保存为栅格以及矢量图像
from osgeo import gdal
from utils.gdal_utils import write_img
from utils.polygon_utils import raster2vector

image = gdal.Open(target_image_path)
write_img(
        f"output/semantic_result/trad_alg/change_result/2020_1_3.tif",
        image.GetProjection(),
        image.GetGeoTransform(),
        detect_change.reshape((1, detect_change.shape[0], detect_change.shape[1])))
raster2vector(
        f"output/semantic_result/trad_alg/change_result/2020_1_3.tif",
        f"output/semantic_result/trad_alg/change_result/2020_1_3.shp")
