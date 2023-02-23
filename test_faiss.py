# encoding:utf-8
"""
通过异常检测的思路进行变化区域识别。
算法思路：
1. 提取所有耕地图斑。
2. 对耕地图斑进行超像素划分。
3. 对每个耕地超像素提取特征向量。
4. 使用faiss创建耕地特征向量库。
5. 对于待检测区域进行超像素划分。
6. 将划分好的超像素使用相同特征提取方法进行特征提取。
7. 将特征输入到faiss特征库中进行检索，并标记出异常区域。

"""
import faiss
import pandas as pd
from PIL import Image
from skimage.segmentation import mark_boundaries, slic

from config import *
from glcm_detect_change import feature_extractor
from remotesensing_alg.fast_glcm import *

warnings.filterwarnings('ignore')
import numpy as np

# 对应地理国情的编号
dict = {"耕地": [1], "建设用地": [3, 4, 5, 8], "林地园地草地": [2]}
class_id_list = np.arange(9)


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


feature_df = pd.read_csv("real_data/trad_alg/extract_features/extracted_features_300_segments_0"
                         ".5_res.csv", index_col="Unnamed: 0")

farmland_feature = feature_df.loc[feature_df['label'] == 0].values[:, :-1]\
    .copy(order='C').astype(np.float32)
building_feature = feature_df.loc[feature_df['label'] == 1].values[:, :-1]\
    .copy(order='C').astype(np.float32)
forest_feature = feature_df.loc[feature_df['label'] == 2].values[:, :-1] \
    .copy(order='C').astype(np.float32)
# 使用faiss构建特征向量库
index = faiss.IndexFlatL2(farmland_feature.shape[1])
index.add(farmland_feature[:-10, :])

# TODO 对要进行变化识别的图片先进行超像素分割，再提取光谱、纹理特征。

# 使用当年三调图斑提取待检测变化的目标时域的遥感影像中的指定区域，并使用slic超像素分割
target_image_path = f"real_data/trad_alg/2020_1_3_res_0.5_耕地.tif"
image = Image.open(target_image_path)
gray_image = np.array(image.convert("L")).astype(np.float32)
image = np.array(image).astype(np.float32)
image_segments = slic_segment(image[..., :3], mask=image[..., 3],
                              visualize=False, num_segments=300)
# 然后提取每个超像素的光谱、纹理特征信息
features = feature_extractor(image, gray_image, image_segments=image_segments, label=0)
features = features[features.columns[:-1]].values.copy(order='C').astype(np.float32)
# 输入到训练好的lightgbm模型中预测每块超像素块的类别信息
# 使用指定阈值进行过滤
score_patches, _ = index.search(features, k=8)
score_patches = score_patches[:, 0]
detect_change = np.zeros_like(image_segments)
for j in range(score_patches.shape[0]):
    detect_change[image_segments == j + 1] = score_patches[j]
plt.imshow(detect_change)
plt.show()