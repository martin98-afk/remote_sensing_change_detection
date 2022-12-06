from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

image = Image.open(
    "/home/xwtech/遥感识别专用/output/semantic_result/change_result/detect_change_block_2_4.tif")
image = np.array(image)
data = np.where(image == 1)
data = np.array([data[0], data[1]]).T
cluster = DBSCAN(eps=20, min_samples=40)
cluster.fit(data)