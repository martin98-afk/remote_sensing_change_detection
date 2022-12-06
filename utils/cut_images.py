from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from glob import glob


def generate_cut_images(target_dir, image_path, mask_path, img_size=512):
    os.makedirs(os.path.join(target_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "val", "labels"), exist_ok=True)

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    image = np.asarray(image)
    mask = np.asarray(mask)
    image_shape = image.shape
    for i in range(2000):
        train_x_rd = int(np.random.random() * (int(image_shape[0] * 0.8) - img_size))
        train_y_rd = int(np.random.random() * (image_shape[1] - img_size))
        train_image = image[train_x_rd:train_x_rd + img_size, train_y_rd:train_y_rd + img_size, :]
        train_mask = mask[train_x_rd:train_x_rd + img_size, train_y_rd:train_y_rd + img_size]
        train_image = Image.fromarray(train_image.astype(np.uint8))
        train_image.save(os.path.join(target_dir, "train", "images",
                                      f"{image_path.split('/')[-1][:-4]}_{i}.png"))
        train_mask = Image.fromarray(train_mask.astype(np.uint8))
        train_mask.save(os.path.join(target_dir, "train", "labels",
                                     f"{image_path.split('/')[-1][:-4]}_{i}.png"))

    for i in range(100):
        val_x_rd = int(np.random.random() * (int(image_shape[0] * 0.2) - img_size)) + int(
            image_shape[0] * 0.8)
        val_y_rd = int(np.random.random() * (image_shape[1] - img_size))
        val_image = image[val_x_rd:val_x_rd + img_size, val_y_rd:val_y_rd + img_size, :]
        val_mask = mask[val_x_rd:val_x_rd + img_size, val_y_rd:val_y_rd + img_size]

        val_image = Image.fromarray(val_image.astype(np.uint8))
        val_image.save(os.path.join(target_dir, "val", "images",
                                    f"{image_path.split('/')[-1][:-4]}_{i}.png"))
        val_mask = Image.fromarray(val_mask.astype(np.uint8))
        val_mask.save(os.path.join(target_dir, "val", "labels",
                                   f"{image_path.split('/')[-1][:-4]}_{i}.png"))


if __name__ == "__main__":
    mask_paths = glob("../real_data/semantic_mask/*.png")
    image_paths = [path.replace("semantic_mask", "processed_data").replace("png", "tif")
                   for path in mask_paths]
    img_size = 512
    for i, (image_path, mask_path) in tqdm(enumerate(zip(image_paths, mask_paths)),
                                           desc="cutting images"):
        generate_cut_images("../real_data/generated_cut_images", image_path, mask_path)
