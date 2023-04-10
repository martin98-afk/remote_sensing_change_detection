# encoding:utf-8
import os
import zipfile


def zip2file(zip_file_name, root):
    """
    将多个文件压缩存储为zip

    :param zip_file_name:
    :param root: 线程缓存的图像地址
    :return:
    """
    if not os.path.exists(root + "/src_image"):
        os.makedirs(root + "/src_image")
    file = zipfile.ZipFile(zip_file_name)
    file.extractall(root + "/src_image")


def file2zip(zip_file_name, file_names):
    """
    将多个文件压缩存储为zip

    :param zip_file_name:
    :param file_names:
    :return:
    """
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in file_names:
            parent_path, name = os.path.split(fn)
            zf.write(fn, arcname=name)