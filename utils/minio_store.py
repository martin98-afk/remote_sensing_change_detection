# encoding:utf-8
from minio import Minio

class MinioStore:
    """文件存储服务器，用于对要处理的图片的读取以及结果图片的保存"""
    def __init__(self, host, access_key, secret_key, bucket, save_dirs):
        self.host = host
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = False
        self.bucket = bucket
        self.save_dirs = save_dirs
        self.client = Minio(
                self.host,
                secure=self.secure,
                access_key=self.access_key,
                secret_key=secret_key
        )

    def __new__(cls, *args, **kw):
        '''
        启用单例模式
        :param args:
        :param kw:
        :return:
        '''
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls)
        return cls._instance

    def get_object(self, file_name):
        found = self.client.bucket_exists(self.bucket)
        if not found:
            self.client.make_bucket(self.bucket)
            print(f"create {self.bucket} success")

        save_path = self.save_dirs + file_name
        data = self.client.get_object(
                self.bucket, file_name
        )

        with open(save_path, "wb") as f:
            for d in data:
                f.write(d)
        return save_path

    def fget_object(self, object_name, file_name):
        self.client.fget_object(
                self.bucket, object_name, file_name
        )

    def fput_object(self, object_name, file_name):
        self.client.fput_object(
                self.bucket, object_name, file_name
        )

    def put_object(self, object_name, raw_data, raw_size):
        found = self.client.bucket_exists(self.bucket)
        if not found:
            self.client.make_bucket(self.bucket)
            print('create {} sucess！'.format(self.bucket))

        return self.client.put_object(
                self.bucket, object_name, raw_data,raw_size
        )