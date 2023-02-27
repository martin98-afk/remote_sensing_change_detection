# encoding:utf-8
import pandas as pd
import pymysql

# mysql连接信息
MYSQL_CONFIG = {
    'host':    '192.168.9.153',
    'user':    'urbanlab_admin',
    'passwd':  'urbanlab_123',
    'db':      'db_landform_dev',
    'port':    3306,
    'charset': 'utf8'
}


class MysqlConnectionTools:

    def __init__(self, **kwargs):
        self.conn = pymysql.connect(**kwargs)
        self.host = MYSQL_CONFIG["host"]
        self.port = MYSQL_CONFIG["port"]
        self.username = MYSQL_CONFIG["user"]
        self.password = MYSQL_CONFIG["passwd"]
        self.db = MYSQL_CONFIG["db"]

    def write_to_mysql_progress(self, sub_id, progress):
        cursor = self.conn.cursor()
        curr_time = pd.datetime.now()
        sql = f"insert into land_subject_progress (subject_id, progress, create_time) " \
              f"values('{sub_id}', '{progress}', '{curr_time}')"
        cursor.execute(sql)
        self.conn.commit()

    def write_to_mysql_relation(self, sub_id, image_url, change_num):
        cursor = self.conn.cursor()
        curr_time = pd.datetime.now()
        sql = f"insert into land_subject_relation (subject_id, res_image_url, create_time, change_num) " \
              f"values('{sub_id}', '{image_url}', '{curr_time}', '{change_num}')"
        cursor.execute(sql)
        self.conn.commit()


if __name__ == "__main__":
    mysql_conn = MysqlConnectionTools(**MYSQL_CONFIG)

    mysql_conn.write_to_mysql_relation("1234", "10%")
