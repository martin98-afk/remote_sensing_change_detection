# encoding:utf-8
import pandas as pd
import pymysql


class MysqlConnectionTools:

    def __init__(self, **kwargs):
        self.conn = pymysql.connect(**kwargs)

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
        sql = f"insert into land_subject_relation (subject_id, res_image_url, create_time, " \
              f"change_num) " \
              f"values('{sub_id}', '{image_url}', '{curr_time}' , '{change_num}')"
        cursor.execute(sql)
        self.conn.commit()

    def update_status(self, sub_id, status, error_message=None):
        """
        向mysql数据库中更新当前任务的状态
        加状态字段 status
            0 新建
            1 发布中
            2 发布失败
            3 发布成功
            4 分析中
            5 分析成功
            6 分析失败

        :param sub_id:
        :param status:
        :param error_message:
        :return:
        """
        cursor = self.conn.cursor()
        sql = f"UPDATE land_subject_relation " \
              f"SET status='{status}'" \
              f"WHERE id='{sub_id}'"
        cursor.execute(sql)
        self.conn.commit()
        if error_message is not None:
            cursor = self.conn.cursor()
            sql = f"UPDATE land_subject_relation " \
                  f"SET err_msg='{error_message}'" \
                  f"WHERE id='{sub_id}'"

            cursor.execute(sql)
            self.conn.commit()

    def update_result(self, sub_id, res_image_url, change_num):
        """
        向mysql中存储结果路径

        :param sub_id:
        :param res_image_url:
        :param change_num:
        :return:
        """
        cursor = self.conn.cursor()
        sql = f"UPDATE land_subject_relation " \
              f"SET res_image_url='{res_image_url}', change_num='{change_num}'" \
              f"WHERE id='{sub_id}'"
        cursor.execute(sql)
        self.conn.commit()


if __name__ == "__main__":
    ...
