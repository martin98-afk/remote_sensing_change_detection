# encoding:utf-8
import json
# from torch.multiprocessing import Process
# import multiprocessing
import time
from threading import Thread

from flask import Flask, request, jsonify
from gevent import pywsgi

from config import minio_store, MYSQL_CONFIG
from detect_change import get_parser
from utils.conn_mysql import *
from utils.detect_change_server_func import DetectTask
from utils.pipeline import RSPipeline

# multiprocessing.set_start_method("spawn", force=True)

app = Flask(__name__)
# 导入detect change中的超参数，同时导入模型及各种参数
args = get_parser()
args = vars(args)
# 导入模型
data, model = RSPipeline.load_model('output/ss_eff_b0_with_glcm.yaml',
                                    model_type=args["model_type"],
                                    device=args["device"],
                                    half=False)

# 连接mysql数据库存储程序运行状态
mysql_conn = MysqlConnectionTools(**MYSQL_CONFIG)


# TODO 使用flask进行多进程创建和调度，使整个识别任务可以并行运行，目前是串行运行。
def trigger_process(info_dict):
    task = DetectTask(info_dict, minio_store, model, mysql_conn, args)
    t = Thread(target=task.process_task)
    t.start()
    return t


@app.errorhandler(400)
def bad_argument(error):
    # TODO 处理错误信息，将错误信息写入mysql数据库
    # mysql_conn.update_status()
    return jsonify({'message': error.description['message']})


@app.route("/detect_change", methods=['POST'])
def detect_change():
    """
    变化识别api接口程序，支持网格化变化识别、图斑变华识别，针对非农化土地变化识别。

    :return:
    """
    info_dict = request.form.to_dict()
    if len(info_dict) == 0:
        info_dict = request.json

    mysql_conn.update_status(sub_id=info_dict["id"],
                             status=4)
    process = trigger_process(info_dict)
    process_dict[info_dict["id"]] = process
    if len(process_dict.keys()) % 5 == 0:
        time.sleep(240)
    return json.loads(json.dumps({"message": "开启成功"}))


if __name__ == '__main__':
    # 进程池
    process_dict = {}
    server = pywsgi.WSGIServer(('192.168.9.161', 8002), app)
    server.serve_forever()
