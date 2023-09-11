# -*- coding=utf-8 -*-
# name: MengHao Tian
# date: 2023/4/18 16:33
import argparse
import os.path
import pickle
import flask

import numpy as np
import cv2
import torch
from train import MLP
import gen

BASE_PATH = os.path.dirname(__file__)
TEMP_PATH = 'D:\\homework\\java\\storage\\face\\temp'
DATASET_PATH = 'D:\\homework\\java\\storage\\face\\dataset'


def test(img_path: str = TEMP_PATH + '\\38f74915-428c-4249-b16c-32f5cf645702.png'):
    print(int(detect(cv2.imread(img_path), debug=True)))


APP = flask.Flask(__name__)
MODEL = torch.load(BASE_PATH + '/model.pth')
MODEL.eval()
with open(BASE_PATH + '/label_to_id.pkl', 'rb') as f:
    LABEL_TO_ID = pickle.load(f)


def detect(img: cv2.Mat = None, confidence: float = 0.80, debug: bool = False) -> int:
    r"""
    检测图像,返回user_id
    :param img: Mat对象
    :param debug: 开启调试(输出日志)
    :param confidence: 最低置信度
    :return: user_id，如果检测失败或者没有达到最低置信度则返回-1
    """
    feature = gen.get_feature('', img, is_mat=True)
    if feature is None:  # 获取特征失败
        return -1
    outputs = MODEL.forward(torch.from_numpy(feature))
    # 测试用的输出语句,实际运行注释掉
    if debug:
        for i, output in enumerate(outputs.tolist()[0]):
            print(i, int(LABEL_TO_ID[i]), '\t{:.16f}'.format(output))
    pred_mx, predicted = torch.max(outputs.data, 1)
    if pred_mx.item() < confidence:  # 准确度不满足要求
        return -1
    return int(LABEL_TO_ID[predicted.item()])


@APP.route('/face/detect', methods=['POST'])
def handle():
    # 获取post请求中data的二进制图像流,转为numpy数组
    data = np.frombuffer(flask.request.data, np.uint8)
    # 将numpy数组解码为图像
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    ret = detect(img)
    return flask.jsonify(ret)


if __name__ == '__main__':
    # 测试
    # test()
    # 正式运行
    APP.run(host='127.0.0.1', port=8015)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, default="")
    # args = parser.parse_args()
    # print(int(detect(args.path)))
