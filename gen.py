# -*- coding=utf-8 -*-
# name: MengHao Tian
# date: 2023/4/20 08:11
import argparse
import os.path
import warnings

from PIL import Image

import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as tt
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)


# D:/homework/py/blue-face
BASE_PATH = os.path.dirname(__file__)
RECOGNITION_PATH = BASE_PATH + '/face_recognition_sface_2021dec.onnx'
DETECTION_PATH = BASE_PATH + '/face_detection_yunet_2022mar.onnx'
recognizer = cv2.FaceRecognizerSF.create(RECOGNITION_PATH, '')
DATASET_PATH = 'D:\\homework\\java\\storage\\face\\dataset'


def get_cur_img_info(dir_path: str, label: int = -1) -> (list, list):
    r"""
    获取当前目录下的所有图像路径
    :param label: 标签
    :param dir_path: 文件夹目录
    :return: 所有图像绝对路径
    """
    img_abs_path_list, id_list = [], []
    if label == -1:
        label = int(dir_path.split('\\')[-1])
    for filename in os.listdir(dir_path):
        abs_path = "%s\\%s" % (dir_path, filename)
        img_abs_path_list.append(abs_path)
        id_list.append(label)
    return img_abs_path_list, id_list


def save_label_user_id(user_id_list: np.ndarray, id_to_label_path: str = BASE_PATH + '/id_to_label.pkl', label_to_id_path: str = BASE_PATH + '/label_to_id.pkl'):
    r"""
    根据用户id生成顺序的label，并保存对应的id->label和label->id的映射文件
    :param user_id_list: 用户id列表
    :param id_to_label_path: id->label文件保存路径
    :param label_to_id_path: label->id文件保存路径
    """
    id_to_label, label_to_id = {}, {}
    if os.path.exists(id_to_label_path):  # 如果存在userid到label的映射文件,则先读取已存在文件
        with open(id_to_label_path, 'rb') as f:
            id_to_label = pickle.load(f)
    for i, user_id in enumerate(user_id_list):  # 将user_id映射为顺序的label
        id_to_label[user_id] = i
        label_to_id[i] = user_id
    with open(id_to_label_path, 'wb') as f:  # 保存user_id映射到label的文件
        pickle.dump(id_to_label, f)
    with open(label_to_id_path, 'wb') as f:  # 保存label映射到user_id的文件
        pickle.dump(label_to_id, f)


def get_img_info(dir_path: str = 'D:\\homework\\web\\blue\\file\\dataset') -> (list, list):
    r"""
    获取图像绝对路径和标签
    :param dir_path: 图像所在爷目录绝对路径(数据集路径)
    :return: 图像绝对路径列表, 标签列表
    """
    img_abs_path_list, id_list = [], []
    for dirname, dir_names, filenames in os.walk(dir_path):
        # sub_dir_name为dataset下的第一层目录名
        for sub_dir_name in dir_names:
            subject_path = os.path.join(dirname, sub_dir_name)
            img_list, ids = get_cur_img_info(subject_path)
            img_abs_path_list = img_abs_path_list + img_list
            id_list = id_list + ids
    return img_abs_path_list, id_list


def save_face(img_path: str, save_path: str):
    r"""
    截取图片中的人脸并保存
    :param img_path: 原图片路径
    :param save_path: 保存路径
    """
    t_img = cv2.imread(img_path)
    img = cv2.resize(t_img, (112, 112))  # 更改图像大小
    # 检测人脸
    faceDetector = cv2.FaceDetectorYN.create(DETECTION_PATH, '', img.shape[:2][::-1])
    # 获取人脸位置信息
    faces = faceDetector.detect(img)
    """ faces是一个元组，第一个是人脸个数，第二个是n*15维的矩阵
    [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
    其中，x1, y1是人脸框左上角坐标,w和h分别是人脸框的宽和高
    {x, y}_{re, le, nt, rcm, lcm}分别是人脸右眼瞳孔、左眼瞳孔、鼻尖、右嘴角和左嘴角的坐标,score是该人脸的得分。
    """
    # color = (255, 0, 0)
    # thickness = 2
    if faces[1] is None:
        return
    face_num, face_list = faces
    for i in range(len(face_list)):
        face_x, face_y, face_width, face_height = face_list[i][0], face_list[i][1], face_list[i][2], face_list[i][3]
        x1, y1, x2, y2 = int(face_x), int(face_y), int(face_x + face_width), int(face_y + face_height)
        if y1-5 < 0 or x1-5 < 0:  # 如果已经是人脸或者截取异常则直接跳过
            continue
        face = cv2.resize(img[y1 - 5:y2 + 5, x1 - 5:x2 + 5], (112, 112))
        cv2.imwrite(save_path, face)
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        # cv2.imshow('picture'+str(i), face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def get_feature(img_path: str, img: cv2.Mat = None, is_mat: bool = False):
    r"""
    获取人脸特征
    :param img_path: 图像路径
    :param img: cv2.Mat对象
    :param is_mat: 传入的是否是Mat，默认传入图像路径
    :return: ndArray,检测失败则返回None
    """
    if not is_mat:
        img = cv2.imread(img_path)
    if img is None:
        return None
    faceDetector = cv2.FaceDetectorYN.create(DETECTION_PATH, '', img.shape[:2][::-1])
    faces = faceDetector.detect(img)
    if faces[1] is None:  # 匹配失败返回None
        return None
    aligned_face = recognizer.alignCrop(img, faces[1][0])
    return recognizer.feature(aligned_face)


def match(img_org_path: str, img_des_path: str):
    feature1 = get_feature(img_org_path)
    feature2 = get_feature(img_des_path)
    if feature1 is None or feature2 is None:  # 有一个没检测出来就匹配失败
        print('匹配失败')
        return None
    cosine_similarity_threshold = 0.363
    score_cosine = recognizer.match(feature1, feature2, 0)

    print('-' * 5 + img_des_path.split('\\')[-1] + '-' * 5)
    # 使用cosine距离作为指标,值越大越好
    if score_cosine >= cosine_similarity_threshold:
        # 'the same identity'
        print('cosine:' + str(score_cosine) + '-' * 5 + '>yes')
    else:
        # 'different identities'
        print('cosine:' + str(score_cosine) + '-' * 5 + '>no')

    # 使用normL2距离作为指标,值越小越好
    l2_similarity_threshold = 1.128
    score_l2 = recognizer.match(feature1, feature2, 1)
    if score_l2 <= l2_similarity_threshold:
        # 'the same identity'
        print('normL2:' + str(score_l2) + '-' * 5 + '>yes')
    else:
        # 'different identities'
        print('normL2:' + str(score_l2) + '-' * 5 + '>no')


def feature_to_csv(img_path_list: list, label_list: list, save_path: str = BASE_PATH + '/data.csv', replace: bool = False, verbose: bool = False) -> np.ndarray:
    r"""
    获取图像特征保存到文件
    :param verbose: 是否显示打印信息
    :param img_path_list:图像绝对路径列表
    :param label_list: 标签列表
    :param save_path: 保存路径
    :param replace: 是否替换csv文件
    :return: user_id列表
    """
    rows = []
    for i, label in enumerate(label_list):
        feature = get_feature(img_path_list[i])
        if feature is not None:
            rows.append(np.append(feature, label))
        if verbose:
            print('%d img finish' % i)
    df = pd.DataFrame(rows, columns=[*['feature_' + str(i) for i in range(len(rows[0]) - 1)], 'user_id'])
    if os.path.exists(save_path) and not replace:  # 如果文件已存在并且不替换则追加
        df = pd.concat([df, pd.read_csv(save_path)], ignore_index=True)
    df.to_csv(save_path, index=False)
    print('img ok!')
    return df['user_id'].unique()


def improve_data_to_csv(img_list: list, label_list: list, save_path: str = BASE_PATH + '/data.csv', replace: bool = False, verbose: bool = False) -> np.ndarray:
    r"""
    处理图像,增加图像数据,保存人脸特征(不替换csv文件)
    :param replace: 是否替换原始文件
    :param verbose: 是否显示打印信息
    :param img_list: 图像绝对路径列表
    :param label_list: 标签列表
    :param save_path: 保存路径
    :return user_id列表
    """
    trans = [
        tt.CenterCrop(224),
        tt.RandomHorizontalFlip(p=0.8),
        tt.Grayscale(),
        tt.GaussianBlur(kernel_size=3),
        tt.RandomRotation(degrees=40, expand=False),
        tt.RandomRotation(degrees=30, expand=False),
        tt.RandomRotation(degrees=20, expand=False),
        tt.RandomRotation(degrees=10, expand=False),
        tt.RandomRotation(degrees=45, expand=True),
        tt.RandomRotation(degrees=35, expand=False),
        tt.RandomRotation(degrees=25, expand=False),
        tt.RandomRotation(degrees=15, expand=False),
        tt.RandomRotation(degrees=5, expand=False)
    ]
    feature = None
    rows = []
    df = pd.DataFrame(columns=[*['feature_' + str(i) for i in range(128)], 'user_id'])
    for i in range(len(img_list)):
        image = Image.open(img_list[i])
        for tran in trans:
            tran_img = tran(image)
            img = cv2.cvtColor(np.asarray(tran_img), cv2.COLOR_RGB2BGR)  # opencv 格式的图像
            feature = get_feature('', img, True)
            if feature is not None:
                rows.append(np.append(feature, label_list[i]))
        if verbose:
            print('%d improve img finish' % i)
    df = pd.concat([df, pd.DataFrame(rows, columns=[*['feature_' + str(i) for i in range(128)], 'user_id'])], ignore_index=True)
    if os.path.exists(save_path) and not replace:  # 如果文件已存在并且不替换则追加
        df = pd.concat([df, pd.read_csv(save_path)], ignore_index=True)
    df.to_csv(save_path, index=False)
    print('improve img ok!')
    return df['user_id'].unique()


if __name__ == '__main__':
    img_path_list, label_list = get_img_info(DATASET_PATH)
    # 增加图像对应的特征
    # user_id_list = improve_data_to_csv(img_path_list, label_list, replace=True, verbose=True)
    # save_label_user_id(user_id_list)
    # 原始图像对应的特征
    user_id_list = feature_to_csv(img_path_list, label_list, replace=False, verbose=True)
    save_label_user_id(user_id_list)

    # 不属于系统的图像特征进一步数据增强
    # img_path_list, label_list = get_cur_img_info(DATASET_PATH+'\\-1')
    # user_id_list = improve_data_to_csv(img_path_list, label_list, replace=False, verbose=True)
    # save_label_user_id(user_id_list)
