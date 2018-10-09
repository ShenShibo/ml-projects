#!/usr/bin/python2
# -*-coding:utf-8-*-
# import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def data_prepare(data_path = None):
    try:
        with open(data_path, 'r') as f:
            lines = f.readlines()
    except Exception, e:
        print e
        return
    train_data = []
    train_label = []
    for i, line in enumerate(lines):
        d = line.split()
        # 数据维度是6
        data = [eval(d[j]) for j in range(6)]
        # 集成
        train_data.append(data)
        train_label.append(eval(d[6]))
    # 数据正则化
    train_data = np.array(train_data, dtype=np.float16)
    train_label = np.array(train_label, dtype=np.int)
    # 随机打散
    rd_idx = np.random.permutation(train_data.shape[0])
    train_data = train_data[rd_idx]
    train_label = train_label[rd_idx]
    # 正则
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data = (train_data - mean) / std
    return train_data, train_label


if __name__ == "__main__":
    path = 'training.data'
    X, y = data_prepare(path)
    # 设置参数区间
    gamma_region = range(-15, 4, 2)
    c_region = range(-5, 16, 2)
    kernal = 'rbf'
    for g in gamma_region:
        gamma = 2 ** g
        for c in c_region:
            c_scale = 2 ** c
            print "gamma:{},c_scale:{}".format(gamma, c_scale)
            clf = SVC(gamma=gamma, C=c_scale, kernel=kernal)
            scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, scoring='f1',n_jobs=-1)
            print "average score:{}".format(scores.mean())
            



