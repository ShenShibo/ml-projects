#!/usr/bin/python2
# -*-coding:utf-8-*-
# import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics.scorer import check_scoring
import pickle as pkl
import time


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
        train_label.append(eval(d[-1]))
    # 数据正则化
    train_data = np.array(train_data, dtype=np.float64)
    train_label = np.array(train_label, dtype=np.int)
    # 随机打散
    rd_idx = np.random.permutation(train_data.shape[0])
    train_data = train_data[rd_idx]
    train_label = train_label[rd_idx]
    # 正则
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    train_data = (train_data - mean) / std
    # print np.mean(train_data, axis=0), np.std(train_data, axis=0)
    print "data prepare done!"
    return train_data, train_label, mean, std


def svm_train(gamma=5, c_scale=1, kernal='rbf', data=None, label=None):
    assert data is not None and label is not None
    # 调用SVM分类器
    classifier = SVC(gamma=gamma, C=c_scale, kernal=kernal, max_iter=1000000)
    # scores = cross_val_score(estimator=classifier, X=data, y=label, cv=5, scoring='f1', n_jobs=-1)
    scoring = 'accurary'
    scorer = check_scoring(classifier, scoring)
    ret = cross_validate(
        estimator=classifier,
        X=data,
        y=label,
        cv=5,
        scoring=scorer,
        return_estimator=True,
        n_jobs=-1
    )
    scores = ret['test_score']
    classifier = ret['estimator'][np.argmax(scores)]
    print "cross validation average test score:{}".format(scores.mean())
    print "train score without whole set trained:{}".format(classifier.score(data, label))
    # 再对全体训练样本训练一次
    classifier.fit(data, label)
    print "final score:{}".format(classifier.score(data, label))
    with open("svm_clf_gamma_{}_c_{}.p".format(gamma, c_scale), 'wb') as f:
        pkl.dump(classifier, f)


def svm_predict(model=None, data_path='testing.data', mean=None, std=None):
    print "start prediction"
    assert model is not None and \
           mean is not None and \
           std is not None
    try:
        with open(data_path, 'r') as f:
            lines = f.readlines()
    except Exception, e:
        print e
        return
    test_data = []
    for i, line in enumerate(lines):
        d = line.split()
        # 数据维度是6
        data = [eval(d[j]) for j in range(6)]
        # 集成
        test_data.append(data)
    # 数据正则化,需要拟合训练集的分布
    test_data = np.array(test_data, dtype=np.float64)
    test_data = (test_data - mean) / std
    start = time.time()
    y = model.predict(test_data)
    print y, len(y)
    print "time cost:{:.2f}".format(time.time() - start)
    print "Done!"


if __name__ == "__main__":
    path = 'training.data'
    X, y, mean, std = data_prepare(path)
    svm_train(
        gamma=5,
        c_scale=1,
        kernal='rbf',
        data=X,
        label=y
    )
    # with open('svm_clf_gamma_5_c_1.p', 'rb') as f:
    #     clf = pkl.load(f)
    # svm_predict(
    #     model=clf,
    #     mean=mean,
    #     std=std
    # )
    # 设置参数区间
    # gamma_region = range(2, 8, 1)
    # c_region = range(2, 9, 1)
    # kernal = 'rbf'
    #
    # for gamma in gamma_region:
    #     for c_scale in c_region:
    #         print "gamma:{},c_scale:{}".format(gamma, c_scale)
    #         clf = SVC(gamma=gamma, C=c_scale, kernel=kernal, max_iter=1000000)
    #         scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, scoring='f1',n_jobs=-1)
    #         print "average score:{}".format(scores.mean())
    # 测试结果gamma = 5， c = 1
    pass



