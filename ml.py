#!/usr/bin/env python
from __future__ import print_function
import os
import pickle

import numpy as np
from sklearn import svm, linear_model
from sklearn import neighbors
from sklearn.metrics import accuracy_score, r2_score
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
import sys
import dataset
import utils

model_dir = 'models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

clf_dict = {
    # classifier
    'lin_svc': svm.LinearSVC,
    'svc': svm.SVC,
    'sgd': linear_model.SGDClassifier,
    'knn': neighbors.KNeighborsClassifier,
    'rf': ensemble.RandomForestClassifier,
    'gb': ensemble.GradientBoostingClassifier,
    'ab': ensemble.AdaBoostClassifier,
    'bg': ensemble.BaggingClassifier,
    # regressor
    'lin_svr': svm.LinearSVR,
    'svr': svm.SVR,
    'sgdr': linear_model.SGDRegressor,
    'knnr': neighbors.KNeighborsRegressor,
    'rfr': ensemble.RandomForestRegressor,
    'gbr': ensemble.GradientBoostingRegressor,
    'abr': ensemble.AdaBoostRegressor,
    'bgr': ensemble.BaggingRegressor
}

forest_estimator_dict = {
    'rf': range(8, 13),
    'ab': range(80, 131, 10),
    'bg': range(8, 13),
    'gb': range(80, 131, 10),
    'rfr': range(8, 13),
    'abr': range(40, 61, 5),
    'bgr': range(8, 13),
    'gbr': range(80, 131, 10),
}


def dict_to_str(data_dict):
    data_list = ['{}_{}'.format(k, v) for k, v in data_dict.items()]
    s = '_'.join(data_list)
    return s


class Monitered(object):
    def __init__(self, clfname, data, split_dict, **kwargs):
        self.data = data
        self.sample_feats = self.data.sample_feats
        self.label_feat = self.data.label_feats[:, INDEX]
        concat = np.column_stack((self.sample_feats, self.label_feat))
        if np.any(np.isnan(self.label_feat)):
            utils.get_logger().warning('contains NAN')
            cleaned = concat[~np.isnan(concat).any(axis=1)]
            self.sample_feats = cleaned[:, :-1]
            self.label_feat = cleaned[:, -1]
            utils.get_logger().warning('sample:{}, label:{}'.format(self.sample_feats.shape, self.label_feat.shape))
        assert np.all(np.isfinite(self.label_feat))
        self.clf_name = clfname
        # training and test
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = \
            train_test_split(self.sample_feats, self.label_feat, **split_dict)
        # model
        train_test_str = '-' + dict_to_str(split_dict)
        if kwargs is not None and len(kwargs) != 0:
            classifier_str = '-' + dict_to_str(kwargs)
            pickle_name = str(INDEX) + '-' + self.clf_name + train_test_str + classifier_str + '.pkl'
        else:
            pickle_name = str(INDEX) + '-' + self.clf_name + train_test_str + '.pkl'
        model_pickle = os.path.join(model_dir, pickle_name)
        if os.path.exists(model_pickle):
            utils.get_logger().warning('{} exists'.format(model_pickle))
            with open(model_pickle, 'rb') as model:
                self.clf = pickle.load(model)
        else:
            utils.get_logger().warning('{} does not exist'.format(model_pickle))
            self.clf = self.get_clf(kwargs)
            self.clf.fit(self.Xtrain, self.ytrain)
            with open(model_pickle, 'wb') as model:
                pickle.dump(self.clf, model)
                # utils.get_logger().warning('CLF info\n{}'.format(self.clf))

    def cross_validation(self):
        ypred = self.clf.predict(self.Xtest)
        try:
            return accuracy_score(self.ytest, ypred)
        except ValueError as e:
            if e.message == 'continuous is not supported':
                utils.get_logger().warning('regression, using r2_score')
                return r2_score(self.ytest, ypred)
            else:
                utils.get_logger().critical('UNKNOWN')
                sys.exit(1)

    def get_clf(self, kwargs):
        clf = clf_dict[self.clf_name](**kwargs)
        return clf


def sgd(name):
    c = Monitered(name, d, split_dict)
    score = c.cross_validation()
    print('{:20s} {:15.6f}'.format(name, score))


def svm_classifier():
    lin_svc = Monitered('lin_svc', d, split_dict)
    svc_linear = Monitered('svc', d, split_dict, kernel='linear')
    svc_rbf = Monitered('svc', d, split_dict)
    svc_poly1 = Monitered('svc', d, split_dict, kernel='poly', degree=3)
    svc_poly2 = Monitered('svc', d, split_dict, kernel='poly', degree=5)
    sgd = Monitered('sgd', d, split_dict)
    for c in [lin_svc, svc_linear, svc_rbf, svc_poly1, svc_poly2, sgd]:
        score = c.cross_validation()
        print('{:20s}: {:15.6f}'.format(c.clf_name, score))


def svm_regression():
    lin_svr = Monitered('lin_svr', d, split_dict)
    svr_linear = Monitered('svr', d, split_dict, kernel='linear')
    svr_rbf = Monitered('svr', d, split_dict, kernel='rbf')
    svr_poly1 = Monitered('svr', d, split_dict, kernel='poly', degree=3)
    svr_poly2 = Monitered('svr', d, split_dict, kernel='poly', degree=5)
    for c in [lin_svr, svr_linear, svr_rbf, svr_poly1, svr_poly2]:
        score = c.cross_validation()
        print('{:20s} {:15.6f}'.format(c.clf_name, score))


def forest(name, estimators):
    scores = []
    for i in estimators:
        rf = Monitered(name, d, split_dict, n_estimators=i)
        score = rf.cross_validation()
        scores.append(score)
    for i, score in zip(estimators, scores):
        print('i={:<2d}, score={:<15.6f}'.format(i, score))


def knn(name):
    scores = []
    knn_range = range(2, 6)
    for i in knn_range:
        knn = Monitered(name, d, split_dict, n_neighbors=i)
        score = knn.cross_validation()
        scores.append(score)
    for i, score in zip(knn_range, scores):
        print('{:<2d} {:<15.6f}'.format(i, score))


def run():
    global INDEX
    for INDEX in [0, 1, 4, 5]:
        svm_classifier()
        sgd('sgd')
        knn('knn')
        # forests
        names = ['rf', 'ab', 'bg']
        for name in names:
            forest(name, forest_estimator_dict[name])
    for INDEX in [2, 3]:
        svm_regression()
        sgd('sgdr')
        knn('knnr')
        names = ['rfr', 'abr', 'bgr']
        for name in names:
            forest(name, forest_estimator_dict[name])


if __name__ == '__main__':
    utils.init_logger()
    d = dataset.DataReader()
    split_dict = {
        'random_state': 42,
        'train_size': 0.8
    }
    run()
