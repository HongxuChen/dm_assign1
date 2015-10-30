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

import dataset
import plotter
import utils

GROUP_NUM = 5

NO_GROUP_LIST = [4, 5]

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


def try_group(label_feat):
    label_feat = label_feat.astype(np.int8)
    uniques = np.unique(label_feat)
    if uniques.shape[0] < GROUP_NUM:
        return label_feat
    utils.get_logger().info('dimension={}, needs fitting'.format(uniques.shape[0]))
    fittted_feat = np.zeros(label_feat.shape, dtype=int)
    separators = np.linspace(np.min(label_feat), np.max(label_feat), GROUP_NUM)
    for i in xrange(label_feat.shape[0]):
        for j in xrange(GROUP_NUM - 1):
            # print(separators[j], label_feat[i], separators[j+1])
            if separators[j] <= label_feat[i] < separators[j + 1]:
                fittted_feat[i] = j
                break
        else:
            fittted_feat[i] = GROUP_NUM - 1
    return fittted_feat


class Monitered(object):
    def __init__(self, ml_name, data, split_dict, **kwargs):
        self.sample_feats, self.label_feat = self.preprocessing(data.sample_feats, data.label_feats, INDEX)
        assert np.all(np.isfinite(self.label_feat))
        self.ml_name = ml_name
        # training and test
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = \
            train_test_split(self.sample_feats, self.label_feat, **split_dict)
        # model
        train_test_str = '-' + dict_to_str(split_dict)
        if kwargs is not None and len(kwargs) != 0:
            classifier_str = '-' + dict_to_str(kwargs)
            pickle_name = str(INDEX) + '-' + self.ml_name + train_test_str + classifier_str + '.pkl'
        else:
            pickle_name = str(INDEX) + '-' + self.ml_name + train_test_str + '.pkl'
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

    @staticmethod
    def preprocessing(sample_feats, label_feats, index):
        label_feat = label_feats[:, index]
        concat = np.column_stack((sample_feats, label_feat))
        if np.any(np.isnan(label_feat)):
            utils.get_logger().warning('contains NAN')
            cleaned = concat[~np.isnan(concat).any(axis=1)]
            sample_feats = cleaned[:, :-1]
            label_feat = cleaned[:, -1]
            utils.get_logger().warning('sample:{}, label:{}'.format(sample_feats.shape, label_feat.shape))
        if utils.isclf_dict[index]:
            utils.get_logger().warning('try grouping {}'.format(plotter.label_dict[index]))
            if index not in NO_GROUP_LIST:
                label_feat = try_group(label_feat)
            plotter.plot_pie(label_feat, index)
        return sample_feats, label_feat

    def cross_validation(self):
        ypred = self.clf.predict(self.Xtest)
        if utils.isclf_dict[INDEX]:
            return accuracy_score(self.ytest, ypred)
        else:
            return r2_score(self.ytest, ypred)

    def get_clf(self, kwargs):
        clf = clf_dict[self.ml_name](**kwargs)
        return clf


def sgd(name):
    print('sgd: {}'.format(name))
    c = Monitered(name, d, split_dict)
    score = c.cross_validation()
    print('{:<10s} {:<15.6f}'.format(name, score))
    print()


def svm_classifier():
    print('svm classification')
    lin_svc = Monitered('lin_svc', d, split_dict)
    svc_linear = Monitered('svc', d, split_dict, kernel='linear')
    svc_rbf = Monitered('svc', d, split_dict)
    svc_poly1 = Monitered('svc', d, split_dict, kernel='poly', degree=3)
    svc_poly2 = Monitered('svc', d, split_dict, kernel='poly', degree=5)
    sgd = Monitered('sgd', d, split_dict)
    for c in [lin_svc, svc_linear, svc_rbf, svc_poly1, svc_poly2, sgd]:
        score = c.cross_validation()
        print('{:<10s}: {:<15.6f}'.format(c.ml_name, score))
    print()


def svm_regression():
    print('---svm regression---')
    lin_svr = Monitered('lin_svr', d, split_dict)
    svr_linear = Monitered('svr', d, split_dict, kernel='linear')
    svr_rbf = Monitered('svr', d, split_dict, kernel='rbf')
    svr_poly1 = Monitered('svr', d, split_dict, kernel='poly', degree=3)
    svr_poly2 = Monitered('svr', d, split_dict, kernel='poly', degree=5)
    for c in [lin_svr, svr_linear, svr_rbf, svr_poly1, svr_poly2]:
        score = c.cross_validation()
        print('{:<10s} {:<15.6f}'.format(c.ml_name, score))
    print()


def forest(name, estimators):
    if name in ['rf', 'ab', 'bg']:
        kind = 'classification'
    else:
        kind = 'regression'
    print('forest, name={}'.format(name))
    scores = []
    for i in estimators:
        rf = Monitered(name, d, split_dict, n_estimators=i)
        score = rf.cross_validation()
        scores.append(score)
    for i, score in zip(estimators, scores):
        print('i={:<2d}, score={:<15.6f}'.format(i, score))
    print()


def knn(name):
    assert name in ['knn', 'knnr']
    if name == 'knn':
        kind = 'classifications'
    else:
        kind = 'regressions'
    print('knn '.format(kind))
    scores = []
    knn_range = range(2, 6)
    for i in knn_range:
        knn = Monitered(name, d, split_dict, n_neighbors=i)
        score = knn.cross_validation()
        scores.append(score)
    for i, score in zip(knn_range, scores):
        print('i={:<2d} {:<15.6f}'.format(i, score))
    print()


def run_once():
    global INDEX
    for INDEX in [0, 1, 4, 5]:
        sgd('sgd')


def run():
    global INDEX
    indexs = range(0, 6)
    for INDEX in indexs:
        print('\n======\n{}\n======'.format(plotter.label_dict[INDEX]))
        sgd('sgd')
        svm_classifier()
        knn('knn')
        # forests
        names = ['rf', 'ab', 'bg']
        for name in names:
            forest(name, forest_estimator_dict[name])
            # for INDEX in [2, 3]:
            #     sgd('sgdr')
            #     svm_regression()
            #     knn('knnr')
            #     names = ['rfr', 'abr', 'bgr']
            #     for name in names:
            #         forest(name, forest_estimator_dict[name])


if __name__ == '__main__':
    utils.init_logger()
    d = dataset.DataReader()
    split_dict = {
        'random_state': 42,
        'train_size': 0.8
    }
    # run_once()
    run()
