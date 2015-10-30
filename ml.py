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
    'abr': range(40, 66, 5),
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
        if ISCLF[index]:
            utils.get_logger().warning('try grouping {}'.format(utils.label_dict[index]))
            if index not in NO_GROUP_LIST:
                label_feat = try_group(label_feat)
            plotter.plot_pie(label_feat, index)
        return sample_feats, label_feat

    def cross_validation(self):
        ypred = self.clf.predict(self.Xtest)
        if ISCLF[INDEX]:
            return accuracy_score(self.ytest, ypred)
        else:
            return r2_score(self.ytest, ypred)

    def get_clf(self, kwargs):
        clf = clf_dict[self.ml_name](**kwargs)
        return clf


def sgd(name):
    print('sgd: {}'.format(name))
    for i in range(3, 8):
        c = Monitered(name, D, SPLIT_DICT, n_iter=i)
        score = c.cross_validation()
        print('iter={:<2d} score={:<15.2f}'.format(i, score * 100))
    print()


def svm_classifier():
    print('svm classification')
    lin_svc = Monitered('lin_svc', D, SPLIT_DICT)
    lin_svc.clf.kernel = 'lin_svc'
    svc_linear = Monitered('svc', D, SPLIT_DICT, kernel='linear')
    svc_rbf = Monitered('svc', D, SPLIT_DICT)
    # score
    for c in [lin_svc, svc_linear, svc_rbf]:
        score = c.cross_validation()
        print('kernel={:<10s} score={:<15.2f}'.format(c.clf.kernel, score * 100))
    scores = []
    degrees = range(2, 6)
    for degree in degrees:
        svc_poly = Monitered('svc', D, SPLIT_DICT, kernel='poly', degree=degree)
        score = svc_poly.cross_validation()
        scores.append(score)
    print('kernel=poly')
    for degree, score in zip(degrees, scores):
        print('  degree={:<2d}, score={:<15.2f}'.format(degree, score * 100))
    print()


def svm_regression():
    print('svm regression')
    lin_svr = Monitered('lin_svr', D, SPLIT_DICT)
    lin_svr.clf.kernel = 'lin_svr'
    svr_linear = Monitered('svr', D, SPLIT_DICT, kernel='linear')
    svr_rbf = Monitered('svr', D, SPLIT_DICT, kernel='rbf')
    for c in [lin_svr, svr_linear, svr_rbf]:
        score = c.cross_validation()
        print('kernel={:<10s} score={:<15.2f}'.format(c.clf.kernel, score * 100))
    degrees = range(2, 6)
    scores = []
    for degree in degrees:
        svr_poly = Monitered('svr', D, SPLIT_DICT, kernel='poly', degree=degree)
        score = svr_poly.cross_validation()
        scores.append(score)
    print('kernel=poly')
    for degree, score in zip(degrees, scores):
        print('\tdegree={:<3d}, score={:<15.2f}'.format(degree, score * 100))
    print()


def forest(name, estimators):
    if name in ['rf', 'ab', 'bg']:
        kind = 'classification'
    else:
        kind = 'regression'
    print('forest {}, name={}'.format(kind, name))
    scores = []
    for i in estimators:
        rf = Monitered(name, D, SPLIT_DICT, n_estimators=i)
        score = rf.cross_validation()
        scores.append(score)
    for i, score in zip(estimators, scores):
        print('estimators={:<3d}, score={:<15.2f}'.format(i, score * 100))
    print()


def knn(name):
    assert name in ['knn', 'knnr']
    if name == 'knn':
        kind = 'classifications'
    else:
        kind = 'regressions'
    print('knn '.format(kind))
    scores = []
    knn_range = range(2, 7)
    for i in knn_range:
        knn = Monitered(name, D, SPLIT_DICT, n_neighbors=i)
        score = knn.cross_validation()
        scores.append(score)
    for i, score in zip(knn_range, scores):
        print('estimators={:<3d} {:<15.2f}'.format(i, score * 100))
    print()


def classification():
    global ISCLF
    ISCLF = {
        0: True,
        1: True,
        2: True,
        3: True,
        4: True,
        5: True
    }
    global INDEX
    # classification
    indexes = range(0, 6)
    for INDEX in indexes:
        print('\n======\n{}\n======'.format(utils.label_dict[INDEX]))
        sgd('sgd')
        svm_classifier()
        knn('knn')
        # forests
        names = ['rf', 'ab', 'bg']
        for name in names:
            forest(name, forest_estimator_dict[name])


def regression():
    global ISCLF
    ISCLF = {
        0: False,
        1: False,
        2: False,
        3: False,
        4: True,
        5: True
    }
    global INDEX
    indexes = range(0, 4)
    for INDEX in indexes:
        print('\n======\n{}\n======'.format(utils.label_dict[INDEX]))
        sgd('sgdr')
        svm_regression()
        knn('knnr')
        names = ['rfr', 'abr', 'bgr']
        for name in names:
            forest(name, forest_estimator_dict[name])


def run():
    classification()
    # regression()


if __name__ == '__main__':
    utils.init_logger()
    D = dataset.DataReader()
    SPLIT_DICT = {
        'random_state': 42,
        'train_size': 0.8
    }
    run()
