#!/usr/bin/env python
from __future__ import print_function
import os
import pickle

import numpy as np
from sklearn import svm, linear_model
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
import dataset
import utils

model_dir = 'models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

clf_dict = {
    'lin_svc': svm.LinearSVC,
    'svc': svm.SVC,
    'sgd': linear_model.SGDClassifier,
    'knn': neighbors.KNeighborsClassifier,
    'rf': ensemble.RandomForestClassifier,
    'gb': ensemble.GradientBoostingClassifier,
    'ab': ensemble.AdaBoostClassifier,
    'bg': ensemble.BaggingClassifier
}

forest_estimator_dict = {
    'rf': range(8, 13),
    'ab': range(80, 135, 5),
    'bg': range(8, 13)
}


def dict_to_str(data_dict):
    data_list = ['{}_{}'.format(k, v) for k, v in data_dict.items()]
    s = '_'.join(data_list)
    return s


class Classifier(object):
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
        return accuracy_score(self.ytest, ypred)

    def get_clf(self, kwargs):
        clf = clf_dict[self.clf_name](**kwargs)
        return clf


def svm_run():
    lin_svc = Classifier('lin_svc', d, split_dict)
    svc_linear = Classifier('svc', d, split_dict, kernel='linear')
    svc_rbf = Classifier('svc', d, split_dict)
    svc_poly1 = Classifier('svc', d, split_dict, kernel='poly', degree=3)
    svc_poly2 = Classifier('svc', d, split_dict, kernel='poly', degree=5)
    sgd = Classifier('sgd', d, split_dict)
    for c in [lin_svc, svc_linear, svc_rbf, svc_poly1, svc_poly2, sgd]:
        score = c.cross_validation()
        print('{:20s}: {:15.6f}'.format(c.clf_name, score))


def forest_run(name, estimators):
    scores = []
    for i in estimators:
        rf = Classifier(name, d, split_dict, n_estimators=i)
        score = rf.cross_validation()
        scores.append(score)
    for i, score in zip(estimators, scores):
        print('i={:<2d}, score={:<15.6f}'.format(i, score))


def knn_run():
    scores = []
    knn_range = range(2, 6)
    for i in knn_range:
        knn = Classifier('knn', d, split_dict, n_neighbors=i)
        score = knn.cross_validation()
        scores.append(score)
    for i, score in zip(knn_range, scores):
        print('{:<2d} {:<15.6f}'.format(i, score))


if __name__ == '__main__':
    utils.init_logger()
    d = dataset.DataReader()
    split_dict = {
        'random_state': 42,
        'train_size': 0.8
    }
    INDEX = 5
    # rf_run()
    # knn_run()
    # gb_run()
    name = 'bg'
    forest_run(name, forest_estimator_dict[name])
