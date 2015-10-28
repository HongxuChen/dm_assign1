#!/usr/bin/env python
import os
import pickle

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import dataset
import utils

model_dir = 'model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

clf_dict = {
    'lin_svc': svm.LinearSVC,
    'svc': svm.SVC
}


class Classifier(object):
    def __init__(self, data, clfname, kwargs):
        self.data = data
        self.sample_feats = self.data.sample_feats
        self.label_feats = self.data.label_feats[:, 5]
        self.clf_name = clfname
        # training and test
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = \
            train_test_split(self.sample_feats, self.label_feats, **kwargs)
        print(self.Xtrain.shape, self.ytrain.shape, self.Xtest.shape, self.ytest.shape)
        # model
        pickle_name = self.clf_name + '.pkl'
        model_pickle = os.path.join(model_dir, pickle_name)
        if os.path.exists(model_pickle):
            utils.get_logger().warning('{} exists'.format(model_pickle))
            with open(model_pickle, 'rb') as model:
                self.clf = pickle.load(model)
        else:
            utils.get_logger().warning('{} does not exist'.format(model_pickle))
            self.clf = self.gen_model(kwargs)
            with open(model_pickle, 'wb') as model:
                pickle.dump(self.clf, model)

    def cross_validation(self):
        ypred = self.clf.predict(self.Xtest)
        return accuracy_score(self.ytest, ypred)

    def gen_model(self, kwargs):
        raise NotImplementedError


class LinearSVC(Classifier):
    def __init__(self, data, **kwargs):
        super(LinearSVC, self).__init__(data, 'lin_svc', kwargs)

    def gen_model(self, kwargs):
        clf = svm.LinearSVC()
        clf.fit(self.Xtrain, self.ytrain)
        return clf


if __name__ == '__main__':
    utils.init_logger()
    d = dataset.DataReader()
    lin_svc = LinearSVC(d, random_state=2)
    score = lin_svc.cross_validation()
    print(score)
