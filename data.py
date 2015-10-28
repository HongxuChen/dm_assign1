#!/usr/bin/env python
from __future__ import print_function
import os
import pickle
import random
import numpy as np
import logging

data_file = 'data.csv'
data_prefix = data_file.split('.')[0]
data_pickle = data_prefix + '.pkl'

SHAPE = (3476, 1248)
GROUP1 = 64, 18
GROUP2 = 5, 18
LABEL_FEAT = SHAPE[1] - GROUP1[0] * GROUP1[1] - GROUP2[0] * GROUP2[1]


def get_logger():
    return logging.getLogger()


class Reader(object):
    def __init__(self):
        if os.path.isfile(data_pickle):
            get_logger().debug('{} does not exist'.format(data_pickle))
            with open(data_pickle, 'rb') as raw_data:
                self.data = pickle.load(raw_data)
        else:
            get_logger().debug('{} exists'.format(data_pickle))
            assert os.path.isfile(data_file)
            self.data = np.genfromtxt(data_file, delimiter=',')
            with open(data_pickle, 'wb') as raw_data:
                pickle.dump(self.data, raw_data)

    @staticmethod
    def shape_validator(data, shape):
        if data.shape != shape:
            raise ValueError('shape error: expected {}, found {}'.format(shape, data.shape))


class DataReader(Reader):
    def __init__(self):
        super(DataReader, self).__init__()
        Reader.shape_validator(self.data, SHAPE)
        self.sample_features = self.data[:, :-LABEL_FEAT]
        self.label_features = self.data[:, -LABEL_FEAT:]

    def get_grp1_feat_n(self, n):
        start = n * GROUP1[0]
        return self.data[:, start:start + GROUP1[1]]

    def get_grp2_feat_n(self, n):
        start = GROUP1[0] * GROUP1[1] + n * GROUP2[0]
        print('start={}'.format(start))
        # return self.data[:, start:start + GROUP2[1]]
        return None

    def get_grp2_feat_instance(self, n):
        feat = self.get_grp1_feat_n(n)
        rnd = random.randint(0, GROUP2[1])
        return feat[:, rnd]


if __name__ == '__main__':
    d = DataReader()
    instance = d.get_grp2_feat_instance(0)
    print(instance)
