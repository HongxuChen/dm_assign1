#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import dataset
import collections


def plot_pie(d):
    y = d.label_feats[:, 4].astype(np.int32)
    v_c_dict = collections.Counter(y)
    print(v_c_dict)


if __name__ == '__main__':
    d = dataset.DataReader()
    plot_pie(d)
