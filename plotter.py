#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import dataset
import collections

gender_dict = {
    -1: 'male',
    -2: 'female'
}

Meaning = {
    
}


def plot_pie(d):
    y = d.label_feats[:, 4].astype(np.int32)
    size = y.shape[0]
    v_c_dict = collections.Counter(y)
    v_p_dict = {v: c / float(size) for v, c in v_c_dict.items()}
    # labels = [gender_dict[v] for v in v_p_dict.keys()]
    labels = v_p_dict.keys()
    values = v_p_dict.values()
    plt.pie(values, labels=labels, autopct='%4.2f%%', shadow=False, startangle=90)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    d = dataset.DataReader()
    plot_pie(d)
