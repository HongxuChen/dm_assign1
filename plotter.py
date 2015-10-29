#!/usr/bin/env python

import collections
import os

import matplotlib.pyplot as plt
import numpy as np

import dataset

info_dir = 'info'
if not os.path.exists(info_dir):
    os.mkdir(info_dir)

meaning = {
    0: 'age',
    1: 'height',
    2: 'weight',
    3: 'bmi',
    4: 'gender',
    5: 'race'
}


def plot_pie(d, index):
    y = d.label_feats[:, index].astype(np.int32)
    size = y.shape[0]
    v_c_dict = collections.Counter(y)
    v_p_dict = {v: c / float(size) for v, c in v_c_dict.items()}
    # labels = [gender_dict[v] for v in v_p_dict.keys()]
    labels = v_p_dict.keys()
    values = v_p_dict.values()
    plt.pie(
        values,
        labels=labels,
        autopct='%4.2f%%',
        shadow=False,
        startangle=90)
    plt.axis('equal')
    title = meaning[index]
    plt.title(title, bbox={'facecolor': '0.8', 'pad': 5})
    fname = os.path.join(info_dir, title + '.png')
    plt.savefig(fname)


if __name__ == '__main__':
    d = dataset.DataReader()
    for i in [0, 1, 4, 5]:
        plot_pie(d, i)
