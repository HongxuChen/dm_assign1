#!/usr/bin/env python

import collections
import os

import matplotlib.pyplot as plt
import numpy as np

import utils
import dataset

info_dir = 'info'
if not os.path.exists(info_dir):
    os.mkdir(info_dir)

label_dict = {
    0: 'age',
    1: 'height',
    2: 'weight',
    3: 'bmi',
    4: 'gender',
    5: 'race'
}


def plot_pie(d, index):
    size = d.shape[0]
    v_c_dict = collections.Counter(d)
    v_p_dict = {v: c / float(size) for v, c in v_c_dict.items()}
    # labels = [gender_dict[v] for v in v_p_dict.keys()]
    labels = v_p_dict.keys()
    values = v_p_dict.values()
    plt.pie(
        values,
        labels=labels,
        autopct='%.1f%%',
        shadow=False,
        startangle=90)
    plt.axis('equal')
    title = label_dict[index] + ' total={}'.format(size)
    # plt.title(title, loc='center')
    plt.title(title, bbox={'facecolor': '0.8', 'pad': 5}, loc='right')
    fname = os.path.join(info_dir, label_dict[index] + '.png')
    plt.savefig(fname)
    plt.clf()


if __name__ == '__main__':
    d = dataset.DataReader()
    for i in [0, 1, 4, 5]:
        plot_pie(d, i)
