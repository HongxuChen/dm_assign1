#!/usr/bin/env python

import collections
import os

import matplotlib as mpl

import matplotlib.pyplot as plt
import utils

info_dir = 'fig'
if not os.path.exists(info_dir):
    os.mkdir(info_dir)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True


def plot_pie(d, index):
    fname = os.path.join(info_dir, utils.label_dict[index] + '.png')
    if os.path.exists(fname):
        return
    size = d.shape[0]
    v_c_dict = collections.Counter(d)
    v_p_dict = {v: c / float(size) * 100 for v, c in v_c_dict.items()}
    values = v_p_dict.values()
    patches, texts = plt.pie(values, shadow=False, startangle=90)
    # plt.axis('equal')
    title = utils.label_dict[index].upper() + ' total=${}$'.format(size)
    # legend
    legend_labels = ['${:.2f}\%$'.format(v) for v in values]
    plt.legend(patches, legend_labels, loc='best')
    plt.title(title, bbox={'facecolor': '0.8', 'pad': 5}, loc='right')
    plt.savefig(fname)
    plt.clf()
