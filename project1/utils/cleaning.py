# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:17:24 2018

@author: bruno
"""

import numpy as np
from utils.proj1_helpers import load_csv_data

def column_replace_invalid_by_mean(col):
    mean = np.mean(col[col != -999])
    col[col == -999] = mean
    return col

def fill_unknown_with_column_mean(dx):
    d = dx.transpose()
    for i in range(len(d)):
        col = d[i]
        d[i] = column_replace_invalid_by_mean(col)
    return d.transpose()

def standardize_train(dx):
    d = dx.transpose()
    means = []
    stds = []
    for i in range(len(d)):
        mean = d[i].mean()
        means.append(mean)
        std = d[i].std()
        stds.append(std)
        d[i] = (d[i] - mean) / std
    return d.transpose(), means, stds

def standardize_test(dx, means, stds):
    d = dx.transpose()
    for i in range(len(d)):
        d[i] = (d[i] - means[i])/stds[i]
    return d.transpose()

def load_clean_standardize_train(file):
    data = load_csv_data(file)
    data_y = data[0]
    ids = data[2]
    data_x, means, stds = standardize_train(fill_unknown_with_column_mean(data[1]))
    return data_x, data_y, ids, means, stds

def load_clean_standardize_test(file, means, stds):
    data = load_csv_data(file)
    data_y = data[0]
    ids = data[2]
    data_x = standardize_test(fill_unknown_with_column_mean(data[1]), means, stds)
    return data_x, data_y, ids



