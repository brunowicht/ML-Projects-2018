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

def clean_and_normalize_train_column(col):
    col_copy=col.copy()
    relevant_data=col_copy[col_copy !=-999]
    mean=np.mean(relevant_data)
    std=np.std(relevant_data)
    col_copy[col_copy==-999]=mean
    print(mean)
    ret=(col_copy-mean)/std
    return ret, mean ,std
def clean_and_normalize_test_column(col,mean,std):
    col_copy=col.copy()
    col_copy[col_copy==-999]=mean
    print(col_copy)
    ret=(col_copy-mean)/std
    return ret


#the next function bugs a bit, test it with small matrix containing -999 kiss kisss Nicolas

def clean_and_normalize(train_x,test_x):
    clean_train_x=np.zeros(train_x.shape)
    clean_test_x=np.zeros(test_x.shape)
    for i in range(train_x.shape[1]):
        clean_train_x[:,i],mean,std=clean_and_normalize_train_column(train_x[:,i])
        clean_test_x[:,i]=clean_and_normalize_test_column(test_x[:,i],mean,std)
    
    return clean_train_x,clean_test_x

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



def load_raw(file,rand = False):
    data = load_csv_data(file)
    
    data_y = data[0]
    data_x = data[1]
    if rand:
        data_y,data_x = shuffle(data[0],data[1])
    
    
    ids = data[2]
    return data_x,data_y,ids


def shuffle(data_y,data_x):
    
    p = np.random.permutation(len(data_x))
    return data_y[p], data_x[p]



