# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:17:24 2018

@author: bruno
"""

import numpy as np
from utils.proj1_helpers import load_csv_data

def find_999_columns(dx):
    tx = dx.transpose()
    cols = []
    for i in range(len(tx)):
        if -999 in tx[i]:
            cols.append(i)
    return cols

def add_columns_for_999(dx):
    d999 = find_999_columns(dx)
    return np.concatenate((dx, (dx[:,d999]==-999).astype(int)), axis=1)
        

def column_replace_invalid_by_mean(col):
    mean = np.mean(col[col != -999])
    col[col == -999] = mean
    return col

def clean_and_normalize_train_column(col):
    col_copy=col.copy()
    relevant_data=col_copy[col_copy !=-999]
    print(relevant_data)
    mean=np.mean(relevant_data)
    std=np.std(relevant_data)
    col_copy[col_copy==-999]=mean
    print(mean)
    print(std)
    print(col_copy)
    ret=(col_copy-mean)/std
    print(col_copy)
    return ret, mean ,std
def clean_and_normalize_test_column(col,mean,std):
    col_copy=col.copy()
    col_copy[col_copy==-999]=mean
    print(col_copy)
    ret=(col_copy-mean)/std
    return ret

def expand_features_degree2(dx):
    res = list()
    a = 0
    for x in dx:
        a+=1
        new = list(x)
        for i in range(len(x)):
            for j in range(i,len(x)):
                new.append(x[i] * x[j])
        res.append(new)
        if a % 1000 == 0:
            print(a)
    print("expand done")
    return np.array(res)

def expand_features_degree3(dx):
    res = list()
    a = 0
    new = list()

    for x in dx:
        a +=1 
        new = []
        for i in range(len(x)):
            for j in range(i,len(x)):
                new.append(x[i] * x[j])
        for i in range(len(x)):
            for j in range(i,len(x)):
                for k in range(j, len(x)):
                    new.append(x[i] * x[j] * x[k])
        res.append(new)
        if a % 1000 == 0:
            print(a)
    return np.array(res)


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
        if std==0:
            std = 1
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
    data_x = add_columns_for_999(data[1])
    data_x = fill_unknown_with_column_mean(data_x)
    data_x, means, stds = standardize_train(data_x)
    data_x = expand_features_degree2(data_x)
    return data_x, data_y, ids, means, stds

def load_clean_standardize_test(file, means, stds):
    data = load_csv_data(file)
    data_y = data[0]
    ids = data[2]
    data_x = add_columns_for_999(data[1])
    data_x = fill_unknown_with_column_mean(data_x)    
    data_x = standardize_test(data_x, means, stds)
    data_x = expand_features_degree2(data_x)
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



