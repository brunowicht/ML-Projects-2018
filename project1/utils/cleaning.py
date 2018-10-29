# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:17:24 2018

@author: bruno
"""

import numpy as np
from utils.proj1_helpers import load_csv_data

def find_999_columns(dx):
    """
    Finds all the columns that contain a value that is -999
    """
    tx = dx.transpose()
    cols = []
    for i in range(len(tx)):
        if -999 in tx[i]:
            cols.append(i)
    return cols

def add_columns_for_999(dx):
    """
    Creates new columns for columns that have -999. 
    This is a new parameter we create. 
    """
    d999 = find_999_columns(dx)
    return np.concatenate((dx, (dx[:,d999]==-999).astype(int)), axis=1)
        

def column_replace_invalid_by_mean(col):
    """
    Where the values are -999 
    we replace by the median in order to ahve standardized values. 
    """
    mean = np.median(col[col != -999])
    col[col == -999] = mean
    return col

def clean_and_normalize_train_column(col):
    """
    Clean an normalize a training column
    Done by : 
        - compute the median and std of value that are not -999
        - set the value of any -999 to the median
        - normalize all values
    """
    col_copy=col.copy()
    relevant_data=col_copy[col_copy !=-999]
    mean=np.median(relevant_data)
    std=np.std(relevant_data)
    col_copy[col_copy==-999]=mean
    ret=(col_copy-mean)/std
    return ret, mean ,std
def clean_and_normalize_test_column(col,mean,std):
    """
    Clean an normalize a training column
    Done by : 
        - set the value of any -999 to the median
        - normalize all values
    """
    col_copy=col.copy()
    col_copy[col_copy==-999]=mean
    ret=(col_copy-mean)/std
    return ret

def expand_features_degree2(dx):
    """
        Do a 2D expansion of the features, 
        For every row we compute : 
            - x[i] * x[j] for all i and j features. This allows for more precise computations
    """
    res = list()
    for x in dx:
        new = list(x)
        for i in range(len(x)):
            for j in range(i,len(x)):
                new.append(x[i] * x[j])
        res.append(new)

    return np.array(res)

def expand_features_degree3(dx):
    """
        We have not used this as it was not precise..
        Does a 3D expansion of features 
    """
    res = list()
    new = list()

    for x in dx:
        new = []
        for i in range(len(x)):
            for j in range(i,len(x)):
                new.append(x[i] * x[j])
        for i in range(len(x)):
            for j in range(i,len(x)):
                for k in range(j, len(x)):
                    new.append(x[i] * x[j] * x[k])
        res.append(new)
        
    return np.array(res)



def clean_and_normalize(train_x,test_x):
    """
        cleans and normalizes the training and testing sets. 
    """
    clean_train_x=np.zeros(train_x.shape)
    clean_test_x=np.zeros(test_x.shape)
    for i in range(train_x.shape[1]):
        clean_train_x[:,i],mean,std=clean_and_normalize_train_column(train_x[:,i])
        clean_test_x[:,i]=clean_and_normalize_test_column(test_x[:,i],mean,std)
    
    return clean_train_x,clean_test_x

def fill_unknown_with_column_mean(dx):
    """
        Fills columns of the data that have -999 values into values that can be standardized. 
    """
    d = dx.transpose()
    for i in range(len(d)):
        col = d[i]
        d[i] = column_replace_invalid_by_mean(col)
    return d.transpose()

def standardize_train(dx):
    """
        Standardize the data. Will compute mean and medians to use during standardization
    """
    d = dx.transpose()
    means = []
    stds = []
    for i in range(len(d)):
        mean = np.median(d[i])
        means.append(mean)
        std = d[i].std()
        if std==0:
            std = 1
        stds.append(std)
        d[i] = (d[i] - mean) / std
    return d.transpose(), means, stds

def standardize_test(dx, means, stds):
    """
        standardize the data according to given means and std. 
    """
    d = dx.transpose()
    for i in range(len(d)):
        d[i] = (d[i] - means[i])/stds[i]
    return d.transpose()

def load_clean_standardize_train(file,expansion=True,replace=True):
    """
        Loads and cleans and standardize the training set. 
        We can decide to expand 2D the parameter and replace the columns -999 if we want. 
        
    """
    data = load_csv_data(file)
    data_y = data[0]
    ids = data[2]
    data_x = data[1]
    if replace:
        data_x = add_columns_for_999(data_x)
    data_x = fill_unknown_with_column_mean(data_x)
    data_x, means, stds = standardize_train(data_x)
    if expansion : 
        data_x = expand_features_degree2(data_x)
    return data_x, data_y, ids, means, stds

def load_clean_standardize_test(file, means, stds,expansion=True,replace=True):
    """
        Loads and cleans the testing set. 
        It will use the given mean and std during the noramlization 
        We can decide to expand 2D the parameter and replace the columns -999 if we want. 
        
    """
    data = load_csv_data(file)
    data_y = data[0]
    ids = data[2]
    if replace:
        data_x = add_columns_for_999(data[1])
    data_x = fill_unknown_with_column_mean(data_x)    
    data_x = standardize_test(data_x, means, stds)
    if expansion:
        data_x = expand_features_degree2(data_x)
    return data_x, data_y, ids



def load_raw(file,rand = False):
    """
        Loads the data and shuffles it if necessary. 
        does not normalize it. 
    """
    data = load_csv_data(file)
    
    data_y = data[0]
    data_x = data[1]
    if rand:
        data_y,data_x = shuffle(data[0],data[1])
    
    
    ids = data[2]
    return data_x,data_y,ids


def shuffle(data_y,data_x):
    """
    shuffles the data_y and data_x so that the rows with same index stay the same. 
    """
    
    p = np.random.permutation(len(data_x))
    return data_y[p], data_x[p]



