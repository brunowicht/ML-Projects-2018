# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 07:10:34 2018

@author: Nicolas
"""
from utils.cleaning import *
from utils.implementations import *
from utils.proj1_helpers import *
import numpy as np

lambda_ = 0.0001
max_iter = 10
gamma = 0.5
proportion=0.1
file_path='feature_selection.csv'


# Load clean and standardize the data
data_x, data_y, train_ids, means, stds = load_clean_standardize_train("train.csv")
print("Cleaning done")
print(data_x.shape)
regressor=np.ones((data_x.shape[0],data_x.shape[1]+1))
regressor[:,1:]=data_x
initial_w=np.array([.0]*(data_x.shape[1]+1))


feature_wrapping(ridge_regression,data_y,data_x,lambda_,max_iter,gamma,proportion,file_path)