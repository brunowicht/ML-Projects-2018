# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:40:38 2018

@author: johan
"""
from cleaning import * 
import numpy as np
data_x, data_y, train_ids, means, stds = load_clean_standardize_train("../../../train.csv")
#test_t,test_y, test_ids = load_clean_standardize_test("../../test.csv")



p_minus = len(data_y[data_y == -1])/len(data_y)
p_plus = len(data_y[data_y == 1])/len(data_y)



## Note that the derived columns depend on the primitive values. 
## Hence we should not use them in our computation. 
## There is 13 derived columns and they are the first. 
## So we take only the 17 last columns

data_x = data_x[:,[range(13,30)]]
## for each column compute means_w and std_w

means_w = np.mean(data_x, axis=0)
std_w = np.std(data_x,axis=0)

