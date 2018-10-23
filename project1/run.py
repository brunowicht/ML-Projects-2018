# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:38:51 2018

@author: bruno
"""

from utils.cleaning import *
from utils.implementations import *
from utils.proj1_helpers import *
import numpy as np

lambda_ = 0.0
max_iters = 100
gamma = 1.
proportion=0.1





# Load clean and standardize the data
data_x, data_y, train_ids,mean__,std__ = load_clean_standardize_train("C:/Users/Test/train.csv")
print("Cleaning done")
data_y_log=data_y/2. +0.5

feature_wrapping(reg_logistic_regression,data_y_log,data_x,lambda_,max_iters,gamma,proportion,'proportion10percent_noLambda_gammaOne_iter100.dat')
#test_x, test_y, test_ids = load_clean_standardize_test("../../MLData2018/test.csv", means, stds)
#y_pred = predict_labels(w, test_x, logistic=False)
#create_csv_submission(test_ids, y_pred, "submission.csv")