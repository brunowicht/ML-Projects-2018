# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 05:37:56 2018

@author: Nicolas
"""

from utils.cleaning import *
from utils.implementations import *
from utils.proj1_helpers import *
import numpy as np

lambda_ = 0.00031
max_iters = 10000
gamma = 0.5
proportion=0.1


# Load clean and standardize the data
#data_x, data_y, train_ids, means, stds = load_clean_standardize_train("C:/Users/Test/train.csv")
print("Cleaning done")
print(data_x.shape)
regressor=np.ones((data_x.shape[0],data_x.shape[1]+1))
regressor[:,1:]=data_x
initial_w=np.array([.0]*(data_x.shape[1]+1))
#output=cross_validation(ridge_regression,data_y,regressor,lambda_,initial_w,max_iters,gamma,proportion)
#print(output[3])
test_errors=list()
train_errors=list()
for lambda_ in np.logspace(-9,0,10):
    print(lambda_)
    a,b,c,test_error,train_error,f,g,h=cross_validation(ridge_regression,data_y,regressor,lambda_,initial_w,max_iters,gamma,proportion)
    #we only want 1-accuracy
    
    print(test_accuracy)
#test_x, test_y, test_ids = load_clean_standardize_test("test.csv", means, stds)
#test_regressor=np.ones((test_x.shape[0],test_x.shape[1]+1))
#test_regressor[:,1:]=test_x
#w,loss_=ridge_regression(data_y,regressor,lambda_)
#y_pred = predict_labels(w, test_regressor, logistic=False)
#create_csv_submission(test_ids, y_pred, "early_morning_submission_4.csv")