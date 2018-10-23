# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:38:51 2018

@author: bruno
"""

from utils.cleaning import *
from utils.implementations import *
from utils.proj1_helpers import *
import numpy as np

lambda_ = 0.8
max_iters = 1000
gamma = 0.3


# Load clean and standardize the data
#data_x, data_y, train_ids, means, stds = load_clean_standardize_train("../../MLData2018/train.csv")
print("Cleaning done")
print(data_x.shape)

gammas = np.linspace(0.000001, 0.000004, 4)
lambdas = np.linspace(0.0000,0.0000002,21)
best_gamma = 0.00
best_loss = 1000
best_lambda = 0.00
#put the first parameter: constant column 1
regressor=np.ones((data_x.shape[0],data_x.shape[1]+1))
regressor[:,1:]=data_x
for l in lambdas:
    print("lambda: "+str(l))
    initial_w = np.zeros(regressor.shape[1])
    data_y_log_reg = (data_y/2) +0.5
    w,loss = ridge_regression(data_y, regressor, l)

    loss = compute_loss(data_y,regressor, w)
    print(loss)
    if loss < best_loss:
        best_loss = loss
        best_lambda = l

print("done")        
print("best loss: "+str(best_loss))
print("best lambda: "+str(best_lambda))
w,loss = ridge_regression(data_y, regressor, best_lambda)


test_x, test_y, test_ids = load_clean_standardize_test("../../MLData2018/test.csv", means, stds)
regressor=np.ones((test_x.shape[0],test_x.shape[1]+1))
regressor[:,1:]=test_x
y_pred = predict_labels(w, regressor, logistic=False)
create_csv_submission(test_ids, y_pred, "submission.csv")