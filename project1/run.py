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
max_iters = 50
gamma = 0.0035


# Load clean and standardize the data
data_x, data_y, train_ids, means, stds = load_clean_standardize_train("../../MLData2018/train.csv")
print("Cleaning done")

gammas = np.linspace(0.01, 1, 101)
best_gamma = 0.00
best_loss = 1000

for g in gammas:
    print(g)
    initial_w = np.zeros(data_x.shape[1])
    data_y_log_reg = (data_y/2) +0.5
    w,loss = least_squares_GD(data_y, data_x,initial_w,max_iter,g)


    loss = compute_loss(data_y, data_x, w)
    print(loss)
    if loss < best_loss:
        best_loss = loss
        best_gamma = g

print(best_loss)
print(best_gamma)
#test_x, test_y, test_ids = load_clean_standardize_test("../../MLData2018/test.csv", means, stds)
#y_pred = predict_labels(w, test_x, logistic=False)
#create_csv_submission(test_ids, y_pred, "submission.csv")