# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:38:51 2018

@author: bruno
"""

from utils.cleaning import load_clean_standardize
from utils.implementations import *
from utils.proj1_helpers import *
import numpy as np

lambda_ = 0.8
max_iter = 100
gamma = 0.7


# Load clean and standardize the data
data_x, data_y, train_ids = load_clean_standardize("../../MLData2018/train.csv")
test_x, test_y, test_ids = load_clean_standardize("../../MLData2018/test.csv")
print("Cleaning done")

initial_w = np.zeros(data_x.shape[1])

w,loss = least_squares(data_y,data_x)
print(w)
print(loss)

#y_pred = predict_labels(w, test_x)
#create_csv_submission(test_ids, y_pred, "submission.csv")