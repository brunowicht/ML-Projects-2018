# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:57:33 2018

@author: Nicolas
"""

from utils.cleaning import *
from utils.implementations import *
from utils.proj1_helpers import *
import numpy as np
import utils.perceptron as prc




# Load clean and standardize the data
data_x, data_y, train_ids, means, stds = load_clean_standardize_train("C:/Users/Test/train.csv")
print("Cleaning done")
print(data_x.shape)
#for i in np.logspace(-9,0,10):
#    w=prc.train(data_y,data_x,np.array([0.]*496),250000,i)
#    print("training done ")
#    y_hat=prc.predictions(data_x[:250000],w)
#    matches=np.heaviside(y_hat*data_y,.5)
#    print(np.sum(matches)/250000)
#    print(i)
#1e-6 is the best one

final_w=prc.bagging(30,data_y,data_x,np.array([.0]*496),250000,1e-6)
print(np.sum(np.heaviside(data_y*prc.predictions(data_x[:250000],final_w),.5))/len(data_y))

#test_x, test_y, test_ids = load_clean_standardize_test("C:/Users/Test/test.csv", means, stds)
#regressor=np.ones((test_x.shape[0],test_x.shape[1]+1))
#regressor[:,1:]=test_x
#y_pred=prc.predictions(test_x[:250000],w)

#create_csv_submission(test_ids, y_pred, "hoc_erat_in_votis.csv")