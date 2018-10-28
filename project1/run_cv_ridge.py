# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 05:37:56 2018

@author: Nicolas
"""

from utils.cleaning import *
from utils.implementations import *
from utils.proj1_helpers import *
import numpy as np
import matplotlib.pyplot as plt
import utils.perceptron as prc

lambda_ = 0.00031
max_iters = 250000
proportion=0.1


# Load clean and standardize the data
#data_x, data_y, train_ids, means, stds = load_clean_standardize_train("../../train.csv",replace=False)
#print("Cleaning done")
#print(data_x.shape)
#regressor=np.ones((data_x.shape[0],data_x.shape[1]+1))
#regressor[:,1:]=data_x
#initial_w=np.array([.0]*(data_x.shape[1]+1))
##output=cross_validation(ridge_regression,data_y,regressor,lambda_,initial_w,max_iters,gamma,proportion)
##print(output[3])
#test_errors=list()
#train_errors=list()
#w_opt = np.zeros(data_x.shape[1])
#test_opt = 100
#best_gamma = 10000
#for gamma in np.linspace(1.6214285714285714e-06,3.8285714285714285e-06,5):
##for gamma in np.linspace(0.000001,0.15,10):
#    print(gamma)
#
#    w,mean_train_l,mean_test_l,train_error,test_error,std_train_l, std_test_l,std_train_e,std_test_e=cross_validation(prc.bagging_t,data_y,data_x,lambda_,initial_w,max_iters,gamma,proportion)
#    #we only want 1-accuracy
#    test_errors.append(test_error)
#    train_errors.append(train_error)
#    if test_opt > test_error: 
#        w_opt = w
#        test_opt = test_error
#        best_gamma = gamma
#    
#
#    
#plt.plot(test_errors)
#plt.plot(train_errors)   
#plt.savefig("crossvalidation_prc_bagging_10.png")
#plt.show()
best_gamma = 1.6214285714285714e-06

w_opt,mean_train_l,mean_test_l,train_error,test_error,std_train_l, std_test_l,std_train_e,std_test_e=cross_validation(prc.bagging_t,data_y,data_x,lambda_,initial_w,max_iters,best_gamma,proportion)
print("Done first cross-val : %f accuracy" % test_error )

## Best selection for now : 450 
final_w_2_opt = np.zeros(data_x.shape[1])
acc = 0
opt_selection = 0
for i in range(100,400,20):
    print("Starting to select only : %d best values" % i)
    liste=list(np.arange(len(w_opt)))
    #list the features from the less important to the most important
    liste.sort(key= lambda i : np.abs(w_opt[i]))
    liste=np.array(liste)-1
    liste=liste[liste>=0] #excluding the -1 than can occur
    selection = i
    reduced_data=data_x[:,liste[selection:]]
    final_w_2=prc.bagging(100,data_y,reduced_data,np.array([.0]*(data_x.shape[1]+1-selection)),250000,best_gamma)
    curr_acc = np.sum(np.heaviside(data_y*prc.predictions(reduced_data,final_w_2),.5))/len(data_y)
    print("curr acc %f" % curr_acc)
    if curr_acc > acc:
        final_w_2_opt = final_w_2
        acc = curr_acc
        opt_selection = selection

# delete this for memory mgmt
del data_x
del data_y   
del reduced_data
del regressor
del train_ids

test_x, test_y, test_ids = load_clean_standardize_test("../../test.csv", means, stds,replace=False)
test_x_r =test_x[:,liste[opt_selection:]]

y_pred=prc.predictions(test_x_r,final_w_2_opt)

create_csv_submission(test_ids, y_pred, "carpe_diem.csv")