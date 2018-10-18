# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:38:51 2018

@author: bruno
"""

from utils.cleaning import load_clean_standardize
import utils.implementations as impl
import utils.proj1_helpers

import numpy as np

# Load clean and standardize the data
data_x, data_y, ids = load_clean_standardize("../../train.csv")

print(data_x)
print(data_y)

# 0.389916
#best_w = []
#best_loss = 1000.0
#lambdas = np.linspace(0,1,50)
#for lambda_ in lambdas:
#    w, loss = impl.ridge_regression(data_y,data_x,lambda_)
#    print("Percent done : "+str(100*lambda_))
#    if loss < best_loss:
#        best_loss = loss
#        best_w = w
#    
#    
#print("Ridge regression : ")
#print(best_loss)

w_init = np.zeros(data_x.shape[1])

#print(impl.least_squares(data_y,data_x))

best_w = w_init
best_loss = 1000.0
gammas = np.linspace(0.0015,0.003,11) #² opt : 0.00195


#data_y_log_reg = data_y/2+0.5
#y,tx,initial_w,max_iter,gamma
#for gamma in gammas:
#    w, loss = impl.logistic_regression(data_y_log_reg,data_x,best_w,50,gamma)
#    print("Gamma done : "+str(gamma)+ " Curr loss : " + str(loss))
#    if loss < best_loss:
#        print("New best loss : " + str(loss))
#        best_loss = loss
#        best_w = w
#lambda_ = 0.00195
#w, loss = impl.logistic_regression(data_y,data_x,best_w,1000,lambda_)
lambdas = np.linspace(1,1.5,5)
#for gamma in gammas:
gamma = 0.00255
for lambda_ in lambdas:
    w, loss = impl.reg_logistic_regression(data_y,data_x,lambda_,best_w,50,gamma)
    print("Gamma : "+str(gamma)+ ", Lamnda : "+ str(lambda_)+" Curr loss : " + str(loss))
    if loss < best_loss:
        print("New best loss : " + str(loss))
        best_loss = loss
        best_w = w

   
    
#
#print(w)
#print(loss)

