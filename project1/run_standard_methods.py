# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:38:51 2018
This file runs the standard methods we have seen in class. 
It does not produce the best results. But we have used it as a starting point. 


@author: bruno
"""

from utils.cleaning import *
from utils.implementations import * 
from utils.proj1_helpers import * 

import matplotlib.pyplot as plt




import numpy as np
import pandas as pd
import seaborn as sns



# Load clean and standardize the data
#data_x, data_y, train_ids, means, stds = load_clean_standardize_train("../../train.csv",expansion=False, replace=False )
#print("Cleaning done")
#print(data_x.shape)

#do the least square method .




def do_standard_run(data_y,data_x):
    """
    This does a standard run of the normal method and prints the losses and returns the min loss and best w. 
    We can pass different data_y and data_x with more or less features. 
    Be aware that using data_x and data_y that has been expanded can lead to overflow computations. 
    """
    
    ws = []
    losses = []
    accuracies = []
    w_ls, loss_ls = least_squares(data_y,data_x)
    ws.append(w_ls)
    losses.append(loss_ls)
    accuracies.append(compute_accuracy(data_y,predict_labels(w_ls,data_x)))
    print("Least square method : %f loss" % loss_ls)
    initial_w = np.zeros(data_x.shape[1])
    gamma = np.linspace(0,1,20)
    loss_ls_gd = 100000
    w_ls_gd = np.zeros(data_x.shape[1])
    for gamma_ in gamma:
        w_Curr, loss_Curr = least_squares_GD(data_y,data_x,initial_w,100,gamma_)
        if loss_Curr < loss_ls_gd: 
            w_Curr = w_ls_gd
            loss_ls_gd = loss_Curr
        
    
    
    print("Least square Gradient descent method : %f loss" % loss_ls_gd)
    ws.append(w_ls_gd)
    losses.append(loss_ls_gd)
    w_ls_sgd = np.zeros(data_x.shape[1])
    accuracies.append(compute_accuracy(data_y,predict_labels(w_ls_gd,data_x)))
    loss_ls_sgd = 1000
    for gamma_ in gamma:
        w_Curr, loss_Curr = least_squares_SGD(data_y,data_x,initial_w,100,gamma_)
        if loss_Curr < loss_ls_sgd: 
            w_ls_sgd = w_Curr
            loss_ls_gd = loss_Curr
    
    
    print("Least square stochastic Gradient descent method (batch size = 1 ): %f loss" % loss_ls_gd)
    ws.append(w_ls_sgd)
    losses.append(loss_ls_sgd)
    
    lambdas = np.linspace(0,1,20)
    w_rr = np.zeros(data_x.shape[1])
    
    accuracies.append(compute_accuracy(data_y,predict_labels(w_ls_sgd,data_x)))
    loss_rr = 10000
    for lambda_ in lambdas:
        w_curr, loss_curr = ridge_regression(data_y,data_x,lambda_)
        if loss_curr < loss_rr:
            w_rr = w_curr
            loss_rr = loss_curr
        
    print("Ridge regression : %f loss" % loss_rr)
    ws.append(w_rr)
    losses.append(loss_rr)
    
    accuracies.append(compute_accuracy(data_y,predict_labels(w_rr,data_x)))
    w_lr = np.zeros(data_x.shape[1])
    loss_lr = 10000
    for gamma_ in gamma:
        w_curr , loss_currr = logistic_regression(data_y,data_x,initial_w,100,gamma_)
        if loss_curr < loss_lr:
            w_lr = w_curr
            loss_lr = loss_curr
    
    
    print("Logistic regression : %f loss "%loss_lr)
    ws.append(w_lr)
    losses.append(loss_lr)
    
    accuracies.append(compute_accuracy(data_y,predict_labels(w_lr,data_x)))
    w_rlr = np.zeros(data_x.shape[1])
    loss_rlr = 1000
    for gamma_ in gamma:
        for lambda_ in lambdas:
            w_curr , loss_currr = reg_logistic_regression(data_y,data_x,lambda_,initial_w,100,gamma_)
            if loss_curr < loss_rlr:
                w_rlr = w_curr
                loss_rlr = loss_curr
    
    print("Logisitc regressions regularized : %f loss " % loss_rlr)
    ws.append(w_rlr)
    losses.append(loss_rlr)
    
    accuracies.append(compute_accuracy(data_y,predict_labels(w_rlr,data_x)))
    
   
    return ws,losses,accuracies  
#    return ws[np.where(np.min(losses)==losses)[0][0]], losses[np.where(np.min(losses)==losses)[0][0]], accuracies[np.where(np.min(losses)==losses)[0][0]]


w, loss,accuracy = do_standard_run(data_y,data_x)





def draw_plot_save(df,filename, hue=False):
    """
    Draws a plot of the dataframe and all their features. 
    Carefull it takes a lot of time if there is a lot of features ( > 10 )
    """
    if hue:
        plot = sns.pairplot(df,hue='y')
        print("plot done ")
        plot.savefig(filename)
        print("Finished saving plot")
    else:
        plot = sns.pairplot(df)
        print("plot done ")
        plot.savefig(filename)
        print("Finished saving plot")