# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:38:51 2018

@author: bruno
"""

from utils.cleaning import *
import utils.implementations as impl
import utils.proj1_helpers



import numpy as np
import pandas as pd
import seaborn as sns

# Load clean and standardize the data
#data_x, data_y, ids <= load_raw("../../train.csv",rand=True)
#df = pd.DataFrame(data_x)
#
#df["y"] = data_y
#print("Done loading df")


#sns.countplot(x=0,hue='y',data=df)

plot = sns.pairplot(df,hue='y')
print("plot done ")
plot.savefig("pairplot_hue.png")
print("Finished saving plot")
#print(data_x)
#print(data_y)
#
#lambda_ = 0.0001
#max_iter =50 
#proportion=0.2
#gamma = 0.5
#model = impl.reg_logistic_regression
#w_init = np.zeros(data_x.shape[1])

#
#w,m_train_loss, m_test_loss,std_train_loss,std_test_loss=impl.cross_validation(model,data_y,data_x,lambda_,w_init,max_iter,gamma,proportion,raw_data=True)
#print(w)
#print(m_train_loss)
#print(m_test_loss)
#
#clean_data_x, clean_data_y, clean_ids, means, stds = load_clean_standardize_train("../../data.csv")


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



#print(impl.least_squares(data_y,data_x))

#best_w = w_init
#best_loss = 1000.0
#gammas = np.linspace(0.0015,0.003,11) #² opt : 0.00195


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
#lambdas = np.linspace(1,1.5,5)
#for gamma in gammas:
#gamma = 0.00255
#for lambda_ in lambdas:
#    w, loss = impl.reg_logistic_regression(data_y,data_x,lambda_,best_w,50,gamma)
#    print("Gamma : "+str(gamma)+ ", Lamnda : "+ str(lambda_)+" Curr loss : " + str(loss))
#    if loss < best_loss:
#        print("New best loss : " + str(loss))
#        best_loss = loss
#        best_w = w

   
    
#
#print(w)
#print(loss)

