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
#data_x, data_y, train_ids, means, stds = load_clean_standardize_train("../../train.csv")
#print("Cleaning done")
#print(data_x.shape)
#best_acc=-1.
#best_gamma=10.
#best_w=np.array([0.]*496)
#for i in np.linspace(0.815e-07,1.5e-06,21):
#
#    w=prc.train(data_y,data_x,np.array([0.]*903),250000,i)
#    print("training done ")
#    y_hat=prc.predictions(data_x[:250000],w)
#    matches=np.heaviside(y_hat*data_y,.5)
#    
#    acc=np.sum(matches)/250000
#    print(acc)
#    print(i)
#    if acc > best_acc:
#        best_acc=acc
#        best_gamma=i
#        best_w=w
#print(best_acc)
#print(best_gamma)
best_gamma = 1.0e-06


##1e-6 is the best one for sgd
#w=prc.train(data_y,data_x,np.array([0.]*(data_x.shape[1]+1)),1500,1e-6)
#final_w=prc.bagging(50,data_y,data_x,np.array([.0]*(data_x.shape[1]+1)),250000,best_gamma)
#print(np.sum(np.heaviside(data_y*prc.predictions(data_x[:250000],final_w),.5))/len(data_y))


#minus 1 is because of the initial 0

## Best selection for now : 450 
#for i in range(440,550,20):
#print("Starting to select only : %d best values" % i)
#liste=list(np.arange(len(final_w)))
##list the features from the less important to the most important
#liste.sort(key= lambda i : np.abs(final_w[i]))
#liste=np.array(liste)-1
#liste=liste[liste>=0] #excluding the -1 than can occur
#selection = 450
#reduced_data=data_x[:,liste[selection:]]
#final_w_2=prc.bagging(100,data_y,reduced_data,np.array([.0]*(data_x.shape[1]+1-selection)),250000,best_gamma)
#print(np.sum(np.heaviside(data_y*prc.predictions(reduced_data,final_w_2),.5))/len(data_y))

#
#print(final_w_2.shape())
#print(test_x.shape())
#test_x, test_y, test_ids = load_clean_standardize_test("../../test.csv", means, stds)
test_x_r =test_x[:,liste[selection:]]

y_pred=prc.predictions(test_x_r,final_w_2)

create_csv_submission(test_ids, y_pred, "carpe_diem.csv")