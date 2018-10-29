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



PATH = "../../"
# Load clean and standardize the data
train_file = PATH+"train.csv"
test_file = PATH+"test.csv"
data_x, data_y, train_ids, means, stds = load_clean_standardize_train(train_file)
best_gamma = 1.0e-06
filename = "ave_maria.csv"

print("Done loading data and cleaning it. \n Starting perceptron with gamma = %f" % best_gamma)



##1e-6 is the best one for sgd

final_w=prc.bagging(50,data_y,data_x,np.array([.0]*(data_x.shape[1]+1)),250000,best_gamma)
print("Found features for the model. \n Now starting to compute selection of best features")


## Best selection for now : 450 

### We now select the best features according to their weight. We found 450 to be a good zone. 
selection = 450
liste=list(np.arange(len(final_w)))

#list the features from the less important to the most important
liste.sort(key= lambda i : np.abs(final_w[i]))
liste=np.array(liste)-1
liste=liste[liste>=0] #excluding the -1 than can occur

reduced_data=data_x[:,liste[selection:]]
final_w_2=prc.bagging(100,data_y,reduced_data,np.array([.0]*(data_x.shape[1]+1-selection)),250000,best_gamma)
print("Found most heavy features\n Now computing final accuracy on training set")
print("Final accuracy of the perceptron on the train set : %f " % np.sum(np.heaviside(data_y*prc.predictions(reduced_data,final_w_2),.5))/len(data_y))

print("Using model on test dataset.")
test_x, test_y, test_ids = load_clean_standardize_test(test_file, means, stds)
test_x_r =test_x[:,liste[selection:]]
print("Predicting values with model")
y_pred=prc.predictions(test_x_r,final_w_2)
print("Creating submission..under "+filename)
create_csv_submission(test_ids, y_pred, filename)

print("Done.")

