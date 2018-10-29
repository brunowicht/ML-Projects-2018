# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:18:46 2018

@author: Nicolas
"""
import numpy as np
import utils.cleaning as cln

def train(y,tx,w,k,gamma):
    """
    takes the labels y, data tx, initial weights w, number of iterations k and
        update rule gamma
    returns weights w after training
    """
    #initial x0 set to -1, else the equation of the classifier is forced to pass to center (0,0)
    x=-np.ones((tx.shape[1]+1,))

    for i in range(k):

        idx=i%len(y)
        x[1:]=tx[idx]
        #update rule
        w=w+gamma*(y[idx]-np.sign(x.dot(w)))*x
        w=w/np.sum(np.abs(w))
    return w

def online_expansion_3d(x):
    """
    this function allows to do a 3rd degree expansion online because of memory 
    management:
        suppose it takes 30 features, it will return exhaustive polynomials
        from degree 0 to 3, which is more the 5000
        it takes less memory than expansion on the whole batch, but is too
        time-consuming on our laptops
    """
    d2x=[]
    d3x=[]
    for i in range(len(x)):
        for j in range(i,len(x)):
            d2x.append(x[i]*x[j])
            
    for i in range(len(x)):
        for j in range(i,len(x)):
            for k in range(j,len(x)):
                d3x.append(x[i]*x[j]*x[k])
    
            
    return np.array([-1.]+list(x)+d2x+d3x)

def predictions(tx,w):
    """
    returns the  output with data tx and trained w, predicted by the simple
    perceptron
    """
    x=-np.ones((tx.shape[1]+1,))
    
    y_pred=np.zeros((tx.shape[0],))
    
    for i in range(tx.shape[0]):
        x[1:]=tx[i]
        y_pred[i]=np.sign(w.dot(x))
    return y_pred

def bagging(nb_experiment,data_y,data_x,w,k,gamma):
    """
    takes initial weights w, number of similar models to generate, k number of
        iterations and the update rate gamma
    returns averaged trained w
    NOTE: nb_experiment models are trained from scratched with the same hyper-
        parameters k and gamma, and same initial w
        
        each models are trained on a different ordered set (use of shuffle)
        
        One assumes that models are independant, have same expected accuracy,
        Averaging them will reduce the variance (=> less overfitting)
        more on https://en.wikipedia.org/wiki/Bootstrap_aggregating
    """
    init_w=w
    res_w=np.zeros((data_x.shape[1]+1,))
    for i in range(nb_experiment):
        data_y,data_x=cln.shuffle(data_y,data_x)
        res_w=res_w+train(data_y,data_x,init_w,k,gamma)
    res_w=res_w/nb_experiment
    return res_w
    
def bagging_t(data_y,data_x,w,k,gamma):
    """
    a fixed bagging for making it as a training method compliant to cross-
    validation
    """
    return bagging(10,data_y,data_x,w,k,gamma)
    

        