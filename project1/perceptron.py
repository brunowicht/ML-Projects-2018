# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:18:46 2018

@author: Nicolas
"""
import numpy as np
import utils.cleaning as cln

def train(y,tx,w,k,gamma):
    x=-np.ones((496,))

    for i in range(k):
#        if i > 249000 and i< 250100:
#            print(i%len(y))
        idx=i%len(y)
#        x=online_expansion_3d(tx[idx])
        x[1:]=tx[i]
        w=w+gamma*(y[idx]-np.sign(x.dot(w)))*x
        #w L1 normalization
        w=w/np.sum(np.abs(w))
    return w

def train_minibatch(y,tx,w,k,gamma,batch_size):
    regressor=-np.ones((batch_size,tx.shape[1]+1))
    for i in range(k):
        indices=np.random.choice(np.arange(len(y)),size=batch_size,replace=False)
        regressor[:,1:]=tx[indices,:]
        delta=regressor.T.dot(np.sign(regressor.dot(w))-y[indices])/batch_size
        w=w-gamma*delta
        #w L1 normalization
        w=w/np.sum(np.abs(w))
    return w
    

#bad idea : highly time-consuming
def online_expansion_3d(x):
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
    x=-np.ones((496,))
    
    y_pred=np.zeros((tx.shape[0],))
    
    for i in range(tx.shape[0]):
#        x=online_expansion_3d(tx[i])
        x[1:]=tx[i]
        y_pred[i]=np.sign(w.dot(x))
    return y_pred

def bagging(nb_experiment,data_y,data_x,w,k,gamma):
    init_w=w
    res_w=np.zeros((data_x.shape[1]+1,))
    for i in range(nb_experiment):
        print(i)
        data_y,data_x=cln.shuffle(data_y,data_x)
        res_w=res_w+train(data_y,data_x,init_w,k,gamma)
    res_w=res_w/nb_experiment
    return res_w
    
    

        