# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:51:24 2018

@author: bruno
"""

import numpy as np

import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient(y,tx,w):
    e = y - tx.dot(w)
    return -tx.T.dot(e)/len(tx)


def least_squares_GD(y,tx,initial_w,max_iters,gamma):
    """
    
    """
    loss = 100000
    w = initial_w
    for i in range(max_iters):
        # compute 
        loss = compute_loss(y,tx,w)
        grad = compute_gradient(y,tx,w)
        w = w - gamma * grad
        
    ## we only care about the final result which will be the most precise
    return w,loss



def least_squares_SGD(y,tx,initial_w,max_iters,gamma):
    
    w = initial_w
    loss = 10000
    for i in range(max_iters):
        # take one random sampling point
        n = np.random.choice(len(tx),1)
        xn = tx[n]
        yn = y[n]
        loss = compute_loss(y,tx,w)
        grad = compute_gradient(yn,xn,w)
        w = w - gamma * grad
        
    return w , loss
        
def least_squares(y,tx):
    
    w = np.linalg.lstsq(tx,y)[0]
    loss = compute_loss(y,tx,w)
    return w , loss
    
def ridge_regression(y,tx,lambda_):
    ## l is mse 
    ## we want (xtx + lambda*2N * I )^-1* xTy
    xty = tx.T.dot(y)
    xtx = tx.T.dot(tx)
    lambdaI = lambda_*2*len(y) * np.identity(tx.shape[1])
    w = np.linalg.inv(xtx + lambdaI).dot(xty)
    loss = compute_loss(y,tx,w)
    return w, loss


def sigmoid(activation):
    """
    takes activation(should look like tw.dot(x))
    returns the sigmoid of the activation
    
    used for logisitic regression
    """
    return 1./(1+np.exp(-activation))


def derivative_of_cross_entropy_error(y,activation):
    """
    takes observed y, and activation (which is tx dot weights)
    returns the derivative of categorical cross entropy cross entropy
    NOTE: the categorical cross entropy has a theoretical basis
          for logistic regression (least squares has none for this regression)
          more on that on https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression
    """
    sigma=sigmoid(activation)
    return(-y*(1-sigma)+(1-y)*sigma)

def logistic_regression(y,tx,initial_w,max_iter,gamma):
    """
    takes observed y, regressors tx, initialized parameters initial_w,
          max number of iterations max_iter and gamma
          for gradient descent
    returns w, the optimal parameters for logistic regression, approximated by gradient descent
    """
    activations=tx.dot(initial_w)
    w=initial_w

    loss = 10000
    for i in range(max_iter):
        delta=tx.T.dot(derivative_of_cross_entropy_error(y,activations))/len(y)
        w=w-gamma*delta
        activations=tx.dot(w)
        loss = compute_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iter,gamma):
    """
    Performs logistic regression like logistic_regression, but with a ridge regression factor
    """
    activations=tx.dot(initial_w)
    w=initial_w
    loss = 10000
    for i in range(max_iter):
        delta=tx.T.dot(derivative_of_cross_entropy_error(y,activations))/len(y)+2*lambda_*w

        w=w-gamma*delta
        activations=tx.dot(w)
        loss = compute_loss(y,tx,w)
    return w, loss


def equipartition(data_y,proportion,fold_idx):
    """
    given the data_y, the proportion of data split and the  episode
    returns bool mask for the split 
    """
    #fold_idx starts at one
    nb_bins=int(1./proportion)

    #find the position where the label 1 stands
    pos_1= np.where(data_y==1)[0]
    size_1=np.sum(data_y==1)
    #find the position  where the label 0 / -1 stands
    pos_0=np.where(data_y !=1)[0]
    size_0=np.sum(data_y !=1)
    
    size=len(data_y)
    per_bins_1=int(np.ceil(size_1*proportion))
    per_bins_0=int(np.ceil(size_0*proportion))
    print(fold_idx)
    if fold_idx <nb_bins-1:
        test_pos_1=pos_1[fold_idx*per_bins_1:(fold_idx+1)*per_bins_1]
        test_pos_0=pos_0[fold_idx*per_bins_0:(fold_idx+1)*per_bins_0]
    else:
        print('coucou')
        test_pos_1=pos_1[fold_idx*per_bins_1:]
        test_pos_0=pos_0[fold_idx*per_bins_0:]
    test_mask=np.array([False]*size)
    #setting the position of label 1
    test_mask[test_pos_1]=True
    #setting the position of the other label
    test_mask[test_pos_0]=True
    train_mask=np.logical_not(test_mask)
    return train_mask,test_mask

def one_fold_validation(model,data_y,data_x,lambda_,initial_w,max_iter,gamma,train_mask,test_mask):


    train_x=data_x[train_mask,:]
    train_y=data_y[train_mask]
    test_x=data_x[test_mask,:]
#        train_x,test_x=fill_unknown_with_column_mean_train_test(train_x, test_x)
#        train_x,mu_x,sigma_x=normal_train(train_x)
#        train_y,mu_y,sigma_y=normal_train(data_y[train_mask])

    test_y=data_y[test_mask]
#        test_x=normal_test(test_x,mu_x,sigma_x)
#        test_y=normal_test(test_y,mu_y,sigma_y)
    w,train_loss=model(train_y,train_x,lambda_,initial_w, max_iter,gamma)
    test_loss=compute_loss(test_y,test_x,w)
    return w,train_loss,test_loss
    


def cross_validation(model,data_y,data_x,lambda_,initial_w,max_iter,gamma,proportion):
    nb_bins=int(1./proportion)
    size=len(data_y)
    train_loss=np.zeros(nb_bins)
    test_loss=np.zeros(nb_bins)
    for i in range(nb_bins):

        train_mask,test_mask=equipartition(data_y,proportion,i)
        w,train_loss[i],test_loss[i]=one_fold_validation(model,data_y,data_x,lambda_,initial_w,max_iter,gamma,train_mask,test_mask)
        print(test_loss[i])
    print(test_mask)


    return w,np.mean(train_loss), np.mean(test_loss),np.std(train_loss),np.std(test_loss)

def normal_train(x):
    mu=x.mean(axis=0)
    std=x.std(axis=0)
    return (x-mu)/(std+1e-17),mu,std
def  normal_test(x,mu,std):
    return (x-mu)/(std+1e-17)

def next_feature(model,data_y,data_x,lambda_,initial_w,max_iter,gamma,proportion,fixed_features):
    nb_features=data_x.shape[1]
    test_error=[1000]*nb_features
    for i in range(nb_features):
        if i not in fixed_features:
            features=fixed_features+[i]
            regressor=np.ones((len(data_y),len(fixed_features)+1))
            regressor[:,1:]=data_x[:,np.array(features)]
            a_,b_,test_error[i]=cross_validation(model,data_y,regressor,lambda_,initial_w,max_iter,gamma,proportion)
    
            



