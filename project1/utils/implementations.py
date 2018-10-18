# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:51:24 2018

@author: bruno
"""

"""
In this file we will put our implementations of ML methods
"""



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
    
    w = np.linalg.lstsq(tx,y,rcond=None)[0]
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