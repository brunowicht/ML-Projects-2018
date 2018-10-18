# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

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
    for i in range(max_iter):
        delta=tx.T.dot(derivative_of_cross_entropy_error(y,activations))/len(y)
        w=w-gamma*delta
        activations=tx.dot(w)
    return w

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iter,gamma):
    """
    Performs logistic regression like logistic_regression, but with a ridge regression factor
    """
    activations=tx.dot(initial_w)
    w=initial_w
    for i in range(max_iter):
        delta=tx.T.dot(derivative_of_cross_entropy_error(y,activations))/len(y)+2*lambda_*w
        w=w-gamma*delta
        activations=tx.dot(w)
    return w