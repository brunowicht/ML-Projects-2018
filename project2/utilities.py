# Imports..
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

#import keras
from IPython.display import SVG
#from keras.optimizers import Adam
#from keras.utils.vis_utils import model_to_dot

# from sklearn.linear_model import Ridge
DATAPATH = "data/"



def ALS(X,Y,matrix,label_positions,lambda_1=.01,lambda_2=0.1):
    """
	Implementation of ALS. Takes an input 
	x  is user*latent_feature matrix
	y is latent feature * movies. 
	matrix is a mtraix user*movies ( x*y ) 
	lamndas_1 and lambdas_2 are the hyper parameter for regularization. 
	
    """
    #fixing Y, optimizing X
    Xnew=X.copy()
    Ynew=Y.copy()
    n_feature=X.shape[1]
    I=np.diag(np.ones((n_feature,)))
	## first we optimize X 
    for u in range(X.shape[0]):
        ratings_user=matrix[u,:]
		# w shutdowns the loss of inexistent vectors. 
        W_user=np.diag(label_positions[u,:].squeeze())
        A=np.linalg.inv(Y.dot(W_user).dot(Y.T)+lambda_1*I)
        Xnew[u,:]=ratings_user.dot(W_user).dot(Y.T).dot(A.T)
        
        if u%100==0:
            print(str(u/100) +'per2cent done')
	## then we optimize Y 
    for m in range(Y.shape[1]):
        ratings_movie=matrix[:,m]
        W_movie=np.diag(label_positions[:,m].squeeze())
        A=np.linalg.inv(Xnew.T.dot(W_movie).dot(Xnew)+lambda_2*I)
        Ynew[:,m]=A.dot(Xnew.T).dot(W_movie).dot(ratings_movie)
        if m%10==0:
            print(str(m/10)+'per2cent done')
	## return the optimized X and Y 
    return Xnew,Ynew


def create_submission(predictors,name):
	## create a submission file with given predictors save it under the name given. 
    f = open(DATAPATH+name, "w")
    f.write("Id,Prediction\n")
    for _,d in predictors.iterrows():
        text = 'r'+str(int(d[0]+1))+'_c'+str(int(d[1]+1))+','+str(int(d[2]))+'\n'
        f.write(text)
    f.close()


def compute_error(X,Y,matrix,label_pos):
	## compute error for matrices. 
    return np.sum((label_pos * (matrix - np.dot(X, Y)))**2)/np.sum(label_pos)




