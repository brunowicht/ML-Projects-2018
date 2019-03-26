
# coding: utf-8
## A file running a non negative matrix factorization on the data. 
## It requires to use keras. 


## Imports.. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import keras
from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from tensorflow.python.client import device_lib


## Loading the data. 
DATAPATH = "data/"
dataset = pd.read_csv(DATAPATH+"cleaned_data_train.csv", names="user_id,movie_id,prediction".split(','))
dataset['user_id'] = dataset['user_id']-1
dataset['movie_id'] = dataset['movie_id']-1
print("Dataset successfully loaded.")
print(dataset.head())

## split in train and test. 
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.1)

n_users, n_movies = len(dataset.user_id.unique()), len(dataset.movie_id.unique())
n_latent_factors = 3

print("Preparing a non negative matrix factorization with %d latent factors" % n_latent_factors)
# In[12]:


from keras.constraints import non_neg

## This part prepares the different layers. they are summed up afterwards. 


## input -> embedding -> flattening
movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding', embeddings_constraint=non_neg())(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

## input -> embedding -> flattening
user_input = keras.layers.Input(shape=[1],name='User')
user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input)
user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)

## Make the dot product. and compile
prod = keras.layers.dot([movie_vec, user_vec],axes=1)
model = keras.Model([user_input, movie_input], prod)
model.compile('adam', 'mean_squared_error')
# display a short summary. 
model.summary()

print("Model successfuly built. Now starting to train over 40 periods")
## train
history = model.fit([train.user_id, train.movie_id], train.prediction, epochs=40, verbose=1)

file_name = "keras_nnmf.png"
print("Training done. A figure of the training error over time can be found in "+file_name)
# In[22]:


pd.Series(history.history['loss']).plot()
plt.xlabel("Epoch")
plt.ylabel("Train Error")
plt.title("Train error over epochs")
plt.savefig(file_name)


## seing how the testing error is. 
y_hat = np.round(model.predict([test.user_id, test.movie_id]),0)
y_hat[y_hat < 1] = 1
y_hat[y_hat > 5] = 5
y_true = test.prediction
from sklearn.metrics import mean_absolute_error, mean_squared_error
mse = mean_squared_error(y_true, y_hat)
print("MSE on test set : %f " %mse)

print("Now predicting on the dataset...")
dataset_to_predict = pd.read_csv(DATAPATH+"cleaned_sample.csv", names="user_id,movie_id,prediction".split(','))
dataset_to_predict ['user_id'] = dataset_to_predict ['user_id']-1
dataset_to_predict ['movie_id'] = dataset_to_predict ['movie_id']-1

predictions = np.round(model.predict([dataset_to_predict.user_id, dataset_to_predict.movie_id]),0)
predictions[predictions < 1] = 1
predictions[predictions > 5] = 5

dataset_to_predict["prediction"] = predictions.astype(int)

sub_file = "submission2.csv"
dataset_to_predict.head()
print("Now creating a submission file under name "+sub_file)
# create a submission..

def create_submission():
    f = open(DATAPATH+sub_file, "w")
    f.write("Id,Prediction\n")
    for _,d in dataset_to_predict.iterrows():
        text = 'r'+str(d[0]+1)+'_c'+str(d[1]+1)+','+str(d[2])+'\n'
        f.write(text)
    f.close()


create_submission()
print("Done. ")
