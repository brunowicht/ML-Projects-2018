import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import keras
from IPython.display import SVG
from keras.optimizers import Adam, SGD
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Load data as a dataframe
DATAPATH = "data/"
dataset = pd.read_csv(DATAPATH+"cleaned_data_train.csv", names="user_id,movie_id,prediction".split(','))

#make user_id and movie_id start from 0
dataset['user_id'] = dataset['user_id']-1
dataset['movie_id'] = dataset['movie_id']-1

# split data into train and test
train, test = train_test_split(dataset, test_size=0.1)

#get number of users and movies
n_users, n_movies = len(dataset.user_id.unique()), len(dataset.movie_id.unique())


#defin the latent factors
n_latent_factors_user = 5
n_latent_factors_movie = 8

#Movie input layer with dropout of 0.2
movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
movie_vec = keras.layers.Dropout(0.2)(movie_vec)

#User input layer with dropout of 0.2
user_input = keras.layers.Input(shape=[1],name='User')
user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input)
user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
user_vec = keras.layers.Dropout(0.2)(user_vec)

#Concatenate both input layers into one
concat = keras.layers.concatenate([movie_vec, user_vec])
concat_dropout = keras.layers.Dropout(0.2)(concat)

#Declare the hidden layers with 300, 150, 75 neurons and a 0.2 dropout for each of them
#Their activation function is linear by default
dense = keras.layers.Dense(300)(concat)
dense = keras.layers.Dropout(0.2)(dense)
dense_2 = keras.layers.Dense(150)(dense)
dense_2 = keras.layers.Dropout(0.2)(dense_2)
dense_3 = keras.layers.Dense(75)(dense_2)
dense_3 = keras.layers.Dropout(0.2)(dense_3)

# define output layer with activation relu. As we are looking for only one value, it has one neuron
result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_3)

#define optimizer
optimizer = SGD(lr=0.05)

#create the model and compile it
model = keras.Model([user_input, movie_input], result)
model.compile(optimizer=optimizer,loss= 'mean_squared_error')

#train the model over 50 epochs
history = model.fit([train.user_id, train.movie_id], train.prediction, epochs=50, verbose=1)


#compute y_hat and y_true and compute MSE
y_hat = np.round(model.predict([test.user_id, test.movie_id]),0)
y_hat[y_hat < 1] = 1
y_hat[y_hat > 5] = 5
y_true = test.prediction

print(mean_squared_error(y_true, y_hat))

#get label of needed prediction
dataset_to_predict = pd.read_csv(DATAPATH+"cleaned_sample.csv", names="user_id,movie_id,prediction".split(','))
dataset_to_predict ['user_id'] = dataset_to_predict ['user_id']-1
dataset_to_predict ['movie_id'] = dataset_to_predict ['movie_id']-1

#compute predictions
predictions = np.round(model.predict([dataset_to_predict.user_id, dataset_to_predict.movie_id]),0)
predictions[predictions < 1] = 1
predictions[predictions > 5] = 5
dataset_to_predict["prediction"] = predictions.astype(int)

#helper function to create submission
def create_submission():
    f = open(DATAPATH+"submission.csv", "w")
    f.write("Id,Prediction\n")
    for _,d in dataset_to_predict.iterrows():
        text = 'r'+str(d[0]+1)+'_c'+str(d[1]+1)+','+str(d[2])+'\n'
        f.write(text)
    f.close()

#create submission    
create_submission()

