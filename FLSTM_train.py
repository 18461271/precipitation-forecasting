import os, time, sys
import numpy as np
from utils import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Lambda
from keras.models import model_from_json

#https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
#https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
#https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
#

path = "data/*.gif"
processed_folder =  "processeddata/"
processed_path = "processeddata/*.jpg"

n_inputs = 100
n_outputs = 7
n_components = 2048

batchsize = 8
epochs = 30
#print(processed_path)
#gif_2_jpg( path, processed_folder)   convert gif to jpg

if os.path.isfile('gray_reduced_2048.dict') :
	imagelist, features = load_gray_data(pca_features = True)
else:
	imagelist,features = gray_imag( processed_path, n_components , pca_features = True)

if not os.path.exists(model_path):
    os.makedirs(model_path)
#imagelist,features,features_reduced = gray_imag( processed_path )
#https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
#print(reframed.head())

X_data , y_data , _ = lstm_data(imagelist, features, n_inputs, n_outputs)

start_time_all = time.time()

print(X_data.shape, y_data.shape)

input_shape = ( X_data.shape[1], X_data.shape[2] )
D = X_data.shape[2]#number of features

# define baseline model
def baseline_model():
	#Build a cnn + lstm model
	model = Sequential()
	model.add(LSTM(100, activation='tanh', input_shape = input_shape))  #input_shape: (N,T,D) ,   n_neurons:50  # activation='relu',
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(30,  activation='tanh',return_sequences=True))  # n_neurons:100
	#model.add(Lambda(lambda x: x[:, -n_inputs:, :]))
	model.add(TimeDistributed(Dense(D)))  # multivariate outputs
	model.compile(optimizer='adam', loss='mse')
	return model


model = baseline_model()
model.fit( X_data , y_data , epochs = epochs, batch_size = batchsize, verbose=0)  #np.concatenate((a, b))


model_path  = "model/"
model_name = " model_mlp "


model_json = model.to_json()
with open(model_path + model_name+ ".json", "w") as json_file:
 json_file.write(model_json)
# save weights
model.save_weights(model_path + model_name+ ".h5")
print("[STATUS] saved model and weights to disk..")
elapsed_time = time.time() - start_time_all
print("elapsed_time:",elapsed_time)
