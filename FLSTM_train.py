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
from keras.utils import plot_model

#from keras.utils import plot_model
#https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
#https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
#https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
#

path = "data/*.gif"
processed_folder =  "processeddata/"
processed_path = "processeddata/*.jpg"
model_path  = "model/"
model_name = "model_flstm"
n_inputs = 10
n_outputs = 7
n_components = 2048

batchsize = 4
epochs = 3
#print(processed_path)
#gif_2_jpg( path, processed_folder)   convert gif to jpg

if os.path.isfile('train_X.dat') :
	train_X= load_array('train_X.dat')
	train_y= load_array('train_y.dat')
else:
	if os.path.isfile('gray_reduced_2048.dict') :
		imagelist, features = load_gray_data(pca_features = True)
	else:
		imagelist,features = gray_imag( processed_path, n_components , pca_features = True)
	train_X , train_y = lstm_data(imagelist, features, n_inputs, n_outputs)

if not os.path.exists(model_path):
    os.makedirs(model_path)
#imagelist,features,features_reduced = gray_imag( processed_path )
#https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
#print(reframed.head())
#print("start_time_all")
#sys.exit()
print("[STATUS] data downloaded !")

start_time_all = time.time()
print("training X shape:",train_X.shape)
print("training y shape:", train_y.shape)

input_shape = ( train_X.shape[1], train_X.shape[2] )
D = train_X.shape[2]#feature dimension

# define baseline model
#The same weights can be used to output each time step in the output sequence by wrapping the Dense layer in a TimeDistributed wrapper.
def baseline_model():
	#Build a cnn + lstm model
	model = Sequential()
	model.add(LSTM(100, activation='relu', input_shape = input_shape))  #input_shape: (N,T,D) ,   n_neurons:50  # activation='tanh',
	model.add(RepeatVector(n_outputs))  #This layer simply repeats the provided 2D input multiple times to create a 3D output.
	model.add(LSTM(30,  activation='relu',return_sequences=True))  # n_neurons:100
	#model.add(Lambda(lambda x: x[:, -n_inputs:, :]))
	model.add(TimeDistributed(Dense(D)))  # multivariate outputs

	model.compile(optimizer='adam', loss='mse')
	return model

print("[STATUS] Start training !")
model = baseline_model()
model.fit( train_X , train_y , epochs = epochs, batch_size = batchsize, verbose=0)  #np.concatenate((a, b))

model_json = model.to_json()
with open(model_path + model_name+ ".json", "w") as json_file:
 json_file.write(model_json)
# save weights
model.save_weights(model_path + model_name+ ".h5")
print("[STATUS] saved model and weights to disk..")
elapsed_time = time.time() - start_time_all
print("elapsed_time:",elapsed_time)
#plot_model(model, show_shapes=True,to_file='FLSTM_model.png')
