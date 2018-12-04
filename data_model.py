from utils import *
from batch_data import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Lambda,  AveragePooling3D, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import model_from_json
import numpy as np,sys
import os
from keras.models import model_from_json
from sklearn.metrics.pairwise import pairwise_distances
import time
import matplotlib.pyplot as plt

##**********************************************************************************************************************--------------------------------------------------------------------------
#1. the first step involves converting gif image to jpg image
original_path ="data_download/*.gif"  #original downloaded precipitation maps
processed_folder =  "processeddata/" # convert gif image to jpg image
gif_2_jpg( original_path, processed_folder)
#**********************************************************************************************************************#--------------------------------------------------------------------------
#2. the second step is to read image data and crop uncessary text in the image, and then build
# a dictionary to store image_id : 3d_image_data
img_shape = 128  #my laptop has only 8GB RAM, depending on RAM, image shape is flexible
img_shape2 = 250
img_shape3 = 360
n_inputs = 7   # the number of prior rainfall maps
n_outputs = 7 # forecasting for the next 7 days
processed_path = "processeddata/*.jpg"


def color_imag(processed_path, resize=False):
	features = dict()
	imagelist = []
	for name in glob.glob(processed_path):
		flbase = os.path.basename(name)
		# load an image from file
		img_data  = scipy.misc.imread(name, flatten=False, mode='RGB') # cv2.imread(name)
		img_data = img_data[20:270,40:,:] #.astype(np.float32)  #crop the unnecessary texts  ( 250, 360, 3)
		#print( img_data.shape)
		#sys.exit()
		if resize == True:
			img_data = cv2.resize(img_data, (img_shape, img_shape))
		#img_data = img_data.reshape([img_data.shape[0], img_data.shape[1],3])
		#print(feature)
		#print(feature.shape, type( feature[0][0] ))
		#sys.exit()
		# get image id
		image_id = flbase.split('.')[0]
		#print(image_id, feature.shape )
		# store feature
		imagelist.append(image_id)
		#print(np.squeeze(img_data), img_data )
		#3sys.exit()

		features[image_id] = img_data# np.squeeze(feature)

	#values = np.array(list(features.values()))/255   #normalization image data
	#values -= np.mean(values, axis=0)
	#values /= np.std(values, axis = 0)
	#imagelist = list(features.keys())
	#values_reduced = pca(values, n_components)  # pca feature reduction
	#print(values_reduced.shape)
	#features = dict(zip(features.keys(), values))

	saveFile("color.dict",features)
	#saveFile("features_reduced_1024.dict",features_reduced)
	saveFile("color_imagelist.csv",imagelist)

	return imagelist,features
#**********************************************************************************************************************#--------------------------------------------------------------------------
#3. the 3rd step is to change unsupervised problem to supervised problem.
def convlstm_data(imagelist, n_inputs, n_outputs =7 ):
    reframed = series_to_supervised(imagelist,n_inputs,n_outputs)  #change unsupervised problem to supervised problem
    indices  = reframed.values  #pandas data to a matrix
    m,n = indices.shape
    #print( indices)
    #X_data , y_data  = load_convlstm_values(features, indices[:,:-n_outputs]  ), load_convlstm_values(features, indices[:, -n_outputs:] )
    num_training = int(m*0.67)
    num_val = m - num_training
    #n_inputs = n-n_outputs
    train_idx = indices[:num_training,:]
    train_X_idx , train_y_idx  = train_idx[:, :-n_outputs], train_idx[:, -n_outputs:]
    val_idx  = indices[num_training:,:]  #get x,y val examples
    val_X_idx , val_y_idx  = val_idx[:, :-n_outputs], val_idx[:, -n_outputs:]

    train_X_idx = train_X_idx.tolist()
    train_y_idx = train_y_idx.tolist()

    val_X_idx = val_X_idx.tolist()
    val_y_idx = val_y_idx.tolist()

    saveFile('color_val_x_index.csv',val_X_idx)
    saveFile('color_val_y_index.csv',val_y_idx)

    return (train_X_idx , train_y_idx, val_X_idx , val_y_idx )

imagelist,features =   color_imag(processed_path) #  load_color_data() #
#imagelist,features =   load_color_data()
print("data loaded")
train_X_idx , train_y_idx, val_X_idx , val_y_idx =  convlstm_data(imagelist,  n_inputs, n_outputs)
#print(train_X_idx.shape )
#train_X , train_y = load_array("train_X.dat")  , load_array("train_y.dat")
#print(train_X.__len__())

#print("train val data splited")
#sys.exit()
#***********************************************************************************************************************************
#4. the 4th step is to build a convolutional LSTM (ConvLSTM) network to preserve all the spatial information
#reference: https://github.com/sxjscience/HKO-7  the author of the papaer  Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model

def convlstm(img_shape= 128 ):

	model = Sequential()
	#model.add(AveragePooling3D(pool_size=(1, 4, 4),input_shape=(n_inputs, img_shape, img_shape, 3),padding='same'))
	#model.add(BatchNormalization())

	model.add(ConvLSTM2D(filters = 32, kernel_size=(3, 3), input_shape=(n_inputs, img_shape2, img_shape3, 3), padding='same', return_sequences=True))  #input_shape=(None, 32, 32, 3),
	#model.add(ConvLSTM2D(filters = 32, kernel_size=(3, 3), input_shape=(n_inputs, img_shape, img_shape, 3), padding='same', return_sequences=True))  #input_shape=(None, 32, 32, 3),
	model.add(BatchNormalization())
	model.add(Dropout(0.25))


	model.add(ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences=True))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(TimeDistributed(Conv3D(filters = 3, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last')))
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.96, amsgrad=False)

	model.compile(loss = 'mse', optimizer = adam)

	return model

model = convlstm()
print(model.summary())

#print(train_X_idx[], type( train_X_idx) )
#sys.exit()
#**********************************************************************************************************************
#5. train the model by reading the data on the fly,
model.fit_generator( generator=DataGenerator(len(train_X_idx ), n_inputs, n_outputs, train_X_idx, train_y_idx, features, batch_size = 1) )
print( "finished training")


model_path  = "model/"
model_name = " model_convlstm "

model_json = model.to_json()
with open(model_path + model_name+ ".json", "w") as json_file:
 json_file.write(model_json)
# save weights
model.save_weights(model_path + model_name+ ".h5")
print("[STATUS] saved model and weights to disk..")

sys.exit()
#**********************************************************************************************************************
#6. Evaluation  the model, the general idea is to utilize cosine similarity
# this part is not provided.

json_file = open(model_path+ model_name+ ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_path + model_name+ ".h5")
# demonstrate prediction

features = loadFile('color.dict')
#val_X , val_y = load_array("val_X.dat"),load_array("val_y.dat")
#vectors = np.array(list( features.values()))
val_y_index = loadFile('color_val_y_index.csv')
#val_y_index = val_y_idx.tolist()
#n_vocab  = len(features)
n_vals = len(val_y_index)

#print(n_vals )
vectors_matrix = np.array(list(features.values()))

print(vectors_matrix.shape, type( vectors_matrix))
#vectors_matrix = vectors_matrix.reshape(vectors_matrix.shape[0],-1  )
#print(vectors_matrix.shape, type( vectors_matrix))
#sys.exit()

val_X =  load_array("val_X.dat")
yhat = model.predict(val_X, batch_size = 8 , verbose=0)
print("prediction finished")
for i in range(n_vals ):
    #yhat[i] = yhat[i][:,:,]
    #yhat[i] = np.array( yhat[i])
    #print( yhat[i].shape,  type(yhat[i] ) )
    y=np.zeros(yhat[i].shape[:3] )
    for idx,j in enumerate(yhat[i]):
        #print(j.shape, np.squeeze(j).shape)
        y[idx,:,:]= np.squeeze(j)
        # the following code is not provided
		print("")
