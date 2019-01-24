import pickle
import sys
import os
import glob
#import cv2
import math
import pickle
import datetime
import pandas as pd
from pandas import DataFrame
from pandas import concat
from PIL import Image
import numpy as np
import scipy.misc
from PIL import Image
from sklearn.decomposition import PCA

import bcolz
#from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import plot_model

#n_inputs = 30
#n_outputs = 7
processed_folder =  "processeddata/"
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def saveFile(fileName, object):
	with open(fileName,'wb') as f:
 	   pickle.dump(object,f)

def loadFile(fileName):
	fileObject = open(fileName,'rb')
	return pickle.load(fileObject,encoding='bytes')



def load_data(pca_features = True):
	if pca_features:
		features = loadFile('features_reduced_1024.dict' )
	else:
		features = loadFile('features.dict' )
	imagelist = loadFile('imagelist.csv')
	return imagelist, features

def load_gray_data(pca_features = True):
	if pca_features:
		features = loadFile('gray_reduced_2048.dict' )
	else:
		features = loadFile('gray.dict' )
	imagelist = loadFile('gray_imagelist.csv')
	return imagelist, features

def load_color_data():

	features = loadFile('color.dict' )
	imagelist = loadFile('color_imagelist.csv')
	return imagelist, features


def gif_2_jpg( original_path, processed_folder):
    for idx,name in enumerate(glob.glob(original_path)):
        flbase = os.path.basename(name)
        new_name = name[:-4]+'.jpg'
        img_data  = Image.open(name).convert('RGB').save(processed_folder + new_name)

    return 1

"""def gif_2_jpg( path, processed_folder):
    for idx,name in enumerate(glob.glob(path)):
        flbase = os.path.basename(name)
        new_name = name[:-4]+'.jpg'
        img_data  = Image.open(name).convert('RGB').save(processed_folder + new_name)
    return 1"""

from sklearn.feature_selection import VarianceThreshold

def gray_imag(processed_path, n_components = 4096, pca_features = True):
	features = dict()
	imagelist = []
	for name in glob.glob(processed_path):
		flbase = os.path.basename(name)
		# load an image from file
		img_data  = scipy.misc.imread(name, flatten=False, mode='L') # cv2.imread(name)#  cv2.imread(name,0)
		img_data = img_data[20:270,40:]  #crop the uncessary texts
		feature = img_data.reshape( 1, -1).astype(np.float32)
		#print(feature)
		#print(feature.shape, type( feature[0][0] ))
		#sys.exit()
		# get image id
		image_id = flbase.split('.')[0]
		#print(image_id, feature.shape )
		# store feature
		imagelist.append(image_id)

		features[image_id] = np.squeeze(feature)

	if pca_features == False:
		saveFile("gray.dict",features)
			#save_array("features_reduced_1024.dict",features_reduced)
		saveFile("gray_imagelist.csv",imagelist)
		#return imagelist,features
	else :
		values = np.array(list(features.values()))/255
		values -= np.mean(values, axis=0)
		#values /= np.std(values, axis = 0)
		#imagelist = list(features.keys())
		values_reduced = pca(values, n_components)
		#print(values_reduced.shape)
		features_reduced = dict(zip(features.keys(), values_reduced))
		features = features_reduced

		#save_array("gray.dict",features)
		saveFile("gray_reduced_2048.dict",features)
		saveFile("gray_imagelist.csv",imagelist)
		#print(values.shape)

	return imagelist,features

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
	#save_array("features_reduced_1024.dict",features_reduced)
	saveFile("color_imagelist.csv",imagelist)

	return imagelist,features

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


def pca(img, n_comp=200):
    pca = PCA( n_components = n_comp)#, svd_solver='full')
    pca.fit(img)
    #print(img_r.shape)
    transformed_pca = pca.fit_transform(img )  # transform
    #restored_img = pca.inverse_transform(transformed_pca) # inverse_transform
    #print(pca.explained_variance_ratio_, pca.singular_values_)
    return transformed_pca


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		#print(df.shift(i) )
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
			#print( "n_vars",n_vars)
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def load_img_values(features, idx):
    a,b = idx.shape
    #dddd= list(features.values())
    #print(dddd[0][0], len( dddd[0][0] ))
    #print( list(imagedict.values())[0])
    D = len(list(features.values())[0])  # number of features
    #print(a,b,D)
    target = np.zeros([a,b,D])
    for i in range(a):
        for j in range(b):
            target[i,j,:] = features[idx[i,j]]
    return target


def lstm_data(imagelist, features, n_inputs, n_outputs = 7 ):
    reframed = series_to_supervised(imagelist,n_inputs,n_outputs)  #change unsupervised problem to supervised problem
    indices  = reframed.values  #pandas data to a matrix
    m,n = indices.shape
    #print( indices)

    X_data , y_data  = load_img_values(features, indices[:,:-n_outputs]  ), load_img_values(features, indices[:, -n_outputs:] )

    num_training = int(m*0.67)
    num_val = m - num_training

    #n_inputs = n-n_outputs
    train_idx = indices[:num_training,:]
    train_X_idx , train_y_idx  = train_idx[:, :-n_outputs], train_idx[:, -n_outputs:]
    train_X  = X_data[:num_training]
    train_y = y_data[:num_training]

    val_idx  = indices[num_training:,:]  #get x,y val examples
    val_X_idx , val_y_idx  = val_idx[:, :-n_outputs], val_idx[:, -n_outputs:]
    #print(indices)
    #sys.exit()


    #print( val_y_idx.shape, val_y_idx[0])
    #sys.exit()
    val_X  = X_data[num_training:]
    #val_y = y_data[num_training:]
    #print(val_X.shape)
    #sys.exit()
    #print(val_X.shape, val_y.shape)
    val_y_index = val_y_idx.tolist()
    #print( val_y_index[0])
    #sys.exit()

    """
    dict_val = {}
    for i in range(num_val):
        groups = val_y_index[i]
        a = []  # it has 7 feature groups for one example  ,
        for j in groups:
            a.append( features[j])
        dict_val[i] = a
    """

    save_array("train_X.dat",train_X)
    save_array("train_y.dat",train_y)
    save_array("val_X.dat",val_X)
    #save_array("val_y.dat",val_y)
    #save_array("X_data.dat",X_data)
    #save_array("y_data.dat",y_data)
    #saveFile("dict_val.dict",dict_val)
    saveFile('val_y_index.csv',val_y_index)
    return  train_X, train_y #X_data , y_data


import matplotlib.pyplot as plt
def self_pca(A, n_components = 250):
	cov = np.dot( A.T, A)/A.shape[0]
	U, s, Vh = np.linalg.svd(cov)
	#print( U.shape, s.shape, Vh.shape  )
	plt.plot(s)
	plt.show()
	#print(s)
	s[n_components:] = 0

	new_A = Xrot_reduced = np.dot(X, U[:,:n_components])  #np.dot(np.dot(U,np.diag(s)),Vh)# np.dot(U, np.dot(np.diag(s), Vh))

	return new_A

def intersect(a, b):
	return list(set(a) & set(b))
