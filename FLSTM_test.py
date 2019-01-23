
import numpy as np,sys
import os
from keras.models import model_from_json
from general_functions import *
from sklearn.metrics.pairwise import pairwise_distances
import time
import matplotlib.pyplot as plt

processed_folder =  "processeddata/"

model_path  = "model/"
model_name = " model_flstm "
start_time_all = time.time()


features = loadFile('gray_reduced_2048.dict')
#test_X , test_y = load_array("test_X.dat"),load_array("test_y.dat")
#vectors = np.array(list( features.values()))
test_y_index = loadFile('val_y_index.csv')
#test_y_index = test_y_idx.tolist()
#n_vocab  = len(features)
n_tests = len(test_y_index)

#print(n_tests )
vectors_matrix = np.array(list(features.values()))
test_X = load_array("test_X.dat")
#model_file = glob.glob(path)

json_file = open(model_path+ model_name+ ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_path + model_name+ ".h5")
# demonstrate prediction

#test_X= test_X.reshape( test_X.shape[0], test_X.shape[1]*test_X.shape[2])

yhat = model.predict(test_X, batch_size = 8 , verbose=0)  #predicted features , with shape ( n_test, n_outputs, feature_dim)

print("prediction finished")
intersect_count = 0
total_count = 0

for i in range(n_tests ):
    #print( yhat[i].shape)  #(7,,4096)

    #print( yhat[i].reshape(7, 4096).sh
    # ape)
    #print(yhat[i])
    #yhat[i] = np.mean( yhat[i], axis=0)
    #print(yhat[i].shape)
    #sys.exit()
    pw_dist = pairwise_distances(vectors_matrix, yhat[i], metric = 'cosine')  #.reshape(1,-1)

    #pw_dist = pairwise_distances(vectors_matrix, yhat[i][0].reshape(1,-1) , metric = 'cosine')  #
    #pca_pw_dist = pca(pw_dist.T )
    #print(pw_dist.shape, pw_dist[0] ,pw_dist[1] )
    #sys.exit()
    #print( yhat[i].shape, pw_dist.shape)
    #arg_sort_list = np.argsort(a = pw_dist, axis=0, kind='quicksort')[:2].T  #.flatten() #[:20]  #it returns the smallest value of a list

    arg_sort_list = np.argsort(a = pw_dist, axis=0)  #, kind='quicksort'            #.flatten()   #[:7]
    #print(arg_sort_list.shape)
    similiar_images = np.take(a = list(features.keys()), indices=arg_sort_list, axis=0)[:2].flatten()
    #print(similiar_images.shape,  len(similiar_images ))
    #sys.exit()

    print("1:",list(similiar_images ))  #, test_y_index[i] )
    #sys.exit()
    print("2:", test_y_index[i] )
    print( "\n")
    #print(pw_dist.shape, similiar_images.shape, type(similiar_images)  )
    intersect_count += len( intersect(list(similiar_images), test_y_index[i]  )  )
    print(intersect_count )
    #sys.exit()
    total_count += len( test_y_index[i] )
score = intersect_count/total_count
#print(score)
elapsed_time = time.time() - start_time_all
print("elapsed_time:",elapsed_time)
print( "score: %.2f%%" %(score*100))
