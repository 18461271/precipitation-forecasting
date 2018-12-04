import numpy as np
#import keras
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
from keras.utils import Sequence
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  n_inputs , n_outputs, idx, idy, features, batch_size=32, dim=(250, 360,3) ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.idx = idx
        self.idy = idy
        #self.timesteps = timesteps
        self.features = features
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.list_IDs / self.batch_size))

    def __getitem__(self, index ):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [list(range(self.list_IDs))[k] for k in indexes]
        n_inputs = self.n_inputs
        n_outputs = self.n_outputs

        # Generate data
        X,y  = self.__data_generation(list_IDs_temp, n_inputs, n_outputs)
        #y = self.__data_generation(list_IDs_temp)

        return X,y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.list_IDs)

    def __data_generation(self, list_IDs_temp, n_inputs, n_outputs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_inputs,  *self.dim))
        y = np.empty((self.batch_size, self.n_outputs,  *self.dim))
        features = self.features
        idx = self.idx
        idy = self.idy
        # Generate data
        for n,i in enumerate(list_IDs_temp):
            for t1 in range(n_inputs):
                #print(idx[n]  )
                # Store sample
                X[n,t1,:,:,:] = features[idx[n][t1]]
            for t2 in range( n_outputs) :
                y[n,t2,:,:,:] = features[idx[n][t2]]

        return X,y
