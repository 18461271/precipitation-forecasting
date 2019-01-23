# precipitation-forecasting

The primary of purpose of this project is to use the last 6 years precipitation map to generate the next 7 days forecasting.

## My first approach is to use the Conv-LSTM, the basic idea is to apply convolution networks on images to encode local spatial temporal information and then decode this information for forecasting.

how to  use :   run `ConvLSTM_train.py`

1. the first step involves converting gif image to jpg image
2. the second step is to read image data and crop uncessary text in the image, and then build a dictionary to store  image_id : 3D_image_data pairs
3. the 3rd step is to change unsupervised problem to supervised problem. Train : data is split as follows, 60% train, 40 validation.
4. the 4th step is to build a ConvLSTM model: many to many multivariate and multi-step model.  ConvLSTM can preserve all the spatial information
5. train the model by reading the data on the fly
6. Evaluation : feed the validation_x data to the trained model,  the general idea is to utilize cosine similarity to output the imagesID lists which have the
shortest distances to the stored features,, and do the intersect_count with regards to ture image_id
this part is not provided.


## My Second approach is to use the fully connected LSTM, the basic idea is to apply recurrent neural networks on images to encode image features and then decode this information for forecasting, a typical encoder-decoder LSTM model for solving sequence-to-sequence prediction problems.

how to  use :   run `FLSTM_train.py` to train the model then run `FLSTM_test.py` to test.
 
1. the first step involves converting gif image to jpg image
2. the second step is to read image data and crop uncessary text in the image, all the images are reshaped  and  PCA dimension deducted and standardized into 1D array with 2048 dimensions, and finally a dictionary is built to store  (image_id : 1d_image_feature) pairs
3. the 3rd step is to change unsupervised problem to supervised problem. Train : data is split as follows, 67% train, 33% validation. X,Y data have shape (n_items, time_dimension, 2048) as required by LSTM layer `input_shape`
4. the 4th step is to build a LSTM model: many to many multivariate and multi-step model. "To summarize, the RepeatVector is used as an adapter to fit the fixed-sized 2D output of the encoder to the direring length and 3D input expected by the decoder. The TimeDistributed
wrapper allows the same output layer to be reused for each element in the output sequence. " 
> *long short term memory networks with python*  P129  
6. Evaluation : feed the validation_x data to the trained model,  the general idea is to utilize cosine similarity to output the imagesId lists which have the shortest distances to the stored features,, and do the intersect_count with regards to ture image_id.

Reference: 


1.[Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model](https://github.com/sxjscience/HKO-7)

2.unpervised learning of video representations of using lstms
                   
3.long short term memory networks with python
