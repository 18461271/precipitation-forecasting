# precipitation-forecasting



How to  use :   run data_model.py

The primary of purpose of this project is to use the last 6 years precipitation map to generate the next 7 days forecasting.

My first approach is to use the Conv-LSTM, the basic idea is to apply convolution networks on images to encode local spatial temporal information
and then decode this information for forecasting, a typical encoder-decoder LSTM model.

1. the first step involves converting gif image to jpg image, the downloaded dataset is extracted in folder "data_download", the converted images are save in folder "processeddata".
2. the second step is to read image data and crop uncessary text in the image, and then build a dictionary to store  image_id : 3D_image_data pairs
3. the 3rd step is to change unsupervised problem to supervised problem. Train : data is split as follows, 60% train, 40 validation.
4. the 4th step is to build a ConvLSTM model: many to many multivariate and multi-step model.  ConvLSTM can preserve all the spatial information
5. train the model by reading the data on the fly
6. Evaluation : feed the validation_x data to the trained model,  the general idea is to utilize cosine similarity, the relevant code is not provided. 

Reference: 1.(Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model)[https://github.com/sxjscience/HKO-7]
           2.unpervised learning of video representations of using lstms
