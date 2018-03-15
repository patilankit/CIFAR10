import numpy as np;
import os
from keras.models import Model
import numpy as np
import random
from computeFans import compute_fans
from ReadImages import load_databatch,load_databatch_test,load_data_full
from HeInitialize import get_model
from Jan_Initialize_v2 import whiten




base_dir = "/home/ankit/Desktop/Dataset/JanResults/CIFAR10";        #save results here...

data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/";      #Server
img_size = 32;
batch_size = 32;
num_classes = 10;
epochs = 10;


#------------------- Get the X_train and Model for temporary use
X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)
model = get_model();
del Y_train, X_test,Y_test;                                                 #Freeing up the space
#---------------------------------------------------------------------------------------------------





def jan_initialize(model,X_train,samp_size = 8):
    # samp_size = 8192;

    totalFiles = list(np.arange(X_train.shape[0]))
    random.shuffle(totalFiles);


    layers = model.layers;
    for num, layer in enumerate(layers):
        if len(layer.get_weights()) != 0:
    #==================================================================== Scaling the Standard deviatin
                fan_in,fan_out = compute_fans(layer.get_weights()[0].shape)
    #==================================================================================================

                sample = random.sample(totalFiles, samp_size);
                sample = X_train[sample]  # now take the only 32 X_train values after taking random on indexes
                data = sample;
                twodim_mat = data.reshape(samp_size,-1);
                mean = np.mean(twodim_mat,axis=1);
                cov = np.cov(twodim_mat,rowvar=True);

                scale = 1.0
                scale /= max(1., float(fan_in + fan_out) / 2)


#-------------------------------------------------------------- CoVariance Scaling
                # cov = np.identity(samp_size, dtype="float32")
                # cov = cov*scale;
                # mean = mean *0;             # mean Scaling
#---------------------------------------------------------------------------------



                temp_weight = np.random.multivariate_normal(mean,cov,size=layer.get_weights()[0].shape)
                mean_weight = np.mean(temp_weight,axis=-1)          #taking the mean of weights across the image dimensions

                mean_weight = whiten(mean_weight,scale)

                weights0 = mean_weight;
                weights1 = np.zeros(layer.get_weights()[1].shape, dtype="float32");

                print("Size of The layer Weights", layer.get_weights()[0].shape)
                print("Size of the corresponding weight matrix", weights0.shape)
                layer.set_weights([weights0,weights1])
                print("Done Conv layer", num)

    return model

def jan_initialize_activation(model, X_train, samp_size=8,act= 'relu'):
    # samp_size = 8192;

    totalFiles = list(np.arange(X_train.shape[0]))
    random.shuffle(totalFiles);

    layers = model.layers;
    for num, layer in enumerate(layers):
        if len(layer.get_weights()) != 0:
            # ==================================================================== Scaling the Standard deviatin
            fan_in, fan_out = compute_fans(layer.get_weights()[0].shape)
            # ==================================================================================================

            sample = random.sample(totalFiles, samp_size);
            sample = X_train[sample]  # now take the only 32 X_train values after taking random on indexes
            data = sample;
            twodim_mat = data.reshape(samp_size, -1);
            mean = np.mean(twodim_mat, axis=1);
            cov = np.cov(twodim_mat, rowvar=True);

            scale = 1.0
            scale /= max(1., float(fan_in + fan_out) / 2)

            # -------------------------------------------------------------- CoVariance Scaling
            # cov = np.identity(samp_size, dtype="float32")
            # cov = cov*scale;
            # mean = mean *0;             # mean Scaling
            # ---------------------------------------------------------------------------------



            temp_weight = np.random.multivariate_normal(mean, cov, size=layer.get_weights()[0].shape)
            mean_weight = np.mean(temp_weight, axis=-1)  # taking the mean of weights across the image dimensions

            mean_weight = whiten(mean_weight, scale)

            weights0 = mean_weight;
            weights1 = np.zeros(layer.get_weights()[1].shape, dtype="float32");

            print("Size of The layer Weights", layer.get_weights()[0].shape)
            print("Size of the corresponding weight matrix", weights0.shape)
            layer.set_weights([weights0, weights1])
            print("Done Conv layer", num)

    return model





        # model2 = jan_initialize(model,X_train,8)              #check the working function :)

# def compute_fans(shape, data_format='channels_last'):
#     """Computes the number of input and output units for a weight shape.
#     # Arguments
#         shape: Integer shape tuple.
#         data_format: Image data format to use for convolution kernels.
#             Note that all kernels in Keras are standardized on the
#             `channels_last` ordering (even when inputs are set
#             to `channels_first`).
#     # Returns
#         A tuple of scalars, `(fan_in, fan_out)`.
#     # Raises
#         ValueError: in case of invalid `data_format` argument.
#     """
#     if len(shape) == 2:
#         fan_in = shape[0]
#         fan_out = shape[1]
#     elif len(shape) in {3, 4, 5}:
#         # Assuming convolution kernels (1D, 2D or 3D).
#         # TH kernel shape: (depth, input_depth, ...)
#         # TF kernel shape: (..., input_depth, depth)
#         if data_format == 'channels_first':
#             receptive_field_size = np.prod(shape[2:])
#             fan_in = shape[1] * receptive_field_size
#             fan_out = shape[0] * receptive_field_size
#         elif data_format == 'channels_last':
#             receptive_field_size = np.prod(shape[:-2])
#             fan_in = shape[-2] * receptive_field_size
#             fan_out = shape[-1] * receptive_field_size
#         else:
#             raise ValueError('Invalid data_format: ' + data_format)
#     else:
#         # No specific assumptions.
#         fan_in = np.sqrt(np.prod(shape))
#         fan_out = np.sqrt(np.prod(shape))
#     return fan_in, fan_out
#
#