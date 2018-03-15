from __future__ import division
# # from __future__ import print_function
#
# import os
from keras.models import Model
import numpy as np;
# import random
#
#
# # dataPath = "/home/ankit/Desktop/dataset/CIFAR_Kaggle/train"
#
#
# def initialize(model, X_train):
#     number = 0; number2 = 0; number3 = 0; number4 = 0;
#     # configfiles = [os.path.join(dirpath, f)
#     #                for dirpath, dirnames, files in os.walk(dataPath)
#     #                for f in files if f.endswith('.png')]  # list all the files with a given extension in directory
#     totalFiles = list(np.arange(X_train.shape[0]))
#     # configfiles = configfiles[:100]  # reduce the file size easy handling
#     random.shuffle(totalFiles);
#     layers = model.layers;
#     for layer in layers:
#         if len(layer.get_weights()) != 0:
#             sample = random.sample(totalFiles, 32);
#             sample = X_train[sample]    #now take the only 32 X_train values after taking random on indexes
#             # gaussian initialization
#             row, col, channels, filters = layer.get_weights()[0].shape;
#             print(number); number += 1;
#             weights0 = np.zeros(layer.get_weights()[0].shape, dtype="float32");
#             weights1 = np.zeros(filters, dtype="float32");
#             mean = np.zeros(row * col, dtype="float32");
#             cov = np.identity(row * col, dtype="float32");
#             for i in range(filters):
#                 for j in range(channels):
#                     weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
#             layer.set_weights([weights0, weights1]);
#             for img in sample:
#                 intermediate_layer_model = Model(inputs=model.input,
#                                                  outputs=layer.output)
#                 intermediate_output = intermediate_layer_model.predict(img.reshape(1,64,64,3));
#                 # intermediate_output = intermediate_layer_model.predict(img);
#                 avg_img = np.average(intermediate_output, axis=1);
#                 for i in range(filters):
#                     # avg = avg_img[ :, :, i].mean;
#                     # std = avg_img[ :, :, i].std;
#                     avg = np.average(avg_img[ :, :, i]);
#                     std = np.std(avg_img[ :, :, i]);
#                     mean = np.ones(row * col, dtype="float32") * avg;
#                     cov = np.identity(row * col, dtype="float32") * (std ** 2);
#                     for j in range(channels):
#                         weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
#                     weights1[i] = avg;
#             layer.set_weights([weights0, weights1]);
#     return model;
#

# model2 = initialize(model= model, X_train=X_train)
#

import os
from keras.models import Model
import numpy as np;
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.preprocessing import normalize
from computeFans import compute_fans



def get_array(fname):
	img = load_img(fname,target_size=(128,128));
	x = img_to_array(img);
	x = x.transpose(2,1,0)
	x = x.reshape((3,128,128))
	x = x/255;
	x -= x.mean(axis=(-2,-1),keepdims=1)
	return x;


def whiten(X,fudge=1E-3):
   Xcov = np.dot(X.T,X)
   d, V = np.linalg.eigh(Xcov)
   D = np.diag(1. / np.sqrt(d+fudge))
   W = np.dot(np.dot(V, D), V.T)
   X_white = np.dot(X, W)
   return X_white

def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X)
    X_white = np.dot(U, Vt)
    return X_white

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)

def zca_whitening_matrix(X):
    sigma = np.cov(X, rowvar=True) # [M x M]
    U,S,V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return np.dot(ZCAMatrix,X)



# def initialize(model,dataPath,batch_size):
#     configfiles = [os.path.join(dirpath, f)
#         for dirpath, dirnames, files in os.walk(dataPath)
#         for f in files if f.endswith('.jpg')]#list all the files with a given extension in directory
#     random.shuffle(configfiles);
#     layers = model.layers;
#     for num,layer in enumerate(layers):
#         if len(layer.get_weights())!=0:
#             if len(layer.get_weights()[0].shape)==4:
#                 sample = random.sample(configfiles,batch_size);
#                 data = np.zeros((batch_size,3,128,128));
#                 for index,img in enumerate(sample):
#                     data[index,:,:,:] = get_array(img);
#                 row,col,channels,filters = layer.get_weights()[0].shape;
#                 dx,dy = row,col;
#                 weights0 = np.zeros(layer.get_weights()[0].shape,dtype="float32");
#                 weights1 = np.zeros(filters,dtype="float32");
#                 intermediate_layer_model = Model(inputs=model.input,outputs=layers[num-1].output)
#                 intermediate_output = intermediate_layer_model.predict(data);
#                 _,pre_channels,row,col = intermediate_output.shape;
#                 w,h = row,col;
#                 x = (1 + (batch_size/pre_channels))*4;
#                 filterData = np.zeros((dx*dy*pre_channels,batch_size*int(x)),dtype='float32');
#                 count = 0;
#                 for i in range(batch_size):
#                     for j in range(int(x)):
#                         X = random.randint(0, w-dx-1);
#                         Y = random.randint(0, h-dy-1);
#                         filterData[:,count] = intermediate_output[i,:,X:X+dx,Y:Y+dy].reshape((pre_channels*dx*dy))
#                         count = count + 1;
#                 print "computing mean..."
#                 mean = np.mean(filterData,axis=1);
#                 print "mean computed, computing cov... ", np.sum(mean);
#                 cov = np.cov(filterData);
#                 print "computed covariance... ", np.sum(cov);
#                 data = np.random.multivariate_normal(mean,cov,filters);
#                 l2norm = np.sqrt((data*data).sum(axis=1))
#                 new_data = data / l2norm[:, np.newaxis]
#                 weights0 = whiten(new_data).reshape((dx,dy,pre_channels,filters));
#                 layer.set_weights([weights0,weights1]);
#             print "done layer " + str(num);
#     return model;
#





def scaledInitialize(model, X_train):

    totalFiles = list(np.arange(X_train.shape[0]))
    # configfiles = configfiles[:100]  # reduce the file size easy handling
    random.shuffle(totalFiles);

    layers = model.layers;
    for num, layer in enumerate(layers):
        if len(layer.get_weights()) != 0:
            #==================================================================== Scaling the Standard deviatin
            fan_in,fan_out = compute_fans(layer.get_weights()[0].shape)
            scale = 2
            scale /= max(1,fan_in)                      #using only He for now
            scaledSD = np.sqrt(2*scale); std = scaledSD;
            #==================================================================================================
#----------------------------------------------------------------------------------------- Convolution Layers
            if len(layer.get_weights()[0].shape) == 4:
                # sample = random.sample(configfiles, 8);
                sample = random.sample(totalFiles, 8);
                sample = X_train[sample]  # now take the only 32 X_train values after taking random on indexes

                # data = np.zeros((8, 3, 64, 64));
                data = np.zeros((8, 64, 64, 3));            #channels last

                # for index, img in enumerate(sample):
                #     data[index, :, :, :] = get_array(img);
                data = sample
                # gaussian initialization
                row, col, channels, filters = layer.get_weights()[0].shape;
                weights0 = np.zeros(layer.get_weights()[0].shape, dtype="float32");
                weights1 = np.zeros(filters, dtype="float32");
                mean = np.zeros(row * col, dtype="float32");
                cov = np.identity(row * col, dtype="float32");
                for i in range(filters):
                    for j in range(channels):
                        weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
                layer.set_weights([weights0, weights1]);

                # for img in sample:
                intermediate_layer_model = Model(inputs=model.input,
                                                 outputs=layer.output)
                intermediate_output = intermediate_layer_model.predict(data);
                avg_img = np.average(intermediate_output, axis=0);
                for i in range(filters):
                    avg = avg_img[ :, :, i].mean();         #both are filters last
                    # std = avg_img[ :, :, i].std(); #print("Std =",std)
                    mean = np.ones(row * col, dtype="float32") * avg;
                    cov = np.identity(row * col, dtype="float32") * (std ** 2); #print("Cov is =", cov)

                    #=============================================================  S Dev Scaling and Mean shifting
                    # std = std * scaledSD /abs(std)
                    mean = mean - (np.average(mean))
                    #=============================================================================
                    for j in range(channels):
                        weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
                    weights1[i] = avg;
                layer.set_weights([weights0, weights1]);
                print("done initializing layer " + str(num));
#--------------------------------------------------------------------------------------- Dense Layers

            elif len(layer.get_weights()[0].shape) == 2:
                sample = random.sample(totalFiles, 8);
                sample = X_train[sample]  # now take the only 32 X_train values after taking random on indexes

                data = np.zeros((8, 64, 64, 3));            #channels last
                data = sample
                weights0 = np.zeros(layer.get_weights()[0].shape, dtype="float32");
                weights1 = np.zeros(layer.get_weights()[1].shape, dtype="float32");
#-------------------------------------------------------------------------------------------------- Intermediate Model
                intermediate_layer_model = Model(inputs=model.input,outputs=layer.output)
                intermediate_output = intermediate_layer_model.predict(data);
                intermediate_output = intermediate_output / max(intermediate_output.flatten())  # random approximation

                avg_img = np.average(intermediate_output, axis=0);
                if avg_img.std() == (float('inf') or float('nan')) : std = (avg_img/avg_img[0]).std();
                else: std = avg_img.std();
                # =============================================================  S Dev Shifting
                std = std * scaledSD / abs(std)
                mean = np.zeros(shape=len(avg_img) ,dtype="float32");
                # =============================================================================
                cov = np.identity(len(avg_img)) * (std ** 2);

#-------------------------------------------------------------------------------------------------- Weights Initialization
                weights1 = avg_img              #bias are initialized for average of the samples
                weights0 = np.random.multivariate_normal(mean = mean,cov= cov,size=(layer.get_weights()[0].shape[0]))
                layer.set_weights([weights0, weights1]);
                print("done initialing layer " + str(num));
    return model;



def initialize(model, X_train):
    # configfiles = [os.path.join(dirpath, f)
    #                for dirpath, dirnames, files in os.walk(dataPath)
    #                for f in files if f.endswith('.jpg')]  # list all the files with a given extension in directory
    # random.shuffle(configfiles);

    totalFiles = list(np.arange(X_train.shape[0]))
    # configfiles = configfiles[:100]  # reduce the file size easy handling
    random.shuffle(totalFiles);

    layers = model.layers;
    for num, layer in enumerate(layers):
        if len(layer.get_weights()) != 0:
#----------------------------------------------------------------------------------------- Convolution Layers
            if len(layer.get_weights()[0].shape) == 4:
                # sample = random.sample(configfiles, 8);
                sample = random.sample(totalFiles, 8);
                sample = X_train[sample]  # now take the only 32 X_train values after taking random on indexes

                # data = np.zeros((8, 3, 64, 64));
                data = np.zeros((8, 64, 64, 3));            #channels last

                # for index, img in enumerate(sample):
                #     data[index, :, :, :] = get_array(img);
                data = sample
                # gaussian initialization
                row, col, channels, filters = layer.get_weights()[0].shape;
                weights0 = np.zeros(layer.get_weights()[0].shape, dtype="float32");
                weights1 = np.zeros(filters, dtype="float32");
                mean = np.zeros(row * col, dtype="float32");
                cov = np.identity(row * col, dtype="float32");
                for i in range(filters):
                    for j in range(channels):
                        weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
                layer.set_weights([weights0, weights1]);

                # for img in sample:
                intermediate_layer_model = Model(inputs=model.input,
                                                 outputs=layer.output)
                # intermediate_output = np.float32(1.0)
                intermediate_output = intermediate_layer_model.predict(data);
                intermediate_output = intermediate_output/max(intermediate_output.flatten())           #random approximation
                avg_img = np.average(intermediate_output, axis=0);
                for i in range(filters):
                    avg = avg_img[ :, :, i].mean();         #both are filters last
                    std = avg_img[ :, :, i].std(); #print("Std =",std)
                    mean = np.ones(row * col, dtype="float32") * avg;
                    cov = np.identity(row * col, dtype="float32") * (std ** 2); #print("Cov is =", cov)
                    for j in range(channels):
                        weights0[:, :, j, i] = np.random.multivariate_normal(mean, cov).reshape((row, col));
                    weights1[i] = avg;
                layer.set_weights([weights0, weights1]);
                print("done initializing layer " + str(num));
#--------------------------------------------------------------------------------------- Dense Layers

            elif len(layer.get_weights()[0].shape) == 2:
                sample = random.sample(totalFiles, 8);
                sample = X_train[sample]  # now take the only 32 X_train values after taking random on indexes

                data = np.zeros((8, 64, 64, 3));            #channels last
                data = sample
                weights0 = np.zeros(layer.get_weights()[0].shape, dtype="float32");
                weights1 = np.zeros(layer.get_weights()[1].shape, dtype="float32");
#-------------------------------------------------------------------------------------------------- Intermediate Model
                intermediate_layer_model = Model(inputs=model.input,outputs=layer.output)
                intermediate_output = intermediate_layer_model.predict(data);
                intermediate_output = intermediate_output / max(intermediate_output.flatten())  # random approximation

                avg_img = np.average(intermediate_output, axis=0);
                if avg_img.std() == (float('inf') or float('nan')) : std = (avg_img/avg_img[0]).std();
                else: std = avg_img.std();

                cov = np.identity(len(avg_img)) * (std ** 2);

#-------------------------------------------------------------------------------------------------- Weights Initialization
                weights1 = avg_img              #bias are initialized for average of the samples
                weights0 = np.random.multivariate_normal(mean = avg_img,cov= cov,size=(layer.get_weights()[0].shape[0]))
                layer.set_weights([weights0, weights1]);
                print("done initialing layer " + str(num));
    return model;






#
#
# model2 = initialize(model= model, X_train=X_train)
#
#
#
