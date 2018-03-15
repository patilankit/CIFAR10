# Author: Ankit Patil
# Train a simple CNN ove CIFAR10 dataset from _______________________ website ka naam
#

import numpy as np
import os
import pickle
from keras.utils import np_utils

# data_folder = "/home/ankit/Desktop/DDP/ImageNet/Train"                #Local System
# data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/"      #Server
# idx = 1
# img_size = 32





def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

def loadHistory(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def saveHistory(file,hist):
    with open(file, 'wb') as fo:
        dict = pickle.dump(hist.history, fo)
    return dict


#
#Input: data_folder is where the data is stored, idx is the index of the training set, img_size is the 32*32 in this case
#Output: X_train, Y_train (categorical) [X_train -> {data_size,3,img_size,img_size},   Y_train -> {data_size,1000}]
#Funciton: Wrapper function for getting directly the X_train and Y_train just by giving the folder path and index number
#

def load_databatch(data_folder, idx, img_size = 32):
    data_file = os.path.join(data_folder, 'data_batch_')
    # data_file = os.path.join(data_folder, 'test_batch')

    d = unpickle(data_file + str(idx))
    # d = unpickle(data_file)

    x = d[b'data']
    y = d[b'labels']

    # x = x[0:200, :]  # to reduce the data to ease the computations
    # y = y[0:200]     #comment these 2 lines while the original execution

    x = x / np.float32(255)

                                                                                                                                                                                                                            # Labels are indexed from 1, shift it so that indexes start at 0
    data_size = x.shape[0]
    #the images are already dstacked (horizontally stacked) with RGB order [1024,1024,1024]
    # x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)            #Channels First
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 1, 2, 3)            #Channels Last

    train_size = data_size
    #===================================================================================================================
    X_train = x[0:train_size, :, :, :]                   #here you can change the data_size of your training data if wished
    Y_train = y[0:train_size]
    #===================================================================================================================

    Y_train = np_utils.to_categorical(Y_train, 10)

    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)



    return X_train, Y_train

#
#Input: data_folder is where the data is stored, idx is the index of the training set, img_size is the 32*32 in this case
#Output: X_test, Y_test (categorical) [X_train -> {data_size,3,img_size,img_size},   Y_train -> {data_size,10}]
#Funciton: Wrapper function for getting directly the X_train and Y_train just by giving the folder path and index number
#

def load_databatch_test(data_folder, idx, img_size = 32):
    # data_file = os.path.join(data_folder, 'data_batch_')
    data_file = os.path.join(data_folder, 'test_batch')

    # d = unpickle(data_file + str(idx),)
    d = unpickle(data_file)

    x = d[b'data']
    y = d[b'labels']

    # x = x[0:200, :]  # to reduce the data to ease the computations
    # y = y[0:200]     #comment these 2 lines while the original execution

    x = x / np.float32(255)

                                                                                                                                                                                                                            # Labels are indexed from 1, shift it so that indexes start at 0
    data_size = x.shape[0]
    #the images are already dstacked (horizontally stacked) with RGB order [1024,1024,1024]
    # x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)            #Channels First
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 1, 2, 3)            #Channels Last

    train_size = data_size
    #===================================================================================================================
    X_train = x[0:train_size, :, :, :]                   #here you can change the data_size of your training data if wished
    Y_train = y[0:train_size]
    #===================================================================================================================
    
    Y_train = np_utils.to_categorical(Y_train, 10)

    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return X_train, Y_train

#
#Input: data_folder is where the data is stored, idx is the index of the training set, img_size is the 32*32 in this case
#Output: X_train,Y_train,X_test, Y_test; Amazing (Insn't it? :P) (categorical) [X_train -> {data_size,3,img_size,img_size},   Y_train -> {data_size,10}]
#Funciton: This is all what you need; just call this function it will do all the dirty lifting.
#

def load_data_full(data_folder, batch_no, img_size = 32):

    for batch in range(1,batch_no + 1,1):
        X_train_temp,Y_train_temp = load_databatch(data_folder=data_folder,idx = batch,img_size=img_size)
        if batch == 1: X_train = X_train_temp; Y_train = Y_train_temp;
        else:
            X_train = np.append(X_train,X_train_temp,axis=0);
            Y_train = np.append(Y_train,Y_train_temp,axis=0);

    X_test, Y_test = load_databatch_test(data_folder=data_folder, idx=batch_no, img_size=img_size)

    return X_train, Y_train,X_test,Y_test



#---------------------------------------------------------------------------------------------------- Tester
# X_train,Y_train = load_databatch(data_folder = data_folder, idx=idx, img_size= img_size)
# X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)



