from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.models import Sequential
from keras.metrics import categorical_accuracy
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout,GaussianNoise
from keras.optimizers import SGD,RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import numpy as np
# from tqdm import tqdm
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier
from keras.constraints import maxnorm
# import cv2
import random
import os
from PIL import Image
import pandas as pd
from pandas import DataFrame
from collections import OrderedDict
# from tqdm import tqdm
import pickle
import datetime
from ReadImages import load_databatch
from ReadImages import saveHistory,loadHistory

base_dir = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/"

data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/"      #Server
idx = 1
img_size = 32


batch_size = 32
num_classes = 10
epochs = 200
save_dir = os.path.join(base_dir,'SavedModels')
model_name = 'Keras_CIFAR10_trainModel_v1.h5'
history_name = 'Keras_CIFAR10_trainModel_v1'

for idx in range(2,6,1):
    #load Dataset
    X_train,Y_train = load_databatch(data_folder = data_folder, idx=idx, img_size= img_size)
    print("Batch {:1d} data is loaded".format(idx))
    # To check the model if working
    # X_train = X_train[0:200, :]  # to reduce the data to ease the computations
    # Y_train = Y_train[0:200]  # comment these 2 lines while the original execution

    load_model_path = os.path.join(save_dir,str('batch_')+str(idx - 1) + model_name);
    print("Model Trained on batch {:1d} is loaded".format(idx -1))
    model = load_model(load_model_path)
    
    filepath = base_dir + "BestModels/FirstModel/"+"batch_{0:01d}_".format(idx) +"Model - {epoch:02d}-{val_acc:.2f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='categorical_accuracy',mode='max', save_best_only=True)
    callback_list = [checkpointer]
    hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_split= 0.2,shuffle=True, callbacks= callback_list);      #hist will store everything


    model.save(os.path.join(save_dir,str('batch_')+str(idx) + model_name))
    saveHistory(os.path.join(save_dir,str('hist_{0:01d}'.format(idx)) + model_name),hist = hist)                   #Save History





