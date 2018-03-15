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
from ReadImages import load_databatch,load_databatch_test
from ReadImages import loadHistory,saveHistory





base_dir = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/"

data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/"      #Server
idx = 1
img_size = 32

#load Dataset
X_train,Y_train = load_databatch(data_folder = data_folder, idx=idx, img_size= img_size)


batch_size = 32
num_classes = 10
epochs = 200
save_dir = os.path.join(base_dir,'SavedModels')
model_name = 'Keras_CIFAR10_trainModel_v1.h5'
history_name = 'Keras_CIFAR10_trainModel_v1'

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape= X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))



opt = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


filepath = base_dir + "BestModels/FirstModel/Model - {epoch:02d}-{val_acc:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='categorical_accuracy',mode='max', save_best_only=True)
callback_list = [checkpointer]

# To check the model if working
# X_train = X_train[0:200, :]  # to reduce the data to ease the computations
# Y_train = Y_train[0:200]  # comment these 2 lines while the original execution

hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_split= 0.2,shuffle=True, callbacks= callback_list);      #hist will store everything

saveHistory(os.path.join(save_dir,str('hist_') + history_name),hist = hist)                   #Save History
# prevHist = loadHistory(os.path.join(save_dir,str('hist_') + model_name)                   #Load Histroy
#-------------------------------------------------------------------------------------------------- Testing
score = model.evaluate(X_train, Y_train, batch_size=32)
model.save(os.path.join(save_dir,model_name))

# model.save(os.path.join(save_dir, str('copy_') + model_name))         #save a copy of the model if you want



#--------------------------------------------------------------------------------------------------- Load Existing Model
load_model_path = os.path.join(save_dir,model_name)
model = load_model(load_model_path)





