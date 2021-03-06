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
from keras.callbacks import ModelCheckpoint,EarlyStopping
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
from ReadImages import load_databatch,load_databatch_test,load_data_full
from ReadImages import loadHistory,saveHistory
from HeInitialize import get_model
from dependant_initialization_v1 import initialize,scaledInitialize


base_dir = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/"

data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/"      #Server
idx = 1
img_size = 32


batch_size = 32
num_classes = 10
epochs = 150
save_dir = os.path.join(base_dir,'SavedModels/ProposedModel_try1')
model_name = 'Keras_CIFAR10_proposed_v1.h5'
history_name = 'Keras_CIFAR10_proposed_v1'

#load Train/Test
X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)
#ModelKeys
initDict = {'LeCun':'lecun_uniform', 'GlorotNorm':'glorot_normal','GlorotUni':'glorot_uniform','HeNorm':'he_normal','HeUni':'he_uniform'}

model = get_model(initDict['HeNorm'],input_shape= X_train.shape[1:],num_classes = num_classes)
model = scaledInitialize(model= model, X_train=X_train)                     #Proposed initialization you want to initialize

initName = 'Proposed_try_1'
optName = 'RMSProp'
lr = 0.0001
opt = RMSprop(lr=lr, decay=1e-6)


model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test),shuffle=True);                                #hist will store everything
earlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='max')
# hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), shuffle=True,callbacks=[earlyStop]);  # With Callback

# ---------------------------------------------- Save History & Model
model.save(os.path.join(save_dir, optName + initName + "_lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name))
HistoryPath = os.path.join(save_dir, optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
saveHistory(HistoryPath, hist)


















# X_train = X_train[0:200]; Y_train = Y_train[0:200];
# X_test = X_test[0:40]; Y_test = Y_test[0:40];

#---------------------------------------------------------------------------------------------------------- Train
lr = 0.00001
opt = RMSprop(lr=lr, decay=1e-6)
# earlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=0, mode='max')
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test),shuffle=True);                              #No Callback
# hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test),shuffle=True, callbacks= [earlyStop]);      #hist will store everything
