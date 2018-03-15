from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.models import Sequential
from keras.metrics import categorical_accuracy
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout,GaussianNoise
from keras.optimizers import SGD,RMSprop,Adadelta,Adagrad,Adam,Nadam,Adamax
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
from dependant_initialization_v1 import initialize
import timeit,datetime,time
from Jan_Initialize_v1 import jan_initialize
import matplotlib.pyplot as plt


base_dir = "/home/ankit/Desktop/Dataset/JanResults/CIFAR10/RMSProp/eight_layers";        #save results here...

data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/"      #Server
idx = 1
img_size = 32



batch_size = 32
num_classes = 10
epochs = 100
save_dir = os.path.join(base_dir,'Jan18')
model_name = 'Jan18'
# history_name = 'Keras_CIFAR10_HeInit_v1'


X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)
# X_train = X_train[0:200];Y_train = Y_train[0:200];
# X_test = X_test[0:40];Y_test = Y_test[0:40];

model = get_model()

samples = 1024;

opt = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model = jan_initialize(model,X_train,samples)              #check the working function :)
#
#
hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test));
#
optName = "RMSProp"; initName = "dataInit";
HistoryPath = os.path.join(save_dir,optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_Jan15".format(lr, epochs) + str(samples) + "_Samples")
saveHistory(HistoryPath, hist)

#
#
# # model.save(os.path.join(save_dir, optName + initName + "_lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name))
# HistoryPath = os.path.join(save_dir,optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
# saveHistory(HistoryPath, hist)
#






epochs = 50;
optName = 'RMSProp';
lr = 0.001;
initDict = {'Proposed': 'data_init','LeCun':'lecun_uniform', 'GlorotNorm':'glorot_normal','GlorotUni':'glorot_uniform','HeNorm':'he_normal','HeUni':'he_uniform'}

for key, initName in initDict.items():
    print(" Training for {:s} initialization with {:s} as optimizer".format(key, optName));

    if initName == 'data_init':
        model = jan_initialize(model, X_train, 8)  # check the working function :)
    else:
        model = get_model(initName, input_shape=X_train.shape[1:], num_classes=num_classes)
    lr = 0.001;

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test));  # With Callback

    HistoryPath = os.path.join(save_dir, optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr,epochs) + model_name)
    saveHistory(HistoryPath, hist)




#------------------------------------------- DataInit

opt = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model = jan_initialize(model,X_train,8)              #check the working function :)
hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test));

optName = "RMSProp"; initName = "dataInit";
HistoryPath = os.path.join(save_dir,optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_Jan15".format(lr, epochs))
saveHistory(HistoryPath, hist)





#------------------------------------------------  Graphs

optName = 'RMSProp'; lr = 0.0001;
initDict = {'Proposed':'dataInit','LeCun':'lecun_uniform', 'GlorotNorm':'glorot_normal','GlorotUni':'glorot_uniform','HeNorm':'he_normal','HeUni':'he_uniform'}


for name, initName in initDict.items():
    # ----------------------------------------------- Load History
    try:
        HistoryPath = os.path.join(save_dir, optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr,epochs) + model_name)
        hist = loadHistory(HistoryPath)

        # --------------------------------------------------------------------------------------------------
        key = 'acc'

        x = np.arange(len(hist[key]))
        y = hist[key]

        plt.plot(x, y, label=name)
    except FileNotFoundError: next;
plt.title("Accuracy Over Initializations for " + optName + " Optimization")
plt.xlabel("No of Epochs ({:d})".format(epochs))
plt.ylabel("Accuracy")
plt.legend()
plt.show()



#-------------------------------------------------------------- Training for both Different Samples and all others
#
# samples = [8,16,32,256,512]
# optName = "RMSProp"; initName = "dataInit";
#
# save_dir = os.path.join(save_dir,'DiffSamp')


base_dir = "/home/ankit/Desktop/Dataset/JanResults/CIFAR10/RMSProp/eight_layers";        #save results here...
save_dir = os.path.join(base_dir,'Jan28')


X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)
# X_train = X_train[0:200];Y_train = Y_train[0:200];
# X_test = X_test[0:40];Y_test = Y_test[0:40];

def DiffSamp(samples):

    for samples in samples:

        model = get_model();
        opt = RMSprop(lr=0.0001, decay=1e-6);
        optName = "RMSProp";initName = "dataInit";
        lr = 0.0001;epochs = 50;

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        model = jan_initialize(model,X_train,samples)              #check the working function :)
        hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test));
        #

        HistoryPath = os.path.join(save_dir,optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_Jan28_".format(lr, epochs) + str(samples) + "_Samples")
        saveHistory(HistoryPath, hist)
    return






epochs = 50;
optName = 'RMSProp';
lr = 0.001; num_classes =10;batch_size = 32;

initDict = {'Proposed': 'data_init', 'LeCun': 'lecun_uniform', 'GlorotNorm': 'glorot_normal',
            'GlorotUni': 'glorot_uniform', 'HeNorm': 'he_normal', 'HeUni': 'he_uniform'}

for key, initName in initDict.items():
    print(" Training for {:s} initialization with {:s} as optimizer".format(key, optName));

    if initName == 'data_init':
        samples = [8,16,32,256,512];
        DiffSamp(samples);
        next;

    model = get_model(initName, input_shape=X_train.shape[1:], num_classes=num_classes)
    lr = 0.001;
    opt = RMSprop(lr=lr, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(X_test, Y_test));  # With Callback

    HistoryPath = os.path.join(save_dir, optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr,epochs) + "Jan28")
    saveHistory(HistoryPath, hist)




#--------------------------------------------------------------------- Only the standard Inits

base_dir = "/home/ankit/Desktop/Dataset/FebResults/CIFAR10/";        #save results here...
save_dir = os.path.join(base_dir,'Feb1/eight_layers')
date = 'Feb1';

X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)


epochs = 50;
optName = 'RMSProp';
lr = 0.001; num_classes =10;batch_size = 32;

initDict = { 'LeCun': 'lecun_uniform', 'GlorotNorm': 'glorot_normal',
            'GlorotUni': 'glorot_uniform', 'HeNorm': 'he_normal', 'HeUni': 'he_uniform'}

for key, initName in initDict.items():
    print(" Training for {:s} initialization with {:s} as optimizer".format(key, optName));

    model = get_model(initName, input_shape=X_train.shape[1:], num_classes=num_classes)
    lr = 0.001;
    opt = RMSprop(lr=lr, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(X_test, Y_test));  # With Callback

    HistoryPath = os.path.join(save_dir, optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr,epochs) + date)
    saveHistory(HistoryPath, hist)





