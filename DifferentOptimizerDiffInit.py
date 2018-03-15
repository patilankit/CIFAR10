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

base_dir = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/"

data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/"      #Server
idx = 1
img_size = 32



batch_size = 32
num_classes = 10
epochs = 50
save_dir = os.path.join(base_dir,'SavedModels/DiffOpt/')
model_name = 'Keras_CIFAR10_HeInit_v1.h5'
history_name = 'Keras_CIFAR10_HeInit_v1'


X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)
#ModelKeys
initDict = {'LeCun':'lecun_uniform', 'GlorotNorm':'glorot_normal','GlorotUni':'glorot_uniform','HeNorm':'he_normal','HeUni':'he_uniform'}


#============================================================================= Optimizer with their defauls
opt1 = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
opt2 = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
opt3 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt4 = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt5 = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
opt6 = RMSprop(lr=0.0001, decay=1e-6)

#============================================================================================================

optDict = {'Adagrad':opt1,'Adadelta':opt2,'Adam':opt3,'Adamax':opt4,'Nadam':opt5, 'RMSProp':opt6}

lr = 0.0001                 #for now unable to assign the value inside the loop


def trainOptInit(X_train,X_test,Y_train,Y_test,initDict,optDict = {'RMSProp':RMSprop(lr=0.0001, decay=1e-6)},mode ='full',saveHist = True):
    if mode == 'test':     #Just test over a few values
        X_train = X_train[0:200];Y_train = Y_train[0:200];
        X_test = X_test[0:40];Y_test = Y_test[0:40];


    for optName,opt in optDict.items():
    #---------------------------------------------- For Different Init and Optimizers paras
        for key,initName in initDict.items():
            print(" Training for {:s} initialization with {:s} as optimizer".format(key,optName))
            model = get_model(initName, input_shape=X_train.shape[1:], num_classes=num_classes)
            # lr = 0.0001
            # opt = RMSprop(lr=lr, decay=1e-6)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            # hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test),shuffle=True);                                #hist will store everything
            #earlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='max')
            hist = model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test),shuffle=True, callbacks= [earlyStop]);      #With Callback
            if (saveHist == True):
                #---------------------------------------------- Save History & Model
                model.save(os.path.join(save_dir, optName+initName + "_lr_{:1}_epoch_{:1d}_".format(lr,epochs) +model_name))
                HistoryPath = os.path.join(save_dir,optName + "_" + initName + "_Hist_"+"lr_{:1}_epoch_{:1d}_".format(lr,epochs) + model_name)
                saveHistory(HistoryPath,hist)
    return 1

# start = timeit.default_timer()
# # res = trainOptInit(X_train,X_test,Y_train,Y_test,initDict=initDict,optDict = optDict, mode ='test')           #test first
# res = trainOptInit(X_train,X_test,Y_train,Y_test,initDict=initDict,optDict = optDict, mode ='full')           #test first
# stop = timeit.default_timer()
# execTime = stop - start
#
# print(" The program Took {:f} time in seconds :P".format(execTime))



#-------------------------------------------------------------------- Test version
res = trainOptInit(X_train,X_test,Y_train,Y_test,initDict=initDict,optDict = optDict, mode ='test',saveHist=False)           #test first


