import numpy as np;
import os
from keras.models import Model
import numpy as np;
import random
from computeFans import compute_fans
from ReadImages import load_databatch,load_databatch_test,load_data_full
from HeInitialize import get_model


base_dir = "/home/ankit/Desktop/Dataset/JanResults/CIFAR10";        #save results here...

data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/";      #Server
img_size = 32;
batch_size = 32;
num_classes = 10;
epochs = 10;


#------------------- Get the X_train and Model for temporary use
# X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)
# model = get_model();
# del Y_train, X_test,Y_test;                                                 #Freeing up the space
#---------------------------------------------------------------------------------------------------



def whiten(mat, new_var):

    mean = np.mean(mat);
    var = np.var(mat);
    #----------------------- Mean and Variance Scaling for the reqd Var.
    mat = mat - mean;
    mat = (np.sqrt(new_var/var) * mat);

    return mat





#----------------------------------------------------------------------------- Test
# mat = np.random.normal(0,1,(3,2))
#
# mat_new = whiten(mat,0.2)
