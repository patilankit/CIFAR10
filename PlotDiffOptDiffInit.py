import os
from ReadImages import loadHistory,saveHistory
import pickle
import matplotlib.pyplot as plt
import numpy as np

base_dir = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/"

data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/"      #Server
idx = 1
img_size = 32



batch_size = 32
num_classes = 10
epochs = 200
save_dir = os.path.join(base_dir,'SavedModels/DiffOpt/')
model_name = 'Keras_CIFAR10_HeInit_v1.h5'
history_name = 'Keras_CIFAR10_HeInit_v1'


# X_train,Y_train,X_test,Y_test = load_data_full(data_folder=data_folder, batch_no = 5,img_size=img_size)
#ModelKeys
initDict = {'LeCun':'lecun_uniform', 'GlorotNorm':'glorot_normal','GlorotUni':'glorot_uniform','HeNorm':'he_normal','HeUni':'he_uniform'}

optDict = {'Adagrad':1,'Adadelta':2,'Adam':3,'Adamax':4,'Nadam':5}


lr = 0.0001
epochs = 200
optName = 'Adagrad'
initName = 'lecun_uniform'


#======================================================================================= Accuracy for all

for optName, opt in optDict.items():
    # ---------------------------------------------- For Different Init and Optimizers paras
    for name, initName in initDict.items():
    #----------------------------------------------- Load History
        try:
            HistoryPath = os.path.join(save_dir,optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
            hist = loadHistory(HistoryPath)

            #--------------------------------------------------------------------------------------------------
            key = 'acc'

            x = np.arange(len(hist[key]))
            y = hist[key]

            plt.plot(x,y, label= name)
        except FileNotFoundError: next;
    plt.title("Accuracy Over Initializations for " + optName + " Optimization")
    plt.xlabel("No of Epochs ({:d})".format(epochs))
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


#======================================================================================= Loss for all
for optName, opt in optDict.items():
    # ---------------------------------------------- For Different Init and Optimizers paras
    for name, initName in initDict.items():
    #----------------------------------------------- Load History
        try:
            HistoryPath = os.path.join(save_dir,optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
            hist = loadHistory(HistoryPath)

            #--------------------------------------------------------------------------------------------------
            key = 'loss'

            x = np.arange(len(hist[key]))
            y = hist[key]

            plt.plot(x,y, label= name)
        except FileNotFoundError: next;
    plt.title("Loss Over Initializations for " + optName + " Optimization")
    plt.xlabel("No of Epochs ({:d})".format(epochs))
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


#======================================================================================= UnNormalized Loss for all
for optName, opt in optDict.items():
    # ---------------------------------------------- For Different Init and Optimizers paras
    for name, initName in initDict.items():
    #----------------------------------------------- Load History
        try:
            HistoryPath = os.path.join(save_dir,optName + "_" + initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
            hist = loadHistory(HistoryPath)

            #--------------------------------------------------------------------------------------------------
            key = 'val_loss'

            x = np.arange(len(hist[key]))
            y = hist[key]

            plt.plot(x,y, label= name)
        except FileNotFoundError: next;
    plt.title("UnNormalized Loss Over Initializations for " + optName + " Optimization")
    plt.xlabel("No of Epochs ({:d})".format(epochs))
    plt.ylabel("UnNormalized Loss")
    plt.legend()
    plt.show()







z = hist[key]
x = np.arange(len(z))



plt.plot(x, z)
plt.title("Accuracy for Data Dependant Initialization over " + optName + " Optimization for CIFAR-10")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



x = np.arange(len(z))

plt.plot(x, z)
plt.title("Loss for Data Dependant Initialization over " + optName + " Optimization for CIFAR-10")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

