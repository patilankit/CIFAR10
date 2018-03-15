import os
from ReadImages import loadHistory,saveHistory
import pickle
import matplotlib.pyplot as plt
import numpy as np

base_dir = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/"
save_dir = os.path.join(base_dir,'SavedModels/HeNormInit')
model_name = 'Keras_CIFAR10_HeInit_v1.h5'
idx = 1


lr = 0.0001
epochs = 200

initDict = {'LeCun':'lecun_uniform', 'Glorot Normal':'glorot_normal','Glorot Uniform':'glorot_uniform','He Normal':'he_normal','He Uniform':'he_uniform'}

# initName = 'glorot_normal'

for name,initName in initDict.items():

#--------------------------------------------------------------------------------------------------Load History
    try:
        HistoryPath = os.path.join(save_dir, initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
        hist = loadHistory(HistoryPath)

        #--------------------------------------------------------------------------------------------------
        key = 'acc'

        x = np.arange(len(hist[key]))
        y = hist[key]

        plt.plot(x,y, 'r-')
        plt.title("Accuracy Per Epoch " + name)
        plt.xlabel("No of Epochs ({:d})".format(epochs))
        plt.ylabel("Accuracy")
        plt.show()
    except FileNotFoundError: next;


#======================================== Accuracy for All ================================================
#--------------------------------------------------------------------------------------------- Overlapping graphs

for name,initName in initDict.items():

#--------------------------------------------------------------------------------------------------Load History
    try:
        HistoryPath = os.path.join(save_dir, initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
        hist = loadHistory(HistoryPath)

        #--------------------------------------------------------------------------------------------------
        key = 'acc'

        x = np.arange(len(hist[key]))
        y = hist[key]

        plt.plot(x,y, label= name)
    except FileNotFoundError: next;
plt.title("Accuracy Per Epoch ")
plt.xlabel("No of Epochs ({:d})".format(epochs))
plt.ylabel("Accuracy")
plt.legend()
plt.show()




#======================================== Loss for All ================================================
#--------------------------------------------------------------------------------------------- Overlapping graphs

for name,initName in initDict.items():

#--------------------------------------------------------------------------------------------------Load History
    try:
        HistoryPath = os.path.join(save_dir, initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
        hist = loadHistory(HistoryPath)

        #--------------------------------------------------------------------------------------------------
        key = 'loss'

        x = np.arange(len(hist[key]))
        y = hist[key]
        y = hist[key]/max(hist[key])

        plt.plot(x,y, label= name)
    except FileNotFoundError: next;
plt.title("Loss Per Epoch ")
plt.xlabel("No of Epochs ({:d})".format(epochs))
plt.ylabel("Loss")
plt.legend()
plt.show()



#======================================== Loss and Accuracy for All ================================================
#--------------------------------------------------------------------------------------------- Overlapping graphs

for name,initName in initDict.items():

#--------------------------------------------------------------------------------------------------Load History
    try:
        HistoryPath = os.path.join(save_dir, initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + model_name)
        hist = loadHistory(HistoryPath)

        #-------------------------------------------- Acc
        key = 'acc'

        x = np.arange(len(hist[key]))
        y = hist[key]
        y = hist[key]

        plt.plot(x,y, label= name)
        #-------------------------------------------- Loss
        key = 'loss'

        x = np.arange(len(hist[key]))
        y = hist[key]
        y = hist[key]/max(hist[key])

        plt.plot(x,y, label= name)
    except FileNotFoundError: next;
plt.title("Accuracy Per Epoch ")
plt.xlabel("No of Epochs ({:d})".format(epochs))
plt.ylabel("Loss and Accuracy")
plt.legend()
plt.show()


