import os
from ReadImages import loadHistory,saveHistory
import pickle
import matplotlib.pyplot as plt
import numpy as np

save_dir = "/home/ankit/Desktop/Dataset/JanResults/CIFAR10/RMSProp/eight_layers/Jan18/DiffSamp"

string = "RMSProp_dataInit_Hist_lr_0.0001_epoch_50_Jan15"

samples = [8,16,32,256,512]

epochs = 50;
for samples in samples:
    new_str = string + str(samples) + "_Samples"

    HistoryPath = os.path.join(save_dir,new_str)
    hist = loadHistory(HistoryPath)

    # -------------------------------------------- Acc
    key = 'acc'

    x = np.arange(len(hist[key]))
    y = hist[key]

    plt.plot(x, y, label="{:d}_Samples".format(samples))
    # -------------------------------------------- Loss
    # key = 'loss'
    #
    # x = np.arange(len(hist[key]))
    # y = hist[key]
    # y = hist[key] / max(hist[key])
    #
    # plt.plot(x, y, label="{:d}_Samples".format(samples))

plt.title("Accuracy Per Epoch ")
plt.xlabel("No of Epochs ({:d})".format(epochs))
plt.ylabel("Accuracy")
plt.legend()
plt.show()


#==============================================================================================


save_dir = "/home/ankit/Desktop/PhaseII/NewResult/CIFAR10/WithDataInit/Jan28"

epochs = 50;
samples = [8,16,32,256,512]

for samples in samples:
    new_str = "RMSProp_dataInit_Hist_lr_0.0001_epoch_50_Jan28_{:d}_Samples".format(samples)
    HistoryPath = os.path.join(save_dir,new_str)
    hist = loadHistory(HistoryPath)

    # -------------------------------------------- Acc
    key = 'acc'

    x = np.arange(len(hist[key]))
    y = hist[key]

    plt.plot(x, y, label="{:d}_Samples".format(samples))

#-------------------------------------------------------- Plot Other Init

save_dir = "/home/ankit/Desktop/PhaseII/NewResult/CIFAR10/FebResults/Feb1/eight_layers"         #where the other files are stored


initDict = {'LeCun':'lecun_uniform', 'Glorot Normal':'glorot_normal','Glorot Uniform':'glorot_uniform','He Normal':'he_normal','He Uniform':'he_uniform'}
# initDict = {'LeCun':'lecun_uniform', 'He Normal':'he_normal','He Uniform':'he_uniform'}
lr = 0.001; epochs = 50;
date = "Feb1"

for name,initName in initDict.items():

#--------------------------------------------------------------------------------------------------Load History
    try:
        HistoryPath = os.path.join(save_dir,"RMSProp_"+ initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + date)
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

















#=========================================================================== Plot both Standard and Proposed using one key for all acc, loss, val_loss
key = 'acc';
keyDict = {'acc':'Accuracy', 'loss':'Loss','val_loss':'UnNormalized Loss '}

save_dir = "/home/ankit/Desktop/PhaseII/NewResult/CIFAR10/WithDataInit/Jan28"

epochs = 50;
samples = [8,16,32,256,512]
# samples = [32,512]

for samples in samples:
    new_str = "RMSProp_dataInit_Hist_lr_0.0001_epoch_50_Jan28_{:d}_Samples".format(samples)
    HistoryPath = os.path.join(save_dir,new_str)
    hist = loadHistory(HistoryPath)

    # -------------------------------------------- Acc
    key = key;

    x = np.arange(len(hist[key]))
    y = hist[key]

    plt.plot(x, y, label="{:d}_Samples".format(samples))

#-------------------------------------------------------- Plot Other Init

save_dir = "/home/ankit/Desktop/PhaseII/NewResult/CIFAR10/FebResults/Feb1/eight_layers"         #where the other files are stored


initDict = {'LeCun':'lecun_uniform', 'Glorot Normal':'glorot_normal','Glorot Uniform':'glorot_uniform','He Normal':'he_normal','He Uniform':'he_uniform'}
# initDict = {'LeCun':'lecun_uniform', 'He Normal':'he_normal','He Uniform':'he_uniform'}
lr = 0.001; epochs = 50;
date = "Feb1"

for name,initName in initDict.items():

#--------------------------------------------------------------------------------------------------Load History
    try:
        HistoryPath = os.path.join(save_dir,"RMSProp_"+ initName + "_Hist_" + "lr_{:1}_epoch_{:1d}_".format(lr, epochs) + date)
        hist = loadHistory(HistoryPath)

        #--------------------------------------------------------------------------------------------------
        key = key;

        x = np.arange(len(hist[key]))
        y = hist[key]

        plt.plot(x,y, label= name)
    except FileNotFoundError: next;
plt.title("{:s} Per Epoch ".format(keyDict[key]))
plt.xlabel("No of Epochs ({:d})".format(epochs))
plt.ylabel("{:s}".format(keyDict[key]))
plt.legend()
plt.show()
