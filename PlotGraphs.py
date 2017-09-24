import os
from ReadImages import loadHistory,saveHistory
import pickle
import matplotlib.pyplot as plt
import numpy as np

base_dir = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/"
save_dir = os.path.join(base_dir,'SavedModels')
model_name = 'Keras_CIFAR10_trainModel_v1.h5'
idx = 1



hist = {}
for idx in range(1,6,1):
    load_model_path = os.path.join(save_dir, str('hist_') + str(idx) + model_name);
    hist["history{:0}".format(idx)] = loadHistory(load_model_path)                   #Load History

histCopy = hist


#-----------------------------------------------------------------------------------------------  For all
all_para = hist['history1'][key] + hist['history2'][key]+ hist['history3'][key] + hist['history4'][key] + hist['history5'][key]
#----------------------------------------------------------------------------------------------- Accuarcy

key = 'acc'
all_para = hist['history1'][key] + hist['history2'][key]+ hist['history3'][key] + hist['history4'][key] + hist['history5'][key]

x = np.arange(len(all_para))
y = all_para

plt.plot(x,y, 'r-')
plt.title("Accuracy per batch ")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
plt.show()

#----------------------------------------------------------------------------------------------- Loss

key = 'loss'
all_para = hist['history1'][key] + hist['history2'][key]+ hist['history3'][key] + hist['history4'][key] + hist['history5'][key]

x = np.arange(len(all_para))
y = all_para

plt.plot(x,y, 'r-')
plt.title("Loss per Batch ")
plt.xlabel("No of Epochs")
plt.ylabel("Loss")
plt.show()


ax = plt.figure()



#----------------------------------------------------------------------------------------------- No of Batch
hist = hist['history1']
#----------------------------------------------------------------------------------------------- Accuracy
key = 'acc'

x = np.arange(len(hist[key]))
y = hist[key]

plt.plot(x,y, 'r-')
plt.title("Accuracy Per Epoch")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
# plt.show()

#----------------------------------------------------------------------------------------------- Loss

key = 'loss'
x = np.arange(len(hist[key]))
y = hist[key]

plt.plot(x,y, 'r-')
plt.title("Loss Per Epoch")
plt.xlabel("No of Epochs")
plt.ylabel("Loss")
plt.show()
#=======================================================================================================================

acc = hist['acc']
loss = hist['loss']/max(hist['loss'])
epochs = np.arange(len(hist['acc']))

plt.plot(epochs,acc,'r',label = 'Acc')
plt.plot(epochs,loss,'b', label = 'loss')
plt.xlabel("Epochs")
plt.ylabel("Loss and Accuracy")
plt.title("Progress Per Epoch")
plt.legend()
plt.show()





