import csv
import pickle





def loadHistory(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def writeDict2CSV(file,Dict):
    with open(file, 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, Dict.keys())
        w.writeheader()
        # w.writerow(Dict)




# histPath = '/home/ankit/Desktop/DDP/Projects/CIFAR10/ProjectData/HeNormInit/glorot_normal_Hist_lr_0.0001_epoch_100_Keras_CIFAR10_HeInit_v1.h5'
#
# dict = loadHistory(histPath)
#
# writeDict2CSV('/home/ankit/Desktop/DDP/Projects/CIFAR10/ProjectData/HeNormInit/CSVFiles/glorot_normal_Hist_lr_0.0001_epoch_100_Keras_CIFAR10_HeInit_v1.csv', Dict=dict)
