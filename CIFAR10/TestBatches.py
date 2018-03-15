from ReadImages import load_databatch_test
from keras.models import Sequential



data_folder = "/home/ankit/Desktop/Dataset/CIFAR10Dataset/cifar-10-batches-py/"      #Server
idx = 1
img_size = 32
batch_size = 32

X_test,Y_test = load_databatch_test(data_folder = data_folder, idx=idx, img_size= img_size)
score = model.evaluate(X_test,Y_test, batch_size=batch_size)


