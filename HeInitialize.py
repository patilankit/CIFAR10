from keras.models import Sequential
from keras.metrics import categorical_accuracy
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Input







# def get_model(init_name='he_normal',input_shape=(32, 32, 3),num_classes = 10):
#
#     model = Sequential()
#
#     model.add(Conv2D(32, (3, 3), padding='same',input_shape= input_shape,data_format="channels_last",kernel_initializer=init_name))
#     model.add(Activation('relu'))
#     model.add(Conv2D(32, (3, 3),kernel_initializer=init_name))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, (3, 3), padding='same',kernel_initializer=init_name))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3),kernel_initializer=init_name))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(512,kernel_initializer=init_name))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes,kernel_initializer=init_name))
#     model.add(Activation('softmax'))
#
#     return model
#

# initDict = {'LeCun':'lecun_uniform', 'GlorotNorm':'glorot_normal','GlorotUni':'glorot_uniform','HeNorm':'he_normal','HeUni':'he_uniform'}
# num_classes = 10
# model = get_model(initDict['HeNorm'],input_shape= X_train.shape[1:],num_classes = num_classes)



#---------------------------------------------------------- to get the network configurations
# layers = model.layers
# for i in range(0,len(layers)):
#
#     try:
#         # print(layers[i].get_output_at(0) )                #print the output shape
#         # print(layers[i].get_weights()[0].shape)           #print the weights
#     except IndexError: print("Not A valid layer")
#



def get_model(init_name='he_normal',input_shape=(32, 32, 3),num_classes = 10):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape= input_shape,data_format="channels_last",kernel_initializer=init_name))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3),padding='same', kernel_initializer=init_name))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3),padding='same', kernel_initializer=init_name))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512,kernel_initializer=init_name))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,kernel_initializer=init_name))
    model.add(Activation('softmax'))

    return model



def get_model_activation(act = 'tanh',init_name='he_normal',input_shape=(32, 32, 3),num_classes = 10):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape= input_shape,data_format="channels_last",kernel_initializer=init_name))
    model.add(Activation(act))
    model.add(Conv2D(32, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation(act))
    model.add(Conv2D(64, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation(act))
    model.add(Conv2D(128, (3, 3),padding='same',kernel_initializer=init_name))
    model.add(Activation(act))

    model.add(Conv2D(256, (3, 3),padding='same', kernel_initializer=init_name))
    model.add(Activation(act))
    model.add(Conv2D(256, (3, 3),padding='same', kernel_initializer=init_name))
    model.add(Activation(act))

    model.add(Flatten())
    model.add(Dense(512,kernel_initializer=init_name))
    model.add(Activation(act))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,kernel_initializer=init_name))
    model.add(Activation('softmax'))

    return model

