#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:13:56 2019

@author: jianan
"""

from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.layers import Conv2D, Input, Flatten, Dense, Dropout, MaxPooling2D
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras import optimizers

import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt

good_examples_dir = "./data/good_data_extend"
bad_examples_dir = "./data/bad_data_extend"
input_shape = (100, 100, 1)
batch_size = 16


def load_data():
    ### load example names ###
    good_examples = os.listdir(good_examples_dir)
    good_examples = [i.split(".")[0] + "_1" for i in good_examples]
    bad_examples = os.listdir(bad_examples_dir)
    bad_examples = [i.split(".")[0] + "_0" for i in bad_examples]

    examples = good_examples + bad_examples
    random.seed(1000)
    random.shuffle(examples)
    
    ### load data and label ###
    train_ratio = 0.9
    data_sum = len(examples)
    train_sum = int(data_sum * train_ratio)
    valid_sum = data_sum - train_sum
    
#    X_train = np.zeros((train_sum, input_shape[0], input_shape[1], input_shape[2]))
#    Y_train = np.zeros((train_sum, 1))
#    X_valid = np.zeros((valid_sum, input_shape[0], input_shape[1], input_shape[2]))
#    Y_valid = np.zeros((valid_sum, 1))
    X_train = []
    Y_train = []
    X_valid = []
    Y_valid = []
    
    for i, data in enumerate(examples):
        data_name = data.split("_")[0] + ".txt"
        data_label = data.split("_")[1]
        if data_label == "1":
            img = np.loadtxt(os.path.join(good_examples_dir, data_name)) / 4096.0
            assert img.shape == input_shape[:2]
        else:
            img = np.loadtxt(os.path.join(bad_examples_dir, data_name)) / 4096.0
            assert img.shape == input_shape[:2]
        
        if i < train_sum:
            X_train.append(img[:, :, np.newaxis])
            Y_train.append([int(data_label)])
        else:
            X_valid.append(img[:, :, np.newaxis])
            Y_valid.append([int(data_label)])
            
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_valid = np.asarray(X_valid)
    Y_valid = np.asarray(Y_valid)
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_valid.shape)
    print(Y_valid.shape)
    
            
    return X_train, Y_train, X_valid, Y_valid

    
def calModel(input_shape):
    input = Input(input_shape, name='input')

    conv1 = Conv2D(16, (3, 3), strides=(1, 1), padding='valid', activation='relu', name='conv1')(input)
    
    conv2 = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu', name='conv2')(conv1)
    
    pl1 = MaxPooling2D(pool_size=(2, 2), padding = 'valid', name = 'pl1')(conv2)

    conv3 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', name='conv3')(pl1)

    pl2 = MaxPooling2D(pool_size=(2, 2), padding = 'valid', name = 'pl2')(conv3)
    
    conv4 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', name='conv4')(pl2)
    
    pl3 = MaxPooling2D(pool_size=(2, 2), padding = 'valid', name = 'pl3')(conv4)
    
    #conv3 = Conv2D(32, (11, 11), strides=(3, 3), padding="valid", activation='relu', name='conv3')(conv2)

    fl = Flatten()(pl3)
    # dp = Dropout(0.2, seed=1000)(fc)
    fc1 = Dense(256, activation='relu', name='fc1')(fl)
    dt1 = Dropout(0.1)(fc1)
    
    fc2 = Dense(128, activation='relu', name='fc2')(dt1)
    dt2 = Dropout(0.1)(fc2)
    
    output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01), name='output')(dt2)

    model = Model(inputs=input, outputs=output, name='calModel')

    return model

    
if __name__ == "__main__":
    # load data
    X_train, Y_train, X_valid, Y_valid = load_data()
    
    # data generator
    train_datagen = ImageDataGenerator()
    valid_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
    valid_generator = valid_datagen.flow(X_valid, Y_valid, batch_size=batch_size)
    
    # load model
    calModel = calModel(input_shape)
    calModel.summary()
    
    # train
    train_steps_per_epoch = math.ceil(len(X_train) / batch_size)
    valid_steps_per_epoch = math.ceil(len(X_valid) / batch_size)
    
    #calModel.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    #sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-04, nesterov=False)
    calModel.compile(optimizer = 'sgd', loss='mean_squared_error', metrics=['accuracy'])
    history = calModel.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=300,
                                     validation_data=valid_generator, validation_steps=valid_steps_per_epoch)
    
    calModel.save("calModel.h5")

    ### draw loss and accuracy ###
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    plt.show()
    
    # lr
    plt.plot(history.history['lr'])
    plt.title('model learning rate')
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.savefig("lr.png")
    plt.show()
    