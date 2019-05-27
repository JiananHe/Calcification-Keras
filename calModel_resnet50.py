#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:35:00 2019

@author: jianan
"""

from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, Flatten, Dense, Dropout, MaxPooling2D
import numpy as np
import random
import os
import matplotlib.pyplot as plt

good_examples_dir = "./data/good_data_extend"
bad_examples_dir = "./data/bad_data_extend"
input_shape = (100, 100, 3)
batch_size = 4


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
    train_ratio = 0.8
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
            img = img[:, :, np.newaxis]
            img = np.tile(img, 3)
            assert img.shape == input_shape
        else:
            img = np.loadtxt(os.path.join(bad_examples_dir, data_name)) / 4096.0
            img = img[:, :, np.newaxis]
            img = np.tile(img, 3)
            assert img.shape == input_shape
        
        if i < train_sum:
            X_train.append(img)
            Y_train.append([int(data_label)])
        else:
            X_valid.append(img)
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



def ResNet50_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling=None, input_shape=input_shape, classes=2)
    for layer in base_model.layers:
        layer.trainable = False
 
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(1, activation='sigmoid')(x)
 
    model = Model(inputs=base_model.input, outputs=predictions)
    #sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 
    return model
    
    
if __name__ == "__main__":
    # load data
    X_train, Y_train, X_valid, Y_valid = load_data()
    
    # data generator
    train_datagen = ImageDataGenerator(rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    valid_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
    valid_generator = valid_datagen.flow(X_valid, Y_valid, batch_size=batch_size)
    
    # load model
    model = ResNet50_model(input_shape)
    model.summary()
    
    # train
    history = model.fit_generator(train_generator, steps_per_epoch=40, epochs=500,
                                     validation_data=valid_generator, validation_steps=20)
    
    model.save("resnet-cal.h5")
    

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
    
    
