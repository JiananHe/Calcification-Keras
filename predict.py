#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:20:20 2019

@author: jianan
"""

from keras.models import load_model
import numpy as np
import os
import random


good_examples_dir = "./data/good_data_extend"
bad_examples_dir = "./data/bad_data_extend"
input_shape = (51, 51, 1)

    
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


if __name__ == "__main__":
    calModel = load_model("calModel.h5")
    random.seed(1000)
    
    # load data
    X_train, Y_train, X_valid, Y_valid = load_data()
    
    thresh = 0.5
    test_num = 40
    right_count = 0
    print("model evaluate: " + str(calModel.evaluate(X_valid, Y_valid, batch_size=8)))
    
    preds = calModel.predict(X_train[0:test_num, :, :, :])
    for i in range(test_num):
        res = 0 if preds[i] < thresh else 1
        print(str(Y_train[i]) + " " + str(res))
        if res == Y_train[i]:
            right_count = right_count + 1
            
    print("right num: " + str(right_count))