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

if __name__ == "__main__":
    calModel = load_model("calModel.h5")
    
    good_examples_dir = "/home/jianan/Projects/calcification/data/good_data"
    bad_examples_dir = "/home/jianan/Projects/calcification/data/bad_data"
    
    good_examples = os.listdir(good_examples_dir)
    print(good_examples)
    print(good_examples)
    bad_examples = os.listdir(bad_examples_dir)
    
    for i in range(10):
        X = np.loadtxt(os.path.join(good_examples_dir, good_examples[i])) / 4096.0
        X = X[np.newaxis, :, :, np.newaxis]
        print(calModel.predict(X))
        
        X = np.loadtxt(os.path.join(bad_examples_dir, bad_examples[i])) / 4096.0
        X = X[np.newaxis, :, :, np.newaxis]
        print(calModel.predict(X))