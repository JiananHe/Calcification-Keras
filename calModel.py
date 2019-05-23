from keras.layers import Conv2D, Input, Flatten, Dense
from keras.models import Model
import os
import random
import numpy as np
import math

good_examples_dir = ".\\data\\good_data"
bad_examples_dir = ".\\data\\bad_data"


def load_data(data_files, input_shape):
    X = np.zeros((len(data_files), input_shape[0], input_shape[1], input_shape[2]))
    Y = np.zeros((len(data_files), 1))

    for i, data in enumerate(data_files):
        data_name = data.split("_")[0] + ".txt"
        data_label = data.split("_")[1]

        if data_label == "1":
            img = np.loadtxt(os.path.join(good_examples_dir, data_name)) / 4096.0
            assert img.shape == input_shape[:2]
        else:
            img = np.loadtxt(os.path.join(bad_examples_dir, data_name)) / 4096.0
            assert img.shape == input_shape[:2]

        X[i, :, :, :] = img[:, :, np.newaxis]
        Y[i, :] = int(data_label)

    return X, Y


def data_generator(data_files, input_shape, batch_size):
    while True:
        for i in range(0, len(data_files), batch_size):
            X, Y = load_data(data_files[i : i + batch_size], input_shape)

            yield X, Y


def calModel(input_shape):
    input = Input(input_shape, name = 'input')
    
    conv1 = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1')(input)
    
    conv2 = Conv2D(16, (7, 7), strides=(2, 2), padding="valid", activation='relu', name='conv2')(conv1)
    
    fc = Flatten()(conv2)
    fc = Dense(1, activation='sigmoid', name='fc')(fc)
     
    model = Model(inputs=input, outputs=fc, name='calModel')

    return model


if __name__ == "__main__":
    input_shape = (51, 51, 1)
    batch_size = 10
    train_valid_ratio = 0.8
    calModel = calModel(input_shape)
    calModel.summary()
    
    ### load data name ###
    
    good_examples = os.listdir(good_examples_dir)
    good_examples = [i.split(".")[0]+"_1" for i in good_examples]
    bad_examples = os.listdir(bad_examples_dir)
    bad_examples = [i.split(".")[0]+"_0" for i in bad_examples]
    
    examples = good_examples + bad_examples
    random.seed(1000)
    random.shuffle(examples)

    data_sum = len(examples)
    train_sum = int(data_sum * train_valid_ratio)
    train_steps_per_epoch = math.floor(train_sum / batch_size)
    valid_steps_per_epoch = math.floor((data_sum - train_sum) / batch_size)
    
    train_data = examples[:train_sum]
    valid_data = examples[train_sum:]

    print("train data counts: " + str(len(train_data)))
    print("train data counts: " + str(len(valid_data)))

    calModel.compile('adam', loss='mean_squared_error', metrics=['accuracy'])
    calModel.fit_generator(generator=data_generator(train_data, input_shape, batch_size),
                           steps_per_epoch = train_steps_per_epoch, epochs=100,
                           validation_data=data_generator(valid_data, input_shape, batch_size),
                           validation_steps=valid_steps_per_epoch)
    calModel.save("calModel.h5")