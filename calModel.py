from keras.layers import Conv2D, Input, Flatten, Dense
from keras.models import Model
import os
import random
import numpy as np

def load_data(input_shape):
    

def calModel(input_shape):
    input = Input(input_shape, name = 'input')
    
    conv1 = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1')(input)
    # bn1 = BatchNormalization()
    
    conv2 = Conv2D(16, (7, 7), strides=(2, 2), padding="valid", activation='relu', name='conv2')(conv1)
    
    #conv3 = Conv2D(32, (7, 7), strides=(2, 2), padding="valid", activation='relu', name='conv3')(conv2)
    
    #conv4 = Conv2D(64, (11, 11), strides=(2, 2), padding="valid", activation='relu', name='conv4')(conv3)
    
    #conv5 = Conv2D(128, (7, 7), strides=(1, 1), padding="valid", activation='relu', name='conv5')(conv4)
    
    fc = Flatten()(conv2)
    fc = Dense(1, activation='sigmoid', name='fc')(fc)
    #output = Conv2D(1, (1, 1), strides=(1, 1), padding="valid", activation='sigmoid', name='output')(conv5)
     
    model = Model(inputs=input, outputs=fc, name='calModel')

    return model


if __name__ == "__main__":
    input_shape = (51, 51, 1)
    calModel = calModel(input_shape)
    calModel.summary()
    
    ### load data ###
    good_examples_dir = "/home/jianan/Projects/calcification/data/good_data"
    bad_examples_dir = "/home/jianan/Projects/calcification/data/bad_data"
    
    good_examples = os.listdir(good_examples_dir)
    good_examples = [i.split(".")[0]+"_1" for i in good_examples]
    bad_examples = os.listdir(bad_examples_dir)
    bad_examples = [i.split(".")[0]+"_0" for i in bad_examples]
    
    examples = good_examples + bad_examples
    random.seed(1000)
    random.shuffle(examples)
    
    X_train_
    
    X_train = np.zeros((len(examples), input_shape[0], input_shape[1], input_shape[2]))
    Y_train = np.zeros((len(examples), 1))
    
    for i, example in enumerate(examples):
        examples_name = example.split("_")[0] + ".txt"
        examples_label = example.split("_")[1]
        
        if examples_label == "1":
            img = np.loadtxt(os.path.join(good_examples_dir, examples_name)) / 4096.0
            assert img.shape == input_shape[:2]
            X_train[i, :, :, :] = img[:, :, np.newaxis]
            Y_train[i, :] = int(examples_label)
        else:
            img = np.loadtxt(os.path.join(bad_examples_dir, examples_name)) / 4096.0
            assert img.shape == input_shape[:2]
            X_train[i, :, :, :] = img[:, :, np.newaxis]
            Y_train[i, :] = int(examples_label)
        
    
    print(X_train.shape)
    print(Y_train.shape)
    
# =============================================================================
#     calModel.compile('adam', loss='mean_squared_error', metrics=['accuracy'])
#     calModel.fit(x=X_train, y=Y_train, epochs=100, validation_data=(X_train, Y_train), steps_per_epoch=10, validation_steps=10)
#     calModel.fit()
#     calModel.save("calModel.h5")
# =============================================================================
    
    
    # print(calModel.predict(X_train[0:10, :, :, :]))