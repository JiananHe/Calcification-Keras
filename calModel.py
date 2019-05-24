import keras.backend as K
from keras.layers import Conv2D, Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import LearningRateScheduler
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt

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
            X, Y = load_data(data_files[i: i + batch_size], input_shape)

            yield X, Y


def scheduler(epoch):
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(calModel.optimizer.lr)
        K.set_value(calModel.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(calModel.optimizer.lr)


def calModel(input_shape):
    input = Input(input_shape, name='input')

    conv1 = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1')(input)

    conv2 = Conv2D(16, (7, 7), strides=(2, 2), padding="valid", activation='relu', name='conv2')(conv1)

    #conv3 = Conv2D(32, (11, 11), strides=(3, 3), padding="valid", activation='relu', name='conv3')(conv2)

    fc = Flatten()(conv2)
    dp = Dropout(0.2, seed=1000)(fc)
    output = Dense(1, activation='sigmoid', name='output')(dp)

    model = Model(inputs=input, outputs=output, name='calModel')

    return model


if __name__ == "__main__":
    input_shape = (51, 51, 1)
    batch_size = 16
    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.1
    calModel = calModel(input_shape)
    calModel.summary()

    ### load data name ###
    good_examples = os.listdir(good_examples_dir)
    good_examples = [i.split(".")[0] + "_1" for i in good_examples]
    bad_examples = os.listdir(bad_examples_dir)
    bad_examples = [i.split(".")[0] + "_0" for i in bad_examples]

    examples = good_examples + bad_examples
    random.seed(1000)
    random.shuffle(examples)

    data_sum = len(examples)
    train_sum = int(data_sum * train_ratio)
    valid_sum = int(data_sum * valid_ratio)
    test_sum = data_sum - train_sum - valid_sum

    train_steps_per_epoch = math.floor(train_sum / batch_size)
    valid_steps_per_epoch = math.floor((data_sum - train_sum) / batch_size)

    train_data = examples[:train_sum]
    valid_data = examples[train_sum:valid_sum+train_sum]
    test_data = examples[valid_sum+train_sum:]

    print("all data counts: " + str(data_sum))
    print("train data counts: " + str(len(train_data)))
    print("train data counts: " + str(len(valid_data)))
    print("test data counts: " + str(len(test_data)))

    calModel.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = LearningRateScheduler(scheduler)
    history = calModel.fit_generator(generator=data_generator(train_data, input_shape, batch_size),
                                     steps_per_epoch=train_steps_per_epoch, epochs=150,
                                     validation_data=data_generator(valid_data, input_shape, batch_size),
                                     validation_steps=valid_steps_per_epoch)
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

    ### evaluate ###
    X_test, Y_test = load_data(valid_data, input_shape)
    print("model evaluate: " + str(calModel.evaluate(X_test, Y_test, batch_size=8)))

    Y_hat = calModel.predict(X_test)
    for i in range(len(Y_test)):
        print(str(Y_test[i]) + " : " + str(Y_hat[i]))
