# Programming Assignment 4
# Alex Yang (ay2344)
# December 8, 2017

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers

# Question for Part 4:
#   Binary classification is easier than classification into 10 categories. This is because
#   having a large number of classes means that each class has more specific defining
#   features; and therefore, the distinctions between classes are more specific and harder
#   to make.


def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    # Construct 1-Hot vectors
    ytrain_1hot = np.zeros((ytrain.shape[0], 10))
    ytrain_1hot[np.arange(ytrain.shape[0]), ytrain.reshape(-1)] = 1

    ytest_1hot = np.zeros((ytest.shape[0], 10))
    ytest_1hot[np.arange(ytest.shape[0]), ytest.reshape(-1)] = 1

    # Normalize data
    xtrain = xtrain/255.0
    xtest = xtest/255.0

    return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_multilayer_nn():
    # Evaluate Output: [1.4848803916931153, 0.48370000000000002]

    nn = Sequential()

    nn.add(Flatten(input_shape=(32, 32, 3)))
    hidden = Dense(units=100, activation="relu", input_shape=(3072,))
    nn.add(hidden)
    output = Dense(units=10, activation="softmax")
    nn.add(output)

    return nn


def train_multilayer_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)        
 

def build_convolution_nn():
    # Evaluate  Output: [0.70585609073638911, 0.755]

    nn = Sequential()

    nn.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)))
    nn.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    nn.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.5))

    nn.add(Flatten())
    nn.add(Dense(units=250, activation="relu"))
    nn.add(Dense(units=100, activation="relu"))
    nn.add(Dense(units=10, activation="softmax"))

    return nn
    

def train_convolution_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)
    

def get_binary_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    # Binary classification - 1 for animal in categories (2, 3, 4, 5, 6, 7)
    ytrain = 1*np.in1d(ytrain, (2, 3, 4, 5, 6, 7))
    ytest = 1*np.in1d(ytest, (2, 3, 4, 5, 6, 7))

    # Normalize data
    xtrain = xtrain / 255.0
    xtest = xtest / 255.0

    return xtrain, ytrain, xtest, ytest


def build_binary_classifier():
    # Evaluate Output: [0.16351540382802487, 0.93820000000000003]

    nn = Sequential()

    nn.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)))
    nn.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    nn.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.5))

    nn.add(Flatten())
    nn.add(Dense(units=250, activation="relu"))
    nn.add(Dense(units=100, activation="relu"))
    nn.add(Dense(units=1, activation="sigmoid"))

    return nn


def train_binary_classifier(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=30, batch_size=32)


if __name__ == "__main__":

    xtrain, ytrain, xtest, ytest = get_binary_cifar10()
    nn = build_binary_classifier()
    train_binary_classifier(nn, xtrain, ytrain)

    print("Evaluation:")
    print(nn.evaluate(xtest, ytest))
