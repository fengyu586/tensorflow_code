#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 1:23
# @Author  : Jing
# @FileName: keras0.py
# @IDE: PyCharm
======================================="""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as K


num_classes = 10
img_rows, img_cols = 28, 28
data_path = './dataset/mnist_data'
(train_X, train_Y), (test_X, test_Y) = mnist.load_data(data_path)

if K.image_data_format() == 'channels_first':
    train_X = train_X.reshape(train_X.shape[0], 1, img_rows, img_cols)
    test_X = test_X.reshape(test_X.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)
else:
    train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)
    test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X /= 255.0
test_X /= 255.0

train_Y = keras.utils.to_categorical(train_Y, num_classes)
test_Y = keras.utils.to_categorical(test_Y, num_classes)


model = keras.Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
model.fit(train_X, train_Y, batch_size=128, epochs=20, validation_data=(test_X, test_Y))
score = model.evaluate(test_X, test_Y)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

