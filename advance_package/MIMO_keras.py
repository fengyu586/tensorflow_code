#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 12:04
# @Author  : Jing
# @FileName: MIMO_keras.py
# @IDE: PyCharm
======================================="""
import keras
from tflearn.layers.core import fully_connected
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model


data_path = './dataset/mnist_data/'
(train_X, train_Y), (test_X, test_Y) = mnist.load_data(data_path)


input1 = Input(shape=(784, ), name='input1')
input2 = Input(shape=(10, ), name='input2')

x = Dense(1, activation='relu')(input1)
output1 = Dense(10, activation='softmax', name='output1')(x)
y = keras.layers.concatenate([x, input2])
output2 = Dense(10, activation='softmax', name='output2')(y)
model = Model(inputs=[input1, input2], outputs=[output1, output2])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(), loss_weights=[1, 0.1],
              metrics=['accuracy'])
model.fit([train_X, train_Y], [train_Y, train_Y], batch_size=128,
          epochs=20, validation_data=([test_X, test_Y], [test_Y, test_Y]))


