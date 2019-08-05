#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 1:13
# @Author  : Jing
# @FileName: lenet1.py
# @IDE: PyCharm
======================================="""
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist


data_path = './dataset/mnist_data'
# 读取数据
train_X, train_Y, test_X, test_Y = mnist.load_data(data_path, one_hot=True)
train_X, test_X = train_X.reshape([-1, 28, 28, 1]), test_X.reshape([-1, 28, 28, 1])

net = input_data(shape=[None, 28, 28, 1], name='input')
net = conv_2d(net, 32, 5, activation='relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 5, activation='relu')
net = max_pool_2d(net, 2)
net = fully_connected(net, 500, activation='relu')
net = fully_connected(net, 10, activation='softmax')

net = regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(train_X, train_Y, n_epoch=20, validation_set=([test_X, test_Y]), show_metric=True)

