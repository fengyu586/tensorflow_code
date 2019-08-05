#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 11:47
# @Author  : Jing
# @FileName: lstm_keras.py
# @IDE: PyCharm
======================================="""
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


# 最多使用的单词数
max_features = 20000
# 循环神经网络的截断长度
max_len = 80
batch_size = 32

# 加载数据
path = './dataset/imdb/'
(train_X, train_Y), (test_X, test_Y) = imdb.load_data(path, num_words=max_features)
print(len(train_X), "train sequence")
print(len(test_X), "test sequence")

train_X = sequence.pad_sequences(train_X, maxlen=max_len)
test_X = sequence.pad_sequences(test_X, maxlen=max_len)

# 构建模型
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_Y, batch_size=batch_size, epochs=15, validation_data=(test_X, test_Y))

score = model.evaluate(test_X, test_Y, batch_size=batch_size)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])



















