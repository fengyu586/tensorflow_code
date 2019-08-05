#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 13:11
# @Author  : Jing
# @FileName: keras_tensorflow.py
# @IDE: PyCharm
======================================="""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


data_path = './dataset/mnist_data/'
mnist = input_data.read_data_sets(data_path, one_hot=True)
x = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.float32, shape=(None, 10))
net = tf.keras.layers.Dense(500, activation='relu')(x)
y = tf.keras.layers.Dense(10, activation='softmax')(net)
loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_, y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

acc_value = tf.reduce_mean(tf.keras.metrics.categorical_crossentropy(y_, y))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10000):
        xs, ys = mnist.train.next_batch(100)
        _, loss_value = sess.run([train_step, loss], feed_dict={x: xs, y_: ys})

        if i % 1000 == 0:
            print("After %d training step(s), loss on training batch is %g" % (i, loss_value))
    print(acc_value.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))





