#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/25 20:36
# @Author  : Jing
# @FileName: mnist_train.py
# @IDE: PyCharm
======================================="""
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from recognize_mnist.CNN import mnist_inference

# set the parameters of the network
BATCH_SIZE = 100
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
LEARNING_RATE = 0.001

# the path for saving model
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'cnn_model.ckpt'

def train(mnist):
    """train the model"""
    x_ = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    x = tf.reshape(x_, [-1, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, 1])
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # define the loss function, learning rate, move average and the process of training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name='train')

    # initial
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # the process of training model
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, acc = sess.run([train_op, loss, global_step, accuracy], feed_dict={x_: xs, y_: ys})

            # print the process of training model
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g. acc on training batch is %.3f'
                      % (i, loss_value, acc))
                # save the model after each 1000 steps for training model
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), i)


def main(argv=None):
    path = '../dataset/mnist_data/'
    mnist = input_data.read_data_sets(path, one_hot=True)
    train(mnist)


if __name__ == '__main__':
    print('start at {}'.format(time.ctime()))
    tf.app.run()
    print('end at {}'.format(time.ctime()))











