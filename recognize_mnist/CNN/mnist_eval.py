#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/26 21:08
# @Author  : Jing
# @FileName: mnist_eval.py
# @IDE: PyCharm
======================================="""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from recognize_mnist.CNN import mnist_inference, mnist_train
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# load the most newest model for each 10 seconds and test the model
EVAL_INTERVAL_SECS = 2


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x_ = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        x = tf.reshape(x_, [-1, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, 1])
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x_: mnist.validation.images, y_: mnist.validation.labels}

        # forward
        y = mnist_inference.inference(x, False, None)

        # predict
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # load model by rename the variable
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # compute accuracy for each 10 seconds
        while True:
            with tf.Session() as sess:

                # get the name of the most newest model file
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:

                    # load the model
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training step(s), validation accuracy = %g'
                          % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    path = '../dataset/mnist_data/'
    mnist = input_data.read_data_sets(path, one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    print('start at {}'.format(time.ctime()))
    tf.app.run()
    print('end at {}'.format(time.ctime()))

















