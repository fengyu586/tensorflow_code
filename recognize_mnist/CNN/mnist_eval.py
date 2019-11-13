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
        # load model by ren                y_output = graph.get_tensor_by_name("y-output")ame the variable
        saver = tf.train.import_meta_graph('model/cnn_model.ckpt-29000.meta')

        # compute accuracy for each 10 seconds
        with tf.Session() as sess:
            # get the name of the most newest model file
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:

                # load the model
                saver.restore(sess, ckpt.model_checkpoint_path)
                graph = tf.get_default_graph()
                x_input = graph.get_tensor_by_name("x-input:0")
                y_input = graph.get_tensor_by_name("y-input:0")
                y_output = graph.get_tensor_by_name("layer6-fc2/y_output:0")
                validate_feed = {x_input: mnist.validation.images, y_input: mnist.validation.labels}
                correct_prediction = tf.equal(tf.argmax(y_input, 1), tf.argmax(y_output, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print('validation accuracy = %g' % accuracy_score)
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

















