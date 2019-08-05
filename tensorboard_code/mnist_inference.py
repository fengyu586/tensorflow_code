#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 14:50
# @Author  : Jing
# @FileName: mnist_inference.py
# @IDE: PyCharm
======================================="""
import tensorflow as tf


# define the parameters of the network
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    """get the variable of the weights"""
    weights = tf.get_variable("weights", shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))

    # put a tensor into a set
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    """define the process of forward"""
    # define the first layer
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)

    # define the second layer
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights)+biases

    return layer2