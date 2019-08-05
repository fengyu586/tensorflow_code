#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 13:25
# @Author  : Jing
# @FileName: base_estimator.py
# @IDE: PyCharm
======================================="""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


tf.logging.set_verbosity(tf.logging.INFO)
data_path = './dataset/mnist_data/'
model_path = 'model/'
mnist = input_data.read_data_sets(data_path, one_hot=True)
feature_columns = [tf.feature_column.numeric_column("image", shape=[784])]

estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                       hidden_units=[500], n_classes=10,
                                       optimizer=tf.train.AdamOptimizer(),
                                       model_dir=model_path)
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"image": mnist.train.images},
                                                    y=mnist.train.labels.astype(np.int32),
                                                    num_epochs=None, batch_size=128, shuffle=True)

estimator.train(input_fn=train_input_fn, steps=10000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"image": mnist.test.images},
                                                   y=mnist.test.labels.astype(np.int32),
                                                   num_epochs=1, batch_size=128, shuffle=False)
accuracy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
print("Test accuracy: %g" % (accuracy_score*100.0))

