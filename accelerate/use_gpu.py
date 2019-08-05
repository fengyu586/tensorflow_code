#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 17:23
# @Author  : Jing
# @FileName: use_gpu.py
# @IDE: PyCharm
======================================="""

import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
d = tf.reshape(a, [1, 3])
c = a + b
sess = tf.Session()
print(sess.run(c))
print(sess.run(d))
