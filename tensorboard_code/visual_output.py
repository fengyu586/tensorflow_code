#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 16:22
# @Author  : Jing
# @FileName: visual_output.py
# @IDE: PyCharm
======================================="""
import tensorflow as tf
from tensorboard_code import mnist_inference
import os
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAIN_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

LOG_DIR = 'log/visual_output_log'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FILE = 'mnist_meta.tsv'
TENSOR_NAME = 'FINAL_LOGITS'


def train(mnist):
    # 输入数据的命名空间
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,
                           [None, mnist_inference.INPUT_NODE],
                           name='x-input')
        y_ = tf.placeholder(tf.float32,
                            [None, mnist_inference.OUTPUT_NODE],
                            name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 处理滑动平均的命名空间。
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 定义损失函数的命名空间
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

    # 定义学习率、优化方法以及每一轮执行训练操作的命名空间。
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY, staircase=True)

        train_step = tf.train.GradientDescentOptimizer(learning_rate)\
            .minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

    # 训练模型。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g"
                      % (i, loss_value))

                # 计算MNIST测试数据对应的输出层矩阵。
        final_result = sess.run(y, feed_dict={x: mnist.test.images})
    return final_result


def visualisation(final_result):
    y = tf.Variable(final_result, name=TENSOR_NAME)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = y.name

    embedding.metadata_path = META_FILE

    embedding.sprite.image_path = SPRITE_FILE

    embedding.sprite.single_image_dim.extend([28, 28])

    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAIN_STEPS)

    summary_writer.close()


def main():
    path = '../dataset/mnist_data/'
    mnist = input_data.read_data_sets(path, one_hot=True)
    final_result = train(mnist)
    visualisation(final_result)


if __name__ == '__main__':
    main()
