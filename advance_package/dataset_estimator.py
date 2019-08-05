#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/29 14:27
# @Author  : Jing
# @FileName: dataset_estimator.py
# @IDE: PyCharm
======================================="""
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):

    def decode_csv(line):
        """解析csv文件中的一行"""
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0.]])

        return {"x": parsed_line[:-1]}, parsed_line[-1:]

    data_set = (tf.data.TextLineDataset(file_path)
                .skip(1).map(decode_csv))
    if perform_shuffle:
        data_set = data_set.shuffle(buffer_size=256)
    data_set = data_set.repeat(repeat_count)
    data_set = data_set.batch(32)
    iterator = data_set.make_one_shot_iterator()

    batch_features, batch_labels = iterator.next()
    return batch_features, batch_labels

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10], n_classes=3)

train_data_path = './data_set/iris/iris_training.csv'
test_data_path = './data_set/iris/iris_test.csv'
classifier.train(input_fn=lambda: my_input_fn(train_data_path, False, 1))
test_results = classifier.evaluate(input_fn=lambda: my_input_fn(test_data_path, False, 1))
print("Test accuracy: %g" % (test_results["accuracy"]*100.0))

