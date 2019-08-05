#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""====================================
# @Time    : 2019/7/28 19:35
# @Author  : Jing
# @FileName: preprocess.py
# @IDE: PyCharm
======================================="""
import tensorflow as tf


MAX_LEN = 50                # 限定句子的最大单词数量。
SOS_ID = 1                  # 目标语言词汇表中<sos>的ID。

# 使用Dataset从一个文件中读取一个语言的数据。
# 数据的格式为每行一句话，单词已经转化为单词编号。
def make_data_set(file_path):
    data_set = tf.data.TextLineDataset(file_path)

    # 根据空格将单词编号切分开并放入一个一维向量。
    data_set = data_set.map(lambda string:tf.string_split([string]).values)

    # 将字符串形式的单词编号转化为整数。
    data_set = data_set.map(lambda string: tf.string_to_number(string, tf.int32))

    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
    data_set = data_set.map(lambda x: (x, tf.size(x)))
    return data_set


# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，
# 并进行填充和batching操作。
def make_src_trg_data_set(src_path, trg_path, batch_size):
    # 首先分别读取源语言数据和目标语言数据。
    src_data = make_data_set(src_path)
    trg_data = make_data_set(trg_path)

    # 通过zip操作将两个Dataset合并为一个Dataset。
    # 现在每个Dataset中每一项数据ds由4个张量组成。
    # 由4个张量组成：
    #   ds[0][0]是源句子
    #   ds[0][1]是源句子长度
    #   ds[1][0]是目标句子
    #   ds[1][1]是目标句子长度
    data_set = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空（只包含<eos>）的句子和长度过长的句子。
    def fileter_length(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)

    data_set = data_set.filter(fileter_length)

    # 上面从文件中读到的目标句子是"X Y Z <eos>"的形式，
    # 我们需要从中生成"<sos> X Y Z"形式并加入到Dataset中。
    def make_trg_input(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    data_set = data_set.map(make_trg_input)

    # 随机打乱训练数据。
    data_set = data_set.shuffle(10000)

    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None])),
        (tf.TensorShape([])),
        (tf.TensorShape([None])),
        (tf.TensorShape([None])),
        (tf.TensorShape([])))

    # 调用padded_batch方法进行batching操作。
    batched_data_set = data_set.padded_batch(batch_size, padded_shapes)
    return batched_data_set
















