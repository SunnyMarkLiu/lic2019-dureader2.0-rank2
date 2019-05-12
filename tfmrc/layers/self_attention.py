#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/11 16:58
"""
import tensorflow as tf
from collections import defaultdict

VERY_NEGATIVE_NUMBER = -1e29


class Layer(object):
    _name_dict = defaultdict(int)

    def __init__(self, name=None):
        if name is None:
            name = "layer"

        self.name = name + "_" + str(self._name_dict[name] + 1)
        self._name_dict[name] += 1


class SelfAttention(Layer):
    def __init__(self, name="self_attention"):
        super(SelfAttention, self).__init__(name)
        # self.similarity_function = similarity_function

    def __call__(self, query, query_len):
        # sim_mat = self.similarity_function(query, query)
        sim_mat = tf.matmul(query, query, transpose_b=True)
        sim_mat += tf.expand_dims(tf.eye(tf.shape(query)[1]) * VERY_NEGATIVE_NUMBER, 0)
        mask = tf.expand_dims(tf.sequence_mask(query_len, tf.shape(query)[1], dtype=tf.float32), axis=1)
        sim_mat = sim_mat + (1. - mask) * VERY_NEGATIVE_NUMBER
        bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
        sim_mat = tf.exp(sim_mat)
        sim_prob = sim_mat / (tf.reduce_sum(sim_mat, axis=2, keepdims=True) + bias)

        return tf.matmul(sim_prob, query)


class TriLinear(Layer):
    def __init__(self, name="tri_linear", bias=False):
        super(TriLinear, self).__init__(name)
        self.projecting_layers = [tf.keras.layers.Dense(1, activation=None, use_bias=False) for _ in range(2)]
        self.dot_w = None
        self.bias = bias

    def __call__(self, t0, t1):
        t0_score = tf.squeeze(self.projecting_layers[0](t0), axis=-1)
        t1_score = tf.squeeze(self.projecting_layers[1](t1), axis=-1)

        if self.dot_w is None:
            hidden_units = t0.shape.as_list()[-1]
            with tf.variable_scope(self.name):
                self.dot_w = tf.get_variable("dot_w", [hidden_units])

        t0_dot_w = t0 * tf.expand_dims(tf.expand_dims(self.dot_w, axis=0), axis=0)
        t0_t1_score = tf.matmul(t0_dot_w, t1, transpose_b=True)

        out = t0_t1_score + tf.expand_dims(t0_score, axis=2) + tf.expand_dims(t1_score, axis=1)
        if self.bias:
            with tf.variable_scope(self.name):
                bias = tf.get_variable("bias", shape=(), dtype=tf.float32)
            out += bias
        return out
