#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/11 17:10
"""
import tensorflow as tf
from collections import defaultdict


class Layer(object):
    _name_dict = defaultdict(int)

    def __init__(self, name=None):
        if name is None:
            name = "layer"

        self.name = name + "_" + str(self._name_dict[name] + 1)
        self._name_dict[name] += 1


def dropout(x, keep_prob, training, noise_shape=None):
    if keep_prob >= 1.0:
        return x
    return tf.cond(training, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape), lambda: x)


class VariationalDropout(Layer):
    def __init__(self, keep_prob=1.0, name="variational_dropout"):
        super(VariationalDropout, self).__init__(name)
        self.keep_prob = keep_prob

    def __call__(self, x, training):
        input_shape = tf.shape(x)
        return dropout(x, self.keep_prob, training, noise_shape=[input_shape[0], 1, input_shape[2]])
