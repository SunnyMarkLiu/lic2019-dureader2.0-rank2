#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/3 21:00
"""
import numpy as np
import tensorflow as tf


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    """
    Sinusoidal Positional_Encoding. See 3.5
    Args:
        inputs: 3d tensor. (N, T, E)
        maxlen: scalar. Must be >= T
        masking: Boolean. If True, padding positions are set to zeros.
        scope: Optional scope for `variable_scope`.
        returns 3d tensor that has the same shape as inputs.
    """
    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def gated_cnn(inputs, num_filters):
    # inputs: [batch_size, seq_len, seq_dim, 1]

    input_shape = inputs.get_shape().as_list()
    filter_height, filter_width = 2, input_shape[2]
    # [filter_height, filter_width, in_channels, out_channels]
    W = tf.Variable(initial_value=tf.truncated_normal([filter_height, filter_width, 1, num_filters], mean=0.0, stddev=0.1), name='linear_conv_w')
    b = tf.Variable(initial_value=tf.constant(0.1, shape=[num_filters]), name='linear_conv_b')
    conv_linear = tf.add(tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME'), b)

    W = tf.Variable(initial_value=tf.truncated_normal([filter_height, filter_width, 1, num_filters], mean=0.0, stddev=0.1), name='gated_conv_w')
    b = tf.Variable(initial_value=tf.constant(0.1, shape=[num_filters]), name='gated_conv_b')
    conv_gated = tf.add(tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME'), b)
    outputs = conv_linear * tf.sigmoid(conv_gated)

    outputs = tf.reduce_mean(outputs, axis=-1)
    return outputs
