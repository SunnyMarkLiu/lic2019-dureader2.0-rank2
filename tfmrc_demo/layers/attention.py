#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/3 15:54
"""
import tensorflow as tf
import tensorflow.contrib.layers as layers


def self_attention(inputs, attention_size, activation, l2_reg_lambda, scope):
    """
    attention mechanism
    """
    with tf.variable_scope(scope or 'self_attention',
                           initializer=layers.xavier_initializer(uniform=True),
                           regularizer=layers.l2_regularizer(scale=l2_reg_lambda)):
        context_vector = tf.get_variable(name='self_attention_context_vector',
                                         shape=[attention_size],
                                         dtype=tf.float32)
        # feed the word encoders through a one-layer MLP to get a hidden representation
        hidden_input_represents = layers.fully_connected(inputs=inputs,
                                                         num_outputs=attention_size,
                                                         activation_fn=activation,
                                                         weights_regularizer=layers.l2_regularizer(scale=l2_reg_lambda))

        # measure the importance of the word as the ** similarity ** of uit with a word level context vector uw
        U_it = activation(tf.multiply(hidden_input_represents, context_vector))
        vector_attn = tf.reduce_sum(U_it, axis=2, keep_dims=True)

        attention_weights = tf.nn.softmax(vector_attn, dim=1)

        weighted_projection = tf.multiply(inputs, attention_weights)

        return weighted_projection
