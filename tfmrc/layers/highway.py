#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/3 16:31
"""
import tensorflow as tf


def highway_layer(inputs, use_bias, transform_bias=-1.0, scope="highway_layer", reuse=None):
    """
    Defines a single highway layer of a highway network
    """
    with tf.variable_scope(scope, reuse=reuse):
        #get hidden dimension d which is 128 in qanet
        dims = inputs.get_shape()[-1]
        #compute the activation using a dense layer with relu activation
        z = tf.layers.dense(inputs, dims, use_bias=use_bias, name="highway_dense_1", reuse=reuse)
        activation = tf.nn.relu(z)
        #compute the transform gate value using a dense layer with sigmoid activation
        transform_gate = tf.layers.dense(inputs, dims, use_bias=use_bias, bias_initializer=tf.constant_initializer(transform_bias), name='highway_dense_2', reuse=reuse)
        transform_gate = tf.nn.sigmoid(transform_gate)
        #apply the highway network equation: (transform_gate * activation) + (carry_gate * inputs) 
        #carry_gate = (1 - transform_gate)
        outputs = transform_gate * activation + (1 - transform_gate) * inputs
        return outputs


def highway_network(inputs, num_layers=2, use_bias=True, transform_bias=-1.0, scope="highway_net", reuse=None):
    """
    Defines a highway network of num_layers and calls highway_layer to construct each layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        for layer_id in range(num_layers):
            #call highway_layer in scope "highway_layer_i", if called again using the same scope, layer i will get reused in scope "highway_layer_i"
            #the highway network is reused on both context and question embedding
            inputs = highway_layer(inputs, use_bias, transform_bias, scope="highway_layer_{}".format(layer_id), reuse=reuse)
        return inputs
