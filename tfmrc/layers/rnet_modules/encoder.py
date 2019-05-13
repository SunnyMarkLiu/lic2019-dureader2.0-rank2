#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/13 17:05
"""
import tensorflow as tf
from layers.rnet_modules.layers import Layer
from layers.rnet_modules.layers import VariationalDropout
from layers.rnet_modules.layers import MultiHeadAttention
from layers.rnet_modules.recurrent import CudnnBiGRU, CudnnGRU


class MultiHeadSelfAttentionEncoder(Layer):
    def __init__(self, heads, input_hidden_size, out_rnn_hidden_size, training, keep_prob=1, name="multi_head_self_attention"):
        super(MultiHeadSelfAttentionEncoder, self).__init__(name)
        self.heads = heads
        self.input_hidden_size = input_hidden_size
        self.keep_prob = keep_prob
        self.training = training
        self.out_rnn_hidden_size = out_rnn_hidden_size
        self.dropout = VariationalDropout(self.keep_prob)

    def __call__(self, x, x_len, residual_connect=True):
        multi_head_attention = MultiHeadAttention(self.heads, self.input_hidden_size, False)

        # mask
        max_x_len = tf.shape(x)[1]
        x_mask = (tf.sequence_mask(x_len, max_x_len, dtype=tf.float32) - 1) * 100

        x_atten_repr = self.dropout(multi_head_attention(x, x, x, x_mask), self.training)
        # multihead attention features + word embedding features
        if residual_connect:
            x_self_atten_rnn_input = tf.concat([x, x_atten_repr], -1)  # B*CL*(H*2)
        else:
            x_self_atten_rnn_input = x_atten_repr

        self_attention_rnn = CudnnBiGRU(self.out_rnn_hidden_size)
        self_attention_output = self.dropout(self_attention_rnn(x_self_atten_rnn_input, x_len)[0], self.training)  # B*CL*(H*2)

        return self_attention_output
