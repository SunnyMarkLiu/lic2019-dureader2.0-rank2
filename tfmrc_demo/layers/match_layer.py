"""
This module implements the core layer of Match-LSTM and BiDAF
"""

import tensorflow as tf
import tensorflow.contrib as tc
from layers.rnet_modules.layers import VariationalDropout, MultiLayerRNN, MultiHeadAttention
from layers.rnet_modules.recurrent import CudnnBiGRU, CudnnGRU


class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context, inputs * attended_context],
                                   -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)


class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state


class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        with tf.variable_scope('bidaf'):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                         [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)
            return concat_outputs, None


class RnetMatchLayer(object):
    """
    Implements r-net match layer
    """
    def __init__(self, hidden_size, is_training, dropout_keep_prob=1.0):
        self.hidden_size = hidden_size
        self.keep_prob = dropout_keep_prob
        self.doc_rnn_layers = 1
        self.question_rnn_layers = 1
        self.heads = 2
        self.training = is_training

    def match(self, encoder_context, encoder_question, context_len, question_len):
        with tf.variable_scope('rnet'):
            # mask
            max_context_len = tf.shape(encoder_context)[1]
            max_question_len = tf.shape(encoder_question)[1]
            context_mask = (tf.sequence_mask(context_len, max_context_len, dtype=tf.float32)-1) * 100
            question_mask = (tf.sequence_mask(question_len, max_question_len, dtype=tf.float32)-1) * 100

            # dropout
            dropout = VariationalDropout(self.keep_prob)

            # co-attention
            co_attention_context = tf.expand_dims(tf.keras.layers.Dense(self.hidden_size)(encoder_context),2) # B*CL*1*H
            co_attention_question = tf.expand_dims(tf.keras.layers.Dense(self.hidden_size)(encoder_question), 1) # B*1*QL*H
            co_attention_score = tf.keras.layers.Dense(1)(tf.nn.tanh(co_attention_context+co_attention_question))[:,:,:,0]+tf.expand_dims(question_mask,1) # B*CL*QL
            co_attention_similarity = tf.nn.softmax(co_attention_score,-1) # B*CL*QL
            co_attention_rnn_input = tf.concat([encoder_context,tf.matmul(co_attention_similarity,encoder_question)],-1) # B*CL*(H*4)
            co_attention_rnn_input = co_attention_rnn_input*tf.keras.layers.Dense(self.hidden_size*4,activation=tf.nn.sigmoid)(co_attention_rnn_input)
            co_attention_rnn = CudnnGRU(self.hidden_size)
            co_attention_output = dropout(co_attention_rnn(co_attention_rnn_input,context_len)[0], self.training)  # B*CL*(H*2)

            # self-attention
            multi_head_attention = MultiHeadAttention(self.heads,self.hidden_size,False)
            self_attention_repr = dropout(multi_head_attention(co_attention_output,co_attention_output,co_attention_output,context_mask),self.training)
            self_attention_rnn_input = tf.concat([co_attention_output,self_attention_repr],-1) # B*CL*(H*2)
            self_attention_rnn_input = self_attention_rnn_input*tf.keras.layers.Dense(self.hidden_size*2,activation=tf.nn.sigmoid)(self_attention_rnn_input)

            # rnn context encode
            self_attention_rnn = CudnnBiGRU(self.hidden_size)
            self_attention_output = dropout(self_attention_rnn(self_attention_rnn_input, context_len)[0], self.training)  # B*CL*(H*2)
            return self_attention_output, None


class AttentionFlowMultiHeadMatchLayer(object):
    """
    Implements match layer that combines Attention Flow and Multi Head Self Attention
    """
    def __init__(self, hidden_size, is_training, dropout_keep_prob=1.0):
        self.hidden_size = hidden_size
        self.keep_prob = dropout_keep_prob
        self.doc_rnn_layers = 1
        self.question_rnn_layers = 1
        self.heads = 2
        self.training = is_training

    def match(self, passage_encodes, question_encodes, context_len, question_len):
        with tf.variable_scope('bidaf-multi-head'):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                            [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)

