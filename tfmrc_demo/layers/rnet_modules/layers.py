import tensorflow as tf
from collections import defaultdict

VERY_NEGATIVE_NUMBER = -1e29


def dropout(x, keep_prob, training, noise_shape=None):
    # if keep_prob >= 1.0:
    #     return x
    return tf.cond(training, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape), lambda: x)


class Layer(object):
    _name_dict = defaultdict(int)

    def __init__(self, name=None):
        if name is None:
            name = "layer"

        self.name = name + "_" + str(self._name_dict[name] + 1)
        self._name_dict[name] += 1


class Dropout(Layer):
    def __init__(self, keep_prob=1.0, name="dropout"):
        super(Dropout, self).__init__(name)
        self.keep_prob = keep_prob

    def __call__(self, x, training):
        return dropout(x, self.keep_prob, training)


class VariationalDropout(Layer):
    def __init__(self, keep_prob=1.0, name="variational_dropout"):
        super(VariationalDropout, self).__init__(name)
        self.keep_prob = keep_prob

    def __call__(self, x, training):
        input_shape = tf.shape(x)
        return dropout(x, self.keep_prob, training, noise_shape=[input_shape[0], 1, input_shape[2]])


class MultiLayerRNN(Layer):
    def __init__(self, layers=None, concat_layer_out=True, input_keep_prob=1.0, name='multi_layer_rnn'):
        super(MultiLayerRNN, self).__init__(name)
        self.concat_layer_output = concat_layer_out
        self.dropout = VariationalDropout(input_keep_prob)
        self.rnn_layers = layers

    def __call__(self, x, x_len, training):
        output = x
        outputs = []
        for layer in self.rnn_layers:
            output, _ = layer(self.dropout(output, training), x_len)
            outputs.append(output)
        if self.concat_layer_output:
            return tf.concat(outputs, axis=-1)
        return outputs[-1]


class MultiHeadAttention(Layer):
    def __init__(self, heads, units, attention_on_itself=True, name='encoder_block'):
        super(MultiHeadAttention, self).__init__(name)
        self.heads = heads
        self.units = units
        self.attention_on_itself = attention_on_itself  # only workable when query==key
        self.dense_layers = [tf.keras.layers.Dense(units) for _ in range(3)]

    def __call__(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        max_query_len = tf.shape(query)[1]
        max_key_len = tf.shape(key)[1]
        wq = tf.transpose(
            tf.reshape(self.dense_layers[0](query), [batch_size, max_query_len, self.heads, self.units // self.heads]),
            [2, 0, 1, 3])  # Head*B*QL*(U/Head)
        wk = tf.transpose(
            tf.reshape(self.dense_layers[1](key), [batch_size, max_key_len, self.heads, self.units // self.heads]),
            [2, 0, 1, 3])  # Head*B*KL*(U/Head)
        wv = tf.transpose(
            tf.reshape(self.dense_layers[2](value), [batch_size, max_key_len, self.heads, self.units // self.heads]),
            [2, 0, 1, 3])  # Head*B*KL*(U/Head)
        attention_score = tf.matmul(wq, wk, transpose_b=True) / tf.sqrt(float(self.units) / self.heads)  # Head*B*QL*KL
        if query == key and not self.attention_on_itself:
            attention_score += tf.matrix_diag(tf.zeros(max_key_len) - 100.0)
        if mask is not None:
            attention_score += tf.expand_dims(mask, 1)
        similarity = tf.nn.softmax(attention_score, -1)  # Head*B*QL*KL
        return tf.reshape(tf.transpose(tf.matmul(similarity, wv), [1, 2, 0, 3]),
                          [batch_size, max_query_len, self.units])  # B*QL*U
