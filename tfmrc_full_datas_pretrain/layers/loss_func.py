import tensorflow as tf


def cul_single_ans_loss(start_probs, end_probs, start_label, end_label):
    """
    基于single answer的loss(baseline)
    """
    def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
        """
        negative log likelyhood loss
        """
        with tf.name_scope(scope, "log_loss"):
            labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
            losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
        return losses

    start_loss = sparse_nll_loss(probs=start_probs, labels=start_label)
    end_loss = sparse_nll_loss(probs=end_probs, labels=end_label)
    loss = tf.reduce_mean(tf.add(start_loss, end_loss))
    return loss

def cul_weighted_avg_loss(start_probs, end_probs, start_label, end_label, match_score):
    """
    计算weighted average loss
    """
    def multi_sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
        """
        multi answer negative log likelyhood loss
        """
        with tf.name_scope(scope, "multi_log_loss"):
            labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
            probs = tf.expand_dims(probs, axis=-1)
            losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), axis=1) # shape=[batch_size, ans_num]
        return losses

    start_loss = multi_sparse_nll_loss(start_probs, start_label)
    end_loss = multi_sparse_nll_loss(end_probs, end_label)
    loss = tf.reduce_mean(tf.reduce_sum(match_score * tf.add(start_loss, end_loss), axis=1), axis=0)
    return loss

def cul_pas_sel_loss(fuse_p_encodes, hidden_size, gold_passage, epsilon=1e-9, scope=None):
    """
    计算Passage selection with multi-answer loss
    """
    with tf.variable_scope(scope or 'pas_sel_loss'):
        # pool attention
        U = tf.contrib.layers.fully_connected(fuse_p_encodes, num_outputs=hidden_size, activation_fn=tf.nn.tanh)
        logits = tf.contrib.layers.fully_connected(U, num_outputs=1, activation_fn=None)
        scores = tf.nn.softmax(logits, axis=1)
        pooled_pas = tf.reduce_sum(fuse_p_encodes * scores, axis=1) # shape = [32*5, 150]

        # match scores
        match_score = tf.contrib.layers.fully_connected(pooled_pas, num_outputs=1,
                                    activation_fn=tf.nn.sigmoid, biases_initializer=None) # shape=[32*5,1]
        match_score = tf.squeeze(match_score, axis=1) # shape=[32*5]

        # pointwise sigmoid loss
        gold_passage = tf.cast(gold_passage, tf.float32)
        loss = -tf.reduce_mean(gold_passage * tf.log(match_score + epsilon) +
                               (1 - gold_passage) * tf.log(1 - match_score + epsilon))
        return loss
