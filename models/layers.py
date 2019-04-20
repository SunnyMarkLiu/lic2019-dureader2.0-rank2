#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/20 10:02
"""
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from utils.functions import masked_softmax, compute_mask, masked_flip


class GloveEmbedding(torch.nn.Module):
    """
    Glove Embedding Layer, also compute the mask of padding index
    Args:
        - dataset_h5_path: glove embedding file path
    Inputs:
        **input** (batch, seq_len): sequence with word index
    Outputs
        **output** (seq_len, batch, embedding_size): tensor that change word index to word embeddings
        **mask** (batch, seq_len): tensor that show which index is padding
    """

    def __init__(self, dataset_h5_path):
        super(GloveEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embeddings, len_embedding, weights = self.load_glove_hdf5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embeddings, embedding_dim=len_embedding,
                                                  _weight=weights)
        self.embedding_layer.weight.requires_grad = False

    def load_glove_hdf5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            f_meta_data = f['meta_data']
            id2vec = np.array(f_meta_data['id2vec'])  # only need 1.11s
            word_dict_size = f.attrs['word_dict_size']
            embedding_size = f.attrs['embedding_size']

        return int(word_dict_size), int(embedding_size), torch.from_numpy(id2vec)

    def forward(self, x):
        mask = compute_mask(x)

        tmp_emb = self.embedding_layer.forward(x)
        out_emb = tmp_emb.transpose(0, 1)

        return out_emb, mask


class CharEmbedding(torch.nn.Module):
    """
    Char Embedding Layer, random weight
    Args:
        - dataset_h5_path: char embedding file path
    Inputs:
        **input** (batch, seq_len, word_len): word sequence with char index
    Outputs
        **output** (batch, seq_len, word_len, embedding_size): tensor contain char embeddings
        **mask** (batch, seq_len, word_len)
    """

    def __init__(self, dataset_h5_path, embedding_size, trainable=False):
        super(CharEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embedding = self.load_dataset_h5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embedding, embedding_dim=embedding_size,
                                                  padding_idx=0)

        # Note that cannot directly assign value. When in predict, it's always False.
        if not trainable:
            self.embedding_layer.weight.requires_grad = False

    def load_dataset_h5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            word_dict_size = f.attrs['char_dict_size']

        return int(word_dict_size)

    def forward(self, x):
        batch_size, seq_len, word_len = x.shape
        x = x.view(-1, word_len)

        mask = compute_mask(x, 0)  # char-level padding idx is zero
        x_emb = self.embedding_layer.forward(x)
        x_emb = x_emb.view(batch_size, seq_len, word_len, -1)
        mask = mask.view(batch_size, seq_len, word_len)

        return x_emb, mask


class CharEncoder(torch.nn.Module):
    """
    char-level encoder with MyRNNBase used
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p):
        super(CharEncoder, self).__init__()

        self.encoder = MyStackedRNN(mode=mode,
                                    input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size)
        x = x.transpose(0, 1)
        char_mask = char_mask.view(-1, word_len)

        _, x_encode = self.encoder.forward(x, char_mask)  # (batch*seq_len, hidden_size)
        x_encode = x_encode.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x_encode = x_encode * word_mask.unsqueeze(-1)

        return x_encode.transpose(0, 1)


class CharCNN(torch.nn.Module):
    """
    Char-level CNN
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, filters_size, filters_num, dropout_p):
        super(CharCNN, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.cnns = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, fn, (fw, emb_size)) for fw, fn in zip(filters_size, filters_num)])

    def forward(self, x, char_mask, word_mask):
        x = self.dropout(x)

        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size).unsqueeze(1)  # (N, 1, word_len, embedding_size)

        x = [F.relu(cnn(x)).squeeze(-1) for cnn in self.cnns]  # (N, Cout, word_len - fw + 1) * fn
        x = [torch.max(cx, 2)[0] for cx in x]  # (N, Cout) * fn
        x = torch.cat(x, dim=1)  # (N, hidden_size)

        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x = x * word_mask.unsqueeze(-1)

        return x.transpose(0, 1)


class Highway(torch.nn.Module):
    def __init__(self, in_size, n_layers, dropout_p):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.normal_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            normal_layer_ret = F.relu(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x
        return x


class CharCNNEncoder(torch.nn.Module):
    """
    char-level cnn encoder with highway networks
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, hidden_size, filters_size, filters_num, dropout_p, enable_highway=True):
        super(CharCNNEncoder, self).__init__()
        self.enable_highway = enable_highway
        self.hidden_size = hidden_size

        self.cnn = CharCNN(emb_size=emb_size,
                           filters_size=filters_size,
                           filters_num=filters_num,
                           dropout_p=dropout_p)

        if enable_highway:
            self.highway = Highway(in_size=hidden_size,
                                   n_layers=2,
                                   dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        o = self.cnn(x, char_mask, word_mask)

        assert o.shape[2] == self.hidden_size
        if self.enable_highway:
            o = self.highway(o)

        return o


class MatchRNNAttention(torch.nn.Module):
    r"""
    attention mechanism in match-rnn
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
    Inputs:
        Hpi(batch, input_size): a context word encoded
        Hq(question_len, batch, input_size): whole question encoded
        Hr_last(batch, hidden_size): last lstm hidden output
    Outputs:
        alpha(batch, question_len): attention vector
    """

    def __init__(self, hp_input_size, hq_input_size, hidden_size):
        super(MatchRNNAttention, self).__init__()

        self.linear_wq = torch.nn.Linear(hq_input_size, hidden_size)
        self.linear_wp = torch.nn.Linear(hp_input_size, hidden_size)
        self.linear_wr = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wg = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hpi, Hq, Hr_last, Hq_mask):
        wq_hq = self.linear_wq(Hq)  # (question_len, batch, hidden_size)
        wp_hp = self.linear_wp(Hpi).unsqueeze(0)  # (1, batch, hidden_size)
        wr_hr = self.linear_wr(Hr_last).unsqueeze(0)  # (1, batch, hidden_size)
        G = F.tanh(wq_hq + wp_hp + wr_hr)  # (question_len, batch, hidden_size), auto broadcast
        wg_g = self.linear_wg(G) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, question_len)
        alpha = masked_softmax(wg_g, m=Hq_mask, dim=1)  # (batch, question_len)
        return alpha


class UniMatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism, one direction, using LSTM cell
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
    Outputs:
        Hr(context_len, batch, hidden_size): question-aware context representation
        alpha(batch, question_len, context_len): used for visual show
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, gated_attention, enable_layer_norm):
        super(UniMatchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gated_attention = gated_attention
        self.enable_layer_norm = enable_layer_norm
        rnn_in_size = hp_input_size + hq_input_size

        self.attention = MatchRNNAttention(hp_input_size, hq_input_size, hidden_size)

        if self.gated_attention:
            self.gated_linear = torch.nn.Linear(rnn_in_size, rnn_in_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(rnn_in_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size=rnn_in_size, hidden_size=hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size=rnn_in_size, hidden_size=hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hp, Hq, Hq_mask):
        batch_size = Hp.shape[1]
        context_len = Hp.shape[0]

        # init hidden with the same type of input data
        h_0 = Hq.new_zeros(batch_size, self.hidden_size)
        hidden = [(h_0, h_0)] if self.mode == 'LSTM' else [h_0]
        vis_para = {}
        vis_alpha = []
        vis_gated = []

        for t in range(context_len):
            cur_hp = Hp[t, ...]  # (batch, input_size)
            attention_input = hidden[t][0] if self.mode == 'LSTM' else hidden[t]

            alpha = self.attention.forward(cur_hp, Hq, attention_input, Hq_mask)  # (batch, question_len)
            vis_alpha.append(alpha)

            question_alpha = torch.bmm(alpha.unsqueeze(1), Hq.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)
            cur_z = torch.cat([cur_hp, question_alpha], dim=1)  # (batch, rnn_in_size)

            # gated
            if self.gated_attention:
                gate = F.sigmoid(self.gated_linear.forward(cur_z))
                vis_gated.append(gate.squeeze(-1))
                cur_z = gate * cur_z

            # layer normalization
            if self.enable_layer_norm:
                cur_z = self.layer_norm(cur_z)  # (batch, rnn_in_size)

            cur_hidden = self.hidden_cell.forward(cur_z, hidden[t])  # (batch, hidden_size), when lstm output tuple
            hidden.append(cur_hidden)

        vis_para['gated'] = torch.stack(vis_gated, dim=-1)  # (batch, context_len)
        vis_para['alpha'] = torch.stack(vis_alpha, dim=2)  # (batch, question_len, context_len)

        hidden_state = list(map(lambda x: x[0], hidden)) if self.mode == 'LSTM' else hidden
        result = torch.stack(hidden_state[1:], dim=0)  # (context_len, batch, hidden_size)
        return result, vis_para


class MatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
        - bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        - gated_attention: If ``True``, gated attention used, see more on R-NET
    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
        Hp_mask(batch, context_len): each context valued length without padding values
        Hq_mask(batch, question_len): each question valued length without padding values
    Outputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, bidirectional, gated_attention,
                 dropout_p, enable_layer_norm):
        super(MatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 1 if bidirectional else 2

        self.left_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                          enable_layer_norm)
        if bidirectional:
            self.right_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                               enable_layer_norm)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hp, Hp_mask, Hq, Hq_mask):
        Hp = self.dropout(Hp)
        Hq = self.dropout(Hq)

        left_hidden, left_para = self.left_match_rnn.forward(Hp, Hq, Hq_mask)
        rtn_hidden = left_hidden
        rtn_para = {'left': left_para}

        if self.bidirectional:
            Hp_inv = masked_flip(Hp, Hp_mask, flip_dim=0)
            right_hidden_inv, right_para_inv = self.right_match_rnn.forward(Hp_inv, Hq, Hq_mask)

            # flip back to normal sequence
            right_alpha_inv = right_para_inv['alpha']
            right_alpha_inv = right_alpha_inv.transpose(0, 1)  # make sure right flip
            right_alpha = masked_flip(right_alpha_inv, Hp_mask, flip_dim=2)
            right_alpha = right_alpha.transpose(0, 1)

            right_gated_inv = right_para_inv['gated']
            right_gated_inv = right_gated_inv.transpose(0, 1)
            right_gated = masked_flip(right_gated_inv, Hp_mask, flip_dim=2)
            right_gated = right_gated.transpose(0, 1)

            right_hidden = masked_flip(right_hidden_inv, Hp_mask, flip_dim=0)

            rtn_para['right'] = {'alpha': right_alpha, 'gated': right_gated}
            rtn_hidden = torch.cat((left_hidden, right_hidden), dim=2)

        real_rtn_hidden = Hp_mask.transpose(0, 1).unsqueeze(2) * rtn_hidden
        last_hidden = rtn_hidden[-1, :]

        return real_rtn_hidden, last_hidden, rtn_para


class PointerAttention(torch.nn.Module):
    r"""
    attention mechanism in pointer network
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        Hk_last(batch, hidden_size): the last hidden output of previous time
    Outputs:
        beta(batch, context_len): question-aware context representation
    """

    def __init__(self, input_size, hidden_size):
        super(PointerAttention, self).__init__()

        self.linear_wr = torch.nn.Linear(input_size, hidden_size)
        self.linear_wa = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wf = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hr, Hr_mask, Hk_pre):
        wr_hr = self.linear_wr(Hr)  # (context_len, batch, hidden_size)
        wa_ha = self.linear_wa(Hk_pre).unsqueeze(0)  # (1, batch, hidden_size)
        f = F.tanh(wr_hr + wa_ha)  # (context_len, batch, hidden_size)

        beta_tmp = self.linear_wf(f) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, context_len)

        beta = masked_softmax(beta_tmp, m=Hr_mask, dim=1)
        return beta


class SeqPointer(torch.nn.Module):
    r"""
    Sequence Pointer Net that output every possible answer position in context
    Args:
    Inputs:
        Hr: question-aware context representation
    Outputs:
        **output** every answer index possibility position in context, no fixed length
    """

    def __init__(self):
        super(SeqPointer, self).__init__()

    def forward(self, *input):
        return NotImplementedError


class UniBoundaryPointer(torch.nn.Module):
    r"""
    Unidirectional Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - enable_layer_norm: Whether use layer normalization
    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0(batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
        **hidden** (batch, hidden_size), [(batch, hidden_size)]: rnn last state
    """
    answer_len = 2

    def __init__(self, mode, input_size, hidden_size, enable_layer_norm):
        super(UniBoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enable_layer_norm = enable_layer_norm

        self.attention = PointerAttention(input_size, hidden_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size, hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size, hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hr, Hr_mask, h_0=None):
        if h_0 is None:
            batch_size = Hr.shape[1]
            h_0 = Hr.new_zeros(batch_size, self.hidden_size)

        hidden = (h_0, h_0) if self.mode == 'LSTM' and isinstance(h_0, torch.Tensor) else h_0
        beta_out = []

        for t in range(self.answer_len):
            attention_input = hidden[0] if self.mode == 'LSTM' else hidden
            beta = self.attention.forward(Hr, Hr_mask, attention_input)  # (batch, context_len)
            beta_out.append(beta)

            context_beta = torch.bmm(beta.unsqueeze(1), Hr.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)

            if self.enable_layer_norm:
                context_beta = self.layer_norm(context_beta)  # (batch, input_size)

            hidden = self.hidden_cell.forward(context_beta, hidden)  # (batch, hidden_size), (batch, hidden_size)

        result = torch.stack(beta_out, dim=0)
        return result, hidden


class BoundaryPointer(torch.nn.Module):
    r"""
    Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - bidirectional: Bidirectional or Unidirectional
        - dropout_p: Dropout probability
        - enable_layer_norm: Whether use layer normalization
    Inputs:
        Hr (context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0 (batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm):
        super(BoundaryPointer, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.left_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        if bidirectional:
            self.right_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)

        left_beta, _ = self.left_ptr_rnn.forward(Hr, Hr_mask, h_0)
        rtn_beta = left_beta
        if self.bidirectional:
            right_beta_inv, _ = self.right_ptr_rnn.forward(Hr, Hr_mask, h_0)
            right_beta = right_beta_inv[[1, 0], :]

            rtn_beta = (left_beta + right_beta) / 2

        # todo: unexplainable
        new_mask = torch.neg((Hr_mask - 1) * 1e-6)  # mask replace zeros with 1e-6, make sure no gradient explosion
        rtn_beta = rtn_beta + new_mask.unsqueeze(0)

        return rtn_beta


class MultiHopBdPointer(torch.nn.Module):
    r"""
    Boundary Pointer Net with Multi-Hops that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - num_hops: Number of max hops
        - dropout_p: Dropout probability
        - enable_layer_norm: Whether use layer normalization
    Inputs:
        Hr (context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0 (batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """

    def __init__(self, mode, input_size, hidden_size, num_hops, dropout_p, enable_layer_norm):
        super(MultiHopBdPointer, self).__init__()
        self.hidden_size = hidden_size
        self.num_hops = num_hops

        self.ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)

        beta_last = None
        for i in range(self.num_hops):
            beta, h_0 = self.ptr_rnn.forward(Hr, Hr_mask, h_0)
            if beta_last is not None and (beta_last == beta).sum().item() == beta.shape[0]:  # beta not changed
                break

            beta_last = beta

        new_mask = torch.neg((Hr_mask - 1) * 1e-6)  # mask replace zeros with 1e-6, make sure no gradient explosion
        rtn_beta = beta + new_mask.unsqueeze(0)

        return rtn_beta


class MyRNNBase(torch.nn.Module):
    """
    RNN with packed sequence and dropout, only one layer
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization
    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.
    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyRNNBase, self).__init__()
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=1,
                                        bidirectional=bidirectional)
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=1,
                                       bidirectional=bidirectional)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, v, mask):
        # layer normalization
        if self.enable_layer_norm:
            seq_len, batch, input_size = v.shape
            v = v.view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(seq_len, batch, input_size)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        v_sort = v.index_select(1, idx_sort)

        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # unsorted o
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len

        # get the last time state
        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)
        o_last = o_unsort.gather(0, len_idx)
        o_last = o_last.squeeze(0)

        return o_unsort, o_last


class MyStackedRNN(torch.nn.Module):
    """
    RNN with packed sequence and dropout, multi-layers used
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: number of rnn layers
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization
    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.
    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyStackedRNN, self).__init__()
        self.num_layers = num_layers
        self.rnn_list = torch.nn.ModuleList([MyRNNBase(mode, input_size, hidden_size, bidirectional, dropout_p,
                                                       enable_layer_norm) for _ in range(num_layers)])

    def forward(self, v, mask):
        v_last = None
        for i in range(self.num_layers):
            v, v_last = self.rnn_list[i].forward(v, mask)

        return v, v_last


class AttentionPooling(torch.nn.Module):
    """
    Attention-Pooling for pointer net init hidden state generate.
    Equal to Self-Attention + MLP
    Modified from r-net.
    Args:
        input_size: The number of expected features in the input uq
        output_size: The number of expected features in the output rq_o
    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.
    Outputs: output
        - **output** (batch, output_size): tensor containing the output features
    """

    def __init__(self, input_size, output_size):
        super(AttentionPooling, self).__init__()

        self.linear_u = torch.nn.Linear(input_size, output_size)
        self.linear_t = torch.nn.Linear(output_size, 1)
        self.linear_o = torch.nn.Linear(input_size, output_size)

    def forward(self, uq, mask):
        q_tanh = F.tanh(self.linear_u(uq))
        q_s = self.linear_t(q_tanh) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, seq_len)

        alpha = masked_softmax(q_s, mask, dim=1)  # (batch, seq_len)
        rq = torch.bmm(alpha.unsqueeze(1), uq.transpose(0, 1)) \
            .squeeze(1)  # (batch, input_size)

        rq_o = F.tanh(self.linear_o(rq))  # (batch, output_size)
        return rq_o


class SelfAttentionGated(torch.nn.Module):
    """
    Self-Attention Gated layer, it`s not weighted sum in the last, but just weighted
    math: \softmax(W*\tanh(W*x)) * x
    Args:
        input_size: The number of expected features in the input x
    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.
    Outputs: output
        - **output** (seq_len, batch, input_size): gated output tensor
    """

    def __init__(self, input_size):
        super(SelfAttentionGated, self).__init__()

        self.linear_g = torch.nn.Linear(input_size, input_size)
        self.linear_t = torch.nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        g_tanh = F.tanh(self.linear_g(x))
        gt = self.linear_t.forward(g_tanh) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, seq_len)

        gt_prop = masked_softmax(gt, x_mask, dim=1)
        gt_prop = gt_prop.transpose(0, 1).unsqueeze(2)  # (seq_len, batch, 1)
        x_gt = x * gt_prop

        return x_gt


class SelfGated(torch.nn.Module):
    """
    Self-Gated layer. math: \sigmoid(W*x) * x
    """

    def __init__(self, input_size):
        super(SelfGated, self).__init__()

        self.linear_g = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        x_l = self.linear_g(x)  # (seq_len, batch, input_size)
        x_gt = F.sigmoid(x_l)

        x = x * x_gt

        return x


class SeqToSeqAtten(torch.nn.Module):
    """
    Args:
        -
    Inputs:
        - h1: (seq1_len, batch, hidden_size)
        - h1_mask: (batch, seq1_len)
        - h2: (seq2_len, batch, hidden_size)
        - h2_mask: (batch, seq2_len)
    Outputs:
        - output: (seq1_len, batch, hidden_size)
        - alpha: (batch, seq1_len, seq2_len)
    """

    def __init__(self):
        super(SeqToSeqAtten, self).__init__()

    def forward(self, h1, h2, h2_mask):
        h1 = h1.transpose(0, 1)
        h2 = h2.transpose(0, 1)

        alpha = h1.bmm(h2.transpose(1, 2))  # (batch, seq1_len, seq2_len)
        alpha = masked_softmax(alpha, h2_mask.unsqueeze(1), dim=2)  # (batch, seq1_len, seq2_len)

        alpha_seq2 = alpha.bmm(h2)  # (batch, seq1_len, hidden_size)
        alpha_seq2 = alpha_seq2.transpose(0, 1)

        return alpha_seq2, alpha


class SelfSeqAtten(torch.nn.Module):
    """
    Args:
        -
    Inputs:
        - h: (seq_len, batch, hidden_size)
        - h_mask: (batch, seq_len)
    Outputs:
        - output: (seq_len, batch, hidden_size)
        - alpha: (batch, seq_len, seq_len)
    """

    def __init__(self):
        super(SelfSeqAtten, self).__init__()

    def forward(self, h, h_mask):
        h = h.transpose(0, 1)
        batch, seq_len, _ = h.shape

        alpha = h.bmm(h.transpose(1, 2))  # (batch, seq_len, seq_len)

        # make element i==j to zero
        mask = torch.eye(seq_len, dtype=torch.uint8, device=h.device)
        mask = mask.unsqueeze(0)
        alpha.masked_fill_(mask, 0.)

        alpha = masked_softmax(alpha, h_mask.unsqueeze(1), dim=2)
        alpha_seq = alpha.bmm(h)

        alpha_seq = alpha_seq.transpose(0, 1)
        return alpha_seq, alpha


class SFU(torch.nn.Module):
    """
    only two input, one input vector and one fusion vector
    Args:
        - input_size:
        - fusions_size:
    Inputs:
        - input: (seq_len, batch, input_size)
        - fusions: (seq_len, batch, fusions_size)
    Outputs:
        - output: (seq_len, batch, input_size)
    """

    def __init__(self, input_size, fusions_size):
        super(SFU, self).__init__()

        self.linear_r = torch.nn.Linear(input_size + fusions_size, input_size)
        self.linear_g = torch.nn.Linear(input_size + fusions_size, input_size)

    def forward(self, input, fusions):
        m = torch.cat((input, fusions), dim=-1)

        r = F.tanh(self.linear_r(m))  # (seq_len, batch, input_size)
        g = F.sigmoid(self.linear_g(m))  # (seq_len, batch, input_size)
        o = g * r + (1 - g) * input

        return o


class MemPtrNet(torch.nn.Module):
    """
    memory pointer net
    Args:
        - input_size: zs and hc size
        - hidden_size:
        - dropout_p:
    Inputs:
        - zs: (batch, input_size)
        - hc: (seq_len, batch, input_size)
        - hc_mask: (batch, seq_len)
    Outputs:
        - ans_out: (ans_len, batch, seq_len)
        - zs_new: (batch, input_size)
    """

    def __init__(self, input_size, hidden_size, dropout_p):
        super(MemPtrNet, self).__init__()

        self.start_net = ForwardNet(input_size=input_size * 3, hidden_size=hidden_size, dropout_p=dropout_p)
        self.start_sfu = SFU(input_size, input_size)
        self.end_net = ForwardNet(input_size=input_size * 3, hidden_size=hidden_size, dropout_p=dropout_p)
        self.end_sfu = SFU(input_size, input_size)

        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, hc, hc_mask, zs):
        hc = self.dropout(hc)

        # start position
        zs_ep = zs.unsqueeze(0).expand(hc.size())  # (seq_len, batch, input_size)
        x = torch.cat((hc, zs_ep, hc * zs_ep), dim=-1)  # (seq_len, batch, input_size*3)
        start_p = self.start_net(x, hc_mask)  # (batch, seq_len)

        us = start_p.unsqueeze(1).bmm(hc.transpose(0, 1)).squeeze(1)  # (batch, input_size)
        ze = self.start_sfu(zs, us)  # (batch, input_size)

        # end position
        ze_ep = ze.unsqueeze(0).expand(hc.size())
        x = torch.cat((hc, ze_ep, hc * ze_ep), dim=-1)
        end_p = self.end_net(x, hc_mask)

        ue = end_p.unsqueeze(1).bmm(hc.transpose(0, 1)).squeeze(1)
        zs_new = self.end_sfu(ze, ue)

        ans_out = torch.stack([start_p, end_p], dim=0)  # (ans_len, batch, seq_len)

        # make sure not nan loss
        new_mask = 1 - hc_mask.unsqueeze(0).type(torch.uint8)
        ans_out.masked_fill_(new_mask, 1e-6)

        return ans_out, zs_new


class ForwardNet(torch.nn.Module):
    """
    one hidden layer and one softmax layer.
    Args:
        - input_size:
        - hidden_size:
        - output_size:
        - dropout_p:
    Inputs:
        - x: (seq_len, batch, input_size)
        - x_mask: (batch, seq_len)
    Outputs:
        - beta: (batch, seq_len)
    """

    def __init__(self, input_size, hidden_size, dropout_p):
        super(ForwardNet, self).__init__()

        self.linear_h = torch.nn.Linear(input_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, 1)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x, x_mask):
        h = F.relu(self.linear_h(x))
        h = self.dropout(h)
        o = self.linear_o(h)
        o = o.squeeze(2).transpose(0, 1)  # (batch, seq_len)

        beta = masked_softmax(o, x_mask, dim=1)
        return beta
