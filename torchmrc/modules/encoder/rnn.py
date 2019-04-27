#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/26 21:50
"""
import torch


class RNNBase(torch.nn.Module):
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
        super(RNNBase, self).__init__()
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
