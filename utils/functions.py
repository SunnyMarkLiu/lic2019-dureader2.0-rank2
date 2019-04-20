#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/20 10:01
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functools import reduce


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
    FROM KERAS
    Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def to_long_tensor(np_array):
    """
    convert to long torch tensor
    :param np_array:
    :return:
    """
    return torch.from_numpy(np_array).type(torch.long)


def to_float_tensor(np_array):
    """
    convert to long torch tensor
    :param np_array:
    :return:
    """
    return torch.from_numpy(np_array).type(torch.float)


def count_parameters(model):
    """
    get parameters count that require grad
    :param model:
    :return:
    """
    parameters_num = 0
    for par in model.parameters():
        if not par.requires_grad:
            continue

        tmp_par_shape = par.size()
        tmp_par_size = 1
        for ele in tmp_par_shape:
            tmp_par_size *= ele
        parameters_num += tmp_par_size
    return parameters_num


def compute_mask(v, padding_idx=0):
    """
    compute mask on given tensor v
    :param v:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(v, padding_idx).float()
    return mask


def generate_mask(batch_length):
    """
    generate mask with given length of each element in batch
    :param batch_length: tensor
    :return:
    """
    sum_one = torch.sum(batch_length)
    one = torch.ones(sum_one.item())

    mask_packed = torch.nn.utils.rnn.PackedSequence(one, batch_length)
    mask, _ = torch.nn.utils.rnn.pad_packed_sequence(mask_packed)

    return mask


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def draw_heatmap(x, xlabels, ylabels, x_top=False):
    """
    draw matrix heatmap with matplotlib
    :param x:
    :param xlabels:
    :param ylabels:
    :param x_top:
    :return:
    """
    # Plot it out
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(x, cmap=plt.cm.Blues, alpha=0.8)

    # Format
    fig = plt.gcf()
    fig.set_size_inches(8, 11)

    # turn off the frame
    ax.set_frame_on(False)
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(x.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(x.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    if x_top:
        ax.xaxis.tick_top()

    ax.set_xticklabels(xlabels, minor=False)
    ax.set_yticklabels(ylabels, minor=False)

    # rotate the
    plt.xticks(rotation=90)

    ax.grid(False)

    # Turn off all the ticks
    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False


def draw_heatmap_sea(x, xlabels, ylabels, answer, save_path, inches=(11, 3), bottom=0.45, linewidths=0.2):
    """
    draw matrix heatmap with seaborn
    :param x:
    :param xlabels:
    :param ylabels:
    :param answer:
    :param save_path:
    :param inches:
    :param bottom:
    :param linewidths:
    :return:
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=bottom)
    plt.title('Answer: ' + answer)
    sns.heatmap(x, linewidths=linewidths, ax=ax, cmap='Blues', xticklabels=xlabels, yticklabels=ylabels)
    fig.set_size_inches(inches)
    fig.savefig(save_path)


def answer_search(answer_prop, mask, max_tokens=15):
    """
    global search best answer for model predict
    :param answer_prop: (batch, answer_len, context_len)
    :return:
    """
    batch_size = answer_prop.shape[0]
    context_len = answer_prop.shape[2]

    # get min length
    lengths = mask.data.eq(1).long().sum(1).squeeze()
    min_length, _ = torch.min(lengths, 0)
    min_length = min_length.item()

    # max move steps
    max_move = max_tokens + context_len - min_length
    max_move = min(context_len, max_move)

    ans_s_p = answer_prop[:, 0, :]
    ans_e_p = answer_prop[:, 1, :]
    b_zero = answer_prop.new_zeros(batch_size, 1)

    # each step, move ans-start-prop matrix to right with one element
    ans_s_e_p_lst = []
    for i in range(max_move):
        tmp_ans_s_e_p = ans_s_p * ans_e_p
        ans_s_e_p_lst.append(tmp_ans_s_e_p)

        ans_s_p = ans_s_p[:, :(context_len - 1)]
        ans_s_p = torch.cat((b_zero, ans_s_p), dim=1)
    ans_s_e_p = torch.stack(ans_s_e_p_lst, dim=2)

    # get the best end position, and move steps
    max_prop1, max_prop_idx1 = torch.max(ans_s_e_p, 1)
    max_prop2, max_prop_idx2 = torch.max(max_prop1, 1)

    ans_e = max_prop_idx1.gather(1, max_prop_idx2.unsqueeze(1)).squeeze(1)
    # ans_e = max_prop_idx1[:, max_prop_idx2].diag()  # notice that only diag element valid, the same with top ways
    ans_s = ans_e - max_prop_idx2

    ans_range = torch.stack((ans_s, ans_e), dim=1)
    return ans_range


def flip(tensor, flip_dim=0):
    """
    flip a tensor on specific dim
    :param tensor:
    :param flip_dim:
    :return:
    """
    idx = [i for i in range(tensor.size(flip_dim) - 1, -1, -1)]
    idx = tensor.new_tensor(idx, dtype=torch.long)
    inverted_tensor = tensor.index_select(flip_dim, idx)
    return inverted_tensor


def del_zeros_right(tensor):
    """
    delete the extra zeros in the right column
    :param tensor: (batch, seq_len)
    :return:
    """

    seq_len = tensor.shape[1]
    last_col = seq_len
    for i in range(seq_len - 1, -1, -1):
        tmp_col = tensor[:, i]
        tmp_sum_col = torch.sum(tmp_col)
        if tmp_sum_col > 0:
            break

        last_col = i

    tensor = tensor[:, :last_col]
    return tensor, last_col


def masked_flip(vin, mask, flip_dim=0):
    """
    flip a tensor
    :param vin: (..., batch, ...), batch should on dim=1, input batch with padding values
    :param mask: (batch, seq_len), show whether padding index
    :param flip_dim: dim to flip on
    :return:
    """
    length = mask.data.eq(1).long().sum(1)
    batch_size = vin.shape[1]

    flip_list = []
    for i in range(batch_size):
        cur_tensor = vin[:, i, :]
        cur_length = length[i]

        idx = list(range(cur_length - 1, -1, -1)) + list(range(cur_length, vin.shape[flip_dim]))
        idx = vin.new_tensor(idx, dtype=torch.long)

        cur_inv_tensor = cur_tensor.unsqueeze(1).index_select(flip_dim, idx).squeeze(1)
        flip_list.append(cur_inv_tensor)
    inv_tensor = torch.stack(flip_list, dim=1)

    return inv_tensor


def pop_dict_keys(d, ks_sub):
    """
    delete keys in dict d that has a substring in ks_sub
    :param d:
    :param ks_sub:
    :return:
    """
    for x in list(d.keys()):
        for sub in ks_sub:
            if sub in x:
                del d[x]
                break

    return d


def multi_scale_ptr(ptr_net, ptr_init_h, hr, hr_mask, scales):
    """
    for multi-scale pointer net output
    :param ptr_net:
    :param ptr_init_h: pointer net init hidden state
    :param hr: (seq_len, batch, hidden_size), question-aware passage representation
    :param hr_mask: (batch, seq_len)
    :param scales: list of different scales, for example: [1, 2, 4]. it should be even numbers
    :return:
    """
    seq_len = hr.shape[0]
    batch_size = hr.shape[1]
    ans_range_prop = hr.new_zeros((2, batch_size, seq_len))
    cut_idx = list(range(seq_len))

    for si, s in enumerate(scales):

        # down sampling
        scale_seq_len = int(seq_len / s)
        down_idx = [i*s + s - 1 for i in range(scale_seq_len)]
        if seq_len % s != 0:
            down_idx.append(seq_len-1)
        down_hr = hr[down_idx]
        down_hr_mask = hr_mask[:, down_idx]

        down_ans_range_prop = ptr_net[si].forward(down_hr, down_hr_mask, ptr_init_h)    # (answer_len, batch, seq_len)

        # up sampling
        down_seq_len = down_ans_range_prop.shape[2]
        up_idx = [[i for _ in range(s)] for i in range(down_seq_len)]
        up_idx = list(reduce(lambda x, y: x+y, up_idx))
        up_ans_range_prop = down_ans_range_prop[:, :, up_idx]
        up_ans_range_prop = up_ans_range_prop[:, :, cut_idx]

        ans_range_prop += up_ans_range_prop

    ans_range_prop /= len(scales)
    ans_range_prop = ans_range_prop.transpose(0, 1)  # (batch, answer_len, seq_len)

    return ans_range_prop
