#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/27 15:27
"""
import torch
import torch.nn.functional as F


class MRCStartEndNLLLoss(torch.nn.modules.loss._Loss):
    """
    a standard negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    Shape:
        - y_pred: (batch, answer_len, prob)
        - y_true: (batch, answer_len)
        - output: loss
    """
    def __init__(self):
        super(MRCStartEndNLLLoss, self).__init__()

    def forward(self, start_probs, start_labels, end_probs, end_labels):
        start_probs_log = torch.log(start_probs)
        start_loss = F.nll_loss(start_probs_log, start_labels)

        end_probs_log = torch.log(end_probs)
        end_loss = F.nll_loss(end_probs_log, end_labels)
        return start_loss + end_loss
