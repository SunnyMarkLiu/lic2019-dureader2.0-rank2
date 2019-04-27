#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/26 20:38
"""
import torch


def compute_mask(v, padding_idx=0):
    """
    compute mask on given tensor v
    :param v:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(v, padding_idx).float()
    return mask
