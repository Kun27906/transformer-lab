# -*- coding: utf-8 -*-
"""QKV 融合版自注意力（成员丙）

题目第 4 节：把 Q = X@Wq、K = X@Wk、V = X@Wv 三次矩阵乘
换成 Wqkv = concat([Wq,Wk,Wv]) 一次矩阵乘后再切开。
"""

import numpy as np

from softmax_layernorm import softmax


def self_attention_from_wqkv(x, Wqkv, Wo):
    """已知拼好的 Wqkv 时做一次单头自注意力，返回 (输出, 注意力矩阵 A)。

    「总融合版」和「总融合·部署态」两个 Block 都调用这一段，
    注意力逻辑因此只有一处实现。
    """
    qkv = x @ Wqkv
    Q, K, V = np.split(qkv, 3, axis=-1)
    A = softmax(Q @ K.T / np.sqrt(Q.shape[-1]))
    return (A @ V) @ Wo, A


def self_attention_fused(x, Wq, Wk, Wv, Wo):
    """QKV 融合版：每次调用时拼接 Wqkv，再交给 self_attention_from_wqkv。

    返回值与 self_attention 一致，都是 (输出, 注意力矩阵 A)。
    """
    Wqkv = np.concatenate([Wq, Wk, Wv], axis=-1)
    return self_attention_from_wqkv(x, Wqkv, Wo)
