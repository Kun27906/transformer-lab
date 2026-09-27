# -*- coding: utf-8 -*-
"""总融合版 Block（成员甲）：INT8 量化 + QKV 融合

三种形态，对应三种开销假设：
    quantized_transformer_block_fused(X, Pq)   每步都反量化 + 拼 Wqkv（把全部开销算进去）
    build_prepared(P) + quantized_block_prepared(X, prepared)
                                               部署态：反量化与拼接只做一次并固定下来
"""

import numpy as np

from feed_forward import feed_forward
from quanti_func import build_pq, dequant_params
from self_attention_fused import self_attention_from_wqkv
from softmax_layernorm import layernorm


def quantized_transformer_block_fused(X, Pq):
    """Pq 为量化参数字典；本函数内部每次调用都反量化并拼接 Wqkv。"""
    W = dequant_params(Pq)
    Wqkv = np.concatenate([W["Wq"], W["Wk"], W["Wv"]], axis=-1)
    O, _ = self_attention_from_wqkv(X, Wqkv, W["Wo"])
    H1 = layernorm(X + O, W["g1"], W["b1"])
    F = feed_forward(H1, W["W1"], W["b1f"], W["W2"], W["b2f"])
    return layernorm(H1 + F, W["g2"], W["b2"])


def build_prepared(P):
    """部署态参数：把量化、反量化、Wqkv 拼接各做一次，之后只剩矩阵乘法。"""
    W = dequant_params(build_pq(P))
    prepared = {k: W[k] for k in ("Wo", "W1", "W2", "g1", "b1", "g2", "b2", "b1f", "b2f")}
    prepared["Wqkv"] = np.concatenate([W["Wq"], W["Wk"], W["Wv"]], axis=-1)
    return prepared


def quantized_block_prepared(X, prepared):
    """用部署态参数直接算，不含任何量化/拼接开销。"""
    O, _ = self_attention_from_wqkv(X, prepared["Wqkv"], prepared["Wo"])
    H1 = layernorm(X + O, prepared["g1"], prepared["b1"])
    F = feed_forward(H1, prepared["W1"], prepared["b1f"], prepared["W2"], prepared["b2f"])
    return layernorm(H1 + F, prepared["g2"], prepared["b2"])
