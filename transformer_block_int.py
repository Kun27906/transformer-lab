# -*- coding: utf-8 -*-
"""单量化版 Block（成员甲）

与 transformer_block.py 的唯一区别：权重先反量化再参与计算。
子层调用完全一致，所以两者可以直接逐元素对比误差。
"""

from feed_forward import feed_forward
from quanti_func import dequant_params
from self_attention import self_attention
from softmax_layernorm import layernorm


def quantized_transformer_block(X, Pq):
    """Pq 里 6 个权重是 (W_int8, s) 元组，γ/β/偏置是浮点。"""
    W = dequant_params(Pq)
    O, _ = self_attention(X, W["Wq"], W["Wk"], W["Wv"], W["Wo"])
    H1 = layernorm(X + O, W["g1"], W["b1"])
    F = feed_forward(H1, W["W1"], W["b1f"], W["W2"], W["b2f"])
    return layernorm(H1 + F, W["g2"], W["b2"])
