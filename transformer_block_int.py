import numpy as np
from softmax_layernorm import softmax
from softmax_layernorm import layernorm
from self_attention import self_attention
from feed_forward import feed_forward
from quanti_func import dequanti_int8

def quantized_transformer_block(X, Pq):
    W = {}
    for k, v in Pq.items():  # 逐对检查
        if isinstance(v, tuple):  # 是 (int8, s) 包裹才处理
            W[k] = dequanti_int8(v[0], v[1])  # 拆包还原，存入新字典
    Q, K, V = X @ W["Wq"], X @ W["Wk"], X @ W["Wv"]
    A = softmax(Q @ K.T / np.sqrt(Q.shape[-1]))
    O = (A @ V) @ W["Wo"]
    H1 = layernorm(X + O, Pq["g1"], Pq["b1"])     # gamma/beta 原样用
    F = np.maximum(0, H1 @ W["W1"] + Pq["b1f"]) @ W["W2"] + Pq["b2f"]
    O1=layernorm(H1 + F, Pq["g2"], Pq["b2"])
    return O1


