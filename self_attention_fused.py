import numpy as np

from softmax_layernorm import softmax

def self_attention_from_wqkv(x, Wqkv, Wo):
    qkv = x @ Wqkv
    Q, K, V = np.split(qkv, 3, axis=-1)
    A = softmax(Q @ K.T / np.sqrt(Q.shape[-1]))
    return (A @ V) @ Wo, A

def self_attention_fused(x, Wq, Wk, Wv, Wo):
    Wqkv = np.concatenate([Wq, Wk, Wv], axis=-1)
    return self_attention_from_wqkv(x, Wqkv, Wo)
