import numpy as np

from common.feed_forward import feed_forward
from common.quanti_func import build_pq, dequant_params
from fused_version.self_attention_fused import self_attention_from_wqkv
from common.softmax_layernorm import layernorm


def quantized_transformer_block_fused(X, Pq):
    W = dequant_params(Pq)
    Wqkv = np.concatenate([W["Wq"], W["Wk"], W["Wv"]], axis=-1)
    O, _ = self_attention_from_wqkv(X, Wqkv, W["Wo"])
    H1 = layernorm(X + O, W["g1"], W["b1"])
    F = feed_forward(H1, W["W1"], W["b1f"], W["W2"], W["b2f"])
    return layernorm(H1 + F, W["g2"], W["b2"])


def build_prepared(P):
    W = dequant_params(build_pq(P))
    prepared = {k: W[k] for k in ("Wo", "W1", "W2", "g1", "b1", "g2", "b2", "b1f", "b2f")}
    prepared["Wqkv"] = np.concatenate([W["Wq"], W["Wk"], W["Wv"]], axis=-1)
    return prepared


def quantized_block_prepared(X, prepared):
    O, _ = self_attention_from_wqkv(X, prepared["Wqkv"], prepared["Wo"])
    H1 = layernorm(X + O, prepared["g1"], prepared["b1"])
    F = feed_forward(H1, prepared["W1"], prepared["b1f"], prepared["W2"], prepared["b2f"])
    return layernorm(H1 + F, prepared["g2"], prepared["b2"])
