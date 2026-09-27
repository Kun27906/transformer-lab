from common.feed_forward import feed_forward
from common.quanti_func import dequant_params
from float_version.self_attention import self_attention
from common.softmax_layernorm import layernorm


def quantized_transformer_block(X, Pq):
    W = dequant_params(Pq)
    O, _ = self_attention(X, W["Wq"], W["Wk"], W["Wv"], W["Wo"])
    H1 = layernorm(X + O, W["g1"], W["b1"])
    F = feed_forward(H1, W["W1"], W["b1f"], W["W2"], W["b2f"])
    return layernorm(H1 + F, W["g2"], W["b2"])
