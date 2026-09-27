from fused_version.self_attention_fused import self_attention_fused
from common.feed_forward import feed_forward
from common.softmax_layernorm import layernorm

def transformer_block_fused(X, P):
    O, _ = self_attention_fused(X, P["Wq"], P["Wk"], P["Wv"], P["Wo"])
    H1 = layernorm(X + O, P["g1"], P["b1"])
    F = feed_forward(H1, P["W1"], P["b1f"], P["W2"], P["b2f"])
    return layernorm(H1 + F, P["g2"], P["b2"])