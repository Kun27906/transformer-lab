import numpy as np
from softmax_layernorm import softmax
from softmax_layernorm import layernorm
from feed_forward import feed_forward
from transformer_block import transformer_block
from transformer_block_int import quantized_transformer_block
from quanti_func import sym_quanti_int8

rng = np.random.default_rng(0)

d_model, d_ff, T = 4, 8, 3
X = rng.normal(size=(T, d_model))
P = dict(
    Wq=rng.normal(size=(d_model, d_model)), Wk=rng.normal(size=(d_model, d_model)),
    Wv=rng.normal(size=(d_model, d_model)), Wo=rng.normal(size=(d_model, d_model)),
    g1=np.ones(d_model), b1=np.zeros(d_model),
    g2=np.ones(d_model), b2=np.zeros(d_model),
    W1=rng.normal(size=(d_model, d_ff)), b1f=np.zeros(d_ff),
    W2=rng.normal(size=(d_ff, d_model)), b2f=np.zeros(d_model),
)
H = transformer_block(X, P)

Pq = {}
for k in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]:
    Pq[k] = sym_quanti_int8(P[k])  # 大矩阵 → (int8, s)
for k in ["g1", "b1", "g2", "b2", "b1f", "b2f"]:
    Pq[k] = P[k]  # 小向量 → 原样浮点

H2q = quantized_transformer_block(X, Pq)

print('final output H2:\n', np.round(H, 4))
print('final output H2q:\n', np.round(H2q, 4))

