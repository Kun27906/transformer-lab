import numpy as np

from params import make_params
from quanti_func import build_pq, inaccuracy
from transformer_block import transformer_block
from transformer_block_int import quantized_transformer_block

X, P = make_params(np.random.default_rng(0), d_model=4, d_ff=8, T=3)
Pq = build_pq(P)                       

H2 = transformer_block(X, P)
H2q = quantized_transformer_block(X, Pq)

print('浮点版 H2:\n', np.round(H2, 4))
print('量化版 H2q:\n', np.round(H2q, 4))

abs_err, rel_err = inaccuracy(H2, H2q)
print(f'最大绝对误差 = {abs_err:.6f}')
print(f'相对误差     = {rel_err * 100:.4f}%')
