# -*- coding: utf-8 -*-
"""方向B 最小对照（成员甲）：浮点版 Block vs 量化版 Block

输出两者结果并排对比，并给出最大绝对误差与相对误差。
完整的五实现对比（含融合、计时、存储账）见 test_compare_all.py。
"""

import numpy as np

from params import make_params
from quanti_func import build_pq, inaccuracy
from transformer_block import transformer_block
from transformer_block_int import quantized_transformer_block

X, P = make_params(np.random.default_rng(0), d_model=4, d_ff=8, T=3)
Pq = build_pq(P)                       # 量化参数统一由 quanti_func.build_pq 生成

H2 = transformer_block(X, P)
H2q = quantized_transformer_block(X, Pq)

print('浮点版 H2:\n', np.round(H2, 4))
print('量化版 H2q:\n', np.round(H2q, 4))

abs_err, rel_err = inaccuracy(H2, H2q)
print(f'最大绝对误差 = {abs_err:.6f}')
print(f'相对误差     = {rel_err * 100:.4f}%')
