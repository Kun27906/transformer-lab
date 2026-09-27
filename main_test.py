# -*- coding: utf-8 -*-
"""第二阶段主入口（成员甲）：最小可跑 demo

打印各步骤的张量维度、注意力矩阵与最终输出。
题目第 3 节的 5 项自动检查、以及误差/存储/计时的完整对比，
统一放在 test_compare_all.py 里，这里只保留最小演示。
"""

import numpy as np

from params import make_params
from self_attention import self_attention
from transformer_block import transformer_block

# 输入与权重统一从 params.make_params 取，保证与组内其他实验完全一致
X, P = make_params(np.random.default_rng(0), d_model=4, d_ff=8, T=3)

print('X shape:', X.shape)                                  # (3, 4)
attn_out, A = self_attention(X, P['Wq'], P['Wk'], P['Wv'], P['Wo'])
print('Q/K/V shape:', attn_out.shape)                       # 注意力输出与输入同形
print('attention matrix A:\n', np.round(A, 4))
print('each row sums to 1:', A.sum(axis=1))

H2 = transformer_block(X, P)
print('final output H2:\n', np.round(H2, 4))
