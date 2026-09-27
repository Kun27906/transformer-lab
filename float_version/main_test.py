import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from common.params import make_params
from float_version.self_attention import self_attention
from float_version.transformer_block import transformer_block

X, P = make_params(np.random.default_rng(0), d_model=4, d_ff=8, T=3)

print('X shape:', X.shape)                                  
attn_out, A = self_attention(X, P['Wq'], P['Wk'], P['Wv'], P['Wo'])
print('Q/K/V shape:', attn_out.shape)                       
print('attention matrix A:\n', np.round(A, 4))
print('each row sums to 1:', A.sum(axis=1))

H2 = transformer_block(X, P)
print('final output H2:\n', np.round(H2, 4))
