import numpy as np
from common.softmax_layernorm import softmax

def self_attention(x, Wq, Wk, Wv, Wo):
    Q =x@Wq
    K =x@Wk
    V =x@Wv

    d_k = Q.shape[-1]
    S = Q@K.T/np.sqrt(d_k)

    P = softmax(S)

    out = (P@V)@Wo

    return out, P
