import numpy as np


# ===== 临时：乙还没交 softmax，先用这个替身 =====
# ===== 交付前必须换成乙的正式版，并删掉注释 =====
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
# =================================================


def self_attention(x, Wq, Wk, Wv, Wo):
    Q =x@Wq     # 空1
    K =x@Wk     # 空2
    V =x@Wv      # 空3

    dk = Q.shape[-1]
    S = Q@K.T/np.sqrt(dk)       # 空4

    P = softmax(S)       # 空5

    out = (P@V)@Wo     # 空6

    return out, P
