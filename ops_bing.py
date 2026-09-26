import numpy as np


# ===== 临时：乙还没交 softmax，先用这个替身 =====
# ===== 交付前必须换成乙的正式版，并删掉注释 =====
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
# =================================================


def self_attention(x, Wq, Wk, Wv, Wo):
    Q =x@Wq     
    K =x@Wk     
    V =x@Wv     

    dk = Q.shape[-1]
    S = Q@K.T/np.sqrt(dk)      

    P = softmax(S)      
    out = (P@V)@Wo     
    return out, P
def feed_forward(x, W1, b1, W2, b2):
    h = x@W1+b1

    h =np.maximum(h,0)

    out =h@W2+b2 

    return out
