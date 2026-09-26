import numpy as np
from ops_bing import softmax
def self_attention_fused(x, Wq, Wk, Wv, Wo):
   
    Wqkv =np.concatenate([Wq,Wk,Wv],axis=-1)
    qkv =x@Wqkv
    Q, K, V =np.split(qkv,3,axis=-1)
    dk = Q.shape[-1]
    S = Q@K.T/np.sqrt(dk)
    P =softmax(S) 
    out = (P@V)@Wo
    return out, P
