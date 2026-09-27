import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)

def softmax(x):
    x = np.asarray(x, dtype=np.float64)             
    x = x - np.max(x, axis=-1, keepdims=True)        
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def layernorm(x,  gamma=None, beta=None ,eps=1e-5,):
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mu) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out
