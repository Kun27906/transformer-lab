import numpy as np
from softmax_layernorm import layernorm
from self_attention import self_attention
from feed_forward import feed_forward

def transformer_block(X,P):
    O,_=self_attention(X,P['Wq'],P['Wk'],P['Wv'],P['Wo'])
    H1=layernorm(X+O, P['g1'], P['b1'])
    F=feed_forward(H1,P['W1'],P['b1f'],P['W2'],P['b2f'])
    H2=layernorm(H1+F, P['g2'], P['b2'])
    return H2