import numpy as np
from softmax_layernorm import softmax
from softmax_layernorm import layernorm
from feed_forward import feed_forward
from transformer_block import transformer_block

rng = np.random.default_rng(0)

d_model, d_ff, T = 4, 8, 3
X = rng.normal(size=(T, d_model))
P = dict(
    Wq=rng.normal(size=(d_model, d_model)),
    Wk=rng.normal(size=(d_model, d_model)),
    Wv=rng.normal(size=(d_model, d_model)),
    Wo=rng.normal(size=(d_model, d_model)),
    g1=np.ones(d_model), b1=np.zeros(d_model),
    g2=np.ones(d_model), b2=np.zeros(d_model),
    W1=rng.normal(size=(d_model, d_ff)), b1f=np.zeros(d_ff),
    W2=rng.normal(size=(d_ff, d_model)), b2f=np.zeros(d_model),
)

print('X shape:', X.shape)
Q, K, V = X @ P['Wq'], X @ P['Wk'], X @ P['Wv']
print('Q/K/V shape:', Q.shape)
S = Q @ K.T / np.sqrt(d_model)
A = softmax(S)
print('attention matrix A:\n', np.round(A, 4))
print('each row sums to 1:', A.sum(axis=1))
H = transformer_block(X, P)
print('final output H2:\n', np.round(H, 4))

