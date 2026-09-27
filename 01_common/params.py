import numpy as np

def make_params(rng, d_model, d_ff, T):
   
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
    return X, P
