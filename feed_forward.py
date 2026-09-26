import numpy as np

def feed_forward(x,W1,b1,W2,b2):
    return np.maximum(0,x@W1+b1)@W2+b2
