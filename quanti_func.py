import numpy as np

INT8_MAX_AS_FLOAT = 127.0

def sym_quanti_int8(W_fp32):
    W_fp64 = np.asarray(W_fp32, dtype=np.float64)
    quant_scale = np.max(np.abs(W_fp64)) / INT8_MAX_AS_FLOAT
    W_int8 = np.clip(np.round(W_fp64 / quant_scale), -INT8_MAX_AS_FLOAT-1, INT8_MAX_AS_FLOAT).astype(np.int8)
    return W_int8, quant_scale

def dequanti_int8(W_int8, quant_scale):
    W_deq = W_int8.astype(np.float32) * quant_scale
    return W_deq

def inaccuracy(W_deq, W_fp32):
    W_fp32 = np.asarray(W_fp32, dtype=np.float32)
    absolute_err = np.max(np.abs(W_deq - W_fp32))
    relative_err = np.round((absolute_err / (W_fp32 + 1e-12)) * 100.0, 2)
    return absolute_err, relative_err