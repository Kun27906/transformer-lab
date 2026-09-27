import numpy as np

INT8_MAX_AS_FLOAT = 127.0
QUANT_KEYS = ("Wq", "Wk", "Wv", "Wo", "W1", "W2")        
PLAIN_KEYS = ("g1", "b1", "g2", "b2", "b1f", "b2f")      


def sym_quanti_int8(W_fp32):
    W_fp64 = np.asarray(W_fp32, dtype=np.float64)
    quant_scale = np.max(np.abs(W_fp64)) / INT8_MAX_AS_FLOAT
    W_int8 = np.clip(np.round(W_fp64 / quant_scale),
                     -INT8_MAX_AS_FLOAT - 1, INT8_MAX_AS_FLOAT).astype(np.int8)
    return W_int8, quant_scale


def dequanti_int8(W_int8, quant_scale):
    W_deq = W_int8.astype(np.float32) * quant_scale
    return W_deq


def build_pq(P):
    Pq = {k: sym_quanti_int8(P[k]) for k in QUANT_KEYS}
    Pq.update({k: P[k] for k in PLAIN_KEYS})
    return Pq


def dequant_params(Pq):
    return {k: (dequanti_int8(*v) if isinstance(v, tuple) else v) for k, v in Pq.items()}


def inaccuracy(reference, candidate):
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    absolute_err = float(np.max(np.abs(cand - ref)))
    scale = float(np.max(np.abs(ref)))
    relative_err = absolute_err / scale if scale > 0 else 0.0
    return absolute_err, relative_err


def storage_bytes(P, Pq=None):
    if Pq is None:
        Pq = build_pq(P)
    fp_bytes = int(sum(np.asarray(P[k]).size * 4 for k in QUANT_KEYS))
    int8_bytes = int(sum(Pq[k][0].size * 1 for k in QUANT_KEYS))
    scale_bytes = int(len(QUANT_KEYS) * 4)
    return fp_bytes, int8_bytes, scale_bytes
