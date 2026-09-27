# -*- coding: utf-8 -*-
"""方向B：对称 INT8 量化 / 反量化 / 误差与存储统计（成员乙）

对称 INT8 量化公式：
    s      = max|W| / 127
    W_int8 = round(W / s)
    W_hat  = s * W_int8          （反量化，之后用浮点做矩阵乘）

本文件是全项目唯一处理"量化参数"的地方：
哪些键要量化、元组怎么反量化、误差怎么算、存了多少字节，都只在这里写一遍。
"""

import numpy as np

INT8_MAX_AS_FLOAT = 127.0
QUANT_KEYS = ("Wq", "Wk", "Wv", "Wo", "W1", "W2")        # 需要量化的 6 个权重矩阵
PLAIN_KEYS = ("g1", "b1", "g2", "b2", "b1f", "b2f")      # 保持浮点的 γ/β 与偏置


def sym_quanti_int8(W_fp32):
    """对称 INT8 量化。返回 (W_int8, s)，s 为量化刻度。"""
    W_fp64 = np.asarray(W_fp32, dtype=np.float64)
    quant_scale = np.max(np.abs(W_fp64)) / INT8_MAX_AS_FLOAT
    W_int8 = np.clip(np.round(W_fp64 / quant_scale),
                     -INT8_MAX_AS_FLOAT - 1, INT8_MAX_AS_FLOAT).astype(np.int8)
    return W_int8, quant_scale


def dequanti_int8(W_int8, quant_scale):
    """反量化：W_hat = s * W_int8"""
    W_deq = W_int8.astype(np.float32) * quant_scale
    return W_deq


def build_pq(P):
    """把浮点参数字典 P 变成量化参数字典 Pq。

    6 个权重量化成 (W_int8, s) 元组，γ/β/偏置原样保留。
    全项目只有这一个入口，避免每个实验各写一遍循环。
    """
    Pq = {k: sym_quanti_int8(P[k]) for k in QUANT_KEYS}
    Pq.update({k: P[k] for k in PLAIN_KEYS})
    return Pq


def dequant_params(Pq):
    """把量化参数字典还原成可直接做矩阵乘的浮点参数字典。

    规则只有一条：值是 (W_int8, s) 元组就反量化，否则原样保留。
    这条规则写在这里一处，Block 们只负责调用。
    """
    return {k: (dequanti_int8(*v) if isinstance(v, tuple) else v) for k, v in Pq.items()}


def inaccuracy(reference, candidate):
    """比较候选结果与基准结果，返回 (最大绝对误差, 相对误差)。

    相对误差 = 最大绝对误差 / 基准张量的最大绝对值（按 max 范数归一）。
    这里不用逐元素 |Δ| / |ref|：基准里只要有一个分量接近 0，
    逐元素相对误差就会爆成很大的数，反而失去参考价值。
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    absolute_err = float(np.max(np.abs(cand - ref)))
    scale = float(np.max(np.abs(ref)))
    relative_err = absolute_err / scale if scale > 0 else 0.0
    return absolute_err, relative_err


def storage_bytes(P, Pq=None):
    """统计 6 个权重矩阵的存储开销，返回 (fp32 字节, int8 字节, 刻度字节)。

    int8 量化后每个权重还要额外保存一个 float32 刻度 s，这笔开销必须算进去，
    否则会高估压缩收益。
    """
    if Pq is None:
        Pq = build_pq(P)
    fp_bytes = int(sum(np.asarray(P[k]).size * 4 for k in QUANT_KEYS))
    int8_bytes = int(sum(Pq[k][0].size * 1 for k in QUANT_KEYS))
    scale_bytes = int(len(QUANT_KEYS) * 4)
    return fp_bytes, int8_bytes, scale_bytes
