# -*- coding: utf-8 -*-
"""最终对比测试（全组）—— 正确性 / 误差 / 存储 / 计时

五种实现的递进关系：
    1. 无优化版        transformer_block                      浮点 + QKV 分开算
    2. 单量化版        quantized_transformer_block            量化 + QKV 分开算
    3. QKV融合版       transformer_block_fused                浮点 + QKV 融合
    4. 总融合版        quantized_transformer_block_fused      量化 + QKV 融合（每次调用都反量化/拼接）
    5. 总融合·部署态   quantized_block_prepared               量化 + QKV 融合（反量化与拼接只做一次）

两部分内容：
    第 1 部分  题目给定参数（X 3×4，d_model=d_k=d_v=4，d_ff=8）
               ① 题目第 3 节要求的 5 项自动检查
               ② 五种实现的绝对误差 + 相对误差（统一调用 quanti_func.inaccuracy）
               ③ 输出矩阵、int8 存储账、计时
    第 2 部分  大模型（128×64，d_ff=256）
               只看计时与存储账，不做误差对比
               （这个规模下误差数值本身没有参考意义，讨论它只会分散结论）

所有数值都来自现成的模块函数，本文件不重新实现任何算子。
"""

import statistics
import time

import numpy as np

from feed_forward import feed_forward
from params import make_params
from quanti_func import build_pq, inaccuracy, storage_bytes
from self_attention import self_attention
from softmax_layernorm import layernorm
from transformer_block import transformer_block
from transformer_block_fused import transformer_block_fused
from transformer_block_int import quantized_transformer_block
from transformer_block_int_fused import (build_prepared,
                                         quantized_block_prepared,
                                         quantized_transformer_block_fused)

# 计时参数：先预热再计时、取中位数（首次调用有缓存与内存分配开销，必须排除）
WARM, REPEAT = 200, 2000            # 小模型：单次极快，多跑几轮
BIG_WARM, BIG_REPEAT = 50, 500      # 大模型：单次开销大，次数相应减少


# ============================================================
# 1. 五种实现：统一在这里构造
# ============================================================

def build_versions(X, P):
    """返回 [(名称, 可调用对象)]，每个可调用对象内部只调用现成的模块函数。"""
    Pq = build_pq(P)
    prepared = build_prepared(P)        # 部署态参数只在构造时做一次
    return [
        ("无优化版",      lambda: transformer_block(X, P)),
        ("单量化版",      lambda: quantized_transformer_block(X, Pq)),
        ("QKV融合版",     lambda: transformer_block_fused(X, P)),
        ("总融合版",      lambda: quantized_transformer_block_fused(X, Pq)),
        ("总融合·部署态", lambda: quantized_block_prepared(X, prepared)),
    ]


# ============================================================
# 2. 正确性自动检查（题目第 3 节 5 项）
# ============================================================

def check_correctness(X, P, tag):
    """返回是否全部通过。所有计算都调用现成算子，不重写公式。"""
    print("=" * 72)
    print(f"[{tag}] 正确性自动检查")
    print("=" * 72)

    # 用现成算子手工搭一遍 Block，顺便验证它与 transformer_block 模块一致
    O, A = self_attention(X, P["Wq"], P["Wk"], P["Wv"], P["Wo"])
    H1 = layernorm(X + O, P["g1"], P["b1"])
    F = feed_forward(H1, P["W1"], P["b1f"], P["W2"], P["b2f"])
    H2_typed = layernorm(H1 + F, P["g2"], P["b2"])
    H2 = transformer_block(X, P)

    Xm = X.copy()
    Xm[0, 0] += 1.0                     # 只动第 1 个 Token 的第 1 个特征
    _, A_mod = self_attention(Xm, P["Wq"], P["Wk"], P["Wv"], P["Wo"])
    H2_mod = transformer_block(Xm, P)

    row_sum = A.sum(axis=1)
    checks = [
        ("注意力矩阵每行之和 ≈ 1",
         bool(np.allclose(row_sum, 1.0, atol=1e-9)),
         f"行和 = {np.round(row_sum, 8)}"),

        ("残差连接两侧形状相同",
         (X + O).shape == X.shape and (H1 + F).shape == H1.shape,
         f"X+attn 为 {X.shape}，H1+FFN 为 {H1.shape}"),

        ("LayerNorm 后每个 Token 均值 ≈ 0",
         bool(np.allclose(H1.mean(axis=1), 0, atol=1e-9) and np.allclose(H2.mean(axis=1), 0, atol=1e-9)),
         f"H1 行均值 = {np.round(H1.mean(axis=1), 10)}"),

        ("大正负数输入不出现 inf/nan",
         bool(np.all(np.isfinite(transformer_block(X * 1000, P)))),
         "X×1000 时输出全部有限"),

        ("改动第 1 个 Token 后结果合理变化",
         bool(np.isfinite(H2_mod).all() and not np.allclose(A_mod, A) and not np.allclose(H2_mod, H2)),
         f"注意力首行变化 {np.round(A_mod[0] - A[0], 4)}，输出最大变化 {np.abs(H2_mod - H2).max():.4f}"),

        ("手工搭的 Block 与 transformer_block 一致",
         bool(np.allclose(H2_typed, H2)),
         "子层函数组合结果与 Block 模块完全一致"),
    ]
    for desc, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        print(f"          {detail}")
    print()
    return all(ok for _, ok, _ in checks)


# ============================================================
# 3. 误差对比（绝对误差 + 相对误差，统一走 inaccuracy）
# ============================================================

def report_errors(tag, X, P, versions):
    """只用于题目给定参数：既要绝对误差，也要相对误差。"""
    print("=" * 72)
    print(f"[{tag}] 误差对比：基准 = 无优化版，X.shape = {X.shape}")
    print("=" * 72)

    base = transformer_block(X, P)                    # 基准只算一次
    rows = []
    for name, fn in versions:
        out = fn()
        abs_err, rel_err = inaccuracy(base, out)      # ← 误差统计统一调用 quanti_func.inaccuracy
        rows.append((name, abs_err, rel_err, out))

    print(f"{'实现':<16}{'最大绝对误差':>14}{'相对误差':>12}")
    print("-" * 72)
    for name, abs_err, rel_err, _ in rows:
        mark = "← 基准" if name == "无优化版" else ""
        print(f"{name:<16}{abs_err:>14.6f}{rel_err * 100:>11.4f}%  {mark}")
    print("-" * 72)
    print("相对误差 = 最大绝对误差 / 基准输出的最大绝对值（按 max 范数归一）\n")

    for name, _, _, out in rows:
        print(f"[{name}] 输出（保留 4 位）：")
        print(np.round(out, 4))
        print("-" * 72)

    fp, iq, sc = storage_bytes(P)
    print(f"int8 存储账（6 个权重矩阵）: fp32 {fp} B → int8 {iq} B + 刻度 {sc} B")
    print(f"  = {iq + sc} B，为 fp32 的 {(iq + sc) / fp * 100:.1f}%（压缩比 {fp / (iq + sc):.2f}x）")
    print()


# ============================================================
# 4. 计时对比
# ============================================================

def bench(fn, warm, n):
    """预热 warm 次后重复 n 次，取中位数。"""
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def report_times(tag, versions, warm, n, show_ratio=True):
    """计时对比。大模型直接调这个，不涉及任何误差计算。"""
    print("=" * 72)
    print(f"[{tag}] 计时对比（预热 {warm} 次，重复 {n} 次，取中位数）")
    print("=" * 72)

    times = [(name, bench(fn, warm, n)) for name, fn in versions]
    base = times[0][1]                                  # 无优化版作为参照
    for name, t in times:
        ratio = f"{t / base:6.3f}x" if show_ratio else ""
        print(f"  {name:<16}{t * 1000:9.4f} ms   {ratio}")
    print()
    return dict(times)


# ============================================================
# 5. 主流程
# ============================================================

if __name__ == "__main__":
    # ---------- 第 1 部分：题目给定参数 ----------
    print("\n########## 第 1 部分：题目给定参数（X 3×4, d_model=4, d_ff=8） ##########\n")
    X, P = make_params(np.random.default_rng(0), d_model=4, d_ff=8, T=3)
    versions = build_versions(X, P)

    check_correctness(X, P, "小模型 3×4")
    report_errors("小模型 3×4", X, P, versions)
    report_times("小模型 3×4", versions, WARM, REPEAT)

    # ---------- 第 2 部分：大模型 ----------
    print("########## 第 2 部分：大模型（X 128×64, d_model=64, d_ff=256） ##########\n")
    X2, P2 = make_params(np.random.default_rng(0), d_model=64, d_ff=256, T=128)
    versions2 = build_versions(X2, P2)

    report_times("大模型 128×64", versions2, BIG_WARM, BIG_REPEAT)

    fp, iq, sc = storage_bytes(P2)
    print(f"int8 存储账（6 个权重矩阵）: fp32 {fp} B → int8 {iq} B + 刻度 {sc} B")
    print(f"  = {iq + sc} B，为 fp32 的 {(iq + sc) / fp * 100:.1f}%（压缩比 {fp / (iq + sc):.2f}x）")
    print("\n说明：大模型部分只比较开销与加速比，不计误差——")
    print("      输入规模放大后，误差数值受随机初始化影响而失去参考意义。")
