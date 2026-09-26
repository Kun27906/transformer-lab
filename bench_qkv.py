import numpy as np
import time
import statistics

rng = np.random.default_rng(0)
X   = rng.standard_normal((128, 64))
Wq  = rng.standard_normal((64, 64))
Wk  = rng.standard_normal((64, 64))
Wv  = rng.standard_normal((64, 64))
Wqkv =np.concatenate([Wq,Wk,Wv],axis=-1)
def separate():
    return X@Wq, X@Wk, X@Wv
def fused():
    qkv = X@Wqkv
    return np.split(qkv,3,axis=-1)
for _ in range(20):
    separate()
    fused()
N = 300
ts, tf = [], []
for _ in range(N):
    t0 = time.perf_counter(); separate(); ts.append(time.perf_counter() - t0)
for _ in range(N):
    t0 = time.perf_counter(); fused();    tf.append(time.perf_counter() - t0)

print("重复次数 =", N)
print("分开版: 最小 %.4f ms   中位数 %.4f ms" % (min(ts)*1000, statistics.median(ts)*1000))
print("融合版: 最小 %.4f ms   中位数 %.4f ms" % (min(tf)*1000, statistics.median(tf)*1000))
print("中位数之比 融合/分开 = %.3f" % (statistics.median(tf)/statistics.median(ts)))
