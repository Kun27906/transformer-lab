# -*- coding: utf-8 -*-
"""全组统一的输入与权重生成（保证 default_rng(0) 下三人结果一致）

固定生成顺序：
    X → Wq → Wk → Wv → Wo → W1 → W2
（γ/β 与偏置的初值直接取 ones/zeros，不消耗随机数）

任何实验都从这里取参数，不要再各自写一遍 rng.normal(...)：
顺序一变，三个人跑出来的数字就对不上了。
"""

import numpy as np


def make_params(rng, d_model, d_ff, T):
    """返回 (X, P)：X 形状 (T, d_model)，P 为参数字典。

    rng 必须是 numpy.random.default_rng(0)。
    """
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
