import numpy as np
from self_attention import self_attention

I2 = np.eye(2)
X  = np.array([[1., 1.],
               [1., 0.]])

print(" 例 A：四个权重都是 I2 ")
outA, PA = self_attention(X, I2, I2, I2, I2)
print("P   =")
print(PA)
print("out =")
print(outA)

print()
print("例 B：只把 Wv 换成 [[1,2],[3,4]] ")
Wv = np.array([[1., 2.],
               [3., 4.]])
outB, PB = self_attention(X, I2, I2, Wv, I2)
print("P   =")
print(PB)
print("out =")
print(outB)
print()
print("两次的 P 一样吗？最大差 =", np.abs(PA - PB).max())
