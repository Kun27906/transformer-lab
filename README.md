# Transformer 小组部署分工方案

题目二：三阶段推进、全员共读、全员实操、两易一难

## 零、分工总则

1. 第一阶段“读懂原理”设为共同工作：三人一起读、一起讨论、一起通过同一份原理验收，防止“只懂自己那块、合上就懵”。
2. 第二、三阶段实操任务每人独立上手写代码/跑实验，但任务难度有意错开：两人做较轻松的基础与对比任务，一人挑战偏难的优化核心。难活单独认领，本文档中称“挑战者”。
3. 两个方向（A：KV Cache；B：INT8 量化）适用同一套分工框架，只是第三阶段的“难活”内容不同，下文分开列出。

| 角色 | 承担任务难度 | 说明 |
| --- | --- | --- |
| 成员甲（挑战者） | 偏难 | 认领第三阶段优化核心（A：KV Cache 缓存版解码器 / B：量化版 Block 与误差排查），要求编程基础最好的一位 |
| 成员乙 | 较轻松 | 负责第二阶段主流程实现与代码整理，模板化、资料全，容易做出成果 |
| 成员丙 | 较轻松 | 负责自动检查与数据对比实验（计时、误差表、存储表），照清单逐项跑即可 |

注意：难度错开不等于工作量错开——乙丙的任务量与甲相当，只是出错风险低、资料全。汇报时三人都有可展示的实操成果。

## 一、第一阶段：读懂 Transformer

本阶段不分工，三人做完全相同的事，以“组内共读 + 轮流讲解 + 同卷验收”推进。

共读材料：《Attention Is All You Need》论文 3.2 节（注意力公式）、配发的 Transformer 原理详解文档；题目二第 1、2 节。

## 二、第二阶段：完成简单架构

本阶段三人都要写代码、都要跑通，但任务难度错开：

| 成员 | 任务 | 难度 | 具体内容 |
| --- | --- | --- | --- |
| 甲（挑战者） | transformer_block 主函数 | 偏难 | 把乙丙写好的模块拼成完整 Post-LN Block（H1=LN(X+Attn(X))，H2=LN(H1+FFN(H1))），处理残差形状、参数字典传递、整体调试——错误往往在这里暴露，需要最强的排错能力 |
| 乙 | softmax + layer_norm | 较轻松 | 两个基础函数：softmax 先减每行最大值防溢出；layer_norm 沿最后一维、eps=1e-5、带可学习 γ/β。模板资料全，写完用小数组手算对答案 |
| 丙 | self_attention + feed_forward | 较轻松 | 按公式照抄的矩阵乘组合：Q=XWq 等三连、S=QK^T/√dk、PV、乘 W_O；FFN 两层带 ReLU。函数签名题目已给出，填空即可 |

协作方式：三人同一 Git 仓库，乙丙先交基础函数，甲拼装；权重统一用 default_rng(0) 生成。

阶段验收：X∈R^3×4 完整前向跑通，打印各步骤维度、注意力矩阵与最终结果；甲负责联调，乙丙各出 2 个手算小例子验证自己的函数。

## 三、第三阶段·方向 A：KV Cache

| 成员 | 任务 | 难度 | 具体内容 |
| --- | --- | --- | --- |
| 甲（挑战者） | 缓存版逐 Token 解码器 | 偏难 | 实现带 KV Cache 的因果自注意力解码：已有 K/V 存表、新 Token 只算自己的 k/v 再拼接、因果掩码（下三角）保证不看未来；索引边界（第 t 步缓存多长）极易差一错误，是最硬的骨头 |
| 乙 | 基线解码器 + 一致性比对 | 较轻松 | 实现“每步重算全部历史 K/V”的朴素基线；逐 Token 循环套用已有 self_attention 即可；然后与甲的缓存版逐元素比对，输出最大误差 |
| 丙 | 重复计算统计 + 计时 + 容量估算 | 较轻松 | ① 数一数两种方法各自重复计算的 K/V 元素个数，做成对比表；② X∈R^128×64、64×64 权重的 QKV 融合计时（Wqkv 只拼一次、先预热、取中位数）；③ 用 2×B×L×T×H_kv×d_h×s_byte 代入具体数字估算缓存容量 |

阶段验收：基线与缓存版最后一个 Token 输出一致（误差 < 1e-6）；重复计算对比表与缓存容量估算各有具体数字；计时报告含预热说明与中位数。

## 四、第三阶段·方向 B：INT8 权重量化

| 成员 | 任务 | 难度 | 具体内容 |
| --- | --- | --- | --- |
| 成员甲（挑战者） | 量化版 Block + 误差排查 | 偏难 | 用乙产出的量化管线替换权重重跑整个 Block；要排查误差来源（round 取整 vs softmax 非线性放大），并解释“存储变小 ≠ NumPy 变快” |
| 成员乙 | 量化管线 + 量化基础实验 | 较轻松 | 实现 quanti_func.py 的三个函数（sym_quanti_int8 / dequanti_int8 / inaccuracy）；对 W_Q/W_K/W_V 分别求 s = max\|W\| / 127、W_int8 = round(W / s)、Ŵ = s·W_int8；浮点 vs INT8 存储字节对比（3 个 4×4 权重：float32 192 B → int8 48 B）；量化前后 Q/K/V 逐元素误差、最大绝对误差统计 |
| 成员丙 | 输出对比 + 计时 + QKV 融合 | 较轻松 | ① 量化前后最终 Block 输出并排对比表；② 计时对比（预热+中位数）；③ 顺带完成题目第 4 节 QKV 融合验证（两种实现输出一致 + 调用次数说明）；④ 说明 BLAS 浮点优化 + 反量化本身还要浮点乘，因此存储变小不等于速度变快 |

阶段验收：误差数据表完整（存储、Q/K/V 最大误差、最终输出变化各有具体数字）；能说清存储变小与速度不变快的关系；融合前后输出一致。

## 五、代码目录与文件说明

目录结构（Python 源码平铺在仓库根目录，模块之间通过 `import` 直接互相调用，全部脚本都在项目根目录下运行）：

```
transformer_team/
├── README.md                          # 本文件：分工方案 + 代码目录说明
├── params.py                          # 全组统一的输入与权重生成（default_rng(0)）
├── softmax_layernorm.py               # 基础算子：softmax / layernorm
├── self_attention.py                  # 单头自注意力（QKV 分开算）
├── self_attention_fused.py            # QKV 融合版自注意力 + 共用的注意力函数
├── feed_forward.py                    # 前馈网络 FFN
├── transformer_block.py               # Block ①：无优化版
├── transformer_block_fused.py         # Block ②：QKV 融合版
├── transformer_block_int.py           # Block ③：单量化版
├── transformer_block_int_fused.py     # Block ④：总融合版 + 部署态参数
├── quanti_func.py                     # 量化 / 反量化 / 误差统计 / 存储账
├── test_compare_all.py                # ★ 最终总测试（结论数据都出自这里）
├── main_test.py                       # 第二阶段最小 demo
├── test_int.py                        # 浮点版 vs 量化版最小对照
├── test_mine.py                       # 注意力手算小例子
└── bench_qkv.py                       # QKV 融合单独计时
```

四个 Block 之间只差一个环节，结构上可以对着看：

| 从 → 到 | 改动的那一步 |
| --- | --- |
| 无优化版 → QKV融合版 | 把分开的三次 QKV 矩阵乘换成拼一次 Wqkv 后一次矩阵乘 |
| 无优化版 → 单量化版 | 把权重换成反量化回来的 Ŵ，子层调用完全不变 |
| 单量化版 → 总融合版 | 在量化基础上再加 QKV 融合 |
| 总融合版 → 总融合·部署态 | 反量化与 Wqkv 拼接只在构造时做一次，测稳态开销 |

各文件作用：

| 文件 | 负责人 | 作用 |
| --- | --- | --- |
| `params.py` | — | 全组统一参数入口。`make_params` 固定生成顺序 X → Wq → Wk → Wv → Wo → W1 → W2，保证三人跑出同一组数字 |
| `softmax_layernorm.py` | 乙 | 两个基础算子。`softmax` 先减每行最大值防溢出、沿 axis=-1 归一化；`layernorm` 沿最后一维计算，eps=1e-5，γ/β 可选（默认 None 等价于 γ=1、β=0） |
| `self_attention.py` | 丙 | 单头自注意力。Q=XWq、K=XWk、V=XWv → S=QK^T/√dk → A=softmax(S) → (A·V)·Wo，返回 (输出, 注意力矩阵 A) 以便外部检查 |
| `self_attention_fused.py` | 丙 | QKV 融合版自注意力。`self_attention_fused` 每次调用拼一次 Wqkv；`self_attention_from_wqkv` 供两个融合 Block 共用，注意力逻辑只此一处 |
| `feed_forward.py` | 丙 | 前馈网络。ReLU(xW1+b1)W2+b2，逐位置独立，中间维度 d_model → d_ff → d_model |
| `transformer_block.py` | 甲 | 无优化版 Block。H1=LN(X+Attn(X))、H2=LN(H1+FFN(H1))，子层全部调用现成函数 |
| `transformer_block_fused.py` | — | QKV 融合版 Block。只把自注意力换成融合实现，其余与无优化版完全一致，便于单独观察融合本身的效果 |
| `transformer_block_int.py` | 甲 | 单量化版 Block。先 `dequant_params` 反量化权重，之后调用与无优化版完全相同的子层 |
| `transformer_block_int_fused.py` | — | 总融合版 Block（每次调用都反量化并拼 Wqkv），并提供 `build_prepared` / `quantized_block_prepared` 表示“反量化与拼接只做一次”的部署态 |
| `quanti_func.py` | 乙 | 量化域的公共函数，全项目只此一处：`sym_quanti_int8`（求 s 并量化为 int8）、`dequanti_int8`（反量化）、`build_pq`（生成量化参数字典）、`dequant_params`（还原成浮点参数字典）、`inaccuracy`（最大绝对误差 + 相对误差）、`storage_bytes`（存储账，含刻度开销） |
| `test_compare_all.py` | — | ★最终总测试。题目第 3 节 5 项自动检查，加五种实现的绝对/相对误差、输出、存储账与计时；大模型部分只测开销与加速比 |
| `main_test.py` | 甲 | 第二阶段最小 demo。打印输入/注意力输出维度、注意力矩阵、行和与最终输出 H2 |
| `test_int.py` | 甲 | 方向B 最小对照。浮点版与量化版 Block 输出并排，并给出最大绝对误差与相对误差 |
| `test_mine.py` | 丙 | 手算小例子。用 2×2 单位权重矩阵核对注意力矩阵与输出，并验证换掉 Wv 不改变 A |
| `bench_qkv.py` | 丙 | QKV 融合单独计时。X∈R^128×64、三个 64×64 权重，Wqkv 只拼一次，预热 20 次后各重复 300 次，输出最小值与中位数 |

> 负责人一栏中标 `—` 的是本次新增的文件，请组内确认后补上。

运行方式（都在项目根目录执行）：

```bash
python test_compare_all.py   # ★最终总测试：正确性检查 + 误差/存储/计时（结论数据出自这里）
python main_test.py          # 第二阶段：最小前向 demo
python test_int.py           # 方向B：浮点版与量化版输出对照
python bench_qkv.py          # QKV 融合单独计时
python test_mine.py          # 注意力手算小例子
```

## 六、成果汇总与汇报分工

结果记录（题目第 6 节）：完整代码甲讲解、张量维度与中间结果乙截取、正确性检查与对比数据表丙整理。

汇报顺序建议：三人各自演示自己的实操成果，甲压轴讲优化难点与解决过程。

可复现性：全组固定 default_rng(0)，记录 Python/NumPy 版本，同一份代码库出结果。
