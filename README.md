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

代码按“通用 → 浮点版 → 单量化版 → 融合版 → 总融合版”五个版本分目录存放，每个目录都是一个标准 Python 包（各含一个 `__init__.py`），同一版本的程序与它的测试放在一起：

```
transformer_team/
├── README.md                              # 本文件：分工方案 + 目录说明 + 调用流程
├── CORRECTIONS.md                         # 重大修正记录（现象 / 原因 / 方法 / 代价 / 数据）
├── common/                                # ① 通用：与版本无关的基础件
│   ├── __init__.py
│   ├── params.py                          # 输入与权重的统一生成入口
│   ├── softmax_layernorm.py               # 基础算子 softmax / layernorm
│   ├── feed_forward.py                    # 前馈网络 FFN
│   └── quanti_func.py                     # 量化 / 反量化 / 误差 / 存储统计
├── float_version/                         # ② 浮点版（无优化）
│   ├── __init__.py
│   ├── self_attention.py                  # 单头自注意力（QKV 分开算）
│   ├── transformer_block.py               # 无优化版 Block
│   ├── main_test.py                       # 最小前向 demo
│   └── test_mine.py                       # 注意力手算小例子
├── quant_version/                         # ③ 单量化版
│   ├── __init__.py
│   ├── transformer_block_int.py           # 单量化版 Block
│   └── test_int.py                        # 浮点版 vs 量化版对照
├── fused_version/                         # ④ QKV 融合版
│   ├── __init__.py
│   ├── self_attention_fused.py            # 融合版自注意力（含共用注意力函数）
│   ├── transformer_block_fused.py         # QKV 融合版 Block
│   └── bench_qkv.py                       # QKV 融合单独计时
└── total_version/                         # ⑤ 总融合版（量化 + 融合）
    ├── __init__.py
    ├── transformer_block_int_fused.py     # 总融合版 Block + 部署态参数
    └── test_compare_all.py                # ★最终总测试
```

**导入约定（重要）**：模块之间一律使用点号绝对导入，例如

```python
from common.quanti_func import build_pq          # 通用包
from float_version.self_attention import self_attention
from total_version.transformer_block_int_fused import build_prepared
```

每个入口脚本（`main_test.py` / `test_mine.py` / `test_int.py` / `test_compare_all.py`）开头都有三行引导，把项目根目录加入模块搜索路径：

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
```

这样既能让 IDE 静态解析出每个 import（不再报“未解析的引用”），也能保证运行时直接跑通。

**IDE 设置**：把 `transformer_team` 这个目录本身作为项目根目录打开（PyCharm / VS Code / Visual Studio 都一样），`common`、`float_version` 等包才会被识别为可导入的包。若之前打开的是别的位置，重新打开一次项目即可。

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
| `common/params.py` | 甲 | 全组统一参数入口。`make_params` 固定生成顺序 X → Wq → Wk → Wv → Wo → W1 → W2，保证三人跑出同一组数字 |
| `common/softmax_layernorm.py` | 乙 | 两个基础算子。`softmax` 先减每行最大值防溢出、沿 axis=-1 归一化；`layernorm` 沿最后一维计算，eps=1e-5，γ/β 可选（默认 None 等价于 γ=1、β=0） |
| `common/feed_forward.py` | 丙 | 前馈网络。ReLU(xW1+b1)W2+b2，逐位置独立，中间维度 d_model → d_ff → d_model |
| `common/quanti_func.py` | 乙 | 量化域公共函数，全项目只此一处：`sym_quanti_int8`（求 s 并量化为 int8）、`dequanti_int8`（反量化）、`build_pq`（生成量化参数字典）、`dequant_params`（还原成浮点参数字典）、`inaccuracy`（最大绝对误差 + 相对误差）、`storage_bytes`（存储账，含刻度开销） |
| `float_version/self_attention.py` | 丙 | 单头自注意力。Q=XWq、K=XWk、V=XWv → S=QK^T/√dk → A=softmax(S) → (A·V)·Wo，返回 (输出, 注意力矩阵 A) 以便外部检查；量化版与融合版都复用它 |
| `float_version/transformer_block.py` | 甲 | 无优化版 Block。H1=LN(X+Attn(X))、H2=LN(H1+FFN(H1))，子层全部调用现成函数 |
| `float_version/main_test.py` | 甲 | 浮点版最小 demo。打印输入维度、注意力矩阵、行和与最终输出 H2 |
| `float_version/test_mine.py` | 丙 | 手算小例子。用 2×2 单位权重矩阵核对注意力矩阵与输出，并验证换掉 Wv 不改变 A |
| `quant_version/transformer_block_int.py` | 甲 | 单量化版 Block。先 `dequant_params` 反量化权重，之后调用与无优化版完全相同的子层，便于逐元素比较误差 |
| `quant_version/test_int.py` | 甲 | 单量化版对照。浮点版与量化版 Block 输出并排，并给出最大绝对误差与相对误差 |
| `fused_version/self_attention_fused.py` | 丙 | 融合版自注意力。`self_attention_fused` 每次调用拼一次 Wqkv；`self_attention_from_wqkv` 供两个融合 Block 共用，注意力逻辑只此一处 |
| `fused_version/transformer_block_fused.py` | 甲 | QKV 融合版 Block。只把自注意力换成融合实现，其余与无优化版完全一致，便于单独观察融合本身的效果 |
| `fused_version/bench_qkv.py` | 丙 | QKV 融合单独计时。X∈R^128×64、三个 64×64 权重，Wqkv 只拼一次，预热 20 次后各重复 300 次，输出最小值与中位数 |
| `total_version/transformer_block_int_fused.py` | 甲 | 总融合版 Block（每次调用都反量化并拼 Wqkv），并提供 `build_prepared` / `quantized_block_prepared` 表示“反量化与拼接只做一次”的部署态 |
| `total_version/test_compare_all.py` | 甲 | ★最终总测试。题目第 3 节 5 项自动检查，加五种实现的绝对/相对误差、输出、存储账与计时；大模型部分只测开销与加速比。计时采用 BLAS 单线程 + 交替采样 + 预热轮 + 报最小值，方法学与实测数据见 `CORRECTIONS.md` 修正 001 |
| `CORRECTIONS.md` | 全组 | 重大修正记录。逐条记录影响结论的修正：现象、原因、修正方法、代价、修正前后实测数据 |

> 计时结果只用于比较各实现的相对开销，其稳定性度量与可信边界见 `CORRECTIONS.md` 修正 001。

运行方式（项目根目录、各版本目录内、项目外均可直接运行）：

```bash
python total_version/test_compare_all.py   # ★最终总测试（结论数据出自这里）
python float_version/main_test.py          # 浮点版最小前向
python float_version/test_mine.py          # 注意力手算例子
python quant_version/test_int.py           # 浮点版 vs 单量化版
python fused_version/bench_qkv.py          # QKV 融合单独计时
```

也可以用模块方式运行（在项目根目录执行，IDE 的“运行”按钮通常就是这种方式）：

```bash
python -m total_version.test_compare_all
```

## 六、调用流程与关系图谱

### 6.1 五个包的依赖方向

箭头表示“被引用”：`A → B` 读作“A 里的模块被 B 引用”。

```
common          →  float_version        （params / 算子被浮点版使用）
common          →  quant_version        （quanti_func 被量化版使用）
common          →  fused_version        （softmax / layernorm / feed_forward 被融合版使用）
common          →  total_version        （quanti_func 被总融合版使用）
float_version   →  quant_version        （quant_version 复用 float_version 的 self_attention）
fused_version   →  total_version        （total_version 复用 fused_version 的 self_attention_from_wqkv）
float_version / quant_version / fused_version  →  total_version   （最终测试要同时调用前四个版本做对比）
                →  各包内的测试脚本
```

一句话概括：**common 是被所有人依赖的地基；float_version 是原始实现；quant_version 在它上面加量化；fused_version 在通用算子上加融合；total_version 把量化和融合合起来，并且只有它同时依赖前四个包。**

### 6.2 一条完整调用链（以“总融合·部署态”为例，从测试一路走到通用函数）

```
total_version/test_compare_all.py                         ← 入口
  │
  ├─► total_version/transformer_block_int_fused.build_prepared(P)
  │      ├─► common/quanti_func.build_pq
  │      │      └─► common/quanti_func.sym_quanti_int8
  │      ├─► common/quanti_func.dequant_params
  │      │      └─► common/quanti_func.dequanti_int8
  │      └─► numpy.concatenate（拼 Wqkv，只做一次）
  │
  ├─► total_version/transformer_block_int_fused.quantized_block_prepared(X, prepared)
  │      ├─► fused_version/self_attention_fused.self_attention_from_wqkv
  │      │      └─► common/softmax_layernorm.softmax
  │      ├─► common/softmax_layernorm.layernorm     （残差后归一化，用了两次）
  │      └─► common/feed_forward.feed_forward
  │
  ├─► common/quanti_func.inaccuracy(基准, 结果)           ← 绝对误差 + 相对误差
  └─► common/quanti_func.storage_bytes(P)               ← int8 存储账
```

### 6.3 五种实现各自的调用链（横向对比）

每一行都是“入口函数 → … → 落到通用函数”的完整路径。

| 实现 | 入口函数 | 调用链 |
| --- | --- | --- |
| 无优化版 | `float_version/transformer_block.transformer_block` | `float_version.transformer_block` → `float_version.self_attention` → `common.softmax`；再 → `common.layernorm` → `common.feed_forward` → `common.layernorm` |
| 单量化版 | `quant_version/transformer_block_int.quantized_transformer_block` | `common.build_pq` → `common.dequant_params` → `quant_version.transformer_block_int` → `float_version.self_attention` → `common.softmax`；再 → `common.layernorm` → `common.feed_forward` → `common.layernorm` |
| QKV融合版 | `fused_version/transformer_block_fused.transformer_block_fused` | `fused_version.transformer_block_fused` → `fused_version.self_attention_fused` → `fused_version.self_attention_from_wqkv` → `common.softmax`；再 → `common.layernorm` → `common.feed_forward` → `common.layernorm` |
| 总融合版 | `total_version/transformer_block_int_fused.quantized_transformer_block_fused` | `common.build_pq` → `common.dequant_params` → `np.concatenate`（拼 Wqkv）→ `total_version.…_fused` → `fused_version.self_attention_from_wqkv` → `common.softmax`；再 → `common.layernorm` → `common.feed_forward` → `common.layernorm` |
| 总融合·部署态 | `total_version/transformer_block_int_fused.quantized_block_prepared` | `common.build_pq` → `common.dequant_params` → `total_version.build_prepared`（反量化与拼接只做一次）→ `total_version.quantized_block_prepared` → `fused_version.self_attention_from_wqkv` → `common.softmax`；再 → `common.layernorm` → `common.feed_forward` → `common.layernorm` |

### 6.4 五个测试脚本各自覆盖的范围

| 脚本 | 所在包 | 入口函数 | 覆盖范围 |
| --- | --- | --- | --- |
| `main_test.py` | float_version | `transformer_block` | 最小前向：维度、注意力矩阵、行和、H2 |
| `test_mine.py` | float_version | `self_attention` | 手算例子核对注意力矩阵（2×2 单位权重） |
| `test_int.py` | quant_version | `transformer_block_int` | 浮点版与单量化版输出并排 + 绝对/相对误差 |
| `bench_qkv.py` | fused_version | 内部 `separate` / `fused` | QKV 融合单独计时（128×64，预热 + 中位数） |
| `test_compare_all.py` | total_version | `build_versions` | 五项自动检查 + 五种实现的误差、输出、存储账、计时 |

## 七、成果汇总与汇报分工

结果记录（题目第 6 节）：完整代码甲讲解、张量维度与中间结果乙截取、正确性检查与对比数据表丙整理。

汇报顺序建议：三人各自演示自己的实操成果，甲压轴讲优化难点与解决过程。

可复现性：全组固定 default_rng(0)，记录 Python/NumPy 版本，同一份代码库出结果。
