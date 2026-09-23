# 题目二 · Transformer 小组作业（方向 B：INT8 权重量化）
## 环境
- Python 3.14.6
- NumPy 2.5.3
- 随机种子统一 `np.random.default_rng(0)`
## 分工
| 成员 | 负责文件 | 内容 |
|---|---|---|
| 甲 | `block_jia.py` | `transformer_block` 主函数 + INT8 量化管线 |
| 乙 | `ops_yi.py` | `softmax`、`layer_norm` |
| 丙 | `ops_bing.py` | `self_attention`、`feed_forward`、对比与计时 |
## 怎么跑
python main_demo.py
## 规矩
1. 一人一个文件，不要编辑别人的文件。需要改接口先在群里说一声。
2. 每天开工前先 `git pull`。
3. 提交信息写清谁做了什么，例如 `丙: 完成 self_attention`。
