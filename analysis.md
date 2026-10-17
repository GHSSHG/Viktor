# 现状与难点分析

更新：2026-10-18。除注明“估算”外，数字均为服务器实测。GPU 为 NVIDIA DRIVE PG199（A100 架构，sm80，**96 个 SM**，32 GB，理论带宽约 **1.44 TB/s**）。

## 1. 结论摘要

- **编码器已经做到 Dorado 的水平。** 用 koi VCS kernel 跑编码器只需 0.96 ms/chunk；配合 koi 束搜索，在 100 条 HG002 read 上和 Dorado 2.1.2 的编辑距离为 0.059%。
- **当前推理速度（B=256，随机权重）：**
  - 4-mer：2.88 ms/chunk，约 **32 万 bp/s**，约为 Dorado（约 69 万）的 46%；
  - 单碱基：8.2 ms/chunk，约 11.4 万 bp/s（B=512 时 7.5 ms，约 12.5 万）。
- **现有架构下，工程优化基本到头了。**
  - 解码 attention 已达到约 80% 的带宽；
  - FlashAttention、cuDNN、FlashInfer 都不比现在的 kernel 快；
  - 融合逐元素操作只能再省 2–3%；
  - MPS 切分 SM 做流水，反而慢了 20%。
- **剩下的提速空间来自改结构**，都需要在训练前决定：
  - 减少每步读取的 KV：MLA、MQA、少层加宽、只让部分层读前缀、局部窗口；
  - 选择 tokenizer（1-mer、2-mer、4-mer）。

  INT8 KV 不改模型，可以在训练后再决定。
- **还没有训练过。** 准确率、真实输出长度下的吞吐，都还没有任何数据。

## 2. 实测结果

### 2.1 和 Dorado 的一致性（100 条 HG002 read，minimap2 比对 hg002v1.2）

| 方案 | 平均 identity | 中位数 |
|---|---|---|
| 官方 Dorado 2.1.2 SUP v5.2.0 | 98.490% | 99.591% |
| 本仓库 koi VCS 编码器 + koi 束搜索 + mux-change 尾部裁剪 | 98.487% | 99.596% |

和 Dorado 逐 read 比较：编辑距离 0.059%，31/100 条完全相同。

这里的 identity 是“匹配数 / 比对块长度”，**和 SUPEncoder 的 chunk 级全局 identity 不是同一口径**。后面评估训练出来的模型时，应该改用 SUPEncoder 的口径（见 5.3 节）。

### 2.2 推理速度（B=256，2048 个真实 chunk 中的前 256 个，单位 ms/chunk）

| | 编码器 | prefill | 解码 | 合计 | 吞吐 |
|---|---|---|---|---|---|
| 4-mer | 0.96 | 0.53 | 1.41 | 2.88–2.93 | 约 32 万 bp/s |
| 单碱基 | 0.96 | 0.53 | 6.66 | 8.18 | 约 11.4 万 bp/s |

### 2.3 这一阶段做过的优化

| 优化 | 效果 |
|---|---|
| koi VCS 编码器，替代 PyTorch 编码器 | 2.86 → 0.96 ms |
| Triton 融合 kernel（add+RMSNorm、QK-norm+RoPE+写 KV、SwiGLU） | prefill 0.87 → 0.59 ms |
| **修正 FlashAttention 的 kernel 选择**：cache 已经提前写好时，FA2 会选通用 kernel；传入恒等 `cache_batch_idx` 并设 `num_splits=1` 后，才会走 split-KV 解码 kernel | 单层 attention 196 → 136 µs；4-mer 解码 1.90 → 1.49 ms |
| 自写的 GQA 解码 attention（每个 KV 头只读一次，同时服务 4 个 Q 头） | 再快约 5%（1.49 → 1.41 ms） |
| prefill 最后一层只计算 BOS 位置 | 0.59 → 0.53 ms |
| 送进 decoder 之前去掉补齐帧 | 不再把重复信号当成 DNA 来解码 |

### 2.4 做过但没有采用的

| 尝试 | 结果 |
|---|---|
| FlashInfer 0.3.1 / 0.6.10（CUDA core 版和 tensor core 版） | 单层 attention 130–135 µs，比现在的 107 µs 慢；完整解码 1.41 → 1.59 ms |
| cuDNN SDPA / torch flash | 110 µs / 187 µs |
| cuBLASLt、Inductor 自动调优的 GEMM | 不比 cuBLAS 快 |
| KV 切分（2 份、4 份） | 没有收益 |
| 选 token 的逻辑融合成一个 kernel；gate_up 与 SwiGLU 融合 | 合计 2.94 → 2.87 ms/chunk（−2.3%）。B=512 时 GEMM 融合反而变慢 |
| 每 4 步录一个 CUDA Graph | 0.13%，说明 kernel 之间的启动间隔不是瓶颈 |
| 编码器和解码器双 stream 跨批次重叠 | +1.5% |
| MPS 静态切分 SM（编码器 60%，解码器 40%） | **慢了 19.8%**。attention 只用 40% 的 SM 只慢 11%，但 GEMM、prefill 和编码器都随 SM 数下降 |

## 3. 成本模型（4-mer，B=256）

| 部分 | 每个 chunk | 性质 |
|---|---|---|
| 编码器 | 0.96 ms，每个采样点约 78 ns | 算力受限，和 chunk 长度成正比 |
| prefill | 0.53 ms，约 100 TFLOPS | 算力受限 |
| 解码 attention | 约 0.90 ms | 读 1.16 GB 的 KV，约 1.15–1.23 TB/s，约为峰值的 80% |
| 解码中的其他部分 | 约 0.5 ms，即约 0.5 ms/步 ÷ batch | **主要是小 GEMM 本身的执行时间**：M=256 时每个 GEMM 至少 7–11 µs，只有 30–53 TFLOPS，而峰值是 248。kernel 之间的启动间隔不是主因 |

由此得出：
- **每个碱基的 KV 读取量和 chunk 长度成正比。** 改成 4096 碱基的 chunk 后，4-mer 估算约 16.5–17.5 万 bp/s，单碱基约 4.2 万。
- **每步固定开销和层数成正比。** 因为每层都有 4 个 GEMM，每个都有一个最低耗时。batch 加大时，GEMM 的耗时不会成倍增加，所以 B 越大，平摊到每条序列越少。
- **现有架构下，单靠工程优化，4-mer 顺序执行的上限约 35–37 万 bp/s（B=512–1024）。**

## 4. 待决定的结构选项（训练前）

以下是 4-mer、B=256 下的估算，按成本模型推出：

| 选项 | 每 token 每层的 KV | 吞吐估算 | 风险 |
|---|---|---|---|
| 现在：8 × 512，GQA 8/2 | 256 个数（512 B） | 32 万 | — |
| MQA | 128 | 约 38 万 | 注意力的表达能力下降，可能影响对齐 |
| MLA，d_c = 128，d_r = 32 | 160 | 约 37 万 | 位置信息只在共享的 RoPE 分量里；需要自写解码 kernel（FlashMLA 只支持 sm90） |
| 4 × 768，GQA，参数量相近 | 层数减半 | 约 41 万 | 层数太少可能完成不了对齐。不建议低于 4 层 |
| 只让部分层读信号前缀 | 前缀读取量减半 | 约 38 万 | 改架构 |
| 前缀局部窗口（按单调对齐） | 和 chunk 长度无关 | 最高 | 推理时要估计窗口中心，跟丢就会脱轨；必须在训练时就加上 |
| 2-mer tokenizer | — | 约 21 万 | 介于单碱基和 4-mer 之间的折中 |

**训练后再决定的：** INT8 前缀 KV。前缀 KV 只写一次，可以按通道统计缩放系数；输出 token 的 KV 保留 bf16。预计 attention 从约 0.90 降到约 0.50 ms。训练全程用 bf16，INT8 只在推理时使用。

## 5. 训练（第一阶段：冻结编码器，用 koi 前向）

### 5.1 已有

- 模型、按碱基数归一化的 teacher-forcing loss、冻结开关。
- 编码器冻结时，`Basecaller.forward` 会自动走 koi，decoder 看到的就是推理时的特征。

### 5.2 还缺

| 部分 | 内容 |
|---|---|
| 训练数据 | 按 Dorado 的方式切 12288 点、重叠 600 的 chunk，标签是每个 chunk 信号范围内的参考碱基。可以复用 BasecallerData 的 CRF 强制比对结果和 mask |
| dataloader | 在线从 pod5 读取，用 `chunking.dorado_scale` 预处理，按长度分桶。第一版只用满长 chunk |
| 训练脚本 | DDP、优化器、学习率调度、bf16、检查点、断点续训 |
| 评估 | 用 `DecodeEngine` 做自由解码，在 chr20 留出集上算 chunk 级全局 identity 和脱轨率 |
| 检查 | 打开 koi 时如果调用 `set_encoder_trainable(True)`，编码器拿不到梯度但不会报错，需要加检查 |

### 5.3 评估口径（沿用 SUPEncoder）

- 全局比对（edlib NW）：identity = 1 − 编辑距离 / 参考长度；另外统计排除 mask 位置的版本。
- 脱轨率：identity < 0.9 的 chunk 比例。
- 基线：Dorado 在 chr20 留出集 1000 个 chunk 上平均 99.24%，中位 99.66%，脱轨率 1.0%。

### 5.4 显存和 batch（估算，需实测确认）

decoder 每个 token 每层保存给反向传播的中间结果约 28 KB。

| | 4-mer | 单碱基 |
|---|---|---|
| 序列长度 | 1272 | 2010 |
| 每个样本的激活 | 约 285 MB | 约 450 MB |
| 每卡 batch 上限 / 建议值 | 约 80–90 / **64** | 约 50–55 / **32–48** |

4-mer 每卡 64 个样本、4 卡时，每步约 25 万碱基。SUPEncoder 每步约 15.7 万碱基。

## 6. 依据

- 实测记录在服务器 `~/Work/Viktor/results/`：`accuracy_100reads.txt`、`speed_bench_koi.jsonl`、`speed_bench_varlen_triton.jsonl`。
- Dorado 参考实现：[TxModel](https://github.com/nanoporetech/dorado/blob/b86d79e9bd6a83c38584d11b7839990f7ea5303e/dorado/basecall/model/TxModel.cpp)、[TxModules](https://github.com/nanoporetech/dorado/blob/b86d79e9bd6a83c38584d11b7839990f7ea5303e/dorado/nn/TxModules.cpp)；koi 用的是 ONT CDN 上的预编译 libkoi 0.7.5。
- 模型：[SUP v5.2.0 官方模型包](https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r10.4.1_e8.2_400bps_sup@v5.2.0.zip)，服务器上的权重已确认和 CDN 版本逐字节相同。
