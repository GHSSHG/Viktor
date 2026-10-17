# 实施计划

日期：2026-10-17。依据：[analysis.md](analysis.md) 和后续讨论。

## 1. 本轮范围

本轮只做两件事：搭好本地仓库，准备好 A100 测速工具。不训练，也不测准确率。

固定的决策：

| 项目 | 决定 |
|---|---|
| Encoder | Dorado SUP `dna_r10.4.1_e8.2_400bps_sup@v5.2.0`，**直接加载 Dorado 发布的训练权重**，不做从零训练 |
| 输入 | **不压缩**。每个 12288 采样点的 chunk 产生 1024 帧 × 512 维，每帧对应一个前缀 token |
| Projector | 逐帧 MLP：Linear(512→512) → GELU → Linear(512→512) |
| Decoder | Qwen3 风格，8 层，d=512，8 个 Q 头 / 2 个 KV 头（GQA），head_dim 64，SwiGLU 1536，QK-norm，RoPE。约 24M 参数 |
| 输出 | 两种都实现：单碱基（词表 7）、4-mer（词表 343：256 个 4-mer + 84 个尾部 1/2/3-mer + 3 个特殊 token） |
| 解码 | 贪心解码。不做 CRF 草稿投机解码，不做 beam search |
| 冻结 | 由 `Basecaller.set_encoder_trainable()` 控制。训练分两个阶段（先冻结、再解冻），具体方案到训练阶段再定 |

本轮不做：输入帧压缩、chunk 拼接与完整 read 输出、质量值、训练数据管线、RL。

## 2. 仓库结构

```
viktor/
  analysis.md          前期分析（CodeX）
  plan.md              本文件
  pyproject.toml
  viktor/
    tokenizer.py       KmerTokenizer(k)：k=1 单碱基，k=4 为 4-mer 加尾部 token
    encoder.py         Dorado SUP encoder 的 PyTorch 实现，以及 .tensor 权重加载
    decoder.py         decoder-only 模型、静态 KV cache
    model.py           Basecaller：encoder + projector + decoder，teacher-forcing loss
    engine.py          优化后的推理：批量 prefill，每个解码步一个 CUDA Graph
    pipeline.py        跨批次的 encoder/decoder 双 stream 流水
  scripts/bench.py     A100 测速脚本
  tests/test_viktor.py CPU 单元测试
```

## 3. 已实现的优化

### Encoder（对齐 Dorado 的计算）

- 数值实现与 Dorado 源码逐项对齐（dorado `TxModules.cpp` / `ConvStack.cpp`，commit b86d79e）：
  - 卷积 + SiLU，batchnorm 已折叠进权重
  - RoPE 用前后半分（NeoX 式），theta = 10000
  - 窗口注意力，左 127 / 右 128
  - SwiGLU 为 `silu(后半) × 前半`
  - DeepNorm 残差 + RMSNorm，eps = 1e-5
- 只加载 `conv.*` 和 `transformer_encoder.*`，跳过上采样层和 CRF 头。
- GPU 上如果装了 flash-attn，就调用 `flash_attn_func(window_size=(127, 128))`；否则退回 SDPA + 带状 mask。
- 默认用 fp16，和 Dorado 一致。

### Decoder

- 每个 chunk 的前缀长度相同（1024 帧 + BOS），所以同一批里所有序列每一步都处在同一个位置，只是结束的时间不同。已结束的序列继续跑、输出 PAD，这样每一步的形状都固定，可以被 CUDA Graph 捕获。
- 静态 KV cache，形状为 `[层, batch, 位置, kv_head, head_dim]`，按最大长度预分配。
- 解码时用 `flash_attn_with_kvcache` 原地写入 KV，并且只读到当前长度。
- prefill 按子批次进行（`--prefill-batch`），用来控制显存峰值。
- 整个解码步只录成一个 CUDA Graph，里面包括：词嵌入、8 层、输出头、argmax、语法约束（尾部 token 后强制 EOS，EOS 之后只能是 PAD）、写回输出缓冲区、位置 +1。
- 每 32 步才检查一次是否全部结束，减少同步。
- 可选 `torch.compile`（`--compile`），先编译再录图。
- Q/K/V 合并为一次投影，gate/up 合并为一次投影。

### 流水线

- 两个 CUDA stream。先把第 i+1 批的 encoder 排进 encoder stream，再在高优先级的 decoder stream 上解码第 i 批。encoder 主要吃算力，decode 主要吃显存带宽，两者可以部分重叠。
- 只有一套 KV cache。下一批的 prefill 必须等当前批解码结束，所以显存里同时存在两批 encoder 输出（batch 512 时约 1 GiB）。
- 用 `overlap=False` 跑同样的工作但完全顺序执行，作为对照。

## 4. 本地验证（已完成）

`pytest tests`，在 CPU、torch 2.8 下 7 项全部通过：

1. 两种 tokenizer 的编码/解码往返；尾部 token 只出现在最后。
2. 两种 tokenizer 下，引擎（静态 cache + 单步解码）的贪心输出，与每一步完整重算的参考实现**逐 token 相同**。
3. 流水模式与顺序模式的输出相同。
4. loss 可以反向传播；冻结 encoder 后，encoder 没有梯度。
5. 用 `VIKTOR_DORADO_MODEL` 指定模型目录时，真实 SUP v5.2.0 权重能 strict 加载，输出形状为 1024×512。

CPU 测试覆盖不到：flash-attn 路径、CUDA Graph、多 stream。这些在下一轮上 GPU 验证。

## 5. 下一轮：A100 测速

### 准备

```bash
rsync -a --exclude .git viktor/ bioinformatics:~/viktor/
ssh bioinformatics
source ~/myenv/bin/activate
pip install -e ~/viktor          # 可选：pip install flash-attn
export CUDA_VISIBLE_DEVICES=0    # GPU 2 是同事在用，不要选
```

需要准备 Dorado SUP v5.2.0 的模型目录。服务器上没有的话，从 ONT CDN 下载。

### 步骤

1. **GPU 正确性**：
   - 在 GPU 上跑 `pytest`。
   - 补一个 GPU 检查：flash 路径（bf16）与 SDPA 路径（fp32）的贪心输出一致率，以及 CUDA Graph 开/关的输出必须相同。
2. **Encoder 数值核对**：同一段真实信号，对比本仓库 encoder 和现有 Whisper 项目里的 Dorado encoder，输出应基本一致（需要你提供那个项目的路径）。
3. **Dorado 基线**：用 `dorado basecaller sup@v5.2.0` 跑一小份 pod5（约 1 万个 chunk），记录日志里的 `Samples/s`，除以 12.5 得到碱基/秒。
4. **本仓库测速**：

   ```bash
   python scripts/bench.py --model-dir MODEL --tokenizer base --batch 512 --json results/bench.jsonl
   python scripts/bench.py --model-dir MODEL --tokenizer 4mer --batch 512 --json results/bench.jsonl
   ```

   测试矩阵：
   - tokenizer：base / 4mer
   - batch：128 / 256 / 512 / 1024
   - CUDA Graph 开/关（`--no-graph`）
   - `--compile` 开/关
   - 有没有 flash-attn

   每次运行输出：encoder、prefill、decode、顺序、流水五项的每 chunk 毫秒数，碱基/秒，以及峰值显存（allocated 和 reserved）。

### 汇总表

| 方案 | encoder ms/chunk | prefill | decode | 顺序 bp/s | 流水 bp/s | 相对 Dorado | 峰值显存 |
|---|---|---|---|---|---|---|---|
| Dorado SUP | — | — | — | | — | 100% | |
| base, B=512 | | | | | | | |
| 4mer, B=512 | | | | | | | |

### 要注意的地方

- decoder 权重是随机的，不会正常结束，所以解码步数用 `--steps` 固定：base 984 步，4mer 247 步。真实输出长度的影响要等训练出模型后再测。
- 吞吐按每个 chunk 新增 935 个碱基计算，即 (12288 − 600) / 12.5。
- 本仓库的 encoder 是 PyTorch 实现。Dorado 用的是 koi 专用 kernel，还会裁掉权重低位来提速，所以 encoder 本身可能比 Dorado 慢。要分开看两点：“encoder 实现的差距”和“decoder 带来的额外开销”。

## 6. 之后的阶段（暂不展开）

1. **训练接入**：
   - 按现有数据格式写 dataloader。
   - 两种 tokenizer 各自预训练 decoder，或者在现有 decoder 上适配。
   - 两阶段 SFT：先冻结 encoder，再解冻 encoder 微调。
   - 评估项按 analysis.md 第 8 节。
2. **推理补全**：chunk 拼接、完整 read 输出、按位置求边缘概率的质量值。
3. **RL（GSPO）**：在 SFT 之后进行。
4. **根据实测结果再决定**：是否需要分页 KV、提前做下一批的 prefill、MQA / FP8 KV、输入帧压缩。
