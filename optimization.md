# 当前架构的推理优化：2026-10-17 实测

本轮保持 encoder、8 层 d512 decoder、GQA 8/2、head_dim 64、tokenizer、贪心解码和 bf16 KV 不变。没有量化、训练、帧压缩或跨层跳算。优化仅删除无人使用的计算、修正 attention 内核分派，并增加专用 decode kernel。

## 测试条件与结果

- 服务器 `~/Work/Viktor`，四张 NVIDIA DRIVE-PG199-PROD（sm80，32 GiB），每张同时最多一个任务；任务外层设置 120 或 180 秒 timeout。
- Python 环境 `~/myenv`：torch 2.9.0+cu128、Triton 3.5.0、FlashAttention 2.8.3。
- 每项测试使用已有数据集的前 **256 个 chunk，B=256，只有一个 batch**。同一个 batch 预热后重复计时三次，报告中位数。
- 每组新旧实现同卡、同数据、同随机权重，固定随机种子 123；4-mer 输出 247 个 token（prefill 产生首个，随后 246 步），单碱基输出 985 个 token。
- decode 直接单独计时，不再用两次计时相减。总耗时包含 encoder、projector、prefill、decode 和输出 token 传回 CPU；输入信号预先在 GPU，未计入文件读取或预处理。
- 功能实现测速 commit `e38d2b1`；`6cfc531` 只增加参数断言、文档和数值检查。

| 配置 | 原 decode ms/chunk | 新 decode | 原总耗时 ms/chunk | 新总耗时 | 原吞吐 bp/s | 新吞吐 bp/s | 吞吐提升 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4-mer，满长 | 1.8998 | 1.4098 | 3.4792 | 2.9312 | 269,288 | 319,637 | 18.7% |
| 4-mer，混合长度 | 1.8997 | 1.3994 | 3.4438 | 2.8890 | 265,624 | 316,632 | 19.2% |
| 单碱基，满长 | 9.0992 | 6.6669 | 10.6834 | 8.2000 | 87,698 | 114,258 | 30.3% |
| 单碱基，混合长度 | 9.0906 | 6.6257 | 10.6394 | 8.1139 | 85,977 | 112,739 | 31.1% |

原始结果保存在 [benchmarks/2026-10-17](benchmarks/2026-10-17)。这是 B=256 的配对比较，不与旧报告 B=512 的数字直接计算加速比。两种混合长度测试的 256 个 chunk 中有 247 个满长。

## 1. FlashAttention 分派问题已确认

旧版 `flash_attn_with_kvcache` 调用传入新 K/V；上一轮 Triton 融合把 cache 写入提前后，不再传 K/V。FA 2.8.3 在默认分片数为 1、没有 cache_batch_idx 的情况下会选择普通 `flash_fwd_kernel`，而不是 `flash_fwd_splitkv_kernel`。

给它传恒等 `cache_batch_idx`、`num_splits=1` 后，profiler 确认调用了解码内核。只设置 `num_splits=1` 不足以触发同一路径。

B=256、长度 1271、单层 attention 微基准：

| 路径 | 时间 µs |
|---|---:|
| 当前普通 FA kernel | 196.2 |
| 恒等行索引，单 split decode kernel | 135.7 |
| 2 splits | 151.8 |
| 4 splits | 153.8 |
| 由 FA 追加 K/V | 149.4 |

实际 kernel 名称与数值误差见 [attention_1271.jsonl](benchmarks/2026-10-17/attention_1271.jsonl)。自动使用单 split 并不代表自动选择了解码内核，这两个概念需区分。

## 2. 专用 Triton decode attention

实现位于 `viktor/decode_attention.py`：每个 CTA 处理一个序列的一个 KV 头，复用 K/V tile 服务四个 Q 头。采用 64-token 分块、4 warps、FP32 在线 softmax 和累积，保留 bf16 KV 存储。直接读取现有 cache，不转置整份 cache，不产生完整 attention score 矩阵。

测试过分块 64/128/256、1/2/4 splits、2/4/8 warps。在当前 B=256 工作负载中，继续切分 KV 没有收益，采用单 split。实现保留可选分片和 log-sum-exp 合并供复现实验使用。

`attention_backend="auto"` 对 sm80、bf16、8/2 头、head_dim=64 使用 `triton64`；其他配置回退到 FA split 路径。可以显式指定 `flash`、`split` 或 `triton64` 比较。

在完整模型中，4-mer 满长 decode 从 FA split 的 1.4861 降至 Triton 的 1.4121 ms/chunk，额外减少约 5%。因此主要收益来自修正分派，专用 kernel 提供了进一步的小幅改善；不存在共享 KV 读取带来额外四倍加速的情况。

## 3. Prefill 去除无用计算

只有最后一层的 BOS 输出会被用于预测，信号位置的最后一层输出没有消费者。所有位置仍做 QKV 投影及 QK-norm/RoPE，完整保存所有层 K/V；最后一层 attention、O 投影、FFN、最终 norm 只计算 BOS。

这不是删层：生成 token 仍经过八层，访问所有原有信号 K/V。BOS 可以读取整条前缀，故最后一层使用单 query 的非因果 attention，与原来最后位置的因果 mask 在数学上相同。

另外，根据 CPU 已知长度构造并缓存 packed prefill 元数据，去掉 GPU 标量回读。每个 prefill 子批次只保存一个长度布局，长度变化时替换，避免缓存无限增长。

4-mer 满长 prefill 从约 0.5924 降至 0.5347 ms/chunk，减少约 9.7%。这包含裁剪和元数据复用的合计收益，未单独归因。单 batch 重复计时使用已缓存布局；长度变化的首次构造成本不包含在这个热态数字中。

逐项实验见 [pipeline4.jsonl](benchmarks/2026-10-17/pipeline4.jsonl)：

| 实现 | 总耗时 ms/chunk |
|---|---:|
| 原 FA + 完整 prefill | 3.4792 |
| FA split + 完整 prefill | 3.0736 |
| FA split + 裁剪 prefill | 3.0053 |
| Triton64 + 裁剪 prefill | 2.9336 |

## 正确性与限制

- Attention 微基准：固定长度、混合长度、1/64/65 边界；与 FP32 SDPA 和原 FA 比较；Graph/eager 完全一致。测过的最大绝对 attention 误差约 0.0024。
- 256 序列、完整八层检查：prefill 前缀 K/V **逐位一致**，包括变更长度布局后的有效 cache；prefill logits 与原实现最大绝对差为 0.0078125（4-mer）、0.00024414（单碱基）。
- 强制相同的非终止 token 输入检查 12 个解码步，避免随机模型提前 EOS 令 PAD 一致率失去意义。logits 最大绝对差为 0.03125（4-mer）、0.015625（单碱基），通过 `atol=0.025, rtol=0.025` 检查。这是浮点归约与 bf16 舍入差异，不是旧新实现逐位一致的承诺。
- 优化版完整 247/985-token 解码：CUDA Graph 与 eager 输出 **逐 token 相同**，cache 有限值检查通过。
- 现有 `tests/gpu_check.py --n 256` 全部通过；其 24 步随机权重 greedy 输出与 PyTorch 后端一致率为 96.4%（4-mer）和 100%（单碱基）。
- CPU 测试 `tests/test_viktor.py`：7 passed，1 skipped（未为 CPU 加载完整 Dorado 权重的可选测试）。
- 现有 `scripts/bench.py` 默认也改为 256 chunk / B=256，标准入口独立复核到混合长度 4-mer 317,656 bp/s。单 batch 无法衡量跨 batch 流水收益。
- 没有测训练后模型的输出长度、准确率或大 batch 的额外收益，本轮只验证当前推理计算成本与数值一致性。

## 复现

在服务器仓库根目录运行，每个命令绑定一张空闲 GPU：

```bash
source ~/myenv/bin/activate
CUDA_VISIBLE_DEVICES=0 timeout -k 5s 180s python -m scripts.bench_decode \
  --model-dir /home/superbig/tools/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
  --chunks results/chunks_full_2048.npz --tokenizer 4mer \
  --variants flash:full,split:full,auto:last --reps 3

CUDA_VISIBLE_DEVICES=1 timeout -k 5s 120s python -m scripts.bench_attention \
  --length 1271 --triton --profile

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 timeout -k 5s 180s python \
  tests/check_decode_optimization.py --tokenizer 4mer --backend auto
```

全部实验日志保留在服务器 `results/optimization/`；关键结果和正确性记录已提交到 `benchmarks/2026-10-17/`。此前的 `analysis.md` 本地未提交修改保持原样，本轮没有覆盖。
