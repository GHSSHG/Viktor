# Viktor

Nanopore basecaller：Dorado SUP v5.2.0 编码器（直接加载官方权重）+ 小型 decoder-only 语言模型（Qwen3 风格），信号帧作为前缀，自回归输出碱基。现状、实测数据和难点见 [analysis.md](analysis.md)。

## 结构

```text
pod5 int16 信号
  → Dorado 预处理与 VCS 切块            viktor/chunking.py
  → Dorado SUP 编码器（koi int8 kernel） viktor/encoder.py, koi.py, csrc/koi_tx.cpp
      只保留真实信号的帧（去掉补齐部分）   DoradoEncoder.encode_chunks
  → 逐帧 Projector                       viktor/model.py
  → Decoder-only（8 层 d512，GQA 8/2）    viktor/decoder.py
      推理：变长 prefill + CUDA Graph 解码 viktor/engine.py, fast_decoder.py,
                                          triton_ops.py, decode_attention.py
  → 单碱基（词表 7）或 4-mer（词表 343）   viktor/tokenizer.py
```

| 文件 | 内容 |
|---|---|
| `viktor/chunking.py` | Dorado ScalerNode 预处理（int16 → 标准化 fp16、开孔电流校正、裁掉开头 10 个点）；VCS 切块（12288 点、重叠 600、末块重复补齐到 768 的倍数） |
| `viktor/encoder.py` | Dorado SUP 编码器的 PyTorch 实现（训练、参考）与权重加载；`use_koi()` 切到 koi；`encode_chunks()` 给 decoder 用，去掉补齐帧 |
| `viktor/koi.py`, `csrc/koi_tx.cpp` | koi VCS 卷积与 Transformer、koi 束搜索的 C++ 扩展（首次运行时 JIT 编译） |
| `viktor/decoder.py` | decoder 参考实现（训练用）和静态 KV cache |
| `viktor/model.py` | `Basecaller` = 编码器 + Projector + decoder；加权 teacher-forcing loss（逐碱基权重，标签长度补齐到固定档位以便 compile）；`set_encoder_trainable()`（koi 编码器不可训练） |
| `viktor/engine.py` | `DecodeEngine`：贪心解码，语法约束（尾部 token 后强制 EOS），每步一个 CUDA Graph |
| `viktor/fast_decoder.py` | 推理前向：融合 kernel + flash-attn；prefill 最后一层只算 BOS |
| `viktor/triton_ops.py` | 融合 kernel：add+RMSNorm、QK-norm+RoPE+写 KV、SwiGLU |
| `viktor/decode_attention.py` | 单 query 的 GQA 解码 attention（sm80 默认使用） |
| `viktor/data.py` | 训练窗口：在线读 pod5 信号（与推理相同的预处理）、从参考取标签、逐碱基权重、`WindowDataset` |
| `scripts/make_chunks.py` | pod5 → chunk 集 `.npz`（signal、lens、valid、new），测速用 |
| `scripts/make_windows.py` | 训练数据：从 BasecallerData 第 6 阶段的强制比对结果切出 Dorado 式 12288 点窗口的索引（`build`），并用 Dorado CRF 抽查标签（`check`）；外部测试集（`external`，如 EF22 细菌，用 SUPEncoder 的强制比对） |
| `scripts/evaluate.py` | 检查点 + Dorado 基线在固定窗口集上的自由解码评估（口径同训练中的 heldout 评估） |
| `scripts/train.py` | 第一阶段 SFT：koi 编码器冻结，单碱基 decoder 从头训练 1 个 epoch；DDP、torch.compile、WSD、wandb、温度保护 |
| `scripts/bench.py` | 端到端测速：编码器、prefill、解码各阶段与总吞吐 |
| `scripts/basecall_crf.py`, `score_paf.py` | 对照路径：koi 编码器 + Dorado CRF 头 + koi 束搜索 + Dorado 拼接，用 minimap2 PAF 打分。用于证明编码器和预处理与 Dorado 一致 |
| `tests/test_viktor.py` | CPU 单元测试 |
| `tests/gpu_check.py` | GPU 数值检查：kernel、koi VCS、融合 decoder 与参考实现、CUDA Graph |

## 环境

服务器 `~/Work/Viktor`，`source ~/myenv/bin/activate`（torch 2.9 + cu128、flash-attn 2.8、triton 3.5）。包没有 pip install，运行时加 `PYTHONPATH=$PWD`。

- koi：ONT CDN 的 `libkoi-0.7.5-Linux-x86_64-cuda-12.8`，解压到 `third_party/`（或设 `VIKTOR_KOI_DIR`）。
- 模型：`~/tools/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0`。
- `results/`、`third_party/` 不进 git。

## 常用命令

```bash
MODEL=~/tools/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0

# chunk 集（--full-only 只保留满长 chunk）
python scripts/make_chunks.py --pod5 READS.pod5 --model-dir $MODEL --n 2048 --out results/chunks_vcs_2048.npz

# 测速
CUDA_VISIBLE_DEVICES=0 python -m scripts.bench --model-dir $MODEL --tokenizer base --chunks results/chunks_vcs_2048.npz

# 训练数据（CPU，约 4–5 分钟）与标签抽查
python -m scripts.make_windows build --out ~/Work/BasecallerData/viktor_win12288 --workers 16
CUDA_VISIBLE_DEVICES=0 python -m scripts.make_windows check --out ~/Work/BasecallerData/viktor_win12288 --model-dir $MODEL

# 外部测试集（EF22 细菌，1024 个窗口）与评估
CUDA_VISIBLE_DEVICES=3 python -m scripts.make_windows external --pod5 /data_nvme/bacteria_ef22/pod5/EF22_main.pod5 \
    --paf /data_nvme/bacteria_ef22/basecalls/ef22_main.paf --ref /data_nvme/bacteria_ef22/reference/ef22_ref.fasta \
    --contigs contig_1 --out ~/Work/BasecallerData/viktor_ef22_n1024.parquet --n 1024
CUDA_VISIBLE_DEVICES=3 python -m scripts.evaluate --model-dir $MODEL --windows ~/Work/BasecallerData/viktor_ef22_n1024.parquet \
    --ref /data_nvme/bacteria_ef22/reference/ef22_ref.fasta --out runs/eval_ef22.json runs/sft2/snap_0030000.pt

# 训练（4 卡，1 个 epoch）
torchrun --nproc-per-node 4 -m scripts.train --model-dir $MODEL --windows ~/Work/BasecallerData/viktor_win12288 --out runs/sft2

# 检查
python -m pytest tests/test_viktor.py
CUDA_VISIBLE_DEVICES=0 python tests/gpu_check.py --model-dir $MODEL --chunks results/chunks_vcs_2048.npz

# Dorado 对照（CRF 路径）
CUDA_VISIBLE_DEVICES=0 python scripts/basecall_crf.py --help
```
