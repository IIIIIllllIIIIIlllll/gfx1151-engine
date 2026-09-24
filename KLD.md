# KLD 测试（与 unsloth / llama.cpp 同口径）

目的：用 unsloth 发布 Qwen3.8-Flash-Next 量化表时用的同一套 KLD 统计口径，量化衡量引擎约 4.5 bit
权重（base.hgn + overlay）相对 BF16 原模型的偏差。unsloth 表里的列（Same top p、Mean KLD、99.9% KLD）
正是 `llama-perplexity --kl-divergence` 打印的统计量。

## 原理

1. 基准模型（BF16 GGUF）跑 `llama-perplexity --kl-divergence-base ref.kld`，写出基准文件：
   文件头、全部 token id，以及每个 chunk 后半段位置上 uint16 量化的 log-softmax
   （范围为 [max-16, max]，约 126 MB/chunk，n_ctx=512 时）。
2. 候选模型读这个文件，喂**同样的 token**（文件里带着，所以分词器差异不影响结果）。
   逐位置算 KL(P_base‖Q)、top-1 是否一致、Δp 等，最后汇总。
3. 每个 chunk 都从空状态开始。只统计位置 n_ctx/2 … n_ctx-2（每 chunk 255 个），
   让每个被统计的 token 至少有 256 个上文。这个 GGUF 的 add_bos=false，token 原样喂入。

引擎的 `--kld-base` 是 llama.cpp `tools/perplexity/perplexity.cpp` 的逐行移植（src/gpu/parts/48_kld.inc），
所以引擎的数和 llama.cpp 各量化的数直接可比。`--kld-save` 按同一格式写出引擎自己的 logits，
llama.cpp 也能读。

```
gdec BASE OVL --kld-base REF.kld [--kld-chunks N] [--kld-save OUT.kld]
gdec BASE OVL --tokens-file IDS --kld-save OUT.kld [--kld-ctx 512] [--kld-chunks N]
```

## 脚本

| 脚本 | 作用 |
|---|---|
| `tools/kld_verify.sh` | 一键自检，不需要 BF16：用 llama Q4_K_XL 当临时基准，检查双向文件兼容、PPL 口径一致和引擎确定性。末行 `KLD VERIFY: PASS/FAIL` |
| `tools/kld_llama.sh base\|cmp` | llama.cpp 侧：`base` 写基准，`cmp` 让候选 GGUF 对比基准 |
| `tools/kld_engine.sh REF` | 引擎侧，使用 `start.sh --check` 的生产 env（同 pp_prod.sh） |
| `tools/kld_bench.sh REF` | 正式对比：引擎 + UD-Q4_K_XL + UD-IQ1_S，输出对比表并附 unsloth 公布值 |
| `tools/kld_table.py` | 解析上面任意日志并出表 |

## 正式流程（需要 BF16）

```bash
# 0) 先自检（约 4 分钟）
bash tools/kld_verify.sh
# 1) 下载 unsloth BF16 GGUF（约 355 GB，放 ~/App/llama.cpp/models/ 下，不要放引擎 models/）
# 2) BF16 写基准：内存放不下，走 CPU + mmap 从 SSD 流权重；把 batch 调大，让每一遍覆盖多个 chunk
LLAMA_ARGS="-ngl 0 -fa on --threads 16" BATCH=4096 CHUNKS=100 \
  bash tools/kld_llama.sh base ~/App/llama.cpp/models/<BF16>/<第一个分片>.gguf data/kld/bf16_c512.kld bf16
# 3) 对比
bash tools/kld_bench.sh data/kld/bf16_c512.kld
```

- 耗时（估算，未实测）：BF16 每一遍都要从 NVMe 读一次约 355 GB，一遍大约 1 分钟起。
  BATCH=4096 时一遍覆盖 8 个 chunk，100 个 chunk 约 13 遍，也就是十几到几十分钟。
  UBATCH 默认等于 BATCH；不要设小，否则每个 ubatch 都要重新流一遍权重。
- 空间：100 chunk 的基准文件约 12.7 GB，放在 data/kld/。
- CHUNKS=100 约 25.5K 个被统计的 token，Mean KLD 的误差条通常已远小于量化档位之间的差距。
  `-1` 表示用满 wikitext-2 test（约 580 chunk，约 73 GB）。

## 读结果

- unsloth 没有公开数据集和 chunk 数，**绝对值不能直接和他们的表比**。要比的是同一基准文件上，
  引擎和 UD-Q4_K_XL/UD-IQ1_S 的相对位置。IQ1_S 顺带验证本机口径：接近 0.396 说明设置和他们相近。
- 引擎的 PPL(base) 必须和 llama.cpp 读同一个基准文件时打印的 PPL(base) 一致（kld_verify 会检查）。
- 引擎自比（读自己 `--kld-save` 的文件）的 KLD 应当约等于 0，只剩 uint16 量化误差。
- 文件只保存 [max-16, max] 范围内的 log-prob，更低的会被截断，所以 PPL(base) 恒比真实 PPL 略低
  （实测约 0.14%，llama.cpp 自己也一样）。要看 PPL 就看 `Mean PPL(Q)`。
- 自检实测（2026-09-24，CHUNKS=8，Q4_K_XL 当基准）：llama 自比 KLD 0.000129 / top-1 99.90%；
  引擎 vs Q4_K_XL KLD 0.237 / 99.9% 6.82 / top-1 89.3%（两个不同的 4-bit 量化互比，
  不代表对 BF16 的偏差）；引擎自比 KLD 0 / top-1 100%；两边读同一个文件时 PPL(base) 6 位有效数字一致。
  **尚未用 BF16 基准跑正式对比**（本机没有 BF16 权重）。

## 注意

- 引擎和 llama.cpp 不能同时跑，内存不够。所有脚本在检测到已有 gdec/llama 进程时直接退出。
- 引擎 512-token chunk 走的是 P<1024 的小 prefill 路径（不走 WMMA GEMM）。数值上与生产的长 chunk
  不 bit 一致，但这是 kernel 累加顺序级别的差异，远小于权重量化带来的 KLD。
