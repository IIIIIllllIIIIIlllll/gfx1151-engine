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
LLAMA_ARGS="-ngl 0 -nkvo -fa on --threads 16 --load-mode mmap --no-host --no-repack --no-op-offload" \
  BATCH=8192 CHUNKS=64 \
  bash tools/kld_llama.sh base ~/Models/BF16/Qwen3.8-Flash-Next-BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf \
  data/kld/bf16_c512.kld bf16_base
# 3) 对比（build/gdec 若早于 KLD 代码，用 BIN= 指向新二进制）
BIN=build/gdec.conc bash tools/kld_bench.sh data/kld/bf16_c512.kld
```

- 这几个 llama 参数缺一不可：load-mode 默认 auto，在 ROCm 上会退回整块分配 354 GB 锁页主机内存
  （失败）；op offload 默认开，会为临时搬到 GPU 的权重申请 482 GB 计算缓冲（失败）。
- **CTX > 2048 时 CPU 必须加 `-nkvo`。** 本机 llama.cpp 分支（qwen4exp.cpp 的
  `qwen4exp_use_block_selection`）只要 `flash_attn && offload_kqv` 就走 maskless 块选择：
  不上传逐 cell 的因果掩码，由 HIP 的 qsa3 kernel 自己做。`-ngl 0` 时 offload_kqv 默认仍为 true，
  CPU 的通用 attention 就会看到未来 token。实测（Q4_K_XL，wikitext 第 1 段）：
  2K 时 GPU 1.394 / CPU 1.376（尚无稀疏，一致）；4K 时 GPU 1.261 / CPU 1.011 / CPU+`-nkvo` 1.259；
  8K 时 GPU 2.10 / CPU 1.008。引擎全量注意力（`GDEC_QSA_DENSE=1`）8K 前 2 段 PPL 2.024，
  所以 1.01 不可能是真实值。512 的表不受影响（全量注意力）。
- 实测耗时（2026-09-24）：加载 2.5 分钟（完整读一遍），BATCH=8192 一遍 16 个 chunk、192 s，
  瓶颈是 CPU 计算（13 核跑满，缺页读盘只有 0.5–2 GB/s）。64 chunk 共 13.4 分钟；
  随后 kld_bench 三项约 5.5 分钟（引擎 139 s、Q4_K_XL 106 s、IQ1_S 83 s）。
  UBATCH 默认等于 BATCH；不要设小，否则每个 ubatch 都要重新流一遍权重。
- 空间：64 chunk 的基准文件 8.1 GB，放在 data/kld/。
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

## 正式结果（2026-09-24，BF16 基准，wikitext-2 test，64 chunk × 512，16320 个统计位置）

| 项 | top-1 % | Mean KLD | 99.9% KLD | 中位数 | PPL(Q) |
|---|---|---|---|---|---|
| BF16（基准） | — | — | — | — | 3.3115 |
| llama UD-Q4_K_XL | 92.78 ± 0.20 | 0.0491 ± 0.0011 | 1.748 | 0.0097 | 3.3426 |
| **引擎 生产（w4b + overlay）** | **86.61 ± 0.27** | **0.1631 ± 0.0030** | 4.387 | 0.0412 | 3.4677 |
| 引擎 不带 overlay | 84.09 ± 0.29 | 0.2378 ± 0.0040 | 5.260 | 0.0626 | 3.7295 |
| 引擎 生产 + fp32 激活（`GDEC_PREFILL_UNFUSED=1 GDEC_MOE_FP32_IO=1`） | 86.53 ± 0.27 | 0.1646 ± 0.0031 | 4.680 | 0.0406 | 3.4754 |
| llama UD-IQ1_S | 76.99 ± 0.33 | 0.4934 ± 0.0075 | 8.999 | 0.1471 | 4.4156 |

- 口径已对上 unsloth：本机 UD-Q4_K_XL 0.0491 / 1.748 / 92.78%，unsloth 公布 0.0469 / 1.547 / 92.26%。
- 引擎约等于 unsloth 的 UD-IQ3_XXS 档（0.165 / 4.04 / 85.4%），权重体积却和 Q4_K_XL 相当
  （w4b 124 GB + overlay 2.5 GB，Q4_K_XL 111 GB）。PPL 高 4.7%，Q4_K_XL 只高 0.9%。
- 中位数 KLD 是 Q4_K_XL 的 4 倍，属于整体性偏差，不是少数位置出问题。
- bf16 激活直传与 fp32 一致，排除激活精度；overlay（官方激活感知量化的稠密层，2.3 GiB）
  把 KLD 从 0.238 降到 0.163，说明偏差主要跟权重量化质量走。剩下的大头推测是主权重里
  无校准 RTN 的 MoE 专家（absmax q4cp）和 fp8 PLE 表，尚未逐项隔离。
- 引擎各行都是生产 env，KV cache 为 BF16（`GDEC_QSA_KV_BF16=1`，代码默认是 FP32）。
  "fp32 激活"一行只改了激活，KV 仍是 BF16；KV 精度的影响尚未单独测。

### 覆盖范围

- 上表 n_ctx=512，统计位置最多到 511；QSA 从位置 2051 起才做稀疏选择
  （512 个 4-token block + 尾部），之前两种模式都是全量注意力。长上下文见下一节。

## 长上下文结果（2026-09-25，BF16 基准，CTX=8192 × 8 chunk，32760 个统计位置）

统计位置 4096…8190，全部在稀疏选择区间内。基准：`-ngl 0 -nkvo`（见上文），
写基准 28 分钟（加载 3 分钟 + 每遍 192 s），文件 16.3 GB（`data/kld/bf16_c8192.kld`）。
引擎：`BIN=build/gdec.conc MAXCTX=8448`，每 chunk 一次 prefill（生产 WMMA 路径）。

| 项 | top-1 % | Mean KLD | 99.9% KLD | 中位数 | PPL(Q) |
|---|---|---|---|---|---|
| BF16（基准） | — | — | — | — | 3.2654 |
| llama UD-Q4_K_XL | 93.83 ± 0.13 | 0.0358 ± 0.0006 | 1.309 | 0.0070 | 3.2790 |
| **引擎 生产（BF16 KV）** | **88.25 ± 0.18** | **0.1219 ± 0.0015** | 3.117 | 0.0335 | 3.3642 |
| 引擎 FP32 KV（去掉 `GDEC_QSA_KV_BF16`） | 88.16 ± 0.18 | 0.1213 ± 0.0015 | 3.043 | 0.0336 | 3.3649 |
| llama UD-IQ1_S | 80.43 ± 0.22 | 0.3566 ± 0.0041 | 7.402 | 0.1061 | 3.9670 |

- 长上下文下所有量化的 KLD 都比 512 低（上下文越长预测越确定）。相对位置不变：
  引擎/Q4_K_XL 的 KLD 比值 8K 为 3.40，512 为 3.32，所以引擎的稀疏注意力实现
  （indexer + 块选择 + WMMA）没有在权重量化之外额外引入可见误差。
- BF16 KV 与 FP32 KV 的差在误差条内（0.1219 对 0.1213），生产用 BF16 KV 没有精度问题。
- 引擎全量注意力（`GDEC_QSA_DENSE=1`）与稀疏前 2 段 PPL 为 2.024 对 2.031：
  稀疏选择本身对这个模型损失很小。
- 偏差仍然主要来自权重：PPL 高 3.0%（Q4_K_XL 高 0.4%）。

### 待测

1. 量化误差逐项隔离：MoE 专家 / fp8 PLE / 稠密层各贡献多少。
2. 更长的上下文（32K）：BF16 CPU 每遍时间随注意力增长，需要单独评估。
- `GDEC_QSA_DENSE=1` 不能当长上下文的"精确对照"：它按全量注意力计算，2051 以上与参考模型
  语义不同。（2026-09-25 起 dense 不再强制 atomic MoE，保留 MoE_LT 确定性路径；decode 从
  k_qsa_step 改为 split flash（selected=nullptr），8K decode 0.9 → 30.6 tok/s，32K 0.2 → 23.6，
  生成 token 与旧实现一致。prefill 仍是 SIMT O(n²)：32K 第二个 chunk 209 tok/s，生产约 1230。）

## 注意

- 引擎和 llama.cpp 不能同时跑，内存不够。所有脚本在检测到已有 gdec/llama 进程时直接退出。
- 引擎 512-token chunk 走的是 P<1024 的小 prefill 路径（不走 WMMA GEMM）。数值上与生产的长 chunk
  不 bit 一致，但这是 kernel 累加顺序级别的差异，远小于权重量化带来的 KLD。
