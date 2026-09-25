# GGUF 权重（hgn → GGUF 迁移，进行中）

目标：引擎直接加载 llama.cpp / Unsloth 的 `Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF，用户不再需要
维护 hgn 这类两个超大的专用权重文件。过渡期 hgn 仍然是默认路径，GGUF 通过环境变量逐步接管。

## 用法（混合模式，仍需 hgn）

```bash
D=~/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
export GDEC_GGUF=$D/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf   # G2：路由专家来自 GGUF
export GDEC_GGUF_DENSE=1                                                 # G3.1：其余非专家张量也来自 GGUF
export GDEC_GGUF_MTP=$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf          # G3.1：MTP 头的非专家张量
```

| 变量 | 作用 |
|---|---|
| `GDEC_GGUF=<第 1 个分片>` | trunk 48 层路由专家直接读 GGUF（hipHostRegister 原地映射，不复制），跑 `26_kernels_moe_gguf.inc` 的 WMMA kernel（移植自 gufo，MIT） |
| `GDEC_GGUF_DENSE=1` | 加载期把 dense / norm / router / HC / embed / lm_head / PLE 投影从 GGUF 转成引擎格式（`src/gguf_map.h`），按 hgn 名覆盖 hgn 记录 |
| `GDEC_GGUF_MTP=<sidecar>` | MTP 头的非专家张量从 Q8_0 sidecar 读 |
| `GDEC_GGUF_DENSE_FILTER=a,b,...` | 调试用：只有名字包含其中某个子串的张量来自 GGUF（`!` 前缀表示取反），用于二分定位 |

仍然来自 hgn：PLE n-gram 表（fp8，GGUF 里是 IQ4_NL）、MTP 路由专家。去掉这两项后才能纯 GGUF 启动（G3.2 / G3.3）。

## 转换规则（gguf_map.h）

- Q8_0 矩阵 → dtype 8 q8g32 planar（无损重排）；F32 矩阵 → bf16；F32 向量 → dtype 1 f32。
- GGUF 的 GDN value head 是 "tiled" 排列：引擎 head h = GGUF head `(h%3)*16 + h/3`，
  作用于 attn_qkv 的 v 行、attn_gate、ssm_alpha/beta、ssm_a、dt_bias、conv1d v 通道、ssm_out 输入列。
- `A_log = log(-ssm_a)`；零中心 RMSNorm 权重在 GGUF 里已 +1（ssm_norm 除外），加载时减 1。
- indexer q_proj(512) + k_proj(128) 拼成 index_qk_proj(640)；MTP eh_proj = [embedding 列 | hidden 列]。
- 逐张量对照检查：`tools/g3_map_check.cpp`（vs 生产 hgn）。

## 结果（2026-09-25，gfx1151，`tools/g3_verify.sh` PASS）

| | hgn | G2（专家 GGUF） | G3.1（+dense GGUF） | llama.cpp Q4_K_XL |
|---|---|---|---|---|
| KLD vs BF16（64×512） | 0.1631 | 0.1494 | **0.0453** | 0.049 |
| top1 一致 | 86.61% | 87.20% | **93.25%** | |
| pp 8K @ chunk 2048 | 786 | 1215 | 1160 | |
| pp 32K @ chunk 16384 | 1242 | 1397 | 1392 | |
| decode @ 32K（tok/s） | 30.1 | 25.4 | 21.0 | |
| MTP 投机 commit/round（8K，γ=3） | 3.78 | | 3.71 | |

- 质量提升主要来自非专家张量（hgn 的 dense 是 4-bit q4cp，GGUF 是 Q8_0）。
- 同一 1024 token 上逐 token decode 与 batched prefill 的 PPL：G3 6.814 / 6.760（0.8%），
  hgn 自身 6.905 / 6.862（0.6%），属于两条计算路径的正常差异。
- 不设 GGUF 变量时与 HEAD 的 hgn 结果逐位相同。
- **已知短板：decode。** 两个原因：P=1 时 WMMA 专家 kernel 16 行只用 1 行；
  Q8_0 dense（含 248320×2560 的 lm_head）每 token 读取字节约为 q4cp 的 2 倍。
  下一步：专用 GGUF decode GEMV。

## 验证脚本

- `tools/g2_verify.sh` — G2（专家）
- `tools/g3_verify.sh` — G3.1：KLD、decode/prefill PPL、hgn 路径逐位不变、MTP、速度
- `tools/g3_smoke.sh [ENV=...]` — 512 token PPL 冒烟（可带过滤器等 env）
- `tools/moe_gguf_test.cu` — MoE kernel vs CPU 参考
