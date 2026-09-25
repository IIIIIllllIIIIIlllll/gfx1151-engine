# GGUF 权重（hgn → GGUF 迁移）

目标：引擎直接加载 llama.cpp / Unsloth 的 `Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF，用户不再需要
维护 hgn 这类两个超大的专用权重文件。G3.3 起可以完全不用 hgn 启动。

## 用法：纯 GGUF（G3.3，不需要任何 .hgn）

Linux 上 hgn 和 GGUF 各有一个启动器，都读项目根目录 `models/`：

```bash
bash start_hgn.sh    # hgn 权重
bash start_gguf.sh   # GGUF 权重（本文）
```

把 Unsloth 的 Qwen3.8-Flash-Next UD-Q4_K_XL 全部文件放进 `models/`（与 llama.cpp 用的是同一份文件，
可以直接软链过去），另需 `models/tokenizer/tokenizer.json`（与 hgn 共用）：

```text
models/
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf  … 00004-of-00004.gguf   # 4 个分片都要
  mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf                                     # MTP sidecar
  mmproj-BF16.gguf                                                            # 视觉塔
  tokenizer/tokenizer.json
```

对应 service.conf 的"GGUF 权重"一段（同名环境变量优先）：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `GGUF_FILE` | `$MODEL_DIR/…-00001-of-00004.gguf` | 第 1 个分片；其余分片须在同一目录，启动器逐个检查 |
| `GGUF_MTP_FILE` | `$MODEL_DIR/mtp-…-Q8_0.gguf` | 设 `""` 不做 MTP 投机 |
| `GGUF_VISION_FILE` | `$MODEL_DIR/mmproj-BF16.gguf` | 设 `""` 纯文本 |

缺任何一个文件时启动器列出全部缺失项并退出 1，不会启动引擎；`bash start_gguf.sh --check` 只检查。
`start_gguf.sh` 只接受 `.gguf`，`start_hgn.sh` 只接受 `.hgn`；两者都会清掉外部残留的 `GDEC_GGUF*`，
权重格式只由所用的启动器决定。

- 启动器执行 `gdec <第 1 个分片>.gguf ...`，引擎看到 `.gguf` 基座即为纯 GGUF 启动，等价于
  `GDEC_GGUF=<该分片> GDEC_GGUF_DENSE=1`，基座 Checkpoint 为空，全部张量来自 GGUF。MTP sidecar 由启动器
  以 `GDEC_GGUF_MTP` 显式传入（空串 = 不用）；直接运行引擎且没设 `GDEC_GGUF_MTP` 时，引擎自动使用分片
  目录里唯一的 `mtp-*.gguf`。
- PLE n-gram 表直接用 GGUF 里的 IQ4_NL（90 B/行，约 27 GiB；hgn fp8 为 160 B/行 47.7 GiB），
  prefill 在 GPU 上解量化（`k_ple_iq4nl_dequant`），decode 在 CPU 上解量化；`PLE_URING=1` 的 io_uring 批量读
  改为指向表所在的分片。
- 目前只支持 Linux（`GDEC_GGUF_DENSE` 还没移植到 Windows；Windows 启动器只读 service.conf 的 hgn 一段）。

## 用法：混合模式（过渡，仍需 hgn；只用于验证脚本直接运行引擎，启动器会清掉这些变量）

```bash
D=~/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
export GDEC_GGUF=$D/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf   # G2：路由专家来自 GGUF
export GDEC_GGUF_DENSE=1                                                 # G3.1：其余非专家张量也来自 GGUF
export GDEC_GGUF_MTP=$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf          # G3.1/G3.2：MTP 头（含路由专家）
# G3.2：视觉塔在命令行给 --vision-tower $D/mmproj-BF16.gguf
```

| 变量 | 作用 |
|---|---|
| `GDEC_GGUF=<第 1 个分片>` | trunk 48 层路由专家直接读 GGUF（hipHostRegister 原地映射，不复制），跑 `26_kernels_moe_gguf.inc` 的 WMMA kernel（移植自 gufo，MIT） |
| `GDEC_GGUF_DENSE=1` | 加载期把 dense / norm / router / HC / embed / lm_head / PLE 投影从 GGUF 转成引擎格式（`src/gguf_map.h`），按 hgn 名覆盖 hgn 记录 |
| `GDEC_GGUF_MTP=<sidecar>` | MTP 头从 Q8_0 sidecar 读：非专家张量需 `GDEC_GGUF_DENSE=1`；路由专家（Q8_0 gate/up/down）只要同时设了 `GDEC_GGUF` 就原地映射，走同一套 MoE kernel |
| `GDEC_GGUF_MTP_EXPERTS=0` | 调试用：MTP 路由专家仍取 hgn |
| `--vision-tower <mmproj.gguf>` | 路径以 `.gguf` 结尾时按 llama.cpp mmproj（clip / qwen3vl_merger）加载，转成与 hgn 视觉文件相同的 bf16 张量（`gguf_map::build_vision`） |
| `GDEC_GGUF_PLE=1` | 混合模式下 PLE n-gram 表也用 GGUF 的 IQ4_NL（默认保留 hgn 的 fp8 表，KLD 更好，见下） |
| `GDEC_GGUF_DENSE_FILTER=a,b,...` | 调试用：只有名字包含其中某个子串的张量来自 GGUF（`!` 前缀表示取反），用于二分定位 |

## 转换规则（gguf_map.h）

- Q8_0 矩阵 → dtype 8 q8g32 planar（无损重排）；F32 矩阵 → bf16；F32 向量 → dtype 1 f32。
- GGUF 的 GDN value head 是 "tiled" 排列：引擎 head h = GGUF head `(h%3)*16 + h/3`，
  作用于 attn_qkv 的 v 行、attn_gate、ssm_alpha/beta、ssm_a、dt_bias、conv1d v 通道、ssm_out 输入列。
- `A_log = log(-ssm_a)`；零中心 RMSNorm 权重在 GGUF 里已 +1（ssm_norm 除外），加载时减 1。
- indexer q_proj(512) + k_proj(128) 拼成 index_qk_proj(640)；MTP eh_proj = [embedding 列 | hidden 列]。
- 逐张量对照检查：`tools/g3_map_check.cpp`（vs 生产 hgn）。
- 视觉塔：Conv3d patch embed 被转换器拆成两个时间切片 `v.patch_embd.weight` / `.weight.1`
  （各 [1152][3][16][16]），拼回引擎的 [1152][1536]（列 = c*512 + t*256 + y*16 + x）；
  其余是改名（`attn_qkv`→`attn.qkv`，`ffn_up/down`→`mlp.linear_fc1/2`，`ln1/2`→`norm1/2`，
  `post_ln`→`merger.norm`，`mm.0/2`→`merger.linear_fc1/2`）。F32 张量是 bf16 原值的上转，转回 bf16 无损。

## 结果（2026-09-25，gfx1151，`tools/g3_verify.sh` PASS）

| | hgn | G2（专家 GGUF） | G3.1（+dense GGUF） | llama.cpp Q4_K_XL |
|---|---|---|---|---|
| KLD vs BF16（64×512） | 0.1631 | 0.1494 | **0.0453** | 0.049 |
| top1 一致 | 86.61% | 87.20% | **93.25%** | |
| pp 8K @ chunk 2048 | 786 | 1215 | 1160 | |
| pp 32K @ chunk 16384 | 1242 | 1397 | 1392 | |
| decode @ 32K（tok/s） | 30.1 | 25.4 | 21.0 → 24.8（decode 提速后） | |
| MTP 投机 commit/round（8K，γ=3） | 3.78 | | 3.71 → 3.82 | |

- 质量提升主要来自非专家张量（hgn 的 dense 是 4-bit q4cp，GGUF 是 Q8_0）。
- 同一 1024 token 上逐 token decode 与 batched prefill 的 PPL：G3 6.814 / 6.760（0.8%），
  hgn 自身 6.905 / 6.862（0.6%），属于两条计算路径的正常差异。
- 不设 GGUF 变量时与改动前提交的 hgn 结果逐位相同（REF 要用同一提交编出的二进制，见 g3_verify.sh 注释）。
- **decode / 投机提速（2026-09-25 下午）。** 原先两个短板：P=1 时 WMMA 专家 kernel 16 行只用 1 行；
  Q8_0 dense 每 token 读取字节约为 q4cp 的 2 倍。已做：
  1. 小 P（≤8，`GDEC_GG_GEMV`）的路由专家走原始块 FP32 GEMV `moe_gg_gemv`（26_kernels_moe_gguf.inc），
     P=1 每层 158 µs（WMMA 约 225）。
  2. 2≤P≤8 时做专家去重（`GDEC_GG_UQ=0` 关闭）：投机验证的几个 token 共享的专家只读一次，
     与不去重的路径逐位相同（`tools/moe_gguf_gemv_test.cu --share 0.5`：MoE 1.07–1.14×；
     实测投机端到端只有约 +1%，在噪声内，没有共享时开销 <1%）。
  3. q8g32 dense GEMV：P=1 `k_q8g32_gemv_lpr`；P>1 `k_q8g32_gemv_mlpr`（每组权重解码一次，复用于 P 行），
     bf16 x 且 cols≥4096 用每行一个 block 的 `k_q8g32_gemv_brow`（HC down 39→23 µs，out_proj 131→93 µs）。
     基准 `tools/q8g32_gemv_bench.cu`。
  4. rows<64 的 bf16 矩阵（HC inject）：P=1 `k_bf16_gemv_row`，P>1 `k_bf16_gemv_row_mp`，替代 rocBLAS（66→4 µs）。

  | 512 token prompt，同机 | hgn | G3 原始 | G3 现在 |
  |---|---|---|---|
  | decode（tok/s） | 32.3 | 23.0 | 26.2 |
  | 投机 8K γ=3（tok/s） | 51.5 | 38.0 | 45.7–46.8 |

  剩余差距主要是 Q8_0 dense 的字节数（约为 q4cp 的 2 倍），带宽已接近上限；
  只有重新量化（牺牲 KLD）才能消除，暂不做。

- **G3.2（MTP 路由专家 + 视觉塔）。**
  MTP sidecar 的路由专家是 Q8_0 gate/up/down（主体是 Q4_K/Q5_K + Q5_1），MoE kernel 加了 Q8_0 gate/up
  实例（WMMA 与小 P GEMV 两条路径，`tools/moe_gguf_*test* --layer 48` PASS）。P=1 每步比 hgn q4cp 多约 100 µs
  （Q8_0 字节多），占一轮投机 <1%。

  | 投机 γ=3，同机 | MTP 专家来自 hgn | 来自 sidecar |
  |---|---|---|
  | 8K prompt tok/s（commit/round） | 46.0 / 46.2（3.82） | 46.3 / 46.1（3.82） |
  | 512 prompt tok/s（commit/round） | 33.9（2.88） | 35.0（2.99） |

  视觉塔：333 个张量与 hgn 视觉文件逐位相同；离线 `--vision-test` 前向的 35 个层 dump 逐字节相同
  （`tools/g3_vision_verify.sh`）。加载 1.2 s（hgn 1.0 s）。

- **G3.3（纯 GGUF 启动）。** 和混合模式唯一的数值差别是 PLE n-gram 表：GGUF 里只有 IQ4_NL，
  比 hgn 的 fp8 粗，KLD 0.0453 → 0.0511。此时引擎和 llama.cpp 用的是逐字节相同的权重（llama.cpp 0.049），
  剩下的 +0.002 来自计算路径（bf16 KV 等）。表变小后 prefill 反而略快（页缓存压力小）。

  | 同机，`tools/g3_pure_verify.sh` | G3.2（hgn fp8 PLE 表） | 纯 GGUF（IQ4_NL PLE 表） |
  |---|---|---|
  | KLD vs BF16（64×512） | 0.0453 | 0.0511 |
  | top1 一致 | 93.25% | 92.64% |
  | 1024 token PPL prefill / decode | 6.814 / 6.760 | 6.862 / 6.802 |
  | pp 8K @ chunk 2048（两轮） | 1126–1155 | 1196–1206 |
  | pp 32K @ chunk 16384 | 1388 | 1403 |
  | decode @ 32K（tok/s） | 24.8 | 24.8 |
  | 投机 8K γ=3 tok/s（commit/round；hgn 51.3 / 3.78） | 46（3.82） | 45.9–46.1（3.82） |

  纯 GGUF 与"hgn 基座 + GGUF 全覆盖 + `GDEC_GGUF_PLE=1`"的 PPL 逐位相同，说明没有任何张量还在读 hgn。

## 同机完整对比（`tools/bench_full.sh`）

两个启动器各自的生产配置（`start_hgn.sh` / `start_gguf.sh --check` 的环境变量与权重参数，prefill chunk 16384），
2026-09-25 重启后全新编译。离线项用 `tools/pp_prod.sh` / `tools/kld_engine.sh`，API 项由启动器真正起服务
（256K 上下文、4 路并发，只关 KVSNAP），数字取自 API / 引擎日志的逐请求统计。

| 项目 | hgn | GGUF |
|---|---|---|
| prefill 8K（tok/s） | 1056 | 1370 |
| prefill 32K（整体 / 末 chunk） | 1250 / 1242 | 1435 / 1441 |
| prefill 64K（整体 / 末 chunk） | 1242 / 1209 | 1409 / 1370 |
| decode @32K，不投机（tok/s） | 30.5 | 25.0 |
| MTP 投机 8K prompt + 256，γ=3（tok/s，commit/round） | 51.9（3.84） | 46.6（3.82） |
| KLD vs BF16 / top1 / PPL | 0.163 / 86.61% / 3.468 | 0.0511 / 92.64% / 3.348 |
| API 就绪后内存 device / RSS（GiB） | 32.6 / 69.4 | 34.8 / 81.1 |
| API 中文散文 512 tok（tok/s，草稿接受率） | 30.1（0.40） | 26.2（0.37） |
| API 代码 512 tok（tok/s，草稿接受率） | 31.1（0.41） | 32.3（0.55） |
| API 8K prompt 首 token（s，prefill tok/s） | 7.6（1016） | 6.2（1237） |
| API 32K prompt 首 token（s，prefill tok/s） | 27.2（1209） | 24.3（1353） |
| API 4 路并发 ×256 tok 总吞吐（tok/s） | 29.8 | 28.5 |

GGUF prefill 快 13–30%、KLD 只有 hgn 的 1/3；decode 慢约 18%（Q8_0 dense 字节多）。
对话类文本草稿接受率只有 0.4 左右，投机收益远小于离线 8K 文档 prompt（commit/round 3.8）。
pp_prod.sh 现在把启动器命令行里的全部权重参数传给引擎，hgn 的投机用的是生产的 8-bit `mtp.hgn`
（以前只传主模型 + overlay，用的是 overlay 内置的 4-bit 草稿头，51.5 / 3.78）。

## hgn 路由专家也走 WMMA：LUT 解码 kernel（2026-09-25，`tools/lut_verify.sh` PASS）

以前 hgn 没吃到 G1 MoE kernel 的收益：它只认 GGUF 块格式，hgn（q4cp）prefill 仍是
dequant + hipBLASLt（`GDEC_MOE_LT`）/ 标量 `k_moe_w4`。q4cp 与 IQ4_NL 结构几乎一样
（32 元素一组、4-bit 查表索引、fp16 scale），所以写了一个 LUT 解码族 kernel
`src/gpu/parts/27_kernels_moe_lut.inc`：`k_moe_lut<kQ4CP | kIQ4NL | kIQ4XS, pair>`，
流水线/tiling/epilogue 与 `k_moe_gg` 相同，解码 = LDS half2 对表（一个字节查出两个元素）× scale。

- **默认开启**（hgn prefill，P > moe_naive_max）；`GDEC_MOE_Q4W=0` 恢复旧路径且与旧提交逐位相同。
  开启时 `GDEC_MOE_LT` 不再生效，它的 dequant/gather 缓冲（`d_moexg` 等，chunk 16384 约 0.8 GiB、
  Windows chunk 8192 约 0.4 GiB）也不分配，`devarena_estimate` 同步。
- decode（P ≤ 16 的 grouped GEMV）完全不变：逐 token decode mean_nll 与旧二进制逐位相同。
- IQ4_NL / IQ4_XS 模板已在 `tools/moe_lut_test.cu --synth iq4nl|iq4xs` 对 CPU 参考通过；
  引擎的 GGUF 路径还没接（手头没有 IQ4 专家的 GGUF 文件），以后给 Windows 用 UD-IQ4_XS 时再接。

kernel（`tools/moe_lut_test.cu --hgn`，真实 q4cp 权重）：P=2048 10.6 ms/层（旧 `k_moe_w4` 28.3，
GGUF Q4_K 11.8）；P=16384 0.168 ms/token（旧 LT 路径 0.291，GGUF 0.173）。

引擎（`start_hgn.sh` 生产配置，同一二进制 off / on）：

| 项目 | off（旧路径） | on（LUT WMMA） | GGUF 参考 |
|---|---|---|---|
| KLD vs BF16 / top1 | 0.1631 / 86.61% | 0.1628 / 86.81% | 0.0511 / 92.64% |
| prefill 8K @ chunk 2048（tok/s） | 816 | 1215（+49%） | 1206 |
| prefill 8K 单 chunk | 1096 | 1361（+24%） | 1370 |
| prefill 32K @ chunk 16384 | 1237 | 1422（+15%） | 1435 |
| decode @32K（tok/s） | 30.5 | 31.2 | 25.0 |
| MTP 投机 8K γ=3（tok/s，commit/round） | 51.6（3.84） | 49.3（3.67） | 46.6（3.82） |

投机每轮耗时不变，commit/round 的差别来自 prefill 数值略变后生成的文本不同（单次 256 token，
脚本门槛 ≥ 0.95×off）。1024 token prefill vs decode PPL 6.875 / 6.905（< 1%）。

结论：hgn 的 prefill 速度追平 GGUF，decode 保持 hgn 的优势，显存比 GGUF 少约 11 GiB——
Windows（96 GiB 上限）继续用 hgn 即可拿到这份 prefill 提速，重新编译即可。质量（KLD）仍是 hgn 的
0.163，下一步是用 imatrix 重新量化出高质量 hgn（dense 8-bit）。

## 验证脚本

- `tools/g2_verify.sh` — G2（专家）
- `tools/g3_verify.sh` — G3.1：KLD、decode/prefill PPL、hgn 路径逐位不变、MTP、速度
- `tools/g3_vision_verify.sh` — G3.2 视觉塔：张量逐位对照 + 端到端前向 dump 逐字节对照
- `tools/g3_pure_verify.sh` — G3.3 纯 GGUF：start_gguf.sh 配置、KLD、与混合启动逐位相同、decode/prefill、
  hgn 路径逐位不变、MTP、速度
- `tools/launcher_verify.sh` — 两个启动器：--check 命令行/环境、缺文件报错、格式互斥、端到端起停
- `tools/bench_full.sh` — hgn vs GGUF 完整性能：prefill 8K/32K/64K、decode、投机、KLD、API 端到端，
  汇总表写到 logs/bench_full.md（`FORMATS=gguf` 只测一种，`SKIP="kld api"` 跳过几项）
- `tools/g3_smoke.sh [ENV=...]` — 512 token PPL 冒烟（可带过滤器等 env）
- `tools/lut_verify.sh` — hgn LUT MoE kernel：开关关闭与旧二进制逐位相同、decode 不变、KLD、
  prefill/decode PPL、MTP、速度（`BIN=` 新二进制，`REF=` 旧提交二进制）
- `tools/moe_lut_test.cu` — LUT kernel（q4cp 真实权重 `--hgn` / 合成 IQ4_NL、IQ4_XS `--synth`）vs CPU 参考
- `tools/moe_gguf_test.cu` — MoE kernel vs CPU 参考
- `tools/moe_gguf_gemv_test.cu` — 小 P 专家 GEMV（含去重路径逐位对照）vs CPU 参考
- `tools/q8g32_gemv_bench.cu` — q8g32 dense GEMV 各变体（P=1 与 P=4）
