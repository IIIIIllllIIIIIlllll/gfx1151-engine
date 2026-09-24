# Prefill 实现说明（PP 性能优化基线文档）

本文档固化 gfx1151-engine prefill 阶段的实现方案，作为后续 PP（prefill
performance）优化工作的基线，避免重复调研。所有行号基于当前 HEAD 的
`src/gpu/gdec.cpp`（14064 行，kernel 与 host 编排同文件）；文件头注释
（`gdec.cpp:1-59`）本身是 prefill 优化史（Phase 3a–3f）的索引，改动
prefill 时请同步更新该注释与本文档。

实测基线（README.md:23）：gfx1151 + 122 GiB 内存，prefill 约
**600–800 tok/s**；prefill chunk 默认 8192（README.md:72）。

2026-09-20 复测（start.sh 生产 flags，`amd_iommu=off`，详见 ~/ppbench/BASELINE.md）：
8K 单 chunk ~963 tok/s，32K ~1086 tok/s。要点：`amd_iommu=off` 带来
+5%（BIOS 开关可能不生效，需内核 cmdline 确认 `iommu_groups` 为空）；
chunk 16384→32768 对 32K prompt +7.4%（单 chunk 摊薄每 chunk 固定开销），
65536 触发内存 PSI 看门狗不可行；当时生产 chunk 改为 32768，
**2026-09-24 核对 start.sh 默认 PREFILL_CHUNK 已是 16384**（§11.9 的数据
都按 16384）。§3/§10 中 WMMA、bf16 KV、MoE Lt 当时为
opt-in，现均已是 start.sh 生产默认；这些开关两两 A/B 均 ±1% 打平。
`GDEC_MOE_ATOMIC`（scatter 免 pairs）实测 -16%，方向已否决。

## 1. 总体架构

模型为 48 层混合架构：`layer % 4 == 3` 为全注意力 QSA 层（12 层），其余
36 层为 GDN 线性注意力（类 Gated DeltaNet，`is_qsa()` gdec.cpp:214）。
关键配置（`Cfg`，gdec.cpp:195-210）：d=2560，qsa_hq=24 / qsa_hkv=2 /
qsa_dh=256，gdn_hk=16 / gdn_hv=48 / dk=dv=128，experts=512 / topk=10 /
moe_mid=640，gr_rank=320，vocab=248320。

**prefill 不逐 token 跑**：按 `maxbatch`（= prefill chunk size，
gdec.cpp:7594）分 chunk，chunk 内全部批量化（GEMM 化）。workspace 按
maxbatch 而非 maxctx 分配（gdec.cpp:7841）。chunk 之间通过 QSA KV cache、
GDN 循环状态 S、conv 状态连续推进（含 kvsnap 分段）。

入口：

- `Model::prefill_batch`（gdec.cpp:9997）— 外层循环，按 maxbatch 切 chunk。
- `Model::prefill_chunk`（gdec.cpp:10022，主体约 10022-10294）— 单个
  chunk 的完整前向：embedding gather → 48 层 → logits。
- serve 路径调用点：gdec.cpp:13300-13370；API 层只统计总耗时
  `prefill_ms`（src/api/main.cpp:240, 386-389）。
- 观测面：每层进度 `\rprefill L%02d/%d`（gdec.cpp:9937）、chunk 级 tok/s
  （gdec.cpp:10294）。
- CPU 参考（逐 token）：src/ref.cpp:782，全程要求与其 bit-exact。

## 2. chunk 内流水线（批量 kernel 链）

按数据流顺序：

1. **Embedding 批量 gather**：Q4C-P 反量化 `k_q4cp_row_b_hc_bf16`
   （gdec.cpp:2572），直接产 bf16。
2. **RoPE**：每 token cos/sin 预计算表 `k_rope_cs`（gdec.cpp:2952；规避
   gfx1151 上 fp64，注释称 fp64 工作量减 24x）+ 批量旋转 `k_rope_b`
   （gdec.cpp:2930）。decode/越界行仍走内联 fp64 路径。
3. **Dense 投影 GEMM**：见 §3 三路分流。
4. **Attention**：QSA 层见 §4，GDN 层见 §5。
5. **PLE**：47.7 GiB 外挂表，逐 token 16×160B 行随机聚集，主机侧异步 I/O
   直读 pinned staging（见 §7）；批量前向 `ple_gpu_b`。
6. **MoE**：GPU 路由（`k_router_topk` gdec.cpp:1966 + rocPRIM
   `radix_sort_pairs` 稳定排序 `GpuMoeRouting` gdec.cpp:1014-1048 +
   `k_moe_tiles` 64-token expert 任务切分 gdec.cpp:1002），默认融合 W4
   GEMM（见 §3）。
7. **GR**（gr_rank=320 分支结构）：prefill 直接产 bf16 供下游 GEMM
   （`k_gr_*_b_hc_bf16` 系列，gdec.cpp:2733-2857）。

## 3. GEMM 三路分流（按 batch 大小 P）

句柄：`rocblas_handle rbh` + `hipblasLtHandle_t lth`（Model 成员，
gdec.cpp:7667-7668，注释 "prefill GEMMs (Phase 3e)"）。

| 路径 | 条件 | 实现 | 要点 |
|---|---|---|---|
| 大 batch GEMM | prefill 主力 | rocBLAS `rocblas_gemm_ex`（gdec.cpp:8971）/ hipBLASLt（gdec.cpp:9017, 9057） | bf16 输入 / fp32 输出；Lt 侧 `hipblasLtMatmulAlgoGetHeuristic` 取 8 候选；bf16 转换缓存特判避免重复转换（gdec.cpp:8945，x==d_Rhatb && K==10240） |
| 小 P | 短 prompt / 投机 verify | 自写多行 GEMV `k_q4cp_gemv_mr<P, XBF16>`（gdec.cpp:1124） | 权重只流一遍、x 驻 L2，DRAM 流量 ~N·K/2 与 P 无关；取代 dequant→bf16→hipBLASLt（后者多 ~4.5x 权重字节流量）。同族：`k_bf16_gemv_mr`（1219）、`k_q8g64_gemv_mr`（1276） |
| MoE 专家 | 默认 | 融合 W4 GEMM `k_moe_w4_up` / `k_moe_w4_down` | 直接读 Q4C-P 码，bf16 `v_dot2` + fp32 累加 |

另有一个 fp16 hipBLASLt 封装（gdec.cpp:7162-7278，失败回退
`rocblas_gemm_ex` f16）。

**融合优化**：`prefill_fused` 分支把 norm/GR 输出直接以 bf16 喂下游 GEMM，
跳过中间 fp32 staging 与重复转换；`GDEC_PREFILL_UNFUSED=1` 回退。
MoE 激活同样直传 bf16（`GDEC_MOE_FP32_IO=1` 回退）。

## 4. QSA 层：稀疏批量 flash attention

**自写 kernel，非移植 flash-attention 库，fp32 计算。**

- 稀疏策略（gdec.cpp:54）：每 query 选 **512 个完整 4-token 块 + 不完整
  尾部**；visible < 2052 时等价 dense（`qsa_source_pos` gdec.cpp:472-477
  做稀疏槽位→真实位置映射）。`GDEC_QSA_DENSE=1` 回退 dense。
- 索引器 kernel 链：`k_index_q`（540）→ `k_index_pool`（559，4-token 池化
  key）→ `k_index_scores_tiled`（901，16q×32k 瓦片，LDS 65/33 pitch，
  K 循环起点 `((key/32)&1)*64` 对齐 rocBLAS gfx1151 MT32x32x8 的已知
  行为）→ top-512 选择：默认 `k_index_select_rs`（668，自写 4-pass MSD
  radix，sort-free，n≤8192）或 rocPRIM block radix sort 版
  `k_index_select`（613）；调度 `index_select()`（954-969）。raw 尾部
  由 4-slot ring `k_index_ring`（574）提供。
- 主 kernel `k_qsa_flash`（gdec.cpp:3003，配置注释 2983-2996）：
  128 线程（4 warp）× 16 q 行 × 1 head/block；K/V 以 8 宽块流式；
  q、K 块驻 LDS（pitch 260 floats 防 bank 冲突）；V 直读全局走 L2；
  **24.4 KiB LDS/block → 每 WGP 2 block**；8-score 块粒度在线 softmax
  （按块顺序扫描天然满足 causal）。多 chunk 部分结果由
  `k_qsa_flash_combine`（3215）归并。
- 稀疏 prefill 把 V 暂存 LDS（SharedV 特化，12 Q 头/KV 头）；
  `GDEC_QSA_GLOBAL_V=1` 回退全局读。
- **WMMA 路径存在但为 opt-in**（gdec.cpp:7639-7646）：`k_qsa_wmma`
  （3404，256 thr）与 dim-split 变体 `k_qsa_wmma6`（3721，128 thr），
  需叠加 bf16 KV 模式，调度点 gdec.cpp:9329-9338。**默认 prefill
  attention 不走 WMMA**，主力是 fp32 FMA 路径——这是已预留的 PP 优化
  方向之一。

## 5. GDN 层：chunked(64) 线性注意力扫描

kernel 链演进（头注释 gdec.cpp:6-15）：

- `k_gdn_chunk`（原始单 kernel）→ 默认拆为 **`k_gdn_intra`**（gdec.cpp:5321，
  chunk 内、状态无关、可并行）+ **`k_gdn_inter`**（gdec.cpp:5656，chunk
  间串行 d-tile pass）→ inter 再拆为 **`k_gdn_inter_strip`**：4 条 32 列
  独立条带，grid (4,48)，~25 KiB LDS/block，2 blocks/CU；32 宽条带把
  kcd/q/attn2/k 的 DRAM 重读降为 16 宽版的 1/4，**32K 时约 1.7x，
  bit-identical**。
- 回退开关：`GDEC_GDN_LOOP`（逐 token `k_gdn_step` 1482）、
  `GDEC_GDN_NOSPLIT`、`GDEC_GDN_NOSTRIP`。
- 配套批量 kernel：conv `k_gdn_conv`（1465）、gates `k_gdn_gates`
  （1708）、q/k L2 norm `k_l2norm_qk`（1676）、gated norm
  `k_gdn_gatednorm_b`（1765，warp-per-token wave32，一 block 覆盖 32
  token；用空 asm barrier 阻止 FMA 融合以保持与 smem 归约树 bit-exact，
  1783-1788，可同时产 bf16 给下游 GEMM）。
- 投机解码相关：`k_gdn_verify`（1530，状态驻寄存器）与快照回滚
  `k_gdn_snap`（1593）/ `k_convst_snap`（1641）/ `k_iraw_restore`
  （1660）。

## 6. KV cache 与状态

- **QSA KV cache**：布局 `(pos, hkv*dh)` 按位置追加；默认 fp32。
  `GDEC_QSA_KV_BF16=1` 时写入端 f2bf RNE 转 bf16 存储，读出在寄存器内
  左移 16 位展开为 fp32，计算仍 fp32（统一访问器 `kv_ld1`/`kv_ld4`，
  gdec.cpp:481-501；模式开关 6028/7638）。bf16 模式下稀疏 prefill 走
  `k_qsa_flash_bf16`（3288，bf16 LDS 瓦片 + fdot2）。
- **GDN**：无传统 KV cache，循环状态 `S[h][128k][128v]` + conv ring。
- 索引器侧：4-slot raw ring + 池化 key cache。
- 逐 token 回退路径 `k_qsa_step`（gdec.cpp:1840，三遍 max→expsum→加权 V；
  注释 1838："current token's k/v already appended"）。

## 7. 硬件级特化（GFX1151）

- **wave32**：多处 kernel 按 wave32 设计；`k_q4cp_gemv_gd_topk_h16`
  （2187）用两个 16-lane 子组消除 K=640 时 12 个空闲 lane。
- **LDS 预算精算**：`k_qsa_flash` 24.4 KiB → 2 blocks/WGP；
  `k_gdn_inter_strip` ~25 KiB → 2 blocks/CU；Q4C-P 16 项 codebook 驻
  LDS（各 GEMV kernel）。
- **dot 指令**：MoE W4 用 bf16 `v_dot2` + fp32 累加；bf16 KV 模式用
  `fdot2`。
- **rocPRIM**：top-512 块选择、MoE 路由稳定排序。
- **fp64 规避**：RoPE cos/sin 预计算表（§2）。
- **PLE 主机 I/O 与 GPU 重叠**：47.7 GiB 表逐 token 16×160B 行随机聚集。
  Linux：raw-syscall io_uring（`PleUring` gdec.cpp:243-364，环深 256，
  直读 pinned staging，绕开 page fault + memcpy；需 `GDEC_PLE_URING`）。
  Windows：IOCP overlapped ReadFile（`PleWin` gdec.cpp:388-455，QD=512，
  NVMe 随机读；热行走缓存）。失败即 fatal，不做半途回退。

## 8. 量化格式在 prefill 中的参与

- **Q4C-P**（4-bit codebook，16 项 + 每 32 列 fp16 group scale；主力权重
  格式，定义见 src/hgn.h 与 HGN-FORMAT.md）：**prefill 不做整体
  dequant**——小 P 走 `k_q4cp_gemv_mr` 直接消费码流；MoE 走
  `k_moe_w4_up/down` 直接读码；仅大 batch GEMM 按需
  `k_dequant_q4cp_bf16`（gdec.cpp:2626）转 bf16 喂 rocBLAS/hipBLASLt。
  embedding 同为 Q4C-P。
- **q8g64**（uint8 + 每 64 列 fp16 (s,m)）：overlay o_proj
  （`k_q8g64_gemv_mr` 1276 / `k_dequant_q8g64_bf16` 2647）。
- **BF16**：GEMM 输入、prefill 融合管线激活、可选 KV 存储。
- 全程 fp32 累加；大量 kernel 强调 bit-exact / bit-identical（相对
  ref.cpp / reference/modeling_qwen4_exp.py，gdec.cpp:53）——**改 kernel
  时这是硬约束**。

## 9. 环境变量开关速查（优化实验用）

| 变量 | 作用 |
|---|---|
| `GDEC_NOPREFILLBATCH=1` | 回退逐 token forward() 循环 |
| `GDEC_PREFILL_UNFUSED=1` | 关闭 norm/GR→GEMM 的 bf16 融合 |
| `GDEC_QSA_LOOP=1` | QSA 回退逐 token `k_qsa_step` |
| `GDEC_QSA_DENSE=1` | 关闭稀疏注意力（dense 对照） |
| `GDEC_QSA_GLOBAL_V=1` | 稀疏 prefill 改全局读 V（关 LDS 暂存） |
| `GDEC_QSA_KV_BF16=1` | KV cache 存 bf16（实测数值持平） |
| `GDEC_QSA_WMMA=1` / `GDEC_QSA_WMMA6=1` | WMMA bf16 稀疏 flash（256/128 thr 两版），需 bf16 KV |
| `GDEC_GDN_LOOP=1` / `GDEC_GDN_NOSPLIT=1` / `GDEC_GDN_NOSTRIP=1` | GDN 三级回退 |
| `GDEC_PLE_LOOP=1` | PLE 回退逐 token |
| `GDEC_PLE_URING=1` | 启用 io_uring PLE 聚集（Linux；service.conf `PLE_URING=1` 默认开，引擎只查变量存在性） |
| `GDEC_MOE_NAIVE=1` / `GDEC_MOE_LT=1` / `GDEC_MOE_LT_BF16=1` | MoE：分组 GEMV 回退 / per-expert hipBLASLt / Lt 输出也 bf16 |
| `GDEC_MOE_HOST_ROUTE=1` / `GDEC_MOE_UNTILED=1` | MoE 路由/切分回退 |
| `GDEC_MOE_FP32_IO=1` | MoE 激活回退 fp32 staging |
| `GDEC_INDEX_OLDSEL=1` | top-512 回退 rocPRIM 排序版 |
| `GDEC_MROPE_DUMP=<file>` | dump RoPE 表（调试用，会同步流） |
| `GDEC_PREFILL_TAIL_SLACK=<n>` | 尾包合并上限（默认 Linux 1024 / Windows 0；0 禁用） |
| `GDEC_GR_SCAT4=1` | GR scatter+norm 单 block/token 实验（实测 -0.8%，勿开） |
| `GDEC_GDN_PIPE=1` | GDN intra/strip 双流窗口流水实验（实测 -1%，勿开） |
| `GDEC_GDN_PIPE2=1` | GDN intra fp16 驻留+WMMA+64-chunk 循环（kernel 2.81×，8K +1.7%；被 FUSED 取代） |
| `GDEC_GDN_FUSED=1` | GDN intra+strip 融合持久化 kernel（start.sh 已开：kernel 3.34×，8K +5.7%、32K +4.4%，ids 0 分歧） |
| `GDEC_GEMM_NO_K64=1` | 关闭 2026-09-24 的 KST=64 GEMM 分流（(2560,6144) 回 d3、(320,10240) 回 Lt） |
| `GDEC_IPROJ_SGEMM=1` | fp32 indexer 投影回退普通 rocblas_sgemm（不挑 solution index） |
| `GDEC_MTP_INDEX_SGEMM=1` | MTP 层 indexer 打分回退 sgemm + index_select（不用 tiled 融合 kernel） |

## 10. PP 优化切入点（2026-09-20 实测刷新）

已实测否决/打平：WMMA/bf16 KV/MoE Lt 两两 A/B 均 ±1%（现已全部为生产默认）；
`GDEC_MOE_ATOMIC` scatter -16%；chunk≥65536 内存不可行；Lt heuristic 8 候选
维护者已扫（index 4 最优，gdec.cpp:8990-8993 注释）。

1. **MoE 是计算受限**（38.6 TFLOP/8K-chunk，v_dot2 管线 MFU ~40-55%），不是
   纯带宽受限：fused W4 的 64-token tile 把热专家权重流 ~3 遍（~6.7GB/层），
   但与 Lt 的"全专家 dequant + Tensile 小 GEMM"实际打平——两者撞同一天花板。
   2026-09-20 已落地：untiled 路径补上了 packed bf16 x/hid（与 tiled packed
   **bit-identical**，8043 token 对拍 0 分歧）+ `GDEC_MOE_PAIRS_BF16`
   （bf16 pairs + 现成 `k_moe_reduce_pw_bf16`，同样 0 分歧）——**性能仍打平**
   （959-967 同噪声带），进一步坐实计算受限结论。WMMA 化两条路线
   2026-09-21 均已实测否决（详见 §11.2）。
2. **dense GEMM 已在库天花板**：rocBLAS solution-index 全量扫描
   （tools/gemm_sol_scan.cu，正确方向 N=out, K=in, P=tokens）显示库上限
   ~35 TFLOPS（sol 1178 = Lt index 4 同款 Tensile kernel），引擎 in-context
   已达 24-63 TFLOPS——**库调优无空间**，更快只能自写 GEMM 超 Tensile。
3. **大 batch GEMM 的 dequant→bf16 转换**：Q4C-P 大 batch 路径每次调用重新
   dequant（gdec.cpp:8925-8932，无常驻缓存）；流量占比不大（dense 权重小），
   多 chunk 时被 chunk 增大摊薄。
4. **观测面**：`GDEC_PHASE=1`（per-phase）+ `GDEC_PROF=1`（gemm/ple host 耗时）
   已够用；kernel 级用 rocprofv3 --kernel-trace。
5. `k_index_scores_tiled` 的 K 循环起点对齐了 rocBLAS gfx1151 的已知
   tile 行为（注释 gdec.cpp:898-899）——改 tile 参数时注意该依赖。
6. GR：`k_gr_scatter_norm_b_hc_bf16` 的 y 被 4 branch 重读、Rhat 双读，
   已近带宽屋顶，剩余收益 ~0.1-0.2s（8K）。

## 11. 后续优化空间（2026-09-21 刷新，按收益/可行性排序）

> **2026-09-21 包含式拆分**：gdec.cpp 已拆为 `src/gpu/parts/*.inc`（15 个，
> 主文件只留头注释 + includes + include 清单；单 TU/构建/预处理 token 流
> 不变，交错 A/B 性能打平，ids token 全同）。本文及 tools/ 注释里的
> gdec.cpp 行号自此漂移，**以符号名 + parts 文件名为准**（如 k_gdn_fused
> 在 parts/25_kernels_gdn.inc，Model::gemm 在 parts/40_model.inc，
> k_gemm_wmma 标记段在 parts/22_kernels_prefill.inc）。

1. **自写 dense GEMM 超 Tensile（已完成，Phase 3g）**：库天花板 ~35 TFLOPS
   （tools/gemm_sol_scan.cu 实测），自写 WMMA kernel
   （`k_gemm_wmma`，gdec.cpp:3926-4104，gwmma_sync/双缓冲/免 bank 冲突布局，
   `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`）达到 37.4-37.9 TFLOPS，
   8/8 形状全胜 sol 1178（大形状 +15-20%）。**实测 gfx1151 WMMA 发射峰值仅
   ~55.4 TFLOPS**（tools/wmma_peak.cu 标定；rocm-smi sclk 读数是 DPM 假象
   不可信），理论 119 不成立。已集成进 `Model::gemm()`（gdec.cpp:9184-9214），
   env `GDEC_GEMM_WMMA` 门控（start.sh 已开）：(N,K)∈{(2560,6144)→d3/GM16,
   (6144/10240/12288,2560)→d9/GM4} 且 P≥1024 时分流，非 bit-exact（K 累加
   顺序不同，对拍 8049 token 仅生成末段分歧 4 个）。端到端：8K +2%、
   32K +3%。健壮性驱动 tools/gemm_wmma_driver.cu（24/24 含 P=53/100/8199
   非对齐尾包）；峰值/bank 冲突探针 tools/wmma_peak.cu、tools/lds_probe.cu。
   剩余空间：store 侧 bank 冲突已消除（(b,b+RPW/2) 行配对映射，
   SQC_LDS_BANK_CONFLICT 39.8M→0/dispatch，driver 24/24）但**零收益**——
   LDS 管 ~700 相位/K-step 远低于 wmma ~1664 clk，冲突被 compute 完全
   掩盖。三缓冲/预取加深 2026-09-21 实测否决（tools/wmma_gemm_proto.cu
   d9p2/t0/t1/t2，证据与 MEASURED VERDICT 注释留在 proto）：KST 32→16
   结构成本 -34%（barrier 频率翻倍）；预取加深本身 +8% 有效，但每组
   预取寄存器都把 256-VGPR 封顶的 d9 推过悬崖进 scratch（d9p2 spill
   160B -32%、t2 三缓冲 spill 320B 仅 5-7 TF）。真正瓶颈是**每步
   barrier + LDS store→load 可见性往返的固定开销**，且 gfx1151 无
   global→LDS async copy——load 必经寄存器，没有免费的预取深度可买。
   再往前只能动 acc/tile 结构（更小 warp tile 换寄存器余量，未试）；
   N=2560 形状 persistent+K 拆分（未试）。
   **窄形状扩展（2026-09-21，Phase 3g 续）**：rocprofv3 + `GDEC_GEMM_DBG`
   全量形状打点（gdec.cpp:9988 的 P≤8 限制已去掉）发现 63 TF（26%）
   dense GEMM 走 Lt 且有效仅 ~17 TFLOPS。kernel `num_n` 改 ceil(N/BP)
   支持 N 尾包（硬要求放宽为仅 K%32==0，旧形状零影响），分流新增
   d9/gm2 三形状：(10240,320) 1.73×、(512,2560) 1.85-2.0×、(2560,640)
   1.24×；留 Lt 两形状带数据：(320,10240) 0.94×（BP 三头堵：N 尾
   62.5% 利用 / X 重读 1.7GB / BP=320 被 LDS 73KB+寄存器双杀）、
   (640,2560) 1.08×（Lt 自身已 32.6）。driver 54/54；ids ndiff=1 仅
   固有噪声位 8048。**重大 harness 修正：`GDEC_GEMM_WMMA` 此前从未
   出现在 ppbench run.sh——914→1043/1177 的全部基线实际是全 Lt，
   start.sh 生产一直有**。flag 开 vs 关：8K 1083/1043（+3.8%）、
   32K **1243/1177（+5.5%）**，gemm_host 32K 1112→535ms。run.sh 已
   补 GDEC_GEMM_WMMA=1 + GDEC_GDN_FUSED=1 对齐生产。
2. ~~**MoE WMMA 化**~~（2026-09-21 已实测否决）：两条主攻路线均失败，数据与
   分析在 ~/ppbench/moe_wmma_notes.md：
   - **融合 dequant+WMMA expert GEMM 原型**（tools/moe_wmma_proto.cu，up GEMM，
     k_gemm_wmma d9 骨架 + 12 种 dequant 变体二分）：最优精确配置 DM15
     （sector 批量加载 + cbp2 float2 字节对表 + 内联 RNE）达 26.6-28.0ms/层
     ≈ 现 v_dot2 的 25-30ms → **持平，无集成价值**。≤15ms 目标物理不可达：
     uniform ne=160 在 BM=128 下 padding 1.6×，15ms 需 57 issued TFLOPS >
     WMMA 实测峰值 55.4；即使 dequant 零成本（fix0 下界实测）也只有 23.1ms。
     根因：① RDNA3.5 WMMA 跑在 VALU，dequant 精确路径每元素 ~5 op（查表+
     FMUL+RNE）与 wmma 直接争用同一执行单元；② expert 小批次 padding；
     ③ scale/codes sector 级 DRAM 浪费（PMC：GL2C 命中仅 13%）。另发现
     gfx1151 无 v_cvt_pk_bf16_f32、v_permlane16_b32 被强制 uniform 广播、
     __shfl_sync lower 成 ds_bpermute 走 LDS——三条硬件捷径均不存在。
     技术副产品：原型数值位等精确（maxrel 3e-4，与参考 bf16(cb*s) 单次
     RNE 权重 bitwise 一致），若未来硬件变强可直接复用。down GEMM 未做
     （同物理约束，预期同样只到 parity）。
   - **MoE Lt deq/GEMM 双流重叠**（`GDEC_MOE_LT_OVL=1`，env 门控保留、默认
     关、关闭时 bit-exact 全等 8051 token）：实测**负收益**（8K -10.4%、
     32K -4.6%）。根因：双槽 52+26MB 超 32MB MALL 失去驻留优势、deq 与
     GEMM 带宽争用、同一 VALU 争用、每层 ~512 次 event 开销。教训：
     legacy 默认流必须用 hipStreamNonBlocking 侧流才能真重叠（blocking
     版本 -16%）。
   结论：MoE 在 gfx1151 上已无已知的"换执行单元"级收益路径；v_dot2 融合
   路径维持为生产实现。Phase 1 untiled+packed+bf16 pairs 基础设施保留
   （bit-exact 验证过），但性能打平未启用。
3. ~~**GDN stream 窗口**~~（2026-09-21 已收口）：window=8 的 illegal memory
   access 已修复——根因是 `d_gdn_split_ws` 分配把 stream 窗口硬编码为 4
   chunk（15.05 MiB），而窗口循环用 `64*gdn_window_chunks`，8 chunk 时
   ws 越界写（gdec.cpp:8207-8210 改为按 `gdn_window_chunks` 分配）。修复
   后 window=8 与默认逐 token bit-exact（8055 token 0 分歧）。但 A/B 实测
   window=2/8 均比默认 4 慢 ~1.5%（32K：w4 1080 vs w2/w8 各 ~1064），
   **默认 4 确认为最优，勿再调**。窗口间流水化也已实现并实测否决
   （`GDEC_GDN_PIPE=1`，intra 留 g_str、strip 走 NonBlocking 侧流、ws
   双槽奇偶轮换 + 事件链，调度对数值中性——ids 分歧与 base-vs-base
   噪声地板同位置同量级）：8K -0.9%、32K -1.1%、pipe+win8 -1.5%。
   负收益根因与 LT_OVL 同物理：intra 已填满 40 CU，strip 并发只是争用
   CU 拉长 intra，另有 ~256 事件/层与 ws 双槽 30MB 逼近 32MB MALL 的
   驻留损失。**此 GPU 没有空闲执行资源可供重叠**——三条双流实验
   （LT_OVL、GR_SCAT4、GDN_PIPE）全部同向否决。
4. ~~**GDN fp16 WMMA 化**~~（2026-09-21 已实测否决，第 1 步终止未集成）：
   对 k_gdn_intra 做相位剖析 + 墙钟消融（tools/gdn_wmma_proto.cu，逐字复刻
   全相位、grid 128×48、NT=1024、63.2KB LDS，fp32 版 20.2ms/iter 与生产
   ~20ms/层精确吻合）。四变体中位墙钟：fp32 完整 20.2ms / fp16 WMMA
   （相位 1/3/4 矩阵化）21.4ms（**1.06× 变慢**）/ 挖掉全部矩阵计算 19.9ms
   ——**矩阵数学只占墙钟 ~2%，WMMA 加速的对象不存在**。真正瓶颈：63KB
   LDS → 1 block/WGP，~30 个 barrier 把 ~12 轮 staging 全局加载串成延迟
   链，等效带宽仅 ~55GB/s（DRAM 地板 ~3.7ms/层 vs 当前 20ms/层）。参考
   引擎的 fp16 WMMA（k_dnc_scan3x）有效的前提是它的 kernel 不是延迟绑。
   附带发现：GDEC_PHASE_PROF 的 clock64 相位计数有正毛刺（相位和 >
   TOTAL），只有墙钟可信。**staging 重构已完成**（2026-09-21，
   tools/gdn_pipe_proto.cu）：胜者是 fp16 驻留 + WMMA + 每块 64-chunk
   循环的 E 变体 @ ncpb=64——21.94→7.82ms/iter（**kernel 级 2.81×**，
   107 GB/s；收益来源是块间去同步而非块内预取，纯寄存器预取变体实测
   无收益）。已集成为 `k_gdn_intra_p2`（gdec.cpp ~5915-6230，ut5 solve
   逐字保留、ws 布局不变），env `GDEC_GDN_PIPE2` 门控默认关。ids 对拍
   过噪声地板（8054 token 仅 3 分歧 @8047/8050/8052，与基线自身
   run-to-run 噪声同签名——近 tie logit 被原子加翻转）。端到端 8K
   +1.7%、32K 持平：window=4 下 64-chunk 循环展不开，窗口开大则
   intra↔strip 的 ws 交接冲出 L2/MALL（w64=252MB）吃掉收益。**维持
   opt-in 不转正**（收益小且非 bit-exact，待融合 kernel 形态明朗）。
5. ~~**strip fp16 ws 降字节**~~（2026-09-21 已实测否决，proto 未集成）：
   tools/gdn_strip_proto.cu 四变体：fp16 ws（字节 -23%）最优 V4（uint4
   16B 读）仅 -8~10% 墙钟且等效带宽反降至 84-88 GB/s——strip 在
   103 GB/s 处不是 DRAM 字节绑，瓶颈是串行 chunk 扫描延迟链（3
   barrier/chunk，nullc 骨架 7.9ms vs 地板 4.94）+ ~2.9ms 计算 +
   4 strip × j 全维收缩的 L2 结构性冗余（raw ~3.1GB/iter）。降冗余
   几何路线同死：4→2 strip（SW=64）慢 53%（LDS 48KB→1 block/WGP、
   寄存器爆），4 strip 已最优。e2e 折算仅 ~0.5%，不抵复杂度。
   **融合路线随后完成**——见下条。
5b. **GDN intra+strip 融合持久化 kernel**（2026-09-21 已完成并转正，
   Phase 3c）：`k_gdn_fused`（gdec.cpp:6434，block-per-head 48 blocks、
   1024 线程、一次 launch 全 P，env `GDEC_GDN_FUSED`，start.sh 已开，
   与 PIPE2 互斥 fused 优先）。ws 全程不落 DRAM，S 驻留寄存器 WMMA
   C-fragment（每 warp 2 个 16×16 fp32，fp16 转置副本经 LDS 复用），
   三个标量 S 收缩（k·S/q·S/S3，rocprofv3 定位占 80% 时间、VALU 发射
   率仅 19%）全部 WMMA 化；LDS 57,344B、192 VGPR 无 spill。迭代路径
   v1 spill 0.32× → v2 消 spill → v3 部分 WMMA → v4 全 WMMA：
   proto 30.3→**9.1ms/层（3.34×，P=8192 达地板 86%）**，P=32832 同
   3.31×。数值 fro-rel 0.040%（S 0.047%），ids 对拍 8054 token **0
   分歧**；端到端 8K 971→**1026.7（+5.7%）**、32K 1111→**1160.3
   （+4.4%）**，默认路径回归在漂移带内。fused 时 d_gdn_split_ws 分配
   已跳过（gdec.cpp:7079/:9061 `!gdn_fused`，8K 省 505 MiB、32K 省
   2.02 GiB，ids 双路径 0 分歧验证）。波次形态后续（2026-09-21）：
   实测占用单位为 **20 WGP**（非 40 CU，VGPR 192×1024 恰好占满
   196,608/WGP → 1 块/WGP），真实形态 48 块/20 槽 = 3 波（20+20+8）；
   劈块回收全部纸面否决——dc-split 后全关联的 attn/solve（P0 大半
   +P1+P2+P4 = 1.78ms ≈ 单块 59%）必须重复，96 块=5 波 12.0ms
   （0.76×）、56/64 块 1.04-1.07×，越劈越亏；3-head/块共享 k·kᵀ
   ~1.17× 但 S×3=48 寄存器必 spill。位级一致微优化 fused2（P0
   float4 staging + P4 并入 P3a 复用 q A-frag + S3 双缓冲）已集成进
   k_gdn_fused 本体（2026-09-21，无新 env，ids token 0 分歧；8K
   1038.9→**1043.4**、32K ~**1177.6**）；P2 向量化/S3 大 staging/P1
   窗口预取/P0 warp 分工/q 双读均被数据否决。最后的结构性形态
   chunk 对半流水（k_fused3：96 块、后半块先 intra 存 ws 再 spin 等
   前半块的 S）proto 实测否决：spin 本身 ~0.02ms 免费，但 ws DRAM
   往返+重 stage 使后半块 +28%，9.211ms vs fused2 8.633（**0.941×
   倒退**）→ **GDN 维度关闭**（已否决形态全清单见
   ~/ppbench/BASELINE.md "Phase 3d 续"）。
   proto：tools/gdn_fused_proto.cu。
6. ~~**小 chunk 尾包固定开销**~~（2026-09-21 已完成，Phase 3h）：尾
   包并入前一个 chunk——`GDEC_PREFILL_TAIL_SLACK`（默认 Linux 1024 /
   Windows 0，0 禁用），workspace 改按 `maxbatch_cap = maxbatch+slack`
   分配（gdec.cpp:8063-8064、8170；mtp 三缓冲同步放大；arena 估算镜像）。
   32821-token 实测：29.2s→28.8s（**+1.5%**，尾包 0.51s 全省）；对拍
   32876 token 仅生成末段 1 个分歧（chunk 边界改变累加顺序，同
   WMMA GEMM 容差级）。收益按 prompt 长度出现在刚超 chunk 边界的
   情形；短尾越小相对收益越大。
7. ~~**GR 去重读**~~（2026-09-21 已实测否决）：`k_gr_scatter_norm_b_hc_bf16_f4`
   （单 block/token，y 进 LDS 只读一遍，串行 4 branch，env `GDEC_GR_SCAT4`
   保留默认关）实测 8K **-0.8%**——原 4 block 并发时 y 的重复读本就命中
   L2，DRAM 节省是纸面的，而 block 数 4P→P 还损失了并行度。Rhat 链
   （scatter 写 → mix gemm 读 → combine 读 → 下层 inject gemm 读）无法
   去重：combine 依赖两个 gemm 的输出，跨 gemm 融合是大改且收益上限
   ~0.1s/8K。GR 维度关闭。
8. **PLE gather 与 GPU 计算的更深重叠**：ple_host 冷态 410-870ms 可见，
   热态已被重叠掩盖；低优先级。2026-09-24 复核：不开 rocprof 时
   ple 90-98 ms、ple_wait 0-1 ms，与 09-21 相同；rocprofv3 trace 下
   ple_wait 变大是 trace 开销，不是回退。
9. **2026-09-24 一轮（生产 env，chunk=16384，32K prompt 取第 2 个 chunk
   稳态 tok/s；工具 tools/pp_prod.sh，kernel 排行 KTRACE=1 →
   tools/ktrace_top.py）**。三项均已默认开启，各有 opt-out（§9），
   GEN=48 贪心 ids 与改动前逐 token 一致，SPEC=256 投机生成 ids 与
   depth acc 完全一致。一键验证 `bash tools/pp_opt_verify.sh`。
   - **KST=64 GEMM 分流**（Model::gemm）：tools/gemm_lt_bench.cu 加了
     KST=64 配置（KST 只能 32/64/128：RPW=32/(KST/8) 须为偶数）。
     (2560,6144) d3 → `<64,128,2,2,2,4,64,128>` gm=1：P=16384 21.5 →
     16.4 ms；(320,10240) Lt → `<64,160,2,2,2,5,64,128>`：5.3 → 3.9 ms。
     K=2560 的 d9 形状 KST=64 全部更慢，保持不变；(640,2560) k1 仅
     1.48 vs 1.68 ms，未采用。端到端 8K 1061 → 1093（+3.1%），32K
     1225 → 1258（+2.7%）。
   - **iproj_sgemm**（fp32 indexer 投影 640×P×2560）：rocblas_sgemm
     自选的 MT32x32x8 仅 ~2.7 TFLOPS。首次 P≥1024 调用时用
     rocblas_gemm_ex_get_solutions 现场计时挑选（快于 0.95× 才换），
     gfx1151 上选中 -607（hipBLASLt 后端，≈2.4×，确定性，maxrel 7e-6，
     与 sgemm 对 fp64 的 4.5e-6 同级；tools/sgemm_iproj_check.cu 全 P
     ALL OK）。32K 省 ~0.28 s kernel 时间，端到端 +0.3%；首个 chunk
     付一次 ~0.17 s 挑选开销，所以 8K 单 chunk 打平。需要
     `ROCBLAS_BETA_FEATURES_API`（gdec.cpp）和 -Wdeprecated pragma。
   - **MTP indexer 打分走 tiled**（mtp_qsa_b）：MTP 层原先用
     rocblas_sgemm（SB MT128x64x12，单层 0.23 s/32K，比 12 个主干层的
     k_index_scores_tiled 合计还多一半），改为与主干相同的 tiled 融合
     kernel + select。32K 第 2 chunk 1266 → 1272（+0.5%）。
   - 合计 32K：1225 → ~1270 tok/s（+3.7%），8K：1061 → ~1092（+2.9%）。
   - 剩余 kernel 排行（32K，新二进制）：k_gemm_wmma 6.9 s（已是自写
     峰值附近）、MoE Lt（Cijk MT128x128x32 1.9 s + MT32x96x32 1.1 s +
     deq 1.4 s + gather/reduce/silu 1.4 s，§11.2 已否决）、k_qsa_wmma
     1.76 s、GDN fused 1.26 s、GR combine/scatter 2.3 s（§11.7 已关闭）。
     还没做的方向只剩 k_qsa_wmma 调优（大工程）以及 MoE 小形状 Lt
     solution 挑选（每个形状收益 <0.3 s，需逐形状确认确定性）。

## 附：本文档的未复核项

本文档由代码调研生成，绝大部分行号已复核；以下细节来自调研记录但未
逐行复核，引用前请确认：`prefill_chunk` 主体内写 QSA cache 的具体语句、
MoE prefill hipBLASLt 段注释（约 4718-4904）、GEMM 分流段（8895-9430）
的完整分支条件。
