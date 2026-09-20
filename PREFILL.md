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
65536 触发内存 PSI 看门狗不可行；生产 chunk 现为 **32768**
（start.sh GDEC_PREFILL_CHUNK）。§3/§10 中 WMMA、bf16 KV、MoE Lt 当时为
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
| `GDEC_PLE_URING=1` | 启用 io_uring PLE 聚集（Linux） |
| `GDEC_MOE_NAIVE=1` / `GDEC_MOE_LT=1` / `GDEC_MOE_LT_BF16=1` | MoE：分组 GEMV 回退 / per-expert hipBLASLt / Lt 输出也 bf16 |
| `GDEC_MOE_HOST_ROUTE=1` / `GDEC_MOE_UNTILED=1` | MoE 路由/切分回退 |
| `GDEC_MOE_FP32_IO=1` | MoE 激活回退 fp32 staging |
| `GDEC_INDEX_OLDSEL=1` | top-512 回退 rocPRIM 排序版 |
| `GDEC_MROPE_DUMP=<file>` | dump RoPE 表（调试用，会同步流） |

## 10. PP 优化切入点（2026-09-20 实测刷新）

已实测否决/打平：WMMA/bf16 KV/MoE Lt 两两 A/B 均 ±1%（现已全部为生产默认）；
`GDEC_MOE_ATOMIC` scatter -16%；chunk≥65536 内存不可行；Lt heuristic 8 候选
维护者已扫（index 4 最优，gdec.cpp:8990-8993 注释）；GDN window=8 触发
illegal memory access（现存 bug，待查）。

1. **MoE 是计算受限**（38.6 TFLOP/8K-chunk，v_dot2 管线 MFU ~40-55%），不是
   纯带宽受限：fused W4 的 64-token tile 把热专家权重流 ~3 遍（~6.7GB/层），
   但与 Lt 的"全专家 dequant + Tensile 小 GEMM"实际打平——两者撞同一天花板。
   2026-09-20 已落地：untiled 路径补上了 packed bf16 x/hid（与 tiled packed
   **bit-identical**，8043 token 对拍 0 分歧）+ `GDEC_MOE_PAIRS_BF16`
   （bf16 pairs + 现成 `k_moe_reduce_pw_bf16`，同样 0 分歧）——**性能仍打平**
   （959-967 同噪声带），进一步坐实计算受限结论。再往下必须 WMMA 化（大工程）。
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

## 11. 后续优化空间（2026-09-20 评估，按收益/可行性排序）

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
   剩余空间：store 侧 bank 冲突（残余 ~64% 冲突主因）、barrier 批处理
   （需三缓冲 LDS，当前 62.5KB 放不下）、N=2560 形状 persistent+K 拆分。
2. **MoE WMMA 化**：MoE ~2.9s/8K-chunk 撞 v_dot2 MFU 天花板（~40-55%），
   WMMA 化理论可到 ~1.2-1.6s（+10-15% PP）。工程量大：LDS dequant→
   fragment 布局重排、占用率重调（MOE_OCC.md 记录多次管线回退）、
   累加顺序变化破坏 bit-exact。Phase 1 的 untiled+packed+bf16 pairs
   基础设施已就位（bit-exact 验证过），是其前置。
3. **GDN stream 窗口串行**：GDEC_GDN_STREAM 下 32 窗口/层严格串行
   （intra→strip 关键路径），长 chunk 下更明显；窗口间流水线化或增大
   窗口（注意 window=8 现触发 illegal memory access，**现存 bug 待查**）。
4. **小 chunk 尾包固定开销**：32K prompt 尾部 53-token mini-chunk 花 0.92s
   （实测 22:04 生产日志）；chunk 对齐或尾包与主 chunk 合并有 ~3% 空间。
5. **GR 去重读**（§10.6）：~0.1-0.2s/8K，小改但收益薄。
6. **PLE gather 与 GPU 计算的更深重叠**：ple_host 冷态 410-870ms 可见，
   热态已被重叠掩盖；低优先级。

## 附：本文档的未复核项

本文档由代码调研生成，绝大部分行号已复核；以下细节来自调研记录但未
逐行复核，引用前请确认：`prefill_chunk` 主体内写 QSA cache 的具体语句、
MoE prefill hipBLASLt 段注释（约 4718-4904）、GEMM 分流段（8895-9430）
的完整分支条件。
