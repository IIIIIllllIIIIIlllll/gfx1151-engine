# gfx1151-engine vs gufo：prefill 差距分析（2026-09-25）

gufo：`gufo-main/src/models/qwen38_flash_next`，同一模型，权重为 Unsloth UD-Q4_K_XL GGUF，
prefill chunk 固定 2048，单 stream，prefill 不用 hipGraph。
它公布的 pp2048@depth（预热，HTTP）：d0 1628 / d4K 1523 / d8K 1499 / d16K 1457 / d32K 1422 / d64K 1304 tok/s。

## 1. 同口径实测（我们，生产 env，32K prompt，远程 logs/cmp_c*.log）

| chunk | 每 chunk tok/s | 说明 |
|---|---|---|
| 2048 | 745(冷) 830 803 802 793 787 800 776 783 762 779 777 771 790 785 762 | 约 800，与 gufo 同口径 → **gufo ≈ 2.0×** |
| 4096 | 852(冷) 920 913 919 912 894 896 903 | |
| 16384（生产） | 1194(冷) 1231 | 即使用我们最优配置，gufo@2048 仍快 1.15–1.3× |

我们的性能严重依赖大 chunk；gufo 在 2048 就到了 1400–1600。

## 2. 逐组件对照（ms/token，kernel 时间）

chunk=2048，8K prompt（logs/kt_c2048.ktrace，合计 10.29 s / 8192 tok）对比 gufo d0 profile：

| 组件 | 我们 | gufo | 差 |
|---|---|---|---|
| 路由 MoE：k_moe_w4_up 3476 ms + k_moe_w4_down 1963 + reduce 209 + router 52 | **0.696** | 0.233 | **+0.463（约 78%）** |
| Dense 投影（k_gemm_wmma + router/iproj Lt + q4cp 反量化；含 HC up/down、shared） | 0.301 | 0.240（dense 0.193 + HC down/shared 0.047） | +0.061 |
| 注意力 + indexer（我们是 0–8K 深度平均） | 0.096 | 0.037 | +0.059 |
| HC combine + scatter_norm | 0.075 | 0.070 | ≈ |
| GDN（fused + conv + gatednorm + l2norm） | 0.070 | 0.063 | ≈ |
| 合计 | 1.256 | 0.665 | 0.591 |

chunk=16384，32K（logs/dab_kt_sp_32k.ktrace，25.72 s / 32768 tok）：MoE 0.291（Lt GEMM 20 种 tile 共 6.52 s，
约 24 TFLOPS；外围 deq_bf16 1.35 s + gather 0.73 + reduce 0.46 + silu 0.25 + router 0.21 ≈ 3.0 s，占 31%），
dense 0.246，注意力 0.092，HC 0.069，GDN 0.068。
> 更正：之前说“我们 MoE 更便宜”是错的，那次只算了 2 种 Lt tile。

## 3. 根因：小 chunk 的 MoE kernel 没有用矩阵核

- chunk < `GDEC_MOE_LT_MIN`(4096) 时走 `k_moe_w4_up/down`（23_kernels_moe_w4.inc）。
- 它在 LDS 里存反量化后的 bf16，然后做**标量 `v_dot2_f32_bf16`**。注释里自己也写着受 LDS 带宽限制。
- 实测：约 7 TFLOPS；每层 1.26 GB 权重读了 28 ms，折合 45 GB/s。**既不在算力上限，也不在带宽上限。**
- gufo 的 `RoutedF16GEMMKernel`（kernels.hip.cpp:3637）做法：
  - 原始 Q4_K/Q5_1 码字以打包形式放进 LDS，两个 K 块只占 11–12 KB，每个 WGP 驻留 5 个 block。
  - 每个 wave 读出自己那 16 行后解码：`perm` 加 0x6400 魔数得到 F16，再一次 `hfma2` 套 scale/bias。
  - F16 WMMA，F32 在整个 K 上累加，不做逐块的 epilogue。
  - gate/up 成对（wave 0-3 算 gate、4-7 算 up），epilogue 里做 SwiGLU，直接写 down 的 F16 输入。
  - down 写 F16 `[token][slot][hidden]`；带权求和放进下一步的 HC combine。
  - 由 host 根据 D2H 回传的专家计数建 tile 表（每层同步一次），tile 按专家分 48/64/128 行三档。
  - 结果：整个 MoE 约 20 TFLOPS。

## 4. 可借鉴点与预估收益（按优先级）

1. **把融合 MoE kernel 改成 WMMA 版（最大杠杆）**
   - 做法：Q4C-P 码字留在 LDS，读出后查 16 项码本转 bf16/f16，再做 WMMA 16x16x16。
   - gate/up 成对并在 epilogue 做 SwiGLU；down 带 router 权重输出。
   - 小 chunk 和大 chunk 用同一个 kernel，这样就不再需要 deq_bf16 + Lt + gather/silu 那条路径。
   - 设计点：码本不是均匀量化，不能直接照搬 gufo 的魔数技巧。可以沿用 cb_pairs（LDS 表），或者把 16 项码本放在寄存器里用 perm 查表。
   - 预估：
     - chunk=2048：MoE 0.70 → 约 0.25，整体约 800 → 约 1200 tok/s（+50%）。
     - 生产 16K chunk：MoE 0.29 → 约 0.20，1231 → 约 1370（+11%）。
     - 之后可以重新扫一遍 chunk 大小：小 chunk 能省约 11 GB workspace，短 prompt 的首 token 延迟也更好。
2. **HC gate 融合（#51，原型已 bit-exact，up GEMM 加 combine 快 2.1×）**：每 32K 省约 1 s，+3.5–4%。gufo 的 HcMix 就是这么做的，gate logits 不落显存。
3. **在生产者 kernel 的 epilogue 里直接写出下一个 GEMM 的输入格式**
   - gufo 的 HcCombine 同时写 F32 残差、F16 xn 和 tiled Q8 xn。
   - QKV GEMM 的 epilogue 里做 q/k RMSNorm、RoPE、KV 写入；SSM GEMM 的 epilogue 里做 conv4 + SiLU。
   - 我们对应的独立 kernel：k_l2norm_qk_b、k_gdn_conv_b、k_qsa_qsplit、k_sigmoid_gate、k_f32_to_bf16_v4、k_dequant_q4cp_bf16。它们在 16K chunk 合计约 1.4 s/32K，预估 +3–5%。
4. **Dense**：chunk=2048 时我们 0.301 对 gufo 0.240。
   - k_dequant_q4cp_bf16 每次调用都要反量化一遍（c2048 下 108 ms/8K）。
   - gufo 是在寄存器里把 Q8_0 解码后直接进 LDS。
   - 对 K 大 N 小的形状（2560×6144、320×10240），gufo 用 int8 WMMA 更快。
5. **注意力（深度越深占比越大）**
   - gufo：F16 KV；一个 block 负责 4 query × 12 head，把 mask 的并集压成 LDS 列表，按 16-key 的 tile 跳过。
   - 可以与 k_qsa_wmma 逐项对照（它在 32K 占 6.9%）。

## 5. 其他说明

- **质量**：gufo 用的 Q4_K_XL，KLD 约 0.049；我们引擎是 0.163（KLD.md）。gufo 在速度和质量上都占优。imatrix 重量化仍然要做。
- **decode**：gufo 文档写的 tg 约 26 tok/s；我们不开投机是 31.9，开 MTP 约 55，decode 不落后。
- **测量口径**：gufo 表里的数都是预热后测的；我们的 pp_prod 是冷启动单次，第一个 chunk 偏低约 5–8%。
