# 高质量 hgn（8-bit dense overlay + imatrix 路由专家）

目标：保持 hgn 格式（Windows 只能用 hgn，95 GiB arena 放不下 GGUF），把 KLD 从 0.163 降到接近
GGUF Q4_K_XL（0.049），速度尽量不掉。新文件都从原始 BF16 safetensors 重新量化，
**格式、张量名与现有文件一致**，旧文件、旧二进制行为不变，换文件即可切换。

## 0. 一键转换：tools/flashnext2hgn.py

集成转换器从原始权重直接生成整套高质量 hgn，**有没有 imatrix 都能转**（只依赖 numpy）：

```bash
# 有 imatrix（llama.cpp 格式：GGUF，如 unsloth 的 imatrix_unsloth.gguf_file；或旧版 imatrix.dat）
python3 tools/flashnext2hgn.py ~/Models/Qwen3.8-Flash-Next --out ~/Models/hq/conv \
    --name qwen38-flash-next-hq \
    --imatrix ~/Models/BF16/Qwen3.8-Flash-Next-BF16/imatrix_unsloth.gguf_file

# 没有 imatrix：去掉 --imatrix 即可（专家改用 x² 加权搜 scale，仍优于旧转换器）
python3 tools/flashnext2hgn.py ~/Models/Qwen3.8-Flash-Next --out ~/Models/hq/conv
```

（远程机器上 numpy 在 `~/Workspace/pylib`，要加 `PYTHONPATH=~/Workspace/pylib`。）

| 输出 | 内容 | 大小 |
|---|---|---|
| `<name>.hgn` | 基座：路由专家加权 q4cp（有 imatrix 用 imatrix），norm bf16，PLE fp8，dense 的 q4cp 备份 | 115.6 GiB |
| `<name>.overlay.hgn` | 8-bit dense：724 个 attention/GDN/HC/shared expert/lm_head/embed 张量（q8g32，少数 bf16） | 4.83 GiB |
| `<name>-mtp.hgn` | 8-bit MTP 草稿（q8g64） | |
| `<name>-vision.hgn` | 视觉塔（bf16） | |
| `tokenizer/`、`start.sh` | 分词器；启动脚本按 base → overlay → mtp 加载 | |

32 核约 1.5 小时（28 个量化进程），内存峰值约 20 GiB，读写的大文件用 fadvise 移出
page cache。写 `.part` 再改名；专家在第 0/24/47 层和 MTP 层回读抽查，加权 scale 搜索的误差必须低于
同一码表的 absmax 量化，否则非 0 退出。

| 选项 | 作用 |
|---|---|
| `--imatrix FILE` | 路由专家按 imatrix 加权；没被路由到的专家用本层均值；MTP 层（imatrix 里没有 blk.48）用 x² |
| `--expert-codebook universal\|trained` | 路由专家码表：universal（默认，生产码表形状，KLD 最好）/ 每张量训练（实验） |
| `--jobs N` | 专家量化进程数（默认 min(32, 核数−4)，每个 ~0.7 GiB）；结果与 N 无关 |
| `--only-overlay` | 只写 `<name>.overlay.hgn`（~1 分钟），配现有基座用 |
| `--skip-overlay` | 不写 overlay（dense 用基座里的 4-bit） |
| `--overlay-skip lm_head` | 这些张量不进 overlay、保持 4-bit：lm_head 省 ~0.3 GiB 显存/每 token 读量，KLD +0.006 |
| `--classic` | 旧的无数据转换器（absmax q4cp，无 overlay），与以前逐字节相同 |
| `--skip-vision` `--skip-mtp-sidecar` `--skip-ple` `--dry-run` | 同以前 |

码表：dense 与其它 q4cp 张量的 16 项码表与 `--classic` 相同（同一 rng 抽样，主 rng 流不变）；
**路由专家默认用固定的 universal 码表**（`UNIVERSAL_EXPERT_CB`，对称，max|cb| = 1，正半边
`0.0421 0.1284 0.2209 0.3244 0.4441 0.5873 0.7653 1.0`）。它就是生产 `w4b.hgn` 的码表形状：
生产文件 98 个专家张量的码表归一化后几乎相同（逐项标准差 ≤ 0.004），是一个通用的重尾形状，
不是本仓库旧转换器产生的。旧转换器的码表是从 linspace 起跑 20 轮的 Lloyd，没有收敛，接近均匀
（`[-1, -0.808, -0.663, -0.532, …]`）。

| 路由专家码表（其余相同：加权 scale + q8 overlay + imatrix） | KLD | top1 | PPL | MTP commit/round |
|---|---|---|---|---|
| 旧转换器码表（接近均匀） | 0.0615 | | | |
| `--expert-codebook trained`：每张量加权 Lloyd + 8 轮交替 scale 搜索 / 最小二乘重拟合 | 0.0587 | 91.80% | 3.355 | 3.78 |
| universal（默认，= 生产码表形状；本行是 09-30 一键转换实测，`hgn_q4i.py` 用逐张量生产码表为 0.0558） | 0.0576 | 92.43% | 3.328 | 3.78 |

注意：按 imatrix 加权的重建误差**不能**用来挑码表。trained 码表在留出专家上的加权误差比旧码表低
1.2–2.6%，4 个张量里有 3 个还低于生产码表，但 KLD 反而更差。所以默认用 universal，trained 只作为
实验选项（每张量约 6 秒，独立 rng，种子为张量名的 crc32）。转换器的回读抽查也只要求“加权 scale
搜索优于同一码表的 absmax”（trained 模式与旧码表比），和旧码表文件的误差对比只打印作参考。
文件大小、张量表都和旧转换器一致。一键验证：

```bash
bash tools/convert_verify.sh               # 有 imatrix；IMATRIX=none 测无 imatrix
```

检查：①合成小模型自测（`tools/convert_selftest.py`，30 秒，不需要 GPU：`--classic` 与旧转换器
逐字节相同、HQ 只改专家、legacy/GGUF imatrix 一致、overlay 张量集合与顺序、`--jobs` 无关）；
②真实权重 `--only-overlay` 与 `tools/hgn_hq.py` 的输出逐字节相同；③完整转换，基座张量表与生产
`w4b.hgn` 一致，start.sh 加载顺序正确；④KLD ≤ 0.060（无 imatrix 0.066）；⑤MTP 投机 commit/round ≥ 3.0。

下面第 2、3 节的两个工具是**在现有生产文件上**分别重建 overlay / 基座的办法（布局与旧文件逐字节
对齐，便于 A/B），效果与一键转换相同：

| 文件 | 替换 | 工具 | 大小 |
|---|---|---|---|
| `qwen38-flash-next-w4b.overlay-q8.hgn` | `qwen38-flash-next-w4b.overlay.hgn` | `tools/hgn_hq.py` | 4.83 GiB |
| `qwen38-flash-next-w4b-imat.hgn` | `qwen38-flash-next-w4b.hgn` | `tools/hgn_q4i.py` | 与旧文件逐字节同大小 |

`qwen38-flash-next-mtp.hgn`（8-bit MTP 草稿）和视觉塔不变。

## 1. 质量瓶颈在 dense 的位宽，不在校准

64 chunk × 512 KLD（BF16 基准，见 [KLD.md](KLD.md)），DENSE_FILTER 二分（`tools/hq_probe.sh`，
dense 从 GGUF Q8_0 取，按组切回 hgn 4-bit）：

| 只有这一组保持 4-bit（其余 dense 8-bit） | KLD |
|---|---|
| （全部 8-bit） | 0.0626 |
| 每层 HC（hcl） | 0.115 |
| GDN 线性注意力 | 0.101 |
| QSA 稀疏注意力 | 0.0706 |
| shared expert | 0.0687 |
| lm_head | 0.0686 |
| embed | 0.0649 |

旧 overlay 的 dense 已经是校准过的 4-bit，仍然 0.163；只把路由专家换成 GGUF 也只到 0.149。
所以 dense 全部改成 8-bit（q8g32，dtype 8，引擎早已支持），embed 也一起改。

## 2. 8-bit dense overlay（tools/hgn_hq.py）

- 从 safetensors 读每个 dense 张量，按旧 overlay 的名字/形状写成 q8g32（Q8_0 同算法：
  32 一组，d = amax/127，q = round(x/d)，planar 布局 [int8 × R·C][fp16 × R·C/32]）。
- 小张量保持 bf16：`in_proj_a/b`、`shared_expert_gate`、`block_inject_weight`、`indexer.index_qk_proj`。
- 各种折叠（HC、qk norm、GDN v-head 重排……）与旧 overlay 完全一致：工具以旧 overlay 的
  反量化结果为对照，`--check N` 抽查 N 个张量的相对误差（最差 6.5e-3，即 q4cp 噪声量级）。
- 引擎：只要有非专家 dtype 8 张量，自动打开 few-row bf16 gemv（日志
  `dense: 8-bit (q8g32) weights present, few-row bf16 gemv on`）；旧 4-bit overlay 逐位不变。

```bash
PYTHONPATH=~/Workspace/pylib python3 tools/hgn_hq.py ~/Models/Qwen3.8-Flash-Next \
    --old-overlay models/qwen38-flash-next-w4b.overlay.hgn \
    --out ~/Models/hq/qwen38-flash-next-w4b.overlay-q8.hgn --embed --check 25
```

32 核上不到 1 分钟。`--q4-keep lm_head` 让 lm_head（248320×2560）保持旧 4-bit：q8 0.63 GiB →
q4cp ~0.33 GiB，arena 与每个 decode token 的读量各少 ~0.3 GiB，代价 KLD +0.006（二分表
lm_head 行），显存或 decode 吃紧时可用。

## 3. imatrix 路由专家（tools/hgn_q4i.py）

q4cp 格式不变（每张量 16 项 fp32 码表、4-bit 码、每 32 列一个 fp16 scale），码表沿用旧文件。
旧量化器：scale = 组内 absmax，取最近码。新量化器（`tools/q4cp_imat.py`）按 llama.cpp
`quantize_row_iq4_nl_impl` 的加权方式逐组搜 scale：

    min Σ_j wt_j (x_j − s·cb[q_j])²,   wt_j = imat_j · sqrt(σ² + x_j²),  σ² = 2·mean(x²)

imat_j 是该专家第 j 个输入通道的平均平方激活（`imatrix_unsloth.gguf_file`，
`blk.N.ffn_{gate,down}_exps.weight.in_sum2 / counts`；count 为 0 的专家用本层均值）。
13 个候选 scale（amax × 0.82…1.18），每个取最近码后再求最小二乘 scale，选最优，再一轮
重分配 + 重拟合，最后用 fp16 舍入后的 scale 重新取码。

抽查（`tools/q4cp_imat_probe.py`，5 个专家，加权相对误差）：L0 gate_up ×1.34、down ×1.51，
L23 ×1.20/×1.21，L47 gate_up ×1.33。旧 blob 与“absmax 量化器复刻”逐位相同（证明对照正确）。

文件布局：头 + 记录表逐字节复制旧基座，每个张量写在旧偏移上；只有 96 个路由专家张量重算，
其余（PLE fp8 表、norm、q4cp dense 回退、MTP 回退）原样拷贝。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=~/Workspace/pylib nice -n 10 python3 tools/hgn_q4i.py \
    ~/Models/Qwen3.8-Flash-Next --old-base models/qwen38-flash-next-w4b.hgn \
    --imatrix ~/Models/BF16/Qwen3.8-Flash-Next-BF16/imatrix_unsloth.gguf_file \
    --out ~/Models/hq/qwen38-flash-next-w4b-imat.hgn --jobs 28
```

32 核约 40 分钟（每层 ~50 s），内存峰值约 20 GiB（page cache 用 fadvise 释放）。
写到 `<out>.part`，结束时检查大小与旧文件相同、抽查层误差确实下降，再改名。

## 4. 结果

KLD：BF16 基准 64 chunk × 512（`data/kld/bf16_c512.kld`）。速度：`tools/hq_verify.sh` 同一轮里
新旧交替测（括号里是同轮旧文件），Strix Halo gfx1151，生产 env。

| 基座 + overlay | KLD | top1 | PPL | MTP commit/round | pp8K@2048 | pp32K@16384 | decode | weight arena |
|---|---|---|---|---|---|---|---|---|
| 旧 `w4b.hgn` + 旧 overlay | 0.1628 | 86.81% | 3.470 | 3.67 | — | — | — | 2.9 GiB |
| 旧基座 + q8 overlay | 0.0627 | 91.91% | — | 3.88（3.67） | 1203（1223） | 1403（1369） | 25.3（30.1） | 5.1 GiB |
| imat 基座 + 旧 overlay | 0.1562 | 86.97% | 3.438 | — | — | — | — | 2.9 GiB |
| **imat 基座 + q8 overlay** | **0.0558** | **92.41%** | **3.329** | 3.76（3.67） | 1173（1099） | 1359（1324） | 24.3（29.0） | 5.1 GiB |
| 一键转换，有 imatrix | 0.0576 | 92.43% | 3.328 | 3.78 | — | — | — | 5.1 GiB |
| 一键转换，无 imatrix | 0.0577 | 92.63% | 3.344 | 3.78 | — | — | — | 5.1 GiB |
| 参考：GGUF UD-Q4_K_XL（本引擎 / llama.cpp） | 0.0511 / 0.049 | | | | | | ~25 | |

- 质量几乎全部来自 8-bit dense（0.163 → 0.063）；imatrix 专家单独只降 0.007，但叠加 8-bit dense
  后再降 0.007，与 GGUF 只差 0.005。
- 一键转换两轮真实权重验证均 PASS（09-30，`convert_verify.sh`）：overlay 与 `hgn_hq.py` 的输出
  逐字节相同，基座张量表与生产 `w4b.hgn` 一致。universal 码表下有/无 imatrix 的 KLD 几乎相同
  （0.0576 / 0.0577，top1 与 PPL 互有胜负，属噪声）——x² 加权已足够，imatrix 变成可选项。
  一键版 MTP sidecar 只含 18 个 q8g64 线性层（生产 sidecar 为 31 个张量），MTP 的 norm 与专家
  回退由基座提供，投机效果相同（commit/round 3.78）。
- prefill 不变（MoE 专家仍是 4-bit，dense 在 prefill 里占比小）；decode −16%：dense 读量翻倍，
  和纯 GGUF（dense 也是 Q8_0）相同。MTP 接受率略升。
- 旧文件在新二进制上逐位不变（hq_verify 第 1 项）。

## 5. 验证

```bash
bash tools/hq_verify.sh                                                   # 只换 overlay
NEWBASE=~/Models/hq/qwen38-flash-next-w4b-imat.hgn bash tools/hq_verify.sh   # overlay + 基座
```

一键打印 PASS/FAIL：①旧文件在新二进制上与 `REF`（fdf805a）逐位相同；②新权重 prefill 与
decode PPL 相差 < 1%；③KLD ≤ KMAX 且比旧低 ≥ 0.08；④MTP commit/round ≥ 0.95×；
⑤速度（decode ≥ 0.7×）；⑥weight arena 增量 ≤ 2.6 GiB。

## 6. 部署

默认配置不变。确认后把两个文件放进 `models/`，在 `service.conf`（hgn 段）改：

```bash
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/qwen38-flash-next-w4b-imat.hgn}"
OVERLAY_FILE="${OVERLAY_FILE-$MODEL_DIR/qwen38-flash-next-w4b.overlay-q8.hgn}"
```

或只在启动时覆盖：`MODEL_FILE=... OVERLAY_FILE=... bash start_hgn.sh`。

一键转换的输出：把 `<name>.hgn`、`<name>.overlay.hgn`、`<name>-mtp.hgn`（和 `-vision.hgn`）放进
`models/`，`MODEL_FILE` / `OVERLAY_FILE` / `MTP_FILE` / `VISION_FILE` 指向它们；或者直接用输出目录
里生成的 `start.sh`（最小启动脚本，已按 base → overlay → mtp 排好）。
kvsnap 指纹含文件路径，换文件后旧快照自动失效。

**Windows**：新 overlay 使 weight arena +2.2 GiB，256K / chunk 8192 估算 91.0 → ~93.2 GiB
（上限 95）。吃紧时用 `--q4-keep lm_head` 版 overlay 或 chunk 4096。
