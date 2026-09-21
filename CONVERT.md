# 转换自有模型(flashnext2hgn)

*English: [CONVERT_EN.md](CONVERT_EN.md)*

`tools/flashnext2hgn.py` 把 HuggingFace safetensors 格式的
Qwen3.8-Flash-Next(qwen4_exp 架构)模型/微调版转换成本引擎的 `.hgn`
权重。单文件脚本,只依赖 Python 3 + numpy(不需要 torch)。

```bash
python3 tools/flashnext2hgn.py /path/to/hf-model --out /path/to/outdir
```

产出(outdir 内):

- `<name>.hgn` — 主权重(q4cp 4-bit 线性层 + bf16 norm/路由/卷积 +
  fp8 PLE 表 + q4cp MTP 兜底),约 116 GiB
- `<name>.overlay.hgn` — 质量覆盖层(可选;启动时作为主权重之后、MTP
  之前的位置参数传给引擎,逐名覆盖主权重张量):723 个稠密线性层
  (注意力投影 / shared expert / hyper-connection / lm_head,不含
  embed_tokens、MoE 专家和 mtp.*)的再量化——12 个全注意力层 o_proj
  升级为 q8g64 8-bit(误差减半),其余 711 个仍为 q4cp 但逐组 scale
  经 MSE 网格搜索优化(优于主权重的 absmax RTN)。约 2.3 GiB
- `<name>.overlay-speed.hgn` — 仅 `--overlay-speed` 时生成:同样 723
  个张量但全部 q4cp(不带 q8g64 o_proj,加载/解码更快)
- `<name>-mtp.hgn` — MTP 8-bit sidecar(q8g64),启动时作为第二个位置
  参数传给引擎即升级草稿头精度
- `<name>-vision.hgn` — 视觉塔(全 bf16),`--vision-tower` 使用
- `tokenizer/` — 从源目录复制的 tokenizer 文件
- `start.sh` — 一键启动脚本(引擎 + OpenAI API,含生产环境变量
  与内存上限看护),转换时自动生成

启动:

```bash
bash /path/to/outdir/start.sh
```

就绪后 API 在 `http://127.0.0.1:8731/v1`。可用环境变量覆盖:
`ENGINE_PORT`(默认 8730)、`API_PORT`(8731)、`MAXCTX`(262144)、
`GAMMA`(MTP 草稿长度,3)、`MEMCAP_GB`(内存上限,86)。
脚本里烘焙的是转换时 `--engine-bin` 指向的引擎目录(默认
`本仓库/build`);引擎挪了位置就用 `--start-script-only --engine-bin
<新build目录>` 重新生成,不用重转模型。

## 说明

- **只支持 qwen4_exp 这一种形状**(48 层 / hidden 2560 / 512 专家 /
  24 头 / vocab 248320 / MTP 1 层 / 视觉 27 层):引擎的维度是硬编码的,
  工具会校验 `config.json`,形状不符直接拒绝。其它的 Flash-Next 微调
  (只动权重不动结构)都可以转。
- **量化无需校准数据**:全部是无数据 RTN,方案细节(四种量化格式、
  张量分文件规则、误差水平)见文末"量化方案"一节。
- 量化按张量多进程并行(`--jobs`,默认满核;大张量自动降并发控内存)。
  12 核 / 32 GiB 内存机器实测:主权重 + MTP + 视觉塔约 3.7 小时,
  overlay 约 1 小时;峰值内存 ~20 GiB,输出约需 121 GiB 磁盘。
- `--skip-vision` / `--skip-ple` / `--skip-mtp-sidecar` / `--skip-overlay`
  可裁掉对应产物;`--overlay-speed` 额外生成全 q4cp 的速度版 overlay;
  `--only-overlay` 只(重)转 overlay(已有主权重时免重跑 116 GiB);
  `--jobs N` 设置量化进程数(默认用满全部核心;每张量独立随机种子,
  任意 N 产出逐位一致);`--dry-run` 只校验映射并打印产出计划;
  `--quant-selftest` 跑量化器数值自检。
- base 文件内含 q4cp 精度的 MTP 权重兜底:不传 sidecar 也能跑投机
  解码,传了 sidecar 草稿质量更好(接受率更高)。

## 量化方案

全部量化都是**无校准数据的 RTN**(round-to-nearest,最近舍入):
不需要跑模型前向,纯 CPU 即可转换,结果可复现。整体精度与 GGUF 的
Q4_K 同级。

### 张量怎么分文件

- **主权重 `<name>.hgn`**:全部 48 层主干。norm / 路由 gate / 卷积 /
  PLE 投影等敏感小算子保留 bf16 直通(`BF16_KEEP` 名单);PLE 配置
  为 i64;PLE ngram 大表为 fp8;其余所有线性层(含 3D 融合专家,按
  `[E*rows, cols]` 平铺)为 q4cp;另含一份 q4cp 精度的 mtp.* 兜底。
- **overlay `<name>.overlay.hgn`**:723 个稠密线性层(注意力投影、
  shared expert、hyper-connection、lm_head;**不含** embed_tokens、
  MoE 专家、mtp.*)的再量化,引擎加载时逐名覆盖主权重。12 个全注意
  力层 o_proj 用 q8g64,其余 711 个用 q4cp_opt。
- **MTP sidecar `<name>-mtp.hgn`**:18 个 mtp.* 张量,全部 q8g64。
- **视觉塔 `<name>-vision.hgn`**:全部 bf16 直通。

### 四种量化格式

| 格式 | 算法 | 用途 |
|---|---|---|
| q4cp | 每张量 16 级 Lloyd 码本(在 32 列组 absmax 归一化空间上拟合,端点锁定 ±1)+ 每组一个 fp16 scale(= 组 absmax);`w = cb[nib] * scale` | 主权重线性层 |
| q4cp_opt | 码本同上不变,但每组 scale 在 0.75–1.25 × absmax 的 21 档网格上搜索权重空间 MSE 最优(允许削掉个别极端值换整体误差) | overlay 稠密层 |
| q8g64 | 每 64 列一组仿射量化:`w = code * scale + min`(fp16 scale/min,uint8 码) | MTP sidecar、overlay o_proj |
| fp8 | e4m3 + 末尾一个 fp32 全局 scale(= absmax/448) | PLE ngram 表 |

### 误差水平与取舍

- RTN 必然有损:q4cp 的相对 RMSE 约 8–10%,但 LLM 权重对这类噪声
  容忍度很高,对下游质量的影响通常在可接受范围。
- q4cp_opt 的 scale 搜索用最大误差换平均误差(实测真实张量 RMSE 降
  ~5%,且因网格含 1.0 档,权重空间 MSE 严格不劣于 absmax RTN)。
- o_proj 从 q4cp 升 q8g64 误差约减半(0.099 → 0.048),是 overlay 最
  主要的增益。
- 曾尝试在 scale 搜索后对码本做二次拟合,实测会发散(朴素 Lloyd 优
  化的是归一化空间误差,忽略了权重空间目标的 s² 加权),已放弃,
  保留"码本不变 + scale 搜索"这一有严格下界保证的方案。
- **官方 halogen overlay 的 q4cp 部分是带校准数据的激活感知量化**
  (GPTQ 类):它优化的是输出误差而非权重误差(部分张量权重空间
  RMSE 反而更差 40%),无校准数据无法逐位复现。如对质量更敏感,
  需要自行引入校准集做激活感知量化,那属于另一个量级的工程。

