# gfx1151-engine

![Strix Halo — Qwen3.8-Flash-Next](media/strix_banner_21x9_v2.png)

*English: [README_EN.md](README_EN.md)*

在单张 AMD Strix Halo APU(gfx1151)上运行 177B MoE 模型的本地推理引擎,
目标模型为 Qwen3.8-Flash-Next(qwen4_exp 架构)及其同结构微调。

路由专家以 4-bit 量化存放在主机内存,GPU kernel 直读,不需要大显存;常驻内存的权重
约 67–77 GiB(视权重格式,PLE n-gram 表留在磁盘按需读),整机 122 GiB 内存即可提供
256K 上下文。

## 性能实测

开发机器为 **GMK EVO-X2(AMD Ryzen AI Max+ 395,Strix Halo / gfx1151)**,
122 GiB 内存:

| 指标 | 数值 |
| --- | --- |
| Prefill(8K,chunk 16384) | 约 1730–1750 tok/s(10-07 实测) |
| Prefill(128K 上下文) | 约 1650 tok/s(10-07 实测) |
| Prefill(256K 上下文) | 原生约 1590、YaRN factor 2 约 1580 tok/s,开销约 0.5%(10-07 实测) |
| Decode(投机,greedy,γ=4) | 8K 约 51、64K 约 50 tok/s(10-07 真实文本) |
| Decode(不投机) | 8K 约 34 tok/s(10-07 实测,hgn 标准权重;HQ/GGUF 每 token 读量大,约低 16%) |
| 平均功耗 | 约 120 W |
| 瞬时最大功耗 | 约 130 W(爆发持续几秒后回落至 120 W 左右) |

投机 decode 随文本重复度变化很大:上面用真实文本 prompt 测得;高重复内容(代码、
模板文本)chain 起草命中率高,同配置实测可达 60 tok/s 以上;采样(自适应 γ)比
greedy 略低。权重格式(GGUF / hgn)不同数字会有出入;ROCm 7.14 与 10.1 实测无差异。
表中日期均为 2026 年,测试环境 ROCm 10.1、生产配置(PREFILL_CHUNK=16384)。

## 权重格式与质量

引擎支持两种权重格式,服务、API、投机解码完全相同,换启动器即可切换:

| | hgn 标准 | hgn 高质量(HQ) | GGUF UD-Q4_K_XL |
| --- | --- | --- | --- |
| 文件 | 当前默认(`qwen38-flash-next-v2.hgn` + ngram/MTP) | 从原始权重转换,见 [HGN-HQ.md](HGN-HQ.md) | Unsloth 发布,与 llama.cpp 同一份文件 |
| 路由专家 | 4-bit(q4cp) | 4-bit(q4cp,imatrix 加权) | Q4_K / Q5_1 为主 |
| dense(注意力、GDN、shared expert、embed、lm_head) | 4-bit | 8-bit(q8g32 overlay) | 8-bit(Q8_0) |
| KLD vs BF16(越低越好) | 0.163 | **0.0558** | 0.0511 |
| top1 与 BF16 一致 | 86.8% | 92.4% | 92.6% |
| 常驻权重 bpw / 大小 | 4.55 / 66.6 GiB | 4.70 / 68.8 GiB | 5.25 / 76.9 GiB |
| Prefill(8K prompt,chunk 16384) | 约 1730–1750 tok/s | 约相同 | 约相同 |
| Decode(不投机) | 约 34 tok/s | 约 29 tok/s | 约 29 tok/s |
| 启动 | `start_hgn.sh` | `start_hgn.sh`(改 `MODEL_FILE` / `OVERLAY_FILE`) | `start_gguf.sh` |
| Windows | 支持 | 支持(尚未实测) | 不支持 |

- KLD:BF16 基准,wikitext-2 64 chunk × 512,测法与 unsloth / llama.cpp 相同(见 [KLD.md](KLD.md));
  llama.cpp 跑同一份 GGUF 为 0.049。
- 质量差距几乎全部来自 dense 的位宽:dense 改成 8-bit 后 KLD 0.163 → 0.063,imatrix 专家再降到
  0.0558。代价是 decode 每 token 读量增加,慢约 16%(与 GGUF 相同);prefill 不受影响。
- 表中性能两行是 2026-10-07 在 hgn 标准(v2 权重)上的实测;HQ / GGUF 列按上述规律推算
  (prefill 对位宽不敏感,decode 慢约 16%)。
- 常驻权重不含 PLE n-gram 表(hgn fp8 47.7 GiB,GGUF IQ4_NL 26.8 GiB),该表留在磁盘按需读。
  `python3 tools/bpw.py` 按类别复核各文件的 bpw(只读文件头,几秒)。
- 默认配置仍是 hgn 标准。HQ 文件已通过 `tools/hq_verify.sh`;部署方法见
  [HGN-HQ.md](HGN-HQ.md) 第 6 节。

## 特性

- **投机解码 chain**:ngram 优先起草、MTP 兜底,逐轮回退;贪心逐位比对,
  采样按目标分布重采样比对,输出分布与串行一致。默认生效,无需参数;
  草稿长度 γ 按模式自选(greedy 固定 4、采样按接受率自适应),也可用
  `MTP_GAMMA=1-8` 固定。高度重复内容(代码注释、模板文本)实测显著加速。
- **独立 8-bit MTP 草稿权重** sidecar,接受率高于内置 4-bit 草稿头。
- **视觉**:支持图像输入(OpenAI `image_url`),KV 复用跨轮生效。
- **OpenAI 兼容 API**:流式、工具调用、`/v1/chat/completions`。
- **原生 256K / YaRN 512K 上下文**,默认保留原生 RoPE,按需开启 YaRN;
  分页 KV 页池 + 两级 prompt 缓存:消息边界的内存检查点(编辑重发秒回)
  + KV 快照跨重启恢复。配置与共享池语义见下方「YaRN 与共享 KV 池配置」。
- **并发请求**:多条请求共享同一个分页 KV 页池(默认 4 路、总共 256K,
  类似 llama.cpp 的共享上下文),GPU 按请求轮转,每条输出与单独运行逐位
  一致;长 prompt prefill 分段让出 GPU,其它会话最长卡顿约 0.6 s。配置与
  语义见 [CONCURRENCY.md](CONCURRENCY.md)。
- **两种权重格式**:自有 `.hgn`(Linux / Windows)与 llama.cpp 的 GGUF
  (Unsloth UD-Q4_K_XL,Linux),对比见上。
- **模型转换工具**:HF safetensors → `.hgn`,默认输出高质量版(8-bit dense overlay
  + 加权 4-bit 专家);有 llama.cpp 格式的 imatrix 就用,没有也能转。可用于自己的
  同架构微调模型(见 [HGN-HQ.md](HGN-HQ.md)、[CONVERT.md](CONVERT.md))。

## 要求

- Linux + ROCm(HIP 7.x),GPU 架构 `gfx1151`;或 Windows + AMD 显卡驱动
  (GPU 需 BIOS 划分显存),见「Windows」一节
- 可用内存 ≥ 100 GiB(权重锁页:hgn 约 68 GiB、GGUF 约 80 GiB,另加 KV)
- 编译依赖:rocBLAS、hipBLASLt、rocPRIM;API 前端另需
  libpng、libjpeg、libwebp（nlohmann/json 已随仓库提供）

## 快速开始

```bash
bash build.sh        # 编译全部所需产物,输出在 build/
bash start_hgn.sh    # hgn 权重:加载模型并启动服务(读取 service.conf)
bash start_gguf.sh   # 或 GGUF 权重(Unsloth UD-Q4_K_XL,与 llama.cpp 同一份文件)
```

两个启动器都从 `./models` 读取权重,缺文件时列出缺失项并退出;配置集中在
`service.conf`(hgn、GGUF 各一段)。GGUF 见 [GGUF.md](GGUF.md)。
详见 [QUICKSTART.md](QUICKSTART.md)。

独立性能测试工具见 [BENCHMARK.md](BENCHMARK.md)。编译脚本不带参数时会
一次性编译引擎、API 前端和 benchmark；Windows 还会编译原生启动器。测试
程序本身的输出使用英语，文档仍以中文为主。

自有微调模型(HF safetensors,同架构)转换:

```bash
# 没有 imatrix
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models
# 有 imatrix(llama.cpp 格式,GGUF 或旧版 imatrix.dat)
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models --imatrix /path/to/imatrix.gguf
```

输出基座 `.hgn`、8-bit dense overlay、8-bit MTP 草稿、视觉塔、分词器,以及可直接运行的
`start.sh`;32 核约 1.5 小时,磁盘约 125 GiB,只依赖 numpy。`--classic` 为旧的无数据转换器
(输出与以前逐字节相同)。高质量转换见 [HGN-HQ.md](HGN-HQ.md),格式与旧转换器见
[CONVERT.md](CONVERT.md)。

## YaRN 与共享 KV 池配置

**先分清三个独立概念:单条请求的上下文上限、整个服务的 KV 池容量、RoPE 缩放。**
本节 `K` 按 1024 token 计:256K = 262144,512K = 524288。上下文包含输入与
生成输出,并包括模板、历史消息和图像对应的 token,不只是用户最后一条消息。

### 原生与 YaRN 的区别

| | 原生 RoPE | YaRN factor 2 |
| --- | --- | --- |
| 定位 | 模型原生 256K 范围,默认模式 | 将单条上下文扩展到 512K 的推理配置 |
| 位置编码 | `ROPE_FACTOR=1`,保持原生频率与默认幅度 | 调整 RoPE 频率,并对主注意力 Q/K 应用幅度缩放 |
| 单条上限 | 配置 `MAX_CONTEXT=262144` | 配置 `MAX_CONTEXT=524288` |
| 短文本行为 | 作为默认基线 | 静态 YaRN 在短文本上也生效,输出分布可能改变 |
| KV 存储 | 由 KV 类型和池容量决定 | 不压缩 KV,不自动扩大共享池 |

`factor` 是位置编码的扩展倍率,不是并发数、KV 压缩率或自动分配的容量。
只改 `MAX_CONTEXT` 不会自动开启 YaRN,只改 `ROPE_FACTOR` 也不会自动改上下文上限。
只需要更多原生短/中上下文并发时,扩大共享池即可,不必开启 YaRN。

单条上限不得超过位置编码能力:YaRN 开启时 `MAX_CONTEXT` 不能超过
`ROPE_FACTOR × ROPE_ORIGINAL_CTX`,超过时启动器和引擎都拒绝启动;
`ROPE_FACTOR=1` 时超过原生 256K 只告警(超出部分的位置编码未验证)。
并发要更多 KV 容量时扩 `KV_POOL_TOKENS`,不要拉伸 `MAX_CONTEXT`。

RoPE 配置是**实例级**的:factor 2 下所有请求都使用 YaRN,不是超过 256K 才临时
切换;同一实例不能让一条请求用原生、另一条用 YaRN。每条序列仍有独立位置,
两条 200K 请求不会拼成一条 400K 序列。短文本优先保持原生;需要同时提供两种
模式时,使用独立实例(检查内存和端口)或停服后切换配置。

### service.conf 参数

编辑已有配置行,不要在文件末尾重复追加。保留 `${变量:-默认值}` 写法时,同名
环境变量优先;修改文件后需要重启引擎和 API 才生效。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `MAX_CONTEXT` | `262144` | 每条请求的输入 + 输出上限,不是所有并发请求的合计;不得超过 `ROPE_FACTOR × ROPE_ORIGINAL_CTX`(超过拒绝启动;factor 1 时超原生值只告警) |
| `ROPE_FACTOR` | `1` | `1` 为原生;512K 配 `2`,必须同时改 `MAX_CONTEXT` |
| `ROPE_ORIGINAL_CTX` | `262144` | 模型原生长度;开 YaRN 后仍保持此值,不要改成 `524288` |
| `ROPE_BETA_FAST` / `ROPE_BETA_SLOW` | `32` / `1` | 频率混合边界,通常不改;引擎要求 `beta_fast >= beta_slow > 0` |
| `ROPE_ATTN_SCALE` | `0` | `0` 自动取幅度系数:原生为 1,YaRN 为 `1 + 0.1 * ln(factor)`;factor 2 约 1.0693,正值表示手工覆盖 |
| `KV_PAGED` | `1` | 分页共享池;`PARALLEL>1` 必须开启,并使用支持分页的 kernel 组合 |
| `KV_POOL_TOKENS` | `0` | 全部槽位共享的物理页池容量,以 token 为单位;`0` 跟随 `MAX_CONTEXT`,小于它也按它算,向上取整到 256 token/页 |
| `PARALLEL` | `4` | 并发槽位数,范围 1–8;不自动乘大池容量,也不把单条上限等分 |
| `GDEC_KV_RESERVE_DECODE` | `4096` | 引擎环境变量,控制准入时最多预约多少 decode token;不是输出上限,有效值为 0–2147483647 |

脚本启动器把 `ROPE_*` 导出为同名的 `GDEC_ROPE_*`,让引擎和 API 使用相同设置。
直接运行引擎时可用 `--rope-factor`、`--rope-original-ctx`、`--rope-beta-fast`、
`--rope-beta-slow`、`--rope-attn-scale`;直接运行 API 则使用 `GDEC_ROPE_*`。
手工启动要确保两侧配置一致;只改 API 的 `/health` 元数据不会改变引擎的 RoPE。

### 两种 512K 配置

`MAX_CONTEXT` 只表达单条上限,并发容量一律用 `KV_POOL_TOKENS` 表达;
`MAX_CONTEXT` 超过 `ROPE_FACTOR × ROPE_ORIGINAL_CTX` 的配置会在启动时被拒绝。

| 使用模式 | `MAX_CONTEXT` | `KV_POOL_TOKENS` | `ROPE_FACTOR` | `PARALLEL` 示例 |
| --- | --- | --- | --- | --- |
| 默认原生 256K 池 | `262144` | `0` | `1` | `4` |
| 原生 256K 单条,512K 总池 | `262144` | `524288` | `1` | `2` |
| YaRN 512K 单条,512K 总池,单并发 | `524288` | `0` 或 `524288` | `2` | `1` |
| YaRN 512K 单条,512K 总池,多并发 | `524288` | `0` 或 `524288` | `2` | `2` |

**模式 A:保持原生,只是扩大共享池。** 两条序列动态共享 512K,每条仍不得超过
256K,不是让单条请求获得 512K。替换 `service.conf` 中对应行:

```bash
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
KV_POOL_TOKENS="${KV_POOL_TOKENS:-524288}"
ROPE_FACTOR="${ROPE_FACTOR:-1}"
KV_PAGED="${KV_PAGED:-1}"
PARALLEL="${PARALLEL:-2}"
```

**模式 B:YaRN 扩展单条到 512K。** 池仍总共 512K;下面开启两槽共享,若希望长请求
单路运行,把 `PARALLEL` 改为 `1`。替换对应行:

```bash
MAX_CONTEXT="${MAX_CONTEXT:-524288}"
KV_POOL_TOKENS="${KV_POOL_TOKENS:-0}"
ROPE_FACTOR="${ROPE_FACTOR:-2}"
ROPE_ORIGINAL_CTX="${ROPE_ORIGINAL_CTX:-262144}"
ROPE_BETA_FAST="${ROPE_BETA_FAST:-32}"
ROPE_BETA_SLOW="${ROPE_BETA_SLOW:-1}"
ROPE_ATTN_SCALE="${ROPE_ATTN_SCALE:-0}"
KV_PAGED="${KV_PAGED:-1}"
PARALLEL="${PARALLEL:-2}"
```

不改文件也可临时覆盖,先检查再启动(去掉 `--check`):

```bash
MAX_CONTEXT=524288 KV_POOL_TOKENS=0 KV_PAGED=1 PARALLEL=2 \
ROPE_FACTOR=2 ROPE_ORIGINAL_CTX=262144 ROPE_BETA_FAST=32 \
ROPE_BETA_SLOW=1 ROPE_ATTN_SCALE=0 bash start_hgn.sh --check
```

Linux GGUF 使用相同变量,换成 `start_gguf.sh`;已完成的 512K 实机验收使用 Linux
hgn + BF16 分页 KV + WMMA + BTV,不代表所有权重/kernel/平台组合都已验证。
Windows 的 `start_win.sh` 与原生 `start_win.exe` 都会读取 `service.conf` 的
`ROPE_*`、做同样的校验并转为 `GDEC_ROPE_*` 传给引擎和 API,但 Windows 512K
尚未实测验收,且需另行确认 arena(95 GiB 上限)/显存能否放下 512K 页池。

### 两个典型场景(参考)

**场景 1:不开 YaRN,2 路并发各自可用满原生 256K。** 对应上方模式 A:
`MAX_CONTEXT=262144` + `KV_POOL_TOKENS=524288` + `PARALLEL=2` + `ROPE_FACTOR=1`。
直觉上"设 512K 让两个并发各吃 256K"的正确表达是:单条上限保持原生 256K 不动,
把 512K 给共享池。两条序列动态共享 512K,单条超过 256K 仍被拒绝。

**场景 2:开 YaRN,2 路并发共享 512K 池。** 对应上方模式 B:
`MAX_CONTEXT=524288` + `ROPE_FACTOR=2` + `KV_POOL_TOKENS=0` + `PARALLEL=2`。
两条请求合计超过 512K 时,后来的请求在准入阶段被拒(API 返回错误,日志
"rejected at admission"),先来的请求不受影响;想让 512K 长请求不被打扰,
把 `PARALLEL` 改为 `1`。

### 并发怎样占用共享池

`PARALLEL` 只规定同时活跃的槽位数。池大小在启动时预分配,不会随并发数自动增长,
也不是每槽固定分到 `池容量 / PARALLEL`。一个槽可以用到单条上限,其他槽用剩余
页;槽位用满时请求排队,槽位可用但 KV 预算不足时,多并发准入门禁会拒绝后来者。
GPU 在安全调度点轮转,并发数翻倍不意味着吞吐翻倍,请求延迟也可能增加。

例如 `MAX_CONTEXT=524288, KV_POOL_TOKENS=0, PARALLEL=2, ROPE_FACTOR=2`:

| 请求情况 | 结果 |
| --- | --- |
| 单条接近 512K(输入 + 输出),另一槽空闲 | 可以使用整池,不会因两槽而把单条上限减半 |
| 两条各约 200K prompt | 加上 decode 预约后仍能放下时,都可准入 |
| 两条各约 300K prompt | 合计超过 512K,后来者在下一安全调度边界被拒,不会完整 prefill 后才发现 |
| 准入时够用,decode 后来增长超出预约 | 仍可能池耗尽,触发逐出/中断兜底;准入不是无限生成保证 |

准入按 256 token/页计算,并对活跃序列保守记账:

```text
目标页数 = ceil(min(MAX_CONTEXT,
                   prompt_tokens + min(max_tokens, GDEC_KV_RESERVE_DECODE)) / 256)
其他活跃槽的预算 = max(目标页数, 当前实际映射页数)
可用预算 = 池总页数 - 其他活跃槽预算之和
新请求需页 = 目标页数; live continuation 则取 max(目标页数, 已映射页数)
```

- 非前缀复用会 reset 旧槽,释放的旧页不再作为新请求的增长基线重复扣除。
- 缓存命中减少 prefill 工作,不等于已有 KV 不占页;活跃序列即使共享前缀,
  准入仍分别记账,不依赖共享省出的物理页允许超订。
- 空闲槽和 RAM 检查点钉住的页也会占物理池,COW 可能需要额外页。门禁把可回收
  页视为可逐出,不把所有缓存永久扣减;真正分配时仍保留压力处理。
- 池耗尽时先逐出 RAM 检查点,再释放空闲槽 KV,仍不足则中断较晚准入的活跃请求;
  需要页的请求若自身是后来者,它可能失败。被中断请求通过 API 返回错误。
- `GDEC_KV_RESERVE_DECODE=4096` 不会把输出裁到 4096;输出仍受请求预算和
  `MAX_CONTEXT` 限制。减小预约可能提高准入率,但增加中途池耗尽风险。
- `PARALLEL=1` 绕过多槽准入门禁,仍受单条上限、实际池容量和缓存逐出机制约束。

想让两条完整 512K 序列同时驻留,需要至少把 `KV_POOL_TOKENS` 设为 `1048576`,
再核算检查点/COW 与其他内存;这不是默认 512K 池的能力,也不在本轮实机验收范围。
增加池容量主要增加 KV 存储,增加槽位还增加 GDN 等每序列状态;相同总池也不保证
不同 `MAX_CONTEXT`/`PARALLEL` 配置的总内存完全相同。

### 启动检查、缓存与验证范围

- 512K 使用 BF16 KV + WMMA + BTV;Linux 启动器已设置此组合。超过原生 256K
  时引擎自动禁用 `GDEC_QSA_UNION`,不要手工强制使用未验证组合。
- 检查 `--check` 的单条上下文、共享池和并发数;服务起来后检查 engine 日志的
  `RoPE: YaRN factor=2`、`[kvpage]` 页数及 KV 类型,以及 `/health` 的
  `context=524288` 和 `rope_scaling.factor=2`。默认原生返回 `rope_scaling=null`;
  API 的上下文以引擎 INFO 为准,不要仅凭 API 命令行判断生效。
- 512K 实测需约 108 GiB 的 gpu-accessible committed 内存(Linux hgn 测试配置),
  不应把 KV 数组大小当成整机预算,也不要把 UMA 的 RSS 与 device 数值直接相加。
  扩池前检查权重、KV/BTV、工作区和系统余量;内存紧张时不要只提高并发/池容量。
- RoPE 参数进入 SSD KV 快照指纹。切换原生/YaRN 或其参数需要重启,不同指纹不会
  错用旧 KV;无需为正确性删除快照,但旧文件仍可能占磁盘配额并被 LRU 逐出。
- 2026-09-30 Linux/gfx1151 实测覆盖 500K prefill/needle、精确 512K 边界、
  factor 2 双槽并发、reset/COW/SSD 恢复、超预约 decode 及取消。256K prefill
  原生 1589.4 vs YaRN 1582.1 tok/s(2026-10-07 复测,约 0.5% 差异),不表示
  任意负载都无开销。
  短文本探针 factor 2 与原生 mean KLD 0.0234、same-top 93.5%,因此不承诺两者
  输出一致。详细实现/限制见 [YARN-512K.md](YARN-512K.md),实测见
  [YARN-512K-RESULTS.md](YARN-512K-RESULTS.md)。

## Windows

Windows 版与 Linux 版功能一致(引擎 + OpenAI API + 多模态),移植记录与
实测见 [PORTING-WINDOWS.md](PORTING-WINDOWS.md)。编译在 Git Bash 中执行
(或双击 `build_win.bat`,仅编译期需要 Git):

```bash
bash build_win.sh           # 全部所需产物:引擎、benchmark、API、启动器
bash build_win.sh api       # OpenAI API 前端
bash build_win.sh launcher  # 免脚本启动器 start_win.exe
```

日常运行双击 `start_win.exe`(原生 Win32,不需要 Git/PowerShell):拉起引擎
+ API 双进程,不开控制台,只在任务栏右下角放托盘图标(右键:打开面板 / 复制
API 地址 / 查看日志 / 退出;双击:打开面板),输出写入 `logs\`;排查问题可用
`start_win.exe --console` 回到控制台模式(Ctrl+C 或关窗停止)。首次双击会弹出
启动配置面板(4 个分页):第 1 页权重(V1/V2/GGUF 版本选择,GGUF 仅保存供
Linux `start_gguf.sh` 用)、第 2 页上下文/并发/YaRN 勾选(扩展倍数按单请求
上限自动推导)、第 3 页监听地址与端口、
第 4 页显存环境检查,每页有独立的检查报错区,确认后写回 `service.conf` 并启动;之后双击不再弹出,
改配置用托盘右键"设置…"或 `start_win.exe --setup`。配置与 Linux
**共用 `service.conf`**(换模型文件名、改上下文窗口都编辑它),环境变量可
临时覆盖。客户端连 `http://<主机>:8731/v1`。

分发:`build/` + `start_win.exe` + `models/` 拷到任意 gfx1151 Windows
机器即用,**无需安装 ROCm/TheRock**;仅需 AMD 显卡驱动,并在 BIOS 为 GPU
划分足够显存(256K 上下文需 96 GiB)。

与 Linux 版的差异:

- 只支持 hgn 权重:Windows 下可用显存上限约 96 GiB,GGUF 权重体积更大
  (hgn 比 GGUF 省约 11 GiB)放不下,`start_gguf.sh` 不适用;hgn 权重由
  转换工具生成,见 [CONVERT.md](CONVERT.md)。高质量 hgn 换文件即可用:
  weight arena +2.2 GiB,256K / chunk 8192 估算约 93.2 GiB(上限 95),
  尚未在 Windows 实测
- 图片解码经 stb_image 支持 PNG/JPEG(WebP 未接)
- prefill chunk 默认 8192
- 冷加载为整权重读盘(分钟级,进度见控制台/日志)
- 启动器未开 `GDEC_GEMM_WMMA` 与 `GDEC_GDN_FUSED`(Linux 启动器已转正的
  自写 WMMA GEMM 与 GDN 融合 kernel,合计约 8-10% PP,TheRock 下未验证——
  故 Windows 端 prefill 走 hipBLASLt + 旧 GDN 路径)

已知问题(原因均未查明;疑难杂症较多,待解决,优先级很低):

- 显存分配超过 41 GiB 或 63 GiB 时失败
- 模型 decode 过程中卡死(疑为控制台输出反压:控制台被点选暂停后,子进程写日志
  阻塞。已修复:启动器改为托盘程序、日志不经过控制台,kvsnap 不再持锁打印;待验证)

编译细节见 [BUILD.md](BUILD.md)。

## 文档

- [QUICKSTART.md](QUICKSTART.md) — 编译、启动、配置
- [BUILD.md](BUILD.md) — 编译环境细节与排错
- [GGUF.md](GGUF.md) — GGUF 权重加载、与 hgn 的性能对比
- [HGN-HQ.md](HGN-HQ.md) — 高质量 hgn:一键转换(可选 imatrix)、结果与部署
- [CONVERT.md](CONVERT.md) — 模型转换工具
- [KLD.md](KLD.md) — 质量测试(KLD,与 unsloth / llama.cpp 同口径)
- [MTP.md](MTP.md) — 投机解码参数与对比方法
- [NGRAM.md](NGRAM.md) — ngram 验证的设计、收益与已知分歧
- [CONCURRENCY.md](CONCURRENCY.md) — 并发请求(PARALLEL)的配置与语义
- [YARN-512K.md](YARN-512K.md) — YaRN 512K 实现、共享池准入与验证范围
- [YARN-512K-RESULTS.md](YARN-512K-RESULTS.md) — Linux/gfx1151 长上下文与并发实测
- [HGN-FORMAT.md](HGN-FORMAT.md) — `.hgn` 权重容器格式
- [GGUF.md](GGUF.md) — 直接用 llama.cpp GGUF 权重运行
- [data/README.md](data/README.md) — 数值回归基准(data/qsa-oracle)说明
- [PORTING-WINDOWS.md](PORTING-WINDOWS.md) — Windows 移植记录与实测

## 测试

```bash
bash build.sh test     # kernel 单测,不加载模型,预期 ALL PASS
python3 tools/bpw.py   # 统计 models/ 下各权重的 bpw(按类别,只读文件头)
```

质量(KLD)测试需要 BF16 基准,流程见 [KLD.md](KLD.md)。

## 致谢

本项目的实现方式借鉴了 peonist-ai 的 [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server);`.hgn` 权重容器格式即 halogen 的 checkpoint 容器格式(见 [HGN-FORMAT.md](HGN-FORMAT.md))。感谢 halogen 作者的工作。

GGUF 支持大量参考了 [gufo](https://github.com/gufo-org/gufo)(MIT 许可证):路由专家 F16 WMMA GEMM kernel 移植自其 RoutedF16GEMMKernel(`src/gpu/parts/26_kernels_moe_gguf.inc`),hgn q4cp / GGUF IQ4 的 LUT 解码版沿用同一条流水线(`27_kernels_moe_lut.inc`),GGUF 与引擎张量间的变换语义参考其 reference.cpp(`src/gguf_map.h`);prefill 的 HC 门控融合、生产者 epilogue 直写下一 GEMM 输入等优化也借鉴了 gufo 的做法(对照分析见 [GUFO-GAP.md](GUFO-GAP.md))。

ngram 投机解码的起草思路与两级 prompt 缓存借鉴了开源项目 [llama.cpp](https://github.com/ggml-org/llama.cpp)(MIT 许可证),GGUF 权重与视觉塔(mmproj)直接复用 llama.cpp 的同一份文件;ngram 声明详见 [NGRAM.md](NGRAM.md) 的「来源与声明」一节。

## 许可证

[AGPL-3.0](LICENSE)
