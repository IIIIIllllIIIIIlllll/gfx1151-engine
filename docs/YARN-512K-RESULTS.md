# YaRN 512K 实机测试结果(2026-09-30)

机器:局域网 192.168.5.12(MarkPC),AMD Ryzen AI MAX+ 395 / Radeon 8060S
(gfx1151,Strix Halo),122 GB 统一内存,ROCm HIP 7.14.60850 / clang 23。
代码:`codex/yarn-512k` @ 78a41cc + 未提交 YaRN 工作区改动(12 个改动文件 +
新增 src/rope.h、tools/yarn_verify.py 等),同步自 Windows 工作机后已修正
CRLF 行尾(否则 bash 脚本无法执行,后续同步需注意)。

配置:`MAX_CONTEXT=524288 ROPE_FACTOR=2 ROPE_ORIGINAL_CTX=262144
ROPE_BETA_FAST=32 ROPE_BETA_SLOW=1 PARALLEL=1 KV_POOL_TOKENS=0` +
启动器默认 `GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1 KV_PAGED=1`。

## 结论:全部必测项通过

| 必测项 | 结果 |
|---|---|
| 1 启动与内存 | PASS |
| 2 长度阶梯 | PASS(34K→500K 六档) |
| 3 300K+ needle | PASS(三档各 3/3) |
| 4 短文本回归 | PASS(factor1 与 main 基线 KLD≈0) |
| 5 KVSNAP 恢复与隔离 | PASS |
| 6 分页 BF16 | PASS |
| 7 满上下文边界(补测) | PASS(524288 满额/超限 0.1s 拒绝;budget-1/25 精确裁剪) |
| 8 长上下文 PPL(补测) | PASS(350K 深处 PPL 退化 +0.4%,无 NaN/spike) |

## 1. 编译与启动

- `GPU_ARCH=gfx1151 bash build.sh test`:**ALL PASS**(含 ixrope_P517/P13/P8192、
  rope_b、qsa_*、gdn_* 全部用例)。
- `build.sh all`:gdec / gdec-api / 工具链编译成功。
- 启动日志关键行:
  - `RoPE: YaRN factor=2 original_ctx=262144 beta_fast=32 beta_slow=1 attention_scale=1.06931`
  - `[kvpage] paged QSA KV on: 2048 pages x 256 tokens (+1 guard), kv=bf16+btv`
  - `memory[ready]: device 42.67 GiB ... HIP free 10.59/120.00 GiB`
  - `kvsnap: dir data/kvsnap, cap 20.0 GB`
- `/health` 正确暴露 `rope_scaling:{type:yarn, factor:2.0, attention_factor:1.0693147180559945,...}`。
- 两个服务周期的引擎日志 grep `error|warn|illegal|stall|fallback|oom`:**0 行**。

## 2. 长度阶梯(单并发单序列,prompt 唯一化防跨档缓存)

| 档位 | 实际 prompt tokens | cached | 墙钟 | prefill 速度 | decode |
|---|---|---|---|---|---|
| 32K  | 34,794  | 0 | 24.5 s | ~1421 tok/s | ✓ |
| 128K | 141,265 | 0 | 104.0 s | ~1358 tok/s | ✓ |
| 256K | 282,513 | 0 | 220.7 s | ~1280 tok/s | ✓(已超原生 262144) |
| 300K | 331,231 | 0 | 269.7 s | ~1228 tok/s | ✓ |
| 400K | 441,567 | 0 | 371.1 s | ~1190 tok/s | ✓ |
| 512K | 500,191 | 0 | 418.8 s | ~1194 tok/s | ✓(160 tok 含 MTP 投机) |

峰值内存:device 45.8 GiB,进程 RSS 76.6 GB,HIP free 10.9 GiB;无 arena
回退、无 OOM、无非法访问、无 stall。

## 3. needle 定位(每档 3 个 codeword,深度 10%/50%/90%;512K 档 5%/50%/97%)

- 331K:3/3 命中(ZEPHYR/OBSIDIAN/LANTERN 全部原文返回)
- 441K:3/3 命中
- 500K:3/3 命中(97% 深度 ≈ 位置 48.5 万,正常定位)

256K 之后未观察到位置错位或检索退化。

## 4. 短文本回归(2048 token 文本,1020 个计分位,离线 KLD)

三方对比(main 分支 @78a41cc 基线引擎 vs YaRN 引擎 factor1 / factor2):

| 组合 | mean KLD | p999 KLD | same-top | PPL |
|---|---|---|---|---|
| factor1 vs main 基线 | 0.000000 | 0.000049 | 100.000% | 6.274121 vs 6.274112 |
| factor2 vs main 基线 | 0.023384 | 0.367746 | 93.529% | 6.207206 vs 6.274112 |

- **factor 1 在该样本上与未改代码数值近似一致**(p999 KLD 约 5e-5、
  same-top 100%)——未发现默认路径回归，但不证明完整 logits 逐 bit 等价。
- factor 2 在短文本上有可量化偏移(mean KLD 0.023,约为 w4b 量化 KLD 0.0576
  的 40%),greedy top-1 有 6.5% 位置不同;短文本实测输出仍连贯正确(中/英/代码
  三题 factor1/factor2 最终答案一致)。印证"短文本流量保持 factor 1"的部署建议。

## 5. KVSNAP(20 GiB 上限,LRU)

- 阶梯期间快照正常写入,LRU 逐出正常工作(500K 快照约 13.4 GB,主要是 MTP 段)。
- **同配置重启**:启动日志 `kvsnap: index 4 checkpoints, 2080 pages, 2080 mtp
  sections, 14.49 GB` —— SSD 索引恢复成功。
- **重启后命中**:32K 探针 cached=32445/32452(0.3 s);500K prompt
  cached=500184/500191(**5.8 s**,新 prefill 需 419 s)。
- **factor 1 隔离**:切换 ROPE_FACTOR=1 后重放同一探针,cached=0,全量重
  prefill(22.3 s)。RoPE 参数指纹隔离正确,无旧 KV 错命中。

## 6. 分页 BF16

- 全部阶梯 prefill 写入 2048 页池(bf16+btv);needle 全深度命中证明分页读取
  正确;decode/MTP 正常(探针请求 draft acceptance 0.625);两次完整重启 +
  页池复用 + 2080 页从 SSD 恢复快照均正常。`GDEC_QSA_UNION` 未开启(>256K 自动
  禁用逻辑已在代码中,未手动强开)。

## 已记录的非阻塞异常

- **`/cache` 端点恒定超时**:API 侧 `handle_cache` 发 CSTAT,但引擎 wire
  协议(PING/INFO/MEM/GEN/X/KVSYNC)从未实现 CSTAT;引擎空闲时同样复现,
  main 分支代码亦然(端点来自旧提交 d255a7a)。**与 YaRN 无关的既有缺陷**,
  建议删除该端点或在引擎补 CSTAT。
- 远端遗留物:`/home/mark/Workspace/engine-baseline`(main 基线 worktree,
  回归用)、`/tmp/yarn_main.kld`(~1 GB)、`/tmp/yarn_prompts/`、
  `data/kvsnap/` ~15 GB 快照。不需要可删。

## 7. 满上下文精确边界(补测,completions 路径,tok_cli 同 tokenizer 验证长度)

| 用例 | prompt tokens | 结果 |
|---|---|---|
| 超限 ~572K | 572,003(估) | **0.1s HTTP 400**,未触发 prefill,服务无损 |
| 满额 budget=0 | 恰好 524,288 | **0.1s HTTP 400**,干净拒绝 |
| budget=1 | 恰好 524,287 | **恰好 1 个 completion token**,finish=length;MTP 裁剪正确 |
| budget=25 | 恰好 524,263 | **恰好 25 token**,finish=length;draft acceptance 1.000(23/23),链在预算处精确截断 |

注:后两例 prompt 与之前超限用例共享前缀,KVSNAP 命中约 418K 页,实测新鲜
prefill 仅 ~106K token(97s);预算裁剪逻辑与 KV 来源无关,结论有效。
发现:客户端断连不会取消引擎侧 prefill(等待写响应时才感知),属既有行为,
断连请求还产生了一条负值 draft acceptance 统计(-1/6),仅账目瑕疵。

## 8. 长上下文 PPL(补测,离线 --ppl,352K token 全程逐位计分,引擎遇 NaN/Inf 会 exit 1)

同一 2048 token 自然文本(README_EN)放在不同位置,三跑均 rc=0、全部 NLL 有限:

| 运行 | 尾部位置 | tail mean NLL | tail PPL | 相对 |
|---|---|---|---|---|
| factor2 深处 | ~350K | 1.726067 | 5.6185 | +0.4% |
| factor2 浅处 | ~2K | 1.721857 | 5.5949 | 基准 |
| factor1 深处 | ~350K | 1.730439 | 5.6431 | +0.9% |

- **YaRN(factor2)在 35 万位置的 PPL 退化仅 +0.4%,无失真、无 spike、无 NaN**。
- factor1(原生 RoPE 超训练上下文)在此探针上也未崩(+0.9%):本模型是
  GDN+周期性全注意力混合架构,位置外推压力远小于纯 RoPE 模型,所以 needle
  检索(已在 500K 三档 3/3)是比 PPL 更敏感的验证手段。两者结论一致:512K
  YaRN 配置在 256K 之后无可测退化。

## 9. 256K prefill 基准(补测,tools/pp.sh,离线直跑)

token 文件 `data/qsa-oracle/262144.tokens`(131072.oracle × 2 拼接),
maxctx=270336,生产内核组合(BF16 KV + WMMA + BTV),每跑 32 个 8192-token
chunk(已剔除 warmup 行):

| 运行 | 总耗时 | 平均速度 | 逐 chunk min/max | gen 1 |
|---|---|---|---|---|
| 原生(factor 默认) | 190.7 s | 1374.4 tok/s | 1270.8 / 1506.4 | chosen=248046 ✓ |
| YaRN factor=2 | 191.3 s | 1370.3 tok/s | 1259.8 / 1517.6 | chosen=248046 ✓ |

**YaRN 在 256K 的 prefill 开销 ≈0.3%(噪声级)**,两跑生成 token 逐位一致。
离线模式 YaRN 日志行打印正确(`RoPE: YaRN factor=2 ... attention_scale=1.06931`)。

## 10. 应用层配置面(补测代码走读)

YaRN 有四层配置入口,全部生效且互相对得上:

| 层 | 入口 | 校验 |
|---|---|---|
| 部署配置 | service.conf 的 `ROPE_FACTOR/ROPE_ORIGINAL_CTX/ROPE_BETA_FAST/ROPE_BETA_SLOW/ROPE_ATTN_SCALE` | 启动器 regex+范围校验(factor≥1、beta>0) |
| 启动器 | start_hgn.sh / start_gguf.sh / start_win.sh 同名环境变量 | 导出 `GDEC_ROPE_*`,`--check` 可见 |
| 引擎 | `GDEC_ROPE_*` 环境变量 + `--rope-factor` 等 CLI(CLI 覆盖 env) | `rope_config_valid`:非法值 rc=2 清晰报错;factor>1 打印 YaRN 行;KVSNAP 指纹含全部 rope 参数 |
| API | 同一 `GDEC_ROPE_*` 环境 | `/health` 暴露 `rope_scaling`(factor=1 时为 null) |

推荐 512K 生产配置(与交接文档一致):
`MAX_CONTEXT=524288 ROPE_FACTOR=2 ROPE_ORIGINAL_CTX=262144 ROPE_BETA_FAST=32 ROPE_BETA_SLOW=1 bash start_hgn.sh`

**发现的配置缺口(建议补,都不阻塞本次验收):**
1. **无"超限未开 YaRN"告警**:`MAX_CONTEXT=524288` 而 `ROPE_FACTOR=1`(默认)时全链路
   无警告,服务正常启动但 256K 之后是原生 RoPE 越界区,质量静默退化。建议启动器在
   `MAX_CONTEXT > ROPE_ORIGINAL_CTX && ROPE_FACTOR == 1` 时 fail 或至少大字警告;
   引擎侧 factor=1 时也应打印一行 `RoPE: native`(目前完全静默)。
2. 启动器端未校验 `beta_fast >= beta_slow`(引擎会 rc=2 兜底,但报错晚一拍)。
3. `MIN_AVAILABLE_GB` 是扁平阈值,不随 MAX_CONTEXT 估算 KV 池体积(实测
   ~43 KiB/token,见 §11),可以把 `base + maxctx×43KiB` 作为预检下限。
4. YaRN 是引擎级全局配置,无每请求覆盖(合理,但文档应写明"同一服务实例
   只能服务一种 RoPE 配置")。

## 11. 512K 内存占用实测(补测)

启动即占稳态(分页池按 MAX_CONTEXT 预分配):

| MAX_CONTEXT | 页数 | device 显存 | gpu-accessible committed |
|---|---|---|---|
| 131072 | 512 | 26.26 GiB | 91.50 GiB |
| 524288 | 2048 | 42.67 GiB | 107.91 GiB |

斜率 ≈ **42.7 KiB/token**(QSA BF16 KV + MTP KV + 页结构);512K 基线 =
43 GiB 显存 + 65 GiB 权重 mmap,机器需 ≥110 GiB 可用内存(默认
`MIN_AVAILABLE_GB=100` 勉强够用,建议 512K 部署提到 110)。

## 12. OOM 路径(补测)

| 用例 | 结果 |
|---|---|
| `MIN_AVAILABLE_GB=200`(实际 118) | 启动器秒拒:`错误:可用内存 118 GiB,要求至少 200 GiB`,rc=1 ✓ |
| `MAX_CONTEXT=2097152`(2M,KV 池 ~75 GiB 超机器总量) | 引擎 `HIP error out of memory at 40_model.inc:524` 清晰报错退出;启动器立即检出"引擎提前退出"并停服,无 360s 挂起、无残留进程 ✓ |

注:run_capped 的 MemoryMax 管不住 GTT 锁页(脚本注释已声明),真正的
内存纪律靠 MIN_AVAILABLE_GB 预检 + HIP 分配检查,两层都验证有效。

## 13. 并发 KV 池共享(补测,PARALLEL=2 + YaRN 512K + 关快照)

- **逐位一致**:两条不同 needle prompt(~24K token,各 3 深度)串行 vs 并发,
  输出逐位相同(A=True B=True),needle 并发下仍各 3/3;并发墙钟 ≈2× 串行
  (GPU 轮转),符合预期。
- **池超限**:两条 ~311K prompt 并发进 524288-token 共享池:先发(C)444s
  完整完成;后发(D)在池耗尽点被干净中断——引擎日志 `pool dry: aborting the
  later request on slot 1 (1024 pages) for slot 0`,API 返回 HTTP 400 且错误
  信息明确提到并发共享上下文。中断后服务健康,新请求正常处理。
  观察:池耗尽是**被动发现**的(D 白跑了 ~400s prefill 才被拒),可优化为
  准入时按剩余页预估拒绝。

## 14. KV 池准入门禁(补测后修复,51_host_cfg.inc)

> **本节数据属于第一版准入门禁,已被 §15 的作者重写版取代,仅留档。**

问题:共享池(vllm 式)在并发超订时只能在池耗尽点被动中断后发请求——
实测 D 白跑 404s prefill 才被拒,还拖慢先发请求(C 444s)。

修复(引擎侧,~70 行):
- `Slot` 增加 `reserved/pages0` 页预约记账;`slot_admit` 在认领槽位时预估
  `need = ceil((新增 prompt token + min(max_tokens, GDEC_KV_RESERVE_DECODE=4096))
  / 256) + 2(COW 余量)`,供给 = 空闲页 + 空闲槽可逐出页 − 在跑请求的未用预约;
  不足即返回 -2 立即拒绝(API 400),不再排队等待。
- decode 预约封顶 4096,因为 API 对未指定 max_tokens 的请求会传整个剩余
  上下文预算,全额预约会把每个长 prompt 都当成独占全池。
- 中途 starve 路径(丢空闲槽 → 中断后发)保留为硬兜底;预约读数是启发式
  估计(对 GPU 线程的池写入有一两页的竞态窗口),注释已声明。

复测(PARALLEL=2,512K YaRN,池 524288 token):
- 超限场景:D **0.3s 被拒**(日志 `req 2 rejected at admission: needs 1209
  pages, only 839 of 2048 pool pages available`),C 241s 完成(比旧方案快
  ~46%,不再为注定失败的请求浪费 GPU)。
- 回归:并发逐位一致(A/B 均 True,needle 3/3)不变;单槽(PARALLEL=1)
  路径不受影响;`build.sh test` ALL PASS。

## 15. 准入门禁作者重写版复测(2026-09-30 晚,五项判定全 PASS)

背景:原作者审查 §14 的第一版门禁后发现四处风险并重写:
1. worker 线程在 g_sm 下直接读 GPU 侧页表/空闲 deque 有竞态;
2. pages0 记账把 reset 释放的旧页重复计入(旧 400 页槽 → 新请求误拒);
3. 固定 +2 页余量使并行单活跃请求无法声明满池;
4. 预约/记账散在 serve 路径里无法主机测试。

重写:准入纯函数抽到 `src/kv_admission.h`(decode 封顶解析 / 目标页 /
need / available);`slot_admit` 先放 g_sm → 取得 GPU turn → 再取 g_sm
复核取消与槽位,在持有 turn 时读 `slot_pages`;预约 = 整序列目标页
(prompt + min(max_tokens, GDEC_KV_RESERVE_DECODE),按 MAX_CONTEXT 裁剪,
无 +2 余量),在跑槽按 max(预约, 已映射) 扣减;成功准入直接持 turn 进
生成(不二次 acquire),拒绝/取消都释放 turn。新增主机回归
`tools/kv_admission_test.py`(抽取生产代码 + 假模型,七组)。

复测环境:远端全量重编(build/gdec 20:08,EXIT=0);配置
`MAX_CONTEXT=524288 ROPE_FACTOR=2 ... PARALLEL=2 KV_POOL_TOKENS=0
GDEC_KV_RESERVE_DECODE=4096`,BF16 KV + WMMA + BTV。

**主机层**:七组全 PASS(g++);`--tsan` 复跑同样全 PASS(无线程竞态
报告);`build.sh test` 已接入该回归,ktest ALL PASS。

**判定1 一致性/超订**(快照关):A/B 串行 vs 并发逐位一致,needle 各 3/3;
2×311K 超订,后来者 5.3s 在下一 GPU 调度边界被拒(日志
`needs 1207 pages (308914 prompt tokens + 16 reserved decode), only 841 of
2048 pool pages available`),先发 242.1s 完成不被中断,事后小请求正常。
拒绝时机按新版设计是等 8192-token prefill 分块的 yield 点,不再是旧版的
固定 ~0.3s,耗时量级符合预期。

**判定2 两个非空槽的非前缀复用**(引擎 token-ID 协议,
`tools/yarn_slot_reuse.py`):先填满两槽各 102400 token(400 页),再并发
R1(23024 token → 目标 90 页)与 R2(460784 token → 目标 1800 页)。日志
双向证实非空槽非前缀复用:`req 103 -> slot 0 (102415 live tokens)` +
`no live prefix`(首 token 即 mismatch)。90+1800=1890<2048,两请求均
准入并完成(R1 28.8s / R2 400.7s),事后健康检查正常——第一版记账在此
场景会把 reset 释放的 400 页重复计入导致 R2 误拒。

**判定3 并行满池边界**:PARALLEL=2 单活跃请求,budget-1(prompt 恰好
524287)→ 恰好 1 个 completion token;budget-25(524263)→ 恰好 25 个;
exact-full(524288)与 oversize(~572K)均 0.4~0.5s 干净 400。移除 +2
余量后单请求可声明全部 2048 页。

**判定4 RAM 检查点 + SSD 恢复**(RCKPT_MAX=8 KVSNAP_MAX_GB=20):
- consistency / slot-reuse 复跑均 PASS(并发相因命中检查点仅更快,
  输出逐位不变);
- 前缀共享 + 部分页 COW(`tools/yarn_cow_prefix.py`):P1 50000 token
  (第 195 页不满)→ P2 = P1 前缀 + 512 新 token,n_cached=50000,
  prefill 0.67s;P2 生成 16 token 与关快照全新重算**逐位一致**;
- 池压力:2×311K 超订触发 rckpt 逐出(23024 tok/90 页、460784 tok/1800
  页两条)与 kvsnap 20GB LRU 逐出,C 242.3s 完成、D 5.3s 被拒,
  无死锁/非法访问;
- 重启后 SSD 恢复:索引 6 检查点/11.73GB;T2(60272 token)经
  `k_60015` 恢复 60015 token(0.62s),gen 与 RAM 层逐位一致;恢复后再
  超订,C 经 SSD 3.18s 恢复 308907 token/1206 页,D 按 max(预约,已映射)
  =1207 页正确扣减后被拒(841<1207)——恢复页参与准入记账无误。

**判定5 超预约 decode 和取消**(`NEW_BIN=build/gdec bash
tools/conc_verify.sh`,溢出阶段脚本固定 GDEC_KV_RESERVE_DECODE=256):
4 路并发 == 单路串行逐位 PASS(10 请求);4 路控制 PASS(负载下 PING
0ms、排队取消 1.0s、运行中取消 30ms、同连接重复 GEN、断连);池溢出
PASS:O1 A(先)超预约增长,池干硬兜底中断后来者 B(1154 token 后
error),A 与单跑逐位一致;O2 B2 准入即拒(0 token),A2 一致;O3
受害者退出后池恢复,B 独立完成 64 token。套件总评 PASS。

观察(非阻塞):合成退化 prompt 上 draft acceptance 出现负值统计
(-1/16),为既有统计口径现象,不影响正确性。

新增工具:`tools/yarn_slot_reuse.py`(两非空槽非前缀复用),
`tools/yarn_cow_prefix.py`(前缀 COW / SSD chain 恢复:
`--chain` 基线 / `--chain-replay` 重启后 SSD 层验证 / `--fresh` 对拍)。

## 回传格式摘要

```text
GPU / ROCm: Radeon 8060S gfx1151 (Strix Halo), HIP 7.14.60850, 122GB UMA
git revision: 78a41cc + 未提交工作区改动(YaRN + src/rope.h + 作者重写的
  src/kv_admission.h 准入门禁,复测见 §15)
config: MAX_CONTEXT=524288 ROPE_FACTOR=2 ROPE_ORIGINAL_CTX=262144
        ROPE_BETA_FAST=32 ROPE_BETA_SLOW=1 PARALLEL=1 KV_POOL_TOKENS=0
        KV_PAGED=1 BF16 KV + WMMA + BTV(启动器默认)
prompt tokens: 34,794 / 141,265 / 282,513 / 331,231 / 441,567 / 500,191
KV mode / paging: BF16 paged, 2048 pages x 256, bf16+btv;UNION 未启用
prefill time: 24.5s / 104.0s / 220.7s / 269.7s / 371.1s / 418.8s(墙钟含 decode)
decode result: 全部正常;needle 331K/441K/500K 各 3/3;短文本 factor1==main
  (KLD 0.000000, same-top 100%),factor2 短文本 mean KLD 0.0234 / same-top 93.5%
memory / arena fallback: device peak 45.8 GiB,RSS 76.6 GB;无回退/无 OOM
KVSNAP result: 重启恢复 2080 页命中(500K 重放 5.8s vs 419s);factor1 下
  cached=0 隔离正确;LRU 20GB 逐出正常
errors or log excerpt: 引擎日志零告警;/cache 端点 CSTAT 未实现(既有缺陷,
  非 YaRN)
```
