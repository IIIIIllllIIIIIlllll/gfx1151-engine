# Ngram verify / GDN 候选优化验证（2026-09-15）

## 2026-09-19 更新：分块 verify（修复 ngram 轮输出卡顿）

**问题**：ngram 轮把 ≤64 token 草稿作为一次 65 行原子 verify（~350ms），
部分接受时还要整段重放已接受前缀（~150ms）；客户端在这段时间内收不到
任何 token，生产日志中表现为 ~1.5s 的流式空洞（"卡顿"）。

**修复**：分块 verify + 自适应块大小。

- 草稿切成 ≤ng_cs 行的块逐块 verify，每块接受即发射（流式平滑）；块内
  拒绝只 restore+replay 该块前缀（上限 ng_cs 行），取代 65 行原子
  verify + 按接受量整段重放。
- 块满接受时检查续接 token 是否等于下一个草稿 token（拼接检查）：相等
  续链下一块；不等视为边界拒绝，状态干净、无需重放，直接结束本轮。
- ng_cs 初始 16，全接受轮翻倍（上限 64），拒绝轮减半（下限 8），作为
  Model 成员跨请求持久。
- env `GDEC_NGRAM_CHUNK`：不设 = 自适应（默认）；`0` = 旧单趟路径
  （原代码保留作对照/回滚）；`N` = 固定块大小。
- 贪心和采样两条 chain 均实现。`CHUNK=64` 固定单块可复现 legacy 的
  全部数字，证明 round 重写本身不改变语义。

**A/B（heretic 模型，贪心，同 prompt，输出逐字节一致）**：

- 代码场景（写快排→加注释，原卡顿场景）turn2：legacy 14.74s /
  36.4 tok/s / p99 gap 431ms / max 592ms → 分块 13.25s /
  40.4 tok/s（+11%）/ p99 191ms / max 441ms。turn1（纯 MTP）不变。
- 重复文本 showcase：legacy 6.58s / 57.1 tok/s；分块热态 6.80s
  （冷启动首个请求 7.52s，自适应从 16 爬坡），接受率 0.767 全同。
- 采样链（temp 0.7 top_p 0.9）跑通，接受率 0.750，输出连贯。

**回归**：ngram_regress ALL PASS；drafter_switch 全过；
snapshot_regress 8/9（cmp4-2 为既有 DIFF）；五组 oracle 全 MATCH；
ktest ALL PASS。

## 2026-09-18 更新：索引 90 分歧根因已定位（近平局，非 bug）

对编号压力用例（prompt 344 / gen 128）的定量结论：

- **MTP 在同一位置同样分歧**：串行 / NGRAM / MTP 三路对比，NGRAM 与 MTP
  在索引 90 同时偏离串行，且两者 128 个 token 逐位相同。分歧与 ngram
  草稿机制无关，是"批量 verify 数学 ≠ 串行 decode 数学"的共性。
- **该位置是近平局**：串行路径 top-2 logprob 为 -0.9553 vs -1.0417
  （差 0.086 nat，概率比 ~1.09）。bf16 中间激活 + 不同累加顺序的扰动
  足以翻过这个差距。
- **大块批量预填与串行同侧**：把 prompt+前 90 个生成 token 一次性
  prefill（P=434 单 chunk）后 argmax 仍是串行选的 3294（gap 0.328 nat）。
  即同一批量数学在不同 chunk/偏移形态下落在平局两侧，属硬币面。
- 结论：编号用例的两次对拍失败不是状态错误（此前的卷积状态 bug 是真
  bug，已修且有 ktest 覆盖），而是投机理应当前允许的近似语义在近平局
  上的正常表现。cmp4-2 的既有 DIFF 是同类现象。
- **ngram_regress.py 判据已更新**：编号用例改为双 gold——与串行一致，
  或与 MTP drafter 输出一致（两者共享主干批量 verify；ngram 专有的
  状态/回滚错误会使两者脱钩）即判通过。其余用例仍严格对拍串行。
- 对 MTP 的含义：MTP 的"与串行逐位相同"是经验结论而非保证，在
  tie-heavy 文本上（如本用例 128 token）同样会翻。README 级表述
  仍成立的只是"投机不改变分布、只提议不强行接受"。

## 本轮改动

- `k_gdn_verify` 将短批次 GDN 状态保留在寄存器中，连续处理验证行，避免按 64-token chunk 填充以及反复读写整个状态。只用于 ngram 的 verify/replay，最多 65 行。
- verify/replay 不再额外计算末行 lm_head 和同步读取 argmax；验证完统一计算所有行 logits。lm_head 按最多 8 行分组复用权重。
- 修正历史重复追加、回滚时 pos/PLE 环位置、跨请求哈希学习状态、生成预算和上下文边界。主干 checkpoint 和 argmax 分配独立于 MTP 权重。
- 修正已有 `k_convst_update` 的短批状态错误：P=1 应保留旧历史的第 1、2 行，P=2 应保留第 2 行；原代码把最老保留行写反。此前的干净源码也有此错误。
- 将短批状态交接测试加入 `tools/ktest.cu`，验证完整历史状态及下一 token 的卷积输出。

该卷积错误不是单纯的浮点微差：修复前 P=1、2 都有 10,240 个状态值与顺序执行不同，下一步输出最大绝对误差分别约 1.833、1.824。修复后 P=1/2/3/4/8/9/16/17/32/33/64/65 的状态及下一步输出全部精确一致。

## 实测收益

设备 gfx1151，模型 `qwen38-flash-next-w4b.hgn` + overlay，单引擎、greedy、关闭磁盘 KV 快照。以下时间仅为 decode，不含 prompt prefill；生成数包含首 token，因此按时间换算 TG 时使用生成数减一。

最终源码、maxctx=40960、prefill chunk=8192，ngram match=24/min=4/max=16：

| 用例 | 串行 decode | Ngram decode（两次） | 输出对拍 |
|---|---:|---:|---|
| 240-token 重复文本，生成 64 | 1.9930 s | 0.8573 / 0.8410 s | 两次一致 |
| 137-token 重复代码，生成 48 | 1.4848 s | 0.8703 / 0.8622 s | 两次一致 |
| 普通问题，生成 32 | 0.9853 s | 0.9782 / 0.9783 s | 一致；无草稿，串行回退 |
| 编号重复段落，生成 128 | 4.0445 s | 4.2117 / 4.1919 s | **两次都在索引 90 分歧** |

重复文本约 2.35 倍、重复代码约 1.71 倍，不能外推到普通问答。编号用例接受不足，验证和回滚使总时间反而增加。

此前同为 max=16 的结构 A/B：恢复旧 GDN 和冗余末行 head 后为 0.9031/0.8949 s，优化路径约 0.841–0.850 s，独立结构收益约 6%。从 max=8 增至 16 的额外收益属于批量摊销，不能全部算作新 GDN 内核收益。该 A/B 发生在本次短卷积修复之前。

隔离 GDN 测试中，P=9 从约 0.1900 ms 降至 0.0237 ms（约 8 倍）；P=17 为 0.1949→0.0342 ms；P=65 为 0.3508→0.1003 ms。每行输出和最终状态对照 `k_gdn_step`，测试的 11 个批次长度均为零绝对误差。隔离加速比不等于整模型加速比。

默认 match=24/min=48/max=64 已补齐整模型测试，同样 maxctx=40960/chunk=8192：

| 用例 | 串行 decode | Ngram decode（两次） | 输出对拍 |
|---|---:|---:|---|
| 重复文本，生成 64 | 1.9936 s | 0.3095 / 0.3012 s | 两次一致 |
| 重复文本，生成 130 | 4.0849 s | 0.6348 / 0.6278 s | 两次一致；覆盖完整 65 行 verify |
| 重复代码，生成 48 | 1.4838 s | 0.4359 / 0.4298 s | 两次一致 |
| 编号段落，生成 128 | 4.0595 s | 4.6494 / 4.6139 s | 两次都在索引 90 分歧 |

默认长草稿下，高重复文本 decode 约 6.5 倍、重复代码约 3.4 倍。这包含更长批量对权重读取和执行开销的摊销，不能归因于单独的 GDN 优化，也不代表实际代码任务普遍能达到这个速度。默认配置共完成 39 个请求，除编号用例两次对拍失败外，其余检查通过；测试总体退出状态仍为失败。

最终源码同为 min=48/max=64 的旧路径对照（`GDEC_NGRAM_LEGACY_GDN=1 GDEC_NGRAM_LEGACY_FINAL=1`）：

| 用例 | 旧 GDN + 旧末行 head（两次） | 新路径（两次） | decode 平均耗时下降 |
|---|---:|---:|---:|
| 重复文本，生成 64 | 0.3265 / 0.3188 s | 0.3095 / 0.3012 s | 约 5.4% |
| 重复代码，生成 48 | 0.4453 / 0.4380 s | 0.4359 / 0.4298 s | 约 2.0% |

这组对照的 token 对拍均通过，串行对照的 decode 也基本稳定；但每种配置仅两次测量，应将百分比视为当前用例的估计。可确认大幅收益主要来自 ngram 长批量验证，新 GDN/末行 head 消除带来额外的小幅收益。

## 正确性与限制

- 最终 `ktest` 全过，包括新加入的短卷积状态交接测试（2026-09-18 复跑
  仍 ALL PASS，含 convstate_handoff P1..P65 共 12 项）。
- 五个现有 oracle（2048、2051、2052、8192、32768）全部 MATCH
  （2026-09-18 复跑确认）。
- cmp4 为 3/4；第 2 个仍是已有 DIFF（与索引 90 同类的近平局，
  见顶部 2026-09-18 更新）。
- 2026-09-18 复跑：ngram_regress 在 helper 配置（min=4/max=16）和
  引擎默认配置（match=24/min=48/max=64）下均 ALL PASS，编号用例
  两次运行均匹配 MTP-spec gold。实测收益复验（默认配置，decode 时间）：
  重复文本 64 token 2.00s→0.30s（6.7×）、重复代码 48 token
  1.49s→0.43s（3.4×）、repeat130 4.11s→0.63s（6.5×）、普通问题
  1.01×（无草稿回退）、编号用例 0.88×（接受不足仍偏慢，见下）。
- 编号压力用例（接受率低的极端形态）verify+回滚开销仍使总时间略差于
  串行（~0.88×）。这是 ngram 草稿在这类内容上的固有取舍，不是正确性
  问题；是否接受该 trade-off 属于部署决策。
- 串行→MTP→ngram→MTP→ngram 的五次冷请求对拍通过，协议报告的 drafter
  正确（2026-09-18 复跑确认）。这只验证模式切换，不代表同一轮融合两种
  草稿源。注意：ngram 请求后 `mtp_live` 置假，同连接后续 cont+MTP
  请求会退到串行，直到下一个冷请求重新预热（与 AGENTS.md 预热铁律
  同源）。
- live KV 续接保留。重复序列生成 11 个 token 再加换行，串行和 ngram
  的 live/cold 比较均一致；生成 12 个再换行，两者均仍有 live/cold
  差异——与索引 90 同为近平局现象（live decode 续算 vs 全量重算的
  数学顺序不同），不是 ngram 专有问题。
- 请求 `drafter: "ngram-mod"` 带采样参数时，当前明确退回串行采样。
  没有实现 ngram 的 rejection sampling，也没有实现 MTP + ngram 的
  混合草稿策略。
- 模型无 MTP 权重的整模型加载未另行测试；目前已从代码上解除 ngram
  checkpoint 对 MTP 分配分支的依赖。

## 默认 drafter(2026-09-18)

请求省略 `drafter` 时由引擎侧环境变量决定,API 不再代填:

- `GDEC_DRAFTER=ngram`(或 `ngram-mod`/`3`):贪心请求默认走 ngram;
  采样请求自动回退既有默认(ngram 只起草贪心)。
- `GDEC_DRAFTER=serial`(或 `0`):默认串行。
- 不设或 `mtp`:既有行为不变(MTP 权重在则 MTP,否则串行)。
- 客户端显式传 `drafter` 始终优先,不受该变量影响。
- 引擎 INFO 第 5 字段上报当前默认;API `/health` 的 `drafter_default`
  如实转发("ngram-mod"/"mtp"/"serial")。
- 注意:ngram 冷启动首轮草稿表为空,D 行 drafter 列回退占位 0,第
  二轮起正常报告 3(协议占位行为,见 12274 行注释)。

## 复现

使用相对路径：

```bash
bash build.sh test
bash build.sh engine
PROBE_TAG=my-ngram-test PROBE_CONTEXT=40960 GDEC_PREFILL_CHUNK=8192 \
  GDEC_NGRAM_MIN=4 GDEC_NGRAM_MAX=16 bash tools/run_ngram_probe.sh
# 引擎日志出现 serve: listening 后，保持 API 停止，再运行：
python3 tools/ngram_regress.py --keep-going --output logs/my-ngram-test.json
python3 tools/snapshot_regress.py
python3 tools/drafter_switch_test.py
```

`run_ngram_probe.sh` 启动单个受内存限制的测试引擎，关闭磁盘快照；helper 默认 min=4/max=16，引擎源码默认则为 min=48/max=64。它不是生产服务启动脚本。

测试客户端串行使用同一个端口，不应同时运行。`--keep-going` 保留并汇总所有生成比较失败，最终仍返回非零；不会把失败当通过。快照脚本的 cmp4 DIFF 会体现在 JSON 和 MATCH/DIFF 输出中，应直接检查这些结果。
