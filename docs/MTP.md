# MTP 参数与对比方法

*English: [MTP_EN.md](MTP_EN.md)*

服务默认 `MTP_GAMMA=0`（auto）：greedy（temperature=0）固定 4；采样模式自适应（每轮按各深度接受率的滑动估计在 3–7 里选，见 `Model::GammaCtl`）。设 1–8 则所有请求都用这个固定长度。要手动比较时每次只改变长度：

```bash
MTP_GAMMA=1 bash start_hgn.sh    # GGUF 权重用 start_gguf.sh
```

停止该轮服务后，再把 1 换成 2 或 3。也可以在 `service.conf` 中修改 `MTP_GAMMA` 的默认值。
**修改草稿长度需要重启引擎，不需要重新编译。** 本次没有替你重启或改变正在运行的服务。

脚本修正说明：服务模式实际读取 `GDEC_SPEC_GAMMA`，旧脚本只传入的 `--gamma` 仅作用于离线 `--spec-gen`。启动器（start_hgn.sh / start_gguf.sh）会把 `MTP_GAMMA` 导出为 `GDEC_SPEC_GAMMA`。手动启动引擎时可以写 `GDEC_SPEC_GAMMA=2 bash tools/run_capped.sh ...`。不要用请求 JSON 的 gamma 字段；当前 API 不支持逐请求修改草稿长度。

## 请求参数

MTP 权重在 `service.conf` 的 `MTP_FILE`（默认指向独立的 8-bit sidecar `qwen38-flash-next-mtp.hgn`；设 `MTP_FILE=""` 退回 overlay 内置的 4-bit 草稿头）。默认 drafter 是 chain（ngram 优先、MTP 兜底，贪心/采样都生效；采样时 ngram 轮按 llama.cpp 语义从目标分布重采样、按 token id 比对接受，MTP 轮维持精确拒绝采样，输出分布与串行采样一致）；`GDEC_DRAFTER=mtp` 纯 MTP，`GDEC_DRAFTER=ngram` 纯 ngram，`GDEC_DRAFTER=serial` 串行基线。drafter 选择不由请求控制：HTTP API 不接受 `drafter` 字段（静默忽略）。未填写 temperature 时 API 默认 `temperature=1.0, top_k=20, top_p=0.95, min_p=0`，是采样模式。

对照测试在相同 prompt 上固定这些字段，比较不同 gamma（改 `MTP_GAMMA` 重启）：

```json
{
  "temperature": 0,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "max_tokens": 512
}
```

temperature=0 是贪心对照，不保证提高接受率。随后用你日常的采样参数重复比较；每组跑 3 次并区分首次预热。修改 temperature/top_k/top_p/min_p 或惩罚项会改变输出分布，不能只把接受率变化当作无损加速收益。采样时固定 seed 可以辅助比较，但不同 gamma 不保证同 seed 逐 token 一致。

本实现没有单独的 draft_temperature 或可调接受阈值：贪心模式验证 token 是否匹配，采样模式按 p/q 拒绝采样。GDEC_MTP_NORM/HAGG 是历史结构调试开关，保留默认值。

## 看日志

```bash
grep -E 'decode-live:|spec(-sample)?:|depth acc:|spec(-sample)? time:' logs/*/engine.log | tail -30
```

重点记录 tok/s、commit/round、depth acc、draft/verify/rollback 时间。depth acc 的 `1:x 2:y 3:z` 是接受前 1/2/3 个草稿的累计比例，不是各层独立命中率。

日志的总体接受率是接受草稿数 / 提出草稿数。缩短 gamma 常会让这个百分比好看，却不一定让生成更快。例如 gamma=3 时 30% 接受率，平均约接受 0.9 个草稿；加上每轮的验证 token，大致是 1.9 token/轮（忽略末尾截断）。

如果第 1 个草稿接受率高、后两项快速下降，优先尝试 gamma=1/2；如果第 1 项也很低，缩短长度主要减少浪费，还需比较 serial 是否更快。最后按稳定的实际生成 tok/s 和输出质量选择，不按接受率单独选择。

## decode 加速开关与离线测试

以下默认开启，仅用于排查时关闭：

| 开关 | 作用 |
|---|---|
| `GDEC_SMS_GPU=0` | 关闭 GPU 采样准备（bias/penalty 修正 + top-k 在 GPU 上做，只拷回候选；top_k 为 1..64 时生效，否则自动回退 host 全词表路径）。输出分布不变，同 seed 的具体 token 可能不同 |
| `GDEC_DRAFT_LM_Q4=0` | draft 改回用生产 lm_head（默认用 base 文件的 4-bit 副本，只影响草稿，不影响提交的 token；多占约 350 MB 显存） |
| `GDEC_ARGMAX_OLD=1` | 改回单 block argmax（只在完全并列时结果不同） |

离线复现 API 的 decode 路径（`--spec-gen N`，或 `tools/pp_prod.sh <tok> LABEL SPEC=256 ...`）：

- `GDEC_SPEC_CHAIN=1`：chain drafter（API 默认）；不设为纯 MTP。
- `GDEC_SPEC_SAMPLE=temp,top_k,top_p,seed[,presence,frequency]`：拒绝采样模式；不设为 greedy。
- `GDEC_SMS_CHECK=1`：每行把 GPU 稀疏分布与 host 全词表 `prepare()` 比对，打印 `[sms-check] n=… bad=…`（很慢，只用于验证）。

一键验证：`bash tools/tg_verify.sh`（kernel 单测 + greedy 逐 token + 采样分布自检 + 采样速度，结尾 PASS/FAIL）。

## 自适应 γ（auto）

控制器按“上一轮是否全部接受”分两套估计各深度接受率：全接受之后的一轮明显更容易接受（真实长文采样：.93/.86/.83 对 .77/.71/.73），γ 越深全接受越少，有效接受率随 γ 下降。单套估计看不到这一点，会往深处漂。两套估计用稳态概率合成期望提交数，再除以耗时 `1 + r·γ` 选 γ。

| 开关 | 作用 |
|---|---|
| `GDEC_SPEC_ADAPT=0` | 采样 auto 退回固定 3 |
| `GDEC_SPEC_ADAPT_GREEDY=1` | greedy auto 也走自适应（默认固定 4；成对测试里两者相差 ±1%，固定更可复现） |
| `GDEC_SPEC_ADAPT_R` | 每轮耗时模型 `1 + r·γ` 的斜率/截距比（实测约 42 + 10.7·γ ms → 0.25；默认 0.3，抵消模型对深 γ 的偏乐观） |
| `GDEC_SPEC_ADAPT_ALPHA` | 各深度接受率 EMA 的步长（默认 0.05） |
| `GDEC_SPEC_ADAPT_HYST` / `_MIN` / `_MAX` | 切换门槛（默认 0.02）/ γ 下限 3（实测卡在 γ2 比 γ3 慢约 20%）/ 上限 7（γ=8 = 9 行 verify，越过 P≤8 快路径） |

离线：`--gamma 0`（pp_prod.sh `GAMMA=0`）走自适应。调参工具：

- `tools/gamma_trace.sh` + `tools/gamma_sim.py`：`GDEC_SPEC_TRACE=1` 打出逐轮 `γ,接受数,ms`，离线回放各种控制器。`MODEL=markov`（两状态生成模型，固定 γ 的每轮提交与实测差约 2%）最准；默认的 pos/round 回放对浅 γ 低估 7–9%，只能用来排序。
- `tools/gamma_force_check.sh`：`GDEC_SPEC_FORCE=<ids 文件>` 让 greedy spec 对照参考序列验收，所有 γ 配置生成完全相同的 token，成对比较只剩计时噪声（约 ±1%）。直接比较不同 γ 时，greedy 轨迹会在近并列处分叉，单个 prompt 差 ±15–27%。
- `tools/gamma_sample_check.sh`：采样模式多起点 × 多 seed 汇总（采样轨迹没法固定）。
