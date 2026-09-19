# MTP 参数与对比方法

*English: [MTP_EN.md](MTP_EN.md)*

目前服务默认草稿长度是 3，范围 1–8。最先比较 1、2、3，每次只改变长度：

```bash
MTP_GAMMA=1 bash start.sh
```

停止该轮服务后，再把 1 换成 2 或 3。也可以在 `service.conf` 中修改 `MTP_GAMMA` 的默认值。
**修改草稿长度需要重启引擎，不需要重新编译。** 本次没有替你重启或改变正在运行的服务。

脚本修正说明：服务模式实际读取 `GDEC_SPEC_GAMMA`，旧脚本只传入的 `--gamma` 仅作用于离线 `--spec-gen`。修正版 start.sh 会把 `MTP_GAMMA` 导出为 `GDEC_SPEC_GAMMA`。手动启动引擎时可以写 `GDEC_SPEC_GAMMA=2 bash tools/run_capped.sh ...`。不要用请求 JSON 的 gamma 字段；当前 API 不支持逐请求修改草稿长度。

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
