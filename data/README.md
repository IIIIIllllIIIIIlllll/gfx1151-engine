# data/qsa-oracle — 数值回归基准

本目录是引擎的**输出对拍基准**:固定 prompt + 固定生成长度的贪心参考输出,
用于验证代码改动没有改变模型的数值行为。任何引擎改动都应通过这组基准
再谈合入。

## 文件组成

- `<N>.json` — 5 组 oracle 基准(N = 2048 / 2051 / 2052 / 8192 / 32768)。
  字段:`prompt_ids`(输入 token 序列)、`gen`(生成长度)、
  `response`(参考贪心输出,OpenAI 响应形状)。
- `<N>.tokens` — 对应 prompt 的原始 token 列表(纯文本,一行一个),
  便于人工查看和复用;`65536` / `131072` 两份是超长 prompt 素材,
  供生成未来的长上下文基准使用。
- `<N>.gdec.log` — 采集基准时引擎侧逐步 top-5 logit 记录,输出分歧时
  用来定位第一个分叉点。
- `server-health.json` — API `/health` 响应形状的参考样本。

## 消费这些文件的工具

- `tools/snapshot_regress.py` — 读 5 组 `.json`,对运行中的引擎逐个
  重放并比对输出文本,打印 MATCH/DIFF(另含 4 组 cmp4 短用例,基准在
  `tools/server_greedy24.json`)。NGRAM.md"复现"一节给出完整流程。
- `tools/oracle8192_one.py` — 单独重放 8192 组。
- `tools/accept_checks.py` — 用 32768 组的 prompt 拼 64K 长上下文
  检查。

参考输出由贪心解码产生;在近平局 token 上,不同批量形态可能合法地落在
平局另一侧(NGRAM.md 有定量分析),因此个别用例的 DIFF 需要人工判断,
不是自动失败信号。
