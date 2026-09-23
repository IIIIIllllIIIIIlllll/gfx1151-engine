# WikiText-2 困惑度测试

## 评测口径

使用 WikiText-2 **raw-v1 的 test split**，不使用 train/valid 作为最终报告。
`tools/prepare_wikitext2.py` 直接从 `wikitext-2.zip` 中读取
`wikitext-2-raw/wiki.test.raw`；仿照 Hugging Face 固定长度模型困惑度示例，
按官方 WikiText 加载器规则将纯空白行变成空字符串、非空行保留结尾的
换行符，再以两个换行符拼接；之后用**当前模型自己的** `tokenizer.json`
（`tokenizers` 快速实现）一次性编码，禁用自动 special tokens，不加入聊天模板。
这和简单地对 `.raw` 文件去掉换行、逐行独立分词或原样全文分词都不是同一个协议。
脚本额外写入 `.tokens.json`，记录数据和 tokenizer 的 SHA-256、计分 token 数。

`--ppl` 进行 teacher forcing，不生成/采样：位置 `p` 的原始 logits 预测
`token[p+1]`，`NLL = logsumexp(logits[p]) - logits[p][token[p+1]]`，
`PPL = exp(sum(NLL) / 计分 token 数)`。第一个 token 没有左侧上下文，不计分。
`--ppl-window W --ppl-stride S` 以不超过 `W` 的重叠上下文覆盖**整个**
test split（`1 <= S < W <= maxctx`）；每个窗口重置模型状态，重算重叠的
上下文，但只计首次进入窗口的目标 token。即使目标位于 prefill chunk
边界，也使用前一行 logits 计分；总分按 token 加权，绝不平均各窗口 PPL。

WikiText-2 的数字只能在相同 raw/非 raw 版本、文本拼接、tokenizer、
是否加入 BOS、窗口和 stride、模型权重/量化下直接比较。WikiText 语料
也不能单凭 PPL 判断指令遵循或 agent 工作能力。对照
`data/qsa-oracle/*.tokens` 可排查 batch/chunk 数值，但它们不是质量基准。

## Linux 运行

将你本机的 `wikitext-2.zip` 拷贝到 Linux 上的任意路径；不要把数据集或生成的
大型 `.tokens`、`.tsv` 文件提交到仓库。停掉服务后，在项目根目录运行：
若在 `/home/mark/Workspace/test` 测试，可先设
`MODEL_DIR=/home/mark/Workspace/gfx1151-engine/models`。

```bash
MODEL_DIR=${MODEL_DIR:-models}
python3 -m pip install --target .ppldeps tokenizers
export PYTHONPATH="${PWD}/.ppldeps${PYTHONPATH:+:$PYTHONPATH}"
python3 tools/prepare_wikitext2.py --zip /path/to/wikitext-2.zip \
  --tokenizer-json "$MODEL_DIR/tokenizer/tokenizer.json" --output wiki-test.tokens
bash build.sh engine
export GDEC_PREFILL_TAIL_SLACK=0
BASE="$MODEL_DIR/qwen38-flash-next-w4b.hgn"
OVL="$MODEL_DIR/qwen38-flash-next-w4b.overlay.hgn"
head -n 1024 wiki-test.tokens > wiki-smoke.tokens
GDEC_PREFILL_CHUNK=512 build/gdec "$BASE" "$OVL" \
  --tokens-file wiki-smoke.tokens --maxctx 512 --ppl \
  --ppl-window 512 --ppl-stride 128 > wiki-smoke.tsv
tail -1 wiki-smoke.tsv  # count 应为 1023；这只是烟测，不是正式 PPL
GDEC_PREFILL_CHUNK=2048 build/gdec "$BASE" "$OVL" \
  --tokens-file wiki-test.tokens --maxctx 2048 --ppl \
  --ppl-window 2048 --ppl-stride 512 > wiki-2048.tsv
GDEC_PREFILL_CHUNK=512 build/gdec "$BASE" "$OVL" \
  --tokens-file wiki-test.tokens --maxctx 2048 --ppl \
  --ppl-window 2048 --ppl-stride 512 > wiki-512.tsv
python3 tools/ppl_compare.py wiki-2048.tsv wiki-512.tsv --chunk 512
tail -1 wiki-2048.tsv
```

首个运行是一个窗口对应一个 prefill chunk；第二个运行在同样 2048-token
上下文中按 512-token chunk 分批。比较工具显示整体 PPL、每 token NLL
偏差及 chunk 边界附近的偏差。如果短序列串行与 batch 即不一致，应先排查
批量算子/精度；若仅 chunk 边界有尖峰，则检查跨 chunk 状态衔接。

## Windows 与短序列对拍

Windows 也可用同一脚本，把 `--zip` 设为
`C:\Users\Mark\Workspace\CProject\my\wikitext-2.zip`，执行
`bash build_win.sh`，引擎路径改为 `build/gdec-win.exe`；其余参数相同。
为避免串行全量测试过慢，先从 `wiki-test.tokens` 截取 256 个 ID，分别用
`GDEC_NOPREFILLBATCH=1`（串行）和 `GDEC_PREFILL_CHUNK=64`
（批量）运行 `--ppl --maxctx 256`，再运行
`python3 tools/ppl_compare.py serial.tsv batch.tsv --chunk 64`。
不设置 `--ppl-window/--ppl-stride` 时仍是旧的单段数值对拍模式，
要求序列长度不超过 `--maxctx`。

输出 `ppl_token<TAB>pos<TAB>next_id<TAB>nll`（原序列绝对位置从 0 起）
及一行 `ppl_summary`。比较工具会校验 token 顺序和 summary 总数，
防止截断结果被误判为完整评测。完整 test split 可能比较慢：
重叠窗口必须反复处理上下文，且每个计分位置都需要 lm_head。
