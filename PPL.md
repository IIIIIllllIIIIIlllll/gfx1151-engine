# Prefill 困惑度诊断

`--ppl` 离线模式对固定 token 序列进行 teacher forcing：位置 `p` 的原始
logits 预测 `token[p+1]`，计算 `NLL = logsumexp(logits) - logits[next]`；
输出每个位置的 NLL 和 `PPL = exp(mean(NLL))`，不采样、不生成。至少两个 token，
最后一个 token 没有下一个目标，因此不计分。只支持纯文本；不要把 API 生成
token 的 `logprobs` 当成这个指标。

先用同一份 token ID 做数值对拍，再用**未参与训练的自然文本**比较绝对 PPL。
仓库的 `data/qsa-oracle/*.tokens` 主要用来诊断边界，不是质量基准。必须保持
模型文件、tokenizer、序列、量化和其他算子开关一致；关闭尾包合并后再比较
chunk 大小。`--maxctx` 至少等于输入长度，离线默认只有 4096。
自然文本先用与模型匹配的 `tools/qwentok.py` 的 `Tokenizer.encode()` 转成
token ID，逐行写入 `.tokens`；不同文档应分别计分，避免跨文档拼接改变上下文。

Windows PowerShell 示例（先执行 `bash build_win.sh`）：

```powershell
$model = 'models/qwen38-flash-next-w4b.hgn'
$overlay = 'models/qwen38-flash-next-w4b.overlay.hgn'
$tokens = 'data/qsa-oracle/8192.tokens'
$env:GDEC_PREFILL_TAIL_SLACK = '0'
$env:GDEC_PREFILL_CHUNK = '1024'
./build/gdec-win.exe $model $overlay --tokens-file $tokens --maxctx 8192 --ppl > ppl-1024.tsv
$env:GDEC_PREFILL_CHUNK = '8192'
./build/gdec-win.exe $model $overlay --tokens-file $tokens --maxctx 8192 --ppl > ppl-8192.tsv
python tools/ppl_compare.py ppl-8192.tsv ppl-1024.tsv --chunk 1024
```

Linux 示例（在项目根目录执行）：

```bash
bash build.sh engine
export GDEC_PREFILL_TAIL_SLACK=0
BASE=models/qwen38-flash-next-w4b.hgn
OVL=models/qwen38-flash-next-w4b.overlay.hgn
IDS=data/qsa-oracle/8192.tokens
GDEC_PREFILL_CHUNK=1024 build/gdec "$BASE" "$OVL" --tokens-file "$IDS" --maxctx 8192 --ppl > ppl-1024.tsv
GDEC_PREFILL_CHUNK=8192 build/gdec "$BASE" "$OVL" --tokens-file "$IDS" --maxctx 8192 --ppl > ppl-8192.tsv
python3 tools/ppl_compare.py ppl-8192.tsv ppl-1024.tsv --chunk 1024
```

先用 256 个相同 token 做串行基线：

```powershell
Get-Content $tokens -TotalCount 256 | Set-Content ppl-short.tokens
$env:GDEC_NOPREFILLBATCH = '1'
./build/gdec-win.exe $model $overlay --tokens-file ppl-short.tokens --maxctx 256 --ppl > ppl-serial.tsv
Remove-Item Env:GDEC_NOPREFILLBATCH
$env:GDEC_PREFILL_CHUNK = '64'
./build/gdec-win.exe $model $overlay --tokens-file ppl-short.tokens --maxctx 256 --ppl > ppl-batch.tsv
python tools/ppl_compare.py ppl-serial.tsv ppl-batch.tsv --chunk 64
```

若短序列都不一致，优先排查批处理算子或量化精度；短序列一致但长序列在
`seam=1024,2048,...` 附近出现明显尖峰，重点检查 chunk 之间的状态衔接。
若逐位置 NLL 基本一致，总 PPL 也一致，优先排查 tokenizer/chat template、
采样参数、模型权重量化与 API/tool-call 链路，而不是 `PREFILL_CHUNK`。

输出数据以 `ppl_token<TAB>pos<TAB>next_id<TAB>nll` 开头，位置从 0 开始；
`ppl_summary` 包含计分 token 数、平均 NLL 与 PPL。不同 chunk 的整体均值
可能掩盖局部问题，因此比较工具会列出最大差异和每个边界两侧的差异。
