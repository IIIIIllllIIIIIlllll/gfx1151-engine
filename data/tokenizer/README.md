# data/tokenizer — 仓库内置兜底词表

qwenox-api 独立运行时，词表按以下顺序解析：

1. `--tokenizer DIR`（显式指定，失败即 503，不回退）
2. `service.conf` 的 `TOKENIZER_DIR`（默认 `models/tokenizer`）
3. **本目录**（`<项目根>/data/tokenizer`）——随仓库分发的兜底

本目录只需一个文件：`tokenizer.json`（Qwen BPE 词表，约 20 MB）。
体量大，与 `models/` 一样本地放置、不入库；首次获取任选其一：

```sh
# 已有模型词表（推荐，与训练版本完全一致）
mkdir -p data/tokenizer && cp models/tokenizer/tokenizer.json data/tokenizer/

# 或从 HuggingFace 下载（Qwen3 的词表，Apache-2.0 可再分发）
curl -L -o data/tokenizer/tokenizer.json \
  https://huggingface.co/Qwen/Qwen3-32B/resolve/main/tokenizer.json
```

来源许可：Qwen tokenizer 随 Qwen 模型以 Apache-2.0 发布，允许再分发。
本目录的词表仅用于 qwenox-api 的 OpenAI 兼容前端（聊天模板套用、token
计数、SNAPS 切点），引擎本身不读它。
