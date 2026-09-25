# 编译和启动

*English: [QUICKSTART_EN.md](QUICKSTART_EN.md)*

在 Linux / ROCm 机器上进入项目目录,先编译,再按手上的权重格式二选一启动:

```bash
bash build.sh
bash start_hgn.sh    # hgn 权重(tools/flashnext2hgn.py 转换得到)
bash start_gguf.sh   # 或 GGUF 权重(Unsloth UD-Q4_K_XL,与 llama.cpp 同一份文件)
```

`build.sh` 一次编译 GPU 引擎和原生 API,输出 `build/gdec`、`build/gdec-api`。
两个启动器除了读哪份权重之外完全相同:先加载引擎,等就绪后启动 API。
默认 256K 上下文、MTP gamma 3。按 **Ctrl+C** 同时停止本次启动的 API 和引擎。
启动不会退出终端;保持 SSH 会话打开,或在 tmux 中运行。旧的 `start.sh`
只打印上面的选择提示并退出。

启动后 API 默认在 `http://<主机>:8731/v1`。
日志按每次启动保存在 `logs/日期时间-PID/engine.log` 和 `api.log`。

## 模型文件

两个启动器都从项目根目录的 `models/` 读取权重(`service.conf` 的
`MODEL_DIR`,默认 `./models`),`tokenizer/` 两种格式共用(只需
`tokenizer.json`,取自原模型 HF 仓库)。只需放自己要用的那一种:

```text
models/
  tokenizer/tokenizer.json                              # 两种都要
  # --- bash start_hgn.sh ---
  qwen38-flash-next-w4b.hgn                             # 主模型
  qwen38-flash-next-w4b.overlay.hgn                     # overlay
  qwen38-flash-next-mtp.hgn                             # 8-bit MTP 草稿
  qwen38-flash-next-vision.hgn                          # 视觉塔
  # --- bash start_gguf.sh ---
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf     # 主模型,4 个分片都要
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
  mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf               # MTP 草稿 sidecar
  mmproj-BF16.gguf                                      # 视觉塔
```

缺文件时启动器不会启动,而是一次列出全部缺失的文件和配置项,退出码 1,例如:

```text
错误：start_gguf.sh（GGUF 权重）缺少以下文件：
  分片 3/4：./models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
```

`start_hgn.sh` 只接受 `.hgn`、`start_gguf.sh` 只接受 `.gguf`,放错会提示
改用另一个启动器。文件名不同时修改 `service.conf` 里对应的一段:

- hgn:`MODEL_FILE`、`OVERLAY_FILE`、`MTP_FILE`、`VISION_FILE`。不需要
  overlay 可设 `OVERLAY_FILE=""`;`MTP_FILE` 是独立的 8-bit MTP 投机草稿
  权重,设 `MTP_FILE=""` 则退回 overlay 内置的 4-bit 草稿头。用
  `tools/flashnext2hgn.py` 转换自有模型时会自动生成这套文件,见 CONVERT.md。
- GGUF:`GGUF_FILE`(填第 1 个分片,其余分片须在同一目录)、
  `GGUF_MTP_FILE`、`GGUF_VISION_FILE`。设 `GGUF_MTP_FILE=""` 则不做 MTP 投机
  (仍有 ngram 草稿)。精度与性能对比见 GGUF.md。

两者纯文本服务都可以把视觉塔设为空(`VISION_FILE=""` / `GGUF_VISION_FILE=""`)。
`TOKENIZER_DIR` 两者共用。

所有相对路径都以脚本所在项目目录为基准,不受打开终端的位置影响。
也可临时指定配置,无需修改文件(同名环境变量优先):

```bash
MODEL_DIR=/data/models VISION_FILE="" bash start_hgn.sh
GGUF_VISION_FILE="" bash start_gguf.sh
```

## 投机解码

默认 drafter 是 chain(ngram 优先、MTP 兜底),引擎侧默认生效,无需任何
参数;贪心与采样请求都走 chain。HTTP API 不暴露 drafter 选择,请求体里
发 `drafter` 字段会被静默忽略。环境变量 `GDEC_DRAFTER=ngram` 纯 ngram、
`=mtp` 纯 MTP、`=serial` 串行基线。

`MTP_GAMMA=1 bash start_hgn.sh`(或 `start_gguf.sh`)可试一轮草稿长度 1,范围 1–8,默认 3;
修改后需重启引擎,不需要编译。参数和接受率的含义见 MTP.md。

## 其他常用命令

```bash
bash start_hgn.sh --check    # 仅检查文件、端口、内存等,不启动服务(start_gguf.sh 同)
bash build.sh engine         # 只编译引擎
bash build.sh api            # 只编译 API
bash build.sh test           # 编译并运行 kernel 单测,不加载模型
```

端口、监听地址、上下文、MTP、内存上限集中在 `service.conf`。默认 API
监听 `0.0.0.0:8731`,可从局域网访问;只需本机访问时改为
`API_HOST="127.0.0.1"`。

运行时内存可通过 `curl http://127.0.0.1:8731/memory` 查询。返回值区分
HIP 设备分配、`mmap + hipHostRegister` 的专家权重、pinned host 内存和
进程 RSS；`gpu_accessible_committed_bytes` 是引擎自身可准确记账的合计，
不会把未注册、可回收的文件 mmap 页缓存误算为显存。

### 控制台与服务端参数覆盖

浏览器打开 `http://<主机>:8731/` 即是控制台,包括健康状态、内存记账、
"采样 / 思考参数(服务端覆盖)"和"对话测试"(流式调用
`/v1/chat/completions`,思考内容单独显示)。

覆盖表用来防止客户端传错采样参数或思考开关。可覆盖的字段:
`enable_thinking`、`reasoning_effort`、`preserve_thinking`、`temperature`、
`top_p`、`top_k`、`min_p`、`presence_penalty`、`frequency_penalty`、
`max_tokens`。每个字段三种模式:

- 跟随客户端:不干预(表里没有该字段)。
- 默认值:仅在客户端没传该字段时使用。
- 强制:无论客户端传什么都改成此值;`max_tokens` 的强制是上限,客户端
  给得更大才压到此值。强制 `temperature=0` 时请求里的 `logprobs` 会被去掉。

覆盖在请求解析之前改写请求体,对 chat / responses / completions 都生效
(responses 的思考强度是 `reasoning.effort`;思考字段对 completions 无意义)。
被改写的字段写进响应头 `X-Gdec-Overrides`(流式同样有)。当前表也出现在
`/health` 的 `server_overrides`。

接口:`GET /admin/overrides` 读取,`POST /admin/overrides` 整表替换,`{}`
表示清空。示例:

```bash
curl -X POST http://127.0.0.1:8731/admin/overrides -H 'Content-Type: application/json' \
  -d '{"enable_thinking":{"mode":"force","value":true},
       "reasoning_effort":{"mode":"default","value":"medium"},
       "temperature":{"mode":"force","value":0.6}}'
```

表持久化在 `data/api-overrides.json`(相对启动目录,重启后自动加载);
`gdec-api --overrides FILE` 可改路径,`--overrides ''` 表示只放内存。默认
任何能访问 API 的人都能修改覆盖表(与 API 本身同样开放)。需要限制时在
启动 API 的环境里设置 `GDEC_API_ADMIN_KEY=<密钥>`,之后 POST 必须带
`X-Admin-Key: <密钥>` 或 `Authorization: Bearer <密钥>`;控制台会显示
密钥输入框。一键验证:`bash tools/api_override_verify.sh`(测试端口
8732/8733,需先停掉生产服务)。

引擎使用 `tools/run_capped.sh`,默认 86 GiB 内存上限、启动前至少
100 GiB 可用内存;另一个引擎或同项目启动任务运行时会拒绝重复启动。
加载阶段持续 60 秒没有日志、引擎 I/O 或 GPU GTT 分配进展会停止;最长
启动时间 360 秒。退出只清理本次启动的进程。

## 编译环境

验证环境:Ubuntu,`gfx1151` GPU,ROCm HIP 7.x / AMD clang。
GPU 编译固定使用 `-O3 -Werror`,链接 rocBLAS 和 hipBLASLt,还需要
rocPRIM 头文件。API 使用 C++17 / `-O2 -Werror`,nlohmann/json 已随仓库提供，
另依赖 libpng、libjpeg、libwebp 和 pthread。Ubuntu 安装命令:

```bash
sudo apt install build-essential libpng-dev libjpeg-dev libwebp-dev
```

脚本优先查找 PATH 中的 `hipcc`,否则使用 `/opt/rocm/bin/hipcc`。
其他 ROCm 位置或 GPU 可显式指定:

```bash
HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151 bash build.sh
```

编译在 8 GiB 内存限额和超时保护下进行(引擎 600 秒、API 各目标
120 秒)。在较慢的机器上超时不代表源码错误;确认编译器仍有进展,
再根据机器能力调整 `build.sh` 的超时。细节见 BUILD.md。
