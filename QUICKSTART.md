# 编译和启动

*English: [QUICKSTART_EN.md](QUICKSTART_EN.md)*

在 Linux / ROCm 机器上进入项目目录,执行:

```bash
bash build.sh
bash start.sh
```

`build.sh` 一次编译 GPU 引擎和原生 API,输出 `build/gdec`、`build/gdec-api`。
`start.sh` 先加载引擎,等就绪后启动 API。默认 256K 上下文、MTP gamma 3。
按 **Ctrl+C** 同时停止本次启动的 API 和引擎。启动不会退出终端;保持
SSH 会话打开,或在 tmux 中运行。

启动后 API 默认在 `http://<主机>:8731/v1`。
日志按每次启动保存在 `logs/日期时间-PID/engine.log` 和 `api.log`。

## 模型文件

编辑根目录 `service.conf`,通常只需要改 `MODEL_DIR`(默认 `./models`):

```bash
MODEL_DIR="./models"
```

目录里应有:

```text
models/
  qwen38-flash-next-w4b.hgn
  qwen38-flash-next-w4b.overlay.hgn
  qwen38-flash-next-mtp.hgn
  qwen38-flash-next-vision.hgn
  tokenizer/tokenizer.json
```

模型名不同时,直接修改 `MODEL_FILE`、`OVERLAY_FILE`、`MTP_FILE`、
`VISION_FILE`、`TOKENIZER_DIR`。纯文本可设 `VISION_FILE=""`;不需要
overlay 可设 `OVERLAY_FILE=""`;`MTP_FILE` 是独立的 8-bit MTP 投机草稿
权重,设 `MTP_FILE=""` 则退回 overlay 内置的 4-bit 草稿头。
用 `tools/flashnext2hgn.py` 转换自有模型时会自动生成这套文件,
见 CONVERT.md。

所有相对路径都以脚本所在项目目录为基准,不受打开终端的位置影响。
也可临时指定配置,无需修改文件:

```bash
MODEL_DIR=./models VISION_FILE="" bash start.sh
```

## 投机解码

默认 drafter 是 chain(ngram 优先、MTP 兜底),引擎侧默认生效,无需任何
参数;贪心与采样请求都走 chain。HTTP API 不暴露 drafter 选择,请求体里
发 `drafter` 字段会被静默忽略。环境变量 `GDEC_DRAFTER=ngram` 纯 ngram、
`=mtp` 纯 MTP、`=serial` 串行基线。

`MTP_GAMMA=1 bash start.sh` 可试一轮草稿长度 1,范围 1–8,默认 3;
修改后需重启引擎,不需要编译。参数和接受率的含义见 MTP.md。

## 其他常用命令

```bash
bash start.sh --check   # 仅检查文件、端口、内存等,不启动服务
bash build.sh engine   # 只编译引擎
bash build.sh api      # 只编译 API
bash build.sh test     # 编译并运行 kernel 单测,不加载模型
```

端口、监听地址、上下文、MTP、内存上限集中在 `service.conf`。默认 API
监听 `0.0.0.0:8731`,可从局域网访问;只需本机访问时改为
`API_HOST="127.0.0.1"`。

运行时内存可通过 `curl http://127.0.0.1:8731/memory` 查询。返回值区分
HIP 设备分配、`mmap + hipHostRegister` 的专家权重、pinned host 内存和
进程 RSS；`gpu_accessible_committed_bytes` 是引擎自身可准确记账的合计，
不会把未注册、可回收的文件 mmap 页缓存误算为显存。

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
