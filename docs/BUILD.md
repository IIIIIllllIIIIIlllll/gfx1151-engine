# 编译与验证

*English: [BUILD_EN.md](BUILD_EN.md)*

## 环境

在 Linux / AMD ROCm 机器上编译。GPU 目标 `gfx1151`(AMD Strix Halo),
编译器为 ROCm 自带 `hipcc`(HIP 7.x / AMD clang)。

引擎需要 HIP、rocBLAS、hipBLASLt、rocPRIM 开发文件和 Linux C++ 标准库。
API 额外需要 g++、libpng、libjpeg、libwebp 开发包；nlohmann/json 已随仓库
放在 `third_party/nlohmann/json.hpp`，不需要系统安装。
Ubuntu 安装命令:

```bash
sudo apt install build-essential libpng-dev libjpeg-dev libwebp-dev
```

ROCm 需要支持 `gfx1151` 的版本。不要把 GPU 架构参数直接套用到其他显卡。

## build.sh(统一入口)

```bash
bash build.sh                 # all:引擎 + benchmark + API 并行编译(默认)
bash build.sh --bundle        # 分发构建：同时打包全部运行依赖
bash build.sh engine [名字]   # 只编引擎 → build/<名字>(默认 gdec)
bash build.sh bench           # 只编独立性能测试工具
bash build.sh api             # 只编 API 服务器 + CLI 工具
bash build.sh test            # 编 ktest 并运行 kernel 单测
```

产物:

| 文件 | 内容 |
|---|---|
| `build/gdec` | GPU 引擎(`src/gpu/gdec.cpp`) |
| `build/gdec-bench` | 独立性能测试工具(`src/gpu/bench_main.cpp`) |
| `build/gdec-api` | OpenAI 兼容 API 服务器(`src/api/*.cpp`) |
| `build/tok_cli` `tpl_cli` `eng_cli` | tokenizer / 模板 / 引擎协议 CLI |
| `build/http_selftest` `toolparse_test` `vision_test` `engine_host_test` | API / 引擎监听地址组件自测 |
| `build/ktest` | 引擎 kernel 单测 |
| `build/lib` | `--bundle` 生成的 ROCm/图像运行库及 gfx1151 kernel db |

行为要点:

- 编译在内存限额(8 GiB)和超时保护下进行:引擎/ktest 600 秒,
  API 各目标 120 秒。在较慢的机器上超时不代表源码错误,确认编译器
  仍有进展后,按机器能力调整 `build.sh` 里 `compile` 调用的超时秒数。
- 每个目标先编到临时文件,成功后原子替换;编译失败保留上次成功的
  对应二进制,并返回非零状态。
- 引擎或 API 正在运行(或另一个编译/启动任务持有锁)时拒绝编译,
  先停止服务再编。
- 环境变量:`HIPCC`(默认 PATH 中的 hipcc,其次 `/opt/rocm/bin/hipcc`)、
  `GPU_ARCH`(默认 `gfx1151`)、`CXX`(默认 `g++`)、`ROCM_PATH`（仅在无法
  从 `HIPCC` 推断 ROCm 根目录时需要）。例:

```bash
HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151 bash build.sh
```

- 默认构建供本机使用，直接加载系统已经安装的 ROCm 和图像运行库，不创建
  `build/lib/`。分发时加 `--bundle`；脚本会根据 ELF 的实际依赖，把 ROCm
  用户态运行库和 API 所需的
  libpng/libjpeg/libwebp 依赖闭包复制到 `build/lib/`，并仅复制
  `GPU_ARCH` 对应的 rocBLAS/hipBLASLt kernel db。二进制带
  `$ORIGIN/lib` RPATH，`start_hgn.sh` / `start_gguf.sh` 也会显式使用这套私有库；部署时把整个
  `build/` 一起复制即可，目标机器不需要另装这些运行库。

## 编译选项(供参考)

引擎:

```bash
hipcc -O3 -Werror --offload-arch=gfx1151 \
  -Wl,-rpath,'$ORIGIN/lib' -Wl,--disable-new-dtags \
  -o build/gdec src/gpu/gdec.cpp -lrocblas -lhipblaslt
```

上面的 RPATH 参数只在 `--bundle` 分发构建中添加。

不要自行加 `-ffast-math`,它会改变数值行为。

API:C++17,`-O2 -Wall -Wextra -Wpedantic -Werror`,链接
`-lpng -ljpeg -lwebp -lpthread`。`tools/build_api.sh` 只是指向
`build.sh api` 的兼容包装。

## Kernel 单测(不加载模型)

`bash build.sh test` 编译并运行 `tools/ktest.cu`,预期末尾为
`ALL PASS`。它是 kernel 单测,不代替完整模型的数值和性能回归
(完整回归见 NGRAM.md 的"复现"一节)。

## 手动启动推理

模型权重、overlay、tokenizer 和可选视觉塔不在源码仓库中,用
`tools/flashnext2hgn.py` 从 HF 模型转换(见 CONVERT.md)。
实际推理前检查 `free -g`:只运行一个引擎,可用内存应至少 100 GiB。
日常服务直接用根目录 `start_hgn.sh`(hgn 权重)或 `start_gguf.sh`(GGUF 权重,见 GGUF.md),
说明见 QUICKSTART.md;以下是 hgn 的手动方式。

生产选项(部分优化由环境变量开启):

```bash
export GDEC_QSA_KV_BF16=1 GDEC_QSA_WMMA=1 GDEC_QSA_WMMA_BTV=1
export GDEC_MOE_LT=1 GDEC_MOE_LT_BF16=1 GDEC_GR_BF16=1
export GDEC_GDN_STREAM=1 GDEC_GDN_WAVE=1 GDEC_NOWARMUP=1
export GDEC_PREFILL_CHUNK=16384
export GDEC_INDEX_FUSED2=1 GDEC_PP_MOE_OUT=1 GDEC_INDEX_STREAM_SELECT=1
MODEL_BASE=./models/qwen38-flash-next-w4b
```

短 token-ID 推理示例(`--tokens` 接收 token ID,文本请走 tokenizer/API):

```bash
bash tools/run_capped.sh 86 -- build/gdec \
  "$MODEL_BASE.hgn" "$MODEL_BASE.overlay.hgn" \
  --tokens 1,2,3 --gen 8 --maxctx 4096
```

启动 256K 服务:

```bash
bash tools/run_capped.sh 86 -- build/gdec \
  "$MODEL_BASE.hgn" "$MODEL_BASE.overlay.hgn" \
  --serve --port 8730 --maxctx 262144 --gamma 3 \
  --vision-tower ./models/qwen38-flash-next-vision.hgn
```

等引擎打印 `serve: listening`,在另一个终端启动 API:

```bash
build/gdec-api --tokenizer ./models/tokenizer \
  --engine 127.0.0.1:8730 --host 127.0.0.1 --port 8731 --context 262144
```

纯文本服务可以省略引擎的 `--vision-tower`。`GDEC_NOWARMUP=1` 会让首个
请求承担预热耗时,首次延迟不能直接当作稳定 prefill 性能。

API 断连回归无需 GPU 或模型权重：测试会在临时目录创建合成词表，启动本地
假引擎与 API，覆盖三个生成端点的流式/非流式断连、预填充、排队、协议收尾
及下一请求。先编译 API，再运行（Windows 的 `--api` 改为
`build/gdec-api-win.exe`）：

```bash
python tools/api_disconnect_test.py --api build/gdec-api
python tools/api_disconnect_test.py --api build/gdec-api --slots 2
```

引擎监听地址与启动配置回归（均无需模型）：

```bash
build/engine_host_test
python tools/engine_host_config_test.py --bash bash
```

Windows 配置回归可额外传入 `--launcher ./start_win.exe`，检查原生启动器。

## Windows（TheRock）

Windows 版有独立入口，与 build.sh / start_hgn.sh 并列，覆盖引擎与 API 前端
（多模态已支持 PNG/JPEG，仅 WebP 未接，见 PORTING-WINDOWS.md）：

```bash
bash build_win.sh           # 全部产物:引擎、benchmark、API、启动器
bash build_win.sh bench     # 只编独立性能测试工具
bash build_win.sh api       # OpenAI API 前端 → build/gdec-api-win.exe
bash build_win.sh launcher  # 免脚本启动器 → ./start_win.exe
bash build_win.sh test      # 编 ktest-win 并运行 kernel 单测
bash start_win.sh           # Git Bash 下起引擎(行协议 8730) + API(8731) 双进程
```

**日常启动用 `start_win.exe`**：原生 Win32 启动器，双击即用，不需要
Git Bash / PowerShell / 任何脚本宿主。拉起引擎 + API 双进程；默认是托盘程序，
不开控制台，子进程输出写入 `logs\`（启动器自己的输出在 `logs\launcher-win-*.log`）。
右键托盘图标：打开面板 / 复制 API 地址 / 查看引擎或 API 日志 / 打开日志文件夹 /
退出（同时收掉两者）；双击图标：已就绪时打开面板。出错或子进程退出会弹框。
两个子进程在一个 Job 里，启动器被任务管理器结束时它们也随之结束。
`start_win.exe --console` 是旧的控制台模式（输出实时显示，Ctrl+C 或关窗停止），
排查问题用。
**首次双击会弹出启动配置面板**（原生 Win32 窗口，4 个分页）：第 1 页权重——
V1（w4b 单文件）/ V2（主体 + 独立 n-gram 文件）/ GGUF 版本选择，各版本只显示
自己相关的文件行（可浏览选择、可选文件可勾选"不使用"）；GGUF 仅保存配置，
Windows 启动器只支持 hgn（Linux 用 `start_gguf.sh` 启动）。第 2 页上下文窗口 /
并发槽数 / 并发总池 / prefill 分段，另有 YaRN 勾选框：单请求上限超过原生
262144 时开启，扩展倍数按上限自动推导（ROPE_FACTOR 无需手填）。
第 3 页监听地址与端口（ENGINE_HOST、API_HOST 与两个端口，含 IPv4 与占用校验）。
第 4 页显存环境检查——可用显存、权重文件合计、并发总池 / prefill 工作区 / 并发
估算与结论，底部汇总其它页的失败项。每页有独立的检查报错区（存在性与格式嗅探、
tokenizer、数值合法性含 MAX_CONTEXT ≤ factor × original_ctx、端口占用等），
任何页有失败项时不能启动。点"启动服务"会把配置写回
`service.conf`（bash 兼容写法，写前备份 `service.conf.bak`）再继续启动；服务首次
就绪后写入 `# start_win: configured` 标记，之后双击不再弹面板。想改配置：托盘右键
"设置…"，或 `start_win.exe --setup` 强制打开（运行中保存的配置下次启动生效）。
**配置改根目录 `service.conf`**（与 Linux 启动器同一个文件；Windows 目前只支持 hgn，
读其中"hgn 权重"一段，GGUF 一段不生效）：换模型
文件名、调上下文窗口、改端口都编辑它；优先级为 环境变量 > service.conf
> 内置默认（`set MAX_CONTEXT=131072 && start_win.exe` 临时覆盖；
`start_win.exe --check` 只检查配置不启动）。客户端连
`http://127.0.0.1:8731/v1`（标准 OpenAI 接口，含流式）；8730 是引擎内部
行协议，由 API 自动桥接，无需直连。

编译入口在 Git Bash 里跑（双击 `build_win.bat` 亦可，自动定位 Git Bash；
只认 Git for Windows 的 bash，刻意避开 WSL 的 `System32\bash.exe`——
编译脚本依赖 Git Bash 路径语义）。**Git 只是编译期依赖，运行期不需要。**

前置（**仅编译期**）：[TheRock](https://github.com/ROCm/TheRock) Windows 多架构包
（默认 `C:\therock-dist-windows-multiarch-10.0.0\...`，可用 `THEROCK=` 覆盖）+
Git Bash。首次构建会把运行期全部依赖备进 `build/`：TheRock DLL ×6、MSVC
运行时 ×3、rocBLAS/hipBLASLt 的 gfx1151 kernel db 真身（共 ~30M，非全架构
1.2G）。**产物自包含：`build/` 拷到任何同架构 Windows 机器即用，无需安装
ROCm/TheRock，也不设 HIP_PATH/ROCM_PATH**（断根实测见 PORTING-WINDOWS.md）。
GPU 需 BIOS 划分足够显存（heretic 68 GiB 权重 + 256K 上下文实测顶格 95 GiB，
划分 96 GiB）。`start_win.exe` / `start_win.sh` 与 Linux 启动器共用根目录
`service.conf`（模型路径、端口、上下文窗口等都在里面改，环境变量可临时覆盖）。

## 可选工具与常见编译问题

CPU 参考实现和权重检查器(不参与 build.sh,按需手动编译):

```bash
g++ -O3 -Werror -std=c++17 -o build/ref src/ref.cpp
g++ -O3 -Werror -std=c++17 -o build/hgn_dump src/hgn_dump.cpp
```

找不到 `hipcc`:使用 `/opt/rocm/bin/hipcc` 并检查 ROCm 安装。
找不到 `rocprim/...`、`hipblaslt/...` 或 `-lhipblaslt`:检查同一套
ROCm 的开发头文件和库是否安装,避免混用版本。
找不到 `nlohmann/json.hpp`:
确认仓库中的 `third_party/nlohmann/json.hpp` 存在。
找不到 `png.h`、`jpeglib.h`、`webp/decode.h`:
安装上方图像开发包。
出现 `no kernel image` / 架构错误:核对显卡与 `--offload-arch`,不要
只删除参数掩盖问题。
`-Werror` 失败:保留诊断并修正对应兼容性问题,不建议直接关闭。
