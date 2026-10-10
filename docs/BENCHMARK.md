# 性能测试工具

`qwenox-bench` 是独立的命令行性能测试程序，不启动 HTTP 服务，也不接受
命令行模型路径。模型和 tokenizer 路径统一从项目根目录的
`service.conf` 读取，避免测试命令和正式配置不一致。

## 编译

无参数执行编译脚本会一次性编译完整的运行所需产物，其中包括引擎、API
前端和性能测试工具；Windows 还会编译原生启动器：

```bash
# Linux
bash build.sh

# Windows（在 Git Bash 中，或双击 build_win.bat）
bash build_win.sh
```

如果只需要重新编译性能测试工具，仍可以使用单目标入口：

```bash
bash build.sh bench
bash build_win.sh bench
```

生成文件：

- Linux：`build/qwenox-bench`
- Windows：`build/qwenox-bench.exe`

## 运行

不带参数直接打开程序会立即退出。显式传入 `run` 才会执行测试：

```bash
# Linux
build/qwenox-bench run

# Windows
build/qwenox-bench.exe run
```

测试程序自身的状态、错误和汇总输出全部使用英语；引擎初始化阶段可能
额外输出引擎已有的诊断信息。

## 测试内容

测试按以下顺序执行，前一步失败时不会继续：

1. **加载模型权重**：只加载 `service.conf` 中当前选定的 HGN 或 GGUF
   配置，包括 overlay 和 MTP 权重。文件不存在、映射失败、HIP 初始化、
   arena、graph 或内存分配等任意异常都会终止测试。
2. **Prefill 性能**：读取 `data/qsa-oracle/*.tokens` 中的 token 文件，
   每个文件单独测量，最后输出每个 prompt 和整体加权平均值。
3. **TG / MTP 解码性能**：使用三类固定的英语测试场景：Python 代码生成、
   创意写作、常识问答。每个场景输出 TG 速度和 MTP 草稿 token 接受率。

## 配置覆盖

这些环境变量只调整测试行为，不改变模型文件的选择：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `QWENOX_BENCH_FORMAT` | 自动 | 强制选择 `hgn` 或 `gguf` |
| `BENCH_PREFILL_REPEATS` | `2` | 每个 prefill prompt 的采样次数 |
| `BENCH_DECODE_REPEATS` | `2` | 每个解码场景的采样次数 |
| `BENCH_DECODE_TOKENS` | `128` | 每次解码生成的 token 数 |
| `PREFILL_CHUNK` | `0` | prefill 分段大小；`0` 使用平台默认值 |

`PREFILL_CHUNK` 是 `service.conf` 的正式配置字段，不要使用单独的
`BENCH_PREFILL_CHUNK`。例如：

```bash
PREFILL_CHUNK=4096 build/qwenox-bench run
```
