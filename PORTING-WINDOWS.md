# Windows 移植可行性分析（2026-09-19）

*English: [PORTING-WINDOWS_EN.md](PORTING-WINDOWS_EN.md)*

> **修订说明**：本文第三版。第一版的"两堵墙"结论基于公开资料推演；
> 第二版基于本机（Windows + gfx1151，BIOS 划分 96 GiB 显存，128 GiB 内存）
> 的**实测数据**。探针源码在 `tools/winprobe/`，可复跑。
> **第三版：移植已完成并验证通过**，见下节。

## 移植结果（2026-09-19 收尾，全部实测）

**移植完成，引擎与 OpenAI 兼容 API 在 Windows + gfx1151 上全功能运行。**
根目录 `build_win.sh` 一键构建（`engine` / `api` / `launcher` / `test` 四个目标，
与 Linux `build.sh` 并列），产物 `build/gdec-win.exe` + `build/gdec-api-win.exe`
+ `start_win.exe`；`start_win.exe` 一键启动引擎 + API 双进程（原生 Win32
启动器，双击即用，不需要 Git Bash / PowerShell；客户端连 8731 走标准
OpenAI 接口）。Git Bash 下 `start_win.sh` 与 Linux `start.sh` 并列。

验证矩阵（全部 PASS）：

| 验证项 | 结果 |
|---|---|
| 内核单测 `ktest-win`（gdn/ple/index/convstate 等） | `== ALL PASS ==` |
| 加载器字节级对拍 `loadcmp`（64 个随机含不对齐区间，NO_BUFFERING 直读 vs 映射 memcpy） | ALL MATCH |
| CLI 冒烟（68 GiB 权重 + MTP，`--tokens 1,2,3 --gen 8`） | 出字正确，prefill 15.9 tok/s，**decode 32.5 tok/s**（30.8 ms/token，与 Linux 同 DRAM 档位） |
| serve 模式 256K（`--maxctx 262144` + vision tower） | arena 估算 91.03 GiB → 顶格预留 95.00 GiB 无溢出；`state allocated (maxctx=262144)`、graph capture、vision tower（333 tensors）、`listening on :8730` 全部通过 |
| socket 协议（PING/INFO/GEN，`tools/winprobe/serve_probe.py`） | GEN 输出 token 序列与 CLI **逐一致**（[4 5 6 24218 10 4838 1665 15]） |
| OpenAI API 全栈验收（`tools/api_live_smoke.py` → `gdec-api-win` → 引擎 8730） | **RESULT PASS**（8/8：health / models / 400 参数校验 / 非流式内容 / 流式与 stop 等价 / logprobs / responses 非流式 + SSE 事件序列） |
| hipGraph | decode graphs captured（ple=1），Windows 可用 |
| hipBLASLt MoE 路径（GDEC_MOE_LT=1） | 随冒烟通过 |
| 32K prefill 实测（32768 随机 id，`--tokens-file`，maxctx=40960） | chunk=8192：**~920 tok/s**（4 chunk 逐次 913.9/936.9/922.3/907.7，无衰减）；chunk=16384：**~979 tok/s**（987.6/970.4，arena 91.95 GiB 无截断）。**chunk 翻倍仅换 ~6% PP**——8192 的 GEMM 效率已接近饱和，256K 下被 95 GiB 上限锁在 8192 的代价很小 |

关键实现（相对 Linux 版的差异点）：

- **单 arena 内存架构**：`g_devarena` + `dalloc_arena`（bump 分配，溢出回退
  hipMalloc 并告警一次），所有大额 device 内存（权重/KV/索引/prefill 工作区）
  装进进程第一笔分配。`devarena_estimate` 镜像 GpuModel 全部分配公式，
  预留 = 估算 + slack（`GDEC_ARENA_SLACK_GB`，默认 4 GiB），封顶
  `GDEC_ARENA_CAP_GB`（默认 **95.0**，对应 96 GiB BIOS 划分的实测上限）。
  Windows 默认 prefill chunk 8192（`eff_maxbatch`；Linux 32768）。
  **256K 下不要设 `GDEC_PREFILL_CHUNK=16384`**（会触发截断 + 溢出回退）；
  maxctx ≤ 40K 时 16384 放得下且 PP 快 ~6%（见验证矩阵实测）。
- **新权重加载器**（`load_arena` Windows 分支）：CreateFile
  （OVERLAPPED|NO_BUFFERING）+ 4 线程 × 4 槽 × 16 MiB pinned + overlapped
  ReadFile 直读 + H2D 直灌，专家权重同样进 arena。实测 **68 GiB / ~9 分钟，
  稳定 ~130 MB/s**（本机 NVMe 弱，磁盘 bound）；旧 mmap+memcpy 路径在 Windows
  页缓存逐出下 54 分钟且末期退化到 ~18 MB/s。
- **rocBLAS kernel db**：TheRock 不随包提供主索引 `TensileLibrary.dat`，
  rocBLAS 会回退枚举 `<exe目录>/rocblas/library/gfx1151/` 分片目录——目录
  在进程启动前就必须可枚举。缺主索引的 "Cannot read TensileLibrary.dat"
  仅为告警。**最终形态是拷真身而非 junction**：只拷 gfx1151 分片
  （rocblas 17M + hipblaslt 13M，全架构则是 703M+530M），`build_win.sh`
  幂等固化；拷完 build/ 自包含，TheRock 可删（已做改名断根实测）。
- **PLE 留盘按需读**：io_uring → IOCP（PleWin，DEPTH=512，自由槽队列），
  prefill 批量读走 `PLE_PUMP` 宏，两平台主体代码同形。
- **socket/serve**：WinSock2 shim（`sock_*`/`os_*`），协议零改动。
- **API 前端**（`src/api` → `gdec-api-win.exe`）：http.cpp / engine_client.cpp
  的 socket 层走同一套 `os_win32.h` shim（`sock_t` 类型、WSAPoll、
  `SO_RCVTIMEO` 用 DWORD 毫秒）；用 TheRock 自带裸 clang++ 编译（无需
  hipcc、零 HIP 依赖），仅链系统 `ws2_32`，nlohmann/json 由 TheRock
  include 提供。注意 WinSock 陷阱：`peer_gone` 探测必须在 recv 失败后
  **立即**取走 `WSAGetLastError()` 再做 `ioctlsocket` 恢复阻塞——成功
  调用会改写 last-error，误判"客户端断开"导致流式请求秒断（已修）。
  图片解码用 stb_image（public domain 单头文件，`src/api/vendor/`）支持
  PNG/JPEG，编译进 exe 零新增 DLL，PNG 图片问答端到端实测通过；WebP
  无等价单头实现，明确报错，是唯一剩余缺口。
- **origami.dll** 是 `libhipblaslt.dll` 的静态依赖，必须随 exe（lld 链接
  报错不点名它，容易漏）。

未尽事项：API 前端已移植（`gdec-api-win.exe`，图片解码经 stb_image 支持
PNG/JPEG 并实测通过，仅 WebP 明确报错未接）；256K 满填 prefill 未实测
（内存布局已验证，全量 256K token 灌入在本机需数小时，意义有限）；
长跑稳定性（WDDM 换页、`HSA_ENABLE_SDMA=0` 等开关）待观察。

## 结论（第二版）

**移植可行**。原判断的两堵墙均已找到通路：

| 原判断的墙 | 实测结果 |
|---|---|
| gfx1151 无 Windows ROCm 支持 | **不成立**。HIP SDK 7.2 与 TheRock 10.0 均原生支持：`hipInfo` 枚举出 8060S（gfx1151，20 CU，107.87 GiB 可见显存），kernel 正常编译执行 |
| WDDM 不允许 65 GiB 锁页直读 | **绕过**。BIOS 划分 96 GiB 显存后，单次大额 `hipMalloc`（实测至 88 GiB）全部落在真 VRAM，带宽 ~235 GB/s（与 Linux 直读 ~220 GB/s 持平），数据完整性校验通过。不再需要 `hipHostRegister` 架构 |

移植形态：权重全部进**单一 device arena**（启动时第一笔分配），PLE 表留盘、
Windows 文件 IO 按需聚集。预期性能与 Linux 版持平（同一块 DRAM，实测带宽一致）。

## 实测记录

环境：HIP SDK 7.2（`C:\Program Files\AMD\ROCm\7.2`）+
TheRock 10.0（`C:\therock-dist-windows-multiarch-10.0.0`），
Adrenalin 驱动 32.0.22018.5（2025-09-17）。主机另有 RTX 2080 Ti（不参与 HIP 枚举）。

### 工具链与库（全部就绪）

- 编译：两套工具链均可编 gfx1151。`gdec.cpp` 用 SDK hipcc 试编，**唯一
  阻断错误是 `sys/mman.h` 不存在**（加 `-std=c++17` 后 rocPRIM 等全部通过）；
  127 个 kernel 的 warp intrinsic / 内联汇编为 ISA 级，原样可编。
- 运行库：rocBLAS SGEMM 实测 3.2 TFLOPS 结果正确；hipBLASLt bf16 GEMM
  正常（heuristic 命中）。两者 + rocPRIM 头文件在 SDK 7.2 与 TheRock 中都有，
  TheRock 自带 gfx1151 的 Tensile kernel db（`bin/rocblas`、`bin/hipblaslt`）。
- 探针：`tools/winprobe/blassmoke.cpp`。

### 显存分配（移植成败的关键，已探明规则）

Windows 上 HIP 分配的内存路由有明确规则（与上游 issue
[ROCm#5940](https://github.com/ROCm/legacy-rocm-build/issues/5940) 一致）：

1. **进程累计 VRAM 分配 < 32 GiB 时，分配落在 VRAM**；超过后新分配路由到
   shared memory（主机内存，本机仅 ~21 GiB 空闲），继续硬撑会被换页或失败。
2. **单次大额分配整体路由**：进程内第一笔就申请大额 → 全额 VRAM。
   实测（TheRock 运行时，`tools/winprobe/bigalloc.cpp`）：

   | 单次 hipMalloc | 结果 | 读带宽 | 完整性 |
   |---|---|---|---|
   | 40–95 GiB | OK | 223–237 GB/s | 通过 |
   | 96 GiB | out of memory | — | — |

   即单进程可用上限 **~95 GiB**——96 GiB 划分几乎全量可用（驱动仅保留 ~1 GiB）。
3. 反例（必须避免的分配模式）：4 GiB 分块连续分配，~44 GiB 后越界——
   SDK 7.2 运行时**静默丢数据**（写后读回损坏，`integrity.cpp` 实测 30 块损坏），
   TheRock 运行时 loud fault。**两份运行时都不允许这种模式，只有
   "启动即单一大块" 是安全路径**。双进程并发各持 40 GiB 完整性通过
   （`holdprobe.cpp`），证明限制是按进程路由而非全局不足。
4. 进程内 `hipFree` 后预算不完全回收（碎片化）；引擎本就一次性分配到底，无碍。

### 内存预算（heretic.hgn，115.5 GiB 文件）

| 内容 | 大小 | Windows 归宿 |
|---|---|---|
| 专家权重 `.mlp.experts.*` | ~65 GiB | arena（启动 memcpy 进入，替代 hipHostRegister） |
| 主干权重 | ~2.8 GiB | arena（与 Linux 相同） |
| KV cache + QSA 索引 @256K（bf16 KV） | ~9.4 GiB | arena |
| prefill 工作区（随 `GDEC_PREFILL_CHUNK` 缩放） | 27.6 / 13.8 / 6.9 GiB @chunk 32768/16384/8192 | arena |
| MTP sidecar + vision tower + 杂项 | ~1.5 GiB | arena |
| **arena 合计 @256K** | **106 / 92.6 / 85.7 GiB** | 上限 ~95 GiB：chunk 16384 紧（余量 2.4），**8192 从容（余量 ~9）** |
| PLE 表（dtype 10） | 47.7 GiB | **留盘**，每 token 16×160 B 按需读（与 Linux 相同架构；io_uring 换成 ReadFile/overlapped IO） |

**关键约束：所有大额 device 内存必须装进"进程第一笔分配"的单一 arena**
（阈值规则见上：第二笔大额分配时累计已 >32 GiB，会被路由到 shared memory）。
实现上把 `load_arena` 的"先求和、后一次性 hipMalloc、内部子分配"模式扩展到
KV/索引/prefill 工作区即可；rocBLAS/Lt 的句柄工作区（~64 MiB 级）在 arena 之后
创建、落到 shared memory，量小无碍。

256K 上下文本身（bf16 KV 6.4 GiB + 索引 ~3 GiB）不是瓶颈；瓶颈是 prefill
工作区。Linux 默认 chunk=32768 是按 122 GiB 系统内存选的，Windows 下用
`GDEC_PREFILL_CHUNK=8192`（引擎已内置此开关，8192 仍是足够大的 GEMM，
prefill 效率基本不受损）。若 `MAX_CONTEXT=131072`，KV 减半，chunk 16384 也宽裕。

PLE 每 token 仅 2.5 KB 随机读（行地址是 token 历史的哈希取模，**无顺序局部性**，
`gdec.cpp:8266`）；Windows 页缓存吸收热行，冷行走 NVMe。decode 每 token 16 行
≈ 最坏 +1 ms（周期 18–33 ms），无碍；**prefill 每 chunk 13.1 万次随机 4K 读
（512 MB 等效），必须用 overlapped IO 批量并发提交（QD≥32，~0.3–0.5 s/chunk），
禁止逐行同步 ReadFile（QD1 需 5+ s/chunk）**。这是 io_uring 路径在 Windows 的
正确等价物。

## 代码移植清单（在实测之后重新评估）

| 项 | 规模 | 方案 |
|---|---|---|
| `sys/mman.h`、`mmap`/`madvise`/`munmap` | ~20 处 | 权重映射改 `MapViewOfFile` 或直接流式读入 arena；`madvise(WILLNEED)` 预取删（PLE 按需读路径替代）；KV 快照用普通 `fread`/`fwrite` |
| io_uring PLE 聚集 | ~150 行 | `ReadFile` + `OVERLAPPED` 偏移批量读，落到 pinned staging（语义与 Linux 路径逐字节一致） |
| 专家权重 `hipHostRegister` 直读 | `load_arena` 一处分支 | Windows 下改为与普通张量一样进 arena（即恢复旧 all-in-arena 形态，仅 Windows 生效） |
| POSIX socket（serve 模式 + `src/api/http.cpp` 等） | ~10 处 | WinSock2（API 相近，机械替换） |
| `clock_gettime`/`localtime_r`/`unistd.h` 杂项 | 少量 | 标准替代 |
| `build.sh` | 单脚本 | `build.bat`/CMake：TheRock hipcc + `-O3 -std=c++17 --offload-arch=gfx1151`，链接 `rocblas.lib` + `libhipblaslt.dll.a`（注意 hipcc 包装器不接受 `-l:` 语法与裸 `.lib` 路径，需改名为 `hipblaslt.lib`） |
| API 前端图片库 | libpng/jpeg/webp | vcpkg 或预编译，与引擎解耦，可后置 |
| `tools/run_capped.sh` | — | 无需移植（本机部署工具） |

## 部署形态（已验证，自包含）

**build/ 拷走即用，目标机器无需安装任何 ROCm/TheRock 运行库**（已断根实测：
TheRock 目录改名、HIP_PATH/ROCM_PATH 清空后 ktest 仍 ALL PASS）。运行期全部
依赖都在 exe 旁：

- TheRock DLL ×6：`amdhip64_7.dll`、`rocm_kpack.dll`、`amd_comgr.dll`、
  `rocblas.dll`、`libhipblaslt.dll`、`origami.dll`（exe 目录优先于
  System32——**必须屏蔽 System32 里 SDK 7.2 的旧 DLL**，旧 DLL 单分配上限
  41 GiB 且越界静默损坏）。ldd 确认无其他隐藏依赖。
- MSVC 运行时 ×3：`msvcp140.dll`、`vcruntime140.dll`、`vcruntime140_1.dll`
  （官方可再分发，覆盖未装 VC++ Redistributable 的裸机）。
- kernel db 真身：`rocblas/library/gfx1151/`（17M）+
  `hipblaslt/library/gfx1151/`（13M），exe 相对路径解析，与 cwd 无关。
- `gdec-api-win.exe`（OpenAI API 前端）零额外依赖：仅链系统 `ws2_32`，
  随 build/ 一起拷走即用；图片解码未接（文本全功能）。
- 运行期不需要 device bitcode（无 JIT：引擎 kernel 编译期内嵌，rocBLAS/Lt
  直接载 hsaco/.co），`HIP_PATH`/`ROCM_PATH` 均不必设。
- 建议引擎初始化时加 arena 自验（写读校验 pattern），对驱动回归一票否决。

## 风险与未尽事项

1. **分配顺序敏感**：arena 必须是进程第一笔大额分配；之后不再有大额
   device 分配（>32 GiB 阈值后会路由到 shared memory）。KV 快照重建等
   路径审查时注意。
2. 启动变慢：68 GiB 权重需真实读盘进 arena（NVMe bound，估分钟级），
   Linux 版 mmap+注册近乎即时。
3. 长跑稳定性未测：WDDM 在主机内存紧张（仅 32 GiB）时的换页行为、
   社区建议的 `HSA_ENABLE_SDMA=0`/`HSA_USE_SVM=0` 稳定性开关待验证。
4. 96 GiB BIOS 划分保持不变即可；无需 VGM 调整（单次大额分配路径不经过
   其限制）。2080 Ti 不影响 HIP 设备枚举。
5. ~~API 前端未动~~ 已移植（见上节）；剩余唯一功能缺口是 WebP 解码，
   systemd 工具链不适用 Windows。

## 下一步（全部完成，留存备查）

1. ~~写 Windows 兼容 shim（mman/socket/unistd）+ arena 化改造 `load_arena`，
   编译 `gdec.cpp` 通过。~~ ✅ `src/gpu/os_win32.h` + 单 arena 架构
2. ~~加载 heretic.hgn 冒烟：CLI 短生成，对拍 Linux 基线输出。~~ ✅ 出字正确，decode 32.5 tok/s
3. ~~PLE 的 Windows 批量读路径 + 性能对拍（prefill/decode tok/s）。~~ ✅ IOCP 路径随冒烟/serve 通过
4. ~~kernel 单测（`tools/ktest.cu`）在 Windows 全绿。~~ ✅ ALL PASS
5. ~~API 前端与服务化收尾。~~ ✅ `gdec-api-win.exe` + `start_win.sh` 双进程，
   `api_live_smoke.py` 8/8 PASS，多模态（PNG 图片问答）端到端实测通过；
   剩余可选：WebP 解码（需 libwebp 预编译库）

实际工作量集中在 arena 架构改造与新加载器（NO_BUFFERING 直读）上，
与"人周量级"的估计一致；运行时风险项已全部实测排除或规避。
