# gfx1151-engine

*English: [README_EN.md](README_EN.md)*

在单张 AMD Strix Halo APU(gfx1151)上运行 177B MoE 模型的本地推理引擎,
目标模型为 Qwen3.8-Flash-Next(qwen4_exp 架构)及其同结构微调。

68 GiB 量化权重以 4-bit 量化存放在主机内存,GPU kernel 直读,不需要大显存;
整机 122 GiB 内存即可提供 256K 上下文。

## 特性

- **投机解码 chain**:ngram 优先起草、MTP 兜底,逐轮回退;贪心逐位比对,
  采样按目标分布重采样比对,输出分布与串行一致。默认生效,无需参数。
  高度重复内容(代码注释、模板文本)实测显著加速。
- **独立 8-bit MTP 草稿权重** sidecar,接受率高于内置 4-bit 草稿头。
- **视觉**:支持图像输入(OpenAI `image_url`),KV 复用跨轮生效。
- **OpenAI 兼容 API**:流式、工具调用、`/v1/chat/completions`。
- **256K 上下文**,KV 快照跨重启恢复。
- **模型转换工具**:HF safetensors → `.hgn`,量化无需校准数据,
  可分发给自己的微调模型使用(见 CONVERT.md)。

实测(gfx1151,122 GiB 内存):prefill 约 1100–1200 tok/s,decode 约
30–55 tok/s(视投机命中率)。

## 要求

- Linux + ROCm(HIP 7.x),GPU 架构 `gfx1151`;或 Windows + AMD 显卡驱动
  (GPU 需 BIOS 划分显存),见「Windows」一节
- 可用内存 ≥ 100 GiB(权重 68 GiB 锁页 + KV)
- 编译依赖:rocBLAS、hipBLASLt、rocPRIM;API 前端另需
  nlohmann-json、libpng、libjpeg、libwebp

## 快速开始

```bash
bash build.sh     # 编译引擎 + API,输出在 build/
bash start.sh     # 加载模型并启动服务(读取 service.conf)
```

模型文件默认从 `./models` 读取,配置集中在 `service.conf`。
详见 [QUICKSTART.md](QUICKSTART.md)。

自有微调模型(HF safetensors,同架构)转换:

```bash
python3 tools/flashnext2hgn.py /path/to/hf-model --out ./models
```

详见 [CONVERT.md](CONVERT.md)。

## Windows

Windows 版与 Linux 版功能一致(引擎 + OpenAI API + 多模态),移植记录与
实测见 [PORTING-WINDOWS.md](PORTING-WINDOWS.md)。编译在 Git Bash 中执行
(或双击 `build_win.bat`,仅编译期需要 Git):

```bash
bash build_win.sh           # 引擎
bash build_win.sh api       # OpenAI API 前端
bash build_win.sh launcher  # 免脚本启动器 start_win.exe
```

日常运行双击 `start_win.exe`(原生 Win32,不需要 Git/PowerShell):拉起引擎
+ API 双进程,输出实时显示并写入 `logs\`,Ctrl+C 或关窗停止。配置与 Linux
**共用 `service.conf`**(换模型文件名、改上下文窗口都编辑它),环境变量可
临时覆盖。客户端连 `http://<主机>:8731/v1`。

分发:`build/` + `start_win.exe` + `models/` 拷到任意 gfx1151 Windows
机器即用,**无需安装 ROCm/TheRock**;仅需 AMD 显卡驱动,并在 BIOS 为 GPU
划分足够显存(256K 上下文需 96 GiB)。差异:图片解码经 stb_image 支持
PNG/JPEG(WebP 未接);prefill chunk 默认 8192;冷加载为整权重读盘
(分钟级,进度见控制台/日志);启动器未开 `GDEC_GEMM_WMMA` 与
`GDEC_GDN_FUSED`(Linux start.sh 已转正的自写 WMMA GEMM 与 GDN 融合
kernel,合计约 8-10% PP,TheRock 下未验证——故 Windows 端 prefill 走
hipBLASLt + 旧 GDN 路径)。编译细节见 [BUILD.md](BUILD.md)。

## 文档

- [QUICKSTART.md](QUICKSTART.md) — 编译、启动、配置
- [BUILD.md](BUILD.md) — 编译环境细节与排错
- [CONVERT.md](CONVERT.md) — 模型转换工具
- [MTP.md](MTP.md) — 投机解码参数与对比方法
- [NGRAM.md](NGRAM.md) — ngram 验证的设计、收益与已知分歧
- [HGN-FORMAT.md](HGN-FORMAT.md) — `.hgn` 权重容器格式
- [data/README.md](data/README.md) — 数值回归基准(data/qsa-oracle)说明
- [PORTING-WINDOWS.md](PORTING-WINDOWS.md) — Windows 移植记录与实测

## 测试

```bash
bash build.sh test   # kernel 单测,不加载模型,预期 ALL PASS
```

## 致谢

本项目的实现方式借鉴了 peonist-ai 的 [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server);`.hgn` 权重容器格式即 halogen 的 checkpoint 容器格式(见 [HGN-FORMAT.md](HGN-FORMAT.md))。感谢 halogen 作者的工作。

## 许可证

[AGPL-3.0](LICENSE)
