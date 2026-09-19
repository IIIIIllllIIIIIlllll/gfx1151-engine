# 转换自有模型(flashnext2hgn)

`tools/flashnext2hgn.py` 把 HuggingFace safetensors 格式的
Qwen3.8-Flash-Next(qwen4_exp 架构)模型/微调版转换成本引擎的 `.hgn`
权重。单文件脚本,只依赖 Python 3 + numpy(不需要 torch)。

```bash
python3 tools/flashnext2hgn.py /path/to/hf-model --out /path/to/outdir
```

产出(outdir 内):

- `<name>.hgn` — 主权重(q4cp 4-bit 线性层 + bf16 norm/路由/卷积 +
  fp8 PLE 表 + q4cp MTP 兜底),约 116 GiB
- `<name>-mtp.hgn` — MTP 8-bit sidecar(q8g64),启动时作为第二个位置
  参数传给引擎即升级草稿头精度
- `<name>-vision.hgn` — 视觉塔(全 bf16),`--vision-tower` 使用
- `tokenizer/` — 从源目录复制的 tokenizer 文件
- `start.sh` — 一键启动脚本(引擎 + OpenAI API,含生产环境变量
  与内存上限看护),转换时自动生成

启动:

```bash
bash /path/to/outdir/start.sh
```

就绪后 API 在 `http://127.0.0.1:8731/v1`。可用环境变量覆盖:
`ENGINE_PORT`(默认 8730)、`API_PORT`(8731)、`MAXCTX`(262144)、
`GAMMA`(MTP 草稿长度,3)、`MEMCAP_GB`(内存上限,86)。
脚本里烘焙的是转换时 `--engine-bin` 指向的引擎目录(默认
`本仓库/build`);引擎挪了位置就用 `--start-script-only --engine-bin
<新build目录>` 重新生成,不用重转模型。

## 说明

- **只支持 qwen4_exp 这一种形状**(48 层 / hidden 2560 / 512 专家 /
  24 头 / vocab 248320 / MTP 1 层 / 视觉 27 层):引擎的维度是硬编码的,
  工具会校验 `config.json`,形状不符直接拒绝。其它的 Flash-Next 微调
  (只动权重不动结构)都可以转。
- **量化无需校准数据**:q4cp = 每张量 16 级 Lloyd 码本 + 32 列组
  absmax(fp16 scale);MTP sidecar = q8g64(64 列组仿射);PLE 表 =
  fp8 e4m3 全局 scale。全部是数据无关的 RTN,质量与 GGUF 的 Q4_K 同级。
- 转换全程约 1-2 小时(177B 参数,纯 CPU),峰值内存 ~12 GiB,
  输出约需 120 GiB 磁盘。
- `--skip-vision` / `--skip-ple` / `--skip-mtp-sidecar` 可裁掉对应产物;
  `--dry-run` 只校验映射并打印产出计划;`--quant-selftest` 跑量化器
  数值自检。
- base 文件内含 q4cp 精度的 MTP 权重兜底:不传 sidecar 也能跑投机
  解码,传了 sidecar 草稿质量更好(接受率更高)。
