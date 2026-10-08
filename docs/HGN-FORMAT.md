# .hgn 权重格式

*English: [HGN-FORMAT_EN.md](HGN-FORMAT_EN.md)*

`.hgn` 是 halogen 推理引擎([halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server))的 checkpoint 容器格式;本项目为加载该格式
的权重文件(以及用 `tools/flashnext2hgn.py` 转换自有模型)实现了对该
容器的读写,仅涉及文件格式层面的互操作。容器结构(magic、头部、张量
表)沿用 halogen 的定义;量化布局以本仓实现为准,权威定义见
`src/hgn.h`(读)和 `tools/flashnext2hgn.py`(写)。

## 文件头(104 字节)

| 偏移 | 类型 | 字段 |
|---|---|---|
| 0x00 | char[4] | magic `"HGN1"` |
| 0x04 | u32 | version(1 或 2) |
| 0x08 | u32 | 张量数量 |
| 0x0c | u32 | 保留(0) |
| 0x10 | u64 | 张量表偏移(通常为 0x68) |
| 0x18 | u64 | 数据区起始偏移 |
| 0x20 | u64 | 文件总大小(须等于实际大小) |
| 0x28 | char[64] | 模型名(NUL 结尾) |

## 张量表

`tensor_count` 条记录,每条 160 字节:

| 偏移 | 类型 | 字段 |
|---|---|---|
| 0x00 | char[96] | 张量名(NUL 结尾,≤95 字符) |
| 0x60 | u32 | dtype(见下表) |
| 0x64 | u32 | 维度数 ndims(1–4) |
| 0x68 | u64[4] | dims(按 dtype 解释) |
| 0x88 | u64 | 数据在文件内的偏移 |
| 0x90 | u64 | 数据字节数 |
| 0x98 | u64 | extra(量化参数,通常为 0) |

## dtype 表

| id | 名称 | 布局 |
|---|---|---|
| 0 | bf16 | 每元素 2 字节,直通 |
| 4 | i64 | u64/i64 数组(PLE 配置等元数据) |
| 5 | q4cp | `[64B: 16 个 fp32 码本][rows*cols/2 B: 4-bit 码,行主序,低半字节在前][scale 记录:每行 cols/32 个 fp16,记录补齐到 16B]`;`w[r,c] = cb[nib] * scale[r][c/32]`。3D 融合专家张量按 `[E*rows, cols]` 平铺,全张量共享一个码本 |
| 7 | q8g64 | 每行:`[cols 个 uint8 码][cols/64 组 (fp16 scale, fp16 min)]`;`w = code*scale + min` |
| 10 | fp8-e4m3 | `[numel 个 uint8 码][末尾 4B fp32 全局 scale]`;`w = e4m3(code) * scale` |

## 文件家族

一次完整部署由若干独立 `.hgn` 文件组成,引擎按需叠加加载:

- `<name>.hgn` — 主权重(48 层主干 + q4cp 精度的 MTP 兜底草稿头)
- `<name>.overlay.hgn` — 覆盖层(可选;q8g64 等更高精度的替换张量,
  逐名覆盖主权重)。`tools/flashnext2hgn.py` 默认生成:723 个稠密线性
  层(不含 embed_tokens / MoE 专家 / mtp.*),其中 12 个全注意力层
  o_proj 为 q8g64,其余为 scale 经 MSE 优化的 q4cp;`--overlay-speed`
  另生成全 q4cp 的 `<name>.overlay-speed.hgn`
- `<name>-mtp.hgn` — 独立 8-bit MTP 草稿权重 sidecar(可选;作为引擎
  第二个位置参数传入,替代内置 4-bit 草稿头,接受率更高)
- `<name>-vision.hgn` — 视觉塔(可选;`--vision-tower` 加载)

## 检查工具

```bash
build/hgn_dump <file.hgn>   # 打印头部与张量表(src/hgn_dump.cpp)
```
