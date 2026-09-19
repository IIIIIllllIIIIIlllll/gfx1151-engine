# .hgn Weight Format

*中文版:[HGN-FORMAT.md](HGN-FORMAT.md)*

`.hgn` is the checkpoint container format of the [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) inference engine. This project implements reading and writing of that container in order to load weight files in this format (and to convert its own models with `tools/flashnext2hgn.py`); the interoperability is purely at the file-format level. The container structure (magic, header, tensor table) follows halogen's definition; the quantization layouts are authoritative in this repository — see `src/hgn.h` (reading) and `tools/flashnext2hgn.py` (writing).

## File Header (104 bytes)

| Offset | Type | Field |
|---|---|---|
| 0x00 | char[4] | magic `"HGN1"` |
| 0x04 | u32 | version (1 or 2) |
| 0x08 | u32 | tensor count |
| 0x0c | u32 | reserved (0) |
| 0x10 | u64 | tensor table offset (usually 0x68) |
| 0x18 | u64 | start offset of the data region |
| 0x20 | u64 | total file size (must equal the actual size) |
| 0x28 | char[64] | model name (NUL-terminated) |

## Tensor Table

`tensor_count` records, 160 bytes each:

| Offset | Type | Field |
|---|---|---|
| 0x00 | char[96] | tensor name (NUL-terminated, ≤95 characters) |
| 0x60 | u32 | dtype (see table below) |
| 0x64 | u32 | number of dimensions ndims (1–4) |
| 0x68 | u64[4] | dims (interpreted per dtype) |
| 0x88 | u64 | data offset within the file |
| 0x90 | u64 | data size in bytes |
| 0x98 | u64 | extra (quantization parameter, usually 0) |

## dtype Table

| id | Name | Layout |
|---|---|---|
| 0 | bf16 | 2 bytes per element, passthrough |
| 4 | i64 | u64/i64 array (metadata such as PLE configuration) |
| 5 | q4cp | `[64B: 16 fp32 codebook][rows*cols/2 B: 4-bit codes, row-major, low nibble first][scale records: cols/32 fp16 per row, record padded to 16B]`; `w[r,c] = cb[nib] * scale[r][c/32]`. A 3D fused-expert tensor is flattened as `[E*rows, cols]`, with a single codebook shared across the whole tensor |
| 7 | q8g64 | per row: `[cols uint8 codes][cols/64 groups of (fp16 scale, fp16 min)]`; `w = code*scale + min` |
| 10 | fp8-e4m3 | `[numel uint8 codes][final 4B fp32 global scale]`; `w = e4m3(code) * scale` |

## File Family

A full deployment consists of several independent `.hgn` files, which the engine stacks and loads as needed:

- `<name>.hgn` — main weights (48-layer backbone + an MTP fallback draft head in q4cp precision)
- `<name>.overlay.hgn` — overlay (optional; replacement tensors in higher precision such as q8g64)
- `<name>-mtp.hgn` — standalone 8-bit MTP draft weights sidecar (optional; passed as the engine's second positional argument, replacing the built-in 4-bit draft head for a higher acceptance rate)
- `<name>-vision.hgn` — vision tower (optional; loaded with `--vision-tower`)

## Inspection Tool

```bash
build/hgn_dump <file.hgn>   # prints the header and tensor table (src/hgn_dump.cpp)
```
