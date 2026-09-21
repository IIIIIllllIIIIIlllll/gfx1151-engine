// gdec.cpp — Phase 2: gfx1151 GPU decode path (single-token forward).
// Phase 3a: prefill_batch — batched (GEMM) prefill; GDEC_NOPREFILLBATCH=1
// falls back to the old per-token forward() loop.
// Phase 3c: k_qsa_flash — batched causal flash attention for prefill;
// GDEC_QSA_LOOP=1 falls back to the per-token k_qsa_step loop.
// Phase 3b: k_gdn_chunk — chunked (64) GDN prefill, fp32;
// GDEC_GDN_LOOP=1 falls back to the per-token recurrence loop.
// Phase 3b-split: k_gdn_intra (per chunk-head, state-independent) +
// k_gdn_inter (per-head serial d-tile pass) replace k_gdn_chunk by default;
// GDEC_GDN_NOSPLIT=1 restores the single-kernel path.
// Phase 3b-strip: k_gdn_inter_strip splits the inter pass into 4 independent
// 32-column S strips per head (grid (4,48), ~25 KiB LDS/block, 2 blocks/CU).
// 32-wide strips quarter the redundant kcd/q/attn2/k DRAM re-reads of the
// original 16-wide version (~1.7x at 32K, bit-identical);
// GDEC_GDN_NOSTRIP=1 restores k_gdn_inter.
// Phase 3b-pipe2: k_gdn_intra_p2 — fp16-resident WMMA intra, 64 chunks per
// block (2.8x over the fp32 intra at P=8192 in proto; fp16 input rounding,
// not bit-exact); GDEC_GDN_PIPE2=1 selects it.
// Phase 3d-1: PLE batched prefill (ple_gpu_b); GDEC_PLE_LOOP=1 falls back
// to the per-token ple_gpu_t loop. MoE fused W4 GEMM (k_moe_w4_up/down read
// Q4C-P codes directly, bf16 v_dot2 with fp32 accumulate); GDEC_MOE_NAIVE=1
// falls back to grouped GEMV.
// Phase 3f: stable GPU MoE routing and 64-token expert tasks.
// GDEC_MOE_HOST_ROUTE=1 / GDEC_MOE_UNTILED=1 restore the respective old paths.
// Sparse prefill stages V in LDS; GDEC_QSA_GLOBAL_V=1 restores global V reads.
// Prefill GR emits cached BF16 inputs; GDEC_PREFILL_UNFUSED=1 restores separate kernels.
// MoE prefill passes BF16 activations directly; GDEC_MOE_FP32_IO=1 restores FP32 staging.
// QSA KV cache stores bf16 (f2bf RNE on write, fp32 reads/compute);
// GDEC_QSA_KV_BF16=1 enables the BF16 KV cache (experiment: measured parity).
// BF16 KV mode also runs sparse prefill through k_qsa_flash_bf16 (bf16 LDS
// tiles + fdot2 scores); the FP32 sparse path is bit-identical to before.
// GDEC_MOE_LT=1: prefill MoE via dequant-to-bf16 + per-expert hipBLASLt GEMM
// (batched-GEMM pipeline; different rounding than the fused W4 kernels, so
// it is opt-in). GDEC_MOE_LT_BF16=1 makes the expert
// GEMM outputs bf16 as well. Both need GPU routing + deterministic mode.
// GDEC_MOE_LT_OVL=1 (needs GDEC_MOE_LT) double-buffers the dequant scratch
// and runs the next 4-expert group's dequant on a side stream overlapping
// the current group's GEMMs; event-chained, bit-identical.
// Phase 3g: self-written bf16 WMMA dense GEMM (k_gemm_wmma) for the four
// prefill projection shapes at P >= 1024, opt-in via GDEC_GEMM_WMMA=1
// (~37-38 vs ~32 TFLOPS hipBLASLt on N=6144..12288 K=2560; N=2560 K=6144
// ~parity-to-+11%). Different K accumulation order than Tensile (not
// bit-identical).
// Phase 3h: tail-merge — a final prefill remainder of <= GDEC_PREFILL_TAIL_SLACK
// (default 1024 on Linux, 0 elsewhere) tokens is absorbed into the previous
// chunk instead of running as its own fixed-cost chunk; batch workspace is
// allocated at maxbatch_cap = maxbatch + slack. GDEC_GDN_WINDOW_CHUNKS now
// also sizes d_gdn_split_ws (was hardcoded 4; window>4 used to be an
// illegal-memory-access bug).
// M-RoPE (vision, stage 1b/1c): requests may carry image grid_thw triples
// (CLI --mrope-grid t,h,w / serve GEN suffix MROPE k t h w ...); the host
// expands them to per-token 3-row positions (reference get_rope_index
// semantics, tools/mrope_ref.py) and fills the k_rope_cs cos/sin table with
// the slot-interleaved frequencies; decode adds the request's rope_delta to
// the rope angle position (*d_pos + *d_rdelta). Text-only requests never
// touch the mrope path (delta 0, device-table fill) and are bitwise
// identical to before. GDEC_MROPE_DUMP=<file> dumps the filled table and
// per-decode-step rope cos/sin (debug; syncs the stream).
// Vision (stage 4): --vision-tower loads the ViT sidecar; GEN requests may
// then carry VIMG k binary patch frames (one per MROPE grid) whose embeds
// are masked-scattered over the image placeholder rows after the embedding
// gather in prefill_chunk (k_img_inject_*, bf16 source). Without the flag
// the engine is text-only and image requests whose placeholders are not
// covered by grid data are rejected, naming the flag).
// All weights (except the 47.7 GiB PLE table) are memcpy'd into one 68 GiB
// hipMalloc arena at startup (4 threads + pinned staging). PLE table rows
// (16 x 160 B per token) are read host-side from the mmap and uploaded
// through a small pinned staging buffer.
//
// Semantics follow reference/modeling_qwen4_exp.py exactly (same as ref.cpp).
// QSA selects 512 complete 4-token blocks plus all incomplete tail tokens.
// GDEC_QSA_DENSE=1 restores the old dense path for performance comparisons.
//
// Build: hipcc -O3 -o gdec gdec.cpp -lrocblas -lhipblaslt
// Usage: gdec <base.hgn> [overlay.hgn ...] --tokens 1,2,3 [--gen N] [--dump]
//        [--maxctx N]   (multiple overlays apply in order; later wins)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <rocprim/block/block_radix_sort.hpp>
#include <rocprim/block/block_scan.hpp>
#include <rocprim/device/device_radix_sort.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include "os_win32.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "../hgn.h"

#include "parts/00_platform.inc"
#include "parts/05_config.inc"
#include "parts/10_ple_io.inc"
#include "parts/20_kernels_gemv.inc"
#include "parts/21_kernels_ple.inc"
#include "parts/22_kernels_prefill.inc"
#include "parts/23_kernels_moe_w4.inc"
#include "parts/24_kernels_moe_lt.inc"
#include "parts/25_kernels_gdn.inc"
#include "parts/30_host_util.inc"
#include "parts/31_vision.inc"
#include "parts/40_model.inc"
#include "parts/49_rckpt.inc"
#include "parts/50_serve.inc"
#include "parts/51_host_cfg.inc"
#include "parts/52_main.inc"
