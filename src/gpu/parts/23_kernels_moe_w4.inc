// ================== fused MoE W4 GEMM (Phase 3d-1) kernels ===================
// Pairs bucketed by expert: expert e owns pairs eoff[e]..eoff[e+1), token of
// pair p = tokidx[p], router weight = pw[p]. Both kernels read Q4C-P codes
// directly (fp32 accum, no bf16 intermediates): block = (expert, MOE_CT-col
// tile), 256 threads with an 8-row x FD_CPT-col accumulator tile each, 64-pair
// row tiles, x staged in LDS in 32-wide K chunks. Weight codes are streamed
// exactly once per expert per launch. Per chunk the block dequantizes the
// code tile ONCE into an fp32 LDS tile ([k][col], padded pitch), so the
// compute loop is pure FMA — per-warp re-decoding of packed codes was the
// bottleneck of the earlier version (measured ~40% of issue slots), and the
// 8-row tile halves LDS bytes per FMA vs 4 rows (LDS bandwidth is the other
// wall: gfx1151 delivers ~1B per fp32 FMA). Numerics: one 32-wide scale
// group per chunk; the chunk partial sums are scaled and folded into the
// accumulators at chunk end, so per-element accumulation order matches the
// grouped GEMV path bit for bit.
// Hard requirements (host-checked): out cols % MOE_CT == 0, K cols % MOE_KC
// == 0 — no runtime guards in the unrolled loops (they defeat unrolling and
// spill the accumulators to scratch).
#define MOE_RT 64
// Output-column tile. 128 is the measured optimum on gfx1151: halving to 64
// halves per-block accumulators/LDS and doubles grid.x, but each 64-row block
// still runs the full K-chunk loop, so x staging and barrier count per unit
// output double — measured 2026-09-09 (tools/MOE_OCC.md): 32K up +12%, down
// +30% slower; occupancy gain does not compensate. Rebuild with
// -DGDEC_MOE_CT64 to rerun that experiment (compile-time switch; the GDEC_*
// runtime fallbacks are unaffected).
#ifdef GDEC_MOE_CT64
#define MOE_CT 64
#else
#define MOE_CT 128
#endif
#define MOE_KC 32             // one 32-wide scale group per chunk
#define MOE_XP 36             // xs row pitch (32 + pad against store conflicts)
#define MOE_WP (MOE_CT + 4)   // wf row pitch (pad: conflict-free stage+compute)
// fdot2_bf16 variant tiling (xs/wf staged as bf162 pairs along k)
#define FD_RT 64
#define FD_CT MOE_CT
// K chunk. Default: 32-wide single-buffered dual-barrier loop. The 16-wide
// double-buffered single-barrier variant (-DGDEC_MOE_KC16DB, stage of chunk
// i+1 overlaps compute of chunk i, bit-identical 32-wide scale fold via
// chunk partials living across two 16-chunks) was MEASURED A REGRESSION
// (2026-09-09: packed up +13%, packed down +8% @32K — halving per-chunk
// compute makes the per-chunk stage/loop overhead dominate, barrier parity
// notwithstanding) and is kept opt-in, like GDEC_MOE_CT64.
#ifndef GDEC_MOE_KC16DB
#define FD_KC 32
#define FD_XP 20              // xs2 row pitch in bf162 (80B, 16B-aligned)
#else
#define FD_KC 16
#define FD_XP 12              // xs2 row pitch in bf162 (48B, 16B-aligned)
#endif
#define FD_WP (FD_CT + 4)     // wf2 row pitch in bf162 (bank-rotating)
#define FD_CPT (FD_CT / 32)   // output columns per thread
// Down-only wide column tile experiment: -DGDEC_MOE_DOWN_CT256 (d=2560 is
// divisible by 256; moe_mid=640 is not, so up always keeps FD_CT).
#ifdef GDEC_MOE_DOWN_CT256
#define FD_CT_DN 256
#else
#define FD_CT_DN FD_CT
#endif
#define FD_WP_DN (FD_CT_DN + 4)
#define FD_CPT_DN (FD_CT_DN / 32)
#if defined(GDEC_MOE_DOWN_CT256) && !defined(GDEC_MOE_KC16DB)
#error "GDEC_MOE_DOWN_CT256 requires the KC16 double-buffered loop (256 columns exceed the 256-thread KC32 staging layout)"
#endif
// Min blocks/CU hint: CT=64 targets 4 resident blocks, CT=128 fits 2.
#ifdef GDEC_MOE_LB2
#define FD_MINB 2
#else
#define FD_MINB (FD_CPT == 2 ? 4 : 2)
#endif

// up: hid[p*mid + c] = silu(g)*u, g = W_e[c].x[tokidx[p]], u = W_e[mid+c].x[.]
// (W_e = [2*mid, cols] Q4C-P). x rows stride cols. Grid is (col-tile, expert)
// so same-expert blocks launch adjacently and share the gathered x rows
// through L2. The next chunk's global loads (codes/scales) are issued right
// after the LDS stores so their latency hides behind the dot-product loop.
// Compute: v_dot2_f32_bf16 (fp32 accum) over bf162 pairs staged in LDS;
// activations and dequantized cb weights are rounded to bf16 on staging
// (bf16-level precision, as elsewhere in the server). The 32-group scale is
// still applied to fp32 chunk partials, keeping the accumulation structure of
// the grouped GEMV path.
// Packed reads cached BF16 x and writes BF16 hidden values; the old down kernel
// performed exactly this rounding when staging the FP32 hidden values.
template <bool Tiled = false, bool Packed = false,
          bool RowDot = Packed, bool PairTable = Packed>
__global__ void __launch_bounds__(256, FD_MINB)
    k_moe_w4_up(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                const float* __restrict__ cb, const float* __restrict__ x,
                const int* __restrict__ tokidx, const int* __restrict__ eoff,
                float* __restrict__ hid, int rows_per, int cols, int mid,
                uint64_t scale_stride, const MoeTile* tiles = nullptr, const int* ntiles = nullptr,
                const uint16_t* packed_x = nullptr, uint16_t* packed_hid = nullptr) {
  if (Tiled && blockIdx.y >= *ntiles) return;
  const int e = Tiled ? tiles[blockIdx.y].expert : blockIdx.y, c0 = blockIdx.x * FD_CT;
  const int n0 = Tiled ? tiles[blockIdx.y].first : eoff[e];
  const int ne = Tiled ? tiles[blockIdx.y].count : eoff[e + 1] - n0;
  if (!ne) return;
#ifndef GDEC_MOE_KC16DB
  // One byte selects two already-rounded codebook values. The first token-map
  // barrier below publishes the table before any weight tile consumes it.
  __shared__ __hip_bfloat162 cb_pairs[PairTable ? 256 : 1];
  if (PairTable)
    cb_pairs[threadIdx.x] = __halves2bfloat162(
        __float2bfloat16(cb[threadIdx.x & 15]),
        __float2bfloat16(cb[threadIdx.x >> 4]));
  auto decode_pair = [&](uint32_t codes2) {
    if constexpr (PairTable) return cb_pairs[codes2 & 255];
    else return __halves2bfloat162(__float2bfloat16(__ldg(cb + (codes2 & 15))),
                                  __float2bfloat16(__ldg(cb + ((codes2 >> 4) & 15))));
  };
  __shared__ __hip_bfloat162 xs2[FD_RT][FD_XP];
  __shared__ __hip_bfloat162 wfg2[FD_KC / 2][FD_WP];  // gate weights, dequantized
  __shared__ __hip_bfloat162 wfu2[FD_KC / 2][FD_WP];  // up weights, dequantized
  __shared__ float wsg[FD_CT], wsu[FD_CT];            // per-column group scales
#else
  // double-buffered: stage of chunk i+1 overlaps compute of chunk i
  __shared__ __hip_bfloat162 xs2[2][FD_RT][FD_XP];
  __shared__ __hip_bfloat162 wfg2[2][FD_KC / 2][FD_WP];  // gate weights
  __shared__ __hip_bfloat162 wfu2[2][FD_KC / 2][FD_WP];  // up weights
  __shared__ float wsg[2][FD_CT], wsu[2][FD_CT];         // per-column group scales
#endif
  const int cg = threadIdx.x & 31, rg = threadIdx.x >> 5;  // FD_CPT cols x 8 rows
  const uint8_t* ec = codes + (size_t)e * rows_per * (cols / 2);
  const uint8_t* es = scales + (size_t)e * rows_per * scale_stride;
  const size_t rowb = (size_t)cols / 2;
  const size_t uoff = (size_t)mid * rowb;
  const size_t usoff = (size_t)mid * scale_stride;
  for (int r0 = 0; r0 < ne; r0 += FD_RT) {
    const int nr = min(FD_RT, ne - r0);
    float accg[FD_CPT][8] = {}, accu[FD_CPT][8] = {};  // [cc][r]
    __shared__ int tk[FD_RT];  // row -> token, resolved once per row tile
    if (threadIdx.x < FD_RT)
      tk[threadIdx.x] = threadIdx.x < nr ? tokidx[n0 + r0 + threadIdx.x] : 0;
    __syncthreads();
#ifndef GDEC_MOE_KC16DB
    // staging slots: codes row tid/2, half tid%2 (8B = 16 nibbles);
    // with FD_CT=64 only the first 128 threads stage weights (sact).
    const int srow = threadIdx.x >> 1, sseg = threadIdx.x & 1;
    const bool sact = srow < FD_CT;
    const uint8_t* rb = ec + (size_t)(c0 + srow) * rowb + sseg * 8;
    const uint8_t* sb = es + (size_t)(c0 + srow) * scale_stride;
    uint2 pgc, puc;
    float sg, su;
    auto stage_load = [&](int k0) {
      pgc = *(const uint2*)(rb + k0 / 2);
      puc = *(const uint2*)(rb + uoff + k0 / 2);
      sg = __half2float(*(const __half*)(sb + (k0 / 32) * 2));
      su = __half2float(*(const __half*)(sb + usoff + (k0 / 32) * 2));
    };
    if (sact) stage_load(0);
    // x gather prefetch: this thread's staging slot is row xr, pairs
    // xp4..xp4+3 (16B per chunk). When enabled, the next chunk's global loads
    // are issued right after the LDS stores, beside stage_load, so the gather
    // latency hides behind the dot-product loop. Bit-identical either way:
    // same bytes gathered, same conversions, only the issue point moves.
    // Measured 2026-09-09 (KC32): pays off only on the fp32 path (2x16B
    // loads, up -3.5%); on the packed bf16 path the 16B gather is too cheap
    // to amortize the register live range (packed up +3%), so it is gated
    // off there. -DGDEC_MOE_NO_XPREF disables it everywhere.
#ifdef GDEC_MOE_NO_XPREF
    constexpr bool XPREF = false;
#else
    constexpr bool XPREF = !Packed;
#endif
    const int xr = threadIdx.x >> 2, xp4 = (threadIdx.x & 3) * 4;
    const bool xact = xr < nr;
    uint4 xp0, xp1;
    auto x_load = [&](int k0) {
      if (!xact) return;
      const size_t xoff = (size_t)tk[xr] * cols + xp4 * 2 + k0;
      if (Packed) {
        xp0 = *(const uint4*)(packed_x + xoff);
      } else {
        *(float4*)&xp0 = *(const float4*)(x + xoff);
        *(float4*)&xp1 = *(const float4*)(x + xoff + 4);
      }
    };
    if (XPREF) x_load(0);
    for (int k0 = 0; k0 < cols; k0 += FD_KC) {
      __syncthreads();
      {
        // gather x rows -> bf162 pairs in LDS: 64 rows x 16 pairs = 1024
        // bf162, 256 threads x 4 bf162 (16B) each
        {
          __hip_bfloat162 v[4];
          if (xact) {  // 4 pairs = 8 k values
            if (Packed) {
              if (XPREF) *(uint4*)v = xp0;
              else *(uint4*)v = *(const uint4*)(packed_x + (size_t)tk[xr] * cols + k0 + xp4 * 2);
            } else {
              float4 x0, x1;
              if (XPREF) {
                x0 = *(const float4*)&xp0;
                x1 = *(const float4*)&xp1;
              } else {
                const float* xp = x + (size_t)tk[xr] * cols + k0 + xp4 * 2;
                x0 = *(const float4*)xp;
                x1 = *(const float4*)(xp + 4);
              }
              v[0] = __halves2bfloat162(__float2bfloat16(x0.x), __float2bfloat16(x0.y));
              v[1] = __halves2bfloat162(__float2bfloat16(x0.z), __float2bfloat16(x0.w));
              v[2] = __halves2bfloat162(__float2bfloat16(x1.x), __float2bfloat16(x1.y));
              v[3] = __halves2bfloat162(__float2bfloat16(x1.z), __float2bfloat16(x1.w));
            }
          } else {
            __hip_bfloat162 z = __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
            v[0] = z; v[1] = z; v[2] = z; v[3] = z;
          }
          *(float4*)&xs2[xr][xp4] = *(float4*)v;
        }
        // dequant staged codes into bf162 wf tiles (pairs along k, raw cb
        // values; the 32-group scale is applied to the chunk partial sums)
        if (sact) {
          uint32_t vg0 = pgc.x, vg1 = pgc.y, vu0 = puc.x, vu1 = puc.y;
          if (sseg == 0) {
            wsg[srow] = sg;
            wsu[srow] = su;
          }
#pragma unroll
          for (int j = 0; j < 4; j++) {
            wfg2[sseg * 8 + j][srow] = decode_pair(vg0);
            wfu2[sseg * 8 + j][srow] = decode_pair(vu0);
            vg0 >>= 8;
            vu0 >>= 8;
          }
#pragma unroll
          for (int j = 0; j < 4; j++) {
            wfg2[sseg * 8 + 4 + j][srow] = decode_pair(vg1);
            wfu2[sseg * 8 + 4 + j][srow] = decode_pair(vu1);
            vg1 >>= 8;
            vu1 >>= 8;
          }
        }
      }
      if (k0 + FD_KC < cols) {
        if (sact) stage_load(k0 + FD_KC);
        if (XPREF) x_load(k0 + FD_KC);
      }
      __syncthreads();
      // compute: v_dot2_f32_bf16 into chunk-partial accumulators (one
      // 32-group per chunk), group scale applied at chunk end. Scales are
      // per output column, from LDS.
      float sgv[FD_CPT], suv[FD_CPT];
#pragma unroll
      for (int cc = 0; cc < FD_CPT; cc++) {
        sgv[cc] = wsg[cg * FD_CPT + cc];
        suv[cc] = wsu[cg * FD_CPT + cc];
      }
      float acg[FD_CPT][8] = {}, acu[FD_CPT][8] = {};
#pragma unroll 1
      for (int i = 0; i < FD_KC / 8; i++) {
        __hip_bfloat162 xv[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++)
          *(float4*)xv[r] = *(const float4*)&xs2[rg * 8 + r][i * 4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          __hip_bfloat162 wgv[FD_CPT], wuv[FD_CPT];
          if (FD_CPT == 4) {
            *(float4*)wgv = *(const float4*)&wfg2[i * 4 + kk][cg * FD_CPT];
            *(float4*)wuv = *(const float4*)&wfu2[i * 4 + kk][cg * FD_CPT];
          } else {
            *(uint2*)wgv = *(const uint2*)&wfg2[i * 4 + kk][cg * FD_CPT];
            *(uint2*)wuv = *(const uint2*)&wfu2[i * 4 + kk][cg * FD_CPT];
          }
          // Interleave independent output columns within each token row.
          // Column-major expansion regressed packed up by about 12% on gfx1151.
          if constexpr (RowDot && FD_CPT == 4) {
#pragma unroll
            for (int r = 0; r < 8; r++) {
              acg[0][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[0], xv[r][kk], acg[0][r], false);
              acg[1][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[1], xv[r][kk], acg[1][r], false);
              acg[2][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[2], xv[r][kk], acg[2][r], false);
              acg[3][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[3], xv[r][kk], acg[3][r], false);
              acu[0][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[0], xv[r][kk], acu[0][r], false);
              acu[1][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[1], xv[r][kk], acu[1][r], false);
              acu[2][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[2], xv[r][kk], acu[2][r], false);
              acu[3][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[3], xv[r][kk], acu[3][r], false);
            }
          } else {
#pragma unroll
            for (int cc = 0; cc < FD_CPT; cc++)
#pragma unroll
              for (int r = 0; r < 8; r++) {
                acg[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[cc], xv[r][kk], acg[cc][r], false);
                acu[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[cc], xv[r][kk], acu[cc][r], false);
              }
          }
        }
      }
#pragma unroll
      for (int cc = 0; cc < FD_CPT; cc++)
#pragma unroll
        for (int r = 0; r < 8; r++) {
          accg[cc][r] += sgv[cc] * acg[cc][r];
          accu[cc][r] += suv[cc] * acu[cc][r];
        }
    }
#else  // FD_KC=16 double-buffered, one barrier per chunk
    // staging slots: thread t stages weight column t%FD_CT, gate (t<FD_CT)
    // or up (t<2*FD_CT) — 8B = 16 nibbles per 16-chunk; x slot is row
    // xr=tid/4, pairs (tid%4)*2..+1 (8B per chunk).
    const int srow = threadIdx.x & (FD_CT - 1), sseg = threadIdx.x / FD_CT;
    const bool sact = sseg < 2;
    const uint8_t* rb = ec + (size_t)(c0 + srow) * rowb + sseg * uoff;
    const uint8_t* sb = es + (size_t)(c0 + srow) * scale_stride + sseg * usoff;
    uint2 pk2;
    float sc;
    auto stage_load = [&](int k0) {  // k0 = 16-wide chunk offset
      pk2 = *(const uint2*)(rb + k0 / 2);
      sc = __half2float(*(const __half*)(sb + (k0 / 32) * 2));
    };
    const int xr = threadIdx.x >> 2, xp2 = (threadIdx.x & 3) * 2;
    const bool xact = xr < nr;
    const size_t xoff = (size_t)tk[xr] * cols + xp2 * 2;
    uint4 xp;  // next chunk's x: 8B packed / 16B fp32 (see k_moe_w4_up notes)
    auto x_load = [&](int k0) {
      if (!xact) return;
      if (Packed) *(uint2*)&xp = *(const uint2*)(packed_x + xoff + k0);
      else *(float4*)&xp = *(const float4*)(x + xoff + k0);
    };
    auto x_store = [&](int buf, int k0) {
      __hip_bfloat162 v[2];
      bool got = xact;
#ifdef GDEC_MOE_NO_XPREF
      if (got) {
        if (Packed) {
          *(uint2*)v = *(const uint2*)(packed_x + xoff + k0);
        } else {
          float4 f = *(const float4*)(x + xoff + k0);
          v[0] = __halves2bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
          v[1] = __halves2bfloat162(__float2bfloat16(f.z), __float2bfloat16(f.w));
        }
      }
#else
      (void)k0;
      if (got) {
        if (Packed) {
          *(uint2*)v = *(const uint2*)&xp;
        } else {
          float4 f = *(const float4*)&xp;
          v[0] = __halves2bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
          v[1] = __halves2bfloat162(__float2bfloat16(f.z), __float2bfloat16(f.w));
        }
      }
#endif
      if (!got) {
        __hip_bfloat162 z = __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
        v[0] = z; v[1] = z;
      }
      *(uint2*)&xs2[buf][xr][xp2] = *(uint2*)v;
    };
    auto stage_w = [&](int buf) {  // dequant staged codes into buffer buf
      if (!sact) return;
      __hip_bfloat162(*wfb)[FD_WP] = (sseg ? wfu2 : wfg2)[buf];
      if (sseg) wsu[buf][srow] = sc; else wsg[buf][srow] = sc;
      uint32_t v0 = pk2.x, v1 = pk2.y;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wfb[j][srow] = __halves2bfloat162(__float2bfloat16(__ldg(cb + (v0 & 0xf))), __float2bfloat16(__ldg(cb + ((v0 >> 4) & 0xf))));
        v0 >>= 8;
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wfb[4 + j][srow] = __halves2bfloat162(__float2bfloat16(__ldg(cb + (v1 & 0xf))), __float2bfloat16(__ldg(cb + ((v1 >> 4) & 0xf))));
        v1 >>= 8;
      }
    };
    // prologue: stage chunk 0 into buffer 0 (the tk barrier above is also
    // the WAR barrier against the previous row tile's compute reads)
    if (sact) stage_load(0);
#ifndef GDEC_MOE_NO_XPREF
    x_load(0);
#endif
    stage_w(0);
    x_store(0, 0);
    __syncthreads();
    // The chunk-partial accumulators live across the two 16-chunks of each
    // 32-wide scale group (k ascending), so the scale fold lands on exactly
    // the same sums as the 32-wide chunk loop: bit-identical.
    float acg[FD_CPT][8] = {}, acu[FD_CPT][8] = {};
    const int nchunk = cols / FD_KC;  // even: cols % 32 == 0 (host-checked)
    for (int ci = 0; ci < nchunk; ci++) {
      const int cur = ci & 1, nxt = cur ^ 1, k0 = ci * FD_KC;
      // issue chunk ci+1's global loads; latency hides behind the compute
      if (ci + 1 < nchunk) {
        if (sact) stage_load(k0 + FD_KC);
#ifndef GDEC_MOE_NO_XPREF
        x_load(k0 + FD_KC);
#endif
      }
      float sgv[FD_CPT], suv[FD_CPT];
#pragma unroll
      for (int cc = 0; cc < FD_CPT; cc++) {
        sgv[cc] = wsg[cur][cg * FD_CPT + cc];
        suv[cc] = wsu[cur][cg * FD_CPT + cc];
      }
#pragma unroll 1
      for (int i = 0; i < FD_KC / 8; i++) {
        __hip_bfloat162 xv[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++)
          *(float4*)xv[r] = *(const float4*)&xs2[cur][rg * 8 + r][i * 4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          __hip_bfloat162 wgv[FD_CPT], wuv[FD_CPT];
          if (FD_CPT == 4) {
            *(float4*)wgv = *(const float4*)&wfg2[cur][i * 4 + kk][cg * FD_CPT];
            *(float4*)wuv = *(const float4*)&wfu2[cur][i * 4 + kk][cg * FD_CPT];
          } else {
            *(uint2*)wgv = *(const uint2*)&wfg2[cur][i * 4 + kk][cg * FD_CPT];
            *(uint2*)wuv = *(const uint2*)&wfu2[cur][i * 4 + kk][cg * FD_CPT];
          }
#pragma unroll
          for (int cc = 0; cc < FD_CPT; cc++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
              acg[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[cc], xv[r][kk], acg[cc][r], false);
              acu[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[cc], xv[r][kk], acu[cc][r], false);
            }
        }
      }
      if (ci & 1) {  // end of a 32-wide scale group: fold chunk partials
#pragma unroll
        for (int cc = 0; cc < FD_CPT; cc++)
#pragma unroll
          for (int r = 0; r < 8; r++) {
            accg[cc][r] += sgv[cc] * acg[cc][r];
            accu[cc][r] += suv[cc] * acu[cc][r];
            acg[cc][r] = 0.f;
            acu[cc][r] = 0.f;
          }
      }
      // stage chunk ci+1 into the other buffer (safe: everyone finished
      // reading it at chunk ci-1, separated by the barrier below)
      if (ci + 1 < nchunk) {
        stage_w(nxt);
        x_store(nxt, k0 + FD_KC);
      }
      __syncthreads();
    }
#endif
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int row = rg * 8 + r;
      if (row >= nr) continue;
      size_t hi = (size_t)(n0 + r0 + row) * mid + c0 + cg * FD_CPT;
#pragma unroll
      for (int cc = 0; cc < FD_CPT; cc++) {
        float g = accg[cc][r], u = accu[cc][r];
        float value = g / (1.f + expf(-g)) * u;
        if (Packed) packed_hid[hi + cc] = f2bf(value);
        else hid[hi + cc] = value;
      }
    }
  }
}

// down: acc[tokidx[p]*rows_per + r] += pw[p] * (W_e[r] . hid[p])
// (W_e = [rows_per, cols] Q4C-P). hid rows have stride cols. Same staging
// structure as k_moe_w4_up; single projection, so wf2 + xs2 stay small enough
// for several resident blocks per CU. Weighted outputs either scatter to tokens or
// occupy separate pair rows for deterministic reduction. bf162/fdot2 compute,
// same numerics note as k_moe_w4_up.
template <bool Scatter = true, bool Tiled = false, bool Packed = false,
          bool PairTable = Packed, bool PairsBf16 = false>
__global__ void __launch_bounds__(256, FD_MINB)
    k_moe_w4_down(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                  const float* __restrict__ cb, const float* __restrict__ hid,
                  const int* __restrict__ tokidx, const float* __restrict__ pw,
                  const int* __restrict__ eoff, float* __restrict__ acc, int rows_per,
                  int cols, uint64_t scale_stride,
                  const MoeTile* tiles = nullptr, const int* ntiles = nullptr,
                  const uint16_t* packed_hid = nullptr) {
  if (Tiled && blockIdx.y >= *ntiles) return;
  const int e = Tiled ? tiles[blockIdx.y].expert : blockIdx.y, c0 = blockIdx.x * FD_CT_DN;
  const int n0 = Tiled ? tiles[blockIdx.y].first : eoff[e];
  const int ne = Tiled ? tiles[blockIdx.y].count : eoff[e + 1] - n0;
  if (!ne) return;
#ifndef GDEC_MOE_KC16DB
  // The first K-loop barrier publishes the table before weight staging.
  __shared__ __hip_bfloat162 cb_pairs[PairTable ? 256 : 1];
  if (PairTable)
    cb_pairs[threadIdx.x] = __halves2bfloat162(
        __float2bfloat16(cb[threadIdx.x & 15]),
        __float2bfloat16(cb[threadIdx.x >> 4]));
  __shared__ __hip_bfloat162 xs2[FD_RT][FD_XP];
  __shared__ __hip_bfloat162 wf2[FD_KC / 2][FD_WP_DN];  // weights, dequantized
  __shared__ float wsc[FD_CT_DN];                       // per-column group scales
#else
  __shared__ __hip_bfloat162 xs2[2][FD_RT][FD_XP];
  __shared__ __hip_bfloat162 wf2[2][FD_KC / 2][FD_WP_DN];  // weights, dequantized
  __shared__ float wsc[2][FD_CT_DN];                       // per-column group scales
#endif
  const int cg = threadIdx.x & 31, rg = threadIdx.x >> 5;
  const uint8_t* ec = codes + (size_t)e * rows_per * (cols / 2);
  const uint8_t* es = scales + (size_t)e * rows_per * scale_stride;
  const size_t rowb = (size_t)cols / 2;
  for (int r0 = 0; r0 < ne; r0 += FD_RT) {
    const int nr = min(FD_RT, ne - r0);
    float accv[FD_CPT_DN][8] = {};  // [cc][r]
#ifndef GDEC_MOE_KC16DB
    const int srow = threadIdx.x >> 1, sseg = threadIdx.x & 1;
    const bool sact = srow < FD_CT_DN;  // CT=64: only half the threads stage
    const uint8_t* rb = ec + (size_t)(c0 + srow) * rowb + sseg * 8;
    const uint8_t* sb = es + (size_t)(c0 + srow) * scale_stride;
    uint2 pk2;
    float sc;
    auto stage_load = [&](int k0) {
      pk2 = *(const uint2*)(rb + k0 / 2);
      sc = __half2float(*(const __half*)(sb + (k0 / 32) * 2));
    };
    if (sact) stage_load(0);
    // x gather prefetch (see k_moe_w4_up): row xr, pairs xp4..xp4+3; loads
    // issued one chunk ahead beside stage_load. Bit-identical either way.
    // Gated to the fp32 path (packed gather is too cheap to amortize the
    // register live range); -DGDEC_MOE_NO_XPREF disables it everywhere.
#ifdef GDEC_MOE_NO_XPREF
    constexpr bool XPREF = false;
#else
    constexpr bool XPREF = !Packed;
#endif
    const int xr = threadIdx.x >> 2, xp4 = (threadIdx.x & 3) * 4;
    const bool xact = xr < nr;
    uint4 xp0, xp1;
    auto x_load = [&](int k0) {
      if (!xact) return;
      const size_t xoff = (size_t)(n0 + r0 + xr) * cols + xp4 * 2 + k0;
      if (Packed) {
        xp0 = *(const uint4*)(packed_hid + xoff);
      } else {
        *(float4*)&xp0 = *(const float4*)(hid + xoff);
        *(float4*)&xp1 = *(const float4*)(hid + xoff + 4);
      }
    };
    if (XPREF) x_load(0);
    for (int k0 = 0; k0 < cols; k0 += FD_KC) {
      __syncthreads();
      {
        {
          __hip_bfloat162 v[4];  // 256 threads x 16B = 64 rows x 16 pairs
          if (xact) {
            if (Packed) {
              if (XPREF) *(uint4*)v = xp0;
              else *(uint4*)v = *(const uint4*)(packed_hid + (size_t)(n0 + r0 + xr) * cols + k0 + xp4 * 2);
            } else {
              float4 x0, x1;
              if (XPREF) {
                x0 = *(const float4*)&xp0;
                x1 = *(const float4*)&xp1;
              } else {
                const float* xp = hid + (size_t)(n0 + r0 + xr) * cols + k0 + xp4 * 2;
                x0 = *(const float4*)xp;
                x1 = *(const float4*)(xp + 4);
              }
              v[0] = __halves2bfloat162(__float2bfloat16(x0.x), __float2bfloat16(x0.y));
              v[1] = __halves2bfloat162(__float2bfloat16(x0.z), __float2bfloat16(x0.w));
              v[2] = __halves2bfloat162(__float2bfloat16(x1.x), __float2bfloat16(x1.y));
              v[3] = __halves2bfloat162(__float2bfloat16(x1.z), __float2bfloat16(x1.w));
            }
          } else {
            __hip_bfloat162 z = __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
            v[0] = z; v[1] = z; v[2] = z; v[3] = z;
          }
          *(float4*)&xs2[xr][xp4] = *(float4*)v;
        }
        if (sact) {
          uint32_t vw0 = pk2.x, vw1 = pk2.y;
          if (sseg == 0) wsc[srow] = sc;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            wf2[sseg * 8 + j][srow] = PairTable ? cb_pairs[vw0 & 255] :
              __halves2bfloat162(__float2bfloat16(__ldg(cb + (vw0 & 0xf))), __float2bfloat16(__ldg(cb + ((vw0 >> 4) & 0xf))));
            vw0 >>= 8;
          }
#pragma unroll
          for (int j = 0; j < 4; j++) {
            wf2[sseg * 8 + 4 + j][srow] = PairTable ? cb_pairs[vw1 & 255] :
              __halves2bfloat162(__float2bfloat16(__ldg(cb + (vw1 & 0xf))), __float2bfloat16(__ldg(cb + ((vw1 >> 4) & 0xf))));
            vw1 >>= 8;
          }
        }
      }
      if (k0 + FD_KC < cols) {
        if (sact) stage_load(k0 + FD_KC);
        if (XPREF) x_load(k0 + FD_KC);
      }
      __syncthreads();
      float sv[FD_CPT_DN];
#pragma unroll
      for (int cc = 0; cc < FD_CPT_DN; cc++) sv[cc] = wsc[cg * FD_CPT_DN + cc];
      float acv[FD_CPT_DN][8] = {};
#pragma unroll 1
      for (int i = 0; i < FD_KC / 8; i++) {
        __hip_bfloat162 xv[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++)
          *(float4*)xv[r] = *(const float4*)&xs2[rg * 8 + r][i * 4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          __hip_bfloat162 wv[FD_CPT_DN];
          if (FD_CPT_DN == 4)
            *(float4*)wv = *(const float4*)&wf2[i * 4 + kk][cg * FD_CPT_DN];
          else
            *(uint2*)wv = *(const uint2*)&wf2[i * 4 + kk][cg * FD_CPT_DN];
#pragma unroll
          for (int cc = 0; cc < FD_CPT_DN; cc++)
#pragma unroll
            for (int r = 0; r < 8; r++)
              acv[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wv[cc], xv[r][kk], acv[cc][r], false);
        }
      }
#pragma unroll
      for (int cc = 0; cc < FD_CPT_DN; cc++)
#pragma unroll
        for (int r = 0; r < 8; r++) accv[cc][r] += sv[cc] * acv[cc][r];
    }
#else  // FD_KC=16 double-buffered, one barrier per chunk
    // staging slots: thread t stages weight column t (t < FD_CT_DN), 8B =
    // 16 nibbles per 16-chunk; x slot is row xr=tid/4, pairs (tid%4)*2..+1.
    const int srow = threadIdx.x & (FD_CT_DN - 1);
    const bool sact = threadIdx.x < FD_CT_DN;
    const uint8_t* rb = ec + (size_t)(c0 + srow) * rowb;
    const uint8_t* sb = es + (size_t)(c0 + srow) * scale_stride;
    uint2 pk2;
    float sc;
    auto stage_load = [&](int k0) {  // k0 = 16-wide chunk offset
      pk2 = *(const uint2*)(rb + k0 / 2);
      sc = __half2float(*(const __half*)(sb + (k0 / 32) * 2));
    };
    const int xr = threadIdx.x >> 2, xp2 = (threadIdx.x & 3) * 2;
    const bool xact = xr < nr;
    const size_t xoff = (size_t)(n0 + r0 + xr) * cols + xp2 * 2;
    uint4 xp;  // next chunk's x: 8B packed / 16B fp32
    auto x_load = [&](int k0) {
      if (!xact) return;
      if (Packed) *(uint2*)&xp = *(const uint2*)(packed_hid + xoff + k0);
      else *(float4*)&xp = *(const float4*)(hid + xoff + k0);
    };
    auto x_store = [&](int buf, int k0) {
      __hip_bfloat162 v[2];
      bool got = xact;
#ifdef GDEC_MOE_NO_XPREF
      if (got) {
        if (Packed) {
          *(uint2*)v = *(const uint2*)(packed_hid + xoff + k0);
        } else {
          float4 f = *(const float4*)(hid + xoff + k0);
          v[0] = __halves2bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
          v[1] = __halves2bfloat162(__float2bfloat16(f.z), __float2bfloat16(f.w));
        }
      }
#else
      (void)k0;
      if (got) {
        if (Packed) {
          *(uint2*)v = *(const uint2*)&xp;
        } else {
          float4 f = *(const float4*)&xp;
          v[0] = __halves2bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
          v[1] = __halves2bfloat162(__float2bfloat16(f.z), __float2bfloat16(f.w));
        }
      }
#endif
      if (!got) {
        __hip_bfloat162 z = __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
        v[0] = z; v[1] = z;
      }
      *(uint2*)&xs2[buf][xr][xp2] = *(uint2*)v;
    };
    auto stage_w = [&](int buf) {
      if (!sact) return;
      wsc[buf][srow] = sc;
      uint32_t v0 = pk2.x, v1 = pk2.y;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wf2[buf][j][srow] = __halves2bfloat162(__float2bfloat16(__ldg(cb + (v0 & 0xf))), __float2bfloat16(__ldg(cb + ((v0 >> 4) & 0xf))));
        v0 >>= 8;
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wf2[buf][4 + j][srow] = __halves2bfloat162(__float2bfloat16(__ldg(cb + (v1 & 0xf))), __float2bfloat16(__ldg(cb + ((v1 >> 4) & 0xf))));
        v1 >>= 8;
      }
    };
    // prologue: stage chunk 0 into buffer 0 (WAR-safe: no barrier since the
    // previous row tile's last compute, but chunk 0 reads nothing and the
    // previous tile ended with a barrier)
    if (sact) stage_load(0);
#ifndef GDEC_MOE_NO_XPREF
    x_load(0);
#endif
    stage_w(0);
    x_store(0, 0);
    __syncthreads();
    // chunk partials live across both 16-chunks of each 32-wide scale group
    // (k ascending) — same fold points as the 32-wide loop, bit-identical.
    float acv[FD_CPT_DN][8] = {};
    const int nchunk = cols / FD_KC;  // even: cols % 32 == 0 (host-checked)
    for (int ci = 0; ci < nchunk; ci++) {
      const int cur = ci & 1, nxt = cur ^ 1, k0 = ci * FD_KC;
      if (ci + 1 < nchunk) {
        if (sact) stage_load(k0 + FD_KC);
#ifndef GDEC_MOE_NO_XPREF
        x_load(k0 + FD_KC);
#endif
      }
      float sv[FD_CPT_DN];
#pragma unroll
      for (int cc = 0; cc < FD_CPT_DN; cc++) sv[cc] = wsc[cur][cg * FD_CPT_DN + cc];
#pragma unroll 1
      for (int i = 0; i < FD_KC / 8; i++) {
        __hip_bfloat162 xv[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++)
          *(float4*)xv[r] = *(const float4*)&xs2[cur][rg * 8 + r][i * 4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          __hip_bfloat162 wv[FD_CPT_DN];
          if (FD_CPT_DN == 8) {
            *(float4*)wv = *(const float4*)&wf2[cur][i * 4 + kk][cg * FD_CPT_DN];
            *(float4*)(wv + 4) = *(const float4*)&wf2[cur][i * 4 + kk][cg * FD_CPT_DN + 4];
          } else if (FD_CPT_DN == 4)
            *(float4*)wv = *(const float4*)&wf2[cur][i * 4 + kk][cg * FD_CPT_DN];
          else
            *(uint2*)wv = *(const uint2*)&wf2[cur][i * 4 + kk][cg * FD_CPT_DN];
#pragma unroll
          for (int cc = 0; cc < FD_CPT_DN; cc++)
#pragma unroll
            for (int r = 0; r < 8; r++)
              acv[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wv[cc], xv[r][kk], acv[cc][r], false);
        }
      }
      if (ci & 1) {  // end of a 32-wide scale group: fold chunk partials
#pragma unroll
        for (int cc = 0; cc < FD_CPT_DN; cc++)
#pragma unroll
          for (int r = 0; r < 8; r++) {
            accv[cc][r] += sv[cc] * acv[cc][r];
            acv[cc][r] = 0.f;
          }
      }
      if (ci + 1 < nchunk) {
        stage_w(nxt);
        x_store(nxt, k0 + FD_KC);
      }
      __syncthreads();
    }
#endif
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int row = rg * 8 + r;
      if (row >= nr) continue;
      int p = n0 + r0 + row;
      size_t base = (size_t)(Scatter ? tokidx[p] : p) * rows_per + c0 + cg * FD_CPT_DN;
      if constexpr (!Scatter && PairsBf16) {
        // Unweighted bf16 pair rows (f2bf RNE); pw is applied in fp32 by
        // k_moe_reduce_pw_bf16, which rounds later than weight-then-round.
#pragma unroll
        for (int cc = 0; cc < FD_CPT_DN; cc++)
          ((uint16_t*)acc)[base + cc] = f2bf(accv[cc][r]);
      } else {
        float w = pw[p];
#pragma unroll
        for (int cc = 0; cc < FD_CPT_DN; cc++) {
          float value = w * accv[cc][r];
          if (Scatter) atomicAdd(acc + base + cc, value);
          else acc[base + cc] = value;
        }
      }
    }
  }
}

__global__ void k_moe_reduce(const float* pairs, const int* pairids, float* out,
                              int P, int k, int D) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)P * D) return;
  int t = i / D, d = i % D;
  float sum = 0.f;
  for (int s = 0; s < k; s++) sum += pairs[(size_t)pairids[t * k + s] * D + d];
  out[i] = sum;
}

// Fixed top-k reduction candidate. Pair ids are shared by all output columns
// in a block, while pair values remain coalesced across the 256 output lanes.
// The accumulation order is exactly slot 0..k-1, matching k_moe_reduce.
template <int K = 10>
__global__ void k_moe_reduce_fast(const float* __restrict__ pairs,
                                  const int* __restrict__ pairids,
                                  float* __restrict__ out, int P, int D) {
  __shared__ int ids[K];
  int tid = threadIdx.x;
  int t = blockIdx.y, d = tid + (int)blockIdx.x * blockDim.x;
  if (t >= P) return;
  if (tid < K) ids[tid] = pairids[t * K + tid];
  __syncthreads();
  if (d >= D) return;
  float sum = 0.f;
#pragma unroll
  for (int s = 0; s < K; s++) sum += pairs[(size_t)ids[s] * D + d];
  out[(size_t)t * D + d] = sum;
}

