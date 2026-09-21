// moe_wmma_proto.cu — Stage 1: grouped expert up-GEMM, Q4C-P codes fused
// dequant to LDS + WMMA compute (gfx1151 prototype; engine untouched).
//
// Contract (mirrors k_moe_w4_up, src/gpu/gdec.cpp:4192):
//   expert e owns pairs tokidx[eoff[e] .. eoff[e+1]); weight W_e = [rows, cols]
//   Q4C-P (4-bit codes row-major, rows*cols/2 bytes; fp16 scale per 32 cols,
//   scale_stride bytes per row; 16-entry fp32 codebook cb).
//   Output (Stage 1 raw, no silu): gu[pair][rows] fp32,
//     gu[n0+p][n] = sum_k X[tokidx[n0+p]][k] * deq(W_e[n][k]),
//     deq: byte b -> k=2b (low nibble), 2b+1 (high); w = bf16(cb[code]*s)
//     — formula identical to k_dequant_q4cp_bf16 (gdec.cpp:2631).
// Kernel: k_gemm_wmma d9 skeleton (128x256 tile, 8 warps 2x4, warp 4x4 frags,
// K-step 32 = exactly one scale group, LDS double buffer, gwmma_sync,
// unconditional staging). Only difference vs dense: B staging is
// codes+scale -> dequant -> LDS instead of a bf16 copy.
// Grid (rows/BP, E): same-expert blocks adjacent, gathered X shared via L2.
// Panel-outer (BM token rows) / K-inner; weight codes re-read per panel
// logically, deduped by L2/MALL (~1 DRAM pass expected).
//
// Build: hipcc -O2 -o /tmp/moe_wmma_proto tools/moe_wmma_proto.cu -lrocblas
// Env:   PROTO_CFG=<substr>  PROTO_CASE=uniform|skew|both (default both)
//        PROTO_NOCHECK=1 (bench only)
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      fprintf(stderr, "hip error %d (%s) at %s:%d\n", (int)e_,               \
              hipGetErrorString(e_), __FILE__, __LINE__);                    \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

using qw_shortx16 = __attribute__((ext_vector_type(16))) short;
using qw_floatx8 = __attribute__((ext_vector_type(8))) float;

__device__ __forceinline__ qw_shortx16 qw_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  qw_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
// LDS-only barrier (__syncthreads() also emits buffer_gl0_inv on gfx11).
__device__ __forceinline__ void gwmma_sync() {
  __asm__ volatile(
      "s_waitcnt lgkmcnt(0)\n\t"
      "s_waitcnt_vscnt null, 0x0\n\t"
      "s_barrier" ::
          : "memory");
}
__host__ __device__ __forceinline__ uint16_t f2bf(float f) {  // RNE
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
__device__ __forceinline__ float bf2f(uint16_t u) {
  uint32_t v = (uint32_t)u << 16;
  float f;
  memcpy(&f, &v, 4);
  return f;
}

// --- k_moe_wmma_up -----------------------------------------------------------
// Block tile BM(pairs) x BP(weight rows), NT threads as MW x PW warps, warp
// owns WM x WP 16x16 frags, K stepped by KST (== one 32-col scale group).
// A side: X rows gathered via tokidx (per-panel tk[] in LDS, tail rows clamp
// to the panel's last valid row; epilogue predicated). B side: thread t owns
// weight row nw0+t: per K-step loads 16B codes + 1 fp16 scale, dequantizes
// (bf16(cb*s), k_dequant_q4cp_bf16 formula) and stores 32 bf16 into the same
// 8-row region layout as the dense kernel (B-frag = one ds_load_b128).
// Hard requirements: rows % BP == 0, cols % KST == 0, BP == NT.
// DM: 0 = fp32 dequant (scb[16] floats, fp32 mul + f2bf per element);
//     1 = cb_pairs bf162 table + hmul2 scale (k_moe_w4_up's PairTable trick;
//         bf16 double-rounding, ~7x less staging VALU);
//     2 = DM1 + dequant to registers before COMPUTE (staging VALU overlaps
//         other warps' wmma issue instead of sitting next to the barrier);
//     3 = lower bound: weights pre-dequantized to bf16 in global (no codes).
template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT,
          int DM>
__global__ void __launch_bounds__(NT)
    k_moe_wmma_up(const uint8_t* __restrict__ codes,
                  const uint8_t* __restrict__ scales,
                  const float* __restrict__ cb,
                  const uint16_t* __restrict__ X,
                  const int* __restrict__ tokidx, const int* __restrict__ eoff,
                  float* __restrict__ gu, int rows, int cols,
                  int64_t scale_stride,
                  const uint16_t* __restrict__ Wbf = nullptr,
                  float* __restrict__ dbg = nullptr) {
  constexpr int LSA = KST + 8;             // LDS row stride (elems)
  constexpr int BREG = 8 * LSA + 8;        // W 8-row region stride (elems)
  constexpr int NW = NT / 32;              // warps per block
  constexpr int AN = BM * (KST / 8) / NT;  // uint4 X loads per thread
  static_assert(MW * PW == NW, "warp grid mismatch");
  static_assert(BM == MW * WM * 16 && BP == PW * WP * 16, "tile mismatch");
  static_assert(BM * (KST / 8) % NT == 0, "staging mismatch");
  static_assert(BP == NT, "one codes row per thread");
  __shared__ uint16_t As[2][BM * LSA];
  __shared__ uint16_t Bs[2][(BP / 8) * BREG];
  __shared__ float scb[(DM == 0 || DM == 10 || DM == 11) ? 16 : 1];
  __shared__ __hip_bfloat162 cbp[(DM >= 1 && DM <= 7 && DM != 3) ? 256 : 1];
  __shared__ float2 cbp2[DM == 15 ? 256 : 1];  // cbp2[byte] = {cb[lo], cb[hi]}
  __shared__ int tk[BM];

  const int e = blockIdx.y, nw0 = blockIdx.x * BP;
  const int n0 = eoff[e], ne = eoff[e + 1] - n0;
  if (!ne) return;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  if (DM == 0 || DM == 10 || DM == 11) {
    if (tid < 16) scb[tid] = cb[tid];
  } else if (DM == 1 || DM == 2 || DM == 4 || DM == 5 || DM == 7) {
    cbp[tid] = __halves2bfloat162(__float2bfloat16(cb[tid & 15]),
                                  __float2bfloat16(cb[tid >> 4]));
  }
  // DM9: cb broadcast via shuffle instead of an LDS table (exact fp32 values).
  float cbreg = 0.f;
  if (DM == 9) cbreg = cb[lane & 15];
  if (DM == 15) {
    float2 pr_;
    pr_.x = cb[tid & 15];
    pr_.y = cb[tid >> 4];
    cbp2[tid] = pr_;
  }
  const size_t rowb = (size_t)cols / 2;
  const uint8_t* ec = codes + ((size_t)e * rows + nw0 + tid) * rowb;
  const uint8_t* es =
      scales + ((size_t)e * rows + nw0 + tid) * scale_stride;
  const uint16_t* gbf =
      Wbf + ((size_t)e * rows + nw0 + tid) * cols;  // DM==3 only

  const int wm0 = (w / PW) * (WM * 16);
  const int wp0 = (w % PW) * (WP * 16);
  // WMMA A layout: lane 2r -> row r, lane 17+2(r-8) -> row 8+r; ignored lanes
  // get complementary dummy rows so every 8-lane LDS phase hits 8 banks.
  const int base_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? base_ : ((base_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((base_ + 4) & 7) : base_);
  const int ksteps = cols / KST;

  uint32_t aoff[WM], boff[WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
    aoff[i] = ((wm0 + i * 16 + qrowA) * LSA) * 2;
#pragma unroll
  for (int j = 0; j < WP; ++j) {
    const int pl = wp0 + j * 16 + (lane & 15);
    boff[j] = ((pl >> 3) * BREG + (pl & 7) * LSA) * 2;
  }
  const uint32_t sboff = ((tid >> 3) * BREG + (tid & 7) * LSA) * 2;

#define MWM_LOAD(RA, RC, RB4, RS, KS)                                  \
  do {                                                                  \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)                      \
        RA[j] = *(const uint4*)(ga[j] + (KS)*KST);                      \
    if (DM == 3) {                                                      \
      _Pragma("unroll") for (int t = 0; t < KST / 8; ++t)               \
          RB4[t] = ((const uint4*)gbf)[(KS) * (KST / 8) + t];           \
    } else {                                                            \
      RC = *(const uint4*)(ec + (KS) * (KST / 2));                      \
      RS = *(const __half*)(es + (KS) * 2);                             \
    }                                                                   \
  } while (0)
#define MWM_DEQ(DQ, RC, RS)                                              \
  do {                                                                  \
    const __hip_bfloat162 sc2_ =                                        \
        __halves2bfloat162(__float2bfloat16(__half2float(RS)),          \
                           __float2bfloat16(__half2float(RS)));         \
    const uint32_t* cw_ = (const uint32_t*)&RC;                         \
    if (DM == 15) {                                                 \
      const float scf_ = __half2float(RS);                          \
      _Pragma("unroll") for (int t = 0; t < 4; ++t) {               \
        const uint32_t v = cw_[t];                                  \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {             \
          const float2 pr_ = cbp2[(v >> (b * 8)) & 255];            \
          uint32_t al_ = __float_as_uint(pr_.x * scf_);             \
          uint32_t ah_ = __float_as_uint(pr_.y * scf_);             \
          al_ += 0x7FFFu + ((al_ >> 16) & 1);                       \
          ah_ += 0x7FFFu + ((ah_ >> 16) & 1);                       \
          const uint32_t pk_ = (al_ >> 16) | (ah_ & 0xFFFF0000u);   \
          ((uint32_t*)DQ)[4 * t + b] = pk_;                         \
        }                                                           \
      }                                                             \
    } else if (DM == 11) {                                          \
      const float scf_ = __half2float(RS);                          \
      _Pragma("unroll") for (int t = 0; t < 4; ++t) {               \
        const uint32_t v = cw_[t];                                  \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {             \
          const uint16_t lo = f2bf(scf_ * scb[(v >> (b * 8)) & 15]); \
          const uint16_t hi =                                       \
              f2bf(scf_ * scb[(v >> (b * 8 + 4)) & 15]);            \
          DQ[4 * t + b] = __halves2bfloat162(                       \
              *(const __hip_bfloat16*)&lo, *(const __hip_bfloat16*)&hi); \
        }                                                           \
      }                                                             \
    } else if (DM == 5 || DM == 7) {                                \
      const float scf_ = __half2float(RS);                              \
      _Pragma("unroll") for (int t = 0; t < 4; ++t) {                   \
        const uint32_t v = cw_[t];                                      \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {                 \
          const __hip_bfloat162 pr = cbp[(v >> (b * 8)) & 255];         \
          uint16_t lo, hi;                                              \
          float f_ = bf2f(*(const uint16_t*)&pr.x);                     \
          lo = f2bf(f_ * scf_);                                         \
          f_ = bf2f(*(const uint16_t*)&pr.y);                           \
          hi = f2bf(f_ * scf_);                                         \
          DQ[4 * t + b] = __halves2bfloat162(                           \
              *(const __hip_bfloat16*)&lo, *(const __hip_bfloat16*)&hi); \
        }                                                               \
      }                                                                 \
    } else if (DM == 6) {                                               \
      const float scf_ = __half2float(RS);                              \
      _Pragma("unroll") for (int t = 0; t < 4; ++t) {                   \
        const uint32_t v = cw_[t];                                      \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {                 \
          const uint16_t lo = f2bf(scf_ * __ldg(cb + ((v >> (b * 8)) & 15))); \
          const uint16_t hi =                                           \
              f2bf(scf_ * __ldg(cb + ((v >> (b * 8 + 4)) & 15)));       \
          DQ[4 * t + b] = __halves2bfloat162(                           \
              *(const __hip_bfloat16*)&lo, *(const __hip_bfloat16*)&hi); \
        }                                                               \
      }                                                                 \
    } else {                                                            \
    _Pragma("unroll") for (int t = 0; t < 4; ++t) {                     \
      const uint32_t v = cw_[t];                                        \
      _Pragma("unroll") for (int b = 0; b < 4; ++b)                     \
          DQ[4 * t + b] = __hmul2(cbp[(v >> (b * 8)) & 255], sc2_);     \
    }                                                                   \
    }                                                                   \
  } while (0)
#define MWM_STORE(BUF, RA, RC, RB4, RS, DQ)                             \
  do {                                                                 \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)                     \
        *(uint4*)((char*)&As[BUF][0] + saoff[j]) = RA[j];              \
    char* wb_ = (char*)&Bs[BUF][0] + sboff;                            \
    if (DM == 3) {                                                     \
      _Pragma("unroll") for (int t = 0; t < KST / 8; ++t)              \
          *(uint4*)(wb_ + t * 16) = RB4[t];                            \
    } else if (DM == 2 || DM == 4 || DM == 11 || DM == 15) {        \
      _Pragma("unroll") for (int t = 0; t < 4; ++t)                    \
          *(uint4*)(wb_ + t * 16) = *(const uint4*)&DQ[t * 4];         \
    } else if (DM == 1 || DM == 5 || DM == 6 || DM == 7) {            \
      MWM_DEQ((DQ), RC, RS);                                           \
      _Pragma("unroll") for (int t = 0; t < 4; ++t)                    \
          *(uint4*)(wb_ + t * 16) = *(const uint4*)&DQ[t * 4];         \
    } else {                                                           \
      const float sc_ = __half2float(RS);                              \
      const uint32_t* cw_ = (const uint32_t*)&RC;                      \
      _Pragma("unroll") for (int t = 0; t < 4; ++t) {                  \
        uint32_t v = cw_[t];                                           \
        uint32_t pk[4];                                                \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {                \
          uint16_t lo, hi;                                             \
          if (DM == 9) {                                               \
            lo = f2bf(sc_ * __shfl_sync(~0ull, cbreg,                  \
                                        (v >> (b * 8)) & 15));         \
            hi = f2bf(sc_ * __shfl_sync(~0ull, cbreg,                  \
                                        (v >> (b * 8 + 4)) & 15));     \
          } else if (DM == 10) {                                       \
            lo = f2bf(sc_ * scb[0]);                                   \
            hi = f2bf(sc_ * scb[0]);                                   \
          } else if (DM == 13) {                                       \
            lo = f2bf(sc_ * (float)((v >> (b * 8)) & 15));             \
            hi = f2bf(sc_ * (float)((v >> (b * 8 + 4)) & 15));         \
          } else {                                                     \
            lo = f2bf(sc_ * scb[(v >> (b * 8)) & 15]);                 \
            hi = f2bf(sc_ * scb[(v >> (b * 8 + 4)) & 15]);             \
          }                                                            \
          pk[b] = (uint32_t)lo | ((uint32_t)hi << 16);                 \
        }                                                              \
        *(uint4*)(wb_ + t * 16) = *(const uint4*)pk;                   \
      }                                                                \
    }                                                                  \
  } while (0)
#define MWM_COMPUTE(CUR)                                                \
  do {                                                                  \
    const char* Ab = (const char*)&As[CUR][0];                          \
    const char* Bb = (const char*)&Bs[CUR][0];                          \
    _Pragma("unroll") for (int kh = 0; kh < KST / 16; ++kh) {          \
      qw_shortx16 af[WM];                                               \
      _Pragma("unroll") for (int i = 0; i < WM; ++i)                    \
          af[i] = qw_ld16((const uint16_t*)(Ab + aoff[i] + kh * 32));   \
      _Pragma("unroll") for (int j = 0; j < WP; ++j) {                  \
        const qw_shortx16 bfj =                                         \
            qw_ld16((const uint16_t*)(Bb + boff[j] + kh * 32));         \
        _Pragma("unroll") for (int i = 0; i < WM; ++i)                  \
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(    \
                af[i], bfj, acc[i][j]);                                 \
      }                                                                 \
    }                                                                   \
  } while (0)

  for (int r0 = 0; r0 < ne; r0 += BM) {
    const int nr = min(BM, ne - r0);
    if (tid < BM) tk[tid] = tokidx[n0 + r0 + min(tid, nr - 1)];
    if (DM == 1 || DM == 2 || DM == 4 || DM == 5 || DM == 7)
      cbp[tid] = __halves2bfloat162(__float2bfloat16(cb[tid & 15]),
                                    __float2bfloat16(cb[tid >> 4]));
    gwmma_sync();  // publish tk + cbp (+ scb on the first panel)
    const uint16_t* ga[AN];
    uint32_t saoff[AN];
#pragma unroll
    for (int j = 0; j < AN; ++j) {
      const int idx = tid + j * NT;
      const int row = idx / (KST / 8), q = idx % (KST / 8);
      ga[j] = X + (size_t)tk[row] * cols + q * 8;
      saoff[j] = (row * LSA + q * 8) * 2;
    }

    qw_floatx8 acc[WM][WP];
#pragma unroll
    for (int i = 0; i < WM; ++i)
#pragma unroll
      for (int j = 0; j < WP; ++j)
#pragma unroll
        for (int e8 = 0; e8 < 8; ++e8) acc[i][j][e8] = 0.f;

    if (DM == 4 || DM == 11 || DM == 15) {
      // sector-batched staging: codes 32B every 2 steps, scales 16B every 8
      // steps (per-step 2B loads in the last 8 steps avoid a trailing 2B
      // over-read past the row's scale region). X stays per-step. The K loop
      // runs in 8-step groups with all register indexing compile-time (a
      // runtime rc2[ks&1]/sc8h[ks&7] index would land the arrays in scratch).
      // Requires cols % (8*KST) == 0 (host-checked).
      uint4 ra0[AN], rc2[2];
      __hip_bfloat162 dq0[16];
#pragma unroll
      for (int j = 0; j < AN; ++j) ra0[j] = *(const uint4*)(ga[j]);
      rc2[0] = *(const uint4*)(ec);
      rc2[1] = *(const uint4*)(ec + (ksteps > 1 ? KST / 2 : 0));
      MWM_DEQ(dq0, rc2[0], *(const __half*)es);
      MWM_STORE(0, ra0, rc2[0], rc2, *(const __half*)es, dq0);
      gwmma_sync();
#pragma unroll 1
      for (int ks8 = 0; ks8 < ksteps; ks8 += 8) {
        const bool tail = (ks8 + 8 >= ksteps);
        __half scv8[8];
        if (tail) {
#pragma unroll
          for (int s = 0; s < 8; ++s)
            scv8[s] = *(const __half*)(es + min(ks8 + s + 1, ksteps - 1) * 2);
        } else {
          *(uint4*)scv8 = *(const uint4*)(es + (ks8 + 1) * 2);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int kt = min(ks8 + s + 1, ksteps - 1);  // tile being stored
#pragma unroll
          for (int j = 0; j < AN; ++j)
            ra0[j] = *(const uint4*)(ga[j] + kt * KST);
          if ((s & 1) == 0) {
            rc2[0] = *(const uint4*)(ec + kt * (KST / 2));
            rc2[1] =
                *(const uint4*)(ec + min(ks8 + s + 2, ksteps - 1) * (KST / 2));
          }
          MWM_COMPUTE(s & 1);
          MWM_DEQ(dq0, rc2[s & 1], scv8[s]);
          MWM_STORE((s & 1) ^ 1, ra0, rc2[s & 1], rc2, scv8[s], dq0);
          gwmma_sync();
        }
      }
    } else {
      uint4 ra0[AN], rc0, rb4[KST / 8];
      __half rs0;
      __hip_bfloat162 dq0[16];
      MWM_LOAD(ra0, rc0, rb4, rs0, 0);
      if (DM == 2) MWM_DEQ(dq0, rc0, rs0);
      MWM_STORE(0, ra0, rc0, rb4, rs0, dq0);
      gwmma_sync();
      if (DM == 7 && dbg && r0 == 0) {
        // dump this block's dequantized tile-0 row (tid) + scale group 0
        float* db = dbg + ((size_t)e * (rows / BP) + blockIdx.x) * (BP * 33);
        const uint16_t* wr = (const uint16_t*)((const char*)&Bs[0][0] + sboff);
#pragma unroll
        for (int t = 0; t < 32; ++t) db[tid * 33 + t] = bf2f(wr[t]);
        db[tid * 33 + 32] = __half2float(rs0);
      }

#pragma unroll 1
      for (int ks = 0; ks < ksteps; ++ks) {
        MWM_LOAD(ra0, rc0, rb4, rs0, min(ks + 1, ksteps - 1));
        if (DM == 2) MWM_DEQ(dq0, rc0, rs0);
        MWM_COMPUTE(ks & 1);
        MWM_STORE((ks & 1) ^ 1, ra0, rc0, rb4, rs0, dq0);
        gwmma_sync();
      }
    }

    const int rh = (lane >> 4) * 8, cl = lane & 15;
#pragma unroll
    for (int i = 0; i < WM; ++i)
#pragma unroll
      for (int j = 0; j < WP; ++j) {
        const int p0 = wm0 + i * 16 + rh;
        const int nn = nw0 + wp0 + j * 16 + cl;
#pragma unroll
        for (int e8 = 0; e8 < 8; ++e8)
          if (p0 + e8 < nr)
            gu[(size_t)(n0 + r0 + p0 + e8) * rows + nn] = acc[i][j][e8];
      }
  }
#undef MWM_LOAD
#undef MWM_DEQ
#undef MWM_STORE
#undef MWM_COMPUTE
}

// --- reference helpers -------------------------------------------------------
// dequant one expert to bf16 (k_dequant_q4cp_bf16 formula)
__global__ void k_dequant_ref(const uint8_t* __restrict__ codes,
                              const uint8_t* __restrict__ scales,
                              const float* __restrict__ cb,
                              uint16_t* __restrict__ out, int rows, int cols,
                              int64_t scale_stride) {
  const int gpr = cols / 32;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= rows * gpr) return;
  const int r = gid / gpr, g = gid % gpr;
  const uint8_t* rc = codes + (size_t)r * (cols / 2) + g * 16;
  const float s = __half2float(*(const __half*)(scales + (size_t)r * scale_stride + g * 2));
  uint16_t* o = out + (size_t)r * cols + g * 32;
#pragma unroll
  for (int b = 0; b < 16; b++) {
    const uint32_t v = rc[b];
    o[2 * b] = f2bf(__ldg(cb + (v & 0xf)) * s);
    o[2 * b + 1] = f2bf(__ldg(cb + (v >> 4)) * s);
  }
}
// gather X rows for one expert: xg[p][k] = X[tokidx[n0+p]][k]
__global__ void k_gather_x(const uint16_t* __restrict__ X,
                           const int* __restrict__ tokidx, int n0, int ne,
                           uint16_t* __restrict__ xg, int cols) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)ne * cols / 8) return;
  const int p = (int)(i / (cols / 8)), q = (int)(i % (cols / 8));
  ((uint4*)xg)[p * (cols / 8) + q] =
      ((const uint4*)X)[(size_t)tokidx[n0 + p] * (cols / 8) + q];
}

static uint32_t xs_state;
static uint32_t xs_next() {
  uint32_t x = xs_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  xs_state = x;
  return x;
}
static float xs_frand() { return (float)((xs_next() >> 8) & 0xFFFF) / 32768.f - 1.f; }

struct KCfg {
  const char* name;
  int bm;
  void (*fn)(const uint8_t*, const uint8_t*, const float*, const uint16_t*,
             const int*, const int*, float*, int, int, int64_t,
             const uint16_t*, unsigned, hipStream_t);
};

template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT,
          int DM>
static void launch_cfg(const uint8_t* codes, const uint8_t* scales,
                       const float* cb, const uint16_t* X, const int* tokidx,
                       const int* eoff, float* gu, int rows, int cols,
                       int64_t ss, const uint16_t* Wbf, unsigned E,
                       hipStream_t st) {
  k_moe_wmma_up<BM, BP, MW, PW, WM, WP, KST, NT, DM>
      <<<dim3(rows / BP, E), NT, 0, st>>>(codes, scales, cb, X, tokidx, eoff,
                                          gu, rows, cols, ss, Wbf);
}

static const KCfg kCfgs[] = {
    {"c0  256t BM128 deq-fp32", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 0>},
    {"c0p 256t BM128 deq-pairs", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 1>},
    {"c0e 256t BM128 deq-early", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 2>},
    {"c0b 256t BM128 bf16-lowb", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 3>},
    {"c0s 256t BM128 deq-sect", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 4>},
    {"c0q 256t BM128 deq-fp32cbp", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 5>},
    {"c0r 256t BM128 deq-ldg", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 6>},
    {"c1p 256t BM64 deq-pairs", 64, launch_cfg<64, 256, 2, 4, 2, 4, 32, 256, 1>},
    {"c0h 256t BM128 deq-shfl", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 9>},
    {"c0f 256t BM128 fix0-diag", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 10>},
    {"c1s 256t BM128 sect-exact", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 11>},
    {"c0n 256t BM128 notbl-diag", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 13>},
    {"c2s 256t BM64 sect-exact", 64, launch_cfg<64, 256, 2, 4, 2, 4, 32, 256, 11>},
    {"c3s 256t BM128 sect-lean", 128, launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 15>},
    {"c4s 256t BM64 sect-lean", 64, launch_cfg<64, 256, 2, 4, 2, 4, 32, 256, 15>},
};

int main(int argc, char** argv) {
  const int E = 512, P = 8192, TOPK = 10;
  const int rows = 1280, cols = 2560;
  const int64_t ss = cols / 32 * 2;  // scale_stride bytes per row
  const int NP = P * TOPK;
  const double flop_eff = 2.0 * NP * rows * cols;
  const char* cfgfilt = getenv("PROTO_CFG");
  const char* casesel = getenv("PROTO_CASE");
  const int nocheck = getenv("PROTO_NOCHECK") != nullptr;

  // ---- host data ----
  xs_state = 0xC0FFEE;
  std::vector<uint8_t> hcodes((size_t)E * rows * cols / 2);
  for (auto& v : hcodes) v = (uint8_t)xs_next();
  std::vector<uint16_t> hscales((size_t)E * rows * cols / 32);
  for (auto& v : hscales) {  // fp16 bits, (0.5, 1.5]
    const __half t = __float2half(0.5f + 0.5f * (xs_frand() + 1.f));
    memcpy(&v, &t, 2);
  }
  std::vector<float> hcb(16);
  for (int i = 0; i < 16; ++i) hcb[i] = -2.f + i * 0.27f + xs_frand() * 0.05f;
  std::vector<uint16_t> hX((size_t)P * cols);
  for (auto& v : hX) v = f2bf(xs_frand());

  uint8_t *dcodes, *dscales;
  float *dcb, *dgu;
  uint16_t* dX;
  int *dtokidx, *deoff;
  CK(hipMalloc(&dcodes, hcodes.size()));
  CK(hipMalloc(&dscales, hscales.size() * 2));
  CK(hipMalloc(&dcb, 16 * 4));
  CK(hipMalloc(&dX, hX.size() * 2));
  CK(hipMalloc(&dgu, (size_t)NP * rows * 4));
  CK(hipMalloc(&dtokidx, NP * 4));
  CK(hipMalloc(&deoff, (E + 1) * 4));
  CK(hipMemcpy(dcodes, hcodes.data(), hcodes.size(), hipMemcpyHostToDevice));
  CK(hipMemcpy(dscales, hscales.data(), hscales.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(dcb, hcb.data(), 64, hipMemcpyHostToDevice));
  CK(hipMemcpy(dX, hX.data(), hX.size() * 2, hipMemcpyHostToDevice));

  // fully dequantized bf16 weights for the DM==3 lower-bound config
  uint16_t* dWbf;
  CK(hipMalloc(&dWbf, (size_t)E * rows * cols * 2));
  for (int e = 0; e < E; ++e)
    k_dequant_ref<<<(rows * cols / 32 + 255) / 256, 256>>>(
        dcodes + (size_t)e * rows * cols / 2, dscales + (size_t)e * rows * ss,
        dcb, dWbf + (size_t)e * rows * cols, rows, cols, ss);
  CK(hipDeviceSynchronize());

  rocblas_handle h;
  rocblas_create_handle(&h);
  hipStream_t st;
  CK(hipStreamCreate(&st));
  rocblas_set_stream(h, st);
  hipEvent_t ev0, ev1;
  CK(hipEventCreate(&ev0));
  CK(hipEventCreate(&ev1));

  // reference scratch (reused per expert)
  uint16_t *dWref, *dXg;
  float* dYref;
  CK(hipMalloc(&dWref, (size_t)rows * cols * 2));
  CK(hipMalloc(&dXg, (size_t)NP * cols * 2));   // skew ne can reach ~20x avg
  CK(hipMalloc(&dYref, (size_t)NP * rows * 4));

  const char* casenames[] = {"uniform", "skew"};
  for (int ci = 0; ci < 2; ++ci) {
    if (casesel && !strstr(casenames[ci], casesel)) continue;
    // routing: ne[] + tokidx (each token exactly TOPK times, shuffled)
    std::vector<int> ne(E), eoff(E + 1, 0), tokidx(NP);
    if (ci == 0) {
      for (int e = 0; e < E; ++e) ne[e] = NP / E;  // 160
    } else {
      std::vector<double> raw(E);
      double sum = 0;
      for (int e = 0; e < E; ++e) { raw[e] = exp(-e / 40.0) + 0.02; sum += raw[e]; }
      int acc = 0, maxe = 0;
      for (int e = 0; e < E; ++e) {
        ne[e] = std::max(1, (int)llround(raw[e] / sum * NP));
        acc += ne[e];
      }
      for (int e = 0; acc != NP; e = (e + 1) % E) { ne[e] += (acc < NP) ? 1 : -1; acc += (acc < NP) ? 1 : -1; if (ne[e] < 1) { ne[e] = 1; } }
      for (int e = 0; e < E; ++e) maxe = std::max(maxe, ne[e]);
      printf("skew: max ne=%d min ne=%d\n", maxe, *std::min_element(ne.begin(), ne.end()));
    }
    for (int e = 0; e < E; ++e) eoff[e + 1] = eoff[e] + ne[e];
    {
      std::vector<int> pool(NP);
      for (int t = 0; t < P; ++t)
        for (int k = 0; k < TOPK; ++k) pool[t * TOPK + k] = t;
      for (int i = NP - 1; i > 0; --i) {  // Fisher-Yates
        const int j = (int)(xs_next() % (uint32_t)(i + 1));
        std::swap(pool[i], pool[j]);
      }
      int pos = 0;
      for (int e = 0; e < E; ++e)
        for (int p = 0; p < ne[e]; ++p) tokidx[eoff[e] + p] = pool[pos++];
    }
    CK(hipMemcpy(dtokidx, tokidx.data(), NP * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(deoff, eoff.data(), (E + 1) * 4, hipMemcpyHostToDevice));
    printf("\n== case %s (E=%d rows=%d cols=%d NP=%d, %.1f GFLOP eff) ==\n",
           casenames[ci], E, rows, cols, NP, flop_eff / 1e9);
    fflush(stdout);

    const int ncfg = (int)(sizeof(kCfgs) / sizeof(kCfgs[0]));
    for (int c = 0; c < ncfg; ++c) {
      if (cfgfilt && !strstr(kCfgs[c].name, cfgfilt)) continue;
      const int BM = kCfgs[c].bm;
      kCfgs[c].fn(dcodes, dscales, dcb, dX, dtokidx, deoff, dgu, rows, cols, ss, dWbf, E, st);
      if (hipStreamSynchronize(st) != hipSuccess) {
        printf("  %-28s LAUNCH/RUN FAIL\n", kCfgs[c].name);
        (void)hipGetLastError();
        continue;
      }
      // ---- correctness vs dequant + rocBLAS, per expert ----
      if (!nocheck) {
        double worst = 0;
        int worst_e = -1;
        for (int e = 0; e < E; ++e) {
          const int n0 = eoff[e], n_e = ne[e];
          k_dequant_ref<<<(rows * cols / 32 + 255) / 256, 256, 0, st>>>(
              dcodes + (size_t)e * rows * cols / 2,
              dscales + (size_t)e * rows * ss, dcb, dWref, rows, cols, ss);
          k_gather_x<<<(unsigned)(((int64_t)n_e * cols / 8 + 255) / 256), 256, 0, st>>>(
              dX, dtokidx, n0, n_e, dXg, cols);
          float one = 1.f, zero = 0.f;
          rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none,
                          rows, n_e, cols, &one, dWref, rocblas_datatype_bf16_r,
                          cols, dXg, rocblas_datatype_bf16_r, cols, &zero, dYref,
                          rocblas_datatype_f32_r, rows, dYref,
                          rocblas_datatype_f32_r, rows, rocblas_datatype_f32_r,
                          rocblas_gemm_algo_standard, 0, 0);
          CK(hipStreamSynchronize(st));
          std::vector<float> vR((size_t)n_e * rows);
          CK(hipMemcpy(vR.data(), dYref, vR.size() * 4, hipMemcpyDeviceToHost));
          std::vector<float> vC((size_t)n_e * rows);
          CK(hipMemcpy(vC.data(), dgu + (size_t)n0 * rows, vC.size() * 4,
                       hipMemcpyDeviceToHost));
          for (size_t i = 0; i < vC.size(); ++i) {
            const double d = fabs((double)vC[i] - (double)vR[i]);
            const double rel = d / std::max(1.0, (double)fabs(vR[i]));
            if (rel > worst) { worst = rel; worst_e = e; }
          }
          const char* dbg = getenv("PROTO_DBG");
          if (dbg && atoi(dbg) == e) {
            int shown = 0;
            for (size_t i = 0; i < vC.size() && shown < 12; ++i) {
              const double d = fabs((double)vC[i] - (double)vR[i]);
              const double rel = d / std::max(1.0, (double)fabs(vR[i]));
              if (rel > 1e-2) {
                printf("    DBG e=%d p=%zu n=%zu kernel=%.4f ref=%.4f rel=%.3f\n",
                       e, i / rows, i % rows, vC[i], vR[i], rel);
                ++shown;
              }
            }
          }
        }
        printf("  %-28s maxrel %.3e (e=%d) %s\n", kCfgs[c].name, worst, worst_e,
               worst < 1e-2 ? "OK" : "BAD");
        fflush(stdout);
      }
      usleep(1500 * 1000);  // thermal spacing
      // ---- bench ----
      for (int i = 0; i < 3; ++i)
        kCfgs[c].fn(dcodes, dscales, dcb, dX, dtokidx, deoff, dgu, rows, cols, ss, dWbf, E, st);
      CK(hipStreamSynchronize(st));
      std::vector<float> ts(20);
      for (int i = 0; i < 20; ++i) {
        CK(hipEventRecord(ev0, st));
        kCfgs[c].fn(dcodes, dscales, dcb, dX, dtokidx, deoff, dgu, rows, cols, ss, dWbf, E, st);
        CK(hipEventRecord(ev1, st));
        CK(hipStreamSynchronize(st));
        CK(hipEventElapsedTime(&ts[i], ev0, ev1));
      }
      std::sort(ts.begin(), ts.end());
      const double ms = ts[ts.size() / 2];
      double issued = 0;
      for (int e = 0; e < E; ++e)
        issued += 2.0 * ((ne[e] + BM - 1) / BM) * BM * rows * cols;
      printf("  %-28s %8.3f ms  eff %5.1f TFLOPS  issued %5.1f TFLOPS\n",
             kCfgs[c].name, ms, flop_eff / ms / 1e9, issued / ms / 1e9);
      fflush(stdout);
      usleep(1500 * 1000);
    }
  }
  rocblas_destroy_handle(h);
  return 0;
}
