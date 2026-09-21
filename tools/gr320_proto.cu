// gr320_proto.cu — feasibility: beat hipBLASLt idx4 (~19.7 TFLOPS) on the
// narrow dense GEMM  y[P][320] f32 = x[P][10240]bf16 @ w[320][10240]bf16^T
// (GR mix, P=16384/32821). k_gemm_wmma's three LDS paths are closed for this
// shape (BP=256 62.5% N-tail waste; BP=64 5x X re-read; BP=320 LDS 73KB>64KB
// + 20-frag register wall). Candidates here:
//   c0: hipBLASLt idx4 (production baseline, mirrors gemm() descriptor path)
//   c1: v_dot2 register-streaming (T=32 tokens x Gn=2 columns per warp-pass,
//       KC=256; w amortized P/T, x streamed once; acc = T*Gn = 64/lane).
//       Analysis: lanes-split-K => acc/lane = Gn*Gt; w amortization needs
//       Gt>=32, so Gn<=2 and the fdot2:x-load ratio caps at ~8:1 -> ceiling
//       ~0.86 * (128 FLOP/instr issue rate) ~ 22 TFLOPS. Measured to confirm.
//   c2: direct-global WMMA, BP=320 exact (no LDS, no staging regs; w frags
//       read straight from global, MALL-hot; x rows via L1). 4096 FLOP/instr
//       sidesteps the v_dot2 density cap. Configs: w2x4 f2x5 (256t, 10 frags)
//       and w1x4 f4x5 (128t, 20 frags).
// Correctness: rocBLAS algo_standard ref + fro-rel vs the Lt output.
// Build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/gr320_proto tools/gr320_proto.cu -lrocblas -lhipblaslt
// Run:   /tmp/gr320_proto [P=32821] [reps=10]   (env CFG=substring filter,
//        GM=swizzle (default 4; winner needs GM=1), OCC=1 prints reg/LDS
//        occupancy of key instantiations)
// Result (2026-09-21, 86% cap, median-of-10): h1 <64,160,2,2,2,5,64,128>
// GM=1 = 27.2/27.0 TF vs Lt 19.5-20.1/19.3 at P=32821/16384 (x1.37-1.45);
// KST=64 (128B contiguous per row per staging step) + BP=160 (num_n=2
// exact) + gm=1 are the three required ingredients. See ~/ppbench/BASELINE.md
// "Phase 3h".
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
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

constexpr int GN = 320, GK = 10240;
#ifndef UNR
#define UNR 8  // t-loop unroll for the v_dot2 kernel (x-load pipeline depth)
#endif

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
__device__ __forceinline__ __hip_bfloat162 ld_bf162(const uint16_t* p) {
  __hip_bfloat162 v;
  memcpy(&v, p, 4);
  return v;
}

// ---------------- c1: v_dot2 register-streaming -----------------------------
// Block: NT threads = NT/32 warps, T tokens (grid = ceil(P/T)). Warp w owns
// column pairs {2w, 2w+1} + 32*r for r < 320/(NT/32)/2 rounds. Per round the
// warp sweeps full K in KC=256 chunks: lanes split the chunk (lane covers
// 8 consecutive bf16 = 1 uint4), w pair chunks live in registers, x chunk is
// loaded once per (i,t) and feeds both columns (8 fdot2 per x-load). acc[t][j]
// is the lane's K-partial for (token t, column j); butterfly-reduced and
// stored at the end of each round.
template <int T, int NT>
__global__ void __launch_bounds__(NT)
    k_gr320_vdot(const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
                 float* __restrict__ Y, int P) {
  constexpr int NW = NT / 32;
  constexpr int RND = GN / (NW * 2);  // n-pair rounds per warp
  constexpr int KC = 256;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int t0 = blockIdx.x * T;
  const int ke = lane * 8;
  for (int r = 0; r < RND; ++r) {
    const int n0 = r * NW * 2 + w * 2;
    float acc[T][2];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t][0] = acc[t][1] = 0.f;
    const uint16_t* w0 = W + (size_t)n0 * GK + ke;
    const uint16_t* w1 = W + (size_t)(n0 + 1) * GK + ke;
#pragma unroll 1
    for (int i = 0; i < GK / KC; ++i) {
      const uint4 wa4 = *(const uint4*)(w0 + i * KC);
      const uint4 wb4 = *(const uint4*)(w1 + i * KC);
      __hip_bfloat162 wa[4], wb[4];
      memcpy(wa, &wa4, 16);
      memcpy(wb, &wb4, 16);
#pragma unroll UNR
      for (int t = 0; t < T; ++t) {
        const int tt = t0 + t < P ? t0 + t : P - 1;
        uint4 x4 = *(const uint4*)(X + (size_t)tt * GK + i * KC + ke);
        __hip_bfloat162 xv[4];
        memcpy(xv, &x4, 16);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc[t][0] = __builtin_amdgcn_fdot2_f32_bf16(xv[e], wa[e], acc[t][0],
                                                      false);
          acc[t][1] = __builtin_amdgcn_fdot2_f32_bf16(xv[e], wb[e], acc[t][1],
                                                      false);
        }
      }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float v = acc[t][j];
#pragma unroll
        for (int off = 16; off; off >>= 1)
          v += __shfl_xor_sync(~0ull, v, off, 32);
        if (lane == 0 && t0 + t < P) Y[(size_t)(t0 + t) * GN + n0 + j] = v;
      }
    }
  }
}

// ---------------- c2: direct-global WMMA, BP=320 ----------------------------
// Block tile BM x 320, NT threads = MW x PW warps, warp owns WM x WP 16x16
// frags (PW*WP*16 = 320). Fragment loads straight from global: A = x rows
// (16B/lane, K-contig, coalesced; ignored A lanes redundantly read row 0 and
// hit L1), B = w rows (MALL-hot after the first blocks). No LDS, no
// barriers, no staging registers -> the 73KB LDS wall and the staging-reg
// spill that killed the LDS BP=320 config do not apply. P tail: A rows clamp
// to P-1, stores predicated. N=320=BP exactly, so no N tail exists.
template <int BM, int MW, int PW, int WM, int WP, int NT>
__global__ void __launch_bounds__(NT)
    k_gr320_direct(const uint16_t* __restrict__ X,
                   const uint16_t* __restrict__ W, float* __restrict__ Y,
                   int P) {
  static_assert(MW * PW == NT / 32, "warp grid");
  static_assert(BM == MW * WM * 16 && PW * WP * 16 == GN, "tile");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int m0 = blockIdx.x * BM;
  const int wm0 = (w / PW) * (WM * 16);
  const int wp0 = (w % PW) * (WP * 16);
  const int qrowA = (lane < 16 && !(lane & 1))   ? lane / 2
                    : (lane >= 17 && (lane & 1)) ? 8 + (lane - 17) / 2
                                                 : 0;
  const uint16_t* pa[WM];
  const uint16_t* pb[WP];
#pragma unroll
  for (int i = 0; i < WM; ++i) {
    const int row = m0 + wm0 + i * 16 + qrowA;
    pa[i] = X + (size_t)(row < P ? row : P - 1) * GK;
  }
#pragma unroll
  for (int j = 0; j < WP; ++j)
    pb[j] = W + (size_t)(wp0 + j * 16 + (lane & 15)) * GK;
  qw_floatx8 acc[WM][WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
#pragma unroll 1
  for (int k0 = 0; k0 < GK; k0 += 32) {
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
      qw_shortx16 af[WM], bf[WP];
#pragma unroll
      for (int i = 0; i < WM; ++i)
        af[i] = qw_ld16(pa[i] + k0 + kh * 16);
#pragma unroll
      for (int j = 0; j < WP; ++j)
        bf[j] = qw_ld16(pb[j] + k0 + kh * 16);
#pragma unroll
      for (int i = 0; i < WM; ++i)
#pragma unroll
        for (int j = 0; j < WP; ++j)
          acc[i][j] =
              __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i], bf[j],
                                                          acc[i][j]);
    }
  }
  const int rh = (lane >> 4) * 8, cl = lane & 15;
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j) {
      const int p0 = m0 + wm0 + i * 16 + rh;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (p0 + e < P)
          Y[(size_t)(p0 + e) * GN + wp0 + j * 16 + cl] = acc[i][j][e];
    }
}

// c2p: software-prefetch variant of k_gr320_direct — fragment loads for
// k0+32 issued before the k0 compute, so global latency hides behind the
// WMMA block instead of being exposed in full every iteration (the measured
// c2 stall). One extra fragment register set.
template <int BM, int MW, int PW, int WM, int WP, int NT>
__global__ void __launch_bounds__(NT)
    k_gr320_direct_pf(const uint16_t* __restrict__ X,
                      const uint16_t* __restrict__ W, float* __restrict__ Y,
                      int P) {
  static_assert(MW * PW == NT / 32, "warp grid");
  static_assert(BM == MW * WM * 16 && PW * WP * 16 == GN, "tile");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int m0 = blockIdx.x * BM;
  const int wm0 = (w / PW) * (WM * 16);
  const int wp0 = (w % PW) * (WP * 16);
  const int qrowA = (lane < 16 && !(lane & 1))   ? lane / 2
                    : (lane >= 17 && (lane & 1)) ? 8 + (lane - 17) / 2
                                                 : 0;
  const uint16_t* pa[WM];
  const uint16_t* pb[WP];
#pragma unroll
  for (int i = 0; i < WM; ++i) {
    const int row = m0 + wm0 + i * 16 + qrowA;
    pa[i] = X + (size_t)(row < P ? row : P - 1) * GK;
  }
#pragma unroll
  for (int j = 0; j < WP; ++j)
    pb[j] = W + (size_t)(wp0 + j * 16 + (lane & 15)) * GK;
  qw_floatx8 acc[WM][WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
  qw_shortx16 af[2][WM], bf[2][WP];
#pragma unroll
  for (int i = 0; i < WM; ++i) af[0][i] = qw_ld16(pa[i]);
#pragma unroll
  for (int j = 0; j < WP; ++j) bf[0][j] = qw_ld16(pb[j]);
  int cur = 0;
#pragma unroll 1
  for (int k0 = 0; k0 < GK; k0 += 16) {
    const int kn = k0 + 16 < GK ? k0 + 16 : 0;
#pragma unroll
    for (int i = 0; i < WM; ++i) af[cur ^ 1][i] = qw_ld16(pa[i] + kn);
#pragma unroll
    for (int j = 0; j < WP; ++j) bf[cur ^ 1][j] = qw_ld16(pb[j] + kn);
#pragma unroll
    for (int i = 0; i < WM; ++i)
#pragma unroll
      for (int j = 0; j < WP; ++j)
        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            af[cur][i], bf[cur][j], acc[i][j]);
    cur ^= 1;
  }
  const int rh = (lane >> 4) * 8, cl = lane & 15;
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j) {
      const int p0 = m0 + wm0 + i * 16 + rh;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (p0 + e < P)
          Y[(size_t)(p0 + e) * GN + wp0 + j * 16 + cl] = acc[i][j][e];
    }
}

#include "gemm_wmma_kernel.inc"

struct KCfg {
  const char* name;
  void (*fn)(const uint16_t*, const uint16_t*, float*, int, hipStream_t);
};

template <int T, int NT>
void launch_vdot(const uint16_t* X, const uint16_t* W, float* Y, int P,
                 hipStream_t st) {
  k_gr320_vdot<T, NT><<<(P + T - 1) / T, NT, 0, st>>>(X, W, Y, P);
}
template <int BM, int MW, int PW, int WM, int WP, int NT>
void launch_direct(const uint16_t* X, const uint16_t* W, float* Y, int P,
                   hipStream_t st) {
  k_gr320_direct<BM, MW, PW, WM, WP, NT>
      <<<(P + BM - 1) / BM, NT, 0, st>>>(X, W, Y, P);
}
template <int BM, int MW, int PW, int WM, int WP, int NT>
void launch_direct_pf(const uint16_t* X, const uint16_t* W, float* Y, int P,
                      hipStream_t st) {
  k_gr320_direct_pf<BM, MW, PW, WM, WP, NT>
      <<<(P + BM - 1) / BM, NT, 0, st>>>(X, W, Y, P);
}
template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT>
void launch_lds(const uint16_t* X, const uint16_t* W, float* Y, int P,
                hipStream_t st) {
  const unsigned grid =
      (unsigned)(((P + BM - 1) / BM) * ((GN + BP - 1) / BP));
  static const int gm = getenv("GM") ? atoi(getenv("GM")) : 4;
  k_gemm_wmma<BM, BP, MW, PW, WM, WP, KST, NT>
      <<<grid, NT, 0, st>>>(X, W, Y, P, GK, GN, gm);
}

template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT>
void occ_lds(const char* name) {
  hipFuncAttributes a;
  hipFuncGetAttributes(&a,
                       (const void*)k_gemm_wmma<BM, BP, MW, PW, WM, WP, KST, NT>);
  int nb = 0;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &nb, (const void*)k_gemm_wmma<BM, BP, MW, PW, WM, WP, KST, NT>, NT, 0);
  printf("occ %-34s regs=%d lds=%zuB blocks/CU=%d\n", name, a.numRegs,
         a.localSizeBytes, nb);
}

static const KCfg kCfgs[] = {
    {"c1 vdot T32xG2 512t", launch_vdot<32, 512>},
    {"c1 vdot T64xG2 512t", launch_vdot<64, 512>},
    {"c2 dir 64x320 w2x4 f2x5 256t", launch_direct<64, 2, 4, 2, 5, 256>},
    {"c2p dir+pf 64x320 f2x5 256t", launch_direct_pf<64, 2, 4, 2, 5, 256>},
    {"c2p dir+pf 128x320 f4x5 256t", launch_direct_pf<128, 2, 4, 4, 5, 256>},
    {"c2p dir+pf 64x320 f4x5 128t", launch_direct_pf<64, 1, 4, 4, 5, 128>},
    {"c4 lds 64x320 w1x4 f4x5 k16 128t",
     launch_lds<64, 320, 1, 4, 4, 5, 16, 128>},
    {"c4 lds 64x320 w1x4 f4x5 k32 128t",
     launch_lds<64, 320, 1, 4, 4, 5, 32, 128>},
    {"c5 ldsN 64x64 w1x4 f4x1 k16 128t",
     launch_lds<64, 64, 1, 4, 4, 1, 16, 128>},
    {"c5 ldsN 64x128 w1x4 f4x2 k16 128t",
     launch_lds<64, 128, 1, 4, 4, 2, 16, 128>},
    {"c5 ldsN 128x128 w2x4 f4x2 k16 256t",
     launch_lds<128, 128, 2, 4, 4, 2, 16, 256>},
    {"c5 ldsN 128x64 w2x2 f4x2 k16 128t",
     launch_lds<128, 64, 2, 2, 4, 2, 16, 128>},
    {"c6 lds 160x320 w2x5 f5x4 k16 320t",
     launch_lds<160, 320, 2, 5, 5, 4, 16, 320>},
    {"c7 lds 320x320 w4x5 f5x4 k16 640t",
     launch_lds<320, 320, 4, 5, 5, 4, 16, 640>},
    {"c7 ldsN 64x192 w1x4 f4x3 k16 128t",
     launch_lds<64, 192, 1, 4, 4, 3, 16, 128>},
    {"c7 ldsN 64x256 w1x4 f4x4 k16 128t",
     launch_lds<64, 256, 1, 4, 4, 4, 16, 128>},
    {"d1 ldsK 128x32 w2x2 f4x2 k32 128t",
     launch_lds<128, 32, 2, 2, 4, 1, 32, 128>},
    {"d1 ldsK 128x32 w2x2 f4x2 k64 128t",
     launch_lds<128, 32, 2, 2, 4, 1, 64, 128>},
    {"d1 ldsK 64x64 w1x4 f4x1 k32 128t",
     launch_lds<64, 64, 1, 4, 4, 1, 32, 128>},
    {"d1 ldsK 64x128 w1x4 f4x2 k32 128t",
     launch_lds<64, 128, 1, 4, 4, 2, 32, 128>},
    {"d1 ldsK 128x32 w2x2 f4x1 k32 128t",
     launch_lds<128, 32, 2, 2, 4, 1, 32, 128>},
    {"e1 ldsK 64x128 w1x4 f4x2 k64 128t",
     launch_lds<64, 128, 1, 4, 4, 2, 64, 128>},
    {"e2 ldsK 64x64 w1x4 f4x1 k64 128t",
     launch_lds<64, 64, 1, 4, 4, 1, 64, 128>},
    {"e3 ldsK 64x160 w2x2 f2x5 k32 128t",
     launch_lds<64, 160, 2, 2, 2, 5, 32, 128>},
    {"e4 ldsK 64x96 w2x2 f2x3 k32 128t",
     launch_lds<64, 96, 2, 2, 2, 3, 32, 128>},
    {"e5 ldsK 128x128 w2x4 f4x2 k32 256t",
     launch_lds<128, 128, 2, 4, 4, 2, 32, 256>},
    {"f1 ldsK 64x32 w1x2 f4x1 k128 64t",
     launch_lds<64, 32, 1, 2, 4, 1, 128, 64>},
    {"f2 ldsK 32x128 w1x4 f2x2 k64 128t",
     launch_lds<32, 128, 1, 4, 2, 2, 64, 128>},
    {"f3 ldsK 64x128 w1x8 f4x1 k64 256t",
     launch_lds<64, 128, 1, 8, 4, 1, 64, 256>},
    {"g1 ldsK 64x128 w2x2 f2x4 k64 128t",
     launch_lds<64, 128, 2, 2, 2, 4, 64, 128>},
    {"g6 ldsK 32x160 w2x2 f1x5 k32 128t",
     launch_lds<32, 160, 2, 2, 1, 5, 32, 128>},
    {"g7 ldsK 64x112 w4x1 f1x7 k64 128t",
     launch_lds<64, 112, 4, 1, 1, 7, 64, 128>},
    {"g8 ldsK 64x80 w4x1 f1x5 k64 128t",
     launch_lds<64, 80, 4, 1, 1, 5, 64, 128>},
    {"g9 ldsK 64x160 w4x1 f1x10 k64 128t",
     launch_lds<64, 160, 4, 1, 1, 10, 64, 128>},
    {"h1 ldsK 64x160 w2x2 f2x5 k64 128t",
     launch_lds<64, 160, 2, 2, 2, 5, 64, 128>},
    {"h2 ldsK 64x160 w4x2 f1x5 k64 256t",
     launch_lds<64, 160, 4, 2, 1, 5, 64, 256>},
};

static uint32_t xs_state;
static uint32_t xs_next() {
  uint32_t x = xs_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  xs_state = x;
  return x;
}
static uint16_t f2bf16(float f) {  // RNE
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static void fill_bf16(uint16_t* p, size_t n, uint32_t seed) {
  xs_state = seed;
  for (size_t i = 0; i < n; ++i) {
    const float f = (float)((xs_next() >> 8) & 0xFFFF) / 32768.f - 1.f;
    p[i] = f2bf16(f);
  }
}

int main(int argc, char** argv) {
  const int P = argc > 1 ? atoi(argv[1]) : 32821;
  const int reps = argc > 2 ? atoi(argv[2]) : 10;
  const char* cfgfilt = getenv("CFG");
  if (getenv("OCC")) {
    occ_lds<64, 320, 1, 4, 4, 5, 16, 128>("c4 64x320 k16 128t");
    occ_lds<64, 320, 1, 4, 4, 5, 32, 128>("c4 64x320 k32 128t");
    occ_lds<64, 64, 1, 4, 4, 1, 16, 128>("c5 64x64 k16 128t");
    occ_lds<64, 256, 1, 4, 4, 4, 16, 128>("c7 64x256 k16 128t");
    occ_lds<160, 320, 2, 5, 5, 4, 16, 320>("c6 160x320 k16 320t");
    occ_lds<320, 320, 4, 5, 5, 4, 16, 640>("c7 320x320 k16 640t");
    occ_lds<64, 128, 1, 4, 4, 2, 32, 128>("d1 64x128 k32 128t");
    occ_lds<64, 128, 1, 4, 4, 2, 64, 128>("e1 64x128 k64 128t");
    occ_lds<64, 128, 1, 8, 4, 1, 64, 256>("f3 64x128 k64 256t");
    occ_lds<64, 160, 4, 1, 1, 10, 64, 128>("g9 64x160 k64 128t");
    occ_lds<64, 160, 4, 2, 1, 5, 64, 256>("h2 64x160 k64 256t");
    return 0;
  }
  const size_t lt_ws_bytes = (size_t)64 << 20;
  const double gflop = 2.0 * GN * GK * P / 1e9;

  rocblas_handle h;
  rocblas_create_handle(&h);
  hipblasLtHandle_t lth;
  hipblasLtCreate(&lth);
  hipStream_t st;
  CK(hipStreamCreate(&st));
  rocblas_set_stream(h, st);
  void* d_ltws;
  CK(hipMalloc(&d_ltws, lt_ws_bytes));
  hipEvent_t ev0, ev1;
  CK(hipEventCreate(&ev0));
  CK(hipEventCreate(&ev1));

  std::vector<uint16_t> hA((size_t)GN * GK), hB((size_t)P * GK);
  fill_bf16(hA.data(), hA.size(), 0xA5A5u + P);
  fill_bf16(hB.data(), hB.size(), 0x5A5Au + P);
  uint16_t *A, *B;
  float *C, *Cref, *Clt;
  CK(hipMalloc(&A, hA.size() * 2));
  CK(hipMalloc(&B, hB.size() * 2));
  CK(hipMalloc(&C, (size_t)GN * P * 4));
  CK(hipMalloc(&Cref, (size_t)GN * P * 4));
  CK(hipMalloc(&Clt, (size_t)GN * P * 4));
  CK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(B, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
  float one = 1.f, zero = 0.f;

  auto bench = [&](auto&& fn) {
    for (int i = 0; i < 3; ++i) fn();
    CK(hipStreamSynchronize(st));
    std::vector<float> ts(reps);
    for (int i = 0; i < reps; ++i) {
      CK(hipEventRecord(ev0, st));
      fn();
      CK(hipEventRecord(ev1, st));
      CK(hipStreamSynchronize(st));
      CK(hipEventElapsedTime(&ts[i], ev0, ev1));
    }
    std::sort(ts.begin(), ts.end());
    return (double)ts[ts.size() / 2];
  };
  std::vector<float> vR((size_t)GN * P), vL((size_t)GN * P);
  auto fro = [&](const float* got, const std::vector<float>& ref) {
    std::vector<float> vC((size_t)GN * P);
    CK(hipMemcpy(vC.data(), got, vC.size() * 4, hipMemcpyDeviceToHost));
    double sa = 0, sb = 0;
    for (size_t i = 0; i < vC.size(); ++i) {
      const double d = (double)vC[i] - (double)ref[i];
      sa += d * d;
      sb += (double)ref[i] * ref[i];
    }
    return sqrt(sa / std::max(1e-30, sb));
  };

  printf("== N=320 K=10240 P=%d (%.1f GFLOP, reps=%d) ==\n", P, gflop, reps);

  // rocBLAS standard ref
  if (rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none,
                      GN, P, GK, &one, A, rocblas_datatype_bf16_r, GK, B,
                      rocblas_datatype_bf16_r, GK, &zero, Cref,
                      rocblas_datatype_f32_r, GN, Cref,
                      rocblas_datatype_f32_r, GN, rocblas_datatype_f32_r,
                      rocblas_gemm_algo_standard, 0, 0) !=
      rocblas_status_success) {
    printf("rocBLAS ref failed\n");
    return 1;
  }
  CK(hipStreamSynchronize(st));
  CK(hipMemcpy(vR.data(), Cref, vR.size() * 4, hipMemcpyDeviceToHost));

  // production hipBLASLt path: heuristic, index 4 validated, else scan
  double lt_ms = 0;
  {
    hipblasLtMatmulDesc_t opd;
    hipblasLtMatmulDescCreate(&opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                    sizeof(opT));
    hipblasLtMatmulDescSetAttribute(opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                    sizeof(opN));
    hipblasLtMatrixLayout_t Al, Bl, Cl, Dl;
    hipblasLtMatrixLayoutCreate(&Al, HIP_R_16BF, GK, GN, GK);
    hipblasLtMatrixLayoutCreate(&Bl, HIP_R_16BF, GK, P, GK);
    hipblasLtMatrixLayoutCreate(&Cl, HIP_R_32F, GN, P, GN);
    hipblasLtMatrixLayoutCreate(&Dl, HIP_R_32F, GN, P, GN);
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
        sizeof(lt_ws_bytes));
    hipblasLtMatmulHeuristicResult_t hres[8];
    int nres = 0;
    hipblasLtMatmulAlgoGetHeuristic(lth, opd, Al, Bl, Cl, Dl, pref, 8, hres,
                                    &nres);
    auto try_run = [&](int i, float* out) {
      hipblasLtMatmul(lth, opd, &one, A, Al, B, Bl, &zero, out, Cl, out, Dl,
                      &hres[i].algo, d_ltws, lt_ws_bytes, st);
      if (hipStreamSynchronize(st) != hipSuccess) {
        (void)hipGetLastError();
        return false;
      }
      return true;
    };
    int bi = -1;
    if (nres > 4 && hres[4].state == HIPBLAS_STATUS_SUCCESS &&
        hres[4].workspaceSize <= lt_ws_bytes && try_run(4, Clt))
      bi = 4;
    else {
      double best = 1e30;
      for (int i = 0; i < nres; i++) {
        if (hres[i].state != HIPBLAS_STATUS_SUCCESS) continue;
        if (hres[i].workspaceSize > lt_ws_bytes) continue;
        if (!try_run(i, Clt)) continue;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 3; r++) try_run(i, Clt);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        if (ms < best) { best = ms; bi = i; }
      }
    }
    if (bi < 0) {
      printf("hipBLASLt: no algo\n");
      return 1;
    }
    lt_ms = bench([&] { try_run(bi, Clt); });
    CK(hipMemcpy(vL.data(), Clt, vL.size() * 4, hipMemcpyDeviceToHost));
    printf("c0 Lt idx%d:           %8.3f ms  %6.1f TFLOPS  fro-vs-rb %.2e\n",
           bi, lt_ms, gflop / lt_ms, fro(Clt, vR));
    fflush(stdout);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(Al);
    hipblasLtMatrixLayoutDestroy(Bl);
    hipblasLtMatrixLayoutDestroy(Cl);
    hipblasLtMatrixLayoutDestroy(Dl);
    hipblasLtMatmulDescDestroy(opd);
  }

  const int ncfg = (int)(sizeof(kCfgs) / sizeof(kCfgs[0]));
  int nbad = 0;
  for (int c = 0; c < ncfg; ++c) {
    if (cfgfilt && !strstr(kCfgs[c].name, cfgfilt)) continue;
    CK(hipMemset(C, 0, (size_t)GN * P * 4));
    kCfgs[c].fn(B, A, C, P, st);
    if (hipStreamSynchronize(st) != hipSuccess) {
      printf("%-28s LAUNCH/RUN FAIL\n", kCfgs[c].name);
      (void)hipGetLastError();
      ++nbad;
      continue;
    }
    const double frb = fro(C, vR), flt = fro(C, vL);
    usleep(200 * 1000);  // thermal spacing
    const double ms = bench([&] { kCfgs[c].fn(B, A, C, P, st); });
    printf("%-28s %8.3f ms  %6.1f TFLOPS  x%.2f vs Lt  fro-rb %.1e fro-Lt %.1e %s\n",
           kCfgs[c].name, ms, gflop / ms, lt_ms / ms, frb, flt,
           flt < 1e-3 ? "OK" : (flt < 3e-3 ? "ok-ish" : "BAD"));
    if (flt >= 3e-3) ++nbad;
    fflush(stdout);
  }
  hipblasLtDestroy(lth);
  rocblas_destroy_handle(h);
  CK(hipStreamDestroy(st));
  printf(nbad ? "RESULT: %d BAD\n" : "RESULT: ALL OK\n", nbad);
  return nbad != 0;
}
