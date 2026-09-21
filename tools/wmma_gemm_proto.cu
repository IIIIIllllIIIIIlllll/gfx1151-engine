// wmma_gemm_proto.cu — self-written bf16 WMMA dense GEMM prototype for gfx1151.
// Contract (mirrors Model::gemm, src/gpu/gdec.cpp:8907):
//   Y[P,N] f32 = X[P,K]bf16 * W[N,K]bf16^T,  all row-major, fp32 accum.
//   (rocBLAS col-major view: m=N, n=P, k=K, A=W transa=T lda=K, B=X ldb=K,
//    C=Y ldc=N.)
// Baseline: rocBLAS gemm_ex solution_index 1178 (fastest in-library).
// Build: hipcc -O2 -o /tmp/wmma_gemm_proto tools/wmma_gemm_proto.cu -lrocblas
// WMMA fragment layouts measured on this GPU (see gdec.cpp:3416):
//   A(r,k): r<8 -> lane 2r elem k; r>=8 -> lane 17+2(r-8) elem k (rest ignored)
//   B(k,c): lane c elem k AND lane 16+c elem k (duplicate)
//   C(r,c): lane c + 16*(r>=8), elem r%8
// Roles: WMMA "A" operand = X tile (token rows, contiguous K), "B" operand =
// W tile (weight rows, contiguous K) — a B fragment for output column n is
// exactly W[n][k0..k0+15], so no transpose exists anywhere. Epilogue lane c
// then walks consecutive n, i.e. coalesced Y[p][n] stores.
// Final design (cfg "d9"): 128x256 block tile, 256 threads (8 warps 2x4),
//   warp tile 64x64 (4x4 fragments), K-step 32, LDS double buffer (62.5 KB).
//   Key lessons: unconditional global->reg->LDS staging (conditional staging
//   made the allocator spill staging regs to scratch, exposing full DRAM
//   latency in the main loop — the single biggest bug); an LDS-only barrier
//   (__syncthreads() also emits buffer_gl0_inv on gfx11, draining the memory
//   pipeline every K-step); probe-calibrated bank-conflict-free fragment
//   addressing (X dummy-row lane assignment, W 8-row region layout);
//   grouped-M CTA swizzle (PROTO_GM, default 4).
// Measured on gfx1151 (ROCm 7.1.1, median of 20, sol 1178 in-process):
//   N=2560 K=6144: ~24.5 vs 23.4 | N=6144/10240/12288 K=2560: ~37.4-37.8
//   vs ~32 TFLOPS. Raw WMMA issue ceiling is only ~55 TFLOPS on this part;
//   a staging-free compute skeleton (PROTO_MODE=2) reaches ~46 TFLOPS, so
//   the remaining gap is LDS staging latency + barrier cost, not DRAM
//   (~130 MB read per dispatch, far under bandwidth).
// Env: PROTO_GM=<group_m> (default 4)  PROTO_SOL=<id|-1> (default 1178)
//      PROTO_CFG=<name substring filter>
#define ROCBLAS_BETA_FEATURES_API
#ifndef PROTO_MODE
#define PROTO_MODE 0  // 0=full, 1=no-WMMA (memory path), 2=no-staging (compute path)
#endif
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
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
// Barrier that synchronizes LDS only: __syncthreads() on gfx11 also emits
// buffer_gl0_inv (global L0 invalidate) every call, which drains the memory
// pipeline once per K-step. Inputs are read-only, so L0 invalidation is pure
// overhead; LDS visibility only needs the lgkmcnt/vscnt drains.
__device__ __forceinline__ void lds_sync() {
  __asm__ volatile(
      "s_waitcnt lgkmcnt(0)\n\t"
      "s_waitcnt_vscnt null, 0x0\n\t"
      "s_barrier" ::
          : "memory");
}

// Block tile BM(tokens) x BP(weights), NT threads = (NT/32) warps as MW x PW,
// warp owns WM x WP 16x16 frags. K stepped by KST (2 WMMA K16 steps at 32).
// X tile staged row-major (stride KST+8 elems, conflict-free 16B units; A
// fragments touch only 8 distinct rows per instr so no region trick needed).
// W tile staged row-major but grouped in 8-row regions, region stride
// 8*(KST+8)+8 elems: B fragment reads 16 consecutive rows and each 8-lane
// phase must see distinct banks. Global staging regs are named scalars (not
// arrays/lambdas) so the allocator keeps them in VGPRs across the compute
// section. Double-buffered LDS, one barrier per K step, prefetch distance
// PFD K-steps.
template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT,
          int PFD>
__global__ void __launch_bounds__(NT)
    k_wmma_gemm(const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
                float* __restrict__ Y, int P, int K, int N, int gm) {
  constexpr int LSA = KST + 8;              // LDS row stride (elems)
  constexpr int BREG = 8 * LSA + 8;         // W 8-row region stride (elems)
  constexpr int NW = NT / 32;               // warps per block
  constexpr int AN = BM * (KST / 8) / NT;   // uint4 X loads per thread
  constexpr int BN = BP * (KST / 8) / NT;   // uint4 W loads per thread
  static_assert(MW * PW == NW, "warp grid mismatch");
  static_assert(BM == MW * WM * 16 && BP == PW * WP * 16, "tile mismatch");
  static_assert(BM * (KST / 8) % NT == 0 && BP * (KST / 8) % NT == 0,
                "staging mismatch");
  __shared__ uint16_t As[2][BM * LSA];
  __shared__ uint16_t Bs[2][(BP / 8) * BREG];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  // grouped swizzle for L2/MALL reuse (group along token blocks)
  const int num_m = P / BM, num_n = N / BP;
  const int bid = blockIdx.x;
  const int per_group = gm * num_n;
  const int gid = bid / per_group, rem = bid % per_group;
  const int first_m = gid * gm;
  const int gs = min(gm, num_m - first_m);
  const int bm = first_m + rem % gs;
  const int bn = rem / gs;
  const int m0 = bm * BM, n0 = bn * BP;

  const int wm0 = (w / PW) * (WM * 16);
  const int wp0 = (w % PW) * (WP * 16);
  // WMMA A layout: lane 2r -> row r, lane 17+2(r-8) -> row 8+r; other lanes
  // are ignored by the hardware. The ignored lanes are assigned complementary
  // dummy rows so every 8-lane LDS phase touches 8 distinct banks (measured
  // with tools/../lds probe: the naive clamp-to-row-0 costs 1 conflict per
  // ds_load_b128; this assignment is conflict-free with plain LSA stride).
  const int base_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? base_ : ((base_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((base_ + 4) & 7) : base_);
  const int ksteps = K / KST;

  qw_floatx8 acc[WM][WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  // loop-invariant LDS byte offsets for this lane's fragment reads
  uint32_t aoff[WM], boff[WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
    aoff[i] = ((wm0 + i * 16 + qrowA) * LSA) * 2;
#pragma unroll
  for (int j = 0; j < WP; ++j) {
    const int pl = wp0 + j * 16 + (lane & 15);
    boff[j] = ((pl >> 3) * BREG + (pl & 7) * LSA) * 2;
  }
  // staging: bank-phase-paired row mapping. A warp stages RPW=32/(KST/8)
  // rows x (KST/8) 16B units; a ds_store_b128 retires in 8-lane phases, and
  // with the LSA=80B row stride any two CONSECUTIVE rows in a phase always
  // collide on one 4-bank group (the second row's 4 blocks always include
  // byte offset 128). Pairing rows (b, b+RPW/2) per phase instead makes each
  // phase cover all 32 banks exactly once. Same warp-global address set, so
  // global coalescing is unchanged; LDS layout and all load paths untouched.
  constexpr int Q8 = KST / 8;
  constexpr int RPW = 32 / Q8;
  static_assert(RPW % 2 == 0 && KST % 8 == 0, "store mapping mismatch");
  const uint16_t* ga[AN];
  const uint16_t* gb[BN];
  uint32_t saoff[AN], sboff[BN];
#pragma unroll
  for (int j = 0; j < AN; ++j) {
    const int idx = tid + j * NT;
    const int q = idx % Q8, b = (idx / Q8) % RPW;
    const int row = (idx / (Q8 * RPW)) * RPW + (b & 1) * (RPW / 2) + (b / 2);
    ga[j] = X + (size_t)(m0 + row) * K + q * 8;
    saoff[j] = (row * LSA + q * 8) * 2;
  }
#pragma unroll
  for (int j = 0; j < BN; ++j) {
    const int idx = tid + j * NT;
    const int q = idx % Q8, b = (idx / Q8) % RPW;
    const int row = (idx / (Q8 * RPW)) * RPW + (b & 1) * (RPW / 2) + (b / 2);
    gb[j] = W + (size_t)(n0 + row) * K + q * 8;
    sboff[j] = ((row >> 3) * BREG + (row & 7) * LSA + q * 8) * 2;
  }

#define STAGE_LOAD_SET(RA, RB, KS)                       \
  do {                                                    \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)        \
        RA[j] = *(const uint4*)(ga[j] + (KS)*KST);        \
    _Pragma("unroll") for (int j = 0; j < BN; ++j)        \
        RB[j] = *(const uint4*)(gb[j] + (KS)*KST);        \
  } while (0)
#define STAGE_STORE_SET(BUF, RA, RB)                               \
  do {                                                               \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)                   \
        *(uint4*)((char*)&As[BUF][0] + saoff[j]) = RA[j];            \
    _Pragma("unroll") for (int j = 0; j < BN; ++j)                   \
        *(uint4*)((char*)&Bs[BUF][0] + sboff[j]) = RB[j];            \
  } while (0)
#define COMPUTE_STEP(CUR)                                                \
  do {                                                                   \
    const char* Ab = (const char*)&As[CUR][0];                           \
    const char* Bb = (const char*)&Bs[CUR][0];                           \
    _Pragma("unroll") for (int kh = 0; kh < KST / 16; ++kh) {            \
      qw_shortx16 af[WM];                                                \
      _Pragma("unroll") for (int i = 0; i < WM; ++i)                     \
          af[i] = qw_ld16((const uint16_t*)(Ab + aoff[i] + kh * 32));    \
      _Pragma("unroll") for (int j = 0; j < WP; ++j) {                   \
        const qw_shortx16 bfj =                                          \
            qw_ld16((const uint16_t*)(Bb + boff[j] + kh * 32));          \
        _Pragma("unroll") for (int i = 0; i < WM; ++i)                   \
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(     \
                af[i], bfj, acc[i][j]);                                  \
      }                                                                  \
    }                                                                    \
  } while (0)

  uint4 ra0[AN], rb0[BN], ra1[AN], rb1[BN];
  STAGE_LOAD_SET(ra0, rb0, 0);
  STAGE_STORE_SET(0, ra0, rb0);
  if (PFD == 2) STAGE_LOAD_SET(ra1, rb1, min(1, ksteps - 1));
  lds_sync();

#pragma unroll 1
  for (int ks = 0; ks < ksteps; ks += PFD) {
#if PROTO_MODE != 2
    STAGE_LOAD_SET(ra0, rb0, min(ks + PFD, ksteps - 1));
#endif
    if (PFD == 2) COMPUTE_STEP(0);
    else COMPUTE_STEP(ks & 1);
#if PROTO_MODE != 2
    if (PFD == 2) STAGE_STORE_SET(1, ra1, rb1);
    else STAGE_STORE_SET((ks & 1) ^ 1, ra0, rb0);
#endif
    lds_sync();
    if (PFD == 2) {
#if PROTO_MODE != 2
      STAGE_LOAD_SET(ra1, rb1, min(ks + 3, ksteps - 1));
#endif
      COMPUTE_STEP(1);
#if PROTO_MODE != 2
      STAGE_STORE_SET(0, ra0, rb0);
#endif
      lds_sync();
    }
  }

  const int rh = (lane >> 4) * 8, cl = lane & 15;
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        Y[(size_t)(m0 + wm0 + i * 16 + rh + e) * N + n0 + wp0 + j * 16 + cl] =
            acc[i][j][e];
}

struct KCfg {
  const char* name;
  int bm, bp;
  void (*fn)(const uint16_t*, const uint16_t*, float*, int, int, int, int,
             hipStream_t);
};

// Direct-global variant: no LDS, no barriers in the K loop. Each warp loads
// its own fragments straight from global memory; intra-block reuse (MW warps
// share W rows, PW warps share X rows; both kh halves share one 64B sector)
// is absorbed by L0/L1, inter-block reuse by MALL. Occupancy is limited only
// by VGPRs, so several blocks per WGP hide load latency.
template <int BM, int BP, int MW, int PW, int WM, int WP, int NT>
__global__ void __launch_bounds__(NT)
    k_wmma_direct(const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
                  float* __restrict__ Y, int P, int K, int N, int gm) {
  constexpr int NW = NT / 32;
  static_assert(MW * PW == NW, "warp grid mismatch");
  static_assert(BM == MW * WM * 16 && BP == PW * WP * 16, "tile mismatch");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int num_m = P / BM, num_n = N / BP;
  const int bid = blockIdx.x;
  const int per_group = gm * num_n;
  const int gid = bid / per_group, rem = bid % per_group;
  const int first_m = gid * gm;
  const int gs = min(gm, num_m - first_m);
  const int m0 = (first_m + rem % gs) * BM;
  const int n0 = (rem / gs) * BP;
  const int wm0 = (w / PW) * (WM * 16);
  const int wp0 = (w % PW) * (WP * 16);
  const int qrowA = (lane < 16 && !(lane & 1))     ? lane / 2
                    : (lane >= 17 && (lane & 1))   ? 8 + (lane - 17) / 2
                                                   : 0;
  const uint16_t* pa[WM];
  const uint16_t* pb[WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
    pa[i] = X + (size_t)(m0 + wm0 + i * 16 + qrowA) * K;
#pragma unroll
  for (int j = 0; j < WP; ++j)
    pb[j] = W + (size_t)(n0 + wp0 + j * 16 + (lane & 15)) * K;
  qw_floatx8 acc[WM][WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
#pragma unroll 1
  for (int k0 = 0; k0 < K; k0 += 32) {
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
              __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[i], bf[j], acc[i][j]);
    }
  }
  const int rh = (lane >> 4) * 8, cl = lane & 15;
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        Y[(size_t)(m0 + wm0 + i * 16 + rh + e) * N + n0 + wp0 + j * 16 + cl] =
            acc[i][j][e];
}

template <int BM, int BP, int MW, int PW, int WM, int WP, int NT>
void launch_direct(const uint16_t* W, const uint16_t* X, float* Y, int N,
                   int K, int P, int gm, hipStream_t st) {
  const int grid = (P / BM) * (N / BP);
  k_wmma_direct<BM, BP, MW, PW, WM, WP, NT>
      <<<grid, NT, 0, st>>>(X, W, Y, P, K, N, gm);
}

template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT,
          int PFD>
void launch_cfg(const uint16_t* W, const uint16_t* X, float* Y, int N, int K,
                int P, int gm, hipStream_t st) {
  const int grid = (P / BM) * (N / BP);
  k_wmma_gemm<BM, BP, MW, PW, WM, WP, KST, NT, PFD>
      <<<grid, NT, 0, st>>>(X, W, Y, P, K, N, gm);
}

static const KCfg kCfgs[] = {
    {"d9 256t 128x256 w2x4 f4x4 k32", 128, 256,
     launch_cfg<128, 256, 2, 4, 4, 4, 32, 256, 1>},
    {"d3 512t 128x256 w2x8 f4x2 k32", 128, 256,
     launch_cfg<128, 256, 2, 8, 4, 2, 32, 512, 1>},
    {"d0 256t 128x128 w4x2 f2x4 k32", 128, 128,
     launch_cfg<128, 128, 4, 2, 2, 4, 32, 256, 1>},
    {"e0 dir 128x128 w4x2 f2x4", 128, 128,
     launch_direct<128, 128, 4, 2, 2, 4, 256>},
};

static uint32_t xs_state = 0x12345678u;
static uint32_t xs_next() {
  uint32_t x = xs_state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
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
  // shapes: N K P triples (default: the four P=8192 + four P=32768)
  std::vector<int> shp;
  for (int i = 1; i + 2 < argc; i += 3) {
    shp.push_back(atoi(argv[i]));
    shp.push_back(atoi(argv[i + 1]));
    shp.push_back(atoi(argv[i + 2]));
  }
  if (shp.empty())
    shp = {2560,  6144, 8192,  6144,  2560,  8192,
           10240, 2560, 8192,  12288, 2560,  8192,
           2560,  6144, 32768, 6144,  2560,  32768,
           10240, 2560, 32768, 12288, 2560, 32768};
  const int gm = getenv("PROTO_GM") ? atoi(getenv("PROTO_GM")) : 4;
  const int sol = getenv("PROTO_SOL") ? atoi(getenv("PROTO_SOL")) : 1178;
  const char* cfgfilt = getenv("PROTO_CFG");  // substring filter, e.g. "d0,d6"
  const int ncfg = (int)(sizeof(kCfgs) / sizeof(kCfgs[0]));

  rocblas_handle h;
  rocblas_create_handle(&h);
  hipStream_t st;
  CK(hipStreamCreate(&st));
  rocblas_set_stream(h, st);
  hipEvent_t ev0, ev1;
  CK(hipEventCreate(&ev0));
  CK(hipEventCreate(&ev1));

  for (size_t s = 0; s < shp.size(); s += 3) {
    const int N = shp[s], K = shp[s + 1], P = shp[s + 2];
    if (N % 256 || K % 32 || P % 256) {
      printf("skip N=%d K=%d P=%d (not tile-divisible)\n", N, K, P);
      continue;
    }
    const double gflop = 2.0 * N * K * P / 1e9;
    printf("\n== N=%d K=%d P=%d (%.1f GFLOP) ==\n", N, K, P, gflop);
    fflush(stdout);

    std::vector<uint16_t> hA((size_t)N * K), hB((size_t)P * K);
    fill_bf16(hA.data(), hA.size(), 0xA5A5u + (unsigned)s);
    fill_bf16(hB.data(), hB.size(), 0x5A5Au + (unsigned)s);
    uint16_t *A, *B;
    float *C, *Cref;
    CK(hipMalloc(&A, hA.size() * 2));
    CK(hipMalloc(&B, hB.size() * 2));
    CK(hipMalloc(&C, (size_t)N * P * 4));
    CK(hipMalloc(&Cref, (size_t)N * P * 4));
    CK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(B, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
    float one = 1.f, zero = 0.f;

    auto rb = [&](rocblas_gemm_algo algo, int idx, float* out) {
      return rocblas_gemm_ex(h, rocblas_operation_transpose,
                             rocblas_operation_none, N, P, K, &one, A,
                             rocblas_datatype_bf16_r, K, B,
                             rocblas_datatype_bf16_r, K, &zero, out,
                             rocblas_datatype_f32_r, N, out,
                             rocblas_datatype_f32_r, N, rocblas_datatype_f32_r,
                             algo, idx, 0);
    };
    auto bench = [&](auto&& fn) {
      for (int i = 0; i < 3; ++i) fn();
      CK(hipStreamSynchronize(st));
      std::vector<float> ts(20);
      for (int i = 0; i < 20; ++i) {
        CK(hipEventRecord(ev0, st));
        fn();
        CK(hipEventRecord(ev1, st));
        CK(hipStreamSynchronize(st));
        CK(hipEventElapsedTime(&ts[i], ev0, ev1));
      }
      std::sort(ts.begin(), ts.end());
      return (double)ts[ts.size() / 2];
    };

    if (rb(rocblas_gemm_algo_standard, 0, Cref) != rocblas_status_success) {
      printf("  rocBLAS ref failed, skip\n");
      goto next;
    }
    CK(hipStreamSynchronize(st));

    {
      rocblas_status r0 = rb(rocblas_gemm_algo_solution_index, sol, Cref);
      CK(hipStreamSynchronize(st));
      const bool use_sol = (r0 == rocblas_status_success);
      if (!use_sol)
        printf("  sol %d unavailable (rc=%d), baseline = algo_standard\n", sol,
               (int)r0);
      const double ms = bench([&] {
        rb(use_sol ? rocblas_gemm_algo_solution_index
                   : rocblas_gemm_algo_standard,
           use_sol ? sol : 0, Cref);
      });
      printf("  rocBLAS %s: %8.3f ms  %6.1f TFLOPS\n",
             use_sol ? "sol1178" : "standard", ms, gflop / ms);
      fflush(stdout);
    }

    for (int c = 0; c < ncfg; ++c) {
      if (P % kCfgs[c].bm || N % kCfgs[c].bp) continue;
      if (cfgfilt && !strstr(kCfgs[c].name, cfgfilt)) continue;
      kCfgs[c].fn(A, B, C, N, K, P, gm, st);
      if (hipStreamSynchronize(st) != hipSuccess) {
        printf("  %-29s LAUNCH/RUN FAIL\n", kCfgs[c].name);
        (void)hipGetLastError();
        continue;
      }
      std::vector<float> vC((size_t)N * P), vR((size_t)N * P);
      CK(hipMemcpy(vC.data(), C, vC.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(vR.data(), Cref, vR.size() * 4, hipMemcpyDeviceToHost));
      double maxabs = 0, maxrel = 0;
      for (size_t i = 0; i < vC.size(); ++i) {
        const double d = fabs((double)vC[i] - (double)vR[i]);
        maxabs = std::max(maxabs, d);
        maxrel = std::max(maxrel, d / std::max(1.0, (double)fabs(vR[i])));
      }
      usleep(1000 * 1000);  // thermal spacing between config benches
      const double ms = bench([&] { kCfgs[c].fn(A, B, C, N, K, P, gm, st); });
      printf("  %-29s %8.3f ms  %6.1f TFLOPS   maxabs %.3e maxrel %.3e %s\n",
             kCfgs[c].name, ms, gflop / ms, maxabs, maxrel,
             maxrel < 1e-2 ? "OK" : "BAD");
      fflush(stdout);
    }
  next:
    CK(hipFree(A));
    CK(hipFree(B));
    CK(hipFree(C));
    CK(hipFree(Cref));
    sleep(3);  // let the iGPU cool between shapes
  }
  rocblas_destroy_handle(h);
  CK(hipStreamDestroy(st));
  return 0;
}
