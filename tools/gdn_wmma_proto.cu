// gdn_wmma_proto.cu — fp16 WMMA prototype for the k_gdn_intra matrix phases
// (src/gpu/gdec.cpp:5586). Replicates the per-(chunk,head) workflow at the
// production geometry (NT=1024, grid (nchunks,48), CH=64, DK=128):
//   phase 0: k tile + gate staging (identical both variants)
//   phase 1: attn = -beta[r]*(K.K^T)*exp(gcum[r]-gcum[j]), j<r   [WMMA-able]
//   phase 2: Ut5 triangular solve (verbatim fp32 copy, both variants)
//   phase 3: vp = attn@(v*beta), kcd = attn@(k*beta*exp(gcum))   [WMMA-able]
//   phase 4: attn2 = (Q.K^T)*exp(gcum[r]-gcum[j]), j<=r          [WMMA-able]
// Variant 0 keeps the production scalar code verbatim; variant 1 does 1/3/4
// with __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 (fp32 accum, fp32 gating
// epilogues; fragment layouts per gdec.cpp:3416).
// fp16 LDS plan (fits the ~63 KiB block budget): k16 [64][72] + bT [128][72]
// (v*beta pass, then k*beta*eg pass — bT is reused), attn16 overlaid on
// s_extra, q16 overlaid on bT in phase 4. Row stride 72 halves = 144 B =
// 36 words: 8-lane ds_b128 phases hit banks 0,4,...,28 (conflict-free).
// Build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/gdn_wmma_proto \
//          tools/gdn_wmma_proto.cu
// Run:   /tmp/gdn_wmma_proto [P=8192] [reps=10]
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                             \
  do {                                                                    \
    hipError_t e_ = (x);                                                  \
    if (e_ != hipSuccess) {                                               \
      fprintf(stderr, "hip error %d (%s) at %d\n", (int)e_,               \
              hipGetErrorString(e_), __LINE__);                           \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

using gp_shortx16 = __attribute__((ext_vector_type(16))) short;
using gp_floatx8 = __attribute__((ext_vector_type(8))) float;

// gfx11 __syncthreads() also emits buffer_gl0_inv, draining the memory
// pipeline (~1-2 us at 1 block/WGP). This kernel only needs LDS visibility +
// block convergence at its barriers (all global traffic is either
// load->reg->LDS staging, whose LDS stores transitively wait the loads, or
// writeout never re-read in-kernel), so an LDS-only barrier is sufficient.
#ifdef GP_LDSSYNC
#define BAR()                                              \
  __asm__ volatile("s_waitcnt lgkmcnt(0)\n\t"              \
                   "s_waitcnt_vscnt null, 0x0\n\t"         \
                   "s_barrier" ::: "memory")
#else
#define BAR() __syncthreads()
#endif

__device__ __forceinline__ gp_shortx16 gp_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  gp_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
__device__ __forceinline__ uint16_t gp_f2h(float f) {
  __half h = __float2half_rn(f);
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}

constexpr int CH = 64, DK = 128, NT = 1024;
constexpr int LDS16 = 72;  // fp16 tile row stride (elems), see header
// ws layout per chunk-head (mirrors gdec.cpp:5537)
constexpr int WS_ATTN2 = 64, WS_VP = WS_ATTN2 + 64 * 64,
              WS_KCD = WS_VP + 64 * 128, WS_FLOATS = WS_KCD + 64 * 128;

__device__ unsigned long long g_ph[4][8];  // [variant][slot]

__device__ __forceinline__ long long ph_clk() {
  __asm__ volatile("" ::: "memory");
  long long v = clock64();
  __asm__ volatile("" ::: "memory");
  return v;
}
#ifdef GP_NOPROF
#define PH_DECL(t)
#define PH_ACC(v, slot, t)
#else
#define PH_DECL(t) long long t = ph_clk()
#define PH_ACC(v, slot, t)                                        \
  do {                                                            \
    BAR();                                              \
    long long e_ = ph_clk();                                      \
    long long d_ = e_ - (t);                                      \
    if (threadIdx.x == 0 && d_ >= 0 && d_ < (1LL << 40))          \
      atomicAdd(&g_ph[v][slot], (unsigned long long)d_);          \
  } while (0)
#endif

// WMMA A-fragment row for this lane: hardware uses even lanes 0-14 (rows
// 0-7) and odd lanes 17-31 (rows 8-15); the ignored lanes get complementary
// dummy rows so every 8-lane LDS phase touches 8 distinct rows
// (probe-calibrated in wmma_gemm_proto.cu).
__device__ __forceinline__ int gp_rowA(int lane) {
  const int base_ = lane >> 1;
  return (lane & 1) ? ((lane & 16) ? base_ : ((base_ + 4) & 7))
                    : ((lane & 16) ? 8 + ((base_ + 4) & 7) : base_);
}

template <int V>
__global__ void __launch_bounds__(NT) k_gdn_proto(
    const float* __restrict__ qkv, const float* __restrict__ gb,
    const float* __restrict__ bb, float* __restrict__ ws, int P, int qkvstride) {
  const int c = blockIdx.x, h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  __shared__ float s_attn[CH][CH];
  __shared__ float s_extra[3072];
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH];
  __shared__ union {
    float kf[CH][DK + 4];                       // V0: fp32 k tile (33.8 KB)
    struct {
      uint16_t k16[CH][LDS16];                  // 9.2 KB
      uint16_t bT[DK][LDS16];                   // 18.4 KB staging (vbT, kbT, q16)
    } f16;
  } u;
  float* wsb = ws + ((size_t)c * 48 + h) * WS_FLOATS;
  float* wvp = wsb + WS_VP;
  float* wkcd = wsb + WS_KCD;
  const int t0 = c * CH;
  const int lane = tid & 31, w = tid >> 5;
  PH_DECL(tk);

  // ---- phase 0: gates + k tile ----
  if (V != 1) {
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      u.kf[t][d] = (t0 + t < P)
                       ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d]
                       : 0.f;
    }
  } else {
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      float v = (t0 + t < P)
                    ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d]
                    : 0.f;
      u.f16.k16[t][d] = gp_f2h(v);
    }
  }
  if (tid < CH) {
    s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
    s_beta[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
  }
  BAR();
  if (tid == 0) {
    float acc = 0.f;
    for (int i = 0; i < CH; i++) {
      acc += s_gcum[i];
      s_gcum[i] = acc;
    }
  }
  BAR();
  if (tid < CH) {
    s_eg[tid] = expf(s_gcum[tid]);
    wsb[tid] = s_gcum[tid];
  }
  PH_ACC(V, 0, tk);
  PH_DECL(t1);

  // ---- phase 1: attn = -beta[r]*(k[r].k[j])*exp(gcum[r]-gcum[j]), j<r ----
  if (V >= 2) {  // null-compute skeleton: same barriers/writes, no dot loop
    for (int i = tid; i < CH * CH; i += NT) s_attn[i / CH][i % CH] = 0.f;
  } else if (V == 0) {
    for (int i = tid; i < CH * CH; i += NT) {
      int r = i / CH, j = i % CH;
      float a = 0.f;
      if (j < r) {
        float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        for (int d = 0; d < DK; d += 4) {
          e0 += u.kf[r][d] * u.kf[j][d];
          e1 += u.kf[r][d + 1] * u.kf[j][d + 1];
          e2 += u.kf[r][d + 2] * u.kf[j][d + 2];
          e3 += u.kf[r][d + 3] * u.kf[j][d + 3];
        }
        float dot = (e0 + e1) + (e2 + e3);
        a = -s_beta[r] * dot * expf(s_gcum[r] - s_gcum[j]);
      }
      s_attn[r][j] = a;
    }
  } else {
    // 16 output frags (4x4), warp w owns frag (w/4, w%4); 8 k-steps of 16.
    if (w < 16) {
      const int fr = (w >> 2) * 16, fc = (w & 3) * 16;
      const uint16_t* arow = &u.f16.k16[fr + gp_rowA(lane)][0];
      const uint16_t* brow = &u.f16.k16[fc + (lane & 15)][0];
      gp_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < DK / 16; kk++) {
        gp_shortx16 a = gp_ld16(arow + kk * 16);
        gp_shortx16 b = gp_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      // C(r,c): lane c + 16*(r>=8), elem r%8
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col < r)
          a = -s_beta[r] * acc[e] * expf(s_gcum[r] - s_gcum[col]);
        s_attn[r][col] = a;
      }
    }
  }
  BAR();
  PH_ACC(V, 1, t1);
  PH_DECL(t2);

  // ---- phase 2: Ut5 solve (verbatim fp32 copy from gdec.cpp:5657) ----
  if (V == 3) {  // null+solve-ablation: keep only the bracketing barriers
    BAR();
  } else {
    const int g = tid >> 8, el = tid & 255;
    const int er = el >> 4, ec = el & 15;
    {
      const int col = er, mg = ec, base = g * 16;
      float solved = 0.f;
      for (int i = 1; i < 16; ++i) {
        float p = 0.f;
        if (mg < i) p = s_attn[base + i][base + mg] * solved;
        float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
        float value = s_attn[base + i][base + col];
#pragma unroll
        for (int gg = 0; gg < 16; gg += 2)
          value += __shfl_sync(~0ull, pair, gg, 16);
        if (i == mg && col < i) solved = value;
      }
      BAR();
      if (col < mg) s_attn[base + mg][base + col] = solved;
    }
    BAR();
    if (g < 3) {
      const int j = g + 1, b = g;
      float a[16];
#pragma unroll
      for (int n = 0; n < 16; n++) a[n] = s_attn[j * 16 + er][b * 16 + n];
      float u = a[ec];
#pragma unroll
      for (int n = 0; n < 16; n++)
        u += a[n] * s_attn[b * 16 + n][b * 16 + ec];
      s_extra[g * 256 + el] = u;
    }
    BAR();
    if (g < 3) {
      const int j = g + 1, b = g;
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[j * 16 + er][j * 16 + m];
      float t = s_extra[g * 256 + el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[g * 256 + m * 16 + ec];
      s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
    }
    BAR();
    if (g < 2) {
      const int j = g + 2, b = g;
      float a0[16], a1[16];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        a0[n] = s_attn[j * 16 + er][b * 16 + n];
        a1[n] = s_attn[j * 16 + er][(b + 1) * 16 + n];
      }
      float u = a0[ec];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        u += a0[n] * s_attn[b * 16 + n][b * 16 + ec];
        u += a1[n] * s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 + ec];
      }
      s_extra[g * 256 + el] = u;
    }
    BAR();
    if (g < 2) {
      const int j = g + 2, b = g;
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[j * 16 + er][j * 16 + m];
      float t = s_extra[g * 256 + el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[g * 256 + m * 16 + ec];
      s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
    }
    BAR();
    if (g == 0) {
      float a0[16], a1[16], a2[16];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        a0[n] = s_attn[48 + er][n];
        a1[n] = s_attn[48 + er][16 + n];
        a2[n] = s_attn[48 + er][32 + n];
      }
      float u = a0[ec];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        u += a0[n] * s_attn[n][ec];
        u += a1[n] * s_extra[1024 + n * 16 + ec];
        u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];
      }
      s_extra[el] = u;
    }
    BAR();
    if (g == 0) {
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[48 + er][48 + m];
      float t = s_extra[el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[m * 16 + ec];
      s_extra[1024 + 3 * 256 + el] = t;
    }
    BAR();
    for (int j = 1; j < 4; j++)
      for (int b = 0; b < j; b++) {
        const int tl = j * (j - 1) / 2 + b;
        for (int e = tid; e < 256; e += NT)
          s_attn[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
              s_extra[1024 + tl * 256 + e];
      }
  }
  BAR();
  PH_ACC(V, 2, t2);
  PH_DECL(t3);

  if (tid < CH) s_attn[tid][tid] += 1.f;
  BAR();

  // ---- phase 3: vp = attn @ (v*beta); kcd = attn @ (k*beta*eg) ----
  if (V >= 2) {  // null: keep band staging + barriers + writeout, drop MACs
    const int d = tid & (DK - 1);
    const int BAND = 16;
    for (int j0 = 0; j0 < CH; j0 += BAND) {
      for (int i = tid; i < BAND * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + j0 + t < P)
                         ? qkv[(size_t)(t0 + j0 + t) * qkvstride + 4096 +
                               h * DK + dd]
                         : 0.f;
      }
      BAR();
      BAR();
    }
    int r = tid >> 7;
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
      wvp[r * DK + d] = 0.f;
      wkcd[r * DK + d] = 0.f;
    }
  } else if (V == 0) {
    const int d = tid & (DK - 1);
    float av[CH * DK / NT], ak[CH * DK / NT];
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++) av[n] = 0.f, ak[n] = 0.f;
    const int BAND = 16;
    for (int j0 = 0; j0 < CH; j0 += BAND) {
      for (int i = tid; i < BAND * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + j0 + t < P)
                         ? qkv[(size_t)(t0 + j0 + t) * qkvstride + 4096 +
                               h * DK + dd]
                         : 0.f;
      }
      BAR();
      for (int jj = 0; jj < BAND; jj++) {
        int j = j0 + jj;
        float vjb = s_extra[jj * DK + d] * s_beta[j];
        float kjb = u.kf[j][d] * s_beta[j] * s_eg[j];
        int r = tid >> 7;
#pragma unroll
        for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
          float a = s_attn[r][j];
          av[n] += a * vjb;
          ak[n] += a * kjb;
        }
      }
      BAR();
    }
    int r = tid >> 7;
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
      wvp[r * DK + d] = av[n];
      wkcd[r * DK + d] = ak[n];
    }
  } else {
    // attn fp16 (RNE) overlaid on s_extra (dead after solve): 9216 B <= 12 KB
    uint16_t(*s_a16)[LDS16] = (uint16_t(*)[LDS16])s_extra;
    for (int i = tid; i < CH * CH; i += NT)
      ((uint16_t*)s_a16)[(i / CH) * LDS16 + (i % CH)] =
          gp_f2h(s_attn[i / CH][i % CH]);
    // vbT pass: stage v*beta transposed fp16. Thread owns column d=tid&127
    // and row-pairs p = jg+8n (jg=tid>>7 in [0,8), n<4 -> p in [0,32));
    // one u32 store per pair. Global reads coalesce per v row.
    {
      const int d = tid & (DK - 1), jg = tid >> 7;
#pragma unroll
      for (int n = 0; n < 4; n++) {
        const int j = (jg + n * 8) * 2;
        float v0 = 0.f, v1 = 0.f;
        if (t0 + j < P)
          v0 = qkv[(size_t)(t0 + j) * qkvstride + 4096 + h * DK + d];
        if (t0 + j + 1 < P)
          v1 = qkv[(size_t)(t0 + j + 1) * qkvstride + 4096 + h * DK + d];
        uint32_t pv = (uint32_t)gp_f2h(v0 * s_beta[j]) |
                      ((uint32_t)gp_f2h(v1 * s_beta[j + 1]) << 16);
        *(uint32_t*)&u.f16.bT[d][j] = pv;
      }
    }
    BAR();
    // 32 output frags (4 rows x 8 cols); warp w owns (w/8, w%8); 4 k-steps.
    const int fr = (w >> 3) * 16, fc = (w & 7) * 16;
    gp_shortx16 a[4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
      a[kk] = gp_ld16(&s_a16[fr + gp_rowA(lane)][kk * 16]);
    {
      const uint16_t* bv = &u.f16.bT[fc + (lane & 15)][0];
      gp_floatx8 accv = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        gp_shortx16 b16 = gp_ld16(bv + kk * 16);
        accv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[kk], b16, accv);
      }
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        wvp[r * DK + col] = accv[e];
      }
    }
    BAR();  // vbT dead; restage as kbT from the resident k16
    {
      const int d = tid & (DK - 1), jg = tid >> 7;
#pragma unroll
      for (int n = 0; n < 4; n++) {
        const int j = (jg + n * 8) * 2;
        float k0 = __half2float(*(const __half*)&u.f16.k16[j][d]);
        float k1 = __half2float(*(const __half*)&u.f16.k16[j + 1][d]);
        uint32_t pk = (uint32_t)gp_f2h(k0 * s_beta[j] * s_eg[j]) |
                      ((uint32_t)gp_f2h(k1 * s_beta[j + 1] * s_eg[j + 1]) << 16);
        *(uint32_t*)&u.f16.bT[d][j] = pk;
      }
    }
    BAR();
    {
      const uint16_t* bk = &u.f16.bT[fc + (lane & 15)][0];
      gp_floatx8 acck = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        gp_shortx16 b16 = gp_ld16(bk + kk * 16);
        acck = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[kk], b16, acck);
      }
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        wkcd[r * DK + col] = acck[e];
      }
    }
  }
  BAR();
  PH_ACC(V, 3, t3);
  PH_DECL(t4);

  // ---- phase 4: attn2 = (q[r].k[j]) * exp(gcum[r]-gcum[j]), j<=r ----
  if (V >= 2) {  // null: keep quarter staging + barriers + writeout
    for (int q0 = 0; q0 < CH; q0 += 16) {
      for (int i = tid; i < 16 * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + q0 + t < P)
                         ? qkv[(size_t)(t0 + q0 + t) * qkvstride + kh * DK + dd]
                         : 0.f;
      }
      BAR();
      BAR();
    }
    for (int i = tid; i < CH * CH; i += NT)
      wsb[WS_ATTN2 + i] = 0.f;
  } else if (V == 0) {
    for (int q0 = 0; q0 < CH; q0 += 16) {
      for (int i = tid; i < 16 * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + q0 + t < P)
                         ? qkv[(size_t)(t0 + q0 + t) * qkvstride + kh * DK + dd]
                         : 0.f;
      }
      BAR();
      for (int i = tid; i < 16 * CH; i += NT) {
        int r = q0 + i / CH, j = i % CH;
        float a = 0.f;
        if (j <= r && t0 + r < P) {
          const float* qr = &s_extra[(i / CH) * DK];
          float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
          for (int dd = 0; dd < DK; dd += 4) {
            e0 += qr[dd] * u.kf[j][dd];
            e1 += qr[dd + 1] * u.kf[j][dd + 1];
            e2 += qr[dd + 2] * u.kf[j][dd + 2];
            e3 += qr[dd + 3] * u.kf[j][dd + 3];
          }
          float dot = (e0 + e1) + (e2 + e3);
          a = dot * expf(s_gcum[r] - s_gcum[j]);
        }
        s_attn[r][j] = a;
      }
      BAR();
    }
    for (int i = tid; i < CH * CH; i += NT)
      wsb[WS_ATTN2 + i] = ((const float*)s_attn)[i];
  } else {
    // stage q fp16 over bT (dead): full 64x128 tile in one pass
    uint16_t(*s_q16)[LDS16] = (uint16_t(*)[LDS16])u.f16.bT;
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, dd = i % DK;
      float v = (t0 + t < P)
                    ? qkv[(size_t)(t0 + t) * qkvstride + kh * DK + dd]
                    : 0.f;
      s_q16[t][dd] = gp_f2h(v);
    }
    BAR();
    if (w < 16) {
      const int fr = (w >> 2) * 16, fc = (w & 3) * 16;
      const uint16_t* arow = &s_q16[fr + gp_rowA(lane)][0];
      const uint16_t* brow = &u.f16.k16[fc + (lane & 15)][0];
      gp_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < DK / 16; kk++) {
        gp_shortx16 a = gp_ld16(arow + kk * 16);
        gp_shortx16 b = gp_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col <= r && t0 + r < P)
          a = acc[e] * expf(s_gcum[r] - s_gcum[col]);
        s_attn[r][col] = a;
      }
    }
    BAR();
    for (int i = tid; i < CH * CH; i += NT)
      wsb[WS_ATTN2 + i] = ((const float*)s_attn)[i];
  }
  PH_ACC(V, 4, t4);
  PH_ACC(V, 5, tk);
}

int main(int argc, char** argv) {
  const int P = argc > 1 ? atoi(argv[1]) : 8192;
  const int reps = argc > 2 ? atoi(argv[2]) : 10;
  const int qkvstride = 10240;
  const int nchunks = (P + CH - 1) / CH;

  float *d_qkv, *d_gb, *d_bb, *d_ws0, *d_ws1;
  CK(hipMalloc(&d_qkv, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_gb, (size_t)P * 48 * 4));
  CK(hipMalloc(&d_bb, (size_t)P * 48 * 4));
  CK(hipMalloc(&d_ws0, (size_t)nchunks * 48 * WS_FLOATS * 4));
  CK(hipMalloc(&d_ws1, (size_t)nchunks * 48 * WS_FLOATS * 4));

  // host-side data gen (deterministic LCG), realistic magnitudes
  {
    std::vector<float> hqkv((size_t)P * qkvstride, 0.f), hgb((size_t)P * 48),
        hbb((size_t)P * 48);
    unsigned long long s = 12345;
    auto rnd = [&]() {
      s = s * 2862933555777941757ULL + 3037000493ULL;
      return (float)(s >> 40) / (float)(1ull << 24);
    };
    for (int t = 0; t < P; t++) {
      for (int d = 0; d < 2048; d++) {  // q seg [0,2048) and k seg [2048,4096)
        // small magnitudes: k/q are layernorm outputs in the real model;
        // oversized synthetic k blows up the (I-A)^-1 solve (measured)
        hqkv[(size_t)t * qkvstride + d] = (rnd() - 0.5f) * 0.3f;
        hqkv[(size_t)t * qkvstride + 2048 + d] = (rnd() - 0.5f) * 0.3f;
      }
      for (int d = 0; d < 6144; d++)  // v seg [4096, 4096+6144)
        hqkv[(size_t)t * qkvstride + 4096 + d] = (rnd() - 0.5f) * 2.f;
      for (int hh = 0; hh < 48; hh++) {
        hgb[(size_t)t * 48 + hh] = -0.001f - rnd() * 0.09f;  // log-decay incr
        hbb[(size_t)t * 48 + hh] = 0.05f + rnd() * 0.85f;    // beta
      }
    }
    CK(hipMemcpy(d_qkv, hqkv.data(), hqkv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_gb, hgb.data(), hgb.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_bb, hbb.data(), hbb.size() * 4, hipMemcpyHostToDevice));
  }

  auto run = [&](int V, float* ws) {
    if (V == 0)
      k_gdn_proto<0><<<dim3(nchunks, 48), NT>>>(d_qkv, d_gb, d_bb, ws, P,
                                                qkvstride);
    else if (V == 1)
      k_gdn_proto<1><<<dim3(nchunks, 48), NT>>>(d_qkv, d_gb, d_bb, ws, P,
                                                qkvstride);
    else if (V == 2)
      k_gdn_proto<2><<<dim3(nchunks, 48), NT>>>(d_qkv, d_gb, d_bb, ws, P,
                                                qkvstride);
    else
      k_gdn_proto<3><<<dim3(nchunks, 48), NT>>>(d_qkv, d_gb, d_bb, ws, P,
                                                qkvstride);
  };
  const char* vn[4] = {"fp32", "fp16wmma", "nullc", "nullc_nosolve"};
  double wall[4] = {0, 0, 0, 0};
  for (int V = 0; V < 4; V++) {
    unsigned long long zero[8] = {};
    CK(hipMemcpyToSymbol(g_ph, zero, sizeof(zero), V * sizeof(zero)));
    for (int i = 0; i < 3; i++) run(V, V ? d_ws1 : d_ws0);
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < reps; i++) run(V, V ? d_ws1 : d_ws0);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    wall[V] = ms / reps;
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    unsigned long long hh[8];
    CK(hipMemcpyFromSymbol(hh, g_ph, sizeof(hh), V * sizeof(hh)));
    const char* nm[6] = {"staging", "p1_dots", "p2_solve", "p3_vpkcd",
                         "p4_attn2", "TOTAL"};
    printf("== %s: wall %.3f ms/iter (grid %dx48) ==\n", vn[V], wall[V],
           nchunks);
    for (int s2 = 0; s2 < 6; s2++)
      printf("  %-9s %12llu cyc\n", nm[s2], hh[s2]);
  }
  printf("wall: fp16/fp32 = %.2fx, nullc = %.2fx, nullc_nosolve = %.2fx\n",
         wall[1] / wall[0], wall[2] / wall[0], wall[3] / wall[0]);

  // ---- precision: vp / kcd / attn2 of fp16 variant vs fp32 variant ----
  {
    std::vector<float> a((size_t)nchunks * 48 * WS_FLOATS),
        b((size_t)nchunks * 48 * WS_FLOATS);
    CK(hipMemcpy(a.data(), d_ws0, a.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), d_ws1, b.size() * 4, hipMemcpyDeviceToHost));
    auto cmp = [&](int off, int n, const char* name) {
      double sa = 0, sb = 0, mx = 0, ref = 0;
      for (size_t i = 0; i < a.size(); i += WS_FLOATS) {
        for (int e = 0; e < n; e++) {
          double d = fabs((double)a[i + off + e] - (double)b[i + off + e]);
          sa += d * d;
          sb += (double)a[i + off + e] * a[i + off + e];
          double r = d / std::max(1.0, (double)fabs(a[i + off + e]));
          if (r > mx) mx = r;
          if (fabs(a[i + off + e]) > ref) ref = fabs(a[i + off + e]);
        }
      }
      printf("  %-6s fro-rel %.4f%%  maxrel %.4f%%  (max|ref| %.2f)\n", name,
             100 * sqrt(sa / std::max(1e-30, sb)), 100 * mx, ref);
    };
    printf("== precision fp16 vs fp32 (all chunk-heads) ==\n");
    cmp(WS_VP, CH * DK, "vp");
    cmp(WS_KCD, CH * DK, "kcd");
    cmp(WS_ATTN2, CH * CH, "attn2");
  }
  return 0;
}
