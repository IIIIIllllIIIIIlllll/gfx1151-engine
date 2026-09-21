// gdn_pipe_proto.cu — staging-overlap restructure prototypes for k_gdn_intra
// (src/gpu/gdec.cpp:5586), after the WMMA rejection (nullc ablation proved
// matrix compute is ~2% of wall; the kernel is staging-latency bound).
// Variants (same geometry: grid (nchunks,48), NT=1024, CH=64, DK=128):
//   V0 : fp32 verbatim reference (production code)
//   V10: route A — v/q read DIRECTLY from global in the compute loops (no
//        LDS band/quarter staging rounds, no staging barriers; s_k stays in
//        LDS). Same values + same accumulation order => bit-exact.
//   V11: route B — route A + k also direct-global (no s_k tile at all);
//        LDS = 28.8 KB -> 2 blocks/WGP co-residency. Bit-exact.
//   V12: route D — all three tiles resident fp16 (RNE), staged in ONE round
//        at phase 0; compute fp32 FMA off fp16 LDS. NOT bit-exact (second
//        tier), isolates the staging-overlap win from WMMA.
// Build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/gdn_pipe_proto \
//          tools/gdn_pipe_proto.cu && /tmp/gdn_pipe_proto [P=8192] [reps=10]
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

constexpr int CH = 64, DK = 128, NT = 1024;
constexpr int WS_ATTN2 = 64, WS_VP = WS_ATTN2 + 64 * 64,
              WS_KCD = WS_VP + 64 * 128, WS_FLOATS = WS_KCD + 64 * 128;

// per-variant LDS tile storage (an in-kernel union would reserve the max for
// every instantiation and kill route B's occupancy)
template <int V> struct TileShr;
template <> struct TileShr<0> { float kf[CH][DK + 4]; };
template <> struct TileShr<10> { float kf[CH][DK + 4]; };
template <> struct TileShr<11> { char none; };
template <> struct TileShr<12> {
  uint16_t k16[CH][DK];  // fp16 tiles, all resident after one staging round
  uint16_t vq[CH][DK];   // v first, re-staged as q after phase 3
};
// V13: fp16 residency + WMMA compute. k16 row stride 136 halfs (272B =
// 68 words, 8-lane ds_b128 phases -> banks 0,4,...,28); bT is the transposed
// staging (v*beta, then k*beta*eg, then q), stride 72 halfs.
template <> struct TileShr<13> {
  uint16_t k16[CH][136];
  uint16_t bT[DK][72];
};
template <> struct TileShr<14> {  // V12 tiles + cross-chunk reg pipeline
  uint16_t k16[CH][DK];
  uint16_t vq[CH][DK];
};

using gp_shortx16 = __attribute__((ext_vector_type(16))) short;
using gp_floatx8 = __attribute__((ext_vector_type(8))) float;

__device__ __forceinline__ gp_shortx16 gp_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  gp_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
// WMMA A-fragment row per lane (hardware lanes + conflict-free dummy rows)
__device__ __forceinline__ int gp_rowA(int lane) {
  const int base_ = lane >> 1;
  return (lane & 1) ? ((lane & 16) ? base_ : ((base_ + 4) & 7))
                    : ((lane & 16) ? 8 + ((base_ + 4) & 7) : base_);
}

__device__ __forceinline__ uint16_t pp_f2h(float f) {
  __half h = __float2half_rn(f);
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}
__device__ __forceinline__ float pp_h2f(uint16_t u) {
  __half h;
  memcpy(&h, &u, 2);
  return __half2float(h);
}

template <int V>
__global__ void __launch_bounds__(NT) k_gdn_pipe(
    const float* __restrict__ qkv, const float* __restrict__ gb,
    const float* __restrict__ bb, float* __restrict__ ws, int P, int qkvstride,
    int upto, int ncpb) {
  const int h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  const int nchunks_t = (P + CH - 1) / CH;
  float pfk[CH * DK / NT], pfv[CH * DK / NT], pfq[CH * DK / NT];  // V14 only
  for (int ic = 0; ic < ncpb; ic++) {
  const int c = blockIdx.x * ncpb + ic;
  if (c >= nchunks_t) break;
  __shared__ float s_attn[CH][CH];
  __shared__ float s_extra[3072];
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH];
  __shared__ TileShr<V> u;
  float* wsb = ws + ((size_t)c * 48 + h) * WS_FLOATS;
  float* wvp = wsb + WS_VP;
  float* wkcd = wsb + WS_KCD;
  const int t0 = c * CH;

  // ---- phase 0: gates + k tile ----
  if constexpr (V == 0 || V == 10) {
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      u.kf[t][d] = (t0 + t < P)
                       ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d]
                       : 0.f;
    }
  } else if constexpr (V == 12) {
    // single staging round: k16 + v16(*beta folded) + gates, one barrier.
    // q is re-staged over vq after phase 3 (v dead by then).
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      float kv = 0.f, vv = 0.f;
      if (t0 + t < P) {
        kv = qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d];
        vv = qkv[(size_t)(t0 + t) * qkvstride + 4096 + h * DK + d];
      }
      u.k16[t][d] = pp_f2h(kv);
      u.vq[t][d] = pp_f2h(vv);  // beta folded at use (fp32) instead
    }
  } else if constexpr (V == 14) {
    // V12 staging + cross-chunk register pipeline: chunk c+1's k/v/q load
    // into registers during chunk c's compute; k/v land in LDS at the next
    // iteration's phase 0, q at its phase 3 end (values identical to V12).
    const int pr = tid >> 7, pd = tid & (DK - 1);
    if (ic == 0) {
      for (int i = tid; i < CH * DK; i += NT) {
        int t = i / DK, d = i % DK;
        float kv = 0.f, vv = 0.f;
        if (t0 + t < P) {
          kv = qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d];
          vv = qkv[(size_t)(t0 + t) * qkvstride + 4096 + h * DK + d];
        }
        u.k16[t][d] = pp_f2h(kv);
        u.vq[t][d] = pp_f2h(vv);
      }
    } else {
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++) {
        u.k16[pr + n * (NT / DK)][pd] = pp_f2h(pfk[n]);
        u.vq[pr + n * (NT / DK)][pd] = pp_f2h(pfv[n]);
      }
    }
    const int cn = c + 1;
    if (ic + 1 < ncpb && cn < nchunks_t) {
      const int tn0 = cn * CH;
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++) {
        const int t = tn0 + pr + n * (NT / DK);
        const bool ok = t < P;
        pfk[n] = ok ? qkv[(size_t)t * qkvstride + 2048 + kh * DK + pd] : 0.f;
        pfv[n] = ok ? qkv[(size_t)t * qkvstride + 4096 + h * DK + pd] : 0.f;
      }
    }
  } else if constexpr (V == 13) {
    // k16 (stride 136) + vbT (transposed, beta folded — read beta from
    // global, same values s_beta stages below), one staging round.
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      float kv = (t0 + t < P)
                     ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d]
                     : 0.f;
      u.k16[t][d] = pp_f2h(kv);
    }
    const int d = tid & (DK - 1), jg = tid >> 7;
#pragma unroll
    for (int n = 0; n < 4; n++) {
      const int j = (jg + n * 8) * 2;
      float v0 = 0.f, v1 = 0.f, b0 = 0.f, b1 = 0.f;
      if (t0 + j < P) {
        v0 = qkv[(size_t)(t0 + j) * qkvstride + 4096 + h * DK + d];
        b0 = bb[(size_t)(t0 + j) * 48 + h];
      }
      if (t0 + j + 1 < P) {
        v1 = qkv[(size_t)(t0 + j + 1) * qkvstride + 4096 + h * DK + d];
        b1 = bb[(size_t)(t0 + j + 1) * 48 + h];
      }
      uint32_t pv =
          (uint32_t)pp_f2h(v0 * b0) | ((uint32_t)pp_f2h(v1 * b1) << 16);
      *(uint32_t*)&u.bT[d][j] = pv;
    }
  }
  if (tid < CH) {
    s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
    s_beta[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0.f;
    for (int i = 0; i < CH; i++) {
      acc += s_gcum[i];
      s_gcum[i] = acc;
    }
  }
  __syncthreads();
  if (tid < CH) {
    s_eg[tid] = expf(s_gcum[tid]);
    wsb[tid] = s_gcum[tid];
  }
  if (upto == 1) return;

  // ---- phase 1: attn = -beta[r]*(k[r].k[j])*exp(gcum[r]-gcum[j]), j<r ----
  if constexpr (V == 0 || V == 10) {
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
  } else if constexpr (V == 11) {
    const int kr0 = t0 < P ? t0 : 0;  // safe row for OOB (result masked)
    for (int i = tid; i < CH * CH; i += NT) {
      int r = i / CH, j = i % CH;
      float a = 0.f;
      if (j < r) {
        const int rr = (t0 + r < P) ? t0 + r : kr0;
        const int jj = (t0 + j < P) ? t0 + j : kr0;
        const float* pr = qkv + (size_t)rr * qkvstride + 2048 + kh * DK;
        const float* pj = qkv + (size_t)jj * qkvstride + 2048 + kh * DK;
        float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        for (int d = 0; d < DK; d += 4) {
          float4 xr = *(const float4*)(pr + d);
          float4 xj = *(const float4*)(pj + d);
          e0 += xr.x * xj.x;
          e1 += xr.y * xj.y;
          e2 += xr.z * xj.z;
          e3 += xr.w * xj.w;
        }
        float dot = (e0 + e1) + (e2 + e3);
        a = -s_beta[r] * dot * expf(s_gcum[r] - s_gcum[j]);
      }
      s_attn[r][j] = a;
    }
  } else if constexpr (V == 12 || V == 14) {  // fp16 tiles, fp32 compute
    for (int i = tid; i < CH * CH; i += NT) {
      int r = i / CH, j = i % CH;
      float a = 0.f;
      if (j < r) {
        float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        for (int d = 0; d < DK; d += 4) {
          e0 += pp_h2f(u.k16[r][d]) * pp_h2f(u.k16[j][d]);
          e1 += pp_h2f(u.k16[r][d + 1]) * pp_h2f(u.k16[j][d + 1]);
          e2 += pp_h2f(u.k16[r][d + 2]) * pp_h2f(u.k16[j][d + 2]);
          e3 += pp_h2f(u.k16[r][d + 3]) * pp_h2f(u.k16[j][d + 3]);
        }
        float dot = (e0 + e1) + (e2 + e3);
        a = -s_beta[r] * dot * expf(s_gcum[r] - s_gcum[j]);
      }
      s_attn[r][j] = a;
    }
  } else {  // V13: WMMA, 16 output frags (4x4), warp w owns (w/4, w%4)
    const int lane = tid & 31, w = tid >> 5;
    if (w < 16) {
      const int fr = (w >> 2) * 16, fc = (w & 3) * 16;
      const uint16_t* arow = &u.k16[fr + gp_rowA(lane)][0];
      const uint16_t* brow = &u.k16[fc + (lane & 15)][0];
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
  __syncthreads();
  if (upto == 2) return;

  // ---- phase 2: Ut5 solve (verbatim fp32 copy from gdec.cpp:5657) ----
  {
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
      __syncthreads();
      if (col < mg) s_attn[base + mg][base + col] = solved;
    }
    __syncthreads();
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
    __syncthreads();
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
    __syncthreads();
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
    __syncthreads();
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
    __syncthreads();
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
    __syncthreads();
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
    __syncthreads();
    for (int j = 1; j < 4; j++)
      for (int b = 0; b < j; b++) {
        const int tl = j * (j - 1) / 2 + b;
        for (int e = tid; e < 256; e += NT)
          s_attn[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
              s_extra[1024 + tl * 256 + e];
      }
  }
  __syncthreads();
  if (upto == 3) return;
  if (tid < CH) s_attn[tid][tid] += 1.f;
  __syncthreads();

  // ---- phase 3: vp = attn @ (v*beta); kcd = attn @ (k*beta*eg) ----
  if constexpr (V == 0) {
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
      __syncthreads();
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
      __syncthreads();
    }
    int r = tid >> 7;
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
      wvp[r * DK + d] = av[n];
      wkcd[r * DK + d] = ak[n];
    }
  } else if constexpr (V == 10 || V == 11) {
    // direct-global v (and k for V11) in the MAC loop; same j-ascending
    // order and same FMA grouping => bit-exact with V0.
    const int d = tid & (DK - 1);
    const int r0 = tid >> 7;
    float av[CH * DK / NT], ak[CH * DK / NT];
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++) av[n] = 0.f, ak[n] = 0.f;
    const float* vcol = qkv + 4096 + h * DK + d;
    const float* kcol = qkv + 2048 + kh * DK + d;
    for (int j = 0; j < CH; j++) {
      const bool okj = (t0 + j) < P;
      const float vj = okj ? vcol[(size_t)(t0 + j) * qkvstride] : 0.f;
      float kj;
      if constexpr (V == 10)
        kj = u.kf[j][d];
      else
        kj = okj ? kcol[(size_t)(t0 + j) * qkvstride] : 0.f;
      float vjb = vj * s_beta[j];
      float kjb = kj * s_beta[j] * s_eg[j];
      int r = r0;
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
        float a = s_attn[r][j];
        av[n] += a * vjb;
        ak[n] += a * kjb;
      }
    }
    int r = r0;
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
      wvp[r * DK + d] = av[n];
      wkcd[r * DK + d] = ak[n];
    }
  } else if constexpr (V == 12 || V == 14) {  // fp16 tiles (v16 has no beta folded; k from k16)
    const int d = tid & (DK - 1);
    const int r0 = tid >> 7;
    float av[CH * DK / NT], ak[CH * DK / NT];
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++) av[n] = 0.f, ak[n] = 0.f;
    for (int j = 0; j < CH; j++) {
      float vjb = pp_h2f(u.vq[j][d]) * s_beta[j];
      float kjb = pp_h2f(u.k16[j][d]) * s_beta[j] * s_eg[j];
      int r = r0;
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
        float a = s_attn[r][j];
        av[n] += a * vjb;
        ak[n] += a * kjb;
      }
    }
    int r = r0;
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
      wvp[r * DK + d] = av[n];
      wkcd[r * DK + d] = ak[n];
    }
    __syncthreads();  // vq dead; restage as q16 for phase 4
    if (V == 14 && ic > 0) {
      const int pr = tid >> 7, pd = tid & (DK - 1);
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++)
        u.vq[pr + n * (NT / DK)][pd] = pp_f2h(pfq[n]);
    } else {
      for (int i = tid; i < CH * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        float qv = (t0 + t < P) ? qkv[(size_t)(t0 + t) * qkvstride + kh * DK + dd]
                                : 0.f;
        u.vq[t][dd] = pp_f2h(qv);
      }
    }
    if constexpr (V == 14) {
      // q prefetch for chunk c+1 (issued here: pfq is consumed at the NEXT
      // iteration's restage, so an earlier issue would clobber chunk c's q)
      const int cn = c + 1;
      if (ic + 1 < ncpb && cn < nchunks_t) {
        const int tn0 = cn * CH;
        const int pr = tid >> 7, pd = tid & (DK - 1);
#pragma unroll
        for (int n = 0; n < CH * DK / NT; n++) {
          const int t = tn0 + pr + n * (NT / DK);
          pfq[n] = (t < P) ? qkv[(size_t)t * qkvstride + kh * DK + pd] : 0.f;
        }
      }
    }
  } else {  // V13: WMMA phase 3. attn fp16 (RNE) overlaid on s_extra (dead
    // after solve), stride 72; bT currently holds v*beta (transposed).
    uint16_t(*s_a16)[72] = (uint16_t(*)[72])s_extra;
    for (int i = tid; i < CH * CH; i += NT)
      ((uint16_t*)s_a16)[(i / CH) * 72 + (i % CH)] =
          pp_f2h(s_attn[i / CH][i % CH]);
    __syncthreads();
    const int lane = tid & 31, w = tid >> 5;
    const int fr = (w >> 3) * 16, fc = (w & 7) * 16;
    gp_shortx16 a[4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
      a[kk] = gp_ld16(&s_a16[fr + gp_rowA(lane)][kk * 16]);
    {
      const uint16_t* bv = &u.bT[fc + (lane & 15)][0];
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
    __syncthreads();  // vbT dead; restage as kbT from k16 (fold beta*eg)
    {
      const int d = tid & (DK - 1), jg = tid >> 7;
#pragma unroll
      for (int n = 0; n < 4; n++) {
        const int j = (jg + n * 8) * 2;
        float k0 = pp_h2f(u.k16[j][d]), k1 = pp_h2f(u.k16[j + 1][d]);
        uint32_t pk = (uint32_t)pp_f2h(k0 * s_beta[j] * s_eg[j]) |
                      ((uint32_t)pp_f2h(k1 * s_beta[j + 1] * s_eg[j + 1]) << 16);
        *(uint32_t*)&u.bT[d][j] = pk;
      }
    }
    __syncthreads();
    {
      const uint16_t* bk = &u.bT[fc + (lane & 15)][0];
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
    __syncthreads();  // bT dead; restage as q16 (no fold) for phase 4.
    // stride 136 like k16: rows are 128 halfs wide, a 72-stride layout
    // overlaps row t with row t+1 (two threads, same address -> race).
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, dd = i % DK;
      float qv = (t0 + t < P)
                     ? qkv[(size_t)(t0 + t) * qkvstride + kh * DK + dd]
                     : 0.f;
      ((uint16_t*)u.bT)[t * 136 + dd] = pp_f2h(qv);
    }
  }
  __syncthreads();
  if (upto == 4) return;

  // ---- phase 4: attn2 = (q[r].k[j]) * exp(gcum[r]-gcum[j]), j<=r ----
  if constexpr (V == 0) {
    for (int q0 = 0; q0 < CH; q0 += 16) {
      for (int i = tid; i < 16 * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + q0 + t < P)
                         ? qkv[(size_t)(t0 + q0 + t) * qkvstride + kh * DK + dd]
                         : 0.f;
      }
      __syncthreads();
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
      __syncthreads();
    }
    for (int i = tid; i < CH * CH; i += NT)
      wsb[WS_ATTN2 + i] = ((const float*)s_attn)[i];
  } else if constexpr (V == 13) {
    // q16 was staged into bT (stride 136) at the end of phase 3; B = k16 rows.
    const int lane = tid & 31, w = tid >> 5;
    const uint16_t(*q16)[136] = (const uint16_t(*)[136])u.bT;
    if (w < 16) {
      const int fr = (w >> 2) * 16, fc = (w & 3) * 16;
      const uint16_t* arow = &q16[fr + gp_rowA(lane)][0];
      const uint16_t* brow = &u.k16[fc + (lane & 15)][0];
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
    __syncthreads();
    for (int i = tid; i < CH * CH; i += NT)
      wsb[WS_ATTN2 + i] = ((const float*)s_attn)[i];
  } else {
    // V10/V11: direct-global q, k from LDS (V10) or global (V11).
    // V12: q16/k16 tiles. Same e0..e3 dot order => V10/V11 bit-exact.
    for (int E = tid; E < CH * CH; E += NT) {
      const int r = E / CH, j = E % CH;
      float a = 0.f;
      if (j <= r && t0 + r < P) {
        const float* qr = qkv + (size_t)(t0 + r) * qkvstride + kh * DK;
        float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        if constexpr (V == 10) {
          for (int dd = 0; dd < DK; dd += 4) {
            float4 q4 = *(const float4*)(qr + dd);
            e0 += q4.x * u.kf[j][dd];
            e1 += q4.y * u.kf[j][dd + 1];
            e2 += q4.z * u.kf[j][dd + 2];
            e3 += q4.w * u.kf[j][dd + 3];
          }
        } else if constexpr (V == 11) {
          const int jj = (t0 + j < P) ? t0 + j : t0;
          const float* kj = qkv + (size_t)jj * qkvstride + 2048 + kh * DK;
          for (int dd = 0; dd < DK; dd += 4) {
            float4 q4 = *(const float4*)(qr + dd);
            float4 k4 = *(const float4*)(kj + dd);
            e0 += q4.x * k4.x;
            e1 += q4.y * k4.y;
            e2 += q4.z * k4.z;
            e3 += q4.w * k4.w;
          }
        } else {
          for (int dd = 0; dd < DK; dd += 4) {
            e0 += pp_h2f(u.vq[r][dd]) * pp_h2f(u.k16[j][dd]);
            e1 += pp_h2f(u.vq[r][dd + 1]) * pp_h2f(u.k16[j][dd + 1]);
            e2 += pp_h2f(u.vq[r][dd + 2]) * pp_h2f(u.k16[j][dd + 2]);
            e3 += pp_h2f(u.vq[r][dd + 3]) * pp_h2f(u.k16[j][dd + 3]);
          }
        }
        float dot = (e0 + e1) + (e2 + e3);
        a = dot * expf(s_gcum[r] - s_gcum[j]);
      }
      s_attn[r][j] = a;
    }
    __syncthreads();
    for (int i = tid; i < CH * CH; i += NT)
      wsb[WS_ATTN2 + i] = ((const float*)s_attn)[i];
  }
  }  // ic chunk loop
}

int main(int argc, char** argv) {
  const int P = argc > 1 ? atoi(argv[1]) : 8192;
  const int reps = argc > 2 ? atoi(argv[2]) : 10;
  const int qkvstride = 10240;
  const int nchunks = (P + CH - 1) / CH;

  float *d_qkv, *d_gb, *d_bb;
  CK(hipMalloc(&d_qkv, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_gb, (size_t)P * 48 * 4));
  CK(hipMalloc(&d_bb, (size_t)P * 48 * 4));
  const int NV = 6;
  const int vs[NV] = {0, 10, 11, 12, 13, 14};
  float* d_ws[NV];
  for (int i = 0; i < NV; i++)
    CK(hipMalloc(&d_ws[i], (size_t)nchunks * 48 * WS_FLOATS * 4));
  {
    std::vector<float> hqkv((size_t)P * qkvstride, 0.f), hgb((size_t)P * 48),
        hbb((size_t)P * 48);
    unsigned long long s = 12345;
    auto rnd = [&]() {
      s = s * 2862933555777941757ULL + 3037000493ULL;
      return (float)(s >> 40) / (float)(1ull << 24);
    };
    for (int t = 0; t < P; t++) {
      for (int d = 0; d < 2048; d++) {
        hqkv[(size_t)t * qkvstride + d] = (rnd() - 0.5f) * 0.3f;
        hqkv[(size_t)t * qkvstride + 2048 + d] = (rnd() - 0.5f) * 0.3f;
      }
      for (int d = 0; d < 6144; d++)
        hqkv[(size_t)t * qkvstride + 4096 + d] = (rnd() - 0.5f) * 2.f;
      for (int hh = 0; hh < 48; hh++) {
        hgb[(size_t)t * 48 + hh] = -0.001f - rnd() * 0.09f;
        hbb[(size_t)t * 48 + hh] = 0.05f + rnd() * 0.85f;
      }
    }
    CK(hipMemcpy(d_qkv, hqkv.data(), hqkv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_gb, hgb.data(), hgb.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_bb, hbb.data(), hbb.size() * 4, hipMemcpyHostToDevice));
  }

  auto run = [&](int vi, int upto, int ncpb) {
    const int gx = (nchunks + ncpb - 1) / ncpb;
    switch (vs[vi]) {
      case 0: k_gdn_pipe<0><<<dim3(gx, 48), NT>>>(d_qkv, d_gb, d_bb, d_ws[vi], P, qkvstride, upto, ncpb); break;
      case 10: k_gdn_pipe<10><<<dim3(gx, 48), NT>>>(d_qkv, d_gb, d_bb, d_ws[vi], P, qkvstride, upto, ncpb); break;
      case 11: k_gdn_pipe<11><<<dim3(gx, 48), NT>>>(d_qkv, d_gb, d_bb, d_ws[vi], P, qkvstride, upto, ncpb); break;
      case 12: k_gdn_pipe<12><<<dim3(gx, 48), NT>>>(d_qkv, d_gb, d_bb, d_ws[vi], P, qkvstride, upto, ncpb); break;
      case 13: k_gdn_pipe<13><<<dim3(gx, 48), NT>>>(d_qkv, d_gb, d_bb, d_ws[vi], P, qkvstride, upto, ncpb); break;
      case 14: k_gdn_pipe<14><<<dim3(gx, 48), NT>>>(d_qkv, d_gb, d_bb, d_ws[vi], P, qkvstride, upto, ncpb); break;
    }
  };
  const char* vn[NV] = {"fp32_ref", "A_directvq", "B_2blk_nosk", "D_fp16res",
                        "E_fp16wmma", "G_fp16pref"};
  double wall[NV];
  const int ncpbs[6] = {1, 8, 16, 32, 64, 128};
  for (int vi = 0; vi < NV; vi++) {
    double gb = ((double)P * 10240 * 4 + (double)nchunks * 48 * WS_FLOATS * 4) / 1e9;
    for (int ci = 0; ci < 6; ci++) {
      const int ncpb = ncpbs[ci];
      // upto decomposition only at ncpb=1; other ncpb at full pipeline
      const int u0 = ncpb == 1 ? 1 : 5;
      for (int upto = u0; upto <= 5; upto++) {
        for (int i = 0; i < 3; i++) run(vi, upto, ncpb);
        CK(hipDeviceSynchronize());
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        CK(hipEventRecord(e0));
        for (int i = 0; i < reps; i++) run(vi, upto, ncpb);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms;
        CK(hipEventElapsedTime(&ms, e0, e1));
        double w = ms / reps;
        CK(hipEventDestroy(e0));
        CK(hipEventDestroy(e1));
        if (upto == 5 && ncpb == 1) wall[vi] = w;
        printf("== %-12s ncpb=%d upto=%d wall %7.3f ms/iter (eff BW %.0f GB/s, floor %.2f ms) ==\n",
               vn[vi], ncpb, upto, w, gb / (w / 1e3), gb / 224.0 * 1e3);
      }
    }
  }

  // correctness: V10/V11 must be BIT-EXACT vs V0; V12 ~1e-3 rel
  {
    std::vector<float> a((size_t)nchunks * 48 * WS_FLOATS), b(a.size());
    CK(hipMemcpy(a.data(), d_ws[0], a.size() * 4, hipMemcpyDeviceToHost));
    for (int vi = 1; vi < NV; vi++) {
      CK(hipMemcpy(b.data(), d_ws[vi], b.size() * 4, hipMemcpyDeviceToHost));
      double sa = 0, sb = 0, mx = 0;
      long ndiff = 0;
      for (size_t i = 0; i < a.size(); i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (a[i] != b[i]) ndiff++;
        sa += d * d;
        sb += (double)a[i] * a[i];
        double r = d / std::max(1.0, (double)fabs(a[i]));
        if (r > mx) mx = r;
      }
      printf("  %-12s vs ref: bitexact=%s (ndiff=%ld) fro-rel %.4f%% maxrel %.4f%%\n",
             vn[vi], ndiff == 0 ? "YES" : "no", ndiff,
             100 * sqrt(sa / std::max(1e-30, sb)), 100 * mx);
    }
    // E (V13) shares D's (V12) fp16 inputs, so their outputs must agree to
    // within accumulation-order noise — this isolates WMMA mapping bugs.
    {
      std::vector<float> a12(a.size());
      CK(hipMemcpy(a12.data(), d_ws[3], a12.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(b.data(), d_ws[4], b.size() * 4, hipMemcpyDeviceToHost));
      double sa = 0, sb = 0, mx = 0;
      long rg[4] = {0, 0, 0, 0};
      for (size_t i = 0; i < a12.size(); i++) {
        double d = fabs((double)a12[i] - (double)b[i]);
        if (a12[i] != b[i]) {
          const int o = i % WS_FLOATS;
          rg[o < WS_ATTN2 ? 0 : o < WS_VP ? 1 : o < WS_KCD ? 2 : 3]++;
        }
        sa += d * d;
        sb += (double)a12[i] * a12[i];
        double r = d / std::max(1.0, (double)fabs(a12[i]));
        if (r > mx) mx = r;
      }
      printf("  E_fp16wmma   vs D (same fp16 inputs): fro-rel %.4f%% maxrel %.4f%%\n",
             100 * sqrt(sa / std::max(1e-30, sb)), 100 * mx);
      printf("    diff elems by region: gcum=%ld attn2=%ld vp=%ld kcd=%ld\n",
             rg[0], rg[1], rg[2], rg[3]);
    }
    // G (V14) runs V12's exact arithmetic through a register pipeline, so it
    // must be BIT-EXACT vs V12 — any diff is a pipelining bug.
    {
      std::vector<float> a12(a.size());
      CK(hipMemcpy(a12.data(), d_ws[3], a12.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(b.data(), d_ws[5], b.size() * 4, hipMemcpyDeviceToHost));
      long ndiff = 0;
      long rg[4] = {0, 0, 0, 0};
      for (size_t i = 0; i < a12.size(); i++)
        if (a12[i] != b[i]) {
          ndiff++;
          const int o = i % WS_FLOATS;
          rg[o < WS_ATTN2 ? 0 : o < WS_VP ? 1 : o < WS_KCD ? 2 : 3]++;
        }
      printf("  G_fp16pref   vs D (same arithmetic): bitexact=%s (ndiff=%ld)\n",
             ndiff == 0 ? "YES" : "no", ndiff);
      printf("    by region: gcum=%ld attn2=%ld vp=%ld kcd=%ld\n", rg[0], rg[1],
             rg[2], rg[3]);
    }
  }
  return 0;
}