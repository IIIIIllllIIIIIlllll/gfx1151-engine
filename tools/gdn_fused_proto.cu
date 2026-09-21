// gdn_fused_proto.cu — fused GDN prefill kernel: one block per head (grid
// 48, NT=1024), serial chunk loop; intra phases (fp16 k/v tiles, WMMA attn,
// ut5 solve) + strip phases (v_new/out/S update) with NO workspace DRAM
// round trip. S is register-resident (16 fp32/thread, dc=tid>>3, j-slice
// (tid&7)*16; 8-lane shuffle reductions). vp/kcd are never materialized:
//   vn[r][d] = sum_l A[r][l]*bb_l*(v[l][d] - eg_l*sum_j k[l][j]*S[j][d])
// is algebraically vp - kcd@S (reassociation only). W = v - eg*(k.S) is
// register-resident fp16 (64 halfs/thread). vn is fp16 in the v16 tile
// (per-column time-disjoint). attn2 = fp32 FMA off global q (V10-style).
// Reference: k_intra_ref (fp32, pipe V0 verbatim) -> DRAM ws -> k_strip_ref
// (strip V0 verbatim) = production split numerics.
// Build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/gdn_fused_proto \
//          tools/gdn_fused_proto.cu && /tmp/gdn_fused_proto [P=8192] [reps=10]
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

constexpr int CH = 64, DK = 128;
constexpr int WS_ATTN2 = 64, WS_VP = WS_ATTN2 + 64 * 64,
              WS_KCD = WS_VP + 64 * 128, WS_FLOATS = WS_KCD + 64 * 128;

using fx_shortx16 = __attribute__((ext_vector_type(16))) short;
using fx_floatx8 = __attribute__((ext_vector_type(8))) float;
using fx_float4 = __attribute__((ext_vector_type(4))) float;

__device__ __forceinline__ uint16_t fx_f2h(float f) {
  return __half_as_ushort(__float2half_rn(f));
}
__device__ __forceinline__ float fx_h2f(uint16_t u) {
  return __half2float(__ushort_as_half(u));
}
__device__ __forceinline__ fx_shortx16 fx_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  fx_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
// WMMA A-fragment row per lane (hardware lanes + conflict-free dummy rows)
__device__ __forceinline__ int fx_rowA(int lane) {
  const int b = lane >> 1;
  return (lane & 1) ? ((lane & 16) ? b : ((b + 4) & 7))
                    : ((lane & 16) ? 8 + ((b + 4) & 7) : b);
}

// ============================ fused kernel =================================
// LDS ledger (budget 65536B):
//   s_v16 64x136h 17408 (v16 -> vn, per-column time-disjoint reuse)
//   s_k16 64x136h 17408
//   s_attn 64x64f 16384 (attn -> solve -> attn2 overwrite)
//   s_extra 3072f 12288 (ut5 U/T tiles)
//   gates 4x64f + glast 1028
//   total 64516B; S (16f/thread) and W (64h/thread) are register-resident.
__global__ void __launch_bounds__(1024, 1)
k_fused(const float* __restrict__ qkv, const float* __restrict__ gb,
        const float* __restrict__ bb, float* __restrict__ Sg,
        float* __restrict__ out, int P, int qkvstride, unsigned phmask = 63) {
  const int NT = 1024;
  const int h = blockIdx.x;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  __shared__ uint16_t s_v16[CH][144];  // v16, then aliased as vnT [128][72]
  __shared__ uint16_t s_k16[CH][136];
  __shared__ uint16_t s_attn16[CH][72];  // attn -> solve -> attn2, fp16
  __shared__ float s_extra[2816];
  __shared__ float s_gcum[CH], s_eg[CH], s_beta[CH], s_bb[CH];
  float* S = Sg + (size_t)h * DK * DK;
  const int lane = tid & 31, warp = tid >> 5;
  // S is register-resident as two persistent 16x16 WMMA C-fragments per
  // warp: tile t covers rows j in [m0+64t, +16), cols dc in [n0, +16).
  // Lane holds C(m,n) = S[m0+64t+(lane<16?e:8+e)][n0+lane&15], e = m%8.
  const int m0 = (warp >> 3) * 16, n0 = (warp & 7) * 16;
  const int dcS = n0 + (lane & 15);
  fx_floatx8 Sr0, Sr1;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    Sr0[e] = S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS];
    Sr1[e] = S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS];
  }
  const int nchunks = (P + CH - 1) / CH;
  for (int c = 0; c < nchunks; c++) {
    const int t0 = c * CH;
    // ---- P0: stage k16/v16 + gates + cumsum ----
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      float kv = 0.f, vv = 0.f;
      if (t0 + t < P) {
        kv = qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d];
        vv = qkv[(size_t)(t0 + t) * qkvstride + 4096 + h * DK + d];
      }
      s_k16[t][d] = fx_f2h(kv);
      s_v16[t][d] = fx_f2h(vv);
    }
    if (tid < CH) {
      s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
      s_bb[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
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
    const float glast = s_gcum[CH - 1];
    if (tid < CH) {
      s_eg[tid] = expf(s_gcum[tid]);
      s_beta[tid] = expf(glast - s_gcum[tid]);  // decay (strip semantics)
    }
    const float egl = expf(glast);
    // ---- P1: attn[r][j] = -bb[r]*(k[r].k[j])*exp(gcum[r]-gcum[j]), j<r ----
    if ((phmask & 1) && warp < 16) {
      const int fr = (warp >> 2) * 16, fc = (warp & 3) * 16;
      const uint16_t* arow = &s_k16[fr + fx_rowA(lane)][0];
      const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < DK / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col < r) a = -s_bb[r] * acc[e] * expf(s_gcum[r] - s_gcum[col]);
        s_attn16[r][col] = fx_f2h(a);
      }
    }
    __syncthreads();
    // ---- P2: ut5 solve, verbatim from k_gdn_intra ----
    if (phmask & 2) {
      const int g = tid >> 8, el = tid & 255;
      const int er = el >> 4, ec = el & 15;
      {
        const int col = er, mg = ec, base = g * 16;
        float solved = 0.f;
        for (int i = 1; i < 16; ++i) {
          float p = 0.f;
          if (mg < i) p = fx_h2f(s_attn16[base + i][base + mg]) * solved;
          float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
          float value = fx_h2f(s_attn16[base + i][base + col]);
#pragma unroll
          for (int gg = 0; gg < 16; gg += 2)
            value += __shfl_sync(~0ull, pair, gg, 16);
          if (i == mg && col < i) solved = value;
        }
        __syncthreads();
        if (col < mg) s_attn16[base + mg][base + col] = fx_f2h(solved);
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float a[16];
#pragma unroll
        for (int n = 0; n < 16; n++) a[n] = fx_h2f(s_attn16[j * 16 + er][b * 16 + n]);
        float u = a[ec];
#pragma unroll
        for (int n = 0; n < 16; n++)
          u += a[n] * fx_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = fx_h2f(s_attn16[j * 16 + er][j * 16 + m]);
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
          a0[n] = fx_h2f(s_attn16[j * 16 + er][b * 16 + n]);
          a1[n] = fx_h2f(s_attn16[j * 16 + er][(b + 1) * 16 + n]);
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * fx_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
          u += a1[n] * s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 + ec];
        }
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 2) {
        const int j = g + 2, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = fx_h2f(s_attn16[j * 16 + er][j * 16 + m]);
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
          a0[n] = fx_h2f(s_attn16[48 + er][n]);
          a1[n] = fx_h2f(s_attn16[48 + er][16 + n]);
          a2[n] = fx_h2f(s_attn16[48 + er][32 + n]);
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * fx_h2f(s_attn16[n][ec]);
          u += a1[n] * s_extra[1024 + n * 16 + ec];
          u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];
        }
        s_extra[el] = u;
      }
      __syncthreads();
      if (g == 0) {
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = fx_h2f(s_attn16[48 + er][48 + m]);
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
            s_attn16[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
                fx_f2h(s_extra[1024 + tl * 256 + e]);
        }
    }
    __syncthreads();
    if (tid < CH) s_attn16[tid][tid] = fx_f2h(fx_h2f(s_attn16[tid][tid]) + 1.f);
    __syncthreads();
    // ---- P3a: ks = k16 @ S16T and qS = q @ S16T (WMMA) ----
    // fp32 S stays register-resident; an fp16 transposed copy S16T[dc][j]
    // is staged into s_extra one j-quarter (32) at a time. Each warp owns
    // one 16x16 C tile of ks (m=l) and one of qS (m=r), accumulated across
    // the four quarters with a shared B-frag.
    float ksc[8], qsc[8], vnc[8];
    if (phmask & 4) {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const uint16_t* karow = &s_k16[r0 + fx_rowA(lane)][0];
      uint16_t* s16 = (uint16_t*)s_extra;  // [128 dc][32 j]
      fx_floatx8 kacc = {0, 0, 0, 0, 0, 0, 0, 0};
      fx_floatx8 qacc = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int qt = 0; qt < 4; qt++) {
        __syncthreads();  // previous quarter's WMMA done with s_extra
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const int mt = m0 + t * 64;
          if ((mt >> 5) == qt) {
            uint16_t tmp[8];
            const fx_floatx8& Srt = t ? Sr1 : Sr0;
#pragma unroll
            for (int e = 0; e < 8; e++) tmp[e] = fx_f2h(Srt[e]);
            *(uint4*)&s16[dcS * 32 + (mt & 31) + (lane < 16 ? 0 : 8)] =
                *(const uint4*)tmp;
          }
        }
        __syncthreads();
        const uint16_t* brow = s16 + (c0 + (lane & 15)) * 32;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          fx_shortx16 bk = fx_ld16(brow + kk * 16);
          fx_shortx16 ak = fx_ld16(karow + (qt * 2 + kk) * 16);
          kacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ak, bk, kacc);
          const int rq = t0 + r0 + fx_rowA(lane);
          const float* qp = qkv + (size_t)(rq < P ? rq : t0) * qkvstride +
                            kh * DK + (qt * 2 + kk) * 16;
          short qt16[16];
#pragma unroll
          for (int q4 = 0; q4 < 4; q4++) {
            float4 x = *(const float4*)(qp + q4 * 4);
            qt16[q4 * 4 + 0] = (short)fx_f2h(x.x);
            qt16[q4 * 4 + 1] = (short)fx_f2h(x.y);
            qt16[q4 * 4 + 2] = (short)fx_f2h(x.z);
            qt16[q4 * 4 + 3] = (short)fx_f2h(x.w);
          }
          fx_shortx16 aq;
          memcpy(&aq, qt16, 32);
          qacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, bk, qacc);
        }
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        ksc[e] = kacc[e];
        qsc[e] = qacc[e];
      }
    }
    __syncthreads();  // ksc complete; s_extra free
    // ---- P3b: WT[dc][l] = bb_l*(v[l][dc] - eg_l*ks[l][dc]) over the v16 tile
    if (phmask & 4) {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const int dc2 = c0 + (lane & 15);
      uint16_t vv[8];
#pragma unroll
      for (int e = 0; e < 8; e++)
        vv[e] = s_v16[r0 + (lane < 16 ? e : 8 + e)][dc2];
      __syncthreads();  // all v reads done before WT overwrites the tile
      uint16_t* WT = (uint16_t*)s_v16;  // [128 dc][72 l]
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int l = r0 + (lane < 16 ? e : 8 + e);
        WT[dc2 * 72 + l] =
            fx_f2h(s_bb[l] * (fx_h2f(vv[e]) - s_eg[l] * ksc[e]));
      }
    }
    __syncthreads();  // WT complete
    // ---- P3c: vnT[dc][r] = sum_l WT[dc][l] * attn16[r][l] (WMMA) ----
    if (phmask & 4) {
      const uint16_t* WT = (const uint16_t*)s_v16;
      const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;  // C[dc][r]
      const uint16_t* arow = WT + (size_t)(m0 + fx_rowA(lane)) * 72;
      const uint16_t* brow = &s_attn16[n0 + (lane & 15)][0];
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < CH / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) vnc[e] = acc[e];
    }
    __syncthreads();  // WT reads done before vnT overwrites the tile
    if (phmask & 4) {
      uint16_t* vnT = (uint16_t*)s_v16;  // [128 dc][72 r]
      const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;
      const int rw = n0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int dcw = m0 + (lane < 16 ? e : 8 + e);
        vnT[dcw * 72 + rw] = fx_f2h(vnc[e]);
      }
    }
    __syncthreads();  // vnT ready; s_attn16 reusable
    // ---- P4: attn2[r][j] = (q[r].k[j]) * exp(gcum[r]-gcum[j]), j<=r ----
    // WMMA: A-frags = q rows straight from global (f2h), B = k16 tile.
    if (phmask & 8) {
      if (warp < 16) {
        const int fr = (warp >> 2) * 16, fc = (warp & 3) * 16;
        const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
        fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < DK / 16; kk++) {
          const int rq = t0 + fr + fx_rowA(lane);
          const float* qp =
              qkv + (size_t)(rq < P ? rq : t0) * qkvstride + kh * DK + kk * 16;
          short qt[16];
#pragma unroll
          for (int q4 = 0; q4 < 4; q4++) {
            float4 x = *(const float4*)(qp + q4 * 4);
            qt[q4 * 4 + 0] = (short)fx_f2h(x.x);
            qt[q4 * 4 + 1] = (short)fx_f2h(x.y);
            qt[q4 * 4 + 2] = (short)fx_f2h(x.z);
            qt[q4 * 4 + 3] = (short)fx_f2h(x.w);
          }
          fx_shortx16 a;
          memcpy(&a, qt, 32);
          fx_shortx16 b = fx_ld16(brow + kk * 16);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
        const int col = fc + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = fr + (lane < 16 ? e : 8 + e);
          float a = 0.f;
          if (col <= r && t0 + r < P)
            a = acc[e] * expf(s_gcum[r] - s_gcum[col]);
          s_attn16[r][col] = fx_f2h(a);
        }
      }
    }
    __syncthreads();
    // ---- S2: out[r][dc] = eg_r*qS[r][dc] + (attn2 @ vnT)[r][dc] ----
    // o2 via WMMA (32 warps, one 16x16 tile each; B-frags = vnT rows, contig
    // r); qS C-frags from P3a share the same tile layout -> single write.
    if (phmask & 16) {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const uint16_t* vnT = (const uint16_t*)s_v16;
      const uint16_t* arow = &s_attn16[r0 + fx_rowA(lane)][0];
      const uint16_t* brow = vnT + (size_t)(c0 + (lane & 15)) * 72;
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < CH / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int dc2 = c0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = r0 + (lane < 16 ? e : 8 + e);
        if (t0 + r < P)
          out[(size_t)(t0 + r) * qkvstride + 4096 + h * DK + dc2] =
              s_eg[r] * qsc[e] + acc[e];
      }
    }
    __syncthreads();  // out complete; vnT still live for S3
    // ---- S3: S[j][dc] = S*egl + sum_r kTdecay[j][r] * vnT[dc][r] ----
    // WMMA accumulate straight into the persistent S C-fragments. A = kT
    // (transposed k16, decay folded in) staged one r-quarter (16) at a time
    // into s_extra; B = vnT rows (contiguous r).
    if (phmask & 32) {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        Sr0[e] *= egl;
        Sr1[e] *= egl;
      }
      uint16_t* ktd = (uint16_t*)s_extra;  // [128 j][16 r]
      const uint16_t* vnT = (const uint16_t*)s_v16;
      for (int qt = 0; qt < 4; qt++) {
        __syncthreads();
        for (int i = tid; i < DK * 16; i += NT) {
          const int j = i >> 4, r = i & 15;
          ktd[j * 16 + r] = fx_f2h(fx_h2f(s_k16[qt * 16 + r][j]) *
                                   s_beta[qt * 16 + r]);
        }
        __syncthreads();
        const uint16_t* arow0 = ktd + (m0 + fx_rowA(lane)) * 16;
        const uint16_t* arow1 = ktd + (m0 + 64 + fx_rowA(lane)) * 16;
        fx_shortx16 b =
            fx_ld16(vnT + (size_t)(n0 + (lane & 15)) * 72 + qt * 16);
        Sr0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow0), b,
                                                         Sr0);
        Sr1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow1), b,
                                                         Sr1);
      }
    }
    __syncthreads();  // k16/vn reads done before next chunk's staging
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr0[e];
    S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr1[e];
  }
}

// ============================ fused2 (Tb attack) ===========================
// Same algorithm as k_fused, two numerics-identical reworks gated by feat:
//   feat&1: P0 staging vectorized (float4 global loads + uint2 LDS stores).
//   feat&2: P4 (attn2) merged into P3a. The q A-fragments P3a already loads
//   from global are reused with k16 B-fragments to accumulate attn2 in the
//   same kk order (0..7) as standalone P4 -> bitwise identical. Result is
//   scaled/masked and written as fp16 [64][72] into s_extra (S16T is dead by
//   then, 9216B <= 11264B); S2 reads its A-frags from there. s_attn16 keeps
//   the solved attn for P3c and is never overwritten. s_extra is free for
//   S3's ktd again after the post-S2 barrier.
__global__ void __launch_bounds__(1024, 1)
k_fused2(const float* __restrict__ qkv, const float* __restrict__ gb,
         const float* __restrict__ bb, float* __restrict__ Sg,
         float* __restrict__ out, int P, int qkvstride, unsigned feat,
         unsigned phmask = 63) {
  const int NT = 1024;
  const int h = blockIdx.x;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  __shared__ uint16_t s_v16[CH][144];  // v16, then aliased as vnT [128][72]
  __shared__ uint16_t s_k16[CH][136];
  __shared__ __align__(16) uint16_t s_attn16[CH][72];  // attn -> solve, fp16
  __shared__ float s_extra[2816];
  __shared__ float s_gcum[CH], s_eg[CH], s_beta[CH], s_bb[CH];
  float* S = Sg + (size_t)h * DK * DK;
  const int lane = tid & 31, warp = tid >> 5;
  const int m0 = (warp >> 3) * 16, n0 = (warp & 7) * 16;
  const int dcS = n0 + (lane & 15);
  fx_floatx8 Sr0, Sr1;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    Sr0[e] = S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS];
    Sr1[e] = S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS];
  }
  const int nchunks = (P + CH - 1) / CH;
  for (int c = 0; c < nchunks; c++) {
    const int t0 = c * CH;
    // ---- P0: stage k16/v16 + gates + cumsum ----
    if (feat & 32) {
      // warps 1..31 stage k16/v16 while warp 0 does gates + serial cumsum in
      // the shadow of the staging loop; one barrier total. Same values, same
      // serial cumsum order as the baseline.
      if (warp > 0) {
        for (int i = tid - 32; i < CH * DK / 4; i += NT - 32) {
          const int t = i >> 5, d4 = (i & 31) << 2;
          float4 k4 = {0.f, 0.f, 0.f, 0.f}, v4 = {0.f, 0.f, 0.f, 0.f};
          if (t0 + t < P) {
            k4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 2048 +
                                  kh * DK + d4);
            v4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 4096 +
                                  h * DK + d4);
          }
          uint16_t kh4[4] = {fx_f2h(k4.x), fx_f2h(k4.y), fx_f2h(k4.z),
                             fx_f2h(k4.w)};
          uint16_t vh4[4] = {fx_f2h(v4.x), fx_f2h(v4.y), fx_f2h(v4.z),
                             fx_f2h(v4.w)};
          *(uint2*)&s_k16[t][d4] = *(const uint2*)kh4;
          *(uint2*)&s_v16[t][d4] = *(const uint2*)vh4;
        }
      } else {
        s_gcum[lane] = (t0 + lane < P) ? gb[(size_t)(t0 + lane) * 48 + h] : 0.f;
        s_gcum[32 + lane] =
            (t0 + 32 + lane < P) ? gb[(size_t)(t0 + 32 + lane) * 48 + h] : 0.f;
        s_bb[lane] = (t0 + lane < P) ? bb[(size_t)(t0 + lane) * 48 + h] : 0.f;
        s_bb[32 + lane] =
            (t0 + 32 + lane < P) ? bb[(size_t)(t0 + 32 + lane) * 48 + h] : 0.f;
        __syncwarp();
        if (lane == 0) {
          float acc = 0.f;
          for (int i = 0; i < CH; i++) {
            acc += s_gcum[i];
            s_gcum[i] = acc;
          }
        }
      }
      __syncthreads();
    } else if (feat & 1) {
      for (int i = tid; i < CH * DK / 4; i += NT) {
        const int t = i >> 5, d4 = (i & 31) << 2;
        float4 k4 = {0.f, 0.f, 0.f, 0.f}, v4 = {0.f, 0.f, 0.f, 0.f};
        if (t0 + t < P) {
          k4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 2048 +
                                kh * DK + d4);
          const float4* vp = (const float4*)(qkv + (size_t)(t0 + t) * qkvstride +
                                             4096 + h * DK + d4);
          if (feat & 8) {
            fx_float4 nt = __builtin_nontemporal_load((const fx_float4*)vp);
            v4.x = nt[0]; v4.y = nt[1]; v4.z = nt[2]; v4.w = nt[3];
          } else {
            v4 = *vp;
          }
        }
        uint16_t kh4[4] = {fx_f2h(k4.x), fx_f2h(k4.y), fx_f2h(k4.z),
                           fx_f2h(k4.w)};
        uint16_t vh4[4] = {fx_f2h(v4.x), fx_f2h(v4.y), fx_f2h(v4.z),
                           fx_f2h(v4.w)};
        *(uint2*)&s_k16[t][d4] = *(const uint2*)kh4;
        *(uint2*)&s_v16[t][d4] = *(const uint2*)vh4;
      }
    } else {
      for (int i = tid; i < CH * DK; i += NT) {
        int t = i / DK, d = i % DK;
        float kv = 0.f, vv = 0.f;
        if (t0 + t < P) {
          kv = qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d];
          vv = qkv[(size_t)(t0 + t) * qkvstride + 4096 + h * DK + d];
        }
        s_k16[t][d] = fx_f2h(kv);
        s_v16[t][d] = fx_f2h(vv);
      }
    }
    if (!(feat & 32)) {
      if (tid < CH) {
        s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
        s_bb[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
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
    }
    const float glast = s_gcum[CH - 1];
    if (tid < CH) {
      s_eg[tid] = expf(s_gcum[tid]);
      s_beta[tid] = expf(glast - s_gcum[tid]);  // decay (strip semantics)
    }
    const float egl = expf(glast);
    // ---- P1: attn[r][j] = -bb[r]*(k[r].k[j])*exp(gcum[r]-gcum[j]), j<r ----
    if ((phmask & 1) && warp < 16) {
      const int fr = (warp >> 2) * 16, fc = (warp & 3) * 16;
      const uint16_t* arow = &s_k16[fr + fx_rowA(lane)][0];
      const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < DK / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col < r) a = -s_bb[r] * acc[e] * expf(s_gcum[r] - s_gcum[col]);
        s_attn16[r][col] = fx_f2h(a);
      }
    }
    __syncthreads();
    // ---- P2: ut5 solve, verbatim from k_gdn_intra ----
    if (phmask & 2) {
      const int g = tid >> 8, el = tid & 255;
      const int er = el >> 4, ec = el & 15;
      {
        const int col = er, mg = ec, base = g * 16;
        float solved = 0.f;
        for (int i = 1; i < 16; ++i) {
          float p = 0.f;
          if (mg < i) p = fx_h2f(s_attn16[base + i][base + mg]) * solved;
          float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
          float value = fx_h2f(s_attn16[base + i][base + col]);
#pragma unroll
          for (int gg = 0; gg < 16; gg += 2)
            value += __shfl_sync(~0ull, pair, gg, 16);
          if (i == mg && col < i) solved = value;
        }
        __syncthreads();
        if (col < mg) s_attn16[base + mg][base + col] = fx_f2h(solved);
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float a[16];
#pragma unroll
        for (int n = 0; n < 16; n++) a[n] = fx_h2f(s_attn16[j * 16 + er][b * 16 + n]);
        float u = a[ec];
#pragma unroll
        for (int n = 0; n < 16; n++)
          u += a[n] * fx_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = fx_h2f(s_attn16[j * 16 + er][j * 16 + m]);
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
          a0[n] = fx_h2f(s_attn16[j * 16 + er][b * 16 + n]);
          a1[n] = fx_h2f(s_attn16[j * 16 + er][(b + 1) * 16 + n]);
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * fx_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
          u += a1[n] * s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 + ec];
        }
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 2) {
        const int j = g + 2, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = fx_h2f(s_attn16[j * 16 + er][j * 16 + m]);
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
          a0[n] = fx_h2f(s_attn16[48 + er][n]);
          a1[n] = fx_h2f(s_attn16[48 + er][16 + n]);
          a2[n] = fx_h2f(s_attn16[48 + er][32 + n]);
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * fx_h2f(s_attn16[n][ec]);
          u += a1[n] * s_extra[1024 + n * 16 + ec];
          u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];
        }
        s_extra[el] = u;
      }
      __syncthreads();
      if (g == 0) {
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = fx_h2f(s_attn16[48 + er][48 + m]);
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
            s_attn16[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
                fx_f2h(s_extra[1024 + tl * 256 + e]);
        }
    }
    __syncthreads();
    if (tid < CH) s_attn16[tid][tid] = fx_f2h(fx_h2f(s_attn16[tid][tid]) + 1.f);
    __syncthreads();
    // ---- P3a: ks = k16 @ S16T and qS = q @ S16T (WMMA) ----
    // feat&2: warps with (warp&7)<4 additionally accumulate attn2 (tile
    // r0 x (warp&3)*16) reusing the same q A-frags with k16 B-frags, same
    // kk order as standalone P4.
    float ksc[8], qsc[8], vnc[8];
    const bool doA2 = (feat & 2) && ((warp & 7) < 4);
    fx_floatx8 a2acc = {0, 0, 0, 0, 0, 0, 0, 0};
    if (phmask & 4) {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const uint16_t* karow = &s_k16[r0 + fx_rowA(lane)][0];
      uint16_t* s16 = (uint16_t*)s_extra;  // [128 dc][32 j]
      const int j0 = (warp & 3) * 16;
      const uint16_t* kbrow = &s_k16[j0 + (lane & 15)][0];
      fx_floatx8 kacc = {0, 0, 0, 0, 0, 0, 0, 0};
      fx_floatx8 qacc = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int qt = 0; qt < 4; qt++) {
        __syncthreads();  // previous quarter's WMMA done with s_extra
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const int mt = m0 + t * 64;
          if ((mt >> 5) == qt) {
            uint16_t tmp[8];
            const fx_floatx8& Srt = t ? Sr1 : Sr0;
#pragma unroll
            for (int e = 0; e < 8; e++) tmp[e] = fx_f2h(Srt[e]);
            *(uint4*)&s16[dcS * 32 + (mt & 31) + (lane < 16 ? 0 : 8)] =
                *(const uint4*)tmp;
          }
        }
        __syncthreads();
        const uint16_t* brow = s16 + (c0 + (lane & 15)) * 32;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          fx_shortx16 bk = fx_ld16(brow + kk * 16);
          fx_shortx16 ak = fx_ld16(karow + (qt * 2 + kk) * 16);
          kacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ak, bk, kacc);
          const int rq = t0 + r0 + fx_rowA(lane);
          const float* qp = qkv + (size_t)(rq < P ? rq : t0) * qkvstride +
                            kh * DK + (qt * 2 + kk) * 16;
          short qt16[16];
#pragma unroll
          for (int q4 = 0; q4 < 4; q4++) {
            float4 x = *(const float4*)(qp + q4 * 4);
            qt16[q4 * 4 + 0] = (short)fx_f2h(x.x);
            qt16[q4 * 4 + 1] = (short)fx_f2h(x.y);
            qt16[q4 * 4 + 2] = (short)fx_f2h(x.z);
            qt16[q4 * 4 + 3] = (short)fx_f2h(x.w);
          }
          fx_shortx16 aq;
          memcpy(&aq, qt16, 32);
          qacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, bk, qacc);
          if (doA2) {
            fx_shortx16 bk2 = fx_ld16(kbrow + (qt * 2 + kk) * 16);
            a2acc =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, bk2, a2acc);
          }
        }
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        ksc[e] = kacc[e];
        qsc[e] = qacc[e];
      }
    }
    __syncthreads();  // S16T B-frag reads done; ksc/qsc complete
    // attn2 write goes straight after this barrier; the P3b/P3c barriers
    // below order it before S2 reads it, and S3's first staging barrier
    // orders it before s_extra is reused as ktd.
    if ((phmask & 4) && doA2) {
      const int r0 = (warp >> 3) * 16;
      const int j0 = (warp & 3) * 16;
      uint16_t* a2w = (uint16_t*)s_extra;  // [64 r][72 j]
      const int col = j0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = r0 + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col <= r && t0 + r < P)
          a = a2acc[e] * expf(s_gcum[r] - s_gcum[col]);
        a2w[r * 72 + col] = fx_f2h(a);
      }
    }
    // ---- P3b: WT[dc][l] = bb_l*(v[l][dc] - eg_l*ks[l][dc]) over the v16 tile
    if (phmask & 4) {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const int dc2 = c0 + (lane & 15);
      uint16_t vv[8];
#pragma unroll
      for (int e = 0; e < 8; e++)
        vv[e] = s_v16[r0 + (lane < 16 ? e : 8 + e)][dc2];
      __syncthreads();  // all v reads done before WT overwrites the tile
      uint16_t* WT = (uint16_t*)s_v16;  // [128 dc][72 l]
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int l = r0 + (lane < 16 ? e : 8 + e);
        WT[dc2 * 72 + l] =
            fx_f2h(s_bb[l] * (fx_h2f(vv[e]) - s_eg[l] * ksc[e]));
      }
    }
    __syncthreads();  // WT complete
    // ---- P3c: vnT[dc][r] = sum_l WT[dc][l] * attn16[r][l] (WMMA) ----
    if (phmask & 4) {
      const uint16_t* WT = (const uint16_t*)s_v16;
      const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;  // C[dc][r]
      const uint16_t* arow = WT + (size_t)(m0 + fx_rowA(lane)) * 72;
      const uint16_t* brow = &s_attn16[n0 + (lane & 15)][0];
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < CH / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) vnc[e] = acc[e];
    }
    __syncthreads();  // WT reads done before vnT overwrites the tile
    if (phmask & 4) {
      uint16_t* vnT = (uint16_t*)s_v16;  // [128 dc][72 r]
      const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;
      const int rw = n0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int dcw = m0 + (lane < 16 ? e : 8 + e);
        vnT[dcw * 72 + rw] = fx_f2h(vnc[e]);
      }
    }
    __syncthreads();  // vnT ready
    // ---- P4: attn2 standalone (skipped when merged into P3a) ----
    if ((phmask & 8) && !(feat & 2)) {
      if (warp < 16) {
        const int fr = (warp >> 2) * 16, fc = (warp & 3) * 16;
        const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
        fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < DK / 16; kk++) {
          const int rq = t0 + fr + fx_rowA(lane);
          const float* qp =
              qkv + (size_t)(rq < P ? rq : t0) * qkvstride + kh * DK + kk * 16;
          short qt[16];
#pragma unroll
          for (int q4 = 0; q4 < 4; q4++) {
            float4 x = *(const float4*)(qp + q4 * 4);
            qt[q4 * 4 + 0] = (short)fx_f2h(x.x);
            qt[q4 * 4 + 1] = (short)fx_f2h(x.y);
            qt[q4 * 4 + 2] = (short)fx_f2h(x.z);
            qt[q4 * 4 + 3] = (short)fx_f2h(x.w);
          }
          fx_shortx16 a;
          memcpy(&a, qt, 32);
          fx_shortx16 b = fx_ld16(brow + kk * 16);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
        const int col = fc + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = fr + (lane < 16 ? e : 8 + e);
          float a = 0.f;
          if (col <= r && t0 + r < P)
            a = acc[e] * expf(s_gcum[r] - s_gcum[col]);
          s_attn16[r][col] = fx_f2h(a);
        }
      }
      __syncthreads();
    }
    // ---- S2: out[r][dc] = eg_r*qS[r][dc] + (attn2 @ vnT)[r][dc] ----
    if (phmask & 16) {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const uint16_t* vnT = (const uint16_t*)s_v16;
      const uint16_t* a2r = (const uint16_t*)s_extra;  // [64][72] when feat&2
      const uint16_t* arow =
          (feat & 2) ? (a2r + (size_t)(r0 + fx_rowA(lane)) * 72)
                     : &s_attn16[r0 + fx_rowA(lane)][0];
      const uint16_t* brow = vnT + (size_t)(c0 + (lane & 15)) * 72;
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < CH / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int dc2 = c0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = r0 + (lane < 16 ? e : 8 + e);
        if (t0 + r < P) {
          float* op = out + (size_t)(t0 + r) * qkvstride + 4096 + h * DK + dc2;
          const float ov = s_eg[r] * qsc[e] + acc[e];
          if (feat & 16)
            __builtin_nontemporal_store(ov, op);
          else
            *op = ov;
        }
      }
    }
    __syncthreads();  // out complete; vnT still live for S3
    // ---- S3: S[j][dc] = S*egl + sum_r kTdecay[j][r] * vnT[dc][r] ----
    if (phmask & 32) {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        Sr0[e] *= egl;
        Sr1[e] *= egl;
      }
      uint16_t* ktd = (uint16_t*)s_extra;  // two [128 j][16 r] buffers
      const uint16_t* vnT = (const uint16_t*)s_v16;
      if (!(feat & 4)) {  // original: one quarter staged per round
        for (int qt = 0; qt < 4; qt++) {
          __syncthreads();
          for (int i = tid; i < DK * 16; i += NT) {
            const int j = i >> 4, r = i & 15;
            ktd[j * 16 + r] = fx_f2h(fx_h2f(s_k16[qt * 16 + r][j]) *
                                     s_beta[qt * 16 + r]);
          }
          __syncthreads();
          const uint16_t* arow0 = ktd + (m0 + fx_rowA(lane)) * 16;
          const uint16_t* arow1 = ktd + (m0 + 64 + fx_rowA(lane)) * 16;
          fx_shortx16 b =
              fx_ld16(vnT + (size_t)(n0 + (lane & 15)) * 72 + qt * 16);
          Sr0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow0), b,
                                                           Sr0);
          Sr1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow1), b,
                                                           Sr1);
        }
      } else {
      // stage(qt+1) overlaps WMMA(qt) on the other buffer; one barrier per
      // quarter instead of two.
      for (int i = tid; i < DK * 16; i += NT) {
        const int j = i >> 4, r = i & 15;
        ktd[j * 16 + r] =
            fx_f2h(fx_h2f(s_k16[r][j]) * s_beta[r]);
      }
      __syncthreads();
      for (int qt = 0; qt < 4; qt++) {
        if (qt + 1 < 4) {
          uint16_t* nb = ktd + ((qt + 1) & 1) * DK * 16;
          for (int i = tid; i < DK * 16; i += NT) {
            const int j = i >> 4, r = i & 15;
            nb[j * 16 + r] = fx_f2h(fx_h2f(s_k16[(qt + 1) * 16 + r][j]) *
                                    s_beta[(qt + 1) * 16 + r]);
          }
        }
        const uint16_t* buf = ktd + (qt & 1) * DK * 16;
        const uint16_t* arow0 = buf + (m0 + fx_rowA(lane)) * 16;
        const uint16_t* arow1 = buf + (m0 + 64 + fx_rowA(lane)) * 16;
        fx_shortx16 b =
            fx_ld16(vnT + (size_t)(n0 + (lane & 15)) * 72 + qt * 16);
        Sr0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow0), b,
                                                         Sr0);
        Sr1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow1), b,
                                                         Sr1);
        if (qt + 1 < 4) __syncthreads();
      }
      }
    }
    __syncthreads();  // k16/vn reads done before next chunk's staging
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr0[e];
    S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr1[e];
  }
}

// ================= fused3: chunk-half pipeline (grid 96) ===================
// block = (half, h) = (blockIdx.x/48, blockIdx.x%48); n2 = ceil(nchunks/2).
// half0 runs the full k_fused2(FEAT=7) pipeline over chunks [0,n2), writes S
// back and releases flag[h]. half1 pass 1 computes the S-independent intra
// phases (P0 k-only staging, P1, P2, standalone P4 attn2) for chunks
// [n2,nchunks) into DRAM ws3, spins on the flag (acquire), loads S, then
// pass 2 runs the strip phases per chunk (restage k16/v16, P3a without
// doA2, P3b, P3c, S2, S3 double-buffered). The fp16/fp32 ws round trip is
// exact and every WMMA keeps k_fused2's accumulation order, so out/S should
// be bitwise identical to k_fused2 FEAT=7. half0-first block ordering makes
// the spin a no-op in the common schedule.
constexpr int WS3_A2 = CH * 72, WS3_F = WS3_A2 + CH * 72,
              WS3_U16 = WS3_F + 2 * (3 * CH + 4);
__global__ void __launch_bounds__(1024, 1)
k_fused3(const float* __restrict__ qkv, const float* __restrict__ gb,
         const float* __restrict__ bb, float* __restrict__ Sg,
         float* __restrict__ out, int P, int qkvstride,
         uint16_t* __restrict__ ws3, int* __restrict__ flags,
         unsigned f3mask = 3) {
  const int NT = 1024;
  const int half = blockIdx.x / 48, h = blockIdx.x % 48;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  __shared__ uint16_t s_v16[CH][144];  // v16, then aliased as vnT [128][72]
  __shared__ uint16_t s_k16[CH][136];
  __shared__ __align__(16) uint16_t s_attn16[CH][72];  // attn -> solve, fp16
  __shared__ __align__(16) float s_extra[2816];
  __shared__ float s_gcum[CH], s_eg[CH], s_beta[CH], s_bb[CH];
  float* S = Sg + (size_t)h * DK * DK;
  const int lane = tid & 31, warp = tid >> 5;
  const int m0 = (warp >> 3) * 16, n0 = (warp & 7) * 16;
  const int dcS = n0 + (lane & 15);
  const int nchunks = (P + CH - 1) / CH;
  const int n2 = (nchunks + 1) >> 1;
  fx_floatx8 Sr0, Sr1;

  if (half == 0) {
#pragma unroll
    for (int e = 0; e < 8; e++) {
      Sr0[e] = S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS];
      Sr1[e] = S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS];
    }
    for (int c = 0; c < n2; c++) {
      const int t0 = c * CH;
      // ---- P0: stage k16/v16 + gates + cumsum (FEAT=7 float4) ----
      for (int i = tid; i < CH * DK / 4; i += NT) {
        const int t = i >> 5, d4 = (i & 31) << 2;
        float4 k4 = {0.f, 0.f, 0.f, 0.f}, v4 = {0.f, 0.f, 0.f, 0.f};
        if (t0 + t < P) {
          k4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 2048 +
                                kh * DK + d4);
          v4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 4096 +
                                h * DK + d4);
        }
        uint16_t kh4[4] = {fx_f2h(k4.x), fx_f2h(k4.y), fx_f2h(k4.z),
                           fx_f2h(k4.w)};
        uint16_t vh4[4] = {fx_f2h(v4.x), fx_f2h(v4.y), fx_f2h(v4.z),
                           fx_f2h(v4.w)};
        *(uint2*)&s_k16[t][d4] = *(const uint2*)kh4;
        *(uint2*)&s_v16[t][d4] = *(const uint2*)vh4;
      }
      if (tid < CH) {
        s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
        s_bb[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
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
      const float glast = s_gcum[CH - 1];
      if (tid < CH) {
        s_eg[tid] = expf(s_gcum[tid]);
        s_beta[tid] = expf(glast - s_gcum[tid]);
      }
      const float egl = expf(glast);
      // ---- P1: attn[r][j] = -bb[r]*(k[r].k[j])*exp(gcum[r]-gcum[j]) ----
      if (warp < 16) {
        const int fr = (warp >> 2) * 16, fc = (warp & 3) * 16;
        const uint16_t* arow = &s_k16[fr + fx_rowA(lane)][0];
        const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
        fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < DK / 16; kk++) {
          fx_shortx16 a = fx_ld16(arow + kk * 16);
          fx_shortx16 b = fx_ld16(brow + kk * 16);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
        const int col = fc + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = fr + (lane < 16 ? e : 8 + e);
          float a = 0.f;
          if (col < r) a = -s_bb[r] * acc[e] * expf(s_gcum[r] - s_gcum[col]);
          s_attn16[r][col] = fx_f2h(a);
        }
      }
      __syncthreads();
      // ---- P2: ut5 solve, verbatim from k_gdn_intra ----
      {
        const int g = tid >> 8, el = tid & 255;
        const int er = el >> 4, ec = el & 15;
        {
          const int col = er, mg = ec, base = g * 16;
          float solved = 0.f;
          for (int i = 1; i < 16; ++i) {
            float p = 0.f;
            if (mg < i) p = fx_h2f(s_attn16[base + i][base + mg]) * solved;
            float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
            float value = fx_h2f(s_attn16[base + i][base + col]);
#pragma unroll
            for (int gg = 0; gg < 16; gg += 2)
              value += __shfl_sync(~0ull, pair, gg, 16);
            if (i == mg && col < i) solved = value;
          }
          __syncthreads();
          if (col < mg) s_attn16[base + mg][base + col] = fx_f2h(solved);
        }
        __syncthreads();
        if (g < 3) {
          const int j = g + 1, b = g;
          float a[16];
#pragma unroll
          for (int n = 0; n < 16; n++)
            a[n] = fx_h2f(s_attn16[j * 16 + er][b * 16 + n]);
          float u = a[ec];
#pragma unroll
          for (int n = 0; n < 16; n++)
            u += a[n] * fx_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
          s_extra[g * 256 + el] = u;
        }
        __syncthreads();
        if (g < 3) {
          const int j = g + 1, b = g;
          float sj[16];
#pragma unroll
          for (int m = 0; m < 16; m++)
            sj[m] = fx_h2f(s_attn16[j * 16 + er][j * 16 + m]);
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
            a0[n] = fx_h2f(s_attn16[j * 16 + er][b * 16 + n]);
            a1[n] = fx_h2f(s_attn16[j * 16 + er][(b + 1) * 16 + n]);
          }
          float u = a0[ec];
#pragma unroll
          for (int n = 0; n < 16; n++) {
            u += a0[n] * fx_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
            u += a1[n] * s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 +
                                 ec];
          }
          s_extra[g * 256 + el] = u;
        }
        __syncthreads();
        if (g < 2) {
          const int j = g + 2, b = g;
          float sj[16];
#pragma unroll
          for (int m = 0; m < 16; m++)
            sj[m] = fx_h2f(s_attn16[j * 16 + er][j * 16 + m]);
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
            a0[n] = fx_h2f(s_attn16[48 + er][n]);
            a1[n] = fx_h2f(s_attn16[48 + er][16 + n]);
            a2[n] = fx_h2f(s_attn16[48 + er][32 + n]);
          }
          float u = a0[ec];
#pragma unroll
          for (int n = 0; n < 16; n++) {
            u += a0[n] * fx_h2f(s_attn16[n][ec]);
            u += a1[n] * s_extra[1024 + n * 16 + ec];
            u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];
          }
          s_extra[el] = u;
        }
        __syncthreads();
        if (g == 0) {
          float sj[16];
#pragma unroll
          for (int m = 0; m < 16; m++)
            sj[m] = fx_h2f(s_attn16[48 + er][48 + m]);
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
              s_attn16[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
                  fx_f2h(s_extra[1024 + tl * 256 + e]);
          }
      }
      __syncthreads();
      if (tid < CH)
        s_attn16[tid][tid] = fx_f2h(fx_h2f(s_attn16[tid][tid]) + 1.f);
      __syncthreads();
      // ---- P3a: ks = k16 @ S16T, qS = q @ S16T, attn2 merged (doA2) ----
      float ksc[8], qsc[8], vnc[8];
      const bool doA2 = (warp & 7) < 4;
      fx_floatx8 a2acc = {0, 0, 0, 0, 0, 0, 0, 0};
      {
        const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
        const uint16_t* karow = &s_k16[r0 + fx_rowA(lane)][0];
        uint16_t* s16 = (uint16_t*)s_extra;  // [128 dc][32 j]
        const int j0 = (warp & 3) * 16;
        const uint16_t* kbrow = &s_k16[j0 + (lane & 15)][0];
        fx_floatx8 kacc = {0, 0, 0, 0, 0, 0, 0, 0};
        fx_floatx8 qacc = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int qt = 0; qt < 4; qt++) {
          __syncthreads();
#pragma unroll
          for (int t = 0; t < 2; t++) {
            const int mt = m0 + t * 64;
            if ((mt >> 5) == qt) {
              uint16_t tmp[8];
              const fx_floatx8& Srt = t ? Sr1 : Sr0;
#pragma unroll
              for (int e = 0; e < 8; e++) tmp[e] = fx_f2h(Srt[e]);
              *(uint4*)&s16[dcS * 32 + (mt & 31) + (lane < 16 ? 0 : 8)] =
                  *(const uint4*)tmp;
            }
          }
          __syncthreads();
          const uint16_t* brow = s16 + (c0 + (lane & 15)) * 32;
#pragma unroll
          for (int kk = 0; kk < 2; kk++) {
            fx_shortx16 bk = fx_ld16(brow + kk * 16);
            fx_shortx16 ak = fx_ld16(karow + (qt * 2 + kk) * 16);
            kacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ak, bk, kacc);
            const int rq = t0 + r0 + fx_rowA(lane);
            const float* qp = qkv + (size_t)(rq < P ? rq : t0) * qkvstride +
                              kh * DK + (qt * 2 + kk) * 16;
            short qt16[16];
#pragma unroll
            for (int q4 = 0; q4 < 4; q4++) {
              float4 x = *(const float4*)(qp + q4 * 4);
              qt16[q4 * 4 + 0] = (short)fx_f2h(x.x);
              qt16[q4 * 4 + 1] = (short)fx_f2h(x.y);
              qt16[q4 * 4 + 2] = (short)fx_f2h(x.z);
              qt16[q4 * 4 + 3] = (short)fx_f2h(x.w);
            }
            fx_shortx16 aq;
            memcpy(&aq, qt16, 32);
            qacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, bk, qacc);
            if (doA2) {
              fx_shortx16 bk2 = fx_ld16(kbrow + (qt * 2 + kk) * 16);
              a2acc =
                  __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, bk2, a2acc);
            }
          }
        }
#pragma unroll
        for (int e = 0; e < 8; e++) {
          ksc[e] = kacc[e];
          qsc[e] = qacc[e];
        }
      }
      __syncthreads();  // S16T reads done; ksc/qsc complete
      if (doA2) {
        const int r0 = (warp >> 3) * 16;
        const int j0 = (warp & 3) * 16;
        uint16_t* a2w = (uint16_t*)s_extra;  // [64 r][72 j]
        const int col = j0 + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = r0 + (lane < 16 ? e : 8 + e);
          float a = 0.f;
          if (col <= r && t0 + r < P)
            a = a2acc[e] * expf(s_gcum[r] - s_gcum[col]);
          a2w[r * 72 + col] = fx_f2h(a);
        }
      }
      // ---- P3b: WT[dc][l] = bb_l*(v[l][dc] - eg_l*ks[l][dc]) ----
      {
        const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
        const int dc2 = c0 + (lane & 15);
        uint16_t vv[8];
#pragma unroll
        for (int e = 0; e < 8; e++)
          vv[e] = s_v16[r0 + (lane < 16 ? e : 8 + e)][dc2];
        __syncthreads();
        uint16_t* WT = (uint16_t*)s_v16;  // [128 dc][72 l]
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int l = r0 + (lane < 16 ? e : 8 + e);
          WT[dc2 * 72 + l] =
              fx_f2h(s_bb[l] * (fx_h2f(vv[e]) - s_eg[l] * ksc[e]));
        }
      }
      __syncthreads();  // WT complete
      // ---- P3c: vnT[dc][r] = sum_l WT[dc][l] * attn16[r][l] ----
      {
        const uint16_t* WT = (const uint16_t*)s_v16;
        const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;
        const uint16_t* arow = WT + (size_t)(m0 + fx_rowA(lane)) * 72;
        const uint16_t* brow = &s_attn16[n0 + (lane & 15)][0];
        fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < CH / 16; kk++) {
          fx_shortx16 a = fx_ld16(arow + kk * 16);
          fx_shortx16 b = fx_ld16(brow + kk * 16);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
#pragma unroll
        for (int e = 0; e < 8; e++) vnc[e] = acc[e];
      }
      __syncthreads();
      {
        uint16_t* vnT = (uint16_t*)s_v16;  // [128 dc][72 r]
        const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;
        const int rw = n0 + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int dcw = m0 + (lane < 16 ? e : 8 + e);
          vnT[dcw * 72 + rw] = fx_f2h(vnc[e]);
        }
      }
      __syncthreads();  // vnT ready
      // ---- S2: out = eg*qS + attn2 @ vnT (attn2 from s_extra) ----
      {
        const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
        const uint16_t* vnT = (const uint16_t*)s_v16;
        const uint16_t* a2r = (const uint16_t*)s_extra;  // [64][72]
        const uint16_t* arow = a2r + (size_t)(r0 + fx_rowA(lane)) * 72;
        const uint16_t* brow = vnT + (size_t)(c0 + (lane & 15)) * 72;
        fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < CH / 16; kk++) {
          fx_shortx16 a = fx_ld16(arow + kk * 16);
          fx_shortx16 b = fx_ld16(brow + kk * 16);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
        const int dc2 = c0 + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = r0 + (lane < 16 ? e : 8 + e);
          if (t0 + r < P)
            out[(size_t)(t0 + r) * qkvstride + 4096 + h * DK + dc2] =
                s_eg[r] * qsc[e] + acc[e];
        }
      }
      __syncthreads();  // out complete; vnT still live for S3
      // ---- S3: S = S*egl + kTdecay @ vnT^T (double-buffered ktd) ----
      {
#pragma unroll
        for (int e = 0; e < 8; e++) {
          Sr0[e] *= egl;
          Sr1[e] *= egl;
        }
        uint16_t* ktd = (uint16_t*)s_extra;  // two [128 j][16 r] buffers
        const uint16_t* vnT = (const uint16_t*)s_v16;
        for (int i = tid; i < DK * 16; i += NT) {
          const int j = i >> 4, r = i & 15;
          ktd[j * 16 + r] = fx_f2h(fx_h2f(s_k16[r][j]) * s_beta[r]);
        }
        __syncthreads();
        for (int qt = 0; qt < 4; qt++) {
          if (qt + 1 < 4) {
            uint16_t* nb = ktd + ((qt + 1) & 1) * DK * 16;
            for (int i = tid; i < DK * 16; i += NT) {
              const int j = i >> 4, r = i & 15;
              nb[j * 16 + r] = fx_f2h(fx_h2f(s_k16[(qt + 1) * 16 + r][j]) *
                                      s_beta[(qt + 1) * 16 + r]);
            }
          }
          const uint16_t* buf = ktd + (qt & 1) * DK * 16;
          const uint16_t* arow0 = buf + (m0 + fx_rowA(lane)) * 16;
          const uint16_t* arow1 = buf + (m0 + 64 + fx_rowA(lane)) * 16;
          fx_shortx16 b =
              fx_ld16(vnT + (size_t)(n0 + (lane & 15)) * 72 + qt * 16);
          Sr0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow0), b,
                                                           Sr0);
          Sr1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow1), b,
                                                           Sr1);
          if (qt + 1 < 4) __syncthreads();
        }
      }
      __syncthreads();  // k16/vn reads done before next chunk's staging
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
      S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr0[e];
      S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr1[e];
    }
    __threadfence();
    if (tid == 0) atomicExch(&flags[h], 1);
    return;
  }

  // ---- half1 pass 1: S-independent intra into ws3 ----
  for (int c = (f3mask & 1) ? n2 : nchunks; c < nchunks; c++) {
    const int t0 = c * CH;
    uint16_t* wb = ws3 + ((size_t)c * 48 + h) * WS3_U16;
    float* wbf = (float*)(wb + WS3_F);
    // P0: stage k16 only (v not needed intra) + gates + cumsum
    for (int i = tid; i < CH * DK / 4; i += NT) {
      const int t = i >> 5, d4 = (i & 31) << 2;
      float4 k4 = {0.f, 0.f, 0.f, 0.f};
      if (t0 + t < P)
        k4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 2048 +
                              kh * DK + d4);
      uint16_t kh4[4] = {fx_f2h(k4.x), fx_f2h(k4.y), fx_f2h(k4.z),
                         fx_f2h(k4.w)};
      *(uint2*)&s_k16[t][d4] = *(const uint2*)kh4;
    }
    if (tid < CH) {
      s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
      s_bb[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
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
    const float glast = s_gcum[CH - 1];
    if (tid < CH) {
      s_eg[tid] = expf(s_gcum[tid]);
      s_beta[tid] = expf(glast - s_gcum[tid]);
    }
    // ---- P1: attn ----
    if (warp < 16) {
      const int fr = (warp >> 2) * 16, fc = (warp & 3) * 16;
      const uint16_t* arow = &s_k16[fr + fx_rowA(lane)][0];
      const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < DK / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col < r) a = -s_bb[r] * acc[e] * expf(s_gcum[r] - s_gcum[col]);
        s_attn16[r][col] = fx_f2h(a);
      }
    }
    __syncthreads();
    // ---- P2: ut5 solve, verbatim ----
    {
      const int g = tid >> 8, el = tid & 255;
      const int er = el >> 4, ec = el & 15;
      {
        const int col = er, mg = ec, base = g * 16;
        float solved = 0.f;
        for (int i = 1; i < 16; ++i) {
          float p = 0.f;
          if (mg < i) p = fx_h2f(s_attn16[base + i][base + mg]) * solved;
          float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
          float value = fx_h2f(s_attn16[base + i][base + col]);
#pragma unroll
          for (int gg = 0; gg < 16; gg += 2)
            value += __shfl_sync(~0ull, pair, gg, 16);
          if (i == mg && col < i) solved = value;
        }
        __syncthreads();
        if (col < mg) s_attn16[base + mg][base + col] = fx_f2h(solved);
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float a[16];
#pragma unroll
        for (int n = 0; n < 16; n++)
          a[n] = fx_h2f(s_attn16[j * 16 + er][b * 16 + n]);
        float u = a[ec];
#pragma unroll
        for (int n = 0; n < 16; n++)
          u += a[n] * fx_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++)
          sj[m] = fx_h2f(s_attn16[j * 16 + er][j * 16 + m]);
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
          a0[n] = fx_h2f(s_attn16[j * 16 + er][b * 16 + n]);
          a1[n] = fx_h2f(s_attn16[j * 16 + er][(b + 1) * 16 + n]);
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * fx_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
          u += a1[n] *
               s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 + ec];
        }
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 2) {
        const int j = g + 2, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++)
          sj[m] = fx_h2f(s_attn16[j * 16 + er][j * 16 + m]);
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
          a0[n] = fx_h2f(s_attn16[48 + er][n]);
          a1[n] = fx_h2f(s_attn16[48 + er][16 + n]);
          a2[n] = fx_h2f(s_attn16[48 + er][32 + n]);
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * fx_h2f(s_attn16[n][ec]);
          u += a1[n] * s_extra[1024 + n * 16 + ec];
          u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];
        }
        s_extra[el] = u;
      }
      __syncthreads();
      if (g == 0) {
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++)
          sj[m] = fx_h2f(s_attn16[48 + er][48 + m]);
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
            s_attn16[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
                fx_f2h(s_extra[1024 + tl * 256 + e]);
        }
    }
    __syncthreads();
    if (tid < CH)
      s_attn16[tid][tid] = fx_f2h(fx_h2f(s_attn16[tid][tid]) + 1.f);
    __syncthreads();
    // ---- P4 standalone: attn2 -> s_extra [64][72]h ----
    if (warp < 16) {
      const int fr = (warp >> 2) * 16, fc = (warp & 3) * 16;
      const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
      uint16_t* a2w = (uint16_t*)s_extra;
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < DK / 16; kk++) {
        const int rq = t0 + fr + fx_rowA(lane);
        const float* qp =
            qkv + (size_t)(rq < P ? rq : t0) * qkvstride + kh * DK + kk * 16;
        short qt[16];
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) {
          float4 x = *(const float4*)(qp + q4 * 4);
          qt[q4 * 4 + 0] = (short)fx_f2h(x.x);
          qt[q4 * 4 + 1] = (short)fx_f2h(x.y);
          qt[q4 * 4 + 2] = (short)fx_f2h(x.z);
          qt[q4 * 4 + 3] = (short)fx_f2h(x.w);
        }
        fx_shortx16 a;
        memcpy(&a, qt, 32);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col <= r && t0 + r < P)
          a = acc[e] * expf(s_gcum[r] - s_gcum[col]);
        a2w[r * 72 + col] = fx_f2h(a);
      }
    }
    __syncthreads();
    // ---- ws write: attn16, attn2, gates ----
    for (int i = tid; i < CH * 72 / 8; i += NT)
      ((uint4*)wb)[i] = ((const uint4*)s_attn16)[i];
    for (int i = tid; i < CH * 72 / 8; i += NT)
      ((uint4*)(wb + WS3_A2))[i] = ((const uint4*)s_extra)[i];
    if (tid < CH) {
      wbf[tid] = s_eg[tid];
      wbf[CH + tid] = s_beta[tid];
      wbf[2 * CH + tid] = s_bb[tid];
    }
    if (tid == 0) wbf[3 * CH] = expf(glast);
    __syncthreads();  // ws writes done before next chunk reuses shared
  }
  // ---- acquire half0's S (skipped when f3mask bit2 set, debug only) ----
  if (!(f3mask & 4) && tid == 0)
    while (atomicAdd(&flags[h], 0) == 0) {
    }
  __syncthreads();
#pragma unroll
  for (int e = 0; e < 8; e++) {
    Sr0[e] = S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS];
    Sr1[e] = S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS];
  }
  // ---- half1 pass 2: strip ----
  for (int c = (f3mask & 2) ? n2 : nchunks; c < nchunks; c++) {
    const int t0 = c * CH;
    const uint16_t* wb = ws3 + ((size_t)c * 48 + h) * WS3_U16;
    const float* wbf = (const float*)(wb + WS3_F);
    // restage k16/v16 (verbatim float4 P0) + gates/attn16 from ws
    for (int i = tid; i < CH * DK / 4; i += NT) {
      const int t = i >> 5, d4 = (i & 31) << 2;
      float4 k4 = {0.f, 0.f, 0.f, 0.f}, v4 = {0.f, 0.f, 0.f, 0.f};
      if (t0 + t < P) {
        k4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 2048 +
                              kh * DK + d4);
        v4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 4096 +
                              h * DK + d4);
      }
      uint16_t kh4[4] = {fx_f2h(k4.x), fx_f2h(k4.y), fx_f2h(k4.z),
                         fx_f2h(k4.w)};
      uint16_t vh4[4] = {fx_f2h(v4.x), fx_f2h(v4.y), fx_f2h(v4.z),
                         fx_f2h(v4.w)};
      *(uint2*)&s_k16[t][d4] = *(const uint2*)kh4;
      *(uint2*)&s_v16[t][d4] = *(const uint2*)vh4;
    }
    if (tid < CH) {
      s_eg[tid] = wbf[tid];
      s_beta[tid] = wbf[CH + tid];
      s_bb[tid] = wbf[2 * CH + tid];
    }
    const float egl = wbf[3 * CH];
    for (int i = tid; i < CH * 72 / 8; i += NT)
      ((uint4*)s_attn16)[i] = ((const uint4*)wb)[i];
    __syncthreads();
    // ---- P3a (no doA2): ks/qS ----
    float ksc[8], qsc[8], vnc[8];
    {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const uint16_t* karow = &s_k16[r0 + fx_rowA(lane)][0];
      uint16_t* s16 = (uint16_t*)s_extra;  // [128 dc][32 j]
      fx_floatx8 kacc = {0, 0, 0, 0, 0, 0, 0, 0};
      fx_floatx8 qacc = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int qt = 0; qt < 4; qt++) {
        __syncthreads();
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const int mt = m0 + t * 64;
          if ((mt >> 5) == qt) {
            uint16_t tmp[8];
            const fx_floatx8& Srt = t ? Sr1 : Sr0;
#pragma unroll
            for (int e = 0; e < 8; e++) tmp[e] = fx_f2h(Srt[e]);
            *(uint4*)&s16[dcS * 32 + (mt & 31) + (lane < 16 ? 0 : 8)] =
                *(const uint4*)tmp;
          }
        }
        __syncthreads();
        const uint16_t* brow = s16 + (c0 + (lane & 15)) * 32;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          fx_shortx16 bk = fx_ld16(brow + kk * 16);
          fx_shortx16 ak = fx_ld16(karow + (qt * 2 + kk) * 16);
          kacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ak, bk, kacc);
          const int rq = t0 + r0 + fx_rowA(lane);
          const float* qp = qkv + (size_t)(rq < P ? rq : t0) * qkvstride +
                            kh * DK + (qt * 2 + kk) * 16;
          short qt16[16];
#pragma unroll
          for (int q4 = 0; q4 < 4; q4++) {
            float4 x = *(const float4*)(qp + q4 * 4);
            qt16[q4 * 4 + 0] = (short)fx_f2h(x.x);
            qt16[q4 * 4 + 1] = (short)fx_f2h(x.y);
            qt16[q4 * 4 + 2] = (short)fx_f2h(x.z);
            qt16[q4 * 4 + 3] = (short)fx_f2h(x.w);
          }
          fx_shortx16 aq;
          memcpy(&aq, qt16, 32);
          qacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, bk, qacc);
        }
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        ksc[e] = kacc[e];
        qsc[e] = qacc[e];
      }
    }
    __syncthreads();  // S16T reads done; ksc/qsc complete; s_extra free
    // attn2 ws -> s_extra [64][72]h; P3b's barrier orders it before S2
    for (int i = tid; i < CH * 72 / 8; i += NT)
      ((uint4*)s_extra)[i] = ((const uint4*)(wb + WS3_A2))[i];
    // ---- P3b: WT ----
    {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const int dc2 = c0 + (lane & 15);
      uint16_t vv[8];
#pragma unroll
      for (int e = 0; e < 8; e++)
        vv[e] = s_v16[r0 + (lane < 16 ? e : 8 + e)][dc2];
      __syncthreads();
      uint16_t* WT = (uint16_t*)s_v16;  // [128 dc][72 l]
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int l = r0 + (lane < 16 ? e : 8 + e);
        WT[dc2 * 72 + l] =
            fx_f2h(s_bb[l] * (fx_h2f(vv[e]) - s_eg[l] * ksc[e]));
      }
    }
    __syncthreads();  // WT complete
    // ---- P3c: vnT ----
    {
      const uint16_t* WT = (const uint16_t*)s_v16;
      const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;
      const uint16_t* arow = WT + (size_t)(m0 + fx_rowA(lane)) * 72;
      const uint16_t* brow = &s_attn16[n0 + (lane & 15)][0];
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < CH / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) vnc[e] = acc[e];
    }
    __syncthreads();
    {
      uint16_t* vnT = (uint16_t*)s_v16;  // [128 dc][72 r]
      const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;
      const int rw = n0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int dcw = m0 + (lane < 16 ? e : 8 + e);
        vnT[dcw * 72 + rw] = fx_f2h(vnc[e]);
      }
    }
    __syncthreads();  // vnT ready
    // ---- S2: out = eg*qS + attn2 @ vnT (attn2 from s_extra) ----
    {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const uint16_t* vnT = (const uint16_t*)s_v16;
      const uint16_t* a2r = (const uint16_t*)s_extra;  // [64][72]
      const uint16_t* arow = a2r + (size_t)(r0 + fx_rowA(lane)) * 72;
      const uint16_t* brow = vnT + (size_t)(c0 + (lane & 15)) * 72;
      fx_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < CH / 16; kk++) {
        fx_shortx16 a = fx_ld16(arow + kk * 16);
        fx_shortx16 b = fx_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int dc2 = c0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = r0 + (lane < 16 ? e : 8 + e);
        if (t0 + r < P)
          out[(size_t)(t0 + r) * qkvstride + 4096 + h * DK + dc2] =
              s_eg[r] * qsc[e] + acc[e];
      }
    }
    __syncthreads();  // out complete; vnT still live for S3
    // ---- S3: double-buffered ktd ----
    {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        Sr0[e] *= egl;
        Sr1[e] *= egl;
      }
      uint16_t* ktd = (uint16_t*)s_extra;  // two [128 j][16 r] buffers
      const uint16_t* vnT = (const uint16_t*)s_v16;
      for (int i = tid; i < DK * 16; i += NT) {
        const int j = i >> 4, r = i & 15;
        ktd[j * 16 + r] = fx_f2h(fx_h2f(s_k16[r][j]) * s_beta[r]);
      }
      __syncthreads();
      for (int qt = 0; qt < 4; qt++) {
        if (qt + 1 < 4) {
          uint16_t* nb = ktd + ((qt + 1) & 1) * DK * 16;
          for (int i = tid; i < DK * 16; i += NT) {
            const int j = i >> 4, r = i & 15;
            nb[j * 16 + r] = fx_f2h(fx_h2f(s_k16[(qt + 1) * 16 + r][j]) *
                                    s_beta[(qt + 1) * 16 + r]);
          }
        }
        const uint16_t* buf = ktd + (qt & 1) * DK * 16;
        const uint16_t* arow0 = buf + (m0 + fx_rowA(lane)) * 16;
        const uint16_t* arow1 = buf + (m0 + 64 + fx_rowA(lane)) * 16;
        fx_shortx16 b =
            fx_ld16(vnT + (size_t)(n0 + (lane & 15)) * 72 + qt * 16);
        Sr0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow0), b,
                                                         Sr0);
        Sr1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fx_ld16(arow1), b,
                                                         Sr1);
        if (qt + 1 < 4) __syncthreads();
      }
    }
    __syncthreads();  // k16/vn reads done before next chunk's staging
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr0[e];
    S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr1[e];
  }
}

// ====================== reference: split kernels ===========================
// k_intra_ref: fp32 intra (pipe V0 semantics), grid (nchunks,48), NT=1024.
__global__ void __launch_bounds__(1024)
k_intra_ref(const float* __restrict__ qkv, const float* __restrict__ gb,
            const float* __restrict__ bb, float* __restrict__ ws, int P,
            int qkvstride) {
  const int NT = 1024;
  const int c = blockIdx.x, h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  __shared__ float s_k[CH][DK + 4];
  __shared__ float s_attn[CH][CH];
  __shared__ float s_extra[3072];
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH];
  float* wsb = ws + ((size_t)c * 48 + h) * WS_FLOATS;
  float* wvp = wsb + WS_VP;
  float* wkcd = wsb + WS_KCD;
  const int t0 = c * CH;
  for (int i = tid; i < CH * DK; i += NT) {
    int t = i / DK, d = i % DK;
    s_k[t][d] =
        (t0 + t < P) ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d]
                     : 0.f;
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
  for (int i = tid; i < CH * CH; i += NT) {
    int r = i / CH, j = i % CH;
    float a = 0.f;
    if (j < r) {
      float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
      for (int d = 0; d < DK; d += 4) {
        e0 += s_k[r][d] * s_k[j][d];
        e1 += s_k[r][d + 1] * s_k[j][d + 1];
        e2 += s_k[r][d + 2] * s_k[j][d + 2];
        e3 += s_k[r][d + 3] * s_k[j][d + 3];
      }
      float dot = (e0 + e1) + (e2 + e3);
      a = -s_beta[r] * dot * expf(s_gcum[r] - s_gcum[j]);
    }
    s_attn[r][j] = a;
  }
  __syncthreads();
  // ut5 solve (same as fused)
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
  if (tid < CH) s_attn[tid][tid] += 1.f;
  __syncthreads();
  // vp/kcd: band-staged v rows, fp32 FMA (production k_gdn_intra order)
  {
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
        float kjb = s_k[j][d] * s_beta[j] * s_eg[j];
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
  }
  __syncthreads();
  // attn2: q quarter tiles
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
          e0 += qr[dd] * s_k[j][dd];
          e1 += qr[dd + 1] * s_k[j][dd + 1];
          e2 += qr[dd + 2] * s_k[j][dd + 2];
          e3 += qr[dd + 3] * s_k[j][dd + 3];
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
}

// k_strip_ref: fp32 strip (strip V0 verbatim), grid (4,48), NT=256.
__global__ void __launch_bounds__(256)
k_strip_ref(const float* __restrict__ qkv, float* __restrict__ S,
            float* __restrict__ out, const float* __restrict__ ws, int P,
            int qkvstride) {
  const int NT = 256;
  const int h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  const int d0 = blockIdx.x * 32;
  __shared__ float s_St[DK][32];
  __shared__ float s_vn[CH][32];
  __shared__ float s_eg[CH], s_beta[CH];
  S += (size_t)h * DK * DK;
  for (int i = tid; i < DK * 32; i += NT)
    s_St[i / 32][i % 32] = S[(i / 32) * DK + d0 + (i % 32)];
  __syncthreads();
  const int nchunks = (P + CH - 1) / CH;
  const int dc = tid & 31;
  const int RG = NT / 32;
  const int RPT = CH * 32 / NT;
  for (int c = 0; c < nchunks; c++) {
    const int t0 = c * CH;
    const float* wsb = ws + ((size_t)c * 48 + h) * WS_FLOATS;
    const float* wvp = wsb + WS_VP;
    const float* wkcd = wsb + WS_KCD;
    const float* wattn2 = wsb + WS_ATTN2;
    if (tid < CH) {
      float gci = wsb[tid];
      s_eg[tid] = expf(gci);
      s_beta[tid] = expf(wsb[CH - 1] - gci);
    }
    const float egl = expf(wsb[CH - 1]);
    {
      int r = tid >> 5;
      const float* w0 = wkcd + r * DK;
      float e[RPT][4];
#pragma unroll
      for (int n = 0; n < RPT; n++)
#pragma unroll
        for (int u = 0; u < 4; u++) e[n][u] = 0.f;
      for (int j = 0; j < DK; j += 4) {
        float s0 = s_St[j][dc], s1 = s_St[j + 1][dc];
        float s2 = s_St[j + 2][dc], s3 = s_St[j + 3][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          float4 w = *(const float4*)(w0 + n * RG * DK + j);
          e[n][0] += w.x * s0;
          e[n][1] += w.y * s1;
          e[n][2] += w.z * s2;
          e[n][3] += w.w * s3;
        }
      }
#pragma unroll
      for (int n = 0; n < RPT; n++, r += RG)
        s_vn[r][dc] =
            wvp[r * DK + d0 + dc] - ((e[n][0] + e[n][1]) + (e[n][2] + e[n][3]));
    }
    __syncthreads();
    {
      int r = tid >> 5;
      const float* q0 = qkv + kh * DK;
      bool v[RPT];
      float eg[RPT], o[RPT];
#pragma unroll
      for (int n = 0; n < RPT; n++) {
        v[n] = (t0 + r + n * RG) < P;
        eg[n] = s_eg[r + n * RG];
        o[n] = 0.f;
      }
      for (int j = 0; j < DK; j += 4) {
        float s0 = s_St[j][dc], s1 = s_St[j + 1][dc];
        float s2 = s_St[j + 2][dc], s3 = s_St[j + 3][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          const float* qr = q0 + (size_t)(v[n] ? t0 + r + n * RG : t0) * qkvstride;
          float4 q4 = *(const float4*)(qr + j);
          o[n] += q4.x * s0 + q4.y * s1 + q4.z * s2 + q4.w * s3;
        }
      }
#pragma unroll
      for (int n = 0; n < RPT; n++) o[n] *= eg[n];
      for (int j = 0; j < CH; j++) {
        float vn = s_vn[j][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++)
          o[n] += wattn2[(r + n * RG) * CH + j] * vn;
      }
#pragma unroll
      for (int n = 0; n < RPT; n++)
        if (v[n])
          out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] =
              o[n];
    }
    __syncthreads();
    {
      const int jg = tid >> 3;
      const int dc2 = (tid & 7) * 4;
      float a[4][4];
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
#pragma unroll
        for (int u = 0; u < 4; u++) a[jj][u] = 0.f;
      for (int r = 0; r < CH; r++) {
        float4 k4 = {0.f, 0.f, 0.f, 0.f};
        if (t0 + r < P)
          k4 = *(const float4*)(qkv + (size_t)(t0 + r) * qkvstride + 2048 +
                                kh * DK + jg * 4);
        const float bt = s_beta[r];
        const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const float vn = s_vn[r][dc2 + u];
#pragma unroll
          for (int jj = 0; jj < 4; jj++) a[jj][u] += kv[jj] * bt * vn;
        }
      }
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const int j = jg * 4 + jj;
#pragma unroll
        for (int u = 0; u < 4; u++)
          s_St[j][dc2 + u] = s_St[j][dc2 + u] * egl + a[jj][u];
      }
    }
    __syncthreads();
  }
  for (int i = tid; i < DK * 32; i += NT)
    S[(i / 32) * DK + d0 + (i % 32)] = s_St[i / 32][i % 32];
}

// ================================ main =====================================
int main(int argc, char** argv) {
  const int P = argc > 1 ? atoi(argv[1]) : 8192;
  const int reps = argc > 2 ? atoi(argv[2]) : 10;
  const int qkvstride = 10240;
  const int nchunks = (P + CH - 1) / CH;

  float *d_qkv, *d_gb, *d_bb, *d_ws, *d_S, *d_out_ref, *d_out_fx, *d_S_ref;
  float *d_S2, *d_out_f2, *d_S3, *d_out_f3;
  uint16_t* d_ws3;
  int* d_flags;
  CK(hipMalloc(&d_qkv, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_gb, (size_t)P * 48 * 4));
  CK(hipMalloc(&d_bb, (size_t)P * 48 * 4));
  CK(hipMalloc(&d_ws, (size_t)nchunks * 48 * WS_FLOATS * 4));
  CK(hipMalloc(&d_S, (size_t)48 * DK * DK * 4));
  CK(hipMalloc(&d_S_ref, (size_t)48 * DK * DK * 4));
  CK(hipMalloc(&d_S2, (size_t)48 * DK * DK * 4));
  CK(hipMalloc(&d_S3, (size_t)48 * DK * DK * 4));
  CK(hipMalloc(&d_out_ref, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_out_fx, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_out_f2, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_out_f3, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_ws3, (size_t)nchunks * 48 * WS3_U16 * 2));
  CK(hipMalloc(&d_flags, 48 * 4));
  std::vector<float> hqkv((size_t)P * qkvstride), hS0(48 * DK * DK);
  {
    std::vector<float> hgb((size_t)P * 48), hbb((size_t)P * 48);
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
    for (auto& x : hS0) x = (rnd() - 0.5f) * 0.5f;
    CK(hipMemcpy(d_qkv, hqkv.data(), hqkv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_gb, hgb.data(), hgb.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_bb, hbb.data(), hbb.size() * 4, hipMemcpyHostToDevice));
  }
  auto reset = [&]() {
    CK(hipMemcpy(d_out_ref, hqkv.data(), hqkv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_out_fx, hqkv.data(), hqkv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_out_f2, hqkv.data(), hqkv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_out_f3, hqkv.data(), hqkv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_S, hS0.data(), hS0.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_S_ref, hS0.data(), hS0.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_S2, hS0.data(), hS0.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_S3, hS0.data(), hS0.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemset(d_flags, 0, 48 * 4));
  };
  auto run_ref = [&]() {
    k_intra_ref<<<dim3(nchunks, 48), 1024>>>(d_out_ref, d_gb, d_bb, d_ws, P,
                                             qkvstride);
    k_strip_ref<<<dim3(4, 48), 256>>>(d_out_ref, d_S_ref, d_out_ref, d_ws, P,
                                      qkvstride);
  };
  const unsigned phmask =
      getenv("PHMASK") ? (unsigned)atoi(getenv("PHMASK")) : 63u;
  const unsigned feat = getenv("FEAT") ? (unsigned)atoi(getenv("FEAT")) : 3u;
  const unsigned f3mask =
      getenv("F3MASK") ? (unsigned)atoi(getenv("F3MASK")) : 3u;
  auto run_fx = [&]() {
    k_fused<<<48, 1024>>>(d_out_fx, d_gb, d_bb, d_S, d_out_fx, P, qkvstride,
                          phmask);
  };
  auto run_fx2 = [&]() {
    k_fused2<<<48, 1024>>>(d_out_f2, d_gb, d_bb, d_S2, d_out_f2, P, qkvstride,
                           feat, phmask);
  };
  auto run_fx3 = [&]() {
    k_fused3<<<96, 1024>>>(d_out_f3, d_gb, d_bb, d_S3, d_out_f3, P, qkvstride,
                           d_ws3, d_flags, f3mask);
  };
  auto time_it = [&](const char* name, auto fn) {
    for (int i = 0; i < 3; i++) fn();
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < reps; i++) fn();
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    printf("== %-14s wall %7.3f ms/iter ==\n", name, ms / reps);
    return ms / reps;
  };

  // correctness from fresh state (single iteration each)
  if (phmask != 63u) {
    printf("phmask=%u feat=%u: timing-only mode\n", phmask, feat);
    reset();
    double wf = time_it("fused", run_fx);
    reset();
    double wf2 = time_it("fused2", run_fx2);
    printf("== phmask %u feat %u wall %.3f / %.3f ms ==\n", phmask, feat, wf,
           wf2);
    return 0;
  }
  reset();
  run_ref();
  run_fx();
  run_fx2();
  run_fx3();
  CK(hipDeviceSynchronize());
  {
    std::vector<float> a((size_t)P * qkvstride), b(a.size());
    CK(hipMemcpy(a.data(), d_out_ref, a.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), d_out_fx, b.size() * 4, hipMemcpyDeviceToHost));
    double sa = 0, sb = 0, mx = 0;
    long nd = 0;
    for (size_t i = 0; i < a.size(); i++) {
      double d = fabs((double)a[i] - (double)b[i]);
      if (a[i] != b[i]) nd++;
      sa += d * d;
      sb += (double)a[i] * a[i];
      double r = d / std::max(1.0, (double)fabs(a[i]));
      if (r > mx) mx = r;
    }
    printf("out: fused vs ref fro-rel %.4f%% maxrel %.4f%% (ndiff %ld/%zu)\n",
           100 * sqrt(sa / std::max(1e-30, sb)), 100 * mx, nd, a.size());
    std::vector<float> sa2(48 * DK * DK), sb2(48 * DK * DK);
    CK(hipMemcpy(sa2.data(), d_S_ref, sa2.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(sb2.data(), d_S, sb2.size() * 4, hipMemcpyDeviceToHost));
    double fa = 0, fb = 0, mx2 = 0;
    for (size_t i = 0; i < sa2.size(); i++) {
      double d = fabs((double)sa2[i] - (double)sb2[i]);
      fa += d * d;
      fb += (double)sa2[i] * sa2[i];
      double r = d / std::max(1.0, (double)fabs(sa2[i]));
      if (r > mx2) mx2 = r;
    }
    printf("S:   fused vs ref fro-rel %.4f%% maxrel %.4f%%\n",
           100 * sqrt(fa / std::max(1e-30, fb)), 100 * mx2);
    // fused2 vs ref (fro-rel) and vs fused (must be bitwise identical)
    std::vector<float> b2(a.size());
    CK(hipMemcpy(b2.data(), d_out_f2, b2.size() * 4, hipMemcpyDeviceToHost));
    double s2a = 0, mx3 = 0;
    long ndbit = 0, nd2 = 0;
    for (size_t i = 0; i < a.size(); i++) {
      double d = fabs((double)a[i] - (double)b2[i]);
      if (a[i] != b2[i]) nd2++;
      s2a += d * d;
      double r = d / std::max(1.0, (double)fabs(a[i]));
      if (r > mx3) mx3 = r;
      if (b[i] != b2[i]) ndbit++;
    }
    printf("out: fused2 vs ref fro-rel %.4f%% maxrel %.4f%% (ndiff %ld)\n",
           100 * sqrt(s2a / std::max(1e-30, sb)), 100 * mx3, nd2);
    printf("out: fused2 vs fused bitwise ndiff %ld (expect 0)\n", ndbit);
    std::vector<float> sc2(48 * DK * DK);
    CK(hipMemcpy(sc2.data(), d_S2, sc2.size() * 4, hipMemcpyDeviceToHost));
    double f2a = 0, mx4 = 0;
    long ndbit2 = 0;
    for (size_t i = 0; i < sa2.size(); i++) {
      double d = fabs((double)sa2[i] - (double)sc2[i]);
      f2a += d * d;
      double r = d / std::max(1.0, (double)fabs(sa2[i]));
      if (r > mx4) mx4 = r;
      if (sb2[i] != sc2[i]) ndbit2++;
    }
    printf("S:   fused2 vs ref fro-rel %.4f%% maxrel %.4f%%\n",
           100 * sqrt(f2a / std::max(1e-30, fb)), 100 * mx4);
    printf("S:   fused2 vs fused bitwise ndiff %ld (expect 0)\n", ndbit2);
    // fused3 vs ref (fro-rel) and vs fused (expect bitwise identical)
    std::vector<float> b3(a.size());
    CK(hipMemcpy(b3.data(), d_out_f3, b3.size() * 4, hipMemcpyDeviceToHost));
    double s3a = 0, mx5 = 0;
    long ndbit3 = 0, nd3 = 0;
    for (size_t i = 0; i < a.size(); i++) {
      double d = fabs((double)a[i] - (double)b3[i]);
      if (a[i] != b3[i]) nd3++;
      s3a += d * d;
      double r = d / std::max(1.0, (double)fabs(a[i]));
      if (r > mx5) mx5 = r;
      if (b[i] != b3[i]) ndbit3++;
    }
    printf("out: fused3 vs ref fro-rel %.4f%% maxrel %.4f%% (ndiff %ld)\n",
           100 * sqrt(s3a / std::max(1e-30, sb)), 100 * mx5, nd3);
    printf("out: fused3 vs fused bitwise ndiff %ld (expect 0)\n", ndbit3);
    std::vector<float> sc3(48 * DK * DK);
    CK(hipMemcpy(sc3.data(), d_S3, sc3.size() * 4, hipMemcpyDeviceToHost));
    double f3a = 0, mx6 = 0;
    long ndbit4 = 0;
    for (size_t i = 0; i < sa2.size(); i++) {
      double d = fabs((double)sa2[i] - (double)sc3[i]);
      f3a += d * d;
      double r = d / std::max(1.0, (double)fabs(sa2[i]));
      if (r > mx6) mx6 = r;
      if (sb2[i] != sc3[i]) ndbit4++;
    }
    printf("S:   fused3 vs ref fro-rel %.4f%% maxrel %.4f%%\n",
           100 * sqrt(f3a / std::max(1e-30, fb)), 100 * mx6);
    printf("S:   fused3 vs fused bitwise ndiff %ld (expect 0)\n", ndbit4);
  }
  // determinism: fresh run twice, out+S must be bit-identical
  {
    reset();
    run_fx2();
    CK(hipDeviceSynchronize());
    std::vector<float> o1((size_t)P * qkvstride), s1(48 * DK * DK);
    CK(hipMemcpy(o1.data(), d_out_f2, o1.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(s1.data(), d_S2, s1.size() * 4, hipMemcpyDeviceToHost));
    reset();
    run_fx2();
    CK(hipDeviceSynchronize());
    std::vector<float> o2((size_t)P * qkvstride), s2(48 * DK * DK);
    CK(hipMemcpy(o2.data(), d_out_f2, o2.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(s2.data(), d_S2, s2.size() * 4, hipMemcpyDeviceToHost));
    long nd = 0;
    for (size_t i = 0; i < o1.size(); i++) nd += o1[i] != o2[i];
    for (size_t i = 0; i < s1.size(); i++) nd += s1[i] != s2[i];
    printf("determinism(fused2): %s (ndiff %ld)\n", nd == 0 ? "YES" : "no",
           nd);
  }
  // determinism(fused3): fresh run twice, out+S must be bit-identical
  {
    reset();
    run_fx3();
    CK(hipDeviceSynchronize());
    std::vector<float> o1((size_t)P * qkvstride), s1(48 * DK * DK);
    CK(hipMemcpy(o1.data(), d_out_f3, o1.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(s1.data(), d_S3, s1.size() * 4, hipMemcpyDeviceToHost));
    reset();
    run_fx3();
    CK(hipDeviceSynchronize());
    std::vector<float> o2((size_t)P * qkvstride), s2(48 * DK * DK);
    CK(hipMemcpy(o2.data(), d_out_f3, o2.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(s2.data(), d_S3, s2.size() * 4, hipMemcpyDeviceToHost));
    long nd = 0;
    for (size_t i = 0; i < o1.size(); i++) nd += o1[i] != o2[i];
    for (size_t i = 0; i < s1.size(); i++) nd += s1[i] != s2[i];
    printf("determinism(fused3): %s (ndiff %ld)\n", nd == 0 ? "YES" : "no",
           nd);
  }

  reset();
  double wr = time_it("ref(intra+strip)", run_ref);
  reset();
  double wf = time_it("fused", run_fx);
  reset();
  double wf2 = time_it("fused2", run_fx2);
  reset();
  double wf3 = time_it("fused3", run_fx3);
  printf("== speedup %.2fx / fused2 %.2fx / fused3 %.2fx (fused3 vs fused2 "
         "%.3fx) ==\n",
         wr / wf, wr / wf2, wr / wf3, wf2 / wf3);
  return 0;
}
