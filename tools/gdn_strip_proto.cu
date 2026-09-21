// gdn_strip_proto.cu — ablation replica of k_gdn_inter_strip
// (src/gpu/gdec.cpp:6085): grid (4,48), NT=256, serial 128-chunk scan per
// block. Variants:
//   V0: full (verbatim fp32 code)
//   V2: null-compute — identical global/LDS traffic and barriers, MAC loops
//       replaced by token adds (loads kept live)
// Question: is strip compute-bound (WMMA would matter) or memory-bound?
// Build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/gdn_strip_proto \
//          tools/gdn_strip_proto.cu && /tmp/gdn_strip_proto [P=8192] [reps=10]
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

constexpr int CH = 64, DK = 128, NT = 256;
constexpr int WS_ATTN2 = 64, WS_VP = WS_ATTN2 + 64 * 64,
              WS_KCD = WS_VP + 64 * 128, WS_FLOATS = WS_KCD + 64 * 128;
// V3: fp16 ws — gcum stays fp32 [0,64), attn2/vp/kcd as halfs after it
constexpr int WS16_ATTN2 = 0, WS16_VP = WS16_ATTN2 + 64 * 64,
              WS16_KCD = WS16_VP + 64 * 128,
              WS16_FLOATS = 64 + (WS16_KCD + 64 * 128) / 2;

__device__ __forceinline__ uint16_t sp_f2h(float f) {
  return __half_as_ushort(__float2half_rn(f));
}
__device__ __forceinline__ float sp_h2f(uint16_t u) {
  return __half2float(__ushort_as_half(u));
}

// host-side RNE fp32->fp16 (bit trick, matches __float2half_rn for normals)
static uint16_t sp_f2h_host(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int e = ((x >> 23) & 0xff) - 112;  // rebias 127->15
  uint32_t m = x & 0x7fffff;
  if (e <= 0) return sign;           // flush tiny to zero (test data is ~1.0)
  if (e >= 31) return sign | 0x7bff; // clamp
  uint32_t h = sign | ((uint32_t)e << 10) | (m >> 13);
  if ((m & 0x1fff) > 0x1000 || ((m & 0x3fff) == 0x3000)) h++;  // RNE
  return (uint16_t)h;
}

template <int V, int SW = 32>
__global__ void __launch_bounds__(NT) k_strip_proto(
    const float* __restrict__ qkv, float* __restrict__ S,
    float* __restrict__ out, const float* __restrict__ ws, int P,
    int qkvstride) {
  const int h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  const int d0 = blockIdx.x * SW;
  __shared__ float s_St[DK][SW];
  __shared__ float s_vn[CH][SW];
  __shared__ float s_eg[CH], s_beta[CH];
  S += (size_t)h * DK * DK;
  for (int i = tid; i < DK * SW; i += NT)
    s_St[i / SW][i % SW] = S[(i / SW) * DK + d0 + (i % SW)];
  __syncthreads();
  const int nchunks = (P + CH - 1) / CH;
  const int dc = tid & (SW - 1);
  const int RG = NT / SW;
  const int RPT = CH * SW / NT;
  for (int c = 0; c < nchunks; c++) {
    const int t0 = c * CH;
    const float* wsb = ws + ((size_t)c * 48 + h) * (V >= 3 ? WS16_FLOATS : WS_FLOATS);
    const uint16_t* ws16 = (const uint16_t*)(wsb + 64);
    const float* wvp = wsb + WS_VP;
    const float* wkcd = wsb + WS_KCD;
    const float* wattn2 = wsb + WS_ATTN2;
    if (tid < CH) {
      float gci = wsb[tid];
      s_eg[tid] = expf(gci);
      s_beta[tid] = expf(wsb[CH - 1] - gci);
    }
    const float egl = expf(wsb[CH - 1]);
    // ---- v_new[r][dc] = value'[r][d0+dc] - kcd[r] . S[:, d0+dc] ----
    if (V == 0) {
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
    } else if (V == 3 || V == 4) {  // fp16 ws reads, fp32 FMA (same order as V0)
      int r = tid >> 5;
      const uint16_t* w0 = ws16 + WS16_KCD + r * DK;
      float e[RPT][4];
#pragma unroll
      for (int n = 0; n < RPT; n++)
#pragma unroll
        for (int u = 0; u < 4; u++) e[n][u] = 0.f;
      if (V == 3) {
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j][dc], s1 = s_St[j + 1][dc];
          float s2 = s_St[j + 2][dc], s3 = s_St[j + 3][dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            uint2 w = *(const uint2*)(w0 + n * RG * DK + j);
            e[n][0] += sp_h2f((uint16_t)w.x) * s0;
            e[n][1] += sp_h2f((uint16_t)(w.x >> 16)) * s1;
            e[n][2] += sp_h2f((uint16_t)w.y) * s2;
            e[n][3] += sp_h2f((uint16_t)(w.y >> 16)) * s3;
          }
        }
      } else {
        // V4: 16B loads (8 halfs), adds still j ascending per residue class
        for (int j = 0; j < DK; j += 8) {
          float s0 = s_St[j][dc], s1 = s_St[j + 1][dc];
          float s2 = s_St[j + 2][dc], s3 = s_St[j + 3][dc];
          float t0 = s_St[j + 4][dc], t1 = s_St[j + 5][dc];
          float t2 = s_St[j + 6][dc], t3 = s_St[j + 7][dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            uint4 w = *(const uint4*)(w0 + n * RG * DK + j);
            e[n][0] += sp_h2f((uint16_t)w.x) * s0;
            e[n][1] += sp_h2f((uint16_t)(w.x >> 16)) * s1;
            e[n][2] += sp_h2f((uint16_t)w.y) * s2;
            e[n][3] += sp_h2f((uint16_t)(w.y >> 16)) * s3;
            e[n][0] += sp_h2f((uint16_t)w.z) * t0;
            e[n][1] += sp_h2f((uint16_t)(w.z >> 16)) * t1;
            e[n][2] += sp_h2f((uint16_t)w.w) * t2;
            e[n][3] += sp_h2f((uint16_t)(w.w >> 16)) * t3;
          }
        }
      }
#pragma unroll
      for (int n = 0; n < RPT; n++, r += RG)
        s_vn[r][dc] =
            sp_h2f(ws16[WS16_VP + r * DK + d0 + dc]) -
            ((e[n][0] + e[n][1]) + (e[n][2] + e[n][3]));
    } else {  // null: same loads, token adds
      int r = tid >> 5;
      const float* w0 = wkcd + r * DK;
      float e[RPT];
#pragma unroll
      for (int n = 0; n < RPT; n++) e[n] = 0.f;
      for (int j = 0; j < DK; j += 4) {
        float s0 = s_St[j][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          float4 w = *(const float4*)(w0 + n * RG * DK + j);
          e[n] += w.x + s0;
        }
      }
#pragma unroll
      for (int n = 0; n < RPT; n++, r += RG)
        s_vn[r][dc] = wvp[r * DK + d0 + dc] - e[n];
    }
    __syncthreads();
    // ---- out[r][d0+dc] = (q[r]*exp(gcum[r])) . S + attn2[r] . v_new ----
    if (V == 0) {
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
    } else if (V == 3 || V == 4) {  // fp16 attn2 reads
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
          o[n] += sp_h2f(ws16[WS16_ATTN2 + (r + n * RG) * CH + j]) * vn;
      }
#pragma unroll
      for (int n = 0; n < RPT; n++)
        if (v[n])
          out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] =
              o[n];
    } else {  // null: same loads/stores, token adds
      int r = tid >> 5;
      const float* q0 = qkv + kh * DK;
      bool v[RPT];
      float o[RPT];
#pragma unroll
      for (int n = 0; n < RPT; n++) {
        v[n] = (t0 + r + n * RG) < P;
        o[n] = 0.f;
      }
      for (int j = 0; j < DK; j += 4) {
        float s0 = s_St[j][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          const float* qr = q0 + (size_t)(v[n] ? t0 + r + n * RG : t0) * qkvstride;
          float4 q4 = *(const float4*)(qr + j);
          o[n] += q4.x + s0;
        }
      }
      for (int j = 0; j < CH; j++) {
        float vn = s_vn[j][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) o[n] += wattn2[(r + n * RG) * CH + j] + vn;
      }
#pragma unroll
      for (int n = 0; n < RPT; n++)
        if (v[n])
          out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] =
              o[n];
    }
    __syncthreads();
    // ---- S[j][d0+dc] = S*egl + sum_r k[r][j]*exp(glast-gcum[r])*vnew[r][dc]
    if (V == 0 || V == 3 || V == 4) {
      constexpr int JB = SW / 32;  // dc-groups per j-pass (SW=64: two j halves)
      const int jg = tid >> (JB == 1 ? 3 : 4);
      const int dc2 = (tid & (JB == 1 ? 7 : 15)) * 4;
#pragma unroll
      for (int jb = 0; jb < JB; jb++) {
        float a[4][4];
#pragma unroll
        for (int jj = 0; jj < 4; jj++)
#pragma unroll
          for (int u = 0; u < 4; u++) a[jj][u] = 0.f;
        const int jbase = jg * 4 + jb * (JB == 1 ? 0 : 64);
        for (int r = 0; r < CH; r++) {
          float4 k4 = {0.f, 0.f, 0.f, 0.f};
          if (t0 + r < P)
            k4 = *(const float4*)(qkv + (size_t)(t0 + r) * qkvstride + 2048 +
                                  kh * DK + jbase);
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
          const int j = jbase + jj;
#pragma unroll
          for (int u = 0; u < 4; u++)
            s_St[j][dc2 + u] = s_St[j][dc2 + u] * egl + a[jj][u];
        }
      }
    } else {  // null
      const int jg = tid >> 3;
      const int dc2 = (tid & 7) * 4 % SW;
      float a[4] = {0.f, 0.f, 0.f, 0.f};
      for (int r = 0; r < CH; r++) {
        float4 k4 = {0.f, 0.f, 0.f, 0.f};
        if (t0 + r < P)
          k4 = *(const float4*)(qkv + (size_t)(t0 + r) * qkvstride + 2048 +
                                kh * DK + jg * 4);
        const float vn = s_vn[r][dc2];
        a[0] += k4.x + s_beta[r] + vn;
      }
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const int j = jg * 4 + jj;
#pragma unroll
        for (int u = 0; u < 4; u++)
          s_St[j][dc2 + u] = s_St[j][dc2 + u] * egl + a[0];
      }
    }
    __syncthreads();
  }
  for (int i = tid; i < DK * SW; i += NT)
    S[(i / SW) * DK + d0 + (i % SW)] = s_St[i / SW][i % SW];
}

int main(int argc, char** argv) {
  const int P = argc > 1 ? atoi(argv[1]) : 8192;
  const int reps = argc > 2 ? atoi(argv[2]) : 10;
  const int qkvstride = 10240;
  const int nchunks = (P + CH - 1) / CH;

  float *d_qkv, *d_S, *d_ws, *d_ws16;
  CK(hipMalloc(&d_qkv, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_S, (size_t)48 * DK * DK * 4));
  CK(hipMalloc(&d_ws, (size_t)nchunks * 48 * WS_FLOATS * 4));
  CK(hipMalloc(&d_ws16, (size_t)nchunks * 48 * WS16_FLOATS * 4));
  {
    std::vector<float> hq((size_t)P * qkvstride, 0.f),
        hw((size_t)nchunks * 48 * WS_FLOATS), hs(48 * DK * DK);
    unsigned long long s = 999;
    auto rnd = [&]() {
      s = s * 2862933555777941757ULL + 3037000493ULL;
      return (float)(s >> 40) / (float)(1ull << 24);
    };
    for (auto& x : hq) x = (rnd() - 0.5f) * 0.3f;
    for (auto& x : hw) x = (rnd() - 0.5f) * 1.0f;
    // gcum region must be a valid decreasing cumsum for expf sanity
    for (size_t i = 0; i < hw.size(); i += WS_FLOATS)
      for (int t = 0; t < CH; t++)
        hw[i + t] = -0.05f * (t + 1) - rnd() * 0.01f;
    for (auto& x : hs) x = (rnd() - 0.5f) * 0.5f;
    CK(hipMemcpy(d_qkv, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_ws, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_S, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    // fp16 ws: same gcum (fp32) + attn2/vp/kcd rounded RNE to half
    std::vector<float> hw16((size_t)nchunks * 48 * WS16_FLOATS);
    for (size_t base = 0, b16 = 0; base < hw.size();
         base += WS_FLOATS, b16 += WS16_FLOATS) {
      for (int t = 0; t < CH; t++) hw16[b16 + t] = hw[base + t];
      uint16_t* h16 = (uint16_t*)(hw16.data() + b16 + 64);
      for (int i = 0; i < 64 * 64 + 2 * 64 * 128; i++)
        h16[i] = sp_f2h_host(hw[base + WS_ATTN2 + i]);
    }
    CK(hipMemcpy(d_ws16, hw16.data(), hw16.size() * 4, hipMemcpyHostToDevice));
  }

  auto run = [&](int V) {
    if (V == 0)
      k_strip_proto<0><<<dim3(4, 48), NT>>>(d_qkv, d_S, d_qkv, d_ws, P,
                                            qkvstride);
    else if (V == 3)
      k_strip_proto<3><<<dim3(4, 48), NT>>>(d_qkv, d_S, d_qkv, d_ws16, P,
                                            qkvstride);
    else if (V == 4)
      k_strip_proto<4><<<dim3(4, 48), NT>>>(d_qkv, d_S, d_qkv, d_ws16, P,
                                            qkvstride);
    else if (V == 5)
      k_strip_proto<0, 64><<<dim3(2, 48), NT>>>(d_qkv, d_S, d_qkv, d_ws, P,
                                                qkvstride);
    else if (V == 6)
      k_strip_proto<4, 64><<<dim3(2, 48), NT>>>(d_qkv, d_S, d_qkv, d_ws16, P,
                                                qkvstride);
    else
      k_strip_proto<2><<<dim3(4, 48), NT>>>(d_qkv, d_S, d_qkv, d_ws, P,
                                            qkvstride);
  };
  const char* vn[6] = {"full", "nullc", "ws16", "ws16v", "sw64", "sw64+16"};
  const int vs[6] = {0, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; vi++) {
    const int V = vs[vi];
    for (int i = 0; i < 3; i++) run(V);
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < reps; i++) run(V);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    double w = ms / reps;
    // unique DRAM traffic per iter (with perfect L2 dedup across 4 strips):
    // per chunk-head: kcd + q 32K + attn2 + k 32K + vp + out 32K
    // (fp32: 32/16/32; fp16 ws halves kcd/attn2/vp)
    const bool h = V == 3 || V == 4 || V == 6;
    const double kcd = h ? 16 : 32, at = h ? 8 : 16, vp = h ? 16 : 32;
    double gb = (double)nchunks * 48 * (kcd + 32 + at + 32 + vp + 32) * 1024 / 1e9;
    printf("== %s: wall %.3f ms/iter (grid 4x48, %d chunks) -> unique-traffic "
           "floor %.3f ms, raw-4x traffic %.1f GB/s equivalent ==\n",
           vn[vi], w, nchunks, gb / 224.0 * 1e3, gb / (w / 1e3));
  }
  return 0;
}
