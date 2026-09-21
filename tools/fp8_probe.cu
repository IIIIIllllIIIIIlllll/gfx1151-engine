// fp8_probe.cu — gfx1151 fp8 capability survey (2026-09-21).
//
// Hardware existence, established by direct assembler probes (clang
// -target amdgcn-amd-amdhsa -mcpu=gfx1151, one instruction per .s file):
//   v_wmma_f32_16x16x16_fp8_fp8 / _bf8_bf8      -> "not supported on this GPU"
//   __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32 (+bf8/mixed) -> undeclared
//   v_cvt_pk_fp8_f32 / v_cvt_pk_bf8_f32 / v_cvt_f32_fp8 / v_cvt_f32_bf8
//     (+ _e4m3 spellings)                        -> not supported / invalid
//   v_dot4_f32_fp8_fp8 / v_dot4_f32_bf8_bf8 / v_dot2_f32_fp8_fp8 /
//   v_dot4_f32_fp8_e4m3 / v_mfma_f32_16x16x32_fp8_fp8 -> not supported
// i.e. gfx1151 has NO fp8 compute, NO fp8 dot, NO fp8 cvt instructions.
// Everything fp8 here is therefore software-emulated (OCP e4m3, RNE,
// saturate-to-448 on overflow, 0x7F/0xFF=NaN); e4m3 values decode exactly
// into bf16 (e4m3 ⊂ bf16), so "fp8 storage + bf16 compute" is lossless
// after the decode step.
//
// This probe measures:
//   Q3-sw: software cvt cost — fp8->bf16 via 256-entry LDS LUT (the cheap
//          direction: weight/KV decode) and bf16->e4m3 arithmetic RNE
//          (the expensive direction: dynamic activation quant).
//   Q5:    e4m3 per-element quantization error (rms/max rel) and GEMM
//          output error vs an f32 reference, compared to the bf16 path:
//          per-tensor vs per-row absmax scaling, uniform + gaussian data,
//          K=2560 accumulation. fp8 GEMM is emulated by dequantizing to
//          bf16 and running the bf16 path (exact for e4m3 values).
//
// Measured results (gfx1151, this build):
//   Q3-sw: fp8->bf16 LUT decode 74.4 Gelem/s (223 GB/s, DRAM-bound, ~2 op/el,
//          ~free when fused); bf16->e4m3 arith RNE encode 57.5 Gelem/s
//          (172 GB/s, ~10 op/el).
//   Q5:    e4m3 element quant err rms 2.6% / max ~6% (uniform); GEMM K=2560
//          output vs f32 truth: fp8-dequant path 3.6-3.8% Frobenius rms
//          (= sqrt(2) x 2.6%, K-invariant for random data), max element
//          10-16%; bf16 path 0.0002% (sanity floor of the GEMM itself).
//
// Build: hipcc -O2 -o /tmp/fp8_probe tools/fp8_probe.cu -lrocblas
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      fprintf(stderr, "hip error %d (%s) at %d\n", (int)e_,                  \
              hipGetErrorString(e_), __LINE__);                              \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static uint16_t f2bf(float f) {  // RNE
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static __host__ __device__ float bf2f(uint16_t u) {
  uint32_t v = (uint32_t)u << 16;
  float f;
  memcpy(&f, &v, 4);
  return f;
}
static uint8_t f2e4m3(float f) {  // OCP e4m3, RNE, saturate to +-448
  uint32_t u;
  memcpy(&u, &f, 4);
  const uint32_t s = u >> 31;
  const uint32_t a = u & 0x7fffffffu;
  if (a == 0) return (uint8_t)(s << 7);
  float af;
  memcpy(&af, &a, 4);
  if (af != af) return 0x7f;
  if (af > 448.f) return (uint8_t)((s << 7) | 0x7e);
  const int e = (int)(a >> 23) - 127;
  const uint32_t mant = a & 0x7fffffu;
  int E = e + 7;
  if (E >= 1) {
    uint32_t M = (mant + 0x7ffffu + ((mant >> 20) & 1)) >> 20;
    if (M == 8) { M = 0; ++E; }
    if (E > 15) return (uint8_t)((s << 7) | 0x7e);
    return (uint8_t)((s << 7) | (E << 3) | M);
  }
  uint32_t M = (uint32_t)lrintf(af * 512.f);  // subnormal quantum 2^-9
  if (M > 7) M = 7;
  return (uint8_t)((s << 7) | M);
}
static float e4m32f(uint8_t v) {
  const int s = v >> 7, E = (v >> 3) & 15, M = v & 7;
  if (E == 15 && M == 7) return nanf("");
  float r = (E == 0) ? (float)M * (1.f / 512.f)
                     : ldexpf(1.f + (float)M / 8.f, E - 7);
  return s ? -r : r;
}
__device__ __forceinline__ uint16_t f2bf_d(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
// arithmetic bf16->e4m3 (RNE, saturate): the dynamic-quant direction.
__device__ __forceinline__ uint8_t f2e4m3_d(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  const uint32_t s = u >> 31;
  const uint32_t a = u & 0x7fffffffu;
  float af;
  memcpy(&af, &a, 4);
  if (!(af > 0.f)) return (uint8_t)(s << 7);
  if (af > 448.f) return (uint8_t)((s << 7) | 0x7e);
  const int e = (int)(a >> 23) - 127;
  const uint32_t mant = a & 0x7fffffu;
  int E = e + 7;
  if (E >= 1) {
    uint32_t M = (mant + 0x7ffffu + ((mant >> 20) & 1)) >> 20;
    if (M == 8) { M = 0; ++E; }
    if (E > 15) return (uint8_t)((s << 7) | 0x7e);
    return (uint8_t)((s << 7) | (E << 3) | M);
  }
  const float v = af * 512.f;
  uint32_t M = (uint32_t)__float2uint_rn(v);
  if (M > 7) M = 7;
  return (uint8_t)((s << 7) | M);
}

// --- Q3-sw microbench -------------------------------------------------------
// fp8 -> bf16 via 256-entry LDS LUT (1 LDS read + extract per element)
__global__ void __launch_bounds__(256) k_dec_lut(const uint8_t* __restrict__ in,
                                                 uint16_t* __restrict__ out,
                                                 size_t n16) {
  __shared__ uint16_t lut[256];
  if (threadIdx.x < 256) {
    const int v = threadIdx.x, s = v >> 7, E = (v >> 3) & 15, M = v & 7;
    float r = (E == 0) ? (float)M * (1.f / 512.f)
                       : ldexpf(1.f + (float)M / 8.f, E - 7);
    lut[threadIdx.x] = f2bf_d(s ? -r : r);
  }
  __syncthreads();
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  const uint4 pk = ((const uint4*)in)[i];  // 16 fp8
  const uint8_t* b = (const uint8_t*)&pk;
  uint4 o0, o1;
  uint16_t* d = (uint16_t*)&o0;
#pragma unroll
  for (int t = 0; t < 8; ++t) d[t] = lut[b[t]];
  d = (uint16_t*)&o1;
#pragma unroll
  for (int t = 0; t < 8; ++t) d[t] = lut[b[8 + t]];
  ((uint4*)out)[2 * i] = o0;
  ((uint4*)out)[2 * i + 1] = o1;
}
// bf16 -> e4m3 arithmetic RNE (dynamic quant direction)
__global__ void __launch_bounds__(256) k_enc_arith(const uint16_t* __restrict__ in,
                                                   uint8_t* __restrict__ out,
                                                   size_t n16) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  const uint4 p0 = ((const uint4*)in)[2 * i];
  const uint4 p1 = ((const uint4*)in)[2 * i + 1];
  const uint16_t* s0 = (const uint16_t*)&p0;
  const uint16_t* s1 = (const uint16_t*)&p1;
  uint4 o;
  uint8_t* d = (uint8_t*)&o;
#pragma unroll
  for (int t = 0; t < 8; ++t) d[t] = f2e4m3_d(bf2f(s0[t]));
#pragma unroll
  for (int t = 0; t < 8; ++t) d[t] = f2e4m3_d(bf2f(s1[t]));
  ((uint4*)out)[i] = o;
}
// copy baseline (same traffic as dec_lut)
__global__ void __launch_bounds__(256) k_copy16(const uint8_t* __restrict__ in,
                                                uint16_t* __restrict__ out,
                                                size_t n16) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  const uint4 pk = ((const uint4*)in)[i];
  const uint8_t* b = (const uint8_t*)&pk;
  uint4 o0, o1;
  uint16_t* d = (uint16_t*)&o0;
#pragma unroll
  for (int t = 0; t < 8; ++t) d[t] = (uint16_t)(b[t] << 2);
  d = (uint16_t*)&o1;
#pragma unroll
  for (int t = 0; t < 8; ++t) d[t] = (uint16_t)(b[8 + t] << 2);
  ((uint4*)out)[2 * i] = o0;
  ((uint4*)out)[2 * i + 1] = o1;
}

static uint32_t xs_state = 0xC0FFEE;
static uint32_t xs_next() {
  uint32_t x = xs_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  xs_state = x;
  return x;
}
static float xs_frand() { return (float)((xs_next() >> 8) & 0xFFFF) / 32768.f - 1.f; }
static float xs_gauss() {  // Box-Muller, sigma 1
  const float u1 = 0.5f * (xs_frand() + 1.f) + 1e-7f;
  const float u2 = 0.5f * (xs_frand() + 1.f);
  return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
}

struct QErr { double fro, rms, mx; };
static QErr rel_err(const std::vector<float>& a, const std::vector<float>& ref) {
  double s = 0, mx = 0, sa = 0, sb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = fabs((double)a[i] - (double)ref[i]);
    sa += d * d;
    sb += (double)ref[i] * ref[i];
    const double r = d / std::max(1.0, (double)fabs(ref[i]));
    s += r * r;
    mx = std::max(mx, r);
  }
  return {sqrt(sa / std::max(1e-30, sb)), sqrt(s / a.size()), mx};
}

int main() {
  printf("== gfx1151 fp8 hardware (assembler-probed, see file header) ==\n");
  printf("  fp8/bf8 WMMA: NO    fp8/bf8 dot: NO    fp8/bf8 cvt: NO\n\n");

  // e4m3 codec self-test: round-trip error bound over the e4m3 grid
  {
    double mx = 0;
    for (int v = 0; v < 256; ++v) {
      const float f = e4m32f((uint8_t)v);
      if (f != f) continue;
      const float g = e4m32f(f2e4m3(f));
      if (g != f) mx = 1;
    }
    printf("e4m3 codec self-test: %s\n", mx == 0 ? "PASS" : "FAIL");
  }

  // --- Q5: quantization error ----------------------------------------------
  const int P = 2048, N = 2560, K = 2560;
  for (int dist = 0; dist < 2; ++dist) {
    xs_state = 0xFEED + dist;
    std::vector<float> hX(P * (size_t)K), hW(N * (size_t)K);
    for (auto& v : hX) v = dist ? xs_gauss() * 0.5f : xs_frand();
    for (auto& v : hW) v = dist ? xs_gauss() * 0.02f : xs_frand() * 0.05f;
    // bf16 versions (the bf16 path's actual inputs)
    std::vector<uint16_t> bX(P * (size_t)K), bW(N * (size_t)K);
    for (size_t i = 0; i < hX.size(); ++i) bX[i] = f2bf(hX[i]);
    for (size_t i = 0; i < hW.size(); ++i) bW[i] = f2bf(hW[i]);
    for (auto& v : hX) v = bf2f(f2bf(v));  // truth = what bf16 path sees
    for (auto& v : hW) v = bf2f(f2bf(v));

    for (int mode = 0; mode < 2; ++mode) {  // 0=per-tensor, 1=per-row
      // quantize X (rows=tokens) and W (rows=channels), dequant to bf16
      std::vector<uint16_t> qX(P * (size_t)K), qW(N * (size_t)K);
      std::vector<float> sx(P, 1.f), sw(N, 1.f);
      float gx = 0, gw = 0;
      if (mode == 0) {
        for (float v : hX) gx = std::max(gx, fabsf(v));
        for (float v : hW) gw = std::max(gw, fabsf(v));
        gx = gx / 448.f; gw = gw / 448.f;
      }
      std::vector<double> qeX, qeW;  // per-element quant rel err
      for (int p = 0; p < P; ++p) {
        float s = gx;
        if (mode == 1) {
          float m = 0;
          for (int k = 0; k < K; ++k) m = std::max(m, fabsf(hX[(size_t)p * K + k]));
          s = m / 448.f;
        }
        sx[p] = s;
        for (int k = 0; k < K; ++k) {
          const float v = hX[(size_t)p * K + k];
          const float d = e4m32f(f2e4m3(v / s)) * s;
          qX[(size_t)p * K + k] = f2bf(d);
          if (fabsf(v) > 1e-6f) qeX.push_back(fabs((double)d - v) / fabs((double)v));
        }
      }
      for (int n = 0; n < N; ++n) {
        float s = gw;
        if (mode == 1) {
          float m = 0;
          for (int k = 0; k < K; ++k) m = std::max(m, fabsf(hW[(size_t)n * K + k]));
          s = m / 448.f;
        }
        sw[n] = s;
        for (int k = 0; k < K; ++k) {
          const float v = hW[(size_t)n * K + k];
          const float d = e4m32f(f2e4m3(v / s)) * s;
          qW[(size_t)n * K + k] = f2bf(d);
          if (fabsf(v) > 1e-6f) qeW.push_back(fabs((double)d - v) / fabs((double)v));
        }
      }
      auto rms = [](const std::vector<double>& v) {
        double s = 0, mx = 0;
        for (double x : v) { s += x * x; mx = std::max(mx, x); }
        printf("rms %.3f%% max %.1f%%", 100 * sqrt(s / v.size()), 100 * mx);
      };
      printf("Q5 %s %s: elem quant err X ", dist ? "gauss " : "uniform",
             mode ? "per-row   " : "per-tensor");
      rms(qeX);
      printf(" | W ");
      rms(qeW);
      printf("\n");

      // GEMM: truth f32(bf16-rounded inputs), bf16 path, fp8-dequant path
      uint16_t *dX, *dW, *dqX, *dqW;
      float *dY, *dT;
      CK(hipMalloc(&dX, hX.size() * 2));
      CK(hipMalloc(&dW, hW.size() * 2));
      CK(hipMalloc(&dqX, hX.size() * 2));
      CK(hipMalloc(&dqW, hW.size() * 2));
      CK(hipMalloc(&dY, (size_t)P * N * 4));
      CK(hipMalloc(&dT, (size_t)P * N * 4));
      CK(hipMemcpy(dX, bX.data(), hX.size() * 2, hipMemcpyHostToDevice));
      CK(hipMemcpy(dW, bW.data(), hW.size() * 2, hipMemcpyHostToDevice));
      CK(hipMemcpy(dqX, qX.data(), hX.size() * 2, hipMemcpyHostToDevice));
      CK(hipMemcpy(dqW, qW.data(), hW.size() * 2, hipMemcpyHostToDevice));
      rocblas_handle h;
      rocblas_create_handle(&h);
      float one = 1.f, zero = 0.f;
      // truth: f32 GEMM on the bf16-rounded values
      {
        std::vector<float> fX(hX.begin(), hX.end()), fW(hW.begin(), hW.end());
        float *dfX, *dfW;
        CK(hipMalloc(&dfX, hX.size() * 4));
        CK(hipMalloc(&dfW, hW.size() * 4));
        CK(hipMemcpy(dfX, fX.data(), hX.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dfW, fW.data(), hW.size() * 4, hipMemcpyHostToDevice));
        (void)rocblas_sgemm(h, rocblas_operation_transpose,
                            rocblas_operation_none, N, P, K, &one, dfW, K, dfX,
                            K, &zero, dT, N);
        CK(hipFree(dfX));
        CK(hipFree(dfW));
      }
      auto bf16_gemm = [&](const uint16_t* A, const uint16_t* B, float* out) {
        (void)rocblas_gemm_ex(h, rocblas_operation_transpose,
                              rocblas_operation_none, N, P, K, &one, (void*)A,
                              rocblas_datatype_bf16_r, K, (void*)B,
                              rocblas_datatype_bf16_r, K, &zero, out,
                              rocblas_datatype_f32_r, N, out,
                              rocblas_datatype_f32_r, N,
                              rocblas_datatype_f32_r,
                              rocblas_gemm_algo_standard, 0, 0);
      };
      std::vector<float> vT((size_t)P * N), vY((size_t)P * N);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(vT.data(), dT, vT.size() * 4, hipMemcpyDeviceToHost));
      bf16_gemm(dW, dX, dY);  // bf16 path
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(vY.data(), dY, vY.size() * 4, hipMemcpyDeviceToHost));
      QErr e16 = rel_err(vY, vT);
      bf16_gemm(dqW, dqX, dY);  // fp8-dequant path (scales already folded
      CK(hipDeviceSynchronize());  // into the dequantized bf16 values)
      CK(hipMemcpy(vY.data(), dY, vY.size() * 4, hipMemcpyDeviceToHost));
      QErr e8 = rel_err(vY, vT);
      printf("    GEMM out vs f32 truth: bf16 path fro %.4f%% (rms %.4f%% max %.2f%%) | "
             "fp8 path fro %.3f%% (rms %.3f%% max %.1f%%)\n",
             100 * e16.fro, 100 * e16.rms, 100 * e16.mx,
             100 * e8.fro, 100 * e8.rms, 100 * e8.mx);
      rocblas_destroy_handle(h);
      CK(hipFree(dX)); CK(hipFree(dW)); CK(hipFree(dqX)); CK(hipFree(dqW));
      CK(hipFree(dY)); CK(hipFree(dT));
    }
  }

  // --- Q3-sw: cvt throughput -------------------------------------------------
  {
    const size_t G = 1 << 26;  // 64M fp8 elements
    const size_t n16 = G / 16;
    uint8_t* di8;
    uint16_t* do16;
    CK(hipMalloc(&di8, G));
    CK(hipMalloc(&do16, G * 2));
    CK(hipMemset(di8, 0x35, G));
    const int grid = (int)((n16 + 255) / 256);
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto bench = [&](const char* name, auto&& fn, double bytes) {
      for (int i = 0; i < 3; ++i) fn();
      CK(hipDeviceSynchronize());
      float best = 1e9f;
      for (int i = 0; i < 10; ++i) {
        CK(hipEventRecord(e0));
        fn();
        CK(hipEventRecord(e1));
        CK(hipDeviceSynchronize());
        float ms;
        CK(hipEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms);
      }
      printf("  %-22s %7.3f ms  %6.1f Gelem/s  %6.1f GB/s\n", name, best,
             G / best / 1e6, bytes / best / 1e6);
    };
    printf("\n== Q3-sw software cvt throughput (%.0f M elements) ==\n", G / 1e6);
    bench("fp8->bf16 LDS LUT", [&] { k_dec_lut<<<grid, 256>>>(di8, do16, n16); },
          (double)G * 3);
    bench("fp8->bf16 copy-ish", [&] { k_copy16<<<grid, 256>>>(di8, do16, n16); },
          (double)G * 3);
    bench("bf16->fp8 arith RNE", [&] { k_enc_arith<<<grid, 256>>>(do16, di8, n16); },
          (double)G * 3);
    CK(hipFree(di8));
    CK(hipFree(do16));
  }
  return 0;
}
