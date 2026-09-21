// Robustness driver for the k_gemm_wmma kernel integrated in gdec.cpp.
// The kernel body is #included verbatim from a .inc sed-extracted from
// src/gpu/parts/22_kernels_prefill.inc between the [gemm-wmma-begin/end] markers
// — same source, not a copy. Regenerate after touching the kernel:
//   sed -n '/\[gemm-wmma-begin\]/,/\[gemm-wmma-end\]/p' src/gpu/parts/22_kernels_prefill.inc > tools/gemm_wmma_kernel.inc
// The qw_* helpers below are the identical one-liners from gdec.cpp:3423-3432
// (outside the markers).
// Build: hipcc -O2 -o build/gemm_wmma_driver tools/gemm_wmma_driver.cu -lrocblas
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#include "gemm_wmma_kernel.inc"

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

int main() {
  const int NK[9][2] = {{2560, 6144}, {6144, 2560}, {10240, 2560},
                        {12288, 2560}, {10240, 320}, {512, 2560},
                        {2560, 640}, {320, 10240}, {640, 2560}};
  const int Ps[] = {53, 100, 1024, 8192, 8199, 32768};
  rocblas_handle h;
  rocblas_create_handle(&h);
  hipStream_t st;
  CK(hipStreamCreate(&st));
  rocblas_set_stream(h, st);
  int nbad = 0;
  for (auto& nk : NK) {
    const int N = nk[0], K = nk[1];
    const bool d3 = (N == 2560 && K == 6144);
    const bool narrow = (N == 10240 && K == 320) || (N == 512 && K == 2560) ||
                        (N == 2560 && K == 640) || (N == 320 && K == 10240) ||
                        (N == 640 && K == 2560);
    const int gm = d3 ? 16 : (narrow ? 2 : 4);
    for (int P : Ps) {
      std::vector<uint16_t> hA((size_t)N * K), hB((size_t)P * K);
      fill_bf16(hA.data(), hA.size(), 0xA5A5u + N + P);
      fill_bf16(hB.data(), hB.size(), 0x5A5Au + K + P);
      uint16_t *A, *B;
      float *C, *Cref;
      CK(hipMalloc(&A, hA.size() * 2));
      CK(hipMalloc(&B, hB.size() * 2));
      CK(hipMalloc(&C, (size_t)N * P * 4));
      CK(hipMalloc(&Cref, (size_t)N * P * 4));
      CK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
      CK(hipMemcpy(B, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
      float one = 1.f, zero = 0.f;
      rocblas_status rc = rocblas_gemm_ex(
          h, rocblas_operation_transpose, rocblas_operation_none, N, P, K,
          &one, A, rocblas_datatype_bf16_r, K, B, rocblas_datatype_bf16_r, K,
          &zero, Cref, rocblas_datatype_f32_r, N, Cref,
          rocblas_datatype_f32_r, N, rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, 0, 0);
      if (rc != rocblas_status_success) {
        printf("N=%d K=%d P=%d rocBLAS ref failed rc=%d\n", N, K, P, (int)rc);
        ++nbad;
        goto next;
      }
      CK(hipStreamSynchronize(st));
      {
        const unsigned grid = (unsigned)(((P + 127) / 128) * ((N + 255) / 256));
        if (d3)
          k_gemm_wmma<128, 256, 2, 8, 4, 2, 32, 512>
              <<<grid, 512, 0, st>>>(B, A, C, P, K, N, gm);
        else
          k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256>
              <<<grid, 256, 0, st>>>(B, A, C, P, K, N, gm);
        if (hipStreamSynchronize(st) != hipSuccess) {
          printf("N=%d K=%d P=%d kernel LAUNCH/RUN FAIL\n", N, K, P);
          (void)hipGetLastError();
          ++nbad;
          goto next;
        }
      }
      {
        std::vector<float> vC((size_t)N * P), vR((size_t)N * P);
        CK(hipMemcpy(vC.data(), C, vC.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(vR.data(), Cref, vR.size() * 4, hipMemcpyDeviceToHost));
        double maxabs = 0, maxrel = 0;
        for (size_t i = 0; i < vC.size(); ++i) {
          const double d = fabs((double)vC[i] - (double)vR[i]);
          maxabs = std::max(maxabs, d);
          maxrel = std::max(maxrel, d / std::max(1.0, (double)fabs(vR[i])));
        }
        printf("N=%-5d K=%-5d P=%-5d (%s gm%d) maxabs %.3e maxrel %.3e %s\n", N,
               K, P, d3 ? "d3" : "d9", gm, maxabs, maxrel,
               maxrel < 1e-2 ? "OK" : "BAD");
        if (maxrel >= 1e-2) ++nbad;
      }
    next:
      CK(hipFree(A));
      CK(hipFree(B));
      CK(hipFree(C));
      CK(hipFree(Cref));
    }
  }
  rocblas_destroy_handle(h);
  CK(hipStreamDestroy(st));
  printf(nbad ? "RESULT: %d BAD\n" : "RESULT: ALL OK\n", nbad);
  return nbad != 0;
}
