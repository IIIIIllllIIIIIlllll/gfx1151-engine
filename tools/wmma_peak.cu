// wmma_peak.cu — raw bf16 WMMA issue-rate ceiling on gfx1151
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
using sx16 = __attribute__((ext_vector_type(16))) short;
using fx8 = __attribute__((ext_vector_type(8))) float;

__global__ void __launch_bounds__(256) k_peak(const short* seed, float* out,
                                              int iters) {
  sx16 a, b;
  short t[16];
  for (int i = 0; i < 16; ++i) t[i] = seed[(threadIdx.x + i) & 63];
  memcpy(&a, t, 32);
  for (int i = 0; i < 16; ++i) t[i] = seed[(threadIdx.x + i + 32) & 63];
  memcpy(&b, t, 32);
  fx8 acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[i]);
  }
  float s = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) s += acc[i][e];
  if (s == 12345.678f) out[threadIdx.x] = s;
}

int main() {
  short* seed;
  float* out;
  hipMalloc(&seed, 1024);
  hipMalloc(&out, 4096);
  hipMemset(seed, 0x3c, 1024);
  const int iters = 20000;
  for (int grid : {20, 40, 80, 160, 320}) {
    k_peak<<<grid, 256>>>(seed, out, iters);
    hipDeviceSynchronize();
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0);
    k_peak<<<grid, 256>>>(seed, out, iters);
    hipEventRecord(e1);
    hipDeviceSynchronize();
    float ms;
    hipEventElapsedTime(&ms, e0, e1);
    double flops = (double)grid * 8 * iters * 8 * (16.0 * 16 * 16 * 2);
    printf("grid %3d: %.3f ms  %.1f TFLOPS\n", grid, ms, flops / ms / 1e9);
  }
  return 0;
}
