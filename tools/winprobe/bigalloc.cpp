// 探针8：单次大额 hipMalloc 的路由行为（>32GiB 是否落在 VRAM）
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  printf("FAIL %s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

__global__ void write_pattern(unsigned long long* p, size_t n, unsigned long long seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t j = i; j < n; j += (size_t)gridDim.x * blockDim.x)
    p[j] = seed ^ j;
}
__global__ void read_kernel(const float4* __restrict__ p, size_t n, float* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  for (size_t j = i; j < n; j += (size_t)gridDim.x * blockDim.x)
    acc += p[j].x + p[j].y + p[j].z + p[j].w;
  if (acc == 1234.5678f) out[0] = acc;
}

int main(int argc, char** argv) {
  size_t gib = (size_t)atoi(argv[1]);
  size_t bytes = gib << 30, N = bytes / 8;
  void* p = nullptr;
  hipError_t e = hipMalloc(&p, bytes);
  if (e != hipSuccess) { printf("hipMalloc %zu GiB FAIL: %s\n", gib, hipGetErrorString(e)); return 1; }
  write_pattern<<<4096, 256>>>((unsigned long long*)p, N, 42);
  CK(hipDeviceSynchronize());   // 若落到未支撑页，这里会炸
  float* out; CK(hipMalloc(&out, 4));
  read_kernel<<<4096, 256>>>((const float4*)p, bytes/16, out);
  CK(hipDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < 2; k++) read_kernel<<<4096, 256>>>((const float4*)p, bytes/16, out);
  CK(hipDeviceSynchronize());
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double bw = (double)bytes * 2 / s / 1e9;
  // 校验首尾
  unsigned long long h[64]; int bad = 0;
  for (int t = 0; t < 2; t++) {
    size_t off = t ? N - 64 : 0;
    CK(hipMemcpy(h, (unsigned long long*)p + off, 512, hipMemcpyDeviceToHost));
    for (int k = 0; k < 64; k++) if (h[k] != (42ull ^ (off + k))) { bad++; break; }
  }
  printf("%3zu GiB single alloc: BW = %6.1f GB/s, integrity %s\n",
         gib, bw, bad ? "CORRUPTED" : "OK");
  return bad != 0;
}
