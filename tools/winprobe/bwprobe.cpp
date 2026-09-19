// 探针5：分配 4 GiB 块直到失败，抽测首/中/尾块的 kernel 读带宽
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

__global__ void read_kernel(const float4* __restrict__ p, size_t n, float* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  for (size_t j = i; j < n; j += (size_t)gridDim.x * blockDim.x)
    acc += p[j].x + p[j].y + p[j].z + p[j].w;
  if (acc == 1234.5678f) out[0] = acc;
}

static double bw(const void* p, size_t bytes) {
  size_t n = bytes / sizeof(float4);
  float* out; 
  if (hipMalloc(&out, 4) != hipSuccess) { printf("  out alloc fail\n"); return -1; }
  read_kernel<<<4096, 256>>>((const float4*)p, n, out);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("  kernel FAIL: %s\n", hipGetErrorString(e)); return -1; }
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < 2; k++) read_kernel<<<4096, 256>>>((const float4*)p, n, out);
  e = hipDeviceSynchronize();
  if (e != hipSuccess) { printf("  kernel FAIL: %s\n", hipGetErrorString(e)); return -1; }
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  hipFree(out);
  return (double)bytes * 2 / s / 1e9;
}

int main() {
  const size_t CHUNK = 4ull << 30;
  std::vector<void*> ptrs;
  while (true) {
    void* p = nullptr;
    if (hipMalloc(&p, CHUNK) != hipSuccess) break;
    hipMemset(p, 0x5a, CHUNK);
    ptrs.push_back(p);
  }
  printf("allocated %zu blocks (%.1f GiB)\n", ptrs.size(), ptrs.size() * CHUNK / 1073741824.0);
  size_t idxs[] = {0, ptrs.size() / 4, ptrs.size() / 2, ptrs.size() * 3 / 4, ptrs.size() - 1};
  for (size_t i : idxs)
    printf("block %2zu: read BW = %.1f GB/s\n", i, bw(ptrs[i], CHUNK));
  return 0;
}
