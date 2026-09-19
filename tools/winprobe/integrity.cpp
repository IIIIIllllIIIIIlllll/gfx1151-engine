// 探针6（决定性）：kernel 写入真实 pattern（非 memset），
// 分配全过程监控带宽与数据完整性，排除零页/静默丢弃。
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstring>

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

static double bench(const void* p, size_t bytes) {
  size_t n = bytes / sizeof(float4);
  float* out; CK(hipMalloc(&out, 4));
  read_kernel<<<4096, 256>>>((const float4*)p, n, out);
  CK(hipDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < 2; k++) read_kernel<<<4096, 256>>>((const float4*)p, n, out);
  CK(hipDeviceSynchronize());
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  CK(hipFree(out));
  return (double)bytes * 2 / s / 1e9;
}

int main() {
  const size_t CHUNK = 4ull << 30;
  const size_t N = CHUNK / 8; // u64 count
  std::vector<void*> ptrs;
  for (int b = 0; b < 40; b++) {
    void* p = nullptr;
    if (hipMalloc(&p, CHUNK) != hipSuccess) break;
    write_pattern<<<4096, 256>>>((unsigned long long*)p, N, 0xABCDEF00 + b);
    CK(hipDeviceSynchronize());
    double w = bench(p, CHUNK);
    printf("block %2d: fresh-write read BW = %7.1f GB/s\n", b, w);
    ptrs.push_back(p);
  }
  printf("total allocated: %.1f GiB\n", ptrs.size() * CHUNK / 1073741824.0);

  // 复测带宽
  for (size_t i = 0; i < ptrs.size(); i += 4)
    printf("re-bench block %2zu: %7.1f GB/s\n", i, bench(ptrs[i], CHUNK));

  // 数据完整性：每块首尾各校验 64 个 u64
  unsigned long long hbuf[64];
  int bad = 0;
  for (size_t i = 0; i < ptrs.size(); i++) {
    for (int tail = 0; tail < 2; tail++) {
      size_t off = tail ? N - 64 : 0;
      CK(hipMemcpy(hbuf, (unsigned long long*)ptrs[i] + off, 512, hipMemcpyDeviceToHost));
      for (int k = 0; k < 64; k++)
        if (hbuf[k] != ((0xABCDEF00 + i) ^ (off + k))) { bad++; break; }
    }
  }
  printf("integrity: %s (%d bad)\n", bad ? "CORRUPTED" : "OK", bad);
  return 0;
}
