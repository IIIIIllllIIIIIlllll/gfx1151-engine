// Windows 移植探针：验证 gfx1151 在 Windows HIP 上的大显存分配与带宽。
// 1) hipMalloc 逐级加大 (GiB)，测 kernel 直读带宽
// 2) hipHostRegister 锁页主机内存逐级加大，测 zero-copy 直读带宽（原引擎架构路径）
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("  FAIL %s: %s\n", #x, hipGetErrorString(e)); return false; } } while (0)

__global__ void read_kernel(const float4* __restrict__ p, size_t n, float* out) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  // 每线程跨步读，覆盖整段
  for (size_t j = i; j < n; j += (size_t)gridDim.x * blockDim.x)
    acc += p[j].x + p[j].y + p[j].z + p[j].w;
  if (acc == 1234.5678f) out[0] = acc; // 防优化，实际不触发
}

static double bench_read(const void* p, size_t bytes, int iters = 3) {
  size_t n = bytes / sizeof(float4);
  float* out; hipMalloc(&out, 4);
  read_kernel<<<4096, 256>>>((const float4*)p, n, out);
  hipDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < iters; k++)
    read_kernel<<<4096, 256>>>((const float4*)p, n, out);
  hipDeviceSynchronize();
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  hipFree(out);
  return (double)bytes * iters / sec / 1e9; // GB/s
}

static bool probe_device(size_t gib) {
  size_t bytes = gib << 30;
  void* p = nullptr;
  hipError_t e = hipMalloc(&p, bytes);
  if (e != hipSuccess) { printf("  hipMalloc %zu GiB FAIL: %s\n", gib, hipGetErrorString(e)); return false; }
  e = hipMemset(p, 0x5a, bytes);
  if (e != hipSuccess) { printf("  hipMemset %zu GiB FAIL: %s\n", gib, hipGetErrorString(e)); hipFree(p); return false; }
  double bw = bench_read(p, bytes);
  printf("  hipMalloc %3zu GiB OK, kernel read BW = %.1f GB/s\n", gib, bw);
  hipFree(p);
  return true;
}

static bool probe_host_register(size_t gib) {
  size_t bytes = gib << 30;
  void* h = malloc(bytes);
  if (!h) { printf("  malloc %zu GiB host FAIL\n", gib); return false; }
  memset(h, 1, bytes < (64u<<20) ? bytes : (64u<<20)); // 触碰一部分即可
  hipError_t e = hipHostRegister(h, bytes, hipHostRegisterDefault);
  if (e != hipSuccess) { printf("  hipHostRegister %zu GiB FAIL: %s\n", gib, hipGetErrorString(e)); free(h); return false; }
  void* d = nullptr;
  e = hipHostGetDevicePointer(&d, h, 0);
  if (e != hipSuccess) { printf("  hipHostGetDevicePointer %zu GiB FAIL: %s\n", gib, hipGetErrorString(e)); hipHostUnregister(h); free(h); return false; }
  double bw = bench_read(d, bytes);
  printf("  hipHostRegister %3zu GiB OK, zero-copy read BW = %.1f GB/s\n", gib, bw);
  hipHostUnregister(h);
  free(h);
  return true;
}

int main(int argc, char** argv) {
  int dev = 0; CK(hipGetDevice(&dev));
  hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, dev));
  printf("device: %s, totalGlobalMem = %.2f GiB\n", prop.name, prop.totalGlobalMem / 1073741824.0);
  size_t freeB, totalB; CK(hipMemGetInfo(&freeB, &totalB));
  printf("hipMemGetInfo: free = %.2f GiB, total = %.2f GiB\n", freeB / 1073741824.0, totalB / 1073741824.0);

  printf("[device memory probe]\n");
  for (size_t g : {1, 8, 32, 68, 90}) if (!probe_device(g)) break;

  printf("[host register probe]\n");
  for (size_t g : {1, 8, 32, 65}) if (!probe_host_register(g)) break;

  // 汇总可用上限：逐级逼近找出 hipMalloc 实际天花板
  printf("[device ceiling scan]\n");
  size_t lo = 0, hi = 100; // GiB
  while (hi - lo > 2) {
    size_t mid = (lo + hi) / 2;
    void* p = nullptr;
    if (hipMalloc(&p, mid << 30) == hipSuccess) { hipFree(p); lo = mid; }
    else hi = mid;
  }
  printf("  max hipMalloc ~= %zu GiB\n", lo);
  return 0;
}
