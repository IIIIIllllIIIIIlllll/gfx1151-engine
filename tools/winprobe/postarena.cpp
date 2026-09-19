// 探针9：大额 arena 之后，小额 hipMalloc 落在哪（VRAM 还是 shared memory）？
// 用法: postarena <arena_GiB> [small_MiB] [count]
// 判定: 小块做 kernel 读写校验 + 带宽实测; 若落到 shared/未支撑页会在 sync 时炸
// 或带宽暴跌. 另测 hipGraph instantiate 在 arena 之后是否仍可用.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

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
  size_t arena_gib = (size_t)atoi(argv[1]);
  size_t small_mib = argc > 2 ? (size_t)atoi(argv[2]) : 32;
  int count = argc > 3 ? atoi(argv[3]) : 64;

  // 1) arena
  void* arena = nullptr;
  CK(hipMalloc(&arena, arena_gib << 30));
  printf("arena %zu GiB OK\n", arena_gib);

  // 2) small buffers after arena
  std::vector<void*> ptrs;
  size_t sb = small_mib << 20;
  for (int i = 0; i < count; i++) {
    void* p = nullptr;
    hipError_t e = hipMalloc(&p, sb);
    if (e != hipSuccess) {
      printf("small #%d hipMalloc FAIL: %s\n", i, hipGetErrorString(e));
      break;
    }
    ptrs.push_back(p);
  }
  printf("allocated %d x %zu MiB after arena\n", (int)ptrs.size(), small_mib);

  // 3) integrity + per-buffer bandwidth (write via kernel, read via kernel)
  float* out; CK(hipMalloc(&out, 4));
  int bad = 0;
  double tsum = 0;
  size_t N = sb / 8;
  for (size_t i = 0; i < ptrs.size(); i++) {
    write_pattern<<<512, 256>>>((unsigned long long*)ptrs[i], N, 0xC0FFEE + i);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("small #%zu kernel write FAIL: %s\n", i, hipGetErrorString(e));
      bad++;
      continue;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < 4; k++)
      read_kernel<<<512, 256>>>((const float4*)ptrs[i], sb / 16, out);
    CK(hipDeviceSynchronize());
    tsum += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    unsigned long long h[8];
    CK(hipMemcpy(h, ptrs[i], 64, hipMemcpyDeviceToHost));
    for (int k = 0; k < 8; k++)
      if (h[k] != ((0xC0FFEE + i) ^ (size_t)k)) { bad++; break; }
  }
  if (!ptrs.empty())
    printf("small-buffer read BW = %.1f GB/s, integrity %s\n",
           (double)sb * 4 * ptrs.size() / tsum / 1e9, bad ? "CORRUPTED" : "OK");

  // 4) pointer attributes of one small buffer (driver may still say Device)
  if (!ptrs.empty()) {
    hipPointerAttribute_t at;
    if (hipPointerGetAttributes(&at, ptrs[0]) == hipSuccess)
      printf("ptr attr: type=%d (0=unregistered/1=host/2=device) device=%d\n",
             (int)at.type, at.device);
  }

  // 5) hipGraph instantiate after arena
  hipGraph_t g = nullptr;
  hipGraphNode_t node = nullptr;
  CK(hipGraphCreate(&g, 0));
  hipMemsetParams mp{};
  mp.dst = ptrs.empty() ? arena : ptrs[0];
  mp.elementSize = 1;
  mp.width = 4096;
  mp.height = 1;
  mp.value = 0;
  CK(hipGraphAddMemsetNode(&node, g, nullptr, 0, &mp));
  hipGraphExec_t ge = nullptr;
  CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
  CK(hipGraphLaunch(ge, hipStreamLegacy));
  CK(hipDeviceSynchronize());
  printf("hipGraph after arena: OK\n");
  printf("POSTARENA PASS\n");
  return bad != 0;
}
