// 探针7：分配 N 块 → 写 pattern → 等信号文件 → 校验完整性（双进程并发用）
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <sys/stat.h>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  printf("FAIL %s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

__global__ void write_pattern(unsigned long long* p, size_t n, unsigned long long seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t j = i; j < n; j += (size_t)gridDim.x * blockDim.x)
    p[j] = seed ^ j;
}

static bool exists(const char* p) { struct stat s; return stat(p, &s) == 0; }

int main(int argc, char** argv) {
  int blocks = argc > 1 ? atoi(argv[1]) : 10;      // 每进程块数（4 GiB/块）
  const char* sig = argc > 2 ? argv[2] : "C:/temp/go.sig";
  const size_t CHUNK = 4ull << 30, N = CHUNK / 8;
  std::vector<void*> ptrs;
  for (int b = 0; b < blocks; b++) {
    void* p = nullptr;
    if (hipMalloc(&p, CHUNK) != hipSuccess) { printf("alloc fail at %d\n", b); break; }
    write_pattern<<<4096, 256>>>((unsigned long long*)p, N, 0xABCDEF00 + b);
    CK(hipDeviceSynchronize());
    ptrs.push_back(p);
  }
  printf("allocated %zu blocks (%.1f GiB), waiting for signal %s\n",
         ptrs.size(), ptrs.size()*CHUNK/1073741824.0, sig);
  fflush(stdout);
  while (!exists(sig)) std::this_thread::sleep_for(std::chrono::milliseconds(200));
  std::this_thread::sleep_for(std::chrono::seconds(2)); // 让另一进程也进入校验
  unsigned long long h[64];
  int bad = 0;
  for (size_t i = 0; i < ptrs.size(); i++)
    for (int t = 0; t < 2; t++) {
      size_t off = t ? N - 64 : 0;
      CK(hipMemcpy(h, (unsigned long long*)ptrs[i] + off, 512, hipMemcpyDeviceToHost));
      for (int k = 0; k < 64; k++)
        if (h[k] != ((0xABCDEF00 + i) ^ (off + k))) { bad++; break; }
    }
  printf("integrity after concurrent hold: %s (%d bad of %zu checks)\n",
         bad ? "CORRUPTED" : "OK", bad, ptrs.size()*2);
  return bad != 0;
}
