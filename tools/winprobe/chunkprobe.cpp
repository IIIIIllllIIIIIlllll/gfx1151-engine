// 探针2：分块 hipMalloc，确认 41 GiB 是单次分配上限还是进程总预算上限。
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

int main() {
  size_t freeB, totalB;
  hipMemGetInfo(&freeB, &totalB);
  printf("start: free = %.2f GiB / %.2f GiB\n", freeB/1073741824.0, totalB/1073741824.0);

  const size_t CHUNK = 4ull << 30; // 4 GiB/块
  std::vector<void*> ptrs;
  for (int i = 0; i < 30; i++) {
    void* p = nullptr;
    hipError_t e = hipMalloc(&p, CHUNK);
    if (e != hipSuccess) {
      hipMemGetInfo(&freeB, &totalB);
      printf("chunk %2d FAIL: %s | free now %.2f GiB | total alloc %.1f GiB\n",
             i, hipGetErrorString(e), freeB/1073741824.0, ptrs.size()*CHUNK/1073741824.0);
      break;
    }
    hipMemset(p, 0x5a, CHUNK); // 真实触碰，排除惰性
    ptrs.push_back(p);
    hipMemGetInfo(&freeB, &totalB);
    printf("chunk %2d OK, cum %.1f GiB, free %.2f GiB\n",
           i, ptrs.size()*CHUNK/1073741824.0, freeB/1073741824.0);
  }
  for (void* p : ptrs) hipFree(p);
  return 0;
}
