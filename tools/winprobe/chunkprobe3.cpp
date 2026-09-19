// 探针4：细粒度块（MiB 参数）新鲜进程总上限
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? (size_t)atoi(argv[1]) : 512;
  size_t chunk = mib << 20;
  std::vector<void*> ptrs;
  while (true) {
    void* p = nullptr;
    if (hipMalloc(&p, chunk) != hipSuccess) break;
    hipMemset(p, 0x5a, chunk);
    ptrs.push_back(p);
  }
  size_t freeB, totalB; hipMemGetInfo(&freeB, &totalB);
  double alloc = (double)ptrs.size() * chunk / 1073741824.0;
  printf("chunk %4zu MiB: allocated %.2f GiB in %zu blocks, free left %.2f GiB\n",
         mib, alloc, ptrs.size(), freeB/1073741824.0);
  return 0;
}
