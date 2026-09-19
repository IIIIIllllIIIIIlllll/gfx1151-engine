// 探针3：不同块大小 / 混合大小的总分配上限
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>

static void scan(size_t chunk_gib) {
  size_t chunk = chunk_gib << 30;
  std::vector<void*> ptrs;
  while (true) {
    void* p = nullptr;
    if (hipMalloc(&p, chunk) != hipSuccess) break;
    hipMemset(p, 0x5a, chunk);
    ptrs.push_back(p);
  }
  size_t freeB, totalB; hipMemGetInfo(&freeB, &totalB);
  printf("chunk %2zu GiB: total %3zu GiB, free left %.2f GiB\n",
         chunk_gib, ptrs.size()*chunk_gib, freeB/1073741824.0);
  for (void* p : ptrs) hipFree(p);
  hipMemGetInfo(&freeB, &totalB);
  printf("  after free: %.2f GiB\n", freeB/1073741824.0);
}

int main(int argc, char** argv) {
  if (argc > 1) { scan((size_t)atoi(argv[1])); return 0; }
  scan(8); scan(4); scan(2); scan(1);
  // 混合：先 4 GiB 到顶，再 1 GiB 续
  std::vector<void*> ptrs;
  auto fill = [&](size_t bytes) {
    while (true) { void* p=nullptr; if (hipMalloc(&p, bytes)!=hipSuccess) break;
      hipMemset(p,0x5a,bytes); ptrs.push_back(p); }
  };
  fill(4ull<<30); fill(1ull<<30); fill(256ull<<20);
  size_t freeB,totalB; hipMemGetInfo(&freeB,&totalB);
  printf("mixed fill: %zu allocs, free left %.2f GiB\n", ptrs.size(), freeB/1073741824.0);
  return 0;
}
