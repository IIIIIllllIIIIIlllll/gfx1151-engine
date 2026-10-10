// brow_bench.cu — q8g32 bf16-x 长行 GEMV：k_q8g32_gemv_brow vs brow_r<P,128,4>，P=1..8。
// 用来定 gemv_multi_q4cp 里 brow_r 的 P 上限（大 P 时 brow_r 的 xv[P][4] 寄存器压力）。
// 形状取 HQ 权重在 spec verify 里走这条路的两类：HC down 320x10240、out_proj 2560x6144。
// 权重轮换多份拷贝（>200 MB）以模拟 decode 的冷权重 DRAM 流，而非 L2/MALL 命中。
// 同时逐 bit 比对 brow_r 与 brow（mism 必须为 0）。
// 编译/运行：hipcc -O3 --offload-arch=gfx1151 -I src/gpu tools/brow_bench.cu \
//              -lrocblas -lhipblaslt -o build/brow_bench && build/brow_bench
#define main qwenox_real_main
#include "qwenox.cpp"
#undef main

#include <random>
#include <vector>

#define BCK(x)                                                              \
  do {                                                                      \
    hipError_t e_ = (x);                                                    \
    if (e_ != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),     \
              __FILE__, __LINE__);                                          \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

static uint16_t bf16_of(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

template <int P>
static void run_brow(bool tiled, const int8_t* q, const __half* s, const uint16_t* x,
                     float* y, uint64_t rows, uint64_t cols) {
  if (tiled)
    k_q8g32_gemv_brow_r<P, 128, 4><<<(unsigned)((rows + 3) / 4), 128>>>(q, s, x, y, rows,
                                                                        cols, cols);
  else
    k_q8g32_gemv_brow<P, true, 128><<<(unsigned)rows, 128>>>(q, s, x, y, rows, cols, cols);
}
static void run_p(int P, bool tiled, const int8_t* q, const __half* s, const uint16_t* x,
                  float* y, uint64_t rows, uint64_t cols) {
  switch (P) {
#define C(PP) case PP: run_brow<PP>(tiled, q, s, x, y, rows, cols); break;
    C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
  }
}

int main() {
  std::mt19937 rng(0xb20);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  struct Shape { const char* name; uint64_t rows, cols; };
  const Shape shapes[] = {{"hc_down 320x10240", 320, 10240}, {"out_proj 2560x6144", 2560, 6144}};
  int bad = 0;
  for (const Shape& sh : shapes) {
    const uint64_t rows = sh.rows, cols = sh.cols, gpr = cols / 32;
    const size_t qb = rows * cols, sb = rows * gpr * 2;
    const int ncopy = (int)std::max<size_t>(4, (size_t)(256ull << 20) / (qb + sb));
    std::vector<int8_t> hq(qb);
    for (auto& v : hq) v = (int8_t)rng();
    std::vector<__half> hs(rows * gpr);
    for (auto& v : hs) v = __float2half(0.01f + 0.005f * U(rng));
    std::vector<uint16_t> hx(8 * cols);
    for (auto& v : hx) v = bf16_of(U(rng));
    std::vector<int8_t*> dq(ncopy);
    std::vector<__half*> ds(ncopy);
    for (int c = 0; c < ncopy; c++) {
      BCK(hipMalloc(&dq[c], qb));
      BCK(hipMalloc(&ds[c], sb));
      BCK(hipMemcpy(dq[c], hq.data(), qb, hipMemcpyHostToDevice));
      BCK(hipMemcpy(ds[c], hs.data(), sb, hipMemcpyHostToDevice));
    }
    uint16_t* dx;
    float *dy0, *dy1;
    BCK(hipMalloc(&dx, hx.size() * 2));
    BCK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    BCK(hipMalloc(&dy0, 8 * rows * 4));
    BCK(hipMalloc(&dy1, 8 * rows * 4));
    printf("== %s  (%d weight copies, %.0f MB)\n", sh.name, ncopy,
           ncopy * (qb + sb) / 1048576.0);
    printf("   P   brow_us  brow_r_us  r/brow   mism\n");
    for (int P = 1; P <= 8; P++) {
      auto timeit = [&](bool tiled) {
        hipEvent_t e0, e1;
        BCK(hipEventCreate(&e0));
        BCK(hipEventCreate(&e1));
        for (int c = 0; c < ncopy; c++) run_p(P, tiled, dq[c], ds[c], dx, dy0, rows, cols);
        const int iters = 4 * ncopy;
        BCK(hipEventRecord(e0));
        for (int i = 0; i < iters; i++)
          run_p(P, tiled, dq[i % ncopy], ds[i % ncopy], dx, dy0, rows, cols);
        BCK(hipEventRecord(e1));
        BCK(hipEventSynchronize(e1));
        float ms = 0;
        BCK(hipEventElapsedTime(&ms, e0, e1));
        BCK(hipEventDestroy(e0));
        BCK(hipEventDestroy(e1));
        return ms * 1000.0 / iters;
      };
      const double tb = timeit(false), tr = timeit(true);
      size_t mism = 0;
      for (uint64_t rr : {rows, rows - 1}) {
        BCK(hipMemset(dy0, 0, 8 * rows * 4));
        BCK(hipMemset(dy1, 0, 8 * rows * 4));
        run_p(P, false, dq[0], ds[0], dx, dy0, rr, cols);
        run_p(P, true, dq[0], ds[0], dx, dy1, rr, cols);
        BCK(hipDeviceSynchronize());
        std::vector<float> y0(8 * rows), y1(8 * rows);
        BCK(hipMemcpy(y0.data(), dy0, y0.size() * 4, hipMemcpyDeviceToHost));
        BCK(hipMemcpy(y1.data(), dy1, y1.size() * 4, hipMemcpyDeviceToHost));
        for (uint64_t p = 0; p < (uint64_t)P; p++)
          for (uint64_t i = 0; i < rr; i++)
            mism += memcmp(&y0[p * rr + i], &y1[p * rr + i], 4) != 0;
      }
      if (mism) bad = 1;
      printf("   %d  %8.1f  %9.1f  %6.2f  %5zu\n", P, tb, tr, tr / tb, mism);
    }
    for (int c = 0; c < ncopy; c++) { BCK(hipFree(dq[c])); BCK(hipFree(ds[c])); }
    BCK(hipFree(dx)); BCK(hipFree(dy0)); BCK(hipFree(dy1));
  }
  puts(bad ? "FAIL (brow_r != brow)" : "PASS (bit-exact)");
  return bad;
}
