// q8g32_gemv_bench.cu — decode (P=1) GEMV variants for q8g32 planar weights
// (GGUF Q8_0 dense, hgn.h dtype 8) on the G3 decode shapes.
//   q8g32_gemv_bench [iters]
// Each shape cycles over enough weight copies (> 64 MB) that the 32 MB MALL
// cannot hold them (production decode streams ~4.5 GB per token). Prints
// us/call and GB/s per variant; results checked vs the production kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// kernels: the q8g32 section of 20_kernels_gemv.inc, cut out at build time:
//   sed -n "/^\/\/ --- q8g32 planar/,/^\/\/ --- BF16 GEMV for few-row/p" \
//     src/gpu/parts/20_kernels_gemv.inc > build/q8g32_part.inc
//   hipcc -O3 -std=c++17 --offload-arch=gfx1151 -Ibuild tools/q8g32_gemv_bench.cu
#include "q8g32_part.inc"

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 200;
  struct Shape { int rows, cols; const char* what; };
  const Shape shapes[] = {{2560, 6144, "out_proj/o_proj"}, {2560, 640, "shexp down"},
                          {640, 2560, "shexp gate/up"},    {320, 10240, "HC down"},
                          {10240, 320, "HC up"},           {512, 2560, "k/v proj"},
                          {10240, 2560, "in_proj_qkv"},    {6144, 2560, "in_proj_z"},
                          {12288, 2560, "q_proj"}};
  std::mt19937 rng(1);
  bool allok = true;
  for (const Shape& sh : shapes) {
    const size_t R = sh.rows, C = sh.cols, qb = R * C, sb = R * (C / 32) * 2, wb = qb + sb;
    const int ncopy = (int)std::max<size_t>(1, (96u << 20) / wb + 1);
    std::vector<uint8_t> h(wb);
    for (size_t i = 0; i < qb; i++) h[i] = (uint8_t)(rng() & 0xFF);
    __half* hs = (__half*)(h.data() + qb);
    for (size_t i = 0; i < R * (C / 32); i++) hs[i] = __float2half(0.001f * (1 + (rng() % 100)));
    uint8_t* dw;
    CK(hipMalloc(&dw, wb * ncopy));
    for (int c = 0; c < ncopy; c++) CK(hipMemcpy(dw + wb * c, h.data(), wb, hipMemcpyHostToDevice));
    std::vector<float> x(C);
    for (auto& v : x) v = (float)((int)(rng() % 2001) - 1000) / 1000.f;
    float *dx, *dy, *dr;
    CK(hipMalloc(&dx, C * 4));
    CK(hipMalloc(&dy, R * 4));
    CK(hipMalloc(&dr, R * 4));
    CK(hipMemcpy(dx, x.data(), C * 4, hipMemcpyHostToDevice));
    auto W = [&](int c) { return (const int8_t*)(dw + wb * c); };
    auto S = [&](int c) { return (const __half*)(dw + wb * c + qb); };
    auto prod = [&](int c, float* y) {
      k_q8g32_gemv_mr<1><<<(unsigned)((R + 31) / 32), 512>>>(W(c), S(c), dx, y, R, C, C);
    };
    prod(0, dr);
    CK(hipDeviceSynchronize());
    std::vector<float> ref(R), got(R);
    CK(hipMemcpy(ref.data(), dr, R * 4, hipMemcpyDeviceToHost));
    auto timeit = [&](auto&& f) {
      for (int i = 0; i < 4; i++) f(i % ncopy);
      hipEvent_t a, b;
      CK(hipEventCreate(&a));
      CK(hipEventCreate(&b));
      CK(hipEventRecord(a, 0));
      for (int i = 0; i < iters; i++) f(i % ncopy);
      CK(hipEventRecord(b, 0));
      CK(hipEventSynchronize(b));
      float ms;
      CK(hipEventElapsedTime(&ms, a, b));
      return ms * 1e3 / iters;
    };
    printf("%-16s %5zux%-5zu %5.2f MB x%d:", sh.what, R, C, wb / 1e6, ncopy);
    const double t0 = timeit([&](int c) { prod(c, dy); });
    printf("  prod %6.1fus %3.0f", t0, wb / t0 / 1e3);
    auto check = [&](const char* nm, auto&& f) {
      CK(hipMemset(dy, 0xFF, R * 4));
      f(0);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(got.data(), dy, R * 4, hipMemcpyDeviceToHost));
      double e = 0, n = 0;
      for (size_t i = 0; i < R; i++) {
        e += (got[i] - ref[i]) * (double)(got[i] - ref[i]);
        n += (double)ref[i] * ref[i];
      }
      const bool ok = std::isfinite(e) && sqrt(e / n) < 1e-5;
      allok &= ok;
      const double t = timeit(f);
      printf(" | %s %6.1fus %3.0f%s", nm, t, wb / t / 1e3, ok ? "" : " BAD");
    };
    check("L8/256", [&](int c) { q8g32_gemv_small<8, 256>(W(c), S(c), dx, dy, R, C, 0); });
    check("L16/256", [&](int c) { q8g32_gemv_small<16, 256>(W(c), S(c), dx, dy, R, C, 0); });
    check("L32/256", [&](int c) { q8g32_gemv_small<32, 256>(W(c), S(c), dx, dy, R, C, 0); });
    check("L16/512", [&](int c) { q8g32_gemv_small<16, 512>(W(c), S(c), dx, dy, R, C, 0); });
    check("L32/128", [&](int c) { q8g32_gemv_small<32, 128>(W(c), S(c), dx, dy, R, C, 0); });
    printf("\n");
    CK(hipFree(dw));
    CK(hipFree(dx));
    CK(hipFree(dy));
    CK(hipFree(dr));
  }
  // multi-row (spec verify P=GAMMA+1): k_q8g32_gemv_mr<P> vs k_q8g32_gemv_mlpr
  printf("\nmulti-row P=4 (fp32 x | bf16 x)\n");
  for (const Shape& sh : shapes) {
    constexpr int PP = 4;
    const size_t R = sh.rows, C = sh.cols, qb = R * C, sb = R * (C / 32) * 2, wb = qb + sb;
    const int ncopy = (int)std::max<size_t>(1, (96u << 20) / wb + 1);
    std::vector<uint8_t> h(wb);
    for (size_t i = 0; i < qb; i++) h[i] = (uint8_t)(rng() & 0xFF);
    __half* hs = (__half*)(h.data() + qb);
    for (size_t i = 0; i < R * (C / 32); i++) hs[i] = __float2half(0.001f * (1 + (rng() % 100)));
    uint8_t* dw;
    CK(hipMalloc(&dw, wb * ncopy));
    for (int c = 0; c < ncopy; c++) CK(hipMemcpy(dw + wb * c, h.data(), wb, hipMemcpyHostToDevice));
    std::vector<float> x(PP * C);
    std::vector<uint16_t> xb(PP * C);
    for (size_t i = 0; i < x.size(); i++) {
      x[i] = (float)((int)(rng() % 2001) - 1000) / 1000.f;
      uint32_t u;
      memcpy(&u, &x[i], 4);
      xb[i] = (uint16_t)(u >> 16);
    }
    float *dx, *dy, *dr;
    uint16_t* dxb;
    CK(hipMalloc(&dx, PP * C * 4));
    CK(hipMalloc(&dxb, PP * C * 2));
    CK(hipMalloc(&dy, PP * R * 4));
    CK(hipMalloc(&dr, PP * R * 4));
    CK(hipMemcpy(dx, x.data(), PP * C * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dxb, xb.data(), PP * C * 2, hipMemcpyHostToDevice));
    auto W = [&](int c) { return (const int8_t*)(dw + wb * c); };
    auto S = [&](int c) { return (const __half*)(dw + wb * c + qb); };
    auto timeit = [&](auto&& f) {
      for (int i = 0; i < 4; i++) f(i % ncopy);
      hipEvent_t a, b;
      CK(hipEventCreate(&a));
      CK(hipEventCreate(&b));
      CK(hipEventRecord(a, 0));
      for (int i = 0; i < iters; i++) f(i % ncopy);
      CK(hipEventRecord(b, 0));
      CK(hipEventSynchronize(b));
      float ms;
      CK(hipEventElapsedTime(&ms, a, b));
      return ms * 1e3 / iters;
    };
    std::vector<float> ref(PP * R), got(PP * R);
    printf("%-16s %5zux%-5zu:", sh.what, R, C);
    for (int bf = 0; bf < 2; bf++) {
      auto prod = [&](int c, float* y) {
        if (bf)
          k_q8g32_gemv_mr<PP, true><<<(unsigned)((R + 31) / 32), 512>>>(W(c), S(c), dxb, y, R, C, C);
        else
          k_q8g32_gemv_mr<PP, false><<<(unsigned)((R + 31) / 32), 512>>>(W(c), S(c), dx, y, R, C, C);
      };
      prod(0, dr);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(ref.data(), dr, PP * R * 4, hipMemcpyDeviceToHost));
      const double t0 = timeit([&](int c) { prod(c, dy); });
      printf(" %s prod %6.1fus", bf ? "| bf16" : "fp32", t0);
      auto check = [&](const char* nm, auto&& f) {
        CK(hipMemset(dy, 0xFF, PP * R * 4));
        f(0);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(got.data(), dy, PP * R * 4, hipMemcpyDeviceToHost));
        double e = 0, n = 0;
        for (size_t i = 0; i < PP * R; i++) {
          e += (got[i] - ref[i]) * (double)(got[i] - ref[i]);
          n += (double)ref[i] * ref[i];
        }
        const bool ok = std::isfinite(e) && sqrt(e / n) < 1e-5;
        allok &= ok;
        const double t = timeit(f);
        printf(" %s %6.1f%s", nm, t, ok ? "" : " BAD");
      };
      const void* xp = bf ? (const void*)dxb : (const void*)dx;
      if (bf) {
        check("L8", [&](int c) { q8g32_gemv_multi<PP, true, 8, 256>(W(c), S(c), xp, dy, R, C, C, 0); });
        check("L16", [&](int c) { q8g32_gemv_multi<PP, true, 16, 256>(W(c), S(c), xp, dy, R, C, C, 0); });
        check("L32", [&](int c) { q8g32_gemv_multi<PP, true, 32, 256>(W(c), S(c), xp, dy, R, C, C, 0); });
        check("B128", [&](int c) { k_q8g32_gemv_brow<PP, true, 128><<<(unsigned)R, 128>>>(W(c), S(c), xp, dy, R, C, C); });
        check("B256", [&](int c) { k_q8g32_gemv_brow<PP, true, 256><<<(unsigned)R, 256>>>(W(c), S(c), xp, dy, R, C, C); });
      } else {
        check("L8", [&](int c) { q8g32_gemv_multi<PP, false, 8, 256>(W(c), S(c), xp, dy, R, C, C, 0); });
        check("L16", [&](int c) { q8g32_gemv_multi<PP, false, 16, 256>(W(c), S(c), xp, dy, R, C, C, 0); });
        check("L32", [&](int c) { q8g32_gemv_multi<PP, false, 32, 256>(W(c), S(c), xp, dy, R, C, C, 0); });
        check("B128", [&](int c) { k_q8g32_gemv_brow<PP, false, 128><<<(unsigned)R, 128>>>(W(c), S(c), xp, dy, R, C, C); });
        check("B256", [&](int c) { k_q8g32_gemv_brow<PP, false, 256><<<(unsigned)R, 256>>>(W(c), S(c), xp, dy, R, C, C); });
      }
    }
    printf("  (%.2f MB)\n", wb / 1e6);
    CK(hipFree(dw));
    CK(hipFree(dx));
    CK(hipFree(dxb));
    CK(hipFree(dy));
    CK(hipFree(dr));
  }
  printf("%s\n", allok ? "PASS" : "FAIL");
  return allok ? 0 : 1;
}
