// hcmix_proto.cu — HC mix-up GEMM + combine fusion (k_gemm_wmma Epi=1) vs the
// production pair: k_gemm_wmma<128,256,2,4,4,4,32,256> (N=10240, K=320, gm=2)
// writing fp32 G[P][10240], then k_gr_combine_b_hc_bf16 reading G + Rhat bf16.
// Checks the fused x (fp32 and bf16) is bit-identical, and times both.
// The kernel is #included from tools/gemm_wmma_kernel.inc (sed-extracted from
// src/gpu/parts/22_kernels_prefill.inc, see tools/gemm_wmma_driver.cu):
//   sed -n '/\[gemm-wmma-begin\]/,/\[gemm-wmma-end\]/p' src/gpu/parts/22_kernels_prefill.inc > tools/gemm_wmma_kernel.inc
// Build: hipcc -O3 --offload-arch=gfx1151 -o /tmp/hcmix_proto tools/hcmix_proto.cu
// Run:   /tmp/hcmix_proto [reps=20]     last line: HCMIX PROTO: PASS/FAIL
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      fprintf(stderr, "hip error %d (%s) at %s:%d\n", (int)e_,               \
              hipGetErrorString(e_), __FILE__, __LINE__);                    \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

using qw_shortx16 = __attribute__((ext_vector_type(16))) short;
using qw_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ qw_shortx16 qw_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  qw_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
// identical to src/gpu/parts/22_kernels_prefill.inc:4-16
__device__ inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
__device__ inline float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

#include "gemm_wmma_kernel.inc"

// verbatim copy of the production combine (22_kernels_prefill.inc)
__global__ void k_gr_combine_b_hc_bf16(
    const float* __restrict__ G, const uint16_t* __restrict__ Rhat,
    float* __restrict__ x, int d, int branches, int P,
    uint16_t* __restrict__ xbf16) {
  int t = blockIdx.x;
  if (t >= P) return;
  const float* Gt = G + (size_t)t * branches * d;
  const uint16_t* Rt = Rhat + (size_t)t * branches * d;
  float* xt = x + (size_t)t * d;
  uint16_t* x16t = xbf16 + (size_t)t * d;
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    float acc = 0.f;
    for (int b = 0; b < branches; b++) {
      float gate = 1.f / (1.f + expf(-Gt[b * d + c]));
      acc += gate * bf2f(Rt[b * d + c]);
    }
    float value = acc / branches;
    xt[c] = value;
    x16t[c] = f2bf(value);
  }
}

static uint32_t xs_state;
static uint32_t xs_next() {
  uint32_t x = xs_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  xs_state = x;
  return x;
}
static uint16_t h_f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static void fill(uint16_t* p, size_t n, uint32_t seed, float scale) {
  xs_state = seed;
  for (size_t i = 0; i < n; ++i)
    p[i] = h_f2bf(((float)((xs_next() >> 8) & 0xFFFF) / 32768.f - 1.f) * scale);
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 20;
  const int d = 2560, N = 4 * d, K = 320;
  const int Ps[] = {16384, 8192, 8199, 1024};
  const int PMAX = 16384 + 128;
  std::vector<uint16_t> hX((size_t)PMAX * K), hW((size_t)N * K),
      hR((size_t)PMAX * N);
  fill(hX.data(), hX.size(), 1, 1.f);
  fill(hW.data(), hW.size(), 2, 0.25f);  // G ~ O(1): sigmoid not saturated
  fill(hR.data(), hR.size(), 3, 2.f);
  uint16_t *X, *W, *R, *xb0, *xb1;
  float *G, *x0, *x1;
  CK(hipMalloc(&X, hX.size() * 2));
  CK(hipMalloc(&W, hW.size() * 2));
  CK(hipMalloc(&R, hR.size() * 2));
  CK(hipMalloc(&G, (size_t)PMAX * N * 4));
  CK(hipMalloc(&x0, (size_t)PMAX * d * 4));
  CK(hipMalloc(&x1, (size_t)PMAX * d * 4));
  CK(hipMalloc(&xb0, (size_t)PMAX * d * 2));
  CK(hipMalloc(&xb1, (size_t)PMAX * d * 2));
  CK(hipMemcpy(X, hX.data(), hX.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(W, hW.data(), hW.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(R, hR.data(), hR.size() * 2, hipMemcpyHostToDevice));
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  auto med = [&](auto&& f) {
    std::vector<float> ts(reps);
    f();
    CK(hipDeviceSynchronize());
    for (int r = 0; r < reps; ++r) {
      CK(hipEventRecord(e0));
      f();
      CK(hipEventRecord(e1));
      CK(hipEventSynchronize(e1));
      CK(hipEventElapsedTime(&ts[r], e0, e1));
    }
    std::sort(ts.begin(), ts.end());
    return ts[reps / 2];
  };
  bool ok = true;
  for (int P : Ps) {
    const unsigned grid = (unsigned)(((P + 127) / 128) * (N / 256));
    auto gemm = [&] {
      k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256>
          <<<grid, 256>>>(X, W, G, P, K, N, 2);
    };
    auto comb = [&] {
      k_gr_combine_b_hc_bf16<<<P, 256>>>(G, R, x0, d, 4, P, xb0);
    };
    auto fused = [&] {
      k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256, 1>
          <<<grid, 256>>>(X, W, x1, P, K, N, 2, R, xb1);
    };
    CK(hipMemset(x1, 0xff, (size_t)P * d * 4));
    CK(hipMemset(xb1, 0xff, (size_t)P * d * 2));
    gemm();
    comb();
    fused();
    CK(hipDeviceSynchronize());
    std::vector<uint32_t> a((size_t)P * d), b((size_t)P * d);
    std::vector<uint16_t> ab((size_t)P * d), bb((size_t)P * d);
    CK(hipMemcpy(a.data(), x0, a.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), x1, b.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ab.data(), xb0, ab.size() * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(bb.data(), xb1, bb.size() * 2, hipMemcpyDeviceToHost));
    size_t bad = 0, badb = 0, first = (size_t)-1;
    double ss = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) { if (first == (size_t)-1) first = i; ++bad; }
      if (ab[i] != bb[i]) ++badb;
      float f;
      memcpy(&f, &a[i], 4);
      ss += (double)f * f;
    }
    const float tg = med(gemm), tc = med(comb), tf = med(fused);
    const double gb_c = ((double)P * N * 4 + (double)P * N * 2 + (double)P * d * 6) / 1e9;
    printf("P=%5d  gemm %.3f ms  combine %.3f ms (%.0f GB/s)  sum %.3f | fused %.3f ms  "
           "x%.2f  save %.3f ms | diff fp32 %zu bf16 %zu rms(x)=%.3f\n",
           P, tg, tc, gb_c / (tc * 1e-3), tg + tc, tf, (tg + tc) / tf, tg + tc - tf,
           bad, badb, std::sqrt(ss / a.size()));
    {
      // tile sweep of the fused kernel (all must stay bit-identical to x0)
      auto check = [&](const char* name, auto&& f) {
        CK(hipMemset(x1, 0xff, (size_t)P * d * 4));
        CK(hipMemset(xb1, 0xff, (size_t)P * d * 2));
        f();
        CK(hipDeviceSynchronize());
        std::vector<uint32_t> c((size_t)P * d);
        std::vector<uint16_t> cb((size_t)P * d);
        CK(hipMemcpy(c.data(), x1, c.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(cb.data(), xb1, cb.size() * 2, hipMemcpyDeviceToHost));
        size_t nb = 0;
        for (size_t i = 0; i < c.size(); ++i) nb += (c[i] != a[i]) + (cb[i] != ab[i]);
        const float t = med(f);
        printf("   %-34s %.3f ms  x%.2f vs pair  diff %zu\n", name, t, (tg + tc) / t, nb);
        if (nb) ok = false;
      };
#define HCFG(BM, BP, MW, PW, WM, KST, NT, GM)                                        \
  check(#BM "x" #BP " w" #MW "x" #PW " f" #WM "x4 k" #KST " t" #NT " gm" #GM, [&] {   \
    k_gemm_wmma<BM, BP, MW, PW, WM, 4, KST, NT, 1>                                   \
        <<<(unsigned)(((P + BM - 1) / BM) * (N / BP)), NT>>>(X, W, x1, P, K, N, GM, R, \
                                                             xb1);                   \
  })
      HCFG(128, 256, 2, 4, 4, 32, 256, 1);
      HCFG(128, 128, 4, 2, 2, 32, 256, 2);
      HCFG(128, 128, 4, 2, 2, 32, 256, 1);
      HCFG(64, 128, 2, 2, 2, 32, 128, 2);
      HCFG(64, 128, 2, 2, 2, 32, 128, 1);
      HCFG(64, 256, 1, 4, 4, 32, 128, 2);
      HCFG(64, 64, 4, 1, 1, 32, 128, 2);
      HCFG(128, 64, 4, 1, 2, 32, 128, 2);
      HCFG(128, 64, 4, 1, 2, 32, 128, 1);
      HCFG(64, 128, 2, 2, 2, 64, 128, 1);
      HCFG(64, 64, 4, 1, 1, 64, 128, 1);
      HCFG(128, 64, 4, 1, 2, 64, 128, 1);
      HCFG(32, 64, 2, 1, 1, 32, 64, 1);
#undef HCFG
    }
    if (bad || badb) {
      ok = false;
      if (first != (size_t)-1) {
        float fa, fb;
        memcpy(&fa, &a[first], 4);
        memcpy(&fb, &b[first], 4);
        printf("   first diff at t=%zu c=%zu: %.9g vs %.9g\n", first / d, first % d, fa, fb);
      }
    }
  }
  printf("HCMIX PROTO: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
