// gemm_lt_bench.cu — per-shape speed trial for extending k_gemm_wmma coverage.
// Benches, for each (N,K,P) triple: the production hipBLASLt path (heuristic,
// index 4 with validation + fallback scan, 64MB workspace — mirrors
// src/gpu/gdec.cpp:10090-10160) vs k_gemm_wmma configs (#included verbatim
// from tools/gemm_wmma_kernel.inc, the sed-extracted production kernel).
// Correctness ref: rocBLAS algo_standard (driver tolerance maxrel < 1e-2).
// Usage: gemm_lt_bench [N K P ...]  (triples; default = the five 32K narrow
// shapes at P=16384 and P=32821). Env: GM=<list, comma> (default 2,4,16),
// CFG=<substring filter>.
// Build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/gemm_lt_bench tools/gemm_lt_bench.cu -lrocblas -lhipblaslt
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
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

#include "gemm_wmma_kernel.inc"

static uint32_t xs_state;
static uint32_t xs_next() {
  uint32_t x = xs_state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  xs_state = x;
  return x;
}
static uint16_t f2bf16(float f) {  // RNE
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static void fill_bf16(uint16_t* p, size_t n, uint32_t seed) {
  xs_state = seed;
  for (size_t i = 0; i < n; ++i) {
    const float f = (float)((xs_next() >> 8) & 0xFFFF) / 32768.f - 1.f;
    p[i] = f2bf16(f);
  }
}

struct KCfg {
  const char* name;
  int bm, bp, nt;
  void (*fn)(const uint16_t*, const uint16_t*, float*, int, int, int, int,
             hipStream_t);
};

template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT>
void launch_cfg(const uint16_t* X, const uint16_t* W, float* Y, int P, int K,
                int N, int gm, hipStream_t st) {
  const unsigned grid =
      (unsigned)(((P + BM - 1) / BM) * ((N + BP - 1) / BP));
  k_gemm_wmma<BM, BP, MW, PW, WM, WP, KST, NT>
      <<<grid, NT, 0, st>>>(X, W, Y, P, K, N, gm);
}

static const KCfg kCfgs[] = {
    {"d9 256t 128x256 f4x4", 128, 256, 256,
     launch_cfg<128, 256, 2, 4, 4, 4, 32, 256>},
    {"d3 512t 128x256 f4x2", 128, 256, 512,
     launch_cfg<128, 256, 2, 8, 4, 2, 32, 512>},
    {"d0 256t 128x128 f2x4", 128, 128, 256,
     launch_cfg<128, 128, 4, 2, 2, 4, 32, 256>},
    {"n128 256t 128x128 f4x2", 128, 128, 256,
     launch_cfg<128, 128, 2, 4, 4, 2, 32, 256>},
    {"n64 256t 128x64 f4x1", 128, 64, 256,
     launch_cfg<128, 64, 2, 4, 4, 1, 32, 256>},
    {"n64w 128t 128x64 f4x2", 128, 64, 128,
     launch_cfg<128, 64, 2, 2, 4, 2, 32, 128>},
    // KST=64 family (128B contiguous per row per staging step; the
    // gr320_proto lesson). LDS = 2*BM*(KST+8)*2 + 2*(BP/8)*(8*(KST+8)+8)*2;
    // the staging map needs 32/(KST/8) even, so KST is 32/64/128 only.
    {"h1 128t 64x160 f2x5 k64", 64, 160, 128,
     launch_cfg<64, 160, 2, 2, 2, 5, 64, 128>},
    {"k1 128t 64x128 f2x4 k64", 64, 128, 128,
     launch_cfg<64, 128, 2, 2, 2, 4, 64, 128>},
    {"k2 256t 64x128 f2x2 k64", 64, 128, 256,
     launch_cfg<64, 128, 2, 4, 2, 2, 64, 256>},
    {"k3 128t 128x64 f4x2 k64", 128, 64, 128,
     launch_cfg<128, 64, 2, 2, 4, 2, 64, 128>},
    {"k4 256t 128x64 f2x2 k64", 128, 64, 256,
     launch_cfg<128, 64, 4, 2, 2, 2, 64, 256>},
    {"k5 256t 64x160 f2x5 k64", 64, 160, 256,
     launch_cfg<64, 160, 4, 2, 1, 5, 64, 256>},
};

int main(int argc, char** argv) {
  std::vector<int> shp;
  for (int i = 1; i + 2 < argc; i += 3) {
    shp.push_back(atoi(argv[i]));
    shp.push_back(atoi(argv[i + 1]));
    shp.push_back(atoi(argv[i + 2]));
  }
  if (shp.empty())
    shp = {320,   10240, 16384, 320,  10240, 32821, 10240, 320,
           16384, 10240, 320,   32821, 640,  2560,  16384, 640,
           2560,  32821, 512,   2560, 16384, 512,   2560,  32821,
           2560,  640,   16384, 2560, 640,   32821};
  std::vector<int> gms = {2, 4, 16};
  if (getenv("GM")) {
    gms.clear();
    char* s = strdup(getenv("GM"));
    for (char* t = strtok(s, ","); t; t = strtok(nullptr, ","))
      gms.push_back(atoi(t));
    free(s);
  }
  const char* cfgfilt = getenv("CFG");
  const size_t lt_ws_bytes = (size_t)64 << 20;

  rocblas_handle h;
  rocblas_create_handle(&h);
  hipblasLtHandle_t lth;
  hipblasLtCreate(&lth);
  hipStream_t st;
  CK(hipStreamCreate(&st));
  rocblas_set_stream(h, st);
  void* d_ltws;
  CK(hipMalloc(&d_ltws, lt_ws_bytes));
  hipEvent_t ev0, ev1;
  CK(hipEventCreate(&ev0));
  CK(hipEventCreate(&ev1));
  const int ncfg = (int)(sizeof(kCfgs) / sizeof(kCfgs[0]));
  int nbad = 0;

  for (size_t s = 0; s < shp.size(); s += 3) {
    const int N = shp[s], K = shp[s + 1], P = shp[s + 2];
    if (K % 32) {
      printf("skip N=%d K=%d P=%d (K %% 32)\n", N, K, P);
      continue;
    }
    const double gflop = 2.0 * N * K * P / 1e9;
    printf("\n== N=%d K=%d P=%d (%.1f GFLOP) ==\n", N, K, P, gflop);
    fflush(stdout);

    std::vector<uint16_t> hA((size_t)N * K), hB((size_t)P * K);
    fill_bf16(hA.data(), hA.size(), 0xA5A5u + N + P);
    fill_bf16(hB.data(), hB.size(), 0x5A5Au + K + P);
    uint16_t *A, *B;
    float *C, *Cref;
    CK(hipMalloc(&A, hA.size() * 2));
    CK(hipMalloc(&B, hB.size() * 2));
    CK(hipMalloc(&C, (size_t)N * P * 4));
    CK(hipMalloc(&Cref, (size_t)N * P * 4));
    CK(hipMemcpy(A, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(B, hB.data(), hB.size() * 2, hipMemcpyHostToDevice));
    float one = 1.f, zero = 0.f;

    auto bench = [&](auto&& fn) {
      for (int i = 0; i < 3; ++i) fn();
      CK(hipStreamSynchronize(st));
      std::vector<float> ts(20);
      for (int i = 0; i < 20; ++i) {
        CK(hipEventRecord(ev0, st));
        fn();
        CK(hipEventRecord(ev1, st));
        CK(hipStreamSynchronize(st));
        CK(hipEventElapsedTime(&ts[i], ev0, ev1));
      }
      std::sort(ts.begin(), ts.end());
      return (double)ts[ts.size() / 2];
    };
    auto check = [&](const float* got, const char* tag) {
      std::vector<float> vC((size_t)N * P), vR((size_t)N * P);
      CK(hipMemcpy(vC.data(), got, vC.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(vR.data(), Cref, vR.size() * 4, hipMemcpyDeviceToHost));
      double maxrel = 0;
      for (size_t i = 0; i < vC.size(); ++i) {
        const double d = fabs((double)vC[i] - (double)vR[i]);
        maxrel = std::max(maxrel, d / std::max(1.0, (double)fabs(vR[i])));
      }
      if (maxrel >= 1e-2) {
        printf("  ** %s BAD maxrel %.3e\n", tag, maxrel);
        ++nbad;
      }
      return maxrel;
    };

    // rocBLAS standard ref
    if (rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none,
                        N, P, K, &one, A, rocblas_datatype_bf16_r, K, B,
                        rocblas_datatype_bf16_r, K, &zero, Cref,
                        rocblas_datatype_f32_r, N, Cref,
                        rocblas_datatype_f32_r, N, rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard, 0, 0) !=
        rocblas_status_success) {
      printf("  rocBLAS ref failed, skip\n");
      goto next;
    }
    CK(hipStreamSynchronize(st));

    // production hipBLASLt path: heuristic, index 4 validated, else scan
    {
      hipblasLtMatmulDesc_t opd;
      hipblasLtMatmulDescCreate(&opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
      hipblasLtMatmulDescSetAttribute(opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
      hipblasLtMatrixLayout_t Al, Bl, Cl, Dl;
      hipblasLtMatrixLayoutCreate(&Al, HIP_R_16BF, K, N, K);
      hipblasLtMatrixLayoutCreate(&Bl, HIP_R_16BF, K, P, K);
      hipblasLtMatrixLayoutCreate(&Cl, HIP_R_32F, N, P, N);
      hipblasLtMatrixLayoutCreate(&Dl, HIP_R_32F, N, P, N);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
          sizeof(lt_ws_bytes));
      hipblasLtMatmulHeuristicResult_t hres[8];
      int nres = 0;
      hipblasLtMatmulAlgoGetHeuristic(lth, opd, Al, Bl, Cl, Dl, pref, 8, hres,
                                      &nres);
      auto try_run = [&](int i) {
        hipblasLtMatmul(lth, opd, &one, A, Al, B, Bl, &zero, C, Cl, C, Dl,
                        &hres[i].algo, d_ltws, lt_ws_bytes, st);
        if (hipStreamSynchronize(st) != hipSuccess) {
          (void)hipGetLastError();
          return false;
        }
        return true;
      };
      int bi = -1;
      if (nres > 4 && hres[4].state == HIPBLAS_STATUS_SUCCESS &&
          hres[4].workspaceSize <= lt_ws_bytes && try_run(4))
        bi = 4;
      else {
        double best = 1e30;
        for (int i = 0; i < nres; i++) {
          if (hres[i].state != HIPBLAS_STATUS_SUCCESS) continue;
          if (hres[i].workspaceSize > lt_ws_bytes) continue;
          if (!try_run(i)) continue;
          auto t0 = std::chrono::steady_clock::now();
          for (int r = 0; r < 3; r++) try_run(i);
          double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
          if (ms < best) {
            best = ms;
            bi = i;
          }
        }
      }
      if (bi < 0) {
        printf("  hipBLASLt: no algo\n");
        ++nbad;
      } else {
        const double ms = bench([&] { try_run(bi); });
        const double mr = check(C, "Lt");
        printf("  Lt idx%d:%18s %8.3f ms  %6.1f TFLOPS   maxrel %.1e\n", bi,
               "(production)", ms, gflop / ms, mr);
        fflush(stdout);
      }
      hipblasLtMatmulPreferenceDestroy(pref);
      hipblasLtMatrixLayoutDestroy(Al);
      hipblasLtMatrixLayoutDestroy(Bl);
      hipblasLtMatrixLayoutDestroy(Cl);
      hipblasLtMatrixLayoutDestroy(Dl);
      hipblasLtMatmulDescDestroy(opd);
    }

    for (int c = 0; c < ncfg; ++c) {
      if (cfgfilt && !strstr(kCfgs[c].name, cfgfilt)) continue;
      if (strstr(kCfgs[c].name, " k64") && K % 64)
        continue;  // kernel hard requirement K % KST == 0
      // correctness once at gm=4, then bench the gm list
      kCfgs[c].fn(B, A, C, P, K, N, 4, st);
      if (hipStreamSynchronize(st) != hipSuccess) {
        printf("  %-22s LAUNCH/RUN FAIL\n", kCfgs[c].name);
        (void)hipGetLastError();
        ++nbad;
        continue;
      }
      const double mr = check(C, kCfgs[c].name);
      for (int gm : gms) {
        usleep(200 * 1000);  // thermal spacing
        const double ms =
            bench([&] { kCfgs[c].fn(B, A, C, P, K, N, gm, st); });
        printf("  %-22s gm=%-2d %8.3f ms  %6.1f TFLOPS   maxrel %.1e %s\n",
               kCfgs[c].name, gm, ms, gflop / ms, mr,
               mr < 1e-2 ? "OK" : "BAD");
        fflush(stdout);
      }
    }
  next:
    CK(hipFree(A));
    CK(hipFree(B));
    CK(hipFree(C));
    CK(hipFree(Cref));
    sleep(2);  // let the iGPU cool between shapes
  }
  hipblasLtDestroy(lth);
  rocblas_destroy_handle(h);
  CK(hipStreamDestroy(st));
  printf(nbad ? "RESULT: %d BAD\n" : "RESULT: ALL OK\n", nbad);
  return nbad != 0;
}
