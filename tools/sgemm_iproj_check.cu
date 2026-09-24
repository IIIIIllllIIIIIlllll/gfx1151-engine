// sgemm_iproj_check.cu — fp32 indexer projection (Model::qsa_b index_f32,
// rocblas_sgemm T/N 640 x P x 2560) vs the rocBLAS solution list.
// For each P: algo_standard (what rocblas_sgemm runs) as reference, then every
// solution index: run twice (bitwise run-to-run determinism — split-K with
// atomics would fail this), maxrel vs reference, median time.
// Usage: sgemm_iproj_check [P ...]   (default 16384 8192 9000 2048 1024 53)
// Build: hipcc -O2 --offload-arch=gfx1151 -o /tmp/sgemm_iproj_check tools/sgemm_iproj_check.cu -lrocblas
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                              \
  do {                                                                     \
    hipError_t e_ = (x);                                                   \
    if (e_ != hipSuccess) {                                                \
      fprintf(stderr, "hip error %d at %s:%d\n", (int)e_, __FILE__, __LINE__); \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

static uint32_t xs = 0x1234567u;
static float frand() {
  xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5;
  return (float)((xs >> 8) & 0xFFFF) / 32768.f - 1.f;
}

int main(int argc, char** argv) {
  std::vector<int> Ps;
  for (int i = 1; i < argc; i++) Ps.push_back(atoi(argv[i]));
  if (Ps.empty()) Ps = {16384, 8192, 9000, 2048, 1024, 53};
  const int M = 640, K = 2560;
  rocblas_handle h;
  rocblas_create_handle(&h);
  hipStream_t st;
  CK(hipStreamCreate(&st));
  rocblas_set_stream(h, st);
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  int nbad = 0;
  for (int P : Ps) {
    std::vector<float> hA((size_t)M * K), hB((size_t)P * K);
    for (auto& v : hA) v = frand() * 0.05f;
    for (auto& v : hB) v = frand();
    float *A, *B, *C, *R;
    CK(hipMalloc(&A, hA.size() * 4));
    CK(hipMalloc(&B, hB.size() * 4));
    CK(hipMalloc(&C, (size_t)M * P * 4));
    CK(hipMalloc(&R, (size_t)M * P * 4));
    CK(hipMemcpy(A, hA.data(), hA.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(B, hB.data(), hB.size() * 4, hipMemcpyHostToDevice));
    const float one = 1.f, zero = 0.f;
    auto run = [&](float* out, rocblas_gemm_algo algo, int idx) {
      return rocblas_gemm_ex(h, rocblas_operation_transpose,
                             rocblas_operation_none, M, P, K, &one, A,
                             rocblas_datatype_f32_r, K, B,
                             rocblas_datatype_f32_r, K, &zero, out,
                             rocblas_datatype_f32_r, M, out,
                             rocblas_datatype_f32_r, M, rocblas_datatype_f32_r,
                             algo, idx, 0);
    };
    auto timeit = [&](auto&& fn) {
      fn();
      CK(hipStreamSynchronize(st));
      std::vector<float> ts(9);
      for (auto& t : ts) {
        CK(hipEventRecord(e0, st));
        fn();
        CK(hipEventRecord(e1, st));
        CK(hipStreamSynchronize(st));
        CK(hipEventElapsedTime(&t, e0, e1));
      }
      std::sort(ts.begin(), ts.end());
      return ts[ts.size() / 2];
    };
    // reference = what rocblas_sgemm does in the engine today
    const float tstd = timeit([&] {
      rocblas_sgemm(h, rocblas_operation_transpose, rocblas_operation_none, M,
                    P, K, &one, A, K, B, K, &zero, R, M);
    });
    std::vector<float> vR((size_t)M * P), v1(vR.size()), v2(vR.size());
    CK(hipMemcpy(vR.data(), R, vR.size() * 4, hipMemcpyDeviceToHost));
    // fp64 host spot check of the reference itself (first 64 columns)
    double refrel = 0;
    for (int p = 0; p < std::min(P, 64); p++)
      for (int m = 0; m < M; m++) {
        double acc = 0;
        for (int k = 0; k < K; k++)
          acc += (double)hA[(size_t)m * K + k] * hB[(size_t)p * K + k];
        refrel = std::max(refrel, fabs(acc - vR[(size_t)p * M + m]) /
                                      std::max(1.0, fabs(acc)));
      }
    printf("\n== P=%d  sgemm(standard) %.3f ms  %.1f TFLOPS  ref-vs-fp64 %.1e ==\n",
           P, tstd, 2.0 * M * K * P / 1e9 / tstd, refrel);
    rocblas_int nsol = 0;
    rocblas_gemm_ex_get_solutions(
        h, rocblas_operation_transpose, rocblas_operation_none, M, P, K, &one,
        A, rocblas_datatype_f32_r, K, B, rocblas_datatype_f32_r, K, &zero, C,
        rocblas_datatype_f32_r, M, C, rocblas_datatype_f32_r, M,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, nullptr,
        &nsol);
    std::vector<rocblas_int> sols(std::max(nsol, 1));
    rocblas_gemm_ex_get_solutions(
        h, rocblas_operation_transpose, rocblas_operation_none, M, P, K, &one,
        A, rocblas_datatype_f32_r, K, B, rocblas_datatype_f32_r, K, &zero, C,
        rocblas_datatype_f32_r, M, C, rocblas_datatype_f32_r, M,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0,
        sols.data(), &nsol);
    for (int i = 0; i < nsol; i++) {
      const int idx = sols[i];
      if (run(C, rocblas_gemm_algo_solution_index, idx) !=
              rocblas_status_success ||
          hipStreamSynchronize(st) != hipSuccess) {
        (void)hipGetLastError();
        printf("  sol %11d: run failed\n", idx);
        continue;
      }
      CK(hipMemcpy(v1.data(), C, v1.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemset(C, 0xff, v1.size() * 4));
      run(C, rocblas_gemm_algo_solution_index, idx);
      CK(hipStreamSynchronize(st));
      CK(hipMemcpy(v2.data(), C, v2.size() * 4, hipMemcpyDeviceToHost));
      const bool det = memcmp(v1.data(), v2.data(), v1.size() * 4) == 0;
      double mr = 0;
      size_t nexact = 0;
      for (size_t j = 0; j < vR.size(); j++) {
        mr = std::max(mr, fabs((double)v1[j] - vR[j]) /
                              std::max(1.0, fabs((double)vR[j])));
        nexact += v1[j] == vR[j];
      }
      const float t = timeit([&] { run(C, rocblas_gemm_algo_solution_index, idx); });
      const bool ok = det && mr < 1e-5;
      nbad += !ok;
      printf("  sol %11d: %8.3f ms  %5.1f TFLOPS  x%.2f  det=%s  maxrel %.1e  bit-equal %.1f%%  %s\n",
             idx, t, 2.0 * M * K * P / 1e9 / t, tstd / t, det ? "yes" : "NO",
             mr, 100.0 * nexact / vR.size(), ok ? "OK" : "BAD");
    }
    CK(hipFree(A)); CK(hipFree(B)); CK(hipFree(C)); CK(hipFree(R));
  }
  printf(nbad ? "RESULT: %d BAD\n" : "RESULT: ALL OK\n", nbad);
  rocblas_destroy_handle(h);
  return 0;
}
