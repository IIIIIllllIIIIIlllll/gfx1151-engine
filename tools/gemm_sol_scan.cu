// gemm_sol_scan.cu — rocBLAS solution-index autotune for prefill dense GEMM shapes.
// Mirrors Model::gemm() (src/gpu/gdec.cpp:8899): C[N,P]f32 = A[N,K]bf16^T * B[K,P]bf16.
// Usage: gemm_sol_scan [N K P ...]  (triples; defaults to the slow prefill shapes)
// Build: hipcc -O2 -o build/gemm_sol_scan tools/gemm_sol_scan.cu -lrocblas
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                           \
  do {                                                                  \
    hipError_t e_ = (x);                                                \
    if (e_ != hipSuccess) {                                             \
      fprintf(stderr, "hip error %d at %s:%d\n", (int)e_, __FILE__, __LINE__); \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

int main(int argc, char** argv) {
  std::vector<int> shp;
  for (int i = 1; i + 2 < argc; i += 3) {
    shp.push_back(atoi(argv[i]));
    shp.push_back(atoi(argv[i + 1]));
    shp.push_back(atoi(argv[i + 2]));
  }
  if (shp.empty()) shp = {8192, 2560, 6144,  8192, 6144, 2560,
                          8192, 12288, 2560, 8192, 10240, 2560,
                          320,  8192, 10240, 8192, 320, 2560};
  rocblas_handle h;
  rocblas_create_handle(&h);
  hipStream_t st;
  CK(hipStreamCreate(&st));
  rocblas_set_stream(h, st);

  for (size_t s = 0; s < shp.size(); s += 3) {
    int N = shp[s], K = shp[s + 1], P = shp[s + 2];
    __hip_bfloat16 *A, *B;
    float* C;
    CK(hipMalloc(&A, (size_t)N * K * 2));
    CK(hipMalloc(&B, (size_t)K * P * 2));
    CK(hipMalloc(&C, (size_t)N * P * 4));
    CK(hipMemset(A, 0x3c, (size_t)N * K * 2));
    CK(hipMemset(B, 0x3c, (size_t)K * P * 2));
    float one = 1.f, zero = 0.f;
    double gflop = 2.0 * N * K * P / 1e9;

    rocblas_int nsol = 0;
    rocblas_status qs = rocblas_gemm_ex_get_solutions(
        h, rocblas_operation_transpose, rocblas_operation_none, N, P, K, &one,
        A, rocblas_datatype_bf16_r, K, B, rocblas_datatype_bf16_r, K, &zero, C,
        rocblas_datatype_f32_r, N, C, rocblas_datatype_f32_r, N,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, nullptr,
        &nsol);
    printf("\n== N=%d K=%d P=%d (%.1f GFLOP) solutions: %d (rc=%d) ==\n", N, K,
           P, gflop, (int)nsol, (int)qs);
    if (qs != rocblas_status_success || nsol <= 0) {
      CK(hipFree(A)); CK(hipFree(B)); CK(hipFree(C));
      continue;
    }
    std::vector<rocblas_int> sols(nsol);
    rocblas_int ngot = nsol;  // list_size is IN when list_array != NULL
    rocblas_status gs = rocblas_gemm_ex_get_solutions(
        h, rocblas_operation_transpose, rocblas_operation_none, N, P, K, &one,
        A, rocblas_datatype_bf16_r, K, B, rocblas_datatype_bf16_r, K, &zero, C,
        rocblas_datatype_f32_r, N, C, rocblas_datatype_f32_r, N,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0,
        sols.data(), &ngot);
    printf("  get_solutions rc=%d ngot=%d first=%d\n", (int)gs, (int)ngot,
           ngot > 0 ? (int)sols[0] : -1);
    double best = 1e30;
    int besti = -1;
    for (int si = 0; si < ngot; si++) {
      rocblas_int idx = sols[si];
      auto run = [&]() {
        return rocblas_gemm_ex(
            h, rocblas_operation_transpose, rocblas_operation_none, N, P, K,
            &one, A, rocblas_datatype_bf16_r, K, B, rocblas_datatype_bf16_r, K,
            &zero, C, rocblas_datatype_f32_r, N, C, rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, idx, 0);
      };
      rocblas_status r0 = run();
      if (r0 != rocblas_status_success) {
        if (si == 0) printf("  (first run rc=%d idx=%d)\n", (int)r0, (int)idx);
        continue;
      }
      if (hipStreamSynchronize(st) != hipSuccess) {
        (void)hipGetLastError();
        continue;
      }
      int reps = 5;
      auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; r++) run();
      CK(hipStreamSynchronize(st));
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count() /
                  reps;
      printf("  sol %4d: %8.3f ms  %6.1f TFLOPS%s\n", (int)idx, ms,
             gflop / ms, ms < best ? " *" : "");
      if (ms < best) {
        best = ms;
        besti = idx;
      }
    }
    auto run0 = [&]() {
      return rocblas_gemm_ex(
          h, rocblas_operation_transpose, rocblas_operation_none, N, P, K, &one,
          A, rocblas_datatype_bf16_r, K, B, rocblas_datatype_bf16_r, K, &zero, C,
          rocblas_datatype_f32_r, N, C, rocblas_datatype_f32_r, N,
          rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    };
    run0();
    CK(hipStreamSynchronize(st));
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; r++) run0();
    CK(hipStreamSynchronize(st));
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count() /
                5;
    printf("  BEST sol %d: %.3f ms (%.1f TFLOPS); algo_standard: %.3f ms (%.1f TFLOPS)\n",
           besti, best, gflop / best, ms, gflop / ms);
    CK(hipFree(A)); CK(hipFree(B)); CK(hipFree(C));
  }
  rocblas_destroy_handle(h);
  CK(hipStreamDestroy(st));
  return 0;
}
