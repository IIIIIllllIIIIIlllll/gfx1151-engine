// 探针9：rocBLAS + hipBLASLt 在 gfx1151/Windows 上的 GEMM 冒烟测试
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>

#define CK(x) do { auto e_ = (x); if (e_ != hipSuccess) { \
  printf("FAIL %s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)
#define CKR(x) do { auto e_ = (x); if (e_ != rocblas_status_success) { \
  printf("FAIL %s: rocblas err %d\n", #x, (int)e_); exit(1); } } while (0)
#define CKL(x) do { auto e_ = (x); if (e_ != HIPBLAS_STATUS_SUCCESS) { \
  printf("FAIL %s: hipblaslt err %d\n", #x, (int)e_); exit(1); } } while (0)

int main() {
  const int M = 4096, N = 4096, K = 4096;
  size_t sz = (size_t)M * N * sizeof(float);
  float *a, *b, *c;
  CK(hipMalloc(&a, sz)); CK(hipMalloc(&b, sz)); CK(hipMalloc(&c, sz));
  std::vector<float> ha((size_t)M*K), hb((size_t)K*N);
  for (auto& x : ha) x = 0.5f; for (auto& x : hb) x = 0.25f;
  CK(hipMemcpy(a, ha.data(), (size_t)M*K*4, hipMemcpyHostToDevice));
  CK(hipMemcpy(b, hb.data(), (size_t)K*N*4, hipMemcpyHostToDevice));

  // rocBLAS SGEMM
  rocblas_handle h; CKR(rocblas_create_handle(&h));
  float alpha = 1.f, beta = 0.f;
  CKR(rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha, a, M, b, K, &beta, c, M));
  CK(hipDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; i++)
    rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                  M, N, K, &alpha, a, M, b, K, &beta, c, M);
  CK(hipDeviceSynchronize());
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  float hc[4]; CK(hipMemcpy(hc, c, 16, hipMemcpyDeviceToHost));
  printf("rocBLAS SGEMM: %.1f TFLOPS, c[0]=%.1f (expect %.1f)\n",
         10.0*2.0*M*N*K/s/1e12, hc[0], (float)K*0.5f*0.25f);
  CKR(rocblas_destroy_handle(h));

  // hipBLASLt bf16 GEMM (引擎 MoE LT 路径用 bf16)
  hipblasLtHandle_t lt; CKL(hipblasLtCreate(&lt));
  hipblasLtMatmulDesc_t op; CKL(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t nop = HIPBLAS_OP_N;
  CKL(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &nop, sizeof(nop)));
  CKL(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &nop, sizeof(nop)));
  hipblasLtMatrixLayout_t la, lb, lc;
  CKL(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, M, K, M));
  CKL(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, N, K));
  CKL(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16BF, M, N, M));
  hipblasLtMatmulPreference_t pref; CKL(hipblasLtMatmulPreferenceCreate(&pref));
  hipblasLtMatmulHeuristicResult_t heur[4]; int nres = 0;
  CKL(hipblasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 4, heur, &nres));
  if (nres == 0) { printf("hipBLASLt: no heuristic (FAIL)\n"); return 1; }
  void *wa, *wb, *wc; CK(hipMalloc(&wa, sz/2)); CK(hipMalloc(&wb, sz/2)); CK(hipMalloc(&wc, sz/2));
  void* ws; CK(hipMalloc(&ws, heur[0].workspaceSize));
  CKL(hipblasLtMatmul(lt, op, &alpha, wa, la, wb, lb, &beta, wc, lc, wc, lc,
                      &heur[0].algo, ws, heur[0].workspaceSize, 0));
  CK(hipDeviceSynchronize());
  printf("hipBLASLt bf16 GEMM: OK (heuristics=%d, ws=%zu)\n", nres, heur[0].workspaceSize);
  printf("ALL BLAS PASS\n");
  return 0;
}
