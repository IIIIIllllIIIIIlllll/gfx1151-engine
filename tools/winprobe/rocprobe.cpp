// 最小 rocBLAS GEMM 探针：验证 TheRock 无主 TensileLibrary.dat 时
// rocBLAS 在 gfx1151 上能否工作（fallback 分片加载是否生效）。
#include <cstdio>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define CK(x)                                                        \
  do {                                                               \
    auto _e = (x);                                                   \
    if (_e != rocblas_status_success) {                              \
      printf("rocBLAS FAIL %s -> %d\n", #x, (int)_e);                \
      return 1;                                                      \
    }                                                                \
  } while (0)

int main() {
  rocblas_handle h;
  CK(rocblas_create_handle(&h));

  const int M = 256, N = 256, K = 256;
  size_t sz = (size_t)M * N * sizeof(float);
  float *dA, *dB, *dC;
  if (hipMalloc(&dA, sz) || hipMalloc(&dB, sz) || hipMalloc(&dC, sz)) {
    printf("hipMalloc FAIL\n");
    return 1;
  }
  float* hA = new float[M * K];
  float* hB = new float[K * N];
  for (int i = 0; i < M * K; i++) hA[i] = 1.0f;
  for (int i = 0; i < K * N; i++) hB[i] = 2.0f;
  hipMemcpy(dA, hA, (size_t)M * K * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, (size_t)K * N * sizeof(float), hipMemcpyHostToDevice);

  float alpha = 1.0f, beta = 0.0f;
  printf("[sgemm] 调用 rocblas_sgemm ...\n");
  CK(rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, M, N, K,
                   &alpha, dA, M, dB, K, &beta, dC, M));
  CK(rocblas_status_success == rocblas_status_success ? rocblas_status_success
                                                      : rocblas_status_success);
  if (hipDeviceSynchronize()) {
    printf("hip sync FAIL\n");
    return 1;
  }
  float out[4];
  hipMemcpy(out, dC, sizeof(out), hipMemcpyDeviceToHost);
  printf("[sgemm] OK  C[0]=%.1f (期望 %.1f)\n", out[0], (float)K * 2.0f);

  // bf16 路径（引擎用的）
  rocblas_datatype type = rocblas_datatype_bf16_r;
  printf("[bf16 gemm] 调用 rocblas_gemm_ex ...\n");
  CK(rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none, M, N,
                     K, &alpha, dA, type, M, dB, type, K, &beta, dC,
                     rocblas_datatype_f32_r, M, dC, rocblas_datatype_f32_r, M,
                     rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
  if (hipDeviceSynchronize()) {
    printf("hip sync FAIL (bf16)\n");
    return 1;
  }
  printf("[bf16 gemm] OK\n");
  printf("== ROCBLAS PROBE PASS ==\n");
  return 0;
}
