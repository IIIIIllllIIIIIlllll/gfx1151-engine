#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
int main() {
  printf("rocblas status ok = %d\n", (int)ROCBLAS_STATUS_SUCCESS);
  printf("hipblaslt status ok = %d\n", (int)HIPBLASLT_STATUS_SUCCESS);
  return 0;
}
