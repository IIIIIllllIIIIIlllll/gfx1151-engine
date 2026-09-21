// moe_lt_scan.cu — hipBLASLt heuristic-candidate sweep for the MoE expert
// GEMM shapes, mirroring the engine's GDEC_MOE_LT path (src/gpu/gdec.cpp
// gemm_bf16, line ~9403):
//   A bf16 [N][K] k-contiguous (op T) x B bf16 [M][K] k-contiguous (op N)
//   -> C/D f32 [N][M] n-contiguous, HIPBLAS_COMPUTE_32F, 64 MB workspace cap.
// For each (shape, M) it enumerates every candidate returned by
// hipblasLtMatmulAlgoGetHeuristic, times each (warmup 3, 15 event-timed reps,
// median), and reports the best candidate vs heuristic index 4 (the engine's
// hardcoded pick).
//
// Usage: moe_lt_scan [maxCand]        (default maxCand = 64)
// Build: hipcc -O2 --offload-arch=gfx1151 -o build/moe_lt_scan \
//          tools/moe_lt_scan.cu -lhipblaslt
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                             \
  do {                                                                    \
    hipError_t e_ = (x);                                                  \
    if (e_ != hipSuccess) {                                               \
      fprintf(stderr, "hip error %d (%s) at %d\n", (int)e_,               \
              hipGetErrorString(e_), __LINE__);                           \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

static const size_t WS_BYTES = (size_t)64 << 20;  // == gdec lt_ws_bytes

int main(int argc, char** argv) {
  int maxcand = argc > 1 ? atoi(argv[1]) : 64;
  const int Ns[2] = {1280, 2560};   // up (gated 2*640), down (d)
  const int Ks[2] = {2560, 640};
  const int Ms[] = {32, 64, 128, 160, 256, 320, 512, 640, 1024, 2048};
  const int nM = sizeof(Ms) / sizeof(Ms[0]);

  hipblasLtHandle_t lth;
  if (hipblasLtCreate(&lth) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "hipblasLtCreate failed\n");
    return 1;
  }
  hipStream_t st;
  CK(hipStreamCreate(&st));
  void* ws;
  CK(hipMalloc(&ws, WS_BYTES));

  for (int s = 0; s < 2; s++) {
    const int N = Ns[s], K = Ks[s];
    __hip_bfloat16 *A, *B;
    float* C;
    CK(hipMalloc(&A, (size_t)N * K * 2));
    CK(hipMalloc(&B, (size_t)2048 * K * 2));  // max M
    CK(hipMalloc(&C, (size_t)N * 2048 * 4));
    CK(hipMemset(A, 0x3c, (size_t)N * K * 2));
    CK(hipMemset(B, 0x3c, (size_t)2048 * K * 2));

    for (int mi = 0; mi < nM; mi++) {
      const int M = Ms[mi];
      const double gflop = 2.0 * N * K * M / 1e9;
      float one = 1.f, zero = 0.f;

      hipblasLtMatmulDesc_t opd;
      hipblasLtMatmulDescCreate(&opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
      hipblasLtMatmulDescSetAttribute(opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
      hipblasLtMatrixLayout_t Al, Bl, Cl, Dl;
      hipblasLtMatrixLayoutCreate(&Al, HIP_R_16BF, K, N, K);
      hipblasLtMatrixLayoutCreate(&Bl, HIP_R_16BF, K, M, K);
      hipblasLtMatrixLayoutCreate(&Cl, HIP_R_32F, N, M, N);
      hipblasLtMatrixLayoutCreate(&Dl, HIP_R_32F, N, M, N);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS_BYTES,
          sizeof(WS_BYTES));

      std::vector<hipblasLtMatmulHeuristicResult_t> hres(maxcand);
      int nres = 0;
      hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(
          lth, opd, Al, Bl, Cl, Dl, pref, maxcand, hres.data(), &nres);
      if (hs != HIPBLAS_STATUS_SUCCESS || nres <= 0) {
        printf("N=%d K=%d M=%d: heuristic failed rc=%d nres=%d\n", N, K, M,
               (int)hs, nres);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(Al);
        hipblasLtMatrixLayoutDestroy(Bl);
        hipblasLtMatrixLayoutDestroy(Cl);
        hipblasLtMatrixLayoutDestroy(Dl);
        hipblasLtMatmulDescDestroy(opd);
        continue;
      }

      // Timing buffers for event pairs (2 per rep).
      const int reps = 15;
      std::vector<hipEvent_t> ev(2 * reps);
      for (auto& e : ev) CK(hipEventCreate(&e));

      struct Row { int idx; double ms, tf; bool ok; };
      std::vector<Row> rows;
      // Dedupe identical algo blobs (heuristic can repeat the same algo).
      std::vector<hipblasLtMatmulAlgo_t> seen;
      for (int i = 0; i < nres; i++) {
        Row r{i, 1e30, 0.0, false};
        if (hres[i].state != HIPBLAS_STATUS_SUCCESS ||
            hres[i].workspaceSize > WS_BYTES) {
          rows.push_back(r);
          continue;
        }
        bool dup = false;
        for (auto& a : seen)
          if (memcmp(&a, &hres[i].algo, sizeof(a)) == 0) { dup = true; break; }
        if (dup) { rows.push_back(r); continue; }
        seen.push_back(hres[i].algo);
        auto call = [&]() {
          return hipblasLtMatmul(lth, opd, &one, A, Al, B, Bl, &zero, C, Cl, C,
                                 Dl, &hres[i].algo, ws, WS_BYTES, st);
        };
        if (call() != HIPBLAS_STATUS_SUCCESS) { rows.push_back(r); continue; }
        for (int w = 0; w < 2; w++) call();
        CK(hipStreamSynchronize(st));
        for (int t = 0; t < reps; t++) {
          CK(hipEventRecord(ev[2 * t], st));
          call();
          CK(hipEventRecord(ev[2 * t + 1], st));
        }
        CK(hipStreamSynchronize(st));
        std::vector<float> ts(reps);
        for (int t = 0; t < reps; t++)
          CK(hipEventElapsedTime(&ts[t], ev[2 * t], ev[2 * t + 1]));
        std::sort(ts.begin(), ts.end());
        r.ms = ts[reps / 2];
        r.tf = gflop / r.ms;
        r.ok = true;
        rows.push_back(r);
      }
      for (auto& e : ev) CK(hipEventDestroy(e));

      int best = -1;
      for (auto& r : rows)
        if (r.ok && (best < 0 || r.ms < rows[best].ms)) best = r.idx;
      const bool has4 = nres > 4 && rows[4].ok;
      printf("\n== N=%d K=%d M=%4d (%.2f GFLOP) candidates=%d (unique=%zu) ==\n",
             N, K, M, gflop, nres, seen.size());
      for (auto& r : rows) {
        if (!r.ok) {
          printf("  algo[%2d]: (invalid/dup)\n", r.idx);
          continue;
        }
        printf("  algo[%2d]: %8.4f ms  %6.2f TFLOPS%s%s\n", r.idx, r.ms, r.tf,
               r.idx == best ? "  BEST" : "", r.idx == 4 ? "  [idx4]" : "");
      }
      if (best >= 0)
        printf("  -> best algo[%d] %.2f TF | idx4 %s%.2f TF | best/idx4 = %.2fx\n",
               best, rows[best].tf, has4 ? "" : "n/a ",
               has4 ? rows[4].tf : 0.0,
               has4 ? rows[best].tf / rows[4].tf : 0.0);

      hipblasLtMatmulPreferenceDestroy(pref);
      hipblasLtMatrixLayoutDestroy(Al);
      hipblasLtMatrixLayoutDestroy(Bl);
      hipblasLtMatrixLayoutDestroy(Cl);
      hipblasLtMatrixLayoutDestroy(Dl);
      hipblasLtMatmulDescDestroy(opd);
    }
    CK(hipFree(A));
    CK(hipFree(B));
    CK(hipFree(C));
  }
  CK(hipFree(ws));
  CK(hipStreamDestroy(st));
  hipblasLtDestroy(lth);
  return 0;
}
