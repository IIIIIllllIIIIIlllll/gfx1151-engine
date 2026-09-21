// gdn_stage_probe.cu — isolate the phase-0 staging cost of k_gdn_intra:
// same grid (128,48), NT=1024, same qkv layout, only the k-tile staging
// loop (fp32 scalar loads vs float4 loads vs different element mappings),
// then exit. Question: why does staging 32KB/block cost ~20us/block?
#include <hip/hip_runtime.h>
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

constexpr int CH = 64, DK = 128, NT = 1024;

// M0: production mapping — scalar loads, i/128/i%128
// M1: float4 loads (2 per thread)
// M2: scalar loads, thread owns row pair (t = tid/16, d = (tid%16)*8..)
template <int M>
__global__ void __launch_bounds__(NT) k_stage(const float* __restrict__ qkv,
                                              float* __restrict__ sink, int P,
                                              int qkvstride) {
  const int c = blockIdx.x, h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  const int t0 = c * CH;
  __shared__ float s_k[CH][DK + 4];
  if (M == 0) {
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      s_k[t][d] = (t0 + t < P)
                      ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d]
                      : 0.f;
    }
  } else if (M == 1) {
#pragma unroll
    for (int n = 0; n < 2; n++) {
      int i = tid + n * NT;  // float4 index space: CH*DK/4 = 2048
      int t = i / (DK / 4), d4 = i % (DK / 4);
      float4 v = {0.f, 0.f, 0.f, 0.f};
      if (t0 + t < P)
        v = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 2048 +
                             kh * DK + d4 * 4);
      s_k[t][d4 * 4] = v.x;
      s_k[t][d4 * 4 + 1] = v.y;
      s_k[t][d4 * 4 + 2] = v.z;
      s_k[t][d4 * 4 + 3] = v.w;
    }
  }
  __syncthreads();
  if (tid == 0) sink[blockIdx.x * 48 + blockIdx.y] = s_k[tid][tid];
}

// R0: read-only, same grid — no LDS, no staging, just the k read pattern
// R1: read-only, grid (48) — each block serially covers all chunks of a head
template <int M>
__global__ void __launch_bounds__(NT) k_read(const float* __restrict__ qkv,
                                             float* __restrict__ sink, int P,
                                             int qkvstride) {
  const int h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  float acc = 0.f;
  if (M == 0) {
    const int t0 = blockIdx.x * CH;
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      if (t0 + t < P)
        acc += qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d];
    }
  } else {
    const int nchunks = (P + CH - 1) / CH;
    for (int c = 0; c < nchunks; c++) {
      const int t0 = c * CH;
      for (int i = tid; i < CH * DK; i += NT) {
        int t = i / DK, d = i % DK;
        if (t0 + t < P)
          acc += qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d];
      }
    }
  }
  if (acc == 1.25f) sink[blockIdx.x * 48 + blockIdx.y] = acc;  // never true
}

int main(int argc, char** argv) {
  const int P = argc > 1 ? atoi(argv[1]) : 8192;
  const int reps = argc > 2 ? atoi(argv[2]) : 10;
  const int qkvstride = 10240;
  const int nchunks = P / CH;
  float *d_qkv, *d_sink;
  CK(hipMalloc(&d_qkv, (size_t)P * qkvstride * 4));
  CK(hipMalloc(&d_sink, nchunks * 48 * 4));
  CK(hipMemset(d_qkv, 0x11, (size_t)P * qkvstride * 4));
  const char* nm[2] = {"scalar i/128", "float4 x2"};
  for (int M = 0; M < 2; M++) {
    auto run = [&]() {
      if (M == 0)
        k_stage<0><<<dim3(nchunks, 48), NT>>>(d_qkv, d_sink, P, qkvstride);
      else
        k_stage<1><<<dim3(nchunks, 48), NT>>>(d_qkv, d_sink, P, qkvstride);
    };
    for (int i = 0; i < 3; i++) run();
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < reps; i++) run();
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    double w = ms / reps, gb = (double)nchunks * 48 * CH * DK * 4 / 1e9;
    printf("== %-14s wall %7.3f ms/iter -> %.0f GB/s (%.2f us/block) ==\n",
           nm[M], w, gb / (w / 1e3), w * 1e3 * 20 / (nchunks * 48) * 1e3 / 1e3);
  }
  // read-only probes
  for (int M = 0; M < 2; M++) {
    auto run = [&]() {
      if (M == 0)
        k_read<0><<<dim3(nchunks, 48), NT>>>(d_qkv, d_sink, P, qkvstride);
      else
        k_read<1><<<dim3(1, 48), NT>>>(d_qkv, d_sink, P, qkvstride);
    };
    for (int i = 0; i < 3; i++) run();
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < reps; i++) run();
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    double w = ms / reps, gb = (double)nchunks * 48 * CH * DK * 4 / 1e9;
    printf("== read-only %-12s wall %7.3f ms/iter -> %.0f GB/s ==\n",
           M == 0 ? "grid(128,48)" : "grid(1,48)loop", w, gb / (w / 1e3));
  }
  return 0;
}
