// lds_conflict_probe.cu — calibrate SQC_LDS_BANK_CONFLICT for known patterns
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

__device__ __forceinline__ uint4 lds128(const char* p) {
  uint4 v = *(const uint4*)p;
  __asm__ volatile("" : "+v"(v.x), "+v"(v.y), "+v"(v.z), "+v"(v.w));
  return v;
}
// V: 0=broadcast, 1=perfect stride16, 2=B-frag region pattern, 3=A-frag qrowA
template <int V>
__global__ void __launch_bounds__(256) k_pat(uint4* out, int iters) {
  extern __shared__ char sm[];
  const int lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < 32768 / 16; i += 256)
    ((uint4*)sm)[i] = make_uint4(i, i ^ 7, i * 3, threadIdx.x);
  __syncthreads();
  // region layout: row r -> (r>>3)*656 + (r&7)*80 bytes (B) ; A: r*80
  uint32_t off;
  const int qrowA = (lane < 16 && !(lane & 1))     ? lane / 2
                    : (lane >= 17 && (lane & 1))   ? 8 + (lane - 17) / 2
                                                   : 0;
  if (V == 0) off = 0;
  else if (V == 1) off = lane * 16;
  else if (V == 2) {
    const int r = lane & 15;
    off = (r >> 3) * 656 + (r & 7) * 80;
  } else if (V == 3) {
    off = qrowA * 80;
  } else if (V == 4) {
    off = (qrowA >> 3) * 656 + (qrowA & 7) * 80;
  } else if (V == 5) {
    const int row = (threadIdx.x & 31) / 4, q = (threadIdx.x & 31) % 4;
    off = row * 80 + q * 16;
  } else if (V == 6) {
    const int row = (threadIdx.x & 31) / 4, q = (threadIdx.x & 31) % 4;
    off = (row >> 3) * 656 + (row & 7) * 80 + q * 16;
  } else if (V == 7) {
    // new A pattern: dummy rows for ignored lanes
    const int b = lane >> 1;
    const int r = (lane & 1) ? ((lane & 16) ? b : ((b + 4) & 7))
                             : ((lane & 16) ? 8 + ((b + 4) & 7) : b);
    off = r * 80;
  } else {
    // new store pattern as load: lane -> row lane&7, q lane>>3
    off = (lane & 7) * 80 + (lane >> 3) * 16;
  }
  uint4 acc = {0, 0, 0, 0};
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const uint4 a = lds128(sm + off + (u & 1) * 16 + ((i & 15) << 10));
      acc.x += a.x;
      acc.y += a.y;
      acc.z += a.z;
      acc.w += a.w;
    }
  }
  if (acc.x == 0xdeadbeef) out[threadIdx.x] = acc;
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 100000;
  uint4* out;
  hipMalloc(&out, 4096);
  const int smem = 32768;
  for (int v = 0; v < 9; ++v) {
    auto run = [&](int v) {
      switch (v) {
        case 0: k_pat<0><<<40, 256, smem>>>(out, iters); break;
        case 1: k_pat<1><<<40, 256, smem>>>(out, iters); break;
        case 2: k_pat<2><<<40, 256, smem>>>(out, iters); break;
        case 3: k_pat<3><<<40, 256, smem>>>(out, iters); break;
        case 4: k_pat<4><<<40, 256, smem>>>(out, iters); break;
        case 5: k_pat<5><<<40, 256, smem>>>(out, iters); break;
        case 6: k_pat<6><<<40, 256, smem>>>(out, iters); break;
        case 7: k_pat<7><<<40, 256, smem>>>(out, iters); break;
        default: k_pat<8><<<40, 256, smem>>>(out, iters); break;
      }
    };
    run(v);
    hipDeviceSynchronize();
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0);
    run(v);
    hipEventRecord(e1);
    hipDeviceSynchronize();
    float ms;
    hipEventElapsedTime(&ms, e0, e1);
    printf("V%d: %.3f ms\n", v, ms);
  }
  return 0;
}
