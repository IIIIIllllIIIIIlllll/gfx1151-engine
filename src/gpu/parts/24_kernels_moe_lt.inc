// ============ MoE prefill via dequant + hipBLASLt (GDEC_MOE_LT, opt-in) ======
// Batched-GEMM prefill pipeline: per layer, dequant Q4C-P expert weights to
// bf16 in small expert groups (scratch reused, MALL-hot), gather pair rows,
// per-expert bf16 GEMM via hipBLASLt, silu, per-expert down GEMM,
// deterministic weighted reduce.
// Numerics differ from the fused W4 kernels: w = bf16(fp16_scale * cb[q])
// (scale folded into the weight instead of applied to fp32 group partials)
// and Tensile WMMA accumulation order. Prototype: tools/moe_lt_bench.cu.
// Requires GPU routing + deterministic mode (uses d_tokidx/d_eoff/d_pairids).

// Dequant Q4C-P rows to bf16 [rows][cols] (k contiguous): one thread per
// 32-wide scale group (16B codes -> 64B bf16).
__global__ void k_moe_deq_bf16(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               __hip_bfloat16* __restrict__ out, int rows,
                               int cols, uint64_t scale_stride) {
  int groups_per_row = cols / 32;
  int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (int64_t)rows * groups_per_row) return;
  int r = (int)(gid / groups_per_row), g = (int)(gid % groups_per_row);
  const uint8_t* cb16 = codes + (size_t)r * (cols / 2) + g * 16;
  float sc = __half2float(*(const __half*)(scales + (size_t)r * scale_stride + g * 2));
  __hip_bfloat16* o = out + (size_t)r * cols + g * 32;
  __hip_bfloat16 w[32];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    uint8_t b = cb16[j];
    w[2 * j] = __float2bfloat16(sc * cb[b & 15]);
    w[2 * j + 1] = __float2bfloat16(sc * cb[b >> 4]);
  }
  *(uint4*)(o) = *(uint4*)&w[0];
  *(uint4*)(o + 8) = *(uint4*)&w[8];
  *(uint4*)(o + 16) = *(uint4*)&w[16];
  *(uint4*)(o + 24) = *(uint4*)&w[24];
}

// Gather activation rows by pair: xg[p][:] = x[tokidx[p]][:] (bf16, K elems).
__global__ void k_moe_gather_bf16(__hip_bfloat16* __restrict__ xg,
                                  const __hip_bfloat16* __restrict__ x,
                                  const int* __restrict__ tokidx, int K) {
  int p = blockIdx.x;
  const __hip_bfloat16* s = x + (size_t)tokidx[p] * K;
  __hip_bfloat16* dd = xg + (size_t)p * K;
  for (int i = threadIdx.x; i < K / 8; i += blockDim.x)
    *(uint4*)(dd + i * 8) = *(const uint4*)(s + i * 8);
}

// silu(g)*u on fp32 guv [slots][2*mid] -> bf16 hid [slots][mid].
__global__ void k_moe_silu_f32_bf16(const float* __restrict__ guv,
                                    __hip_bfloat16* __restrict__ hid, int mid,
                                    int64_t nslots) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nslots * mid) return;
  int64_t slot = i / mid;
  int r = (int)(i % mid);
  float a = guv[slot * 2 * mid + r];
  hid[i] = __float2bfloat16(a / (1.f + expf(-a)) * guv[slot * 2 * mid + mid + r]);
}

// silu(g)*u on bf16 guv [slots][2*mid] -> bf16 hid [slots][mid].
__global__ void k_moe_silu_bf16(const __hip_bfloat16* __restrict__ guv,
                                __hip_bfloat16* __restrict__ hid, int mid,
                                int64_t nslots) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nslots * mid) return;
  int64_t slot = i / mid;
  int r = (int)(i % mid);
  float a = __bfloat162float(guv[slot * 2 * mid + r]);
  hid[i] = __float2bfloat16(a / (1.f + expf(-a)) *
                            __bfloat162float(guv[slot * 2 * mid + mid + r]));
}

// Weighted deterministic reduce: out[t][d] = sum_s pw[ids[s]] * pairs[ids[s]][d]
// (same slot order as k_moe_reduce_fast; the router weight is applied here
// instead of in the down kernel).
template <int K>
__global__ void k_moe_reduce_pw(const float* __restrict__ pairs,
                                const int* __restrict__ pairids,
                                const float* __restrict__ pw,
                                float* __restrict__ out, int P, int Dd) {
  __shared__ int ids[K];
  int tid = threadIdx.x;
  int t = blockIdx.y, d = tid + (int)blockIdx.x * blockDim.x;
  if (t >= P) return;
  if (tid < K) ids[tid] = pairids[t * K + tid];
  __syncthreads();
  if (d >= Dd) return;
  float sum = 0.f;
#pragma unroll
  for (int s = 0; s < K; s++) sum += pw[ids[s]] * pairs[(size_t)ids[s] * Dd + d];
  out[(size_t)t * Dd + d] = sum;
}

// bf16 pair variant (fp32 accumulate).
template <int K>
__global__ void k_moe_reduce_pw_bf16(const __hip_bfloat16* __restrict__ pairs,
                                     const int* __restrict__ pairids,
                                     const float* __restrict__ pw,
                                     float* __restrict__ out, int P, int Dd) {
  __shared__ int ids[K];
  int tid = threadIdx.x;
  int t = blockIdx.y, d = tid + (int)blockIdx.x * blockDim.x;
  if (t >= P) return;
  if (tid < K) ids[tid] = pairids[t * K + tid];
  __syncthreads();
  if (d >= Dd) return;
  float sum = 0.f;
#pragma unroll
  for (int s = 0; s < K; s++)
    sum += pw[ids[s]] * __bfloat162float(pairs[(size_t)ids[s] * Dd + d]);
  out[(size_t)t * Dd + d] = sum;
}

// Debug instrumentation (GDEC_MOE_LT_CMP): block-reduced max abs diff of the
// LT path's intermediates against the fused W4 reference run in the same
// process on identical inputs. out = [max|diff|, max|ref|] as ordered ints
// (non-negative floats order like their int bit patterns).
__global__ void k_cmp_absmax_bf16(const __hip_bfloat16* __restrict__ a,
                                  const __hip_bfloat16* __restrict__ b,
                                  int64_t n, int* __restrict__ out) {
  float md = 0.f, mr = 0.f;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    float r = __bfloat162float(b[i]);
    md = fmaxf(md, fabsf(__bfloat162float(a[i]) - r));
    mr = fmaxf(mr, fabsf(r));
  }
  for (int o = 16; o; o >>= 1) {
    md = fmaxf(md, __shfl_xor_sync(~0ull, md, o));
    mr = fmaxf(mr, __shfl_xor_sync(~0ull, mr, o));
  }
  __shared__ float sd[8], sr[8];
  int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
  if (!ln) { sd[w] = md; sr[w] = mr; }
  __syncthreads();
  if (!w) {
    md = ln < 8 ? sd[ln] : 0.f;
    mr = ln < 8 ? sr[ln] : 0.f;
    for (int o = 4; o; o >>= 1) {
      md = fmaxf(md, __shfl_xor_sync(~0ull, md, o));
      mr = fmaxf(mr, __shfl_xor_sync(~0ull, mr, o));
    }
    if (!ln) {
      atomicMax(out, __float_as_int(md));
      atomicMax(out + 1, __float_as_int(mr));
    }
  }
}

// Same, but ref holds router-weighted fp32 pairs while lt holds unweighted
// bf16 pairs: diff = ref[i] - pw[i/dd] * lt[i].
__global__ void k_cmp_absmax_pw(const float* __restrict__ ref,
                                const __hip_bfloat16* __restrict__ lt,
                                const float* __restrict__ pw, int npairs, int dd,
                                int* __restrict__ out) {
  float md = 0.f, mr = 0.f;
  int64_t n = (int64_t)npairs * dd;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    float r = ref[i];
    md = fmaxf(md, fabsf(r - pw[i / dd] * __bfloat162float(lt[i])));
    mr = fmaxf(mr, fabsf(r));
  }
  for (int o = 16; o; o >>= 1) {
    md = fmaxf(md, __shfl_xor_sync(~0ull, md, o));
    mr = fmaxf(mr, __shfl_xor_sync(~0ull, mr, o));
  }
  __shared__ float sd[8], sr[8];
  int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
  if (!ln) { sd[w] = md; sr[w] = mr; }
  __syncthreads();
  if (!w) {
    md = ln < 8 ? sd[ln] : 0.f;
    mr = ln < 8 ? sr[ln] : 0.f;
    for (int o = 4; o; o >>= 1) {
      md = fmaxf(md, __shfl_xor_sync(~0ull, md, o));
      mr = fmaxf(mr, __shfl_xor_sync(~0ull, mr, o));
    }
    if (!ln) {
      atomicMax(out, __float_as_int(md));
      atomicMax(out + 1, __float_as_int(mr));
    }
  }
}

