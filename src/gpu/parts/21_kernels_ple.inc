// ================== PLE batched prefill (Phase 3d-1) kernels =================

// --- k_ple_gate over P tokens: block (m, t); kn/qn/U rows are [P, branches*d]
__global__ void k_ple_gate_b(const float* __restrict__ kn,
                             const float* __restrict__ qn,
                             const float* __restrict__ v, float* __restrict__ U, int d,
                             int branches) {
  int m = blockIdx.x, t = blockIdx.y;
  size_t row = ((size_t)t * branches + m) * d;
  const float* knm = kn + row;
  const float* qnm = qn + row;
  const float* vt = v + (size_t)t * d;
  float* Um = U + row;
  __shared__ float sm[1024];
  int i = threadIdx.x;
  float acc = 0.f;
  for (int j = i; j < d; j += blockDim.x) acc += knm[j] * qnm[j];
  sm[i] = acc;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (i < off) sm[i] += sm[i + off];
    __syncthreads();
  }
  float g = sm[0] / sqrtf((float)d);
  g = (g < 0 ? -1.f : 1.f) * sqrtf(fmaxf(fabsf(g), 1e-6f));
  float alpha = 1.f / (1.f + expf(-g));
  for (int j = i; j < d; j += blockDim.x) Um[j] = alpha * vt[j];
}

// --- batched PLE conv (dilation 3, k=4): the 9-slot ring aliases tap back-9
// to the current position, so out[t] = silu(cw0*Un[t] + cw1*Un[t-6] +
// cw2*Un[t-3] + cw3*Un[t]); missing (negative) positions read as zero.
__global__ void k_ple_conv_b(const float* __restrict__ Un,
                             const float* __restrict__ cw, float* __restrict__ out,
                             int P, int n, int base = 0,
                             const float* __restrict__ ring = nullptr) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= (int64_t)P * n) return;
  int t = (int)(i / n), ch = (int)(i % n);
  int a = base + t;  // absolute position
  const float* c4 = cw + (size_t)ch * 4;
  float v = c4[0] * Un[i];
  // taps reach back 6/3 positions; before the batch start they read the ring
  // (holds the last 9 absolute positions), at sequence start they vanish
  if (a >= 6) v += c4[1] * (t >= 6 ? Un[i - (size_t)6 * n]
                                    : ring[(size_t)((a - 6) % 9) * n + ch]);
  if (a >= 3) v += c4[2] * (t >= 3 ? Un[i - (size_t)3 * n]
                                    : ring[(size_t)((a - 3) % 9) * n + ch]);
  v += c4[3] * Un[i];
  out[i] = v / (1.f + expf(-v));
}

// --- PLE ring handoff: ring slot (base+P-9+j)%9 = Un[P-9+j] (negative pos skipped)
__global__ void k_ple_ring_wr(const float* __restrict__ Un, float* __restrict__ ring,
                              int P, int n, int base = 0) {
  int pos = P - 9 + blockIdx.x;
  if (pos < 0) return;
  const float* src = Un + (size_t)pos * n;
  float* dst = ring + (size_t)((base + pos) % 9) * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// --- PLE fp8 staging dequant: raw e4m3 bytes (16 rows x 160 per token, packed
// 2560/token) -> f32 embeddings. Bit-exact with hgn::fp8e4m3_to_f32 * scale.
__device__ inline float fp8e4m3_f32_dev(uint8_t v) {
  uint32_t s = v >> 7, e = (v >> 3) & 0xf, m = v & 7;
  float r;
  if (e == 0) r = ldexpf((float)m, -9);
  else if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
  else r = ldexpf((float)(8 + m), (int)e - 10);
  return s ? -r : r;
}

__global__ void k_ple_fp8_dequant(const uint8_t* __restrict__ src,
                                  float* __restrict__ dst, float scale,
                                  size_t n16) {  // n16 = elements/16
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  uint4 raw = ((const uint4*)src)[i];
  const uint8_t* b = (const uint8_t*)&raw;
  float t[16];
#pragma unroll
  for (int j = 0; j < 16; j++) t[j] = fp8e4m3_f32_dev(b[j]) * scale;
  float4* d4 = (float4*)dst + i * 4;
  d4[0] = make_float4(t[0], t[1], t[2], t[3]);
  d4[1] = make_float4(t[4], t[5], t[6], t[7]);
  d4[2] = make_float4(t[8], t[9], t[10], t[11]);
  d4[3] = make_float4(t[12], t[13], t[14], t[15]);
}

// --- argmax over n floats -> writes int id
__global__ void k_argmax(const float* __restrict__ x, int n, int* __restrict__ out) {
  __shared__ float bv[1024];
  __shared__ int bi[1024];
  int t = threadIdx.x;
  float best = -1e30f;
  int bidx = 0;
  for (int i = t; i < n; i += blockDim.x) {
    if (x[i] > best) {
      best = x[i];
      bidx = i;
    }
  }
  bv[t] = best;
  bi[t] = bidx;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off && bv[t + off] > bv[t]) {
      bv[t] = bv[t + off];
      bi[t] = bi[t + off];
    }
    __syncthreads();
  }
  if (t == 0) out[0] = bi[0];
}

__global__ void k_batch_nll(const float* __restrict__ logits, int vocab,
                            const int* __restrict__ targets,
                            float* __restrict__ losses) {
  __shared__ float reduction[256];
  int row = blockIdx.x;
  int lane = threadIdx.x;
  const float* x = logits + (size_t)row * vocab;
  float maximum = -INFINITY;
  for (int i = lane; i < vocab; i += blockDim.x)
    maximum = fmaxf(maximum, x[i]);
  reduction[lane] = maximum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) reduction[lane] = fmaxf(reduction[lane], reduction[lane + offset]);
    __syncthreads();
  }
  maximum = reduction[0];
  float total = 0.0f;
  for (int i = lane; i < vocab; i += blockDim.x)
    total += expf(x[i] - maximum);
  reduction[lane] = total;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) reduction[lane] += reduction[lane + offset];
    __syncthreads();
  }
  if (lane == 0) losses[row] = maximum + logf(reduction[0]) - x[targets[row]];
}

// vector copy / fill helpers
__global__ void k_repeat4(const float* __restrict__ x, float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) {
    R[i] = x[i];
    R[d + i] = x[i];
    R[2 * d + i] = x[i];
    R[3 * d + i] = x[i];
  }
}

// batched embedding gather for prefill: one block per token, out rows are
// outstride apart (the 4*d Rb row). Same math as k_q4cp_row.
__global__ void k_q4cp_row_b(const uint8_t* __restrict__ codes,
                             const uint8_t* __restrict__ scales,
                             const float* __restrict__ cb,
                             const int* __restrict__ d_tokens, uint64_t cols,
                             uint64_t scale_stride, float* __restrict__ out,
                             uint64_t outstride) {
  uint64_t row = (uint64_t)d_tokens[blockIdx.x];
  out += blockIdx.x * outstride;
  uint64_t g = threadIdx.x;  // one block per token, cols/32 threads
  if (g * 32 >= cols) return;
  const uint8_t* rc = codes + row * (cols / 2) + g * 16;
  const uint8_t* rs = scales + row * scale_stride;
  float s = __half2float(*(const __half*)(rs + g * 2));
  uint4 pk = *(const uint4*)rc;
  uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t v = w[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      out[g * 32 + i * 8 + j] = __ldg(cb + (v & 0xf)) * s;
      v >>= 4;
    }
  }
}

// batched repeat4 over P rows of 4*d contiguous floats (x == R in-place safe)
__global__ void k_repeat4_b(const float* __restrict__ x, float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  size_t base = (size_t)t * 4 * d + c;
  float v = x[base];
  R[base] = v;
  R[base + d] = v;
  R[base + 2 * d] = v;
  R[base + 3 * d] = v;
}
