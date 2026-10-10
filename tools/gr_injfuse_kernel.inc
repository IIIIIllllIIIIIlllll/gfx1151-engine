// [gr-injfuse-begin]  (sed-extracted by tools/injfuse_proto.cu — needs f2bf/bf2f)
// BF16 residual state and BF16 normalized output. The rounded residual is
// accumulated into the norm (scatter/store-before-norm order), halving HC
// state traffic.
__global__ void k_gr_scatter_norm_b_hc_bf16(
    const float* __restrict__ w4, uint16_t* __restrict__ R,
    const float* __restrict__ y, const float* __restrict__ w,
    uint16_t* __restrict__ Rhat, int d, int branches, int P, float eps) {
  int group = blockIdx.x;
  if (group >= P * branches) return;
  int t = group / branches;
  int b = group - t * branches;
  int lane = threadIdx.x;
  uint16_t* Rg = R + (size_t)group * d;
  const float* yg = y + (size_t)t * d;
  const float* wg = w + (size_t)b * d;
  uint16_t* og = Rhat + (size_t)group * d;
  __shared__ float sm[256];
  __shared__ float inject;
  if (lane == 0)
    inject = 2.f / (1.f + expf(-w4[(size_t)t * branches + b] * 0.25f));
  __syncthreads();

  float values[10];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 10; j++) {
    int c = lane + j * 256;
    float rounded = 0.f;
    if (c < d) {
      uint16_t r16 = f2bf(bf2f(Rg[c]) + inject * yg[c]);
      Rg[c] = r16;
      rounded = bf2f(r16);
      ss += rounded * rounded;
    }
    values[j] = rounded;
  }
  sm[lane] = ss;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (lane < off) sm[lane] += sm[lane + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / d + eps);
#pragma unroll
  for (int j = 0; j < 10; j++) {
    int c = lane + j * 256;
    if (c < d) og[c] = f2bf(values[j] * inv * (1.f + wg[c]));
  }
}

// HC inject fusion (QWENOX_HC_INJ_FUSE). w4[t][k] = sum_c Winj[k][c] *
// Rhat[t][c] over K = 4 * d is the next site's inject projection (an N=4
// GEMM that re-reads the whole P x 10240 Rhat). Summed in a fixed order:
// per-branch partials p[t][b][k], then b = 0..3 by inj_psum.
__device__ inline float inj_psum(const float* __restrict__ p) {
  float x = p[0];
  x += p[4];
  x += p[8];
  x += p[12];
  return x;
}
// Partials -> w4 [P][4] for the plain consumers (k_gr_write_b_hc_bf16).
__global__ void k_inj_psum(const float* __restrict__ pin, float* __restrict__ w4, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * 4) return;
  w4[i] = inj_psum(pin + (size_t)(i >> 2) * 16 + (i & 3));
}
// k_gr_scatter_norm_b_hc_bf16 (d <= 2560, 4 branches) + the next site's
// inject partials pout[t][b][k] = sum_{c<d} Winj[k][b*d+c] * Rhat[t][b][c]
// (Rhat as the stored bf16, fp32 accumulate, fixed wave-shuffle + LDS order).
// Block = (T tokens, branch b), 1-D grid 4 * ceil(P / T) with the branch
// fastest (the 4 blocks sharing a token's y run together -> y L2 hits; a
// branch-major grid re-reads y from DRAM 4x and is 2x slower). The block's
// 4 x d bf16 inject slice stays in registers across its T tokens. The scatter / norm arithmetic per (t, b) is
// copied verbatim, so R and Rhat are bit-identical to the unfused kernel for
// the same inject scalar; only w4 changes (summation order vs the GEMM).
// The scalar comes from the GEMM's w4 (pfull [P][4]) or from the previous
// call's partials (pin [P][4][4]); pin != pout (other blocks still read pin).
template <int T>
__global__ void __launch_bounds__(256) k_gr_scatter_norm_inj_bf16(
    const float* __restrict__ pfull, const float* __restrict__ pin,
    uint16_t* __restrict__ R, const float* __restrict__ y,
    const float* __restrict__ w, uint16_t* __restrict__ Rhat,
    const uint16_t* __restrict__ winj, float* __restrict__ pout, int d, int P,
    float eps) {
  const int b = blockIdx.x & 3;
  const int t0 = (blockIdx.x >> 2) * T;
  const int lane = threadIdx.x;
  __shared__ float sm[256];
  __shared__ float red[8][4];
  __shared__ float s_inj[T];
  if (lane < T) {
    const int t = t0 + lane;
    float v = 0.f;
    if (t < P) {
      const float x = pin ? inj_psum(pin + (size_t)t * 16 + b) : pfull[(size_t)t * 4 + b];
      v = 2.f / (1.f + expf(-x * 0.25f));
    }
    s_inj[lane] = v;
  }
  const float* wg = w + (size_t)b * d;
  float wn[10], wr[4][10];
#pragma unroll
  for (int j = 0; j < 10; j++) {
    const int c = lane + j * 256;
    wn[j] = c < d ? 1.f + wg[c] : 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++)
      wr[k][j] = c < d ? bf2f(winj[(size_t)k * 4 * d + (size_t)b * d + c]) : 0.f;
  }
  __syncthreads();
  for (int i = 0; i < T; i++) {
    const int t = t0 + i;
    if (t >= P) break;  // block-uniform
    uint16_t* Rg = R + ((size_t)t * 4 + b) * d;
    const float* yg = y + (size_t)t * d;
    uint16_t* og = Rhat + ((size_t)t * 4 + b) * d;
    const float inject = s_inj[i];
    float values[10];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 10; j++) {
      int c = lane + j * 256;
      float rounded = 0.f;
      if (c < d) {
        uint16_t r16 = f2bf(bf2f(Rg[c]) + inject * yg[c]);
        Rg[c] = r16;
        rounded = bf2f(r16);
        ss += rounded * rounded;
      }
      values[j] = rounded;
    }
    sm[lane] = ss;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
      if (lane < off) sm[lane] += sm[lane + off];
      __syncthreads();
    }
    float inv = rsqrtf(sm[0] / d + eps);
    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 10; j++) {
      int c = lane + j * 256;
      if (c < d) {
        const uint16_t o = f2bf(values[j] * inv * wn[j]);
        og[c] = o;
        const float of = bf2f(o);
#pragma unroll
        for (int k = 0; k < 4; k++) a[k] += of * wr[k][j];
      }
    }
#pragma unroll
    for (int k = 0; k < 4; k++)
#pragma unroll
      for (int o = 16; o; o >>= 1) a[k] += __shfl_xor(a[k], o, 32);
    if ((lane & 31) == 0)
#pragma unroll
      for (int k = 0; k < 4; k++) red[lane >> 5][k] = a[k];
    __syncthreads();
    if (lane < 4) {
      float s = red[0][lane];
#pragma unroll
      for (int wv = 1; wv < 8; wv++) s += red[wv][lane];
      pout[((size_t)t * 4 + b) * 4 + lane] = s;
    }
  }
}
// [gr-injfuse-end]
