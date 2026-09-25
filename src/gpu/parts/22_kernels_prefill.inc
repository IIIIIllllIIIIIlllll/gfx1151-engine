// ================== prefill batch (Phase 3a) kernels =========================

// fp32 -> bf16 (round to nearest even)
__device__ inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

__device__ inline float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

__global__ void k_bf16_to_f32(const uint16_t* __restrict__ x,
                              float* __restrict__ out, uint64_t n) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = bf2f(x[i]);
}

// Direct embedding gather into the four-branch BF16 hyper-connection state.
__global__ void k_q4cp_row_b_hc_bf16(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const float* __restrict__ cb, const int* __restrict__ d_tokens,
    uint64_t cols, uint64_t scale_stride, uint16_t* __restrict__ out) {
  uint64_t row = (uint64_t)d_tokens[blockIdx.x];
  out += (size_t)blockIdx.x * 4 * cols;
  uint64_t g = threadIdx.x;
  if (g * 32 >= cols) return;
  const uint8_t* rc = codes + row * (cols / 2) + g * 16;
  const uint8_t* rs = scales + row * scale_stride;
  float s = __half2float(*(const __half*)(rs + g * 2));
  uint4 pk = *(const uint4*)rc;
  uint32_t packed[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t v = packed[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int c = (int)(g * 32 + i * 8 + j);
      uint16_t value = f2bf(__ldg(cb + (v & 0xf)) * s);
      out[c] = value;
      out[cols + c] = value;
      out[2 * cols + c] = value;
      out[3 * cols + c] = value;
      v >>= 4;
    }
  }
}

__global__ void k_rmsnorm_zc_grouped_bf16(
    const uint16_t* __restrict__ x, const float* __restrict__ w,
    uint16_t* __restrict__ out, int gsize, float eps, int ngroups) {
  const uint16_t* xg = x + (size_t)blockIdx.x * gsize;
  const float* wg = w + (size_t)(blockIdx.x % ngroups) * gsize;
  uint16_t* og = out + (size_t)blockIdx.x * gsize;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < gsize; i += blockDim.x) {
    float value = bf2f(xg[i]);
    ss += value * value;
  }
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / gsize + eps);
  for (int i = t; i < gsize; i += blockDim.x)
    og[i] = f2bf(bf2f(xg[i]) * inv * (1.f + wg[i]));
}

// --- Q4C-P full dequant to bf16: one thread per 32-col group
__global__ void k_dequant_q4cp_bf16(const uint8_t* __restrict__ codes,
                                    const uint8_t* __restrict__ scales,
                                    const float* __restrict__ cb,
                                    uint16_t* __restrict__ out, uint64_t rows,
                                    uint64_t cols, uint64_t scale_stride) {
  uint64_t gpr = cols / 32;
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= rows * gpr) return;
  uint64_t r = gid / gpr, g = gid % gpr;
  const uint8_t* rc = codes + r * (cols / 2) + g * 16;
  float s = __half2float(*(const __half*)(scales + r * scale_stride + g * 2));
  uint16_t* o = out + r * cols + g * 32;
#pragma unroll
  for (int b = 0; b < 16; b++) {
    uint32_t v = rc[b];
    o[2 * b] = f2bf(__ldg(cb + (v & 0xf)) * s);
    o[2 * b + 1] = f2bf(__ldg(cb + (v >> 4)) * s);
  }
}

// --- q8g64 full dequant to bf16: w'[r][c] = code*s + m (one thread per 64-col group)
__global__ void k_dequant_q8g64_bf16(const uint8_t* __restrict__ w,
                                     uint16_t* __restrict__ out, uint64_t rows,
                                     uint64_t cols) {
  uint64_t gpr = cols / 64;
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= rows * gpr) return;
  uint64_t r = gid / gpr, g = gid % gpr;
  const uint8_t* rp = w + r * (cols + gpr * 4);
  const uint8_t* codes = rp + g * 64;
  const __half* sm = (const __half*)(rp + cols + g * 4);
  float s = __half2float(sm[0]), mn = __half2float(sm[1]);
  uint16_t* o = out + r * cols + g * 64;
#pragma unroll
  for (int j = 0; j < 64; j++) o[j] = f2bf(codes[j] * s + mn);
}

// --- fp32 [P,n] (row stride xs) -> compact bf16 [P,n]
__global__ void k_f32_to_bf16(const float* __restrict__ x, uint16_t* __restrict__ out,
                              uint64_t xs, int P, int n) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= (uint64_t)P * n) return;
  uint64_t t = i / n, c = i % n;
  out[i] = f2bf(x[t * xs + c]);
}

// 4 floats/thread variant; requires n % 4 == 0, xs % 4 == 0 and 16B-aligned
// base pointers (host-checked, scalar kernel is the fallback)
__global__ void k_f32_to_bf16_v4(const float* __restrict__ x,
                                 uint16_t* __restrict__ out, uint64_t xs, int P,
                                 int n) {
  uint64_t n4 = (uint64_t)n / 4;
  uint64_t i4 = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i4 >= (uint64_t)P * n4) return;
  uint64_t t = i4 / n4;
  uint64_t c = (i4 - t * n4) * 4;
  float4 v = *(const float4*)(x + t * xs + c);
  uint16_t o[4] = {f2bf(v.x), f2bf(v.y), f2bf(v.z), f2bf(v.w)};
  uint2 pk;
  pk.x = (uint32_t)o[0] | ((uint32_t)o[1] << 16);
  pk.y = (uint32_t)o[2] | ((uint32_t)o[3] << 16);
  *(uint2*)(out + i4 * 4) = pk;
}

// Compact BF16 [P,512] V rows into block-transposed
// [block][kvh][dim][slot] storage. Each thread owns one dimension of one KV
// head and packs the four token slots, so the WMMA V gather can issue one
// contiguous 64-bit load per four-key group. The leading partial block keeps
// its slots < base (earlier rows). Slots >= base + P of the trailing partial
// block are written as ZERO: k_qsa_wmma<true> loads whole 4-slot groups and
// multiplies the masked slots by p = 0 inside the WMMA, and on gfx1151 the
// result is NOT independent of those V bits (stale finite values changed
// the output LSBs; A5 off-vs-paged divergence, 2026-09-24). Zero keeps the
// output a function of valid rows only, identical for paged / unpaged.
__global__ void k_f32_to_bf16_v4_bt(const float* __restrict__ x,
                                    uint16_t* __restrict__ out, uint64_t xs,
                                    int P, int n, int base) {
  const int dims = n / 2;
  const int b0 = base / 4;
  const int b1 = (base + P + 3) / 4;
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  uint64_t work = (uint64_t)(b1 - b0) * n;
  if (gid >= work) return;
    int b = b0 + (int)(gid / n);
    int hd = (int)(gid % n);
  int kvh = hd / dims;
  int dim = hd - kvh * dims;
  uint16_t v[4] = {};
  int valid = 0;
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    int abs = b * 4 + slot;
    if (abs >= base) {
      if (abs < base + P) v[slot] = f2bf(x[(size_t)(abs - base) * xs + kvh * (n / 2) + dim]);
      valid |= 1 << slot;
    }
  }
  uint16_t* dst = out + (((size_t)b * 2 + kvh) * (n / 2) + dim) * 4;
  if (valid == 0xf) {
    uint2 pk;
    pk.x = (uint32_t)v[0] | ((uint32_t)v[1] << 16);
    pk.y = (uint32_t)v[2] | ((uint32_t)v[3] << 16);
    *(uint2*)dst = pk;
  } else {
#pragma unroll
    for (int slot = 0; slot < 4; ++slot)
      if (valid & (1 << slot)) dst[slot] = v[slot];
  }
}

// Rebuild block-transposed V for absolute rows [r0, r1) from the row-major
// BF16 cache (kvsnap restore: the snapshot stores only the row-major cache).
// Same packing as k_f32_to_bf16_v4_bt (n = 512); bits are copied, so the
// result equals what the live prefill/decode stores wrote. Slots < r0 are left
// untouched; slots >= r1 of the trailing partial block are zeroed (same rule
// as k_f32_to_bf16_v4_bt).
__global__ void k_bf16_rows_to_bt(const uint16_t* __restrict__ vc,
                                  uint16_t* __restrict__ vct, int r0, int r1) {
  const int b0 = r0 / 4, b1 = (r1 + 3) / 4;
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= (uint64_t)(b1 - b0) * 512) return;
  const int b = b0 + (int)(gid / 512), i = (int)(gid % 512);
  uint16_t* dst = vct + ((size_t)b * 512 + i) * 4;
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int t = b * 4 + slot;
    if (t >= r0) dst[slot] = t < r1 ? vc[(size_t)t * 512 + i] : (uint16_t)0;
  }
}

// Debug (GDEC_VCT_CHECK): count logical rows [0, r1) whose transposed-V slot
// differs from the row-major BF16 V. out = {count, first row, last row}.
__global__ void k_vct_check(const uint16_t* __restrict__ vc,
                            const uint16_t* __restrict__ vct,
                            const int* __restrict__ ptab, int r1,
                            unsigned* __restrict__ out) {
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= (uint64_t)r1 * 512) return;
  const int t = (int)(gid / 512), i = (int)(gid % 512), row = kv_row(ptab, t);
  if (vct[((size_t)(row >> 2) * 512 + i) * 4 + (t & 3)] != vc[(size_t)row * 512 + i]) {
    atomicAdd(&out[0], 1u);
    atomicMin(&out[1], (unsigned)t);
    atomicMax(&out[2], (unsigned)t);
  }
}
// Zero the transposed-V slots of logical rows [r0, r1) (grid (r1-r0)*512/256).
__global__ void k_vct_zero_slots(uint16_t* __restrict__ vct,
                                 const int* __restrict__ ptab, int r0, int r1,
                                 uint16_t val = 0) {
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= (uint64_t)(r1 - r0) * 512) return;
  const int t = r0 + (int)(gid / 512), i = (int)(gid % 512), row = kv_row(ptab, t);
  if (val == 0xFFFF) {  // debug: per-element pseudo-random finite bf16 (+-2^-7..2^7)
    unsigned h = (unsigned)(t * 512 + i) * 2654435761u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    val = (uint16_t)((h & 0x807f) | ((0x78 + (h >> 8) % 15) << 7));
  }
  vct[((size_t)(row >> 2) * 512 + i) * 4 + (t & 3)] = val;
}
// Debug (GDEC_ATTN_HASH): order-independent hashes. Logical rows [r0, r1) of a
// row-major bf16 cache (512 per row) / of the transposed V (per slot), and a
// raw float buffer.
__device__ __forceinline__ unsigned long long dbg_mix(unsigned long long x) {
  x ^= x >> 31; x *= 0x9E3779B97F4A7C15ull; x ^= x >> 29; return x;
}
__global__ void k_dbg_hash_rows(const uint16_t* __restrict__ c, const uint16_t* __restrict__ vct,
                                const int* __restrict__ ptab, int r0, int r1,
                                unsigned long long* __restrict__ out) {
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= (uint64_t)(r1 - r0) * 512) return;
  const int t = r0 + (int)(gid / 512), i = (int)(gid % 512), row = kv_row(ptab, t);
  const uint16_t v = vct ? vct[((size_t)(row >> 2) * 512 + i) * 4 + (t & 3)]
                         : c[(size_t)row * 512 + i];
  atomicAdd(out, dbg_mix(((unsigned long long)t * 512 + i) * 65537ull + v + 1));
}
__global__ void k_dbg_hash_f32(const float* __restrict__ x, uint64_t n,
                               unsigned long long* __restrict__ out) {
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= n) return;
  atomicAdd(out, dbg_mix(gid * 0x100000001ull + __float_as_uint(x[gid]) + 1));
}
__global__ void k_dbg_hash_i32(const int* __restrict__ x, uint64_t n,
                               unsigned long long* __restrict__ out) {
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= n) return;
  atomicAdd(out, dbg_mix(gid * 0x100000001ull + (unsigned)x[gid] + 1));
}
// Debug (GDEC_VCT_CHECK): non-finite bf16 values in n raw uint16 of vct.
// out = {count, first physical row, last physical row}.
__global__ void k_vct_nonfinite(const uint16_t* __restrict__ vct, uint64_t n,
                                unsigned* __restrict__ out) {
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= n) return;
  if ((vct[gid] & 0x7f80u) == 0x7f80u) {
    const unsigned row = (unsigned)((gid / 2048) * 4 + gid % 4);  // [blk][512][4]
    atomicAdd(&out[0], 1u);
    atomicMin(&out[1], row);
    atomicMax(&out[2], row);
  }
}

// --- batched GR combine: x[t*d+i] = mean_b G[t][b*d+i] * Rhat[t][b*d+i]
template <bool Fused = false>
__global__ void k_gr_combine_b(const float* __restrict__ G, const float* __restrict__ Rhat,
                               float* __restrict__ x, int d, int branches, int P,
                               uint16_t* __restrict__ bf16 = nullptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * d) return;
  int t = i / d, c = i - t * d;
  const float* Gt = G + (size_t)t * branches * d;
  const float* Rt = Rhat + (size_t)t * branches * d;
  float acc = 0.f;
  for (int b = 0; b < branches; b++) {
    float gate = Gt[b * d + c];
    if (Fused) gate = 1.f / (1.f + expf(-gate));
    acc += gate * Rt[b * d + c];
  }
  float value = acc / branches;
  x[i] = value;
  if (Fused) bf16[i] = f2bf(value);
}

// BF16 HC state; one block per token.
__global__ void k_gr_combine_b_hc_bf16(
    const float* __restrict__ G, const uint16_t* __restrict__ Rhat,
    float* __restrict__ x, int d, int branches, int P,
    uint16_t* __restrict__ xbf16) {
  int t = blockIdx.x;
  if (t >= P) return;
  const float* Gt = G + (size_t)t * branches * d;
  const uint16_t* Rt = Rhat + (size_t)t * branches * d;
  float* xt = x + (size_t)t * d;
  uint16_t* x16t = xbf16 + (size_t)t * d;
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    float acc = 0.f;
    for (int b = 0; b < branches; b++) {
      float gate = 1.f / (1.f + expf(-Gt[b * d + c]));
      acc += gate * bf2f(Rt[b * d + c]);
    }
    float value = acc / branches;
    xt[c] = value;
    x16t[c] = f2bf(value);
  }
}

// --- batched GR write: R[t][b*d+i] += 2*sigmoid(w4[t][b]/4) * y[t*d+i]
__global__ void k_gr_write_b(const float* __restrict__ w4, float* __restrict__ R,
                             const float* __restrict__ y, int d, int branches, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * d) return;
  int t = i / d, c = i - t * d;
  float* Rt = R + (size_t)t * branches * d;
  const float* wt = w4 + t * branches;
  for (int b = 0; b < branches; b++) {
    float s = 2.f / (1.f + expf(-wt[b] * 0.25f));
    Rt[b * d + c] += s * y[i];
  }
}

__global__ void k_gr_write_b_hc_bf16(const float* __restrict__ w4,
                                     uint16_t* __restrict__ R,
                                     const float* __restrict__ y, int d,
                                     int branches, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * d) return;
  int t = i / d, c = i - t * d;
  uint16_t* Rt = R + (size_t)t * branches * d;
  const float* wt = w4 + t * branches;
  for (int b = 0; b < branches; b++) {
    float s = 2.f / (1.f + expf(-wt[b] * 0.25f));
    Rt[b * d + c] = f2bf(bf2f(Rt[b * d + c]) + s * y[i]);
  }
}

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

// One-block-per-token variant: y is staged in LDS once instead of being
// re-read by each of the 4 branch blocks. Per-branch arithmetic (inject,
// bf16 round-trip, 256-lane reduction tree, norm) is kept identical to
// k_gr_scatter_norm_b_hc_bf16, so results are bit-exact.
__global__ void k_gr_scatter_norm_b_hc_bf16_f4(
    const float* __restrict__ w4, uint16_t* __restrict__ R,
    const float* __restrict__ y, const float* __restrict__ w,
    uint16_t* __restrict__ Rhat, int d, int branches, int P, float eps) {
  int t = blockIdx.x;
  if (t >= P) return;
  int lane = threadIdx.x;
  const float* yg = y + (size_t)t * d;
  __shared__ float sm[256];
  __shared__ float s_y[2560];
  __shared__ float inject;
  for (int c = lane; c < d; c += 256) s_y[c] = yg[c];
  __syncthreads();
  for (int b = 0; b < branches; b++) {
    uint16_t* Rg = R + ((size_t)t * branches + b) * d;
    const float* wg = w + (size_t)b * d;
    uint16_t* og = Rhat + ((size_t)t * branches + b) * d;
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
        uint16_t r16 = f2bf(bf2f(Rg[c]) + inject * s_y[c]);
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
    __syncthreads();
  }
}
// Fuse a hyper-connection residual update with the following grouped norm.
// The model uses d=2560 and 1024-thread blocks, so each lane retains at most
// three updated values across the reduction instead of reloading R.
template <bool Bf16 = false>
__global__ void k_gr_scatter_norm_b(const float* __restrict__ w4,
                                    float* __restrict__ R,
                                    const float* __restrict__ y,
                                    const float* __restrict__ w,
                                    float* __restrict__ Rhat, int d,
                                    int branches, int P, float eps,
                                    uint16_t* __restrict__ bf16 = nullptr) {
  int group = blockIdx.x;
  if (group >= P * branches) return;
  int t = group / branches;
  int b = group - t * branches;
  int lane = threadIdx.x;
  float* Rg = R + (size_t)group * d;
  const float* yg = y + (size_t)t * d;
  const float* wg = w + (size_t)b * d;
  float* og = Rhat + (size_t)group * d;
  __shared__ float sm[1024];
  __shared__ float inject;
  if (lane == 0)
    inject = 2.f / (1.f + expf(-w4[(size_t)t * branches + b] * 0.25f));
  __syncthreads();

  float values[3] = {0.f, 0.f, 0.f};
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 3; j++) {
    int c = lane + j * blockDim.x;
    if (c < d) {
      float value = Rg[c] + inject * yg[c];
      Rg[c] = value;
      values[j] = value;
      ss += value * value;
    }
  }
  sm[lane] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (lane < off) sm[lane] += sm[lane + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / d + eps);
#pragma unroll
  for (int j = 0; j < 3; j++) {
    int c = lane + j * blockDim.x;
    if (c < d) {
      float value = values[j] * inv * (1.f + wg[c]);
      og[c] = value;
      if (Bf16) bf16[(size_t)group * d + c] = f2bf(value);
    }
  }
}

// --- batched RoPE: grid.y = token row t (pos = t); rotation identical to k_rope
// --- RoPE pow table: g_rope_tab[i] = pow(theta, -2.0*i/rotary), filled on
// device with the same libm pow and the same argument expression k_rope_b
// used to compute inline -> per-element angles are bit-identical, the table
// just removes 33M redundant pow calls per prefill layer.
__device__ double g_rope_tab[64];
__global__ void k_rope_tab_init(double theta, int rotary) {
  int i = threadIdx.x;
  if (i < rotary / 2) g_rope_tab[i] = pow(theta, -2.0 * i / rotary);
}
static void ensure_rope_tab(double theta, int rotary) {
  static double cur_theta = -1;
  static int cur_rotary = -1;
  if (cur_theta != theta || cur_rotary != rotary) {
    k_rope_tab_init<<<1, 64, 0, g_str>>>(theta, rotary);
    cur_theta = theta;
    cur_rotary = rotary;
  }
}

__global__ void k_rope_b(float* __restrict__ vec, int nheads, int dh, int rotary,
                         double theta, int base = 0) {
  int pos = blockIdx.y;      // batch-local row
  int apos = base + pos;     // absolute position (rope angle)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // over nheads * rotary/2
  int half = rotary / 2;
  if (idx >= nheads * half) return;
  int h = idx / half, i = idx % half;
  double ang = apos * g_rope_tab[i];
  float cs = (float)cos(ang), sn = (float)sin(ang);
  float* p = vec + ((size_t)pos * nheads + h) * dh;
  float x0 = p[i], x1 = p[i + half];
  p[i] = x0 * cs - x1 * sn;
  p[i + half] = x0 * sn + x1 * cs;
  (void)theta;
}

// Per-(token, i) rope cos/sin table: one fp64 transcendental pair per (pos, i)
// instead of one per (head, pos, i) — 24x less fp64 work on gfx1151, and the
// consumers keep full thread parallelism. Values are bit-identical to the
// inline expressions in k_rope_b / the fused QSA prep kernels (same libm,
// same argument expression).
__global__ void k_rope_cs(float2* __restrict__ tab, int base, int P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= P * 32) return;
  double ang = (base + (idx >> 5)) * g_rope_tab[idx & 31];
  tab[idx] = make_float2((float)cos(ang), (float)sin(ang));
}

// M-RoPE debug dump: the 32 cos/sin pairs k_rope would use this decode step
// (same inline fp64 expression), for rope position *d_pos + *d_rdelta.
__global__ void k_rope_dump(float2* __restrict__ out, int* __restrict__ rpos,
                            const int* __restrict__ d_pos,
                            const int* __restrict__ d_rdelta, double theta) {
  int i = threadIdx.x;
  int pos = *d_pos + (d_rdelta ? *d_rdelta : 0);
  if (i == 0) *rpos = pos;
  if (i < 32) {
    double ang = pos * pow(theta, -2.0 * i / 64.0);
    out[i] = make_float2((float)cos(ang), (float)sin(ang));
  }
}

// M-RoPE slot interleave (mrope_section = [11, 11, 10], see
// reference/flash_next_config.json and tools/mrope_ref.py): freq index i
// takes its position from row H when i % 3 == 1 and i < 3*11 (1,4,...,31),
// from row W when i % 3 == 2 and i < 3*10 (2,5,...,29), else from row T.
static int mrope_slot_row(int i) {
  if (i % 3 == 1 && i < 33) return 1;
  if (i % 3 == 2 && i < 30) return 2;
  return 0;
}

// --- QSA flash prefill: full causal attention over P positions, fp32.
// Tiled: block = 128 threads (4 warps) x 16 q rows x one head; positions
// stream in 8-wide chunks. q and K chunks live in LDS (pitch 260 floats:
// float4-aligned, bank-spread; an LDS b128 wavefront costs ~4 cycles
// regardless of broadcast, measured on gfx1151); V is read straight from
// the global cache (L2) with contiguous 4-row-broadcast wavefronts.
// 24.4 KiB LDS per block -> two blocks per WGP so one block's staging and
// syncthreads overlap the other's compute.
// Online softmax per 8-score chunk; a row's scores live in the 8 lanes
// (r%4)*8..+7 of its warp; e crosses to the O accumulation via shfl.
// fp32 throughout.
// SharedV specializes sparse prefill for 12 query heads per KV head: 12 Q
// rows plus 8 K/V rows use 29120 bytes LDS. The fourth warp stages data but
// skips head arithmetic. Score, softmax, and V accumulation order are unchanged.
#define QSA_TR 16   // q rows per block
#define QSA_KT 8    // positions per chunk
#define QSA_LP 260  // LDS row pitch (floats, float4-aligned, bank-spread)
#define QSA_DEC_NSPLIT 32  // decode flash-decoding splits (grid.x = splits, y = kvh)
template <bool Sparse = false, bool SharedV = false, typename KVT = float,
          bool Split = false>
__global__ void k_qsa_flash(const float* __restrict__ q, const KVT* __restrict__ kc,
                            const KVT* __restrict__ vc, float* __restrict__ out, int P,
                            int dh, int hkv, int hq, const int* selected = nullptr,
                            int first = 0, const int* pos = nullptr,
                            float* pacc = nullptr, float* pml = nullptr,
                            int kbase = 0,
                            const int* __restrict__ ptab = nullptr) {
  static_assert(!SharedV || Sparse, "shared V is a sparse-attention specialization");
  const int h = blockIdx.y;
  const int kvh = Sparse ? h : h / (hq / hkv);
  const int token = pos ? *pos : first + (int)blockIdx.x;
  // Dense decode tokens (< 2051) are handled here too: ntok = token+1 and
  // qsa_source_pos degrades to identity, so the split partials cover the
  // whole visible range (empty splits write m=-1e30, l=0 and merge as 0).
  const int* blocks =
      Sparse ? selected + (size_t)(Split ? 0 : blockIdx.x) * 512 : nullptr;
  const int ts = blockIdx.x * QSA_TR;   // first q row of this tile
  // Sparse: q/out rows are batch-local (token - kbase); token stays absolute
  // for ntok/source bookkeeping. Dense: rows local, keys/mask absolute (kbase).
  const int qrow = token - kbase;
  // selected == nullptr (GDEC_QSA_DENSE decode): every visible position.
  const int ntok = Sparse ? ((token < 2051 || !selected) ? token + 1
                                                         : 2048 + (token + 1) % 4)
                          : min(kbase + P, kbase + ts + QSA_TR);
  constexpr int QR = SharedV ? 12 : QSA_TR;
  __shared__ float qs[QR][QSA_LP];
  __shared__ float kt[QSA_KT][QSA_LP];
  __shared__ float vt[SharedV ? QSA_KT : 1][SharedV ? QSA_LP : 1];
  const int tid = threadIdx.x;
  // stage q tile (rows >= P read as zero; those rows never write out)
  for (int i4 = tid; i4 < QR * (dh / 4); i4 += 128) {
    int rl = i4 / (dh / 4), d4 = i4 % (dh / 4);
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (Sparse ? rl < hq / hkv : ts + rl < P)
      v = *(const float4*)(q + (Sparse ? (size_t)(pos ? 0 : qrow) * hq + kvh * (hq / hkv) + rl
                                      : (size_t)(ts + rl) * hq + h) * dh + d4 * 4);
    *(float4*)&qs[rl][d4 * 4] = v;
  }
  const int w = tid / warpSize, lane = tid % warpSize;
  const int rl = lane / QSA_KT;      // row within warp: rows 4w+rl
  const int pl = lane % QSA_KT;      // position within chunk (score phase)
  const int row = 4 * w + rl;
  const int rg = Sparse ? token : ts + row;
  float m_run = -1e30f, l_run = 0.f; // online softmax state for row rg
  float4 acc[8];                     // dims dg*4 + 32*f, dg = lane % 8
  for (int f = 0; f < 8; f++) acc[f] = make_float4(0.f, 0.f, 0.f, 0.f);
  const float scale = 1.f / sqrtf((float)dh);
  // K chunk staging slice: thread tid stages row tid/16, floats (tid%16)*16..
  const int spi = tid / 16, spf = (tid % 16) * 16;
  const KVT* kg = kc + (size_t)kvh * dh + spf;
  float4 st0, st1, st2, st3;
  if (!Sparse) {
    int source = min(spi, ntok - 1);
    if (Sparse) source = qsa_source_pos(source, token + 1, blocks);
    size_t off = (size_t)kv_row(ptab, source) * hkv * dh;
    st0 = kv_ld4(kg + off);
    st1 = kv_ld4(kg + off + 4);
    st2 = kv_ld4(kg + off + 8);
    st3 = kv_ld4(kg + off + 12);
    if (spi >= ntok) st0 = st1 = st2 = st3 = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  // Split (decode flash-decoding): this block covers a contiguous slice of
  // the logical position space; unnormalized partials go to pacc/pml and are
  // merged by k_qsa_flash_combine.
  int p_lo = 0, p_hi = ntok;
  if (Split) {
    int nchunks = (ntok + QSA_KT - 1) / QSA_KT;
    int cpb = (nchunks + QSA_DEC_NSPLIT - 1) / QSA_DEC_NSPLIT;
    p_lo = blockIdx.x * cpb * QSA_KT;
    p_hi = min(ntok, p_lo + cpb * QSA_KT);
  }
  for (int p0 = p_lo; p0 < p_hi; p0 += QSA_KT) {
    // make the prefetched chunk visible, then prefetch the next one
    if (Sparse) {
      // Sparse address bookkeeping plus the 16-float prefetch spills VGPRs.
      int source = kv_row(ptab, qsa_source_pos(min(p0 + spi, ntok - 1), token + 1, blocks));
      const KVT* src = kg + (size_t)source * hkv * dh;
      *(float4*)&kt[spi][spf] = kv_ld4(src);
      *(float4*)&kt[spi][spf + 4] = kv_ld4(src + 4);
      *(float4*)&kt[spi][spf + 8] = kv_ld4(src + 8);
      *(float4*)&kt[spi][spf + 12] = kv_ld4(src + 12);
      if (SharedV) {
        const KVT* vs = vc + ((size_t)source * hkv + kvh) * dh + spf;
        *(float4*)&vt[spi][spf] = kv_ld4(vs);
        *(float4*)&vt[spi][spf + 4] = kv_ld4(vs + 4);
        *(float4*)&vt[spi][spf + 8] = kv_ld4(vs + 8);
        *(float4*)&vt[spi][spf + 12] = kv_ld4(vs + 12);
      }
    } else {
      *(float4*)&kt[spi][spf] = st0;
      *(float4*)&kt[spi][spf + 4] = st1;
      *(float4*)&kt[spi][spf + 8] = st2;
      *(float4*)&kt[spi][spf + 12] = st3;
    }
    __syncthreads();
    if (!Sparse && p0 + QSA_KT < ntok) {
      int pos = p0 + QSA_KT + spi;
      int source = min(pos, ntok - 1);
      if (Sparse) source = qsa_source_pos(source, token + 1, blocks);
      size_t off = (size_t)kv_row(ptab, source) * hkv * dh;
      st0 = kv_ld4(kg + off);
      st1 = kv_ld4(kg + off + 4);
      st2 = kv_ld4(kg + off + 8);
      st3 = kv_ld4(kg + off + 12);
      if (pos >= ntok) st0 = st1 = st2 = st3 = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    // scores: lane (rl, pl) owns (row 4w+rl, pos p0+pl); float4 reads are
    // bank-conflict-free (4 rl / 8 pl groups tile all 32 banks)
    if (!SharedV || row < QR) {
      const float4* q4 = (const float4*)&qs[4 * w + rl][0];
      const float4* k4 = (const float4*)&kt[pl][0];
      float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);  // 4 partials: break the
      for (int d4 = 0; d4 < dh / 4; d4++) {         // serial FMA chain (ILP)
        float4 qv = q4[d4];
        float4 kv = k4[d4];
        s4.x += qv.x * kv.x;
        s4.y += qv.y * kv.y;
        s4.z += qv.z * kv.z;
        s4.w += qv.w * kv.w;
      }
      float s = ((s4.x + s4.y) + (s4.z + s4.w)) * scale;
      bool masked = Sparse ? p0 + pl >= ntok : p0 + pl > kbase + rg;
      if (masked) s = -1e30f;
      // chunk max/sum over the 8 lanes of this row (width-8 shfl groups)
      float cm = s;
      for (int off = QSA_KT / 2; off > 0; off >>= 1)
        cm = fmaxf(cm, __shfl_down_sync(~0ull, cm, off, QSA_KT));
      cm = __shfl_sync(~0ull, cm, 0, QSA_KT);
      float mn = fmaxf(m_run, cm);
      float c = expf(m_run - mn);    // 1.0 when both are -1e30 (masked chunk)
      float e = expf(s - mn);
      if (masked) e = 0.f;
      float cl = e;
      for (int off = QSA_KT / 2; off > 0; off >>= 1)
        cl += __shfl_down_sync(~0ull, cl, off, QSA_KT);
      cl = __shfl_sync(~0ull, cl, 0, QSA_KT);
      l_run = l_run * c + cl;
      m_run = mn;
      // O accumulation: lane (rl, dg) owns dims dg*4 + 32*f. Rescale by c once
      // per chunk, then accumulate e_p * V_p; e_p comes from lane rl*8+p.
      const int dg = lane % QSA_KT;
      for (int f = 0; f < 8; f++) {
        acc[f].x *= c;
        acc[f].y *= c;
        acc[f].z *= c;
        acc[f].w *= c;
      }
      auto accumulate_v = [&](int p) {
        float ev = __shfl_sync(~0ull, e, rl * QSA_KT + p);
        // clamp the row: masked positions multiply by ev == 0 (their cache
        // rows may be stale/NaN or, in unit tests, out of bounds)
        if (SharedV) {
          const float4* v4 = (const float4*)&vt[p][dg * 4];
          for (int f = 0; f < 8; f++) {
            float4 vv = v4[f * 8];
            acc[f].x += ev * vv.x;
            acc[f].y += ev * vv.y;
            acc[f].z += ev * vv.z;
            acc[f].w += ev * vv.w;
          }
        } else {
          int pr = min(p0 + p, ntok - 1);
          if (Sparse) pr = qsa_source_pos(pr, token + 1, blocks);
          const KVT* vg = vc + ((size_t)kv_row(ptab, pr) * hkv + kvh) * dh + dg * 4;
          for (int f = 0; f < 8; f++) {
            float4 vv = kv_ld4(vg + f * 32);
            acc[f].x += ev * vv.x;
            acc[f].y += ev * vv.y;
            acc[f].z += ev * vv.z;
            acc[f].w += ev * vv.w;
          }
        }
      };
      if (Sparse) {
#pragma unroll 1
        for (int p = 0; p < QSA_KT; p++) accumulate_v(p);
      } else {
#pragma unroll
        for (int p = 0; p < QSA_KT; p++) accumulate_v(p);
      }
    }
    __syncthreads();
  }
  if (Split) {
    // unnormalized partial: acc is relative to this slice's m_run/l_run
    if (row < hq / hkv) {
      size_t base =
          ((size_t)h * QSA_DEC_NSPLIT + blockIdx.x) * (hq / hkv) + row;
      float* pa = pacc + base * dh + (lane % QSA_KT) * 4;
      for (int f = 0; f < 8; f++) *(float4*)(pa + 32 * f) = acc[f];
      if (lane % QSA_KT == 0) {
        pml[base * 2] = m_run;
        pml[base * 2 + 1] = l_run;
      }
    }
    return;
  }
  if (Sparse ? row < hq / hkv : rg < P) {
    float inv = 1.f / l_run;
    float* o = out + (Sparse ? (size_t)(pos ? 0 : qrow) * hq + kvh * (hq / hkv) + row
                             : (size_t)rg * hq + h) * dh + (lane % QSA_KT) * 4;
    for (int f = 0; f < 8; f++) {
      float4 v = acc[f];
      v.x *= inv;
      v.y *= inv;
      v.z *= inv;
      v.w *= inv;
      *(float4*)(o + 32 * f) = v;
    }
  }
}

// --- k_qsa_flash_combine: merge QSA_DEC_NSPLIT split partials (decode) ------
// Standard flash-decoding merge: out = sum_s exp(m_s - M) * acc_s / L with
// L = sum_s exp(m_s - M) * l_s. One block per (q row, kv head).
__global__ void k_qsa_flash_combine(const float* __restrict__ pml,
                                    const float* __restrict__ pacc,
                                    float* __restrict__ out, int qpg, int dh,
                                    const int* __restrict__ d_pos) {
  (void)d_pos;  // dense tokens now take this path too (no <2051 early-out)
  const int h = blockIdx.y, row = blockIdx.x;
  const float* pm = pml + ((size_t)h * QSA_DEC_NSPLIT * qpg + row) * 2;
  float M = -1e30f, L = 0.f;
  for (int s = 0; s < QSA_DEC_NSPLIT; s++) {
    float m = pm[(size_t)s * qpg * 2], l = pm[(size_t)s * qpg * 2 + 1];
    float mn = fmaxf(M, m);
    L = L * expf(M - mn) + l * expf(m - mn);
    M = mn;
  }
  float inv = 1.f / L;
  for (int d4 = threadIdx.x; d4 < dh / 4; d4 += blockDim.x) {
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < QSA_DEC_NSPLIT; s++) {
      float w = expf(pm[(size_t)s * qpg * 2] - M);
      float4 a = *(const float4*)(pacc +
                                  (((size_t)h * QSA_DEC_NSPLIT + s) * qpg + row) * dh +
                                  d4 * 4);
      o.x += w * a.x;
      o.y += w * a.y;
      o.z += w * a.z;
      o.w += w * a.w;
    }
    o.x *= inv;
    o.y *= inv;
    o.z *= inv;
    o.w *= inv;
    *(float4*)(out + ((size_t)h * qpg + row) * dh + d4 * 4) = o;  // token 0
  }
}

// --- k_qsa_flash_bf16: BF16 LDS variant of k_qsa_flash<true, true> ----------
// Used only by the BF16 KV mode (!qsa_kv_fp32) sparse prefill SharedV launch;
// the FP32 sparse path above is untouched. qs/kt/vt tiles are stored bf16
// (fp32 global -> f2bf RNE on stage) for ~15KB LDS and 4 blocks/CU (vs 2 at
// 29120B); QK uses fdot2_f32_bf16 with fp32 accumulators; AV stays fp32 FMA
// on bf16->fp32 expanded V. Row pitch QSA_LPB bf16: 264 (=528B) keeps the
// fp32 pitch's 4-bank row rotation (528 mod 128 = 16), so the 8-row kt access
// pattern stays conflict-free. Measured (tools/qsa_bench.cu): 32K sparse
// kernel 1179->459ms, max abs diff 2.8e-3 vs the fp32 staged kernel.
// Override with -DQSA_LPB=... for pitch experiments.
#ifndef QSA_LPB
#define QSA_LPB 264
#endif

__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  return (uint32_t)f2bf(a) | ((uint32_t)f2bf(b) << 16);
}

// stage 16 K/V elements from global into a bf16 LDS row slice. FP32 global
// is converted with f2bf RNE; BF16 global (the GDEC_QSA_KV_BF16 production
// layout) is a straight copy with no conversion work.
template <typename KVT>
__device__ __forceinline__ void stage16_bf16(uint16_t* dst, const KVT* src) {
  if constexpr (sizeof(KVT) == 4) {
    float4 a = *(const float4*)src, b = *(const float4*)(src + 4);
    float4 c = *(const float4*)(src + 8), d = *(const float4*)(src + 12);
    *(uint4*)dst = {pack_bf2(a.x, a.y), pack_bf2(a.z, a.w),
                    pack_bf2(b.x, b.y), pack_bf2(b.z, b.w)};
    *(uint4*)(dst + 8) = {pack_bf2(c.x, c.y), pack_bf2(c.z, c.w),
                          pack_bf2(d.x, d.y), pack_bf2(d.z, d.w)};
  } else {
    const uint4* s = (const uint4*)src;  // 16 bf16 = 32B
    *(uint4*)dst = s[0];
    *(uint4*)(dst + 8) = s[1];
  }
}

template <typename KVT = float>
__global__ void k_qsa_flash_bf16(const float* __restrict__ q,
                                 const KVT* __restrict__ kc,
                                 const KVT* __restrict__ vc,
                                 float* __restrict__ out, int P, int dh, int hkv,
                                 int hq, const int* __restrict__ selected,
                                 int first, int kbase = 0) {
  const int kvh = blockIdx.y;
  const int token = first + (int)blockIdx.x;
  const int qrow = token - kbase;  // batch-local q/out row; token stays absolute
  const int* blocks = selected + (size_t)blockIdx.x * 512;
  const int ntok = token < 2051 ? token + 1 : 2048 + (token + 1) % 4;
  constexpr int QR = 12;
  __shared__ __align__(16) uint16_t qs[QR][QSA_LPB];
  __shared__ __align__(16) uint16_t kt[QSA_KT][QSA_LPB];
  __shared__ __align__(16) uint16_t vt[QSA_KT][QSA_LPB];
  const int tid = threadIdx.x;
  // stage q tile (12 rows; fp32 global -> bf16 LDS)
  for (int i4 = tid; i4 < QR * (dh / 4); i4 += 128) {
    int r = i4 / (dh / 4), d4 = i4 % (dh / 4);
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < hq / hkv)
      v = *(const float4*)(q + ((size_t)qrow * hq + kvh * (hq / hkv) + r) * dh + d4 * 4);
    uint2 pk = {pack_bf2(v.x, v.y), pack_bf2(v.z, v.w)};
    *(uint2*)&qs[r][d4 * 4] = pk;
  }
  const int w = tid / warpSize, lane = tid % warpSize;
  const int rl = lane / QSA_KT;      // row within warp: rows 4w+rl
  const int pl = lane % QSA_KT;      // position within chunk (score phase)
  const int row = 4 * w + rl;
  float m_run = -1e30f, l_run = 0.f; // online softmax state
  float4 acc[8];                     // dims dg*4 + 32*f, dg = lane % 8
  for (int f = 0; f < 8; f++) acc[f] = make_float4(0.f, 0.f, 0.f, 0.f);
  const float scale = 1.f / sqrtf((float)dh);
  // K/V chunk staging slice: thread tid stages row tid/16, elems (tid%16)*16..
  const int spi = tid / 16, spf = (tid % 16) * 16;
  const KVT* kg = kc + (size_t)kvh * dh + spf;
  for (int p0 = 0; p0 < ntok; p0 += QSA_KT) {
    // stage K and V chunks into the bf16 LDS tiles
    int source = qsa_source_pos(min(p0 + spi, ntok - 1), token + 1, blocks);
    stage16_bf16(&kt[spi][spf], kg + (size_t)source * hkv * dh);
    stage16_bf16(&vt[spi][spf], vc + ((size_t)source * hkv + kvh) * dh + spf);
    __syncthreads();
    if (row < QR) {
      // scores: lane (rl, pl) owns (row 4w+rl, pos p0+pl); uint4 (8 bf16)
      // reads keep the fp32 float4 pattern's conflict-free bank layout
      const uint4* q8 = (const uint4*)&qs[row][0];
      const uint4* k8 = (const uint4*)&kt[pl][0];
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;  // 4 partials (ILP)
      for (int d8 = 0; d8 < dh / 8; d8++) {
        uint4 qv = q8[d8];
        uint4 kv = k8[d8];
        const __hip_bfloat162* qp = (const __hip_bfloat162*)&qv;
        const __hip_bfloat162* kp = (const __hip_bfloat162*)&kv;
        s0 = __builtin_amdgcn_fdot2_f32_bf16(qp[0], kp[0], s0, false);
        s1 = __builtin_amdgcn_fdot2_f32_bf16(qp[1], kp[1], s1, false);
        s2 = __builtin_amdgcn_fdot2_f32_bf16(qp[2], kp[2], s2, false);
        s3 = __builtin_amdgcn_fdot2_f32_bf16(qp[3], kp[3], s3, false);
      }
      float s = ((s0 + s1) + (s2 + s3)) * scale;
      bool masked = p0 + pl >= ntok;
      if (masked) s = -1e30f;
      // chunk max/sum over the 8 lanes of this row (width-8 shfl groups)
      float cm = s;
      for (int off = QSA_KT / 2; off > 0; off >>= 1)
        cm = fmaxf(cm, __shfl_down_sync(~0ull, cm, off, QSA_KT));
      cm = __shfl_sync(~0ull, cm, 0, QSA_KT);
      float mn = fmaxf(m_run, cm);
      float c2 = expf(m_run - mn);   // 1.0 when both are -1e30 (masked chunk)
      float e = expf(s - mn);
      if (masked) e = 0.f;
      float cl = e;
      for (int off = QSA_KT / 2; off > 0; off >>= 1)
        cl += __shfl_down_sync(~0ull, cl, off, QSA_KT);
      cl = __shfl_sync(~0ull, cl, 0, QSA_KT);
      l_run = l_run * c2 + cl;
      m_run = mn;
      // O accumulation: lane (rl, dg) owns dims dg*4 + 32*f, fp32 FMA on
      // bf16->fp32 expanded V (ev == 0 kills stale masked rows, as before)
      const int dg = lane % QSA_KT;
      for (int f = 0; f < 8; f++) {
        acc[f].x *= c2;
        acc[f].y *= c2;
        acc[f].z *= c2;
        acc[f].w *= c2;
      }
      auto accumulate_v = [&](int p) {
        float ev = __shfl_sync(~0ull, e, rl * QSA_KT + p);
        const uint16_t* vrow = &vt[p][dg * 4];
        for (int f = 0; f < 8; f++) {
          float4 vv = kv_ld4(vrow + f * 32);
          acc[f].x += ev * vv.x;
          acc[f].y += ev * vv.y;
          acc[f].z += ev * vv.z;
          acc[f].w += ev * vv.w;
        }
      };
#pragma unroll 1
      for (int p = 0; p < QSA_KT; p++) accumulate_v(p);
    }
    __syncthreads();
  }
  if (row < hq / hkv) {
    float inv = 1.f / l_run;
    float* o = out + ((size_t)qrow * hq + kvh * (hq / hkv) + row) * dh +
               (lane % QSA_KT) * 4;
    for (int f = 0; f < 8; f++) {
      float4 v = acc[f];
      v.x *= inv;
      v.y *= inv;
      v.z *= inv;
      v.w *= inv;
      *(float4*)(o + 32 * f) = v;
    }
  }
}

// --- k_qsa_wmma: WMMA bf16 sparse flash attention (256 thr = 8 warps) ---------
// Opt-in (GDEC_QSA_WMMA=1, BF16 KV mode only): replaces k_qsa_flash_bf16 for
// the sparse tail. One block per (token, kvh); keys stream in 128-key block
// iterations and warp w scores its own 16-key stripe (S = Q·Kᵀ WMMA, dual
// accumulators for ILP). Softmax uses a BLOCK-wide online maximum (two-stage
// LDS reduction), so every warp owns a fixed 32-dim slice of O and the
// accumulators never need a cross-warp merge; l is reduced once at the end.
// P is quantized bf16 (RNE) into a double-buffered LDS tile and read back as
// the PV A fragment — the WMMA C layout cannot feed back as A, and an LDS
// round-trip is the standard fix. V B fragments come from either the
// transposed cache (TransposedV, vct) or a row-major gather + in-register
// 16x16 transpose (tr16x16_bf16, no LDS).
// gfx1151 WMMA fragment layout (measured with local probe kernels on this
// GPU — hardware behavior, not implementation choice):
//   A(r,k): r<8 -> lane 2r elem k; r>=8 -> lane 17+2(r-8) elem k (rest ignored)
//   B(k,c): lane c elem k AND lane 16+c elem k (duplicate)
//   C(r,c): lane c + 16*(r>=8), elem r%8
// Numerics: bf16 P + WMMA accumulation (not bitwise fp32); measured maxabs
// 1.6e-4 vs the bf16-P fp32 reference path.
using qw_shortx16 = __attribute__((ext_vector_type(16))) short;
using qw_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ qw_shortx16 qw_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  qw_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
__device__ __forceinline__ qw_shortx16 qw_zero16() {
  short tmp[16] = {};
  qw_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
// In-register 16x16 bf16 transpose within each 16-lane half (gfx1151,
// wave32). Input: lane l (of the half) holds u32 a[8], a[i] =
// pack(M[l][2i], M[l][2i+1]) — one matrix row, 32B. Output: lane c holds
// u32 a[j] = pack(M[2j][c], M[2j+1][c]) — one matrix column, i.e. a WMMA
// B fragment tile. 20 shfls, no LDS.
// Derivation (CPU-sim verified in tools/qsa_wmma_rm.cu): lane l = 2q+g.
// Even lanes hold rows of the 8x8 u32 matrix E[s][i], odd lanes O[s][i].
// Stage A: 8x8 u32 transpose per sub-group (masks 2,4,8); butterfly swaps
// slot j on lane q with slot j^p on lane q^p, so the sent value is a[j^p]
// and each step snapshots into t[] (partner may overwrite a slot before
// our loop iteration reads it). Stage B (mask 1): adjacent lanes (2i,
// 2i+1) swap u32s and pick halves by g. Every lane executes every shfl
// (convergence), selecting afterwards.
__device__ __forceinline__ void tr16x16_bf16(unsigned a[8]) {
  const unsigned l = threadIdx.x & 15;
  const unsigned q = l >> 1;
#pragma unroll
  for (int m = 2; m <= 8; m <<= 1) {
    const unsigned p = m >> 1;
    unsigned t[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) t[j] = a[j];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned peer = __shfl_xor_sync(~0ull, t[j ^ p], m);
      if (((unsigned)j & p) != (q & p)) a[j] = peer;
    }
  }
  const unsigned g = l & 1;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const unsigned peer = __shfl_xor_sync(~0ull, a[j], 1);
    a[j] = g ? ((peer >> 16) | (a[j] & 0xFFFF0000u))
             : ((a[j] & 0xFFFFu) | (peer << 16));
  }
}
#define QW_DH 256
#define QW_QR 12
#define QW_KTILE 16
#define QW_KITER 128
#define QW_NITER 17   // ceil(2051 / 128)
#define QW_NKSTEP (QW_KITER / QW_KTILE)
// Paged (A1c): KV rows / vct blocks go through the page table ptab
// (kv_row / ik_blk). Compile-time switch so the unpaged production
// instantiations keep their exact codegen (this kernel is VGPR-sensitive).
template <bool TransposedV = false, bool Paged = false>
__global__ void __launch_bounds__(256)
    k_qsa_wmma(const float* __restrict__ q, const uint16_t* __restrict__ kc,
               const uint16_t* __restrict__ vc, const uint16_t* __restrict__ vct,
               float* __restrict__ out, int P,
               int dh, int hkv, int hq, const int* __restrict__ selected,
               int first, int kbase = 0, const int* __restrict__ ptab = nullptr) {
  (void)P;
  (void)ptab;
  const int token = first + (int)blockIdx.x, kvh = blockIdx.y;
  const int qrow = token - kbase;  // batch-local q/out row; token stays absolute
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __shared__ uint16_t Qs[16][QW_DH + 8];   // 8704B
  __shared__ uint16_t Pbuf[2][16][136];    // [phase][qrow][key], double-buffered
  __shared__ float Mx[8][16];
  __shared__ float Lf[8][16];
  // Neither V path uses an LDS tile: TransposedV gathers from the
  // transposed cache, the row-major path transposes in registers.
  // stage q tile (fp32 global -> bf16 RNE LDS; rows >= 12 stay zero)
  {
    const float* qb = q + ((size_t)qrow * hq + kvh * (hq / hkv)) * dh;
    for (int i = threadIdx.x; i < 16 * (QW_DH / 8); i += 256) {
      int r = i / (QW_DH / 8), c8 = i % (QW_DH / 8);
      uint4 v = {0, 0, 0, 0};
      if (r < hq / hkv) {
        const float4* q4 = (const float4*)(qb + r * dh + c8 * 8);
        float4 a = q4[0], b = q4[1];
        v.x = pack_bf2(a.x, a.y); v.y = pack_bf2(a.z, a.w);
        v.z = pack_bf2(b.x, b.y); v.w = pack_bf2(b.z, b.w);
      }
      *(uint4*)&Qs[r][c8 * 8] = v;
    }
  }
  __syncthreads();
  const int* blocks = selected + (size_t)blockIdx.x * 512;
  const int ntok = 2048 + (token + 1) % 4;
  const long tail0 = ((token + 1) / 4) * 4L;   // dense tail (visible = token+1)
  float m_run[8], l_run[8];   // per-elem qrow state (row = lane<16 ? e : 8+e)
#pragma unroll
  for (int e = 0; e < 8; ++e) { m_run[e] = -1e30f; l_run[e] = 0.f; }
  float acc[2][8];            // 2 d-tiles; elem e -> qrow (lane<16 ? e : 8+e),
                              // lane c16 -> dim w*32 + dt*16 + c16
#pragma unroll
  for (int dt = 0; dt < 2; ++dt)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[dt][e] = 0.f;
  const float scale = 1.f / sqrtf((float)dh);
  int qrowA = -1;             // A=Q role lane -> qrow
  if (lane < 16 && !(lane & 1)) qrowA = lane / 2;
  else if (lane >= 17 && (lane & 1)) qrowA = 8 + (lane - 17) / 2;
  const int keylane = lane & 15;
#pragma unroll 1
  for (int it = 0; it < QW_NITER; ++it) {
    const int kbase = it * QW_KITER + w * QW_KTILE;
    // ---------------- QK: S = Q·Kᵀ ----------------
    qw_floatx8 s0 = {0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0;
    int gkey = kbase + keylane;
    long src = gkey < ntok ? (gkey < 2048 ? 4L * blocks[gkey / 4] + gkey % 4
                                          : tail0 + gkey - 2048)
                           : 0;
    if constexpr (Paged) src = kv_row(ptab, (int)src);
    const uint16_t* kp = kc + ((size_t)src * hkv + kvh) * dh;
#pragma unroll
    for (int k4 = 0; k4 < 4; ++k4) {
      qw_shortx16 a[4], b[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        a[j] = qrowA >= 0 ? qw_ld16(&Qs[qrowA][(k4 * 4 + j) * 16]) : qw_zero16();
        b[j] = qw_ld16(kp + (k4 * 4 + j) * 16);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if ((k4 * 4 + j) & 1)
          s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[j], b[j], s1);
        else
          s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[j], b[j], s0);
      }
    }
    qw_floatx8 s;
#pragma unroll
    for (int e = 0; e < 8; ++e) s[e] = (s0[e] + s1[e]) * scale;
    if (gkey >= ntok)
#pragma unroll
      for (int e = 0; e < 8; ++e) s[e] = -1e30f;
    // warp-level row max via butterflies (halves stay separate: xor < 16)
    float pm[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float m = s[e];
      m = fmaxf(m, __shfl_xor_sync(~0ull, m, 1));
      m = fmaxf(m, __shfl_xor_sync(~0ull, m, 2));
      m = fmaxf(m, __shfl_xor_sync(~0ull, m, 4));
      m = fmaxf(m, __shfl_xor_sync(~0ull, m, 8));
      pm[e] = m;
    }
    // row j (j<8): lane j elem j; row 8+j: lane 16+j elem j
    if (lane < 8) Mx[w][lane] = pm[lane];
    else if (lane >= 16 && lane < 24) Mx[w][8 + lane - 16] = pm[lane - 16];
    __syncthreads();
    // single-barrier block max: lane l<16 reduces row l, broadcast via shfl
    float mrow = -1e30f;
    if (lane < 16) {
      float m = Mx[0][lane];
#pragma unroll
      for (int ww = 1; ww < 8; ++ww) m = fmaxf(m, Mx[ww][lane]);
      mrow = m;
    }
    // per-elem: p = exp(s - mn), l partial, cf rescale
    uint16_t pb[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      float mn = fmaxf(m_run[e], __shfl_sync(~0ull, mrow, r));
      float p = (s[e] <= -1e29f) ? 0.f : __expf(s[e] - mn);
      pb[e] = f2bf(p);
      float cf = __expf(m_run[e] - mn);
      float ls = p;
      ls += __shfl_xor_sync(~0ull, ls, 1);
      ls += __shfl_xor_sync(~0ull, ls, 2);
      ls += __shfl_xor_sync(~0ull, ls, 4);
      ls += __shfl_xor_sync(~0ull, ls, 8);
      l_run[e] = l_run[e] * cf + ls;
      m_run[e] = mn;
      acc[0][e] *= cf;
      acc[1][e] *= cf;
    }
    // ---------------- P to LDS transposed ----------------
    {
      int wkey = w * QW_KTILE + keylane;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int r = (lane < 16) ? e : 8 + e;
        Pbuf[it & 1][r][wkey] = pb[e];
      }
    }
    __syncthreads();
    // ---------------- PV: warp w owns dims [w*32, +32) ----------------
    int pr = (lane < 16) ? lane / 2 : 8 + (lane - 17) / 2;
    bool pact = (lane < 16 && !(lane & 1)) || (lane >= 17 && (lane & 1));
#pragma unroll 1
    for (int ks = 0; ks < QW_NKSTEP; ++ks) {
      qw_shortx16 ap = pact ? qw_ld16(&Pbuf[it & 1][pr][ks * 16]) : qw_zero16();
      // B = V frag: lane c elem k = V[key k][dim dt*16 + c16]. The
      // transposed cache path loads four slots per block in one 64-bit read
      // (base pointers per 4-key block, all 8 loads issued before the 2
      // wmmas to hide gather latency); the row-major path gathers one
      // 16-key x 16-dim tile per half-warp (same wide global pattern as
      // the old LDS path) and transposes it in registers (tr16x16_bf16),
      // then swaps tiles between halves — no LDS, no vct.
      if constexpr (TransposedV) {
        const uint16_t* vb[4];
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
          int vk = it * QW_KITER + ks * QW_KTILE + blk * 4;
          long vs = vk < ntok ? (vk < 2048 ? 4L * blocks[vk / 4] + vk % 4
                                           : tail0 + vk - 2048)
                              : 0;
          // vs is 4-aligned here (blk*4 steps, 4-aligned tail0)
          if constexpr (Paged) vs = (long)ik_blk(ptab, (int)(vs / 4)) * 4;
          vb[blk] = vct + ((((size_t)vs / 4) * hkv + kvh) * dh + w * 32) * 4;
        }
        unsigned long long pk[2][4];
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          int dim = dt * 16 + (lane & 15);
#pragma unroll
          for (int blk = 0; blk < 4; ++blk)
            pk[dt][blk] = *(const unsigned long long*)(vb[blk] + dim * 4);
        }
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          short btmp[16];
#pragma unroll
          for (int blk = 0; blk < 4; ++blk)
#pragma unroll
            for (int slot = 0; slot < 4; ++slot)
              btmp[blk * 4 + slot] = (short)(pk[dt][blk] >> (slot * 16));
          qw_shortx16 bv;
          memcpy(&bv, btmp, 32);
          qw_floatx8 o;
#pragma unroll
          for (int e = 0; e < 8; ++e) o[e] = acc[dt][e];
          o = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ap, bv, o);
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[dt][e] = o[e];
        }
      } else {
        // lane&15 = key, lane>>4 = d-tile: 2 x uint4 = 16 dims per lane.
        int vk = it * QW_KITER + ks * QW_KTILE + (lane & 15);
        long vs = vk < ntok ? (vk < 2048 ? 4L * blocks[vk / 4] + vk % 4
                                         : tail0 + vk - 2048)
                            : 0;
        if constexpr (Paged) vs = kv_row(ptab, (int)vs);
        const uint16_t* vp =
            vc + ((size_t)vs * hkv + kvh) * dh + w * 32 + (lane >> 4) * 16;
        unsigned va[8];
        *(uint4*)&va[0] = *(const uint4*)vp;
        *(uint4*)&va[4] = *(const uint4*)(vp + 8);
        tr16x16_bf16(va);  // half (lane>>4) now holds B cols of d-tile (lane>>4)
        unsigned vo[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) vo[j] = __shfl_xor_sync(~0ull, va[j], 16);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          unsigned bs[8];
#pragma unroll
          for (int j = 0; j < 8; ++j)
            bs[j] = ((unsigned)(lane >> 4) == (unsigned)dt) ? va[j] : vo[j];
          qw_shortx16 bv;
          memcpy(&bv, bs, 32);
          qw_floatx8 o;
#pragma unroll
          for (int e = 0; e < 8; ++e) o[e] = acc[dt][e];
          o = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ap, bv, o);
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[dt][e] = o[e];
        }
      }
    }
    // no reuse barrier: double-buffered Pbuf + next iter's max sync is enough
  }
  // ---------------- final: block-wide l sum + write ----------------
  if (lane < 8) Lf[w][lane] = l_run[lane];
  else if (lane >= 16 && lane < 24) Lf[w][8 + lane - 16] = l_run[lane - 16];
  __syncthreads();
  float lfin[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int r = (lane < 16) ? e : 8 + e;
    float t = 0;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) t += Lf[ww][r];
    lfin[e] = 1.f / t;
  }
  int c16 = lane & 15;
  float* ob = out + ((size_t)qrow * hq + kvh * (hq / hkv)) * dh;
#pragma unroll
  for (int dt = 0; dt < 2; ++dt)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      if (r < hq / hkv) ob[(size_t)r * dh + w * 32 + dt * 16 + c16] = acc[dt][e] * lfin[e];
    }
}

// --- k_qsa_wmma6: dim-split WMMA sparse flash attention (128 thr = 4 warps) --
// Opt-in (GDEC_QSA_WMMA6=1, on top of the transposed-V BF16 path). Same data
// layout and masking as k_qsa_wmma, but warp w owns dims [w*64, +64) for BOTH
// QK and PV (no key/dim role switch): Q lives in registers as four resident
// bf16 A fragments (fp32 global -> RNE on load, rows >= 12 zero). Per 16-key
// group the four warps' QK partial S fragments are summed through a
// double-buffered LDS area in fixed w0->w3 order; every warp then runs the
// full online softmax redundantly (warp-local butterflies only — the warp
// already sees all 16 keys — so no block-wide max/l reductions). P is
// quantized bf16 RNE into a warp-private LDS tile and read back as the PV A
// fragment; PV B fragments come straight from the transposed V cache (four
// 16-dim C tiles per warp, four 4-key slots per 64-bit load). Fragment
// layouts as documented on k_qsa_wmma.
__global__ void __launch_bounds__(128)
    k_qsa_wmma6(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                const uint16_t* __restrict__ vc, const uint16_t* __restrict__ vct,
                float* __restrict__ out, int P,
                int dh, int hkv, int hq, const int* __restrict__ selected,
                int first, int kbase = 0) {
  (void)P;
  (void)vc;
  const int token = first + (int)blockIdx.x, kvh = blockIdx.y;
  const int qrow = token - kbase;  // batch-local q/out row; token stays absolute
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __shared__ int Sel[512];              // selected block table, 2KB
  __shared__ float Sbuf[2][4][16][17];  // [phase][warp][qrow][key], 8.7KB
  __shared__ uint16_t Pbuf[4][16][24];  // [warp][qrow][key], 3KB
  const int* blocks = selected + (size_t)blockIdx.x * 512;
  ((int4*)Sel)[threadIdx.x] = ((const int4*)blocks)[threadIdx.x];
  // Resident Q: A-fragment lanes (even 0-14 / odd 17-31) hold q row
  // lane/2 / 8+(lane-17)/2, dims [w*64, +64) as four 16-dim fragments.
  int qrowA = -1;
  if (lane < 16 && !(lane & 1)) qrowA = lane / 2;
  else if (lane >= 17 && (lane & 1)) qrowA = 8 + (lane - 17) / 2;
  qw_shortx16 qa[4];
  {
    const float* qb = q + ((size_t)qrow * hq + kvh * (hq / hkv)) * dh;
#pragma unroll
    for (int ss = 0; ss < 4; ++ss) {
      if (qrowA >= 0 && qrowA < hq / hkv) {
        const float4* q4 =
            (const float4*)(qb + (size_t)qrowA * dh + w * 64 + ss * 16);
        float4 a = q4[0], b = q4[1], c = q4[2], d = q4[3];
        uint32_t pk[8] = {pack_bf2(a.x, a.y), pack_bf2(a.z, a.w),
                          pack_bf2(b.x, b.y), pack_bf2(b.z, b.w),
                          pack_bf2(c.x, c.y), pack_bf2(c.z, c.w),
                          pack_bf2(d.x, d.y), pack_bf2(d.z, d.w)};
        memcpy(&qa[ss], pk, 32);
      } else {
        qa[ss] = qw_zero16();
      }
    }
  }
  __syncthreads();  // Sel visible block-wide
  const int ntok = 2048 + (token + 1) % 4;
  const long tail0 = ((token + 1) / 4) * 4L;  // dense tail (visible = token+1)
  const float scale = 1.f / sqrtf((float)dh);
  const int keylane = lane & 15;
  float m_run[8], l_run[8];  // per-elem qrow state (row = lane<16 ? e : 8+e)
#pragma unroll
  for (int e = 0; e < 8; ++e) { m_run[e] = -1e30f; l_run[e] = 0.f; }
  float acc[4][8];           // 4 d-tiles; lane c16 -> dim w*64 + dt*16 + c16
#pragma unroll
  for (int dt = 0; dt < 4; ++dt)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[dt][e] = 0.f;
  const int ngroups = (ntok + 15) >> 4;
#pragma unroll 1
  for (int g = 0; g < ngroups; ++g) {
    // ---------------- QK: partial S over this warp's 64 dims ----------------
    const int gkey = g * 16 + keylane;
    long src = gkey < ntok ? (gkey < 2048 ? 4L * Sel[gkey >> 2] + (gkey & 3)
                                          : tail0 + gkey - 2048)
                           : 0;
    const uint16_t* kp = kc + ((size_t)src * hkv + kvh) * dh + w * 64;
    qw_floatx8 s = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int ss = 0; ss < 4; ++ss) {
      qw_shortx16 b = qw_ld16(kp + ss * 16);
      s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qa[ss], b, s);
    }
    // ---------------- cross-warp S reduction, fixed w0 -> w3 ----------------
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      Sbuf[g & 1][w][r][keylane] = s[e];
    }
    // Double-buffered: the single barrier per group also orders next group's
    // writes against this group's reads (they land in the other phase).
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      float t = Sbuf[g & 1][0][r][keylane] + Sbuf[g & 1][1][r][keylane];
      t += Sbuf[g & 1][2][r][keylane];
      t += Sbuf[g & 1][3][r][keylane];
      s[e] = t * scale;
    }
    if (gkey >= ntok)
#pragma unroll
      for (int e = 0; e < 8; ++e) s[e] = -1e30f;
    // ---------------- online softmax (warp-redundant, identical) -----------
    uint16_t pb[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float pm = s[e];
      pm = fmaxf(pm, __shfl_xor_sync(~0ull, pm, 1));
      pm = fmaxf(pm, __shfl_xor_sync(~0ull, pm, 2));
      pm = fmaxf(pm, __shfl_xor_sync(~0ull, pm, 4));
      pm = fmaxf(pm, __shfl_xor_sync(~0ull, pm, 8));
      float mn = fmaxf(m_run[e], pm);
      float p = (s[e] <= -1e29f) ? 0.f : __expf(s[e] - mn);
      pb[e] = f2bf(p);
      float cf = __expf(m_run[e] - mn);
      float ls = p;
      ls += __shfl_xor_sync(~0ull, ls, 1);
      ls += __shfl_xor_sync(~0ull, ls, 2);
      ls += __shfl_xor_sync(~0ull, ls, 4);
      ls += __shfl_xor_sync(~0ull, ls, 8);
      float lc = l_run[e] * cf;
      asm volatile("" : "+v"(lc));  // keep the rescale out of fma contraction
      l_run[e] = lc + ls;
      m_run[e] = mn;
#pragma unroll
      for (int dt = 0; dt < 4; ++dt) acc[dt][e] *= cf;
    }
    // ---------------- P to warp-private LDS, back as PV A fragment ---------
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      Pbuf[w][r][keylane] = pb[e];
    }
    __syncwarp();
    int pr = (lane < 16) ? lane / 2 : 8 + (lane - 17) / 2;
    bool pact = (lane < 16 && !(lane & 1)) || (lane >= 17 && (lane & 1));
    qw_shortx16 ap = pact ? qw_ld16(&Pbuf[w][pr][0]) : qw_zero16();
    // ---------------- PV: B from the transposed V cache, 4 C tiles ---------
    const uint16_t* vb[4];
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
      int vk = g * 16 + blk * 4;  // 4-aligned: slots 0-3 of one vct block
      long vs = vk < ntok ? (vk < 2048 ? 4L * Sel[vk >> 2] + (vk & 3)
                                       : tail0 + vk - 2048)
                          : 0;
      vb[blk] = vct + ((((size_t)vs >> 2) * hkv + kvh) * dh + w * 64) * 4;
    }
    unsigned long long pk[4][4];
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      int dim = dt * 16 + keylane;
#pragma unroll
      for (int blk = 0; blk < 4; ++blk)
        pk[dt][blk] = *(const unsigned long long*)(vb[blk] + dim * 4);
    }
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      short btmp[16];
#pragma unroll
      for (int blk = 0; blk < 4; ++blk)
#pragma unroll
        for (int slot = 0; slot < 4; ++slot)
          btmp[blk * 4 + slot] = (short)(pk[dt][blk] >> (slot * 16));
      qw_shortx16 bv;
      memcpy(&bv, btmp, 32);
      qw_floatx8 o;
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = acc[dt][e];
      o = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ap, bv, o);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[dt][e] = o[e];
    }
  }
  // ---------------- final: warp-local l, each warp writes its 64 dims ------
  float lfin[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) lfin[e] = 1.f / l_run[e];
  float* ob = out + ((size_t)qrow * hq + kvh * (hq / hkv)) * dh;
#pragma unroll
  for (int dt = 0; dt < 4; ++dt)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      if (r < hq / hkv) ob[(size_t)r * dh + w * 64 + dt * 16 + keylane] = acc[dt][e] * lfin[e];
    }
}

// --- GDEC_QSA_UNION: 4-query 块并集 WMMA 稀疏 prefill attention ---------------
// 移植自 llama.cpp-strix-halo qsa.cu 的 qsa3 rows/merge/attn（同模型、同 GPU
// 验证过），适配引擎数据布局。4 个连续（4 对齐）稀疏 token 的 top-512 块列表
// 由 k_qsa_union_merge 合成并集（实测重叠 ~2/3，K/V gather 量约 ÷3），
// k_qsa_wmma_u 一个 block 处理 1 组 × 1 个 KV 头：48 个 Q 行（4 token x 12 头
// = 3 个 16 行 WMMA tile）共享同一条 K/V 流。
// 与参考的差异：
//  - bf16（参考是 f16），fragment 布局以 k_qsa_wmma 注释的本机实测为准
//    （参考的 key 交错/掩码映射与该布局自洽，照搬）；
//  - K 走行主序 bf16 cache 按键 gather（同 k_qsa_wmma），V 走 vct 转置
//    cache（[block][kvh][dim][slot]，与参考 packed pv 同构）；不上 packed K；
//  - 引擎选择列表是块粒度且升序、无无效项（稀疏 token n>512），merge 直接
//    4 路归并块号，整块快路径退化为常数；省掉参考的 rows 排序 kernel；
//  - 参考的尾部 key 由 mask 张量表达；引擎的 dense 尾部 = 组内自身块的
//    slot 0..qi（qi<3），attn 内对自身块做因果覆盖（slot cc 对 query qi<3
//    可见 iff cc <= qi；qi=3 无尾部，仅靠 mask），零额外 K/V 流量；自身块
//    若未被 token g+3 选中则由 merge 补登（mask=0，只剩因果覆盖生效，
//    语义与逐 token 路径一致）。
//  - scale 在跨 warp 归约后乘（1/16，与 k_qsa_wmma 一致），softmax 用 __expf。
#define QSA_UCAP 2052  // 每组并集容量（块）：4x512 + 自身块补登，pad 到 4 倍数

// 4 条升序 top-512 块号列表 -> 并集 + 每 query 有效性掩码（整块 0xF）。
// 128 lanes：lane 负责块号子区间（row0 分位数划分，真实负载均衡）；两遍归并
// （pass1 计数 + Hillis-Steele 前缀和，pass2 写出），pad 0xFFFF。
__global__ void __launch_bounds__(128)
    k_qsa_union_merge(const int* __restrict__ sel, int g0,
                      uint16_t* __restrict__ ublk, uint16_t* __restrict__ umask,
                      int* __restrict__ ucount, int cap, int sbase = 0) {
  __shared__ int rows_s[4][512];
  __shared__ int scan_s[128];
  const int lane = threadIdx.x, g = blockIdx.x;
  const int t0 = g0 + 4 * g;  // 组首 token（绝对，4 对齐）
#pragma unroll
  for (int qi = 0; qi < 4; ++qi) {
    // sel 行号 = 绝对 token - sbase（A1b：d_iselected 按 batch 局部行存放）
    const int* src = sel + (size_t)(t0 + qi - sbase) * 512;
    for (int j = lane; j < 512; j += 128) rows_s[qi][j] = src[j];
  }
  __syncthreads();
  const int own = t0 >> 2;
  int kmax = rows_s[0][511];
#pragma unroll
  for (int qi = 1; qi < 4; ++qi) kmax = max(kmax, rows_s[qi][511]);
  // lane 的块号子区间 [lo, hi)：row0 的分位数
  const int lo = lane == 0 ? 0 : rows_s[0][4 * lane];
  const int hi = lane == 127 ? kmax + 1 : rows_s[0][4 * (lane + 1)];
  int p[4], pe[4];
#pragma unroll
  for (int qi = 0; qi < 4; ++qi) {
    int a = 0, b = 512;
    while (a < b) {
      const int mid = (a + b) >> 1;
      if (rows_s[qi][mid] < lo) a = mid + 1; else b = mid;
    }
    p[qi] = a;
    b = 512;
    while (a < b) {
      const int mid = (a + b) >> 1;
      if (rows_s[qi][mid] < hi) a = mid + 1; else b = mid;
    }
    pe[qi] = a;
  }
  // pass 1：计数（p[] 保留给 pass 2）
  int n = 0;
  {
    int pp[4], head[4];
#pragma unroll
    for (int qi = 0; qi < 4; ++qi) {
      pp[qi] = p[qi];
      head[qi] = pp[qi] < pe[qi] ? rows_s[qi][pp[qi]] : 0x7FFFFFFF;
    }
    while (true) {
      int bmin = 0x7FFFFFFF;
#pragma unroll
      for (int qi = 0; qi < 4; ++qi) bmin = min(bmin, head[qi]);
      if (bmin == 0x7FFFFFFF) break;
#pragma unroll
      for (int qi = 0; qi < 4; ++qi)
        if (head[qi] == bmin) {
          ++pp[qi];
          head[qi] = pp[qi] < pe[qi] ? rows_s[qi][pp[qi]] : 0x7FFFFFFF;
        }
      ++n;
    }
  }
  scan_s[lane] = n;
  __syncthreads();
#pragma unroll
  for (int off = 1; off < 128; off <<= 1) {
    const int v = lane >= off ? scan_s[lane - off] : 0;
    __syncthreads();
    scan_s[lane] += v;
    __syncthreads();
  }
  const int total = scan_s[127];
  int o = scan_s[lane] - n;
  uint16_t* ob = ublk + (size_t)g * cap;
  uint16_t* om = umask + (size_t)g * cap;
  // pass 2：写出（整块 -> 该 query 4 slot 全置）
  {
    int head[4];
#pragma unroll
    for (int qi = 0; qi < 4; ++qi)
      head[qi] = p[qi] < pe[qi] ? rows_s[qi][p[qi]] : 0x7FFFFFFF;
    while (true) {
      int bmin = 0x7FFFFFFF;
#pragma unroll
      for (int qi = 0; qi < 4; ++qi) bmin = min(bmin, head[qi]);
      if (bmin == 0x7FFFFFFF) break;
      uint32_t mk = 0;
#pragma unroll
      for (int qi = 0; qi < 4; ++qi)
        if (head[qi] == bmin) {
          mk |= 0xFu << (4 * qi);
          ++p[qi];
          head[qi] = p[qi] < pe[qi] ? rows_s[qi][p[qi]] : 0x7FFFFFFF;
        }
      ob[o] = (uint16_t)bmin;
      om[o] = (uint16_t)mk;
      ++o;
    }
  }
  if (lane == 127) {
    int t = total;
    // 防御：自身块必须落在并集里（qi<3 的 dense 尾部挂在它的 slot 上）。
    // 正常总被 token g+3 选中（row3 末位即 own）；万一缺失则补登，
    // mask=0：qi=3 不可见它（与逐 token 路径一致），qi<3 靠因果覆盖。
    if (rows_s[3][511] != own) {
      ob[t] = (uint16_t)own;
      om[t] = 0;
      ++t;
    }
    while (t & 3) {
      ob[t] = 0xFFFFu;
      om[t] = 0;
      ++t;
    }
    ucount[g] = t;
  }
}

// 并集 attention：grid (ngroups, 2)，256 线程 = 8 warp。结构同参考
// qsa3_attn_kernel：warp w 持有全部 48 行的 [32w,+32) 维 Q 切片（寄存器常驻
// bf16 A... 实为 B 操作数：wmma(K, Q) 得 Sᵀ），QK 跨 warp 经 part[] LDS 归约；
// warp 0..2 是 owner（行 16w+r），负责 online softmax 并把 P 量化 bf16 经
// ptile LDS 回转成 PV 的 B 操作数；PV 的 A = V（vct 转置 cache，4 slot 一次
// 64-bit 读），C 的行是 perm 后的维度、列是 Q 行。每 chunk 恰 2 次
// __syncthreads()，K 描述符与 fragment 软件流水预取。
__global__ void __launch_bounds__(256)
    k_qsa_wmma_u(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                 const uint16_t* __restrict__ vct, float* __restrict__ out,
                 int dh, int hkv, int hq,
                 const uint16_t* __restrict__ ublk,
                 const uint16_t* __restrict__ umask,
                 const int* __restrict__ ucount, int cap, int g0, int kbase) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane & 15,
            hi = lane >> 4;
  const int g = blockIdx.x, kvh = blockIdx.y;
  __shared__ float4 part[3][8][64];
  __shared__ uint4 ptile[3][32];
  __shared__ float alpha_s[48];
  __shared__ float l_s[48];
  const uint16_t* gblk = ublk + (size_t)g * cap;
  const uint16_t* gmsk = umask + (size_t)g * cap;
  const int nchunks = ucount[g] >> 2;
  const int own_blk = (g0 >> 2) + g;
  const int tok0 = g0 + 4 * g - kbase;  // 组首 token 的 batch 行
  const int hqk = hq / hkv;             // 12

  // Q stage：warp w 负责全部 48 行的 [32w,+32) 维切片（fp32 -> bf16 RNE）
  uint16_t* stage = reinterpret_cast<uint16_t*>(&part[0][0][0]) + w * (48 * 32);
#pragma unroll
  for (int c = 0; c < 12; ++c) {
    const int row = 4 * c + (lane >> 3), dq = 4 * (lane & 7);
    const int qi = row / 12, h = row - 12 * qi;
    const float4 v = *(const float4*)(q + ((size_t)(tok0 + qi) * hq +
                                           kvh * hqk + h) * dh + 32 * w + dq);
    uint2 pk;
    pk.x = pack_bf2(v.x, v.y);
    pk.y = pack_bf2(v.z, v.w);
    *(uint2*)(stage + row * 32 + dq) = pk;
  }
  __syncthreads();
  qw_shortx16 qf[3][2];
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int t = 0; t < 2; ++t)
      qf[i][t] = qw_ld16(stage + (16 * i + r) * 32 + 16 * t);
  __syncthreads();

  qw_floatx8 O[3][2];
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) O[i][t][e] = 0.f;
  float m = -1e30f, l = 0.f;  // owner 状态（w<3）：行 16w+r
  const int own_qi = (16 * w + r) / 12;

  auto load_desc = [&](const int c, uint32_t& b01, uint32_t& b23, uint32_t& m01,
                       uint32_t& m23) {
    const uint2 bb = *(const uint2*)(gblk + 4 * c);
    const uint2 mm = *(const uint2*)(gmsk + 4 * c);
    b01 = bb.x; b23 = bb.y; m01 = mm.x; m23 = mm.y;
  };
  auto blk_of = [&](const uint32_t b01, const uint32_t b23, const int j) -> int {
    const uint32_t v = (j < 2 ? b01 : b23) >> (16 * (j & 1)) & 0xFFFFu;
    return v == 0xFFFFu ? 0 : (int)v;
  };
  // K：行主序 bf16 cache 按键 gather；lane r 持有 chunk 内第 r 个 key 的
  // [32w,+32) 维（硬件 A 布局实际取 lane 2r/17+2(r-8) -> key 交错
  // 0,2,4,...,14,1,3,...,15，掩码侧按同一 perm 映射）。
  auto load_k = [&](const uint32_t b01, const uint32_t b23, qw_shortx16* kf) {
    const int src = 4 * blk_of(b01, b23, r >> 2) + (r & 3);
    const uint16_t* krow = kc + ((size_t)src * hkv + kvh) * dh + w * 32;
#pragma unroll
    for (int t = 0; t < 2; ++t) kf[t] = qw_ld16(krow + 16 * t);
  };

  uint32_t b01 = 0, b23 = 0, m01 = 0, m23 = 0;
  qw_shortx16 kf[2];
  if (nchunks > 0) {
    load_desc(0, b01, b23, m01, m23);
    load_k(b01, b23, kf);
  }

#pragma unroll 1
  for (int c = 0; c < nchunks; ++c) {
    qw_floatx8 sc[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
      for (int e = 0; e < 8; ++e) sc[i][e] = 0.f;
#pragma unroll
      for (int t = 0; t < 2; ++t)
        sc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(kf[t], qf[i][t], sc[i]);
    }
    // V：vct 转置 cache，lane r 持有维 32w+16t+r 的 4 块 x 4 slot（16 key）
    qw_shortx16 vf[2];
    {
      const uint16_t* vb[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        vb[j] = vct + (((size_t)blk_of(b01, b23, j) * hkv + kvh) * dh) * 4;
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const int d = 32 * w + 16 * t + r;
        uint2 pk[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) pk[j] = *(const uint2*)(vb[j] + d * 4);
        short tmp[16];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          tmp[4 * j + 0] = (short)(pk[j].x & 0xFFFFu);
          tmp[4 * j + 1] = (short)(pk[j].x >> 16);
          tmp[4 * j + 2] = (short)(pk[j].y & 0xFFFFu);
          tmp[4 * j + 3] = (short)(pk[j].y >> 16);
        }
        memcpy(&vf[t], tmp, 32);
      }
    }
    // 预取下一 chunk 的描述符与 K fragment
    uint32_t nb01 = 0, nb23 = 0, nm01 = 0, nm23 = 0;
    qw_shortx16 kfn[2];
    if (c + 1 < nchunks) {
      load_desc(c + 1, nb01, nb23, nm01, nm23);
      load_k(nb01, nb23, kfn);
    } else {
      kfn[0] = kf[0];
      kfn[1] = kf[1];
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      if (i != w) {
        part[i][w][lane * 2 + 0] = make_float4(sc[i][0], sc[i][1], sc[i][2], sc[i][3]);
        part[i][w][lane * 2 + 1] = make_float4(sc[i][4], sc[i][5], sc[i][6], sc[i][7]);
      }
    }
    __syncthreads();
    if (w < 3) {
      // owner：归约 8 个 warp 的 32 维 partial -> 全 256 维 Sᵀ，再 softmax
      float scf[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) scf[e] = 0.f;
#pragma unroll
      for (int w2 = 0; w2 < 8; ++w2) {
        if (w2 == w) {
#pragma unroll
          for (int e = 0; e < 8; ++e) scf[e] += sc[w][e];
        } else {
          const float4 a = part[w][w2][lane * 2 + 0], b = part[w][w2][lane * 2 + 1];
          scf[0] += a.x; scf[1] += a.y; scf[2] += a.z; scf[3] += a.w;
          scf[4] += b.x; scf[5] += b.y; scf[6] += b.z; scf[7] += b.w;
        }
      }
      // 有效性 + 自身块因果覆盖（elem e -> 块 j=e>>1、slot cc=2*(e&1)+hi，
      // 与硬件 A 布局的 key 交错一致）
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        scf[e] *= 0.0625f;  // 1/sqrt(256)
        const int j = e >> 1, cc = 2 * (e & 1) + hi;
        const uint32_t mk = ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xFFFFu;
        bool kv = (mk >> (own_qi * 4 + cc)) & 1u;
        // 自身块因果覆盖：qi<3 的 dense 尾部 = 自身块 slot 0..qi（恒可见）；
        // qi=3 无尾部，自身块仅在真被选中（mask 0xF000）时可见
        if (own_qi < 3 && blk_of(b01, b23, j) == own_blk && cc <= own_qi)
          kv = true;
        if (!kv) scf[e] = -1e30f;
      }
      float mloc = scf[0];
#pragma unroll
      for (int e = 1; e < 8; ++e) mloc = fmaxf(mloc, scf[e]);
      mloc = fmaxf(mloc, __shfl_xor_sync(~0ull, mloc, 16));
      const float mnew = fmaxf(m, mloc);
      const float alpha = __expf(m - mnew);  // 全掩码 chunk：m==mnew -> 1
      float p[8], ls = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        p[e] = scf[e] <= -1e29f ? 0.f : __expf(scf[e] - mnew);
        ls += p[e];
      }
      ls += __shfl_xor_sync(~0ull, ls, 16);
      l = l * alpha + ls;
      m = mnew;
      // P 量化 bf16 并按自然 key 序摆进 ptile（WMMA C 布局回不去 A，LDS 回转）
      uint32_t pp[4], po[4];
#pragma unroll
      for (int e2 = 0; e2 < 4; ++e2) {
        pp[e2] = (uint32_t)f2bf(p[2 * e2]) | ((uint32_t)f2bf(p[2 * e2 + 1]) << 16);
        po[e2] = (uint32_t)__shfl_xor_sync(~0ull, (int)pp[e2], 16);
      }
      uint16_t ph[16];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const uint16_t mine = (e & 1) ? (uint16_t)(pp[e >> 1] >> 16)
                                      : (uint16_t)(pp[e >> 1] & 0xffffu);
        const uint16_t other = (e & 1) ? (uint16_t)(po[e >> 1] >> 16)
                                       : (uint16_t)(po[e >> 1] & 0xffffu);
        ph[2 * e] = hi ? other : mine;
        ph[2 * e + 1] = hi ? mine : other;
      }
      if (hi == 0) {
        ptile[w][r * 2 + 0] = make_uint4(
            (uint32_t)ph[0] | ((uint32_t)ph[1] << 16),
            (uint32_t)ph[2] | ((uint32_t)ph[3] << 16),
            (uint32_t)ph[4] | ((uint32_t)ph[5] << 16),
            (uint32_t)ph[6] | ((uint32_t)ph[7] << 16));
        ptile[w][r * 2 + 1] = make_uint4(
            (uint32_t)ph[8] | ((uint32_t)ph[9] << 16),
            (uint32_t)ph[10] | ((uint32_t)ph[11] << 16),
            (uint32_t)ph[12] | ((uint32_t)ph[13] << 16),
            (uint32_t)ph[14] | ((uint32_t)ph[15] << 16));
        alpha_s[16 * w + r] = alpha;
      }
    }
    __syncthreads();
    float al[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) al[i] = alpha_s[16 * i + r];
    if (__ballot_sync(~0ull, al[0] != 1.f || al[1] != 1.f || al[2] != 1.f)) {
#pragma unroll
      for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
          for (int e = 0; e < 8; ++e) O[i][t][e] *= al[i];
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const qw_shortx16 pf = qw_ld16((const uint16_t*)&ptile[i][0] + r * 16);
#pragma unroll
      for (int t = 0; t < 2; ++t)
        O[i][t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(vf[t], pf, O[i][t]);
    }
    kf[0] = kfn[0];
    kf[1] = kfn[1];
    b01 = nb01; b23 = nb23; m01 = nm01; m23 = nm23;
  }

  if (w < 3 && hi == 0) l_s[16 * w + r] = l;
  __syncthreads();
  // 输出：每 tile 先经 LDS 摆成 [16 行 x 32 维] fp32，再 float4 合并写全局
  float* ostage = reinterpret_cast<float*>(&part[0][0][0]) + w * (16 * 32);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const float li = l_s[16 * i + r];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        ostage[r * 32 + 16 * t + 2 * e + hi] = li > 0.f ? O[i][t][e] / li : 0.f;
    __syncthreads();
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int rr = 4 * c + (lane >> 3), dq = 4 * (lane & 7);
      const int row = 16 * i + rr, qi = row / 12, h = row - 12 * qi;
      float* dst = out + ((size_t)(tok0 + qi) * hq + kvh * hqk + h) * dh +
                   32 * w + dq;
      *(float4*)dst = *(const float4*)(ostage + rr * 32 + dq);
    }
    __syncthreads();
  }
}

// forward decls (defined below main model code)
template <bool Rope = false>
__global__ void k_qsa_qsplit(const float* __restrict__ qg, const float* __restrict__ qnw,
                             float* __restrict__ qs, float* __restrict__ gs, int dh,
                             float eps, int rope_base = 0,
                             const float2* __restrict__ ropecs = nullptr);
template <bool Bf16 = false>
__global__ void k_qsa_kprep(const float* __restrict__ kb, const float* __restrict__ knw,
                            float* __restrict__ kout, uint16_t* __restrict__ kc,
                            float eps, int rope_base,
                            const float2* __restrict__ ropecs);
__global__ void k_axpy_sg(float* __restrict__ acc, const float* __restrict__ y,
                          const float* __restrict__ sg, int n, int total);

// [gemm-wmma-begin]
// --- k_gemm_wmma: self-written bf16 WMMA dense GEMM (Phase 3g) --------------
// Y[P,N] f32 = X[P,K]bf16 * W[N,K]bf16^T, all row-major (ld K, K, N), fp32
// accum. WMMA "A" operand = X rows (K-contig), "B" operand = W rows (K-contig)
// — a B fragment for output column n is exactly W[n][k0..k0+15], so no
// transpose exists anywhere; the epilogue walks lane-contiguous n (coalesced).
// Fragment layouts are the hardware ones documented above k_qsa_wmma.
// Block tile BM(tokens) x BP(weights), NT threads = (NT/32) warps as MW x PW,
// warp owns WM x WP 16x16 frags, K stepped by KST, LDS double buffer:
//   - X tile row-major, row stride KST+8; A-fragment lanes use a dummy-row
//     assignment so every 8-lane LDS phase touches 8 distinct banks
//     (probe-calibrated conflict-free; naive clamp-to-row-0 costs 1 extra
//     phase per ds_load_b128).
//   - W tile grouped in 8-row regions, region stride 8*(KST+8)+8: B-fragment
//     loads are one ds_load_b128 each, broadcast-friendly, conflict-free.
//   - Unconditional global->reg->LDS staging (tail K-step index clamped to
//     ksteps-1, recomputed into a dead buffer): conditional staging makes the
//     allocator spill the staging registers to scratch, exposing full DRAM
//     latency in the main loop (the single biggest perf bug found in the
//     prototype, tools/wmma_gemm_proto.cu).
//   - gwmma_sync(): __syncthreads() also emits buffer_gl0_inv on gfx11,
//     draining the memory pipeline every K-step; inputs are read-only, so
//     only the lgkmcnt/vscnt drains are needed.
//   - Grouped-M CTA swizzle (gm) for L2 reuse.
// P tail: staging rows clamp to row P-1 (duplicate compute, discarded) and
// the epilogue is predicated, so any P works. N tail: W staging rows clamp to
// row N-1 the same way and the epilogue is already n < N predicated, so any N
// works (the last BP tile just wastes some compute). Hard requirement (host-
// checked, like the MoE kernels below): K % KST == 0 — no runtime guards in
// the unrolled loops (they defeat unrolling and spill the accumulators to
// scratch).
// Measured on gfx1151 (proto, median of 20, vs hipBLASLt ~23-32 TFLOPS):
// N=2560 K=6144 ~24.5 TFLOPS (d3 cfg 512t w2x8 f4x2, gm=16);
// N=6144/10240/12288 K=2560 ~37.4-37.9 TFLOPS (d9 cfg 256t w2x4 f4x4, gm=4).
__device__ __forceinline__ void gwmma_sync() {
  __asm__ volatile(
      "s_waitcnt lgkmcnt(0)\n\t"
      "s_waitcnt_vscnt null, 0x0\n\t"
      "s_barrier" ::
          : "memory");
}
// Same as f2bf/bf2f at the top of this file; repeated inside the markers so
// the sed-extracted tools/gemm_wmma_kernel.inc stays self-contained.
__device__ __forceinline__ uint16_t gw_f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float gw_bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// Epi == 1: HC mix-up + combine fusion (replaces the (10240,320) G GEMM plus
// k_gr_combine_b_hc_bf16). W is the [4*d, K] input_mix_weight_up, logically
// permuted at staging time so logical column r maps to W row
//   ((r >> 4) & 3) * d + (r >> 6) * 16 + (r & 15),  d = N / 4,
// i.e. the WP=4 fragments of a warp are the 4 branches of the same 16
// channels, and one lane holds all 4 gates of its channel for 8 tokens. The
// epilogue then does exactly what the combine kernel did (same ops, same
// branch order): x = sum_b sigmoid(G_b) * Rhat_b / 4, fp32 to Y[P,d] and bf16
// to Yb. Only the column->block assignment changes, not the K accumulation,
// so G and x are bit-identical to the unfused pair. Requires WP == 4,
// PW * 64 == BP and N % BP == 0 (host-checked); G never touches DRAM.
template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT,
          int Epi = 0>
__global__ void __launch_bounds__(NT)
    k_gemm_wmma(const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
                float* __restrict__ Y, int P, int K, int N, int gm,
                const uint16_t* __restrict__ Rh = nullptr,
                uint16_t* __restrict__ Yb = nullptr) {
  static_assert(Epi == 0 || (WP == 4 && PW * WP * 16 == BP), "hc epi layout");
  constexpr int LSA = KST + 8;        // LDS row stride (elems)
  constexpr int BREG = 8 * LSA + 8;   // W 8-row region stride (elems)
  constexpr int NW = NT / 32;         // warps per block
  constexpr int AN = BM * (KST / 8) / NT;  // uint4 X loads per thread
  constexpr int BN = BP * (KST / 8) / NT;  // uint4 W loads per thread
  static_assert(MW * PW == NW, "warp grid mismatch");
  static_assert(BM == MW * WM * 16 && BP == PW * WP * 16, "tile mismatch");
  static_assert(BM * (KST / 8) % NT == 0 && BP * (KST / 8) % NT == 0,
                "staging mismatch");
  __shared__ uint16_t As[2][BM * LSA];
  __shared__ uint16_t Bs[2][(BP / 8) * BREG];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  // grouped swizzle for L2/MALL reuse (group along token blocks)
  const int num_m = (P + BM - 1) / BM, num_n = (N + BP - 1) / BP;
  const int bid = blockIdx.x;
  const int per_group = gm * num_n;
  const int gid = bid / per_group, rem = bid % per_group;
  const int first_m = gid * gm;
  const int gs = min(gm, num_m - first_m);
  const int bm = first_m + rem % gs;
  const int bn = rem / gs;
  const int m0 = bm * BM, n0 = bn * BP;

  const int wm0 = (w / PW) * (WM * 16);
  const int wp0 = (w % PW) * (WP * 16);
  // WMMA A layout: lane 2r -> row r, lane 17+2(r-8) -> row 8+r; other lanes
  // are ignored by the hardware. The ignored lanes are assigned complementary
  // dummy rows so every 8-lane LDS phase touches 8 distinct banks.
  const int base_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? base_ : ((base_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((base_ + 4) & 7) : base_);
  const int ksteps = K / KST;

  qw_floatx8 acc[WM][WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  // loop-invariant LDS byte offsets for this lane's fragment reads
  uint32_t aoff[WM], boff[WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
    aoff[i] = ((wm0 + i * 16 + qrowA) * LSA) * 2;
#pragma unroll
  for (int j = 0; j < WP; ++j) {
    const int pl = wp0 + j * 16 + (lane & 15);
    boff[j] = ((pl >> 3) * BREG + (pl & 7) * LSA) * 2;
  }
  // staging: bank-phase-paired row mapping. A warp stages RPW=32/(KST/8)
  // rows x (KST/8) 16B units; a ds_store_b128 retires in 8-lane phases, and
  // with the LSA=80B row stride any two CONSECUTIVE rows in a phase always
  // collide on one 4-bank group (the second row's 4 blocks always wrap onto
  // byte offset 128 == bank 0). Pairing rows (b, b+RPW/2) per phase makes
  // each phase cover all 32 banks exactly once (SQC_LDS_BANK_CONFLICT:
  // 39.8M -> 0 per dispatch). Same warp-global address set, so global
  // coalescing is unchanged; LDS layout and all load paths untouched.
  constexpr int Q8 = KST / 8;
  constexpr int RPW = 32 / Q8;
  static_assert(RPW % 2 == 0 && KST % 8 == 0, "store mapping mismatch");
  const uint16_t* ga[AN];
  const uint16_t* gb[BN];
  uint32_t saoff[AN], sboff[BN];
#pragma unroll
  for (int j = 0; j < AN; ++j) {
    const int idx = tid + j * NT;
    const int q = idx % Q8, b = (idx / Q8) % RPW;
    const int row = (idx / (Q8 * RPW)) * RPW + (b & 1) * (RPW / 2) + (b / 2);
    ga[j] = X + (size_t)min(m0 + row, P - 1) * K + q * 8;
    saoff[j] = (row * LSA + q * 8) * 2;
  }
#pragma unroll
  for (int j = 0; j < BN; ++j) {
    const int idx = tid + j * NT;
    const int q = idx % Q8, b = (idx / Q8) % RPW;
    const int row = (idx / (Q8 * RPW)) * RPW + (b & 1) * (RPW / 2) + (b / 2);
    int wrow = min(n0 + row, N - 1);
    if constexpr (Epi == 1)
      wrow = ((wrow >> 4) & 3) * (N >> 2) + (wrow >> 6) * 16 + (wrow & 15);
    gb[j] = W + (size_t)wrow * K + q * 8;
    sboff[j] = ((row >> 3) * BREG + (row & 7) * LSA + q * 8) * 2;
  }

#define GWMMA_STAGE_LOAD(RA, RB, KS)                    \
  do {                                                  \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)      \
        RA[j] = *(const uint4*)(ga[j] + (KS)*KST);      \
    _Pragma("unroll") for (int j = 0; j < BN; ++j)      \
        RB[j] = *(const uint4*)(gb[j] + (KS)*KST);      \
  } while (0)
#define GWMMA_STAGE_STORE(BUF, RA, RB)                          \
  do {                                                          \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)              \
        *(uint4*)((char*)&As[BUF][0] + saoff[j]) = RA[j];       \
    _Pragma("unroll") for (int j = 0; j < BN; ++j)              \
        *(uint4*)((char*)&Bs[BUF][0] + sboff[j]) = RB[j];       \
  } while (0)
#define GWMMA_COMPUTE(CUR)                                              \
  do {                                                                  \
    const char* Ab = (const char*)&As[CUR][0];                          \
    const char* Bb = (const char*)&Bs[CUR][0];                          \
    _Pragma("unroll") for (int kh = 0; kh < KST / 16; ++kh) {          \
      qw_shortx16 af[WM];                                               \
      _Pragma("unroll") for (int i = 0; i < WM; ++i)                    \
          af[i] = qw_ld16((const uint16_t*)(Ab + aoff[i] + kh * 32));   \
      _Pragma("unroll") for (int j = 0; j < WP; ++j) {                  \
        const qw_shortx16 bfj =                                         \
            qw_ld16((const uint16_t*)(Bb + boff[j] + kh * 32));         \
        _Pragma("unroll") for (int i = 0; i < WM; ++i)                  \
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(    \
                af[i], bfj, acc[i][j]);                                 \
      }                                                                 \
    }                                                                   \
  } while (0)

  uint4 ra0[AN], rb0[BN];
  GWMMA_STAGE_LOAD(ra0, rb0, 0);
  GWMMA_STAGE_STORE(0, ra0, rb0);
  gwmma_sync();

#pragma unroll 1
  for (int ks = 0; ks < ksteps; ++ks) {
    GWMMA_STAGE_LOAD(ra0, rb0, min(ks + 1, ksteps - 1));
    GWMMA_COMPUTE(ks & 1);
    GWMMA_STAGE_STORE((ks & 1) ^ 1, ra0, rb0);
    gwmma_sync();
  }
#undef GWMMA_STAGE_LOAD
#undef GWMMA_STAGE_STORE
#undef GWMMA_COMPUTE

  const int rh = (lane >> 4) * 8, cl = lane & 15;
  if constexpr (Epi == 1) {
    const int d = N >> 2;
    const int c = (n0 >> 2) + (wp0 >> 2) + cl;
#pragma unroll
    for (int i = 0; i < WM; ++i) {
      const int p0 = m0 + wm0 + i * 16 + rh;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int p = p0 + e;
        if (p < P) {
          const uint16_t* Rt = Rh + (size_t)p * 4 * d + c;
          float a = 0.f;
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            float gate = 1.f / (1.f + expf(-acc[i][b][e]));
            a += gate * gw_bf2f(Rt[b * d]);
          }
          const float value = a / 4;
          Y[(size_t)p * d + c] = value;
          Yb[(size_t)p * d + c] = gw_f2bf(value);
        }
      }
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j) {
      const int p0 = m0 + wm0 + i * 16 + rh;
      const int n = n0 + wp0 + j * 16 + cl;
      if (n < N)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          if (p0 + e < P) Y[(size_t)(p0 + e) * N + n] = acc[i][j][e];
    }
}
// [gemm-wmma-end]

