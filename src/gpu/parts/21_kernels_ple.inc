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
                             const float* __restrict__ ring = nullptr,
                             const int* __restrict__ d_vb = nullptr) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= (int64_t)P * n) return;
  if (d_vb) base = *d_vb;  // P3-A: device-mirrored base (graph-replayable)
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
                              int P, int n, int base = 0,
                              const int* __restrict__ d_vb = nullptr) {
  int pos = P - 9 + blockIdx.x;
  if (pos < 0) return;
  if (d_vb) base = *d_vb;  // P3-A: device-mirrored base (graph-replayable)
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

// --- PLE IQ4_NL staging dequant (GGUF table, dtype 11): raw 18 B blocks
// [fp16 d][16 B nibbles], 5 per 160-wide row, packed 16 rows per token ->
// f32 embeddings (block b -> dst[32b..32b+31]). Bit-exact with
// gguf::dequant_row(IQ4_NL) (fp16->f32 exact, one f32 multiply).
__constant__ int8_t c_kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                           1,    13,   25,  38,  53,  69,  89,  113};
__global__ void k_ple_iq4nl_dequant(const uint8_t* __restrict__ src,
                                    float* __restrict__ dst, size_t nblk) {
  size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk) return;
  const uint8_t* p = src + b * 18;
  const float d = __half2float(__ushort_as_half((uint16_t)(p[0] | (p[1] << 8))));
  float* o = dst + b * 32;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const uint8_t q = p[2 + j];
    o[j] = d * (float)c_kvalues_iq4nl[q & 0xF];
    o[j + 16] = d * (float)c_kvalues_iq4nl[q >> 4];
  }
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

// [tgk-begin] (extracted by tools/tg_kernels_test.cu via tools/tg_verify.sh)
// Two-stage argmax: stage 1 reduces each contiguous chunk to one (val, id)
// candidate packed as a sortable u64 (monotonic fp32 bits high, ~id low, so
// max key = largest value, ties -> smallest id = first occurrence). Stage 2
// reduces the few candidates. Same output value as k_argmax except on exact
// fp32 ties, where the old kernel's modulo-thread order is replaced by a
// first-occurrence rule (k_argmax stays for bit-sensitive A/B).
// Rows are blockIdx.y; cand layout [row][40].
__device__ __forceinline__ unsigned long long amx_pack(float v, int id) {
  unsigned u = __float_as_uint(v);
  u ^= (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  return ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - (unsigned)id);
}

__global__ void k_argmax_p1(const float* __restrict__ x, int n,
                            unsigned long long* __restrict__ cand) {
  const int chunk = (n + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
  const float* xr = x + (size_t)blockIdx.y * n;
  unsigned long long best = 0;  // packs below any finite/inf value
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    unsigned long long k = amx_pack(xr[i], i);
    if (k > best) best = k;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    unsigned long long o = __shfl_down_sync(~0ull, best, off);
    if (o > best) best = o;
  }
  __shared__ unsigned long long wv[32];
  const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
  if (lane == 0) wv[w] = best;
  __syncthreads();
  if (w == 0) {
    best = lane < (blockDim.x + warpSize - 1) / warpSize ? wv[lane] : 0;
    for (int off = 16; off > 0; off >>= 1) {
      unsigned long long o = __shfl_down_sync(~0ull, best, off);
      if (o > best) best = o;
    }
    if (lane == 0) cand[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = best;
  }
}

__global__ void k_argmax_p2(const unsigned long long* __restrict__ cand,
                            int mcand, int* __restrict__ out) {
  const unsigned long long* cr = cand + (size_t)blockIdx.y * mcand;
  unsigned long long best = 0;
  for (int i = threadIdx.x; i < mcand; i += blockDim.x)
    if (cr[i] > best) best = cr[i];
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    unsigned long long o = __shfl_down_sync(~0ull, best, off);
    if (o > best) best = o;
  }
  if (threadIdx.x == 0)
    out[blockIdx.y] = (int)(0xFFFFFFFFu - (unsigned)(best & 0xFFFFFFFFull));
}

// 40 blocks/row: one per CU on gfx1151; cand scratch is rows * 40 * 8 bytes.
#define ARGMAX_B1 40
static void argmax_fast_rows(const float* x, int n, int rows,
                             unsigned long long* d_cand, int* out,
                             hipStream_t st) {
  k_argmax_p1<<<dim3(ARGMAX_B1, rows), 256, 0, st>>>(x, n, d_cand);
  // p2 is a single-wave reduction (no cross-wave step): launch exactly one
  // wave. A 64-thread block on wave32 dropped candidates 32..39.
  k_argmax_p2<<<dim3(1, rows), 32, 0, st>>>(d_cand, ARGMAX_B1, out);
}

// Default: two-stage. QWENOX_ARGMAX_OLD restores the single-block kernel
// (identical values except on exact fp32 ties; see k_argmax_p1).
static const bool g_argmax_old = getenv("QWENOX_ARGMAX_OLD") != nullptr;
static void argmax_rows(const float* x, int n, int rows,
                        unsigned long long* d_cand, int* out, hipStream_t st) {
  if (!g_argmax_old && rows <= 65) {
    argmax_fast_rows(x, n, rows, d_cand, out, st);
    return;
  }
  for (int r = 0; r < rows; r++)
    k_argmax<<<1, 1024, 0, st>>>(x + (size_t)r * n, n, out + r);
}

// --- GPU sampling prep (spec sampling fast path) ---
// k_corr_apply replays HostSampler::prepare's correction phases on device:
// bias adds first, then penalty subtracts, then this row's draft-prefix
// penalty subtracts. Each list has unique ids, so within a phase the adds are
// independent; phases are ordered by __syncthreads (one block per row).
__global__ void k_corr_apply(float* __restrict__ x, int stride,
                             const int* __restrict__ b_ids,
                             const float* __restrict__ b_vals, int nb,
                             const int* __restrict__ h_ids,
                             const float* __restrict__ h_vals, int nh,
                             const int* __restrict__ p_ids,
                             const float* __restrict__ p_vals,
                             const int* __restrict__ p_n, int pcap) {
  float* xr = x + (size_t)blockIdx.y * stride;
  for (int i = threadIdx.x; i < nb; i += blockDim.x) xr[b_ids[i]] += b_vals[i];
  __syncthreads();
  for (int i = threadIdx.x; i < nh; i += blockDim.x) xr[h_ids[i]] -= h_vals[i];
  __syncthreads();
  const int np = p_n ? p_n[blockIdx.y] : 0;
  const int* pr_ids = p_ids + (size_t)blockIdx.y * pcap;
  const float* pr_vals = p_vals + (size_t)blockIdx.y * pcap;
  for (int i = threadIdx.x; i < np; i += blockDim.x) xr[pr_ids[i]] -= pr_vals[i];
}

// Radix top-k over fp32 logits via their monotonic u32 keys (same map as
// amx_pack). Two 12-bit histogram passes narrow the k-th value to a 24-bit
// prefix; collect then writes every key >= T (the true top-k is a subset).
// ctrl per row (8 ints): 0=prefix, 1=strictly-in count, 2=k, 3=done, 4=T,
// 5=expected collect count.
__device__ __forceinline__ unsigned tk_key(float v) {
  unsigned u = __float_as_uint(v);
  return u ^ ((u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

template <int PASS>
__global__ void k_tk_hist(const float* __restrict__ x, int n, int stride,
                          const int* __restrict__ ctrl,
                          unsigned* __restrict__ hist) {
  __shared__ unsigned sh[4096];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  const int row = blockIdx.y;
  if (!PASS || !ctrl[(size_t)row * 8 + 3]) {
    const unsigned pfx = PASS ? (unsigned)ctrl[(size_t)row * 8 + 0] : 0u;
    const float* xr = x + (size_t)row * stride;
    const int chunk = (n + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      const unsigned u = tk_key(xr[i]);
      if (PASS && (u >> 20) != pfx) continue;
      atomicAdd(&sh[PASS ? ((u >> 8) & 0xFFFu) : (u >> 20)], 1u);
    }
  }
  __syncthreads();
  unsigned* h = hist + (size_t)row * 4096;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x)
    if (sh[i]) atomicAdd(&h[i], sh[i]);
}

template <int PASS>
__global__ void k_tk_scan(const unsigned* __restrict__ hist,
                          int* __restrict__ ctrl, int* __restrict__ cand_n) {
  const int row = blockIdx.y;
  int* c = ctrl + (size_t)row * 8;
  if (PASS && c[3]) return;
  const unsigned* h = hist + (size_t)row * 4096;
  const int t = threadIdx.x;
  const int needed = c[2] - c[1];
  __shared__ unsigned wsum[8];
  __shared__ unsigned s_d, s_gt, s_bin;
  if (t == 0) s_d = 0xFFFFFFFFu;
  unsigned v[16];
  unsigned lsum = 0;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    v[j] = h[t * 16 + j];
    lsum += v[j];
  }
  unsigned x = lsum;
  // Inclusive scan within each 32-lane group (all 256 threads active), then a
  // serial scan of the 8 group sums. No *_sync shuffles in a partial wave:
  // HIP traps when the mask names inactive lanes.
  for (int off = 1; off < 32; off <<= 1) {
    unsigned y = __shfl_up(x, off, 32);
    if ((t & 31) >= off) x += y;
  }
  if ((t & 31) == 31) wsum[t >> 5] = x;
  __syncthreads();
  if (t == 0)
    for (int w = 1; w < 8; w++) wsum[w] += wsum[w - 1];
  __syncthreads();
  const unsigned total = wsum[7];
  unsigned run = x - lsum + (t >= 32 ? wsum[(t >> 5) - 1] : 0);
  unsigned best = 0xFFFFFFFFu, bgt = 0, bbin = 0;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const unsigned s_ge = total - run, s_gt = s_ge - v[j];
    if (s_ge >= (unsigned)needed && s_gt < (unsigned)needed) {
      best = t * 16 + j;
      bgt = s_gt;
      bbin = v[j];
    }
    run += v[j];
  }
  if (best != 0xFFFFFFFFu) {
    s_d = best;
    s_gt = bgt;
    s_bin = bbin;
  }
  __syncthreads();
  if (t == 0) {
    if (s_d == 0xFFFFFFFFu) {  // total < needed: impossible for k <= n
      c[4] = 0;
      c[5] = 1 << 30;
      c[3] = 1;
      cand_n[row] = 0;
      return;
    }
    const unsigned pfx = PASS ? (((unsigned)c[0] << 12) | s_d) : s_d;
    c[0] = (int)pfx;
    c[1] += (int)s_gt;
    if (PASS) {
      c[4] = (int)(pfx << 8);
      c[5] = c[1] + (int)s_bin;
      c[3] = 1;
      cand_n[row] = 0;
    }
  }
}

__global__ void k_tk_init(int* __restrict__ ctrl, int k) {
  int* c = ctrl + (size_t)blockIdx.x * 8;
  if (threadIdx.x == 0) {
    c[0] = 0;
    c[1] = 0;
    c[2] = k;
    c[3] = 0;
    c[4] = 0;
    c[5] = 0;
  }
}

__global__ void k_tk_collect(const float* __restrict__ x, int n, int stride,
                             const int* __restrict__ ctrl,
                             int* __restrict__ ids, float* __restrict__ vals,
                             int* __restrict__ cand_n, int tcap) {
  const int row = blockIdx.y;
  const unsigned T = (unsigned)ctrl[(size_t)row * 8 + 4];
  const float* xr = x + (size_t)row * stride;
  const int chunk = (n + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const float xv = xr[i];
    if (tk_key(xv) >= T) {
      const int idx = atomicAdd(&cand_n[row], 1);
      if (idx < tcap) {
        ids[(size_t)row * tcap + idx] = i;
        vals[(size_t)row * tcap + idx] = xv;
      }
    }
  }
}

// [tgk-end]

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
