// ================== GDN chunked prefill (Phase 3b) kernels ===================

// --- batched conv+silu: out[t][ch] = silu(sum_j cw[ch][j]*hist_j + cw[ch][3]*x[t])
// hist_j = x[t-3+j], t-3+j < 0 reads convst (recent 3 pre-prefill inputs, oldest
// first — matches k_gdn_conv). Not in-place: out must differ from x.
// float4 over channels (qkv % 4 == 0): per-lane op order identical to the
// scalar version -> bit-exact, 4x fewer threads.
__global__ void k_gdn_conv_b(const float* __restrict__ x, const float* __restrict__ cw,
                             const float* __restrict__ convst, float* __restrict__ out,
                             int P, int qkv) {
  int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) * 4;
  if (i >= (int64_t)P * qkv) return;
  int t = (int)(i / qkv), ch = (int)(i - (int64_t)t * qkv);
  float4 xv = *(const float4*)(x + i);
  float4 a;
  a.x = cw[(ch + 0) * 4 + 3] * xv.x;
  a.y = cw[(ch + 1) * 4 + 3] * xv.y;
  a.z = cw[(ch + 2) * 4 + 3] * xv.z;
  a.w = cw[(ch + 3) * 4 + 3] * xv.w;
#pragma unroll
  for (int j = 0; j < 3; j++) {
    int tt = t - 3 + j;
    const float* src =
        tt >= 0 ? x + (size_t)tt * qkv + ch : convst + (tt + 3) * qkv + ch;
    float4 v = *(const float4*)src;
    a.x += cw[(ch + 0) * 4 + j] * v.x;
    a.y += cw[(ch + 1) * 4 + j] * v.y;
    a.z += cw[(ch + 2) * 4 + j] * v.z;
    a.w += cw[(ch + 3) * 4 + j] * v.w;
  }
  float4 o;
  o.x = a.x / (1.f + expf(-a.x));
  o.y = a.y / (1.f + expf(-a.y));
  o.z = a.z / (1.f + expf(-a.z));
  o.w = a.w / (1.f + expf(-a.w));
  *(float4*)(out + i) = o;
}

// --- write back conv state: convst[j] = last-3 inputs after P tokens.
// Per-channel independent; old values are read into registers before writing.
__global__ void k_convst_update(float* __restrict__ convst, const float* __restrict__ x,
                                int P, int qkv) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= qkv) return;
  float o1 = convst[qkv + ch], o2 = convst[2 * qkv + ch];
  float n2 = x[(size_t)(P - 1) * qkv + ch];
  float n1 = P >= 2 ? x[(size_t)(P - 2) * qkv + ch] : o2;
  // P=1 keeps old rows 1,2; P=2 keeps only old row 2.
  float n0 = P >= 3 ? x[(size_t)(P - 3) * qkv + ch] : (P == 2 ? o2 : o1);
  convst[ch] = n0;
  convst[qkv + ch] = n1;
  convst[2 * qkv + ch] = n2;
}

// --- batched GDN q/k L2 norm + q scale: grid (hk, P) over [P, qkvstride] rows.
// q heads at t*qkvstride + h*dk, k heads at t*qkvstride + hk*dk + h*dk.
__global__ void k_l2norm_qk_b(float* __restrict__ qkv, int dk, float qscale,
                              int qkvstride, int hk) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  float* qh = qkv + (size_t)blockIdx.y * qkvstride + h * dk;
  float* kh = qkv + (size_t)blockIdx.y * qkvstride + (size_t)hk * dk + h * dk;
  float v = (t < dk) ? qh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float iq = rsqrtf(sm[0] + 1e-6f);
  __syncthreads();
  float kv = (t < dk) ? kh[t] : 0.f;
  sm[t] = kv * kv;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float ik = rsqrtf(sm[0] + 1e-6f);
  if (t < dk) {
    qh[t] = v * iq * qscale;
    kh[t] = kv * ik;
  }
}

// --- GDN chunked prefill (torch_chunk_gated_delta_rule equivalent), fp32.
// One block (256 threads) per v-head; chunks of 64 tokens serially.
// qkv: conv-ed rows [P, qkvstride]; v-head h reads q,k at kh=h/3 (GVA mapping
// of k_gdn_step: 48 v-heads share 16 k-heads), v at 4096+h*128. Output is
// written in place over v. q arrives already l2normed and scaled by
// 1/sqrt(128) (k_l2norm_qk), so no extra scale here. g/beta rows [P, 48];
// g is the per-token log-decay (chunk-local cumsum, padded zeros contribute
// nothing). S [128,128] per head is read, updated per chunk, written back.
// ws: per-head scratch [2][64][128] fp32 (value'/v_new, k_cumdecay).
// Optimized (Phase 3d-2): the naive version streamed S from global three
// times per chunk (~8MB/chunk/head) and re-read q/v rows per output element.
// Here S is processed once per chunk in 16-column LDS d-tiles that fuse
// v_new = value' - kcd@S, out = qeg@S + attn2@v_new and the S update into a
// single read+write; the v'/kcd pass maps each thread to one d-column (32
// rows) so v rows are read once with no staging; the 63-step forward
// substitution runs warp-synchronously (one warp, lanes over columns, no
// block barriers); expf(gcum) / expf(glast-gcum) are hoisted to per-chunk
// vectors (bit-identical values). Per-element accumulation orders (j/d/r
// ascending) are unchanged, so results match the naive version bit for bit.
#ifndef GDN_NT
#define GDN_NT 1024  // k_gdn_chunk block size (occupancy tuning knob)
#endif
__global__ void k_gdn_chunk(const float* __restrict__ qkv, const float* __restrict__ gb,
                            const float* __restrict__ bb, float* __restrict__ S,
                            float* __restrict__ out, float* __restrict__ ws, int P,
                            int qkvstride) {
  const int CH = 64, DK = 128;
  const int NT = GDN_NT;  // launch bound: blockDim.x must match
  int h = blockIdx.x;
  int kh = h / 3;
  int tid = threadIdx.x;
  __shared__ float s_k[CH][DK + 4];  // row pad: 32-way bank conflicts -> 4-way
  __shared__ float s_attn[CH][CH];
  __shared__ float s_extra[3072];  // q quarter tile [16][128] | S tile [128][16] + vnew [64][16]
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH], s_glast;
  // s_beta is reused for expf(glast-gcum) after the v'/kcd pass
  S += (size_t)h * DK * DK;
  float* wvp = ws + (size_t)h * 2 * CH * DK;  // value' then v_new
  float* wkcd = wvp + CH * DK;                // k_cumdecay
  int nchunks = (P + CH - 1) / CH;
  for (int c = 0; c < nchunks; c++) {
    int t0 = c * CH;
    // load k tile, g, beta; per-chunk cumsum of g (pad rows: k=0, g=0, beta=0)
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      s_k[t][d] =
          (t0 + t < P) ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d] : 0.f;
    }
    if (tid < CH) {
      s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
      s_beta[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
    }
    __syncthreads();
    if (tid == 0) {
      float acc = 0.f;
      for (int i = 0; i < CH; i++) {
        acc += s_gcum[i];
        s_gcum[i] = acc;
      }
      s_glast = s_gcum[CH - 1];
    }
    __syncthreads();
    if (tid < CH) s_eg[tid] = expf(s_gcum[tid]);
    // attn[r][j] = -beta[r] * (k[r].k[j]) * exp(gcum[r]-gcum[j]) for j < r, else 0
    for (int i = tid; i < CH * CH; i += NT) {
      int r = i / CH, j = i % CH;
      float a = 0.f;
      if (j < r) {
        float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        for (int d = 0; d < DK; d += 4) {
          e0 += s_k[r][d] * s_k[j][d];
          e1 += s_k[r][d + 1] * s_k[j][d + 1];
          e2 += s_k[r][d + 2] * s_k[j][d + 2];
          e3 += s_k[r][d + 3] * s_k[j][d + 3];
        }
        float dot = (e0 + e1) + (e2 + e3);
        a = -s_beta[r] * dot * expf(s_gcum[r] - s_gcum[j]);
      }
      s_attn[r][j] = a;
    }
    __syncthreads();
    // forward substitution: attn[i,:i] += sum_m attn[i,m](old) * attn[m,:i](updated).
    // Rows are serially dependent; one warp processes them with lanes over
    // columns (m ascending per element — same order as the naive loop).
    {
      // NT threads = NT/64 m-groups (tid>>6) x 64 columns (tid&63); per row each
      // group sums its strided m range into s_extra, then group 0 merges.
      const int j = tid & (CH - 1), mg = tid >> 6;
      for (int i = 1; i < CH; i++) {
        float p = 0.f;
        for (int m = mg; m < i; m += NT / CH) p += s_attn[i][m] * s_attn[m][j];
        s_extra[mg * CH + j] = p;
        __syncthreads();
        if (mg == 0 && j < i) {
          float s = s_attn[i][j];
#pragma unroll
          for (int g = 0; g < NT / CH; g += 2)
            s += s_extra[g * CH + j] + s_extra[(g + 1) * CH + j];
          s_attn[i][j] = s;
        }
        __syncthreads();
      }
    }
    __syncthreads();
    if (tid < CH) s_attn[tid][tid] += 1.f;
    __syncthreads();
    // value' = attn @ (v*beta); k_cumdecay = attn @ (k*beta*exp(gcum)).
    // Thread owns one d-column (rows r = tid/128 + 2n): each v row is loaded
    // once per thread and s_attn[r][j] reads broadcast within thread pairs.
    {
      const int d = tid & (DK - 1);
      float av[CH * DK / NT], ak[CH * DK / NT];
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++) av[n] = 0.f, ak[n] = 0.f;
      // software-pipeline the global v row reads one step ahead
      float vj_nxt = (t0 < P) ? qkv[(size_t)t0 * qkvstride + 4096 + h * DK + d] : 0.f;
      for (int j = 0; j < CH; j++) {
        float vj = vj_nxt;
        if (j + 1 < CH) {
          int tn = t0 + j + 1;
          vj_nxt = (tn < P) ? qkv[(size_t)tn * qkvstride + 4096 + h * DK + d] : 0.f;
        }
        float vjb = vj * s_beta[j];
        float kjb = s_k[j][d] * s_beta[j] * s_eg[j];
        int r = tid >> 7;
#pragma unroll
        for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
          float a = s_attn[r][j];
          av[n] += a * vjb;
          ak[n] += a * kjb;
        }
      }
      int r = tid >> 7;
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
        wvp[r * DK + d] = av[n];
        wkcd[r * DK + d] = ak[n];
      }
    }
    __syncthreads();
    // attn2[r][j] = (q[r].k[j]) * exp(gcum[r]-gcum[j]) for j <= r (overwrite
    // s_attn); q staged in 16-row quarter tiles
    for (int q0 = 0; q0 < CH; q0 += 16) {
      for (int i = tid; i < 16 * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + q0 + t < P)
                         ? qkv[(size_t)(t0 + q0 + t) * qkvstride + kh * DK + dd]
                         : 0.f;
      }
      __syncthreads();
      for (int i = tid; i < 16 * CH; i += NT) {
        int r = q0 + i / CH, j = i % CH;
        float a = 0.f;
        if (j <= r && t0 + r < P) {
          const float* qr = &s_extra[(i / CH) * DK];
          float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
          for (int dd = 0; dd < DK; dd += 4) {
            e0 += qr[dd] * s_k[j][dd];
            e1 += qr[dd + 1] * s_k[j][dd + 1];
            e2 += qr[dd + 2] * s_k[j][dd + 2];
            e3 += qr[dd + 3] * s_k[j][dd + 3];
          }
          float dot = (e0 + e1) + (e2 + e3);
          a = dot * expf(s_gcum[r] - s_gcum[j]);
        }
        s_attn[r][j] = a;
      }
      __syncthreads();
    }
    // fused d-tile pass over S (16 columns per tile): v_new, out, S update.
    // expf(glast - gcum) reuses s_beta (dead after the v'/kcd pass)
    if (tid < CH) s_beta[tid] = expf(s_glast - s_gcum[tid]);
    float egl = expf(s_glast);
    __syncthreads();
    float* s_St = s_extra;          // [DK][16]
    float* s_vn = s_extra + 2048;   // [CH][16]
    const int dc = tid & 15;
    const int RPT = CH * 16 / NT;   // S-phase rows per thread
    const int JPT = DK * 16 / NT;   // S-phase j-rows per thread
    const int RG = NT / 16;         // row-group stride
    for (int d0 = 0; d0 < DK; d0 += 16) {
      for (int i = tid; i < DK * 16; i += NT) s_St[i] = S[(i / 16) * DK + d0 + (i % 16)];
      __syncthreads();
      // v_new[r][dc] = value'[r][d] - kcd[r] . S[:, d]  (j ascending)
      {
        // j-outer: s_St tile loads shared across the rows per thread
        int r = tid >> 4;
        const float* w0 = wkcd + r * DK;
        float e[RPT][4];
#pragma unroll
        for (int n = 0; n < RPT; n++)
#pragma unroll
          for (int u = 0; u < 4; u++) e[n][u] = 0.f;
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j * 16 + dc], s1 = s_St[(j + 1) * 16 + dc];
          float s2 = s_St[(j + 2) * 16 + dc], s3 = s_St[(j + 3) * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            float4 w = *(const float4*)(w0 + n * RG * DK + j);
            e[n][0] += w.x * s0;
            e[n][1] += w.y * s1;
            e[n][2] += w.z * s2;
            e[n][3] += w.w * s3;
          }
        }
#pragma unroll
        for (int n = 0; n < RPT; n++, r += RG)
          s_vn[r * 16 + dc] =
              wvp[r * DK + d0 + dc] - ((e[n][0] + e[n][1]) + (e[n][2] + e[n][3]));
      }
      __syncthreads();
      // out[r][d] = (q[r]*exp(gcum[r])) . S[:, d] + attn2[r] . v_new[:, d]
      {
        // j-outer with row accumulators: s_St / s_vn loads shared across rows
        int r = tid >> 4;
        const float* q0 = qkv + kh * DK;
        bool v[RPT];
        float eg[RPT], o[RPT];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          v[n] = (t0 + r + n * RG) < P;
          eg[n] = s_eg[r + n * RG];
          o[n] = 0.f;
        }
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j * 16 + dc], s1 = s_St[(j + 1) * 16 + dc];
          float s2 = s_St[(j + 2) * 16 + dc], s3 = s_St[(j + 3) * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            const float* qr = q0 + (size_t)(v[n] ? t0 + r + n * RG : t0) * qkvstride;
            float4 q4 = *(const float4*)(qr + j);
            o[n] += q4.x * s0 + q4.y * s1 + q4.z * s2 + q4.w * s3;
          }
        }
#pragma unroll
        for (int n = 0; n < RPT; n++) o[n] *= eg[n];
        for (int j = 0; j < CH; j++) {
          float vn = s_vn[j * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++)
            o[n] += ((const float*)s_attn)[(r + n * RG) * CH + j] * vn;
        }
#pragma unroll
        for (int n = 0; n < RPT; n++)
          if (v[n])
            out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] = o[n];
      }
      // S[j][d] = S[j][d]*exp(glast) + sum_r k[r][j]*exp(glast-gcum[r])*vnew[r][d]
      {
        // r-outer with j accumulators: s_vn / s_beta loads shared across j
        int j0 = tid >> 4;
        float a[JPT];
#pragma unroll
        for (int n = 0; n < JPT; n++) a[n] = 0.f;
        for (int r = 0; r < CH; r++) {
          float vn = s_vn[r * 16 + dc];
          float bt = s_beta[r];
          const float* kr = &s_k[r][0];
#pragma unroll
          for (int n = 0; n < JPT; n++) a[n] += kr[j0 + n * RG] * bt * vn;
        }
#pragma unroll
        for (int n = 0; n < JPT; n++) {
          int j = j0 + n * RG;
          S[j * DK + d0 + dc] = s_St[j * 16 + dc] * egl + a[n];
        }
      }
      __syncthreads();
    }
  }
}

// --- Phase 3b split: k_gdn_chunk as an intra/inter kernel pair --------------
// k_gdn_intra is state-independent: one block per (chunk, head), grid
// (nchunks, 48); it runs the load/cumsum/attn/forward-substitution/value'/kcd/
// attn2 phases verbatim from k_gdn_chunk and stages the chunk-local results
// fp32 in ws. k_gdn_inter keeps the per-head serial chunk scan but only runs
// the fused d-tile pass (v_new / out / S update), re-reading k and q from qkv
// (out still overwrites the v segment in place; v was fully consumed by
// k_gdn_intra). All per-element accumulation orders (j/d/r ascending) and the
// hoisted expf() values match k_gdn_chunk, so out and S are bit-identical to
// the single-kernel path. GDEC_GDN_NOSPLIT=1 selects the old kernel instead.
// Workspace layout per (chunk c, head h), base = ws + (c*48+h)*GDN_SPLIT_WS_FLOATS:
//   [0,64) gcum (inclusive; [63] = glast) | [64,4160) attn2 [64][64] |
//   [4160,12352) value' [64][128] | [12352,20544) kcd [64][128]
//   (20544 floats = 80.25 KiB per chunk-head; 16B-aligned sections)
#define GDN_WS_ATTN2 64
#define GDN_WS_VP (GDN_WS_ATTN2 + 64 * 64)
#define GDN_WS_KCD (GDN_WS_VP + 64 * 128)
#define GDN_SPLIT_WS_FLOATS (GDN_WS_KCD + 64 * 128)
// dev-only per-phase clock64 accounting (build with -DGDEC_PHASE_PROF;
// zeroed per prefill_chunk call, dumped on GDEC_PHASE_PROF env). Slots:
// 0 staging+cumsum, 1 attn dots, 2 solve, 3 vp/kcd, 4 attn2+writeout,
// 5 strip S staging, 6 strip v_new, 7 strip out, 8 strip S update,
// 9 intra total, 10 strip total (per-block cycles, tid0 atomicAdd).
#ifdef GDEC_PHASE_PROF
__device__ unsigned long long g_ph[16];
// gfx1151 has no s_memrealtime; clock64 is per-CU. Compiler barriers pin the
// reads in place, and the guard drops wrapped/preempted samples (2^64 junk
// seen with the unguarded version).
__device__ __forceinline__ long long ph_clk() {
  __asm__ volatile("" ::: "memory");
  long long v = clock64();
  __asm__ volatile("" ::: "memory");
  return v;
}
#define PH_DECL(t) long long t = ph_clk()
#define PH_ACC(slot, t)                          \
  do {                                           \
    __syncthreads();                             \
    long long e = ph_clk();                      \
    long long d = e - (t);                       \
    if (threadIdx.x == 0 && d >= 0 && d < (1LL << 40)) \
      atomicAdd(&g_ph[slot], (unsigned long long)d); \
  } while (0)
#else
#define PH_DECL(t)
#define PH_ACC(slot, t)
#endif
#ifndef GDN_NT_INTRA
// NOTE: values other than GDN_NT change the forward-substitution m-grouping
// and break bit-identity with k_gdn_chunk (still well within ktest tolerance).
#define GDN_NT_INTRA GDN_NT
#endif

// k_gdn_intra is templated on the block size; NT=GDN_NT keeps the
// forward-substitution m-grouping (and thus the results) bit-identical to
// k_gdn_chunk. (gfx1151 has only 64 KiB LDS per WGP, so the ~63 KiB block
// can never co-reside with a second one — smaller NT does not raise
// occupancy; measured 512 threads merely trades barrier cost for compute.)
template <int NT, bool WaveSolve = false, bool Ut5 = false>
__global__ void k_gdn_intra(const float* __restrict__ qkv, const float* __restrict__ gb,
                            const float* __restrict__ bb, float* __restrict__ ws, int P,
                            int qkvstride) {
  const int CH = 64, DK = 128;
  int c = blockIdx.x, h = blockIdx.y;
  int kh = h / 3;
  int tid = threadIdx.x;
  __shared__ float s_k[CH][DK + 4];  // row pad: 32-way bank conflicts -> 4-way
  __shared__ float s_attn[CH][CH];
  __shared__ float s_extra[3072];  // q quarter tile [16][128]
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH], s_glast;
  float* wsb = ws + ((size_t)c * 48 + h) * GDN_SPLIT_WS_FLOATS;
  float* wvp = wsb + GDN_WS_VP;    // value'
  float* wkcd = wsb + GDN_WS_KCD;  // k_cumdecay
  int t0 = c * CH;
  PH_DECL(tk);
  // load k tile, g, beta; per-chunk cumsum of g (pad rows: k=0, g=0, beta=0)
  for (int i = tid; i < CH * DK; i += NT) {
    int t = i / DK, d = i % DK;
    s_k[t][d] =
        (t0 + t < P) ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d] : 0.f;
  }
  if (tid < CH) {
    s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
    s_beta[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0.f;
    for (int i = 0; i < CH; i++) {
      acc += s_gcum[i];
      s_gcum[i] = acc;
    }
    s_glast = s_gcum[CH - 1];
  }
  __syncthreads();
  if (tid < CH) {
    s_eg[tid] = expf(s_gcum[tid]);
    wsb[tid] = s_gcum[tid];  // stage gcum for k_gdn_inter
  }
  PH_ACC(0, tk);
  PH_DECL(t1);
  // attn[r][j] = -beta[r] * (k[r].k[j]) * exp(gcum[r]-gcum[j]) for j < r, else 0
  for (int i = tid; i < CH * CH; i += NT) {
    int r = i / CH, j = i % CH;
    float a = 0.f;
    if (j < r) {
      float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
      for (int d = 0; d < DK; d += 4) {
        e0 += s_k[r][d] * s_k[j][d];
        e1 += s_k[r][d + 1] * s_k[j][d + 1];
        e2 += s_k[r][d + 2] * s_k[j][d + 2];
        e3 += s_k[r][d + 3] * s_k[j][d + 3];
      }
      float dot = (e0 + e1) + (e2 + e3);
      a = -s_beta[r] * dot * expf(s_gcum[r] - s_gcum[j]);
    }
    s_attn[r][j] = a;
  }
  __syncthreads();
  PH_ACC(1, t1);
  PH_DECL(t2);
  if constexpr (Ut5) {
  // ut5-style blocked inverse of (I - A), A strictly lower in s_attn.
  // Diagonal 16x16 blocks: the same serial row recurrence as WaveSolve on
  // four independent 256-thread groups; off-block T entries are exact zeros
  // in the full recurrence, so diag blocks stay bit-identical. Off-diagonal
  // blocks: T_jb = D_j^-1 . (A_jb + sum_{k=b}^{j-1} A_jk T_kb) in three
  // dependency stages of 16x16x16 fp32 tile GEMMs (reassociation only).
  // s_extra layout: U tiles [0,768), T tiles [1024,2560); slot of (j,b) is
  // tl = j*(j-1)/2 + b.
  static_assert(NT == 1024, "ut5 solve uses 4x256 threads");
  {
    const int g = tid >> 8, el = tid & 255;
    const int er = el >> 4, ec = el & 15;
    {
      const int col = er, mg = ec, base = g * 16;
      float solved = 0.f;
      for (int i = 1; i < 16; ++i) {
        float p = 0.f;
        if (mg < i) p = s_attn[base + i][base + mg] * solved;
        float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
        float value = s_attn[base + i][base + col];
#pragma unroll
        for (int gg = 0; gg < 16; gg += 2)
          value += __shfl_sync(~0ull, pair, gg, 16);
        if (i == mg && col < i) solved = value;
      }
      __syncthreads();  // all warps done reading A before it is overwritten
      if (col < mg) s_attn[base + mg][base + col] = solved;
    }
    __syncthreads();
    // stage A: T_10, T_21, T_32 (U = A_jb + A_jb . S_bb)
    if (g < 3) {
      const int j = g + 1, b = g;
      float a[16];
#pragma unroll
      for (int n = 0; n < 16; n++) a[n] = s_attn[j * 16 + er][b * 16 + n];
      float u = a[ec];
#pragma unroll
      for (int n = 0; n < 16; n++)
        u += a[n] * s_attn[b * 16 + n][b * 16 + ec];
      s_extra[g * 256 + el] = u;
    }
    __syncthreads();
    if (g < 3) {
      const int j = g + 1, b = g;
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[j * 16 + er][j * 16 + m];
      float t = s_extra[g * 256 + el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[g * 256 + m * 16 + ec];
      s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
    }
    __syncthreads();
    // stage B: T_20, T_31 (U = A_jb + A_jb . S_bb + A_j,b+1 . T_b+1,b)
    if (g < 2) {
      const int j = g + 2, b = g;
      float a0[16], a1[16];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        a0[n] = s_attn[j * 16 + er][b * 16 + n];
        a1[n] = s_attn[j * 16 + er][(b + 1) * 16 + n];
      }
      float u = a0[ec];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        u += a0[n] * s_attn[b * 16 + n][b * 16 + ec];
        u += a1[n] * s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 + ec];
      }
      s_extra[g * 256 + el] = u;
    }
    __syncthreads();
    if (g < 2) {
      const int j = g + 2, b = g;
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[j * 16 + er][j * 16 + m];
      float t = s_extra[g * 256 + el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[g * 256 + m * 16 + ec];
      s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
    }
    __syncthreads();
    // stage C: T_30 (U = A_30 + A_30.S_00 + A_31.T_10 + A_32.T_20)
    if (g == 0) {
      float a0[16], a1[16], a2[16];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        a0[n] = s_attn[48 + er][n];
        a1[n] = s_attn[48 + er][16 + n];
        a2[n] = s_attn[48 + er][32 + n];
      }
      float u = a0[ec];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        u += a0[n] * s_attn[n][ec];
        u += a1[n] * s_extra[1024 + n * 16 + ec];        // T_10
        u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];  // T_20
      }
      s_extra[el] = u;
    }
    __syncthreads();
    if (g == 0) {
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[48 + er][48 + m];
      float t = s_extra[el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[m * 16 + ec];
      s_extra[1024 + 3 * 256 + el] = t;
    }
    __syncthreads();
    // publish the off-diagonal blocks into s_attn
    for (int j = 1; j < 4; j++)
      for (int b = 0; b < j; b++) {
        const int tl = j * (j - 1) / 2 + b;
        for (int e = tid; e < 256; e += NT)
          s_attn[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
              s_extra[1024 + tl * 256 + e];
      }
  }

  } else if constexpr (WaveSolve) {
  // Each 16-lane group owns a column of the triangular inverse. Lane g
  // keeps rows g,g+16,g+32,g+48 in registers. Preserve the old 16 partial
  // sums and its sequential pair merges, without publishing every row in LDS.
  static_assert(NT == 1024, "the bit-exact solve uses 16 m-groups");
  {
    const int col = tid / 16, mg = tid % 16;
    float solved[4] = {};
    for (int i = 1; i < CH; ++i) {
      float p = 0.f;
#pragma unroll
      for (int slot = 0; slot < 4; ++slot) {
        int m = mg + 16 * slot;
        if (m < i) p += s_attn[i][m] * solved[slot];
      }
      float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
      float value = s_attn[i][col];
#pragma unroll
      for (int g = 0; g < 16; g += 2)
        value += __shfl_sync(~0ull, pair, g, 16);
#pragma unroll
      for (int slot = 0; slot < 4; ++slot)
        if (i == mg + 16 * slot && col < i) solved[slot] = value;
    }
    // The attn2 workspace is not live yet. Use it as a transpose buffer;
    // all waves must finish reading the original matrix before LDS changes.
#pragma unroll
    for (int slot = 0; slot < 4; ++slot)
      wsb[GDN_WS_ATTN2 + col * CH + mg + 16 * slot] = solved[slot];
    __syncthreads();
    for (int i = tid; i < CH * CH; i += NT)
      s_attn[i / CH][i % CH] = wsb[GDN_WS_ATTN2 + (i % CH) * CH + i / CH];
  }

  } else {
  // forward substitution: attn[i,:i] += sum_m attn[i,m](old) * attn[m,:i](updated).
  // Rows are serially dependent; one warp processes them with lanes over
  // columns (m ascending per element — same order as the naive loop).
  {
    // NT threads = NT/64 m-groups (tid>>6) x 64 columns (tid&63); per row each
    // group sums its strided m range into s_extra, then group 0 merges.
    const int j = tid & (CH - 1), mg = tid >> 6;
    for (int i = 1; i < CH; i++) {
      float p = 0.f;
      for (int m = mg; m < i; m += NT / CH) p += s_attn[i][m] * s_attn[m][j];
      s_extra[mg * CH + j] = p;
      __syncthreads();
      if (mg == 0 && j < i) {
        float s = s_attn[i][j];
#pragma unroll
        for (int g = 0; g < NT / CH; g += 2)
          s += s_extra[g * CH + j] + s_extra[(g + 1) * CH + j];
        s_attn[i][j] = s;
      }
      __syncthreads();
    }
  }

  }
  __syncthreads();
  PH_ACC(2, t2);
  PH_DECL(t3);
  if (tid < CH) s_attn[tid][tid] += 1.f;
  __syncthreads();
  // value' = attn @ (v*beta); k_cumdecay = attn @ (k*beta*exp(gcum)).
  // Thread owns one d-column (rows r = tid/128 + 2n). v rows are staged
  // through s_extra in 16-row bands (s_extra is dead between the substitution
  // and the attn2 q staging) so the 64 serial global-read latencies collapse
  // into four coalesced band loads; s_attn[r][j] reads broadcast within
  // thread pairs. Values and accumulation order are unchanged.
  {
    const int d = tid & (DK - 1);
    float av[CH * DK / NT], ak[CH * DK / NT];
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++) av[n] = 0.f, ak[n] = 0.f;
    const int BAND = 16;
    for (int j0 = 0; j0 < CH; j0 += BAND) {
      for (int i = tid; i < BAND * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + j0 + t < P)
                         ? qkv[(size_t)(t0 + j0 + t) * qkvstride + 4096 + h * DK + dd]
                         : 0.f;
      }
      __syncthreads();
      for (int jj = 0; jj < BAND; jj++) {
        int j = j0 + jj;
        float vjb = s_extra[jj * DK + d] * s_beta[j];
        float kjb = s_k[j][d] * s_beta[j] * s_eg[j];
        int r = tid >> 7;
#pragma unroll
        for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
          float a = s_attn[r][j];
          av[n] += a * vjb;
          ak[n] += a * kjb;
        }
      }
      __syncthreads();
    }
    int r = tid >> 7;
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
      wvp[r * DK + d] = av[n];
      wkcd[r * DK + d] = ak[n];
    }
  }
  __syncthreads();
  PH_ACC(3, t3);
  PH_DECL(t4);
  // attn2[r][j] = (q[r].k[j]) * exp(gcum[r]-gcum[j]) for j <= r (overwrite
  // s_attn); q staged in 16-row quarter tiles
  for (int q0 = 0; q0 < CH; q0 += 16) {
    for (int i = tid; i < 16 * DK; i += NT) {
      int t = i / DK, dd = i % DK;
      s_extra[i] = (t0 + q0 + t < P)
                       ? qkv[(size_t)(t0 + q0 + t) * qkvstride + kh * DK + dd]
                       : 0.f;
    }
    __syncthreads();
    for (int i = tid; i < 16 * CH; i += NT) {
      int r = q0 + i / CH, j = i % CH;
      float a = 0.f;
      if (j <= r && t0 + r < P) {
        const float* qr = &s_extra[(i / CH) * DK];
        float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        for (int dd = 0; dd < DK; dd += 4) {
          e0 += qr[dd] * s_k[j][dd];
          e1 += qr[dd + 1] * s_k[j][dd + 1];
          e2 += qr[dd + 2] * s_k[j][dd + 2];
          e3 += qr[dd + 3] * s_k[j][dd + 3];
        }
        float dot = (e0 + e1) + (e2 + e3);
        a = dot * expf(s_gcum[r] - s_gcum[j]);
      }
      s_attn[r][j] = a;
    }
    __syncthreads();
  }
  // stage attn2 for k_gdn_inter
  for (int i = tid; i < CH * CH; i += NT) wsb[GDN_WS_ATTN2 + i] = ((const float*)s_attn)[i];
  PH_ACC(4, t4);
  PH_ACC(9, tk);
}

// GDEC_GDN_PIPE2: fp16-resident k_gdn_intra with WMMA matrix phases and a
// 64-chunk loop per block, grid (ceil(nchunks/64), 48). Same ws layout and
// phase math as k_gdn_intra (ut5 solve verbatim), but k/v/q tiles live in
// LDS as fp16 and the attn/vp/kcd/attn2 dots run on WMMA. fp16 input
// rounding makes this NOT bit-exact (proto vs the fp32 kernel at P=8192:
// fro-rel 0.019%); wall 7.8 ms vs 21.9 ms for the fp32 reference
// (tools/gdn_pipe_proto.cu V13 @ ncpb=64).
#define GDN_PIPE2_NCPB 64

using gdn2_shortx16 = __attribute__((ext_vector_type(16))) short;
using gdn2_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ uint16_t gdn2_f2h(float f) {
  return __half_as_ushort(__float2half_rn(f));
}
__device__ __forceinline__ float gdn2_h2f(uint16_t u) {
  return __half2float(__ushort_as_half(u));
}
__device__ __forceinline__ gdn2_shortx16 gdn2_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  gdn2_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
// WMMA A-fragment row per lane (hardware lanes + conflict-free dummy rows)
__device__ __forceinline__ int gdn2_rowA(int lane) {
  const int b = lane >> 1;
  return (lane & 1) ? ((lane & 16) ? b : ((b + 4) & 7))
                    : ((lane & 16) ? 8 + ((b + 4) & 7) : b);
}

__global__ void __launch_bounds__(1024)
k_gdn_intra_p2(const float* __restrict__ qkv, const float* __restrict__ gb,
               const float* __restrict__ bb, float* __restrict__ ws, int P,
               int qkvstride) {
  const int CH = 64, DK = 128, NT = 1024;
  const int h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  const int nchunks_t = (P + CH - 1) / CH;
  for (int ic = 0; ic < GDN_PIPE2_NCPB; ic++) {
    const int c = blockIdx.x * GDN_PIPE2_NCPB + ic;
    if (c >= nchunks_t) break;
    __shared__ float s_attn[CH][CH];
    __shared__ float s_extra[3072];  // solve U/T tiles, then attn fp16 overlay
    __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH];
    __shared__ uint16_t s_k16[CH][136];  // 272B rows: conflict-free WMMA tiles
    __shared__ uint16_t s_bT[DK][72];    // v*beta -> k*beta*eg (transposed)
    float* wsb = ws + ((size_t)c * 48 + h) * GDN_SPLIT_WS_FLOATS;
    float* wvp = wsb + GDN_WS_VP;    // value'
    float* wkcd = wsb + GDN_WS_KCD;  // k_cumdecay
    const int t0 = c * CH;

    // staging: k16 + v*beta transposed in one round; gates; per-chunk cumsum
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      float kv = (t0 + t < P)
                     ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d]
                     : 0.f;
      s_k16[t][d] = gdn2_f2h(kv);
    }
    {
      const int d = tid & (DK - 1), jg = tid >> 7;
#pragma unroll
      for (int n = 0; n < 4; n++) {
        const int j = (jg + n * 8) * 2;
        float v0 = 0.f, v1 = 0.f, b0 = 0.f, b1 = 0.f;
        if (t0 + j < P) {
          v0 = qkv[(size_t)(t0 + j) * qkvstride + 4096 + h * DK + d];
          b0 = bb[(size_t)(t0 + j) * 48 + h];
        }
        if (t0 + j + 1 < P) {
          v1 = qkv[(size_t)(t0 + j + 1) * qkvstride + 4096 + h * DK + d];
          b1 = bb[(size_t)(t0 + j + 1) * 48 + h];
        }
        *(uint32_t*)&s_bT[d][j] =
            (uint32_t)gdn2_f2h(v0 * b0) | ((uint32_t)gdn2_f2h(v1 * b1) << 16);
      }
    }
    if (tid < CH) {
      s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
      s_beta[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
    }
    __syncthreads();
    if (tid == 0) {
      float acc = 0.f;
      for (int i = 0; i < CH; i++) {
        acc += s_gcum[i];
        s_gcum[i] = acc;
      }
    }
    __syncthreads();
    if (tid < CH) {
      s_eg[tid] = expf(s_gcum[tid]);
      wsb[tid] = s_gcum[tid];  // stage gcum for k_gdn_inter
    }

    // attn[r][j] = -beta[r] * (k[r].k[j]) * exp(gcum[r]-gcum[j]) for j < r
    {
      const int lane = tid & 31, w = tid >> 5;
      if (w < 16) {  // 16 warps own a 4x4 grid of 16x16 frags
        const int fr = (w >> 2) * 16, fc = (w & 3) * 16;
        const uint16_t* arow = &s_k16[fr + gdn2_rowA(lane)][0];
        const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
        gdn2_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < DK / 16; kk++) {
          gdn2_shortx16 a = gdn2_ld16(arow + kk * 16);
          gdn2_shortx16 b = gdn2_ld16(brow + kk * 16);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
        // C(r,c): lane c + 16*(r>=8), elem r%8
        const int col = fc + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = fr + (lane < 16 ? e : 8 + e);
          float a = 0.f;
          if (col < r)
            a = -s_beta[r] * acc[e] * expf(s_gcum[r] - s_gcum[col]);
          s_attn[r][col] = a;
        }
      }
    }
    __syncthreads();

    // ut5-style blocked inverse of (I - A), verbatim from k_gdn_intra.
    {
      const int g = tid >> 8, el = tid & 255;
      const int er = el >> 4, ec = el & 15;
      {
        const int col = er, mg = ec, base = g * 16;
        float solved = 0.f;
        for (int i = 1; i < 16; ++i) {
          float p = 0.f;
          if (mg < i) p = s_attn[base + i][base + mg] * solved;
          float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
          float value = s_attn[base + i][base + col];
#pragma unroll
          for (int gg = 0; gg < 16; gg += 2)
            value += __shfl_sync(~0ull, pair, gg, 16);
          if (i == mg && col < i) solved = value;
        }
        __syncthreads();  // all warps done reading A before it is overwritten
        if (col < mg) s_attn[base + mg][base + col] = solved;
      }
      __syncthreads();
      // stage A: T_10, T_21, T_32 (U = A_jb + A_jb . S_bb)
      if (g < 3) {
        const int j = g + 1, b = g;
        float a[16];
#pragma unroll
        for (int n = 0; n < 16; n++) a[n] = s_attn[j * 16 + er][b * 16 + n];
        float u = a[ec];
#pragma unroll
        for (int n = 0; n < 16; n++)
          u += a[n] * s_attn[b * 16 + n][b * 16 + ec];
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = s_attn[j * 16 + er][j * 16 + m];
        float t = s_extra[g * 256 + el];
#pragma unroll
        for (int m = 0; m < 16; m++)
          t += sj[m] * s_extra[g * 256 + m * 16 + ec];
        s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
      }
      __syncthreads();
      // stage B: T_20, T_31 (U = A_jb + A_jb . S_bb + A_j,b+1 . T_b+1,b)
      if (g < 2) {
        const int j = g + 2, b = g;
        float a0[16], a1[16];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          a0[n] = s_attn[j * 16 + er][b * 16 + n];
          a1[n] = s_attn[j * 16 + er][(b + 1) * 16 + n];
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * s_attn[b * 16 + n][b * 16 + ec];
          u += a1[n] * s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 + ec];
        }
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 2) {
        const int j = g + 2, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = s_attn[j * 16 + er][j * 16 + m];
        float t = s_extra[g * 256 + el];
#pragma unroll
        for (int m = 0; m < 16; m++)
          t += sj[m] * s_extra[g * 256 + m * 16 + ec];
        s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
      }
      __syncthreads();
      // stage C: T_30 (U = A_30 + A_30.S_00 + A_31.T_10 + A_32.T_20)
      if (g == 0) {
        float a0[16], a1[16], a2[16];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          a0[n] = s_attn[48 + er][n];
          a1[n] = s_attn[48 + er][16 + n];
          a2[n] = s_attn[48 + er][32 + n];
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * s_attn[n][ec];
          u += a1[n] * s_extra[1024 + n * 16 + ec];        // T_10
          u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];  // T_20
        }
        s_extra[el] = u;
      }
      __syncthreads();
      if (g == 0) {
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = s_attn[48 + er][48 + m];
        float t = s_extra[el];
#pragma unroll
        for (int m = 0; m < 16; m++)
          t += sj[m] * s_extra[m * 16 + ec];
        s_extra[1024 + 3 * 256 + el] = t;
      }
      __syncthreads();
      // publish the off-diagonal blocks into s_attn
      for (int j = 1; j < 4; j++)
        for (int b = 0; b < j; b++) {
          const int tl = j * (j - 1) / 2 + b;
          for (int e = tid; e < 256; e += NT)
            s_attn[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
                s_extra[1024 + tl * 256 + e];
        }
    }
    __syncthreads();
    if (tid < CH) s_attn[tid][tid] += 1.f;
    __syncthreads();

    // value' = attn @ (v*beta); k_cumdecay = attn @ (k*beta*exp(gcum)) (WMMA).
    // attn is rounded to fp16 (RNE) into an s_extra overlay as the A operand.
    {
      uint16_t(*s_a16)[72] = (uint16_t(*)[72])s_extra;
      for (int i = tid; i < CH * CH; i += NT)
        ((uint16_t*)s_a16)[(i / CH) * 72 + (i % CH)] =
            gdn2_f2h(s_attn[i / CH][i % CH]);
      __syncthreads();
      const int lane = tid & 31, w = tid >> 5;
      const int fr = (w >> 3) * 16, fc = (w & 7) * 16;  // 32 warps: 4x8 frags
      gdn2_shortx16 a[4];
#pragma unroll
      for (int kk = 0; kk < 4; kk++)
        a[kk] = gdn2_ld16(&s_a16[fr + gdn2_rowA(lane)][kk * 16]);
      {
        const uint16_t* bv = &s_bT[fc + (lane & 15)][0];
        gdn2_floatx8 accv = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          gdn2_shortx16 b16 = gdn2_ld16(bv + kk * 16);
          accv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[kk], b16, accv);
        }
        const int col = fc + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = fr + (lane < 16 ? e : 8 + e);
          wvp[r * DK + col] = accv[e];
        }
      }
      __syncthreads();  // vbT dead; restage as kbT from k16 (fold beta*eg)
      {
        const int d = tid & (DK - 1), jg = tid >> 7;
#pragma unroll
        for (int n = 0; n < 4; n++) {
          const int j = (jg + n * 8) * 2;
          float k0 = gdn2_h2f(s_k16[j][d]), k1 = gdn2_h2f(s_k16[j + 1][d]);
          uint32_t pk =
              (uint32_t)gdn2_f2h(k0 * s_beta[j] * s_eg[j]) |
              ((uint32_t)gdn2_f2h(k1 * s_beta[j + 1] * s_eg[j + 1]) << 16);
          *(uint32_t*)&s_bT[d][j] = pk;
        }
      }
      __syncthreads();
      {
        const uint16_t* bk = &s_bT[fc + (lane & 15)][0];
        gdn2_floatx8 acck = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          gdn2_shortx16 b16 = gdn2_ld16(bk + kk * 16);
          acck = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[kk], b16, acck);
        }
        const int col = fc + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = fr + (lane < 16 ? e : 8 + e);
          wkcd[r * DK + col] = acck[e];
        }
      }
      __syncthreads();  // bT dead; restage as q16 for attn2. stride 136 like
      // k16: rows are 128 halfs wide; a 72-stride layout would overlap row t
      // with row t+1 (two threads, same address -> race).
      for (int i = tid; i < CH * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        float qv = (t0 + t < P)
                       ? qkv[(size_t)(t0 + t) * qkvstride + kh * DK + dd]
                       : 0.f;
        ((uint16_t*)s_bT)[t * 136 + dd] = gdn2_f2h(qv);
      }
    }
    __syncthreads();

    // attn2[r][j] = (q[r].k[j]) * exp(gcum[r]-gcum[j]) for j <= r (WMMA)
    {
      const int lane = tid & 31, w = tid >> 5;
      const uint16_t(*q16)[136] = (const uint16_t(*)[136])s_bT;
      if (w < 16) {
        const int fr = (w >> 2) * 16, fc = (w & 3) * 16;
        const uint16_t* arow = &q16[fr + gdn2_rowA(lane)][0];
        const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
        gdn2_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int kk = 0; kk < DK / 16; kk++) {
          gdn2_shortx16 a = gdn2_ld16(arow + kk * 16);
          gdn2_shortx16 b = gdn2_ld16(brow + kk * 16);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
        const int col = fc + (lane & 15);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int r = fr + (lane < 16 ? e : 8 + e);
          float a = 0.f;
          if (col <= r && t0 + r < P)
            a = acc[e] * expf(s_gcum[r] - s_gcum[col]);
          s_attn[r][col] = a;
        }
      }
      __syncthreads();
      // stage attn2 for k_gdn_inter
      for (int i = tid; i < CH * CH; i += NT)
        wsb[GDN_WS_ATTN2 + i] = ((const float*)s_attn)[i];
    }
  }
}

#ifndef GDN_NT_INTER
#define GDN_NT_INTER GDN_NT  // k_gdn_inter block size (bit-exact at any NT)
#endif

template <int NT>
__global__ void k_gdn_inter(const float* __restrict__ qkv, float* __restrict__ S,
                            float* __restrict__ out, const float* __restrict__ ws,
                            int P, int qkvstride) {
  const int CH = 64, DK = 128;
  int h = blockIdx.x;
  int kh = h / 3;
  int tid = threadIdx.x;
  __shared__ float s_k[CH][DK + 4];  // row pad: 32-way bank conflicts -> 4-way
  __shared__ float s_attn[CH][CH];   // attn2 staged from ws
  __shared__ float s_extra[3072];    // S tile [128][16] + vnew [64][16]
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH];
  // s_beta holds expf(glast-gcum); glast = gcum[CH-1]
  S += (size_t)h * DK * DK;
  int nchunks = (P + CH - 1) / CH;
  for (int c = 0; c < nchunks; c++) {
    int t0 = c * CH;
    const float* wsb = ws + ((size_t)c * 48 + h) * GDN_SPLIT_WS_FLOATS;
    const float* wvp = wsb + GDN_WS_VP;    // value'
    const float* wkcd = wsb + GDN_WS_KCD;  // k_cumdecay
    // stage k rows (S update; pad rows: k=0), gcum, attn2 and the first S tile
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      s_k[t][d] =
          (t0 + t < P) ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d] : 0.f;
    }
    if (tid < CH) s_gcum[tid] = wsb[tid];
    for (int i = tid; i < CH * CH; i += NT)
      ((float*)s_attn)[i] = wsb[GDN_WS_ATTN2 + i];
    for (int i = tid; i < DK * 16; i += NT)
      s_extra[i] = S[(i / 16) * DK + (i % 16)];  // S tile d0=0
    __syncthreads();
    // fused d-tile pass over S (16 columns per tile): v_new, out, S update.
    if (tid < CH) {
      s_eg[tid] = expf(s_gcum[tid]);
      s_beta[tid] = expf(s_gcum[CH - 1] - s_gcum[tid]);  // expf(glast - gcum)
    }
    float egl = expf(s_gcum[CH - 1]);  // expf(glast)
    __syncthreads();
    float* s_St = s_extra;          // [DK][16]
    float* s_vn = s_extra + 2048;   // [CH][16]
    const int dc = tid & 15;
    const int RPT = CH * 16 / NT;   // S-phase rows per thread
    const int JPT = DK * 16 / NT;   // S-phase j-rows per thread
    const int RG = NT / 16;         // row-group stride
    for (int d0 = 0; d0 < DK; d0 += 16) {
      // prefetch the next S tile into registers; the loads overlap this
      // tile's compute (disjoint columns, no hazard with the S update below)
      float pf[JPT];
      if (d0 + 16 < DK) {
        int i = tid;
#pragma unroll
        for (int n = 0; n < JPT; n++, i += NT)
          pf[n] = S[(i / 16) * DK + d0 + 16 + (i % 16)];
      }
      // v_new[r][dc] = value'[r][d] - kcd[r] . S[:, d]  (j ascending)
      {
        // j-outer: s_St tile loads shared across the rows per thread
        int r = tid >> 4;
        const float* w0 = wkcd + r * DK;
        float e[RPT][4];
#pragma unroll
        for (int n = 0; n < RPT; n++)
#pragma unroll
          for (int u = 0; u < 4; u++) e[n][u] = 0.f;
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j * 16 + dc], s1 = s_St[(j + 1) * 16 + dc];
          float s2 = s_St[(j + 2) * 16 + dc], s3 = s_St[(j + 3) * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            float4 w = *(const float4*)(w0 + n * RG * DK + j);
            e[n][0] += w.x * s0;
            e[n][1] += w.y * s1;
            e[n][2] += w.z * s2;
            e[n][3] += w.w * s3;
          }
        }
#pragma unroll
        for (int n = 0; n < RPT; n++, r += RG)
          s_vn[r * 16 + dc] =
              wvp[r * DK + d0 + dc] - ((e[n][0] + e[n][1]) + (e[n][2] + e[n][3]));
      }
      __syncthreads();
      // out[r][d] = (q[r]*exp(gcum[r])) . S[:, d] + attn2[r] . v_new[:, d]
      {
        // j-outer with row accumulators: s_St / s_vn loads shared across rows
        int r = tid >> 4;
        const float* q0 = qkv + kh * DK;
        bool v[RPT];
        float eg[RPT], o[RPT];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          v[n] = (t0 + r + n * RG) < P;
          eg[n] = s_eg[r + n * RG];
          o[n] = 0.f;
        }
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j * 16 + dc], s1 = s_St[(j + 1) * 16 + dc];
          float s2 = s_St[(j + 2) * 16 + dc], s3 = s_St[(j + 3) * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            const float* qr = q0 + (size_t)(v[n] ? t0 + r + n * RG : t0) * qkvstride;
            float4 q4 = *(const float4*)(qr + j);
            o[n] += q4.x * s0 + q4.y * s1 + q4.z * s2 + q4.w * s3;
          }
        }
#pragma unroll
        for (int n = 0; n < RPT; n++) o[n] *= eg[n];
        for (int j = 0; j < CH; j++) {
          float vn = s_vn[j * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++)
            o[n] += ((const float*)s_attn)[(r + n * RG) * CH + j] * vn;
        }
#pragma unroll
        for (int n = 0; n < RPT; n++)
          if (v[n])
            out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] = o[n];
      }
      // S[j][d] = S[j][d]*exp(glast) + sum_r k[r][j]*exp(glast-gcum[r])*vnew[r][d]
      {
        // r-outer with j accumulators: s_vn / s_beta loads shared across j
        int j0 = tid >> 4;
        float a[JPT];
#pragma unroll
        for (int n = 0; n < JPT; n++) a[n] = 0.f;
        for (int r = 0; r < CH; r++) {
          float vn = s_vn[r * 16 + dc];
          float bt = s_beta[r];
          const float* kr = &s_k[r][0];
#pragma unroll
          for (int n = 0; n < JPT; n++) a[n] += kr[j0 + n * RG] * bt * vn;
        }
#pragma unroll
        for (int n = 0; n < JPT; n++) {
          int j = j0 + n * RG;
          S[j * DK + d0 + dc] = s_St[j * 16 + dc] * egl + a[n];
        }
      }
      __syncthreads();  // all reads of the current S tile are done
      // commit the prefetched tile over the dead one, then re-sync
      if (d0 + 16 < DK) {
        int i = tid;
#pragma unroll
        for (int n = 0; n < JPT; n++, i += NT) s_St[i] = pf[n];
      }
      __syncthreads();
    }
  }
}

// --- GDEC_GDN_FUSED: one block per head (grid 48, NT=1024), serial chunk
// loop; intra (fp16 tiles, WMMA attn, ut5 solve) + strip (v_new/out/S) fused
// with NO workspace DRAM round trip. S is register-resident as two persistent
// 16x16 WMMA C-fragments per warp (fp32 master); k.S/q.S run as WMMA against
// an fp16 transposed S copy staged quarter-wise in the ut5 scratch; W/vn live
// transposed in the v16 tile; the S update accumulates directly into the S
// fragments (A = transposed k*decay staged per r-quarter, B = vnT). proto:
// tools/gdn_fused_proto.cu (fro-rel out 0.040% / S 0.047% vs the fp32 split
// kernels, deterministic, 3.3x proto wall @ P=8192). Nonzero initial S
// supported; S written back at the end. GDEC_GDN_FUSED=1 selects it.
__global__ void __launch_bounds__(1024, 1)
k_gdn_fused(const float* __restrict__ qkv, const float* __restrict__ gb,
        const float* __restrict__ bb, float* __restrict__ Sg,
        float* __restrict__ out, int P, int qkvstride) {
  const int CH = 64, DK = 128, NT = 1024;
  const int h = blockIdx.x;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  __shared__ uint16_t s_v16[CH][144];  // v16, then aliased as vnT [128][72]
  __shared__ uint16_t s_k16[CH][136];
  __shared__ uint16_t s_attn16[CH][72];  // attn -> solve -> attn2, fp16
  __shared__ float s_extra[2816];
  __shared__ float s_gcum[CH], s_eg[CH], s_beta[CH], s_bb[CH];
  float* S = Sg + (size_t)h * DK * DK;
  const int lane = tid & 31, warp = tid >> 5;
  // S is register-resident as two persistent 16x16 WMMA C-fragments per
  // warp: tile t covers rows j in [m0+64t, +16), cols dc in [n0, +16).
  // Lane holds C(m,n) = S[m0+64t+(lane<16?e:8+e)][n0+lane&15], e = m%8.
  const int m0 = (warp >> 3) * 16, n0 = (warp & 7) * 16;
  const int dcS = n0 + (lane & 15);
  gdn2_floatx8 Sr0, Sr1;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    Sr0[e] = S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS];
    Sr1[e] = S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS];
  }
  const int nchunks = (P + CH - 1) / CH;
  for (int c = 0; c < nchunks; c++) {
    const int t0 = c * CH;
    // ---- P0: stage k16/v16 + gates + cumsum ----
    for (int i = tid; i < CH * DK / 4; i += NT) {
      const int t = i >> 5, d4 = (i & 31) << 2;
      float4 k4 = {0.f, 0.f, 0.f, 0.f}, v4 = {0.f, 0.f, 0.f, 0.f};
      if (t0 + t < P) {
        k4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 2048 +
                              kh * DK + d4);
        v4 = *(const float4*)(qkv + (size_t)(t0 + t) * qkvstride + 4096 +
                              h * DK + d4);
      }
      uint16_t kh4[4] = {gdn2_f2h(k4.x), gdn2_f2h(k4.y), gdn2_f2h(k4.z),
                         gdn2_f2h(k4.w)};
      uint16_t vh4[4] = {gdn2_f2h(v4.x), gdn2_f2h(v4.y), gdn2_f2h(v4.z),
                         gdn2_f2h(v4.w)};
      *(uint2*)&s_k16[t][d4] = *(const uint2*)kh4;
      *(uint2*)&s_v16[t][d4] = *(const uint2*)vh4;
    }
    if (tid < CH) {
      s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
      s_bb[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
    }
    __syncthreads();
    if (tid == 0) {
      float acc = 0.f;
      for (int i = 0; i < CH; i++) {
        acc += s_gcum[i];
        s_gcum[i] = acc;
      }
    }
    __syncthreads();
    const float glast = s_gcum[CH - 1];
    if (tid < CH) {
      s_eg[tid] = expf(s_gcum[tid]);
      s_beta[tid] = expf(glast - s_gcum[tid]);  // decay (strip semantics)
    }
    const float egl = expf(glast);
    // ---- P1: attn[r][j] = -bb[r]*(k[r].k[j])*exp(gcum[r]-gcum[j]), j<r ----
    if (warp < 16) {
      const int fr = (warp >> 2) * 16, fc = (warp & 3) * 16;
      const uint16_t* arow = &s_k16[fr + gdn2_rowA(lane)][0];
      const uint16_t* brow = &s_k16[fc + (lane & 15)][0];
      gdn2_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < DK / 16; kk++) {
        gdn2_shortx16 a = gdn2_ld16(arow + kk * 16);
        gdn2_shortx16 b = gdn2_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int col = fc + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = fr + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col < r) a = -s_bb[r] * acc[e] * expf(s_gcum[r] - s_gcum[col]);
        s_attn16[r][col] = gdn2_f2h(a);
      }
    }
    __syncthreads();
    // ---- P2: ut5 solve, verbatim from k_gdn_intra ----
    {
      const int g = tid >> 8, el = tid & 255;
      const int er = el >> 4, ec = el & 15;
      {
        const int col = er, mg = ec, base = g * 16;
        float solved = 0.f;
        for (int i = 1; i < 16; ++i) {
          float p = 0.f;
          if (mg < i) p = gdn2_h2f(s_attn16[base + i][base + mg]) * solved;
          float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
          float value = gdn2_h2f(s_attn16[base + i][base + col]);
#pragma unroll
          for (int gg = 0; gg < 16; gg += 2)
            value += __shfl_sync(~0ull, pair, gg, 16);
          if (i == mg && col < i) solved = value;
        }
        __syncthreads();
        if (col < mg) s_attn16[base + mg][base + col] = gdn2_f2h(solved);
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float a[16];
#pragma unroll
        for (int n = 0; n < 16; n++) a[n] = gdn2_h2f(s_attn16[j * 16 + er][b * 16 + n]);
        float u = a[ec];
#pragma unroll
        for (int n = 0; n < 16; n++)
          u += a[n] * gdn2_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 3) {
        const int j = g + 1, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = gdn2_h2f(s_attn16[j * 16 + er][j * 16 + m]);
        float t = s_extra[g * 256 + el];
#pragma unroll
        for (int m = 0; m < 16; m++)
          t += sj[m] * s_extra[g * 256 + m * 16 + ec];
        s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
      }
      __syncthreads();
      if (g < 2) {
        const int j = g + 2, b = g;
        float a0[16], a1[16];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          a0[n] = gdn2_h2f(s_attn16[j * 16 + er][b * 16 + n]);
          a1[n] = gdn2_h2f(s_attn16[j * 16 + er][(b + 1) * 16 + n]);
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * gdn2_h2f(s_attn16[b * 16 + n][b * 16 + ec]);
          u += a1[n] * s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 + ec];
        }
        s_extra[g * 256 + el] = u;
      }
      __syncthreads();
      if (g < 2) {
        const int j = g + 2, b = g;
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = gdn2_h2f(s_attn16[j * 16 + er][j * 16 + m]);
        float t = s_extra[g * 256 + el];
#pragma unroll
        for (int m = 0; m < 16; m++)
          t += sj[m] * s_extra[g * 256 + m * 16 + ec];
        s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
      }
      __syncthreads();
      if (g == 0) {
        float a0[16], a1[16], a2[16];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          a0[n] = gdn2_h2f(s_attn16[48 + er][n]);
          a1[n] = gdn2_h2f(s_attn16[48 + er][16 + n]);
          a2[n] = gdn2_h2f(s_attn16[48 + er][32 + n]);
        }
        float u = a0[ec];
#pragma unroll
        for (int n = 0; n < 16; n++) {
          u += a0[n] * gdn2_h2f(s_attn16[n][ec]);
          u += a1[n] * s_extra[1024 + n * 16 + ec];
          u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];
        }
        s_extra[el] = u;
      }
      __syncthreads();
      if (g == 0) {
        float sj[16];
#pragma unroll
        for (int m = 0; m < 16; m++) sj[m] = gdn2_h2f(s_attn16[48 + er][48 + m]);
        float t = s_extra[el];
#pragma unroll
        for (int m = 0; m < 16; m++)
          t += sj[m] * s_extra[m * 16 + ec];
        s_extra[1024 + 3 * 256 + el] = t;
      }
      __syncthreads();
      for (int j = 1; j < 4; j++)
        for (int b = 0; b < j; b++) {
          const int tl = j * (j - 1) / 2 + b;
          for (int e = tid; e < 256; e += NT)
            s_attn16[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
                gdn2_f2h(s_extra[1024 + tl * 256 + e]);
        }
    }
    __syncthreads();
    if (tid < CH) s_attn16[tid][tid] = gdn2_f2h(gdn2_h2f(s_attn16[tid][tid]) + 1.f);
    __syncthreads();
    // ---- P3a: ks = k16 @ S16T and qS = q @ S16T (WMMA) ----
    // fp32 S stays register-resident; an fp16 transposed copy S16T[dc][j]
    // is staged into s_extra one j-quarter (32) at a time. Each warp owns
    // one 16x16 C tile of ks (m=l) and one of qS (m=r), accumulated across
    // the four quarters with a shared B-frag. Warps with (warp&7)<4 also
    // accumulate attn2 (tile r0 x (warp&3)*16) reusing the same q A-frags
    // with k16 B-frags, same kk order as the standalone P4 it replaces.
    float ksc[8], qsc[8], vnc[8];
    const bool doA2 = (warp & 7) < 4;
    gdn2_floatx8 a2acc = {0, 0, 0, 0, 0, 0, 0, 0};
    {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const uint16_t* karow = &s_k16[r0 + gdn2_rowA(lane)][0];
      uint16_t* s16 = (uint16_t*)s_extra;  // [128 dc][32 j]
      const int j0 = (warp & 3) * 16;
      const uint16_t* kbrow = &s_k16[j0 + (lane & 15)][0];
      gdn2_floatx8 kacc = {0, 0, 0, 0, 0, 0, 0, 0};
      gdn2_floatx8 qacc = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int qt = 0; qt < 4; qt++) {
        __syncthreads();  // previous quarter's WMMA done with s_extra
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const int mt = m0 + t * 64;
          if ((mt >> 5) == qt) {
            uint16_t tmp[8];
            const gdn2_floatx8& Srt = t ? Sr1 : Sr0;
#pragma unroll
            for (int e = 0; e < 8; e++) tmp[e] = gdn2_f2h(Srt[e]);
            *(uint4*)&s16[dcS * 32 + (mt & 31) + (lane < 16 ? 0 : 8)] =
                *(const uint4*)tmp;
          }
        }
        __syncthreads();
        const uint16_t* brow = s16 + (c0 + (lane & 15)) * 32;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          gdn2_shortx16 bk = gdn2_ld16(brow + kk * 16);
          gdn2_shortx16 ak = gdn2_ld16(karow + (qt * 2 + kk) * 16);
          kacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ak, bk, kacc);
          const int rq = t0 + r0 + gdn2_rowA(lane);
          const float* qp = qkv + (size_t)(rq < P ? rq : t0) * qkvstride +
                            kh * DK + (qt * 2 + kk) * 16;
          short qt16[16];
#pragma unroll
          for (int q4 = 0; q4 < 4; q4++) {
            float4 x = *(const float4*)(qp + q4 * 4);
            qt16[q4 * 4 + 0] = (short)gdn2_f2h(x.x);
            qt16[q4 * 4 + 1] = (short)gdn2_f2h(x.y);
            qt16[q4 * 4 + 2] = (short)gdn2_f2h(x.z);
            qt16[q4 * 4 + 3] = (short)gdn2_f2h(x.w);
          }
          gdn2_shortx16 aq;
          memcpy(&aq, qt16, 32);
          qacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, bk, qacc);
          if (doA2) {
            gdn2_shortx16 bk2 = gdn2_ld16(kbrow + (qt * 2 + kk) * 16);
            a2acc =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aq, bk2, a2acc);
          }
        }
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        ksc[e] = kacc[e];
        qsc[e] = qacc[e];
      }
    }
    __syncthreads();  // S16T B-frag reads done; ksc/qsc complete
    // attn2 write goes straight after this barrier; the P3b/P3c barriers
    // below order it before S2 reads it, and S3's first staging barrier
    // orders it before s_extra is reused as ktd.
    if (doA2) {
      const int r0 = (warp >> 3) * 16;
      const int j0 = (warp & 3) * 16;
      uint16_t* a2w = (uint16_t*)s_extra;  // [64 r][72 j]
      const int col = j0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = r0 + (lane < 16 ? e : 8 + e);
        float a = 0.f;
        if (col <= r && t0 + r < P)
          a = a2acc[e] * expf(s_gcum[r] - s_gcum[col]);
        a2w[r * 72 + col] = gdn2_f2h(a);
      }
    }
    // ---- P3b: WT[dc][l] = bb_l*(v[l][dc] - eg_l*ks[l][dc]) over the v16 tile
    {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const int dc2 = c0 + (lane & 15);
      uint16_t vv[8];
#pragma unroll
      for (int e = 0; e < 8; e++)
        vv[e] = s_v16[r0 + (lane < 16 ? e : 8 + e)][dc2];
      __syncthreads();  // all v reads done before WT overwrites the tile
      uint16_t* WT = (uint16_t*)s_v16;  // [128 dc][72 l]
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int l = r0 + (lane < 16 ? e : 8 + e);
        WT[dc2 * 72 + l] =
            gdn2_f2h(s_bb[l] * (gdn2_h2f(vv[e]) - s_eg[l] * ksc[e]));
      }
    }
    __syncthreads();  // WT complete
    // ---- P3c: vnT[dc][r] = sum_l WT[dc][l] * attn16[r][l] (WMMA) ----
    {
      const uint16_t* WT = (const uint16_t*)s_v16;
      const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;  // C[dc][r]
      const uint16_t* arow = WT + (size_t)(m0 + gdn2_rowA(lane)) * 72;
      const uint16_t* brow = &s_attn16[n0 + (lane & 15)][0];
      gdn2_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < CH / 16; kk++) {
        gdn2_shortx16 a = gdn2_ld16(arow + kk * 16);
        gdn2_shortx16 b = gdn2_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) vnc[e] = acc[e];
    }
    __syncthreads();  // WT reads done before vnT overwrites the tile
    {
      uint16_t* vnT = (uint16_t*)s_v16;  // [128 dc][72 r]
      const int m0 = (warp >> 2) * 16, n0 = (warp & 3) * 16;
      const int rw = n0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int dcw = m0 + (lane < 16 ? e : 8 + e);
        vnT[dcw * 72 + rw] = gdn2_f2h(vnc[e]);
      }
    }
    __syncthreads();  // vnT ready; s_attn16 reusable
    // ---- S2: out[r][dc] = eg_r*qS[r][dc] + (attn2 @ vnT)[r][dc] ----
    // o2 via WMMA (32 warps, one 16x16 tile each; B-frags = vnT rows, contig
    // r); qS C-frags from P3a share the same tile layout -> single write.
    // attn2 A-frags come from the s_extra copy written by P3a.
    {
      const int r0 = (warp >> 3) * 16, c0 = (warp & 7) * 16;
      const uint16_t* vnT = (const uint16_t*)s_v16;
      const uint16_t* a2r = (const uint16_t*)s_extra;  // [64][72]
      const uint16_t* arow = a2r + (size_t)(r0 + gdn2_rowA(lane)) * 72;
      const uint16_t* brow = vnT + (size_t)(c0 + (lane & 15)) * 72;
      gdn2_floatx8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kk = 0; kk < CH / 16; kk++) {
        gdn2_shortx16 a = gdn2_ld16(arow + kk * 16);
        gdn2_shortx16 b = gdn2_ld16(brow + kk * 16);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      const int dc2 = c0 + (lane & 15);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int r = r0 + (lane < 16 ? e : 8 + e);
        if (t0 + r < P)
          out[(size_t)(t0 + r) * qkvstride + 4096 + h * DK + dc2] =
              s_eg[r] * qsc[e] + acc[e];
      }
    }
    __syncthreads();  // out complete; vnT still live for S3
    // ---- S3: S[j][dc] = S*egl + sum_r kTdecay[j][r] * vnT[dc][r] ----
    // WMMA accumulate straight into the persistent S C-fragments. A = kT
    // (transposed k16, decay folded in) staged one r-quarter (16) at a time
    // into s_extra; B = vnT rows (contiguous r).
    {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        Sr0[e] *= egl;
        Sr1[e] *= egl;
      }
      uint16_t* ktd = (uint16_t*)s_extra;  // two [128 j][16 r] buffers
      const uint16_t* vnT = (const uint16_t*)s_v16;
      // stage(qt+1) overlaps WMMA(qt) on the other buffer; one barrier per
      // quarter instead of two.
      for (int i = tid; i < DK * 16; i += NT) {
        const int j = i >> 4, r = i & 15;
        ktd[j * 16 + r] = gdn2_f2h(gdn2_h2f(s_k16[r][j]) * s_beta[r]);
      }
      __syncthreads();
      for (int qt = 0; qt < 4; qt++) {
        if (qt + 1 < 4) {
          uint16_t* nb = ktd + ((qt + 1) & 1) * DK * 16;
          for (int i = tid; i < DK * 16; i += NT) {
            const int j = i >> 4, r = i & 15;
            nb[j * 16 + r] = gdn2_f2h(gdn2_h2f(s_k16[(qt + 1) * 16 + r][j]) *
                                      s_beta[(qt + 1) * 16 + r]);
          }
        }
        const uint16_t* buf = ktd + (qt & 1) * DK * 16;
        const uint16_t* arow0 = buf + (m0 + gdn2_rowA(lane)) * 16;
        const uint16_t* arow1 = buf + (m0 + 64 + gdn2_rowA(lane)) * 16;
        gdn2_shortx16 b =
            gdn2_ld16(vnT + (size_t)(n0 + (lane & 15)) * 72 + qt * 16);
        Sr0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gdn2_ld16(arow0), b,
                                                         Sr0);
        Sr1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gdn2_ld16(arow1), b,
                                                         Sr1);
        if (qt + 1 < 4) __syncthreads();
      }
    }
    __syncthreads();  // k16/vn reads done before next chunk's staging
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    S[(size_t)(m0 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr0[e];
    S[(size_t)(m0 + 64 + (lane < 16 ? e : 8 + e)) * DK + dcS] = Sr1[e];
  }
}

// --- Phase 3b-strip: k_gdn_inter with the S v-dimension split into 8 strips.
// grid (8 strips, 48 heads); block (strip s, head h) owns S columns
// [16s, 16s+16) for the whole chunk scan. The strip's S tile [128][16] stays
// resident in LDS (loaded once, written back once; nonzero initial state
// supported), k/attn2 are re-read from global instead of staged, so a block
// needs ~13 KiB LDS and 4 blocks co-reside per CU (vs 1 for k_gdn_inter).
// Every output element keeps k_gdn_inter's accumulation order exactly
// (v_new/out: j ascending, 4-grouped (e0+e1)+(e2+e3); S update: r ascending,
// single thread per element; eg/beta/egl from the same ws gcum values), so
// out and S are bit-identical to k_gdn_inter and k_gdn_chunk. 3 barriers per
// chunk. Selected by default; GDEC_GDN_NOSTRIP=1 restores k_gdn_inter.
#define GDN_NT_STRIP 256

__global__ void __launch_bounds__(GDN_NT_STRIP)
    k_gdn_inter_strip(const float* __restrict__ qkv, float* __restrict__ S,
                      float* __restrict__ out, const float* __restrict__ ws, int P,
                      int qkvstride) {
  const int CH = 64, DK = 128, NT = GDN_NT_STRIP;
  const int h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  const int d0 = blockIdx.x * 32;  // strip base column (32-wide: 4 blocks/head)
  __shared__ float s_St[DK][32];   // resident S strip (16 KiB)
  __shared__ float s_vn[CH][32];   // v_new (8 KiB)
  __shared__ float s_eg[CH], s_beta[CH];
  S += (size_t)h * DK * DK;
  PH_DECL(tk);
  // load the strip's initial S (may be a nonzero state)
  for (int i = tid; i < DK * 32; i += NT)
    s_St[i / 32][i % 32] = S[(i / 32) * DK + d0 + (i % 32)];
  __syncthreads();
  PH_ACC(5, tk);
  const int nchunks = (P + CH - 1) / CH;
  const int dc = tid & 31;
  const int RG = NT / 32;        // row-group stride (8)
  const int RPT = CH * 32 / NT;  // rows per thread in the v_new/out phases (8)
  for (int c = 0; c < nchunks; c++) {
    const int t0 = c * CH;
    const float* wsb = ws + ((size_t)c * 48 + h) * GDN_SPLIT_WS_FLOATS;
    const float* wvp = wsb + GDN_WS_VP;    // value'
    const float* wkcd = wsb + GDN_WS_KCD;  // k_cumdecay
    const float* wattn2 = wsb + GDN_WS_ATTN2;
    // gates straight from ws (no staging); consumed after the v_new barrier
    if (tid < CH) {
      float gci = wsb[tid];
      s_eg[tid] = expf(gci);
      s_beta[tid] = expf(wsb[CH - 1] - gci);  // expf(glast - gcum)
    }
    const float egl = expf(wsb[CH - 1]);  // expf(glast)
    PH_DECL(t6);
    // v_new[r][dc] = value'[r][d0+dc] - kcd[r] . S[:, d0+dc]  (j ascending)
    {
      int r = tid >> 5;
      const float* w0 = wkcd + r * DK;
      float e[RPT][4];
#pragma unroll
      for (int n = 0; n < RPT; n++)
#pragma unroll
        for (int u = 0; u < 4; u++) e[n][u] = 0.f;
      for (int j = 0; j < DK; j += 4) {
        float s0 = s_St[j][dc], s1 = s_St[j + 1][dc];
        float s2 = s_St[j + 2][dc], s3 = s_St[j + 3][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          float4 w = *(const float4*)(w0 + n * RG * DK + j);
          e[n][0] += w.x * s0;
          e[n][1] += w.y * s1;
          e[n][2] += w.z * s2;
          e[n][3] += w.w * s3;
        }
      }
#pragma unroll
      for (int n = 0; n < RPT; n++, r += RG)
        s_vn[r][dc] =
            wvp[r * DK + d0 + dc] - ((e[n][0] + e[n][1]) + (e[n][2] + e[n][3]));
    }
    __syncthreads();  // s_vn / s_eg / s_beta ready
    PH_ACC(6, t6);
    PH_DECL(t7);
    // out[r][d0+dc] = (q[r]*exp(gcum[r])) . S[:, d0+dc] + attn2[r] . v_new[:, dc]
    {
      int r = tid >> 5;
      const float* q0 = qkv + kh * DK;
      bool v[RPT];
      float eg[RPT], o[RPT];
#pragma unroll
      for (int n = 0; n < RPT; n++) {
        v[n] = (t0 + r + n * RG) < P;
        eg[n] = s_eg[r + n * RG];
        o[n] = 0.f;
      }
      for (int j = 0; j < DK; j += 4) {
        float s0 = s_St[j][dc], s1 = s_St[j + 1][dc];
        float s2 = s_St[j + 2][dc], s3 = s_St[j + 3][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          const float* qr = q0 + (size_t)(v[n] ? t0 + r + n * RG : t0) * qkvstride;
          float4 q4 = *(const float4*)(qr + j);
          o[n] += q4.x * s0 + q4.y * s1 + q4.z * s2 + q4.w * s3;
        }
      }
#pragma unroll
      for (int n = 0; n < RPT; n++) o[n] *= eg[n];
      for (int j = 0; j < CH; j++) {
        float vn = s_vn[j][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++)
          o[n] += wattn2[(r + n * RG) * CH + j] * vn;
      }
#pragma unroll
      for (int n = 0; n < RPT; n++)
        if (v[n])
          out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] = o[n];
    }
    __syncthreads();  // all reads of the old S strip are done
    PH_ACC(7, t7);
    PH_DECL(t8);
    // S[j][d0+dc] = S*egl + sum_r k[r][j]*exp(glast-gcum[r])*vnew[r][dc]
    // (r ascending, single thread per element; lane holds 4 consecutive j so
    // the k reads coalesce into float4)
    {
      const int jg = tid >> 3;        // j = 4*jg .. 4*jg+3
      const int dc2 = (tid & 7) * 4;  // dc = dc2 .. dc2+3
      float a[4][4];
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
#pragma unroll
        for (int u = 0; u < 4; u++) a[jj][u] = 0.f;
      for (int r = 0; r < CH; r++) {
        float4 k4 = {0.f, 0.f, 0.f, 0.f};  // tail chunks: k zero-padded
        if (t0 + r < P)
          k4 = *(const float4*)(qkv + (size_t)(t0 + r) * qkvstride + 2048 +
                                kh * DK + jg * 4);
        const float bt = s_beta[r];
        const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const float vn = s_vn[r][dc2 + u];
#pragma unroll
          for (int jj = 0; jj < 4; jj++) a[jj][u] += kv[jj] * bt * vn;
        }
      }
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const int j = jg * 4 + jj;
#pragma unroll
        for (int u = 0; u < 4; u++)
          s_St[j][dc2 + u] = s_St[j][dc2 + u] * egl + a[jj][u];
      }
    }
    __syncthreads();  // S strip updated; s_vn reusable next chunk
    PH_ACC(8, t8);
  }
  // write back the strip's final state
  for (int i = tid; i < DK * 32; i += NT)
    S[(i / 32) * DK + d0 + (i % 32)] = s_St[i / 32][i % 32];
  PH_ACC(10, tk);
}

