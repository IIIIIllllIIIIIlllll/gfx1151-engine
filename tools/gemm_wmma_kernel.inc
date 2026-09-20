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
// the epilogue is predicated, so any P works. Hard requirements (host-
// checked, like the MoE kernels below): N % BP == 0, K % KST == 0 — no
// runtime guards in the unrolled loops (they defeat unrolling and spill the
// accumulators to scratch).
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

template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT>
__global__ void __launch_bounds__(NT)
    k_gemm_wmma(const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
                float* __restrict__ Y, int P, int K, int N, int gm) {
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
  const int num_m = (P + BM - 1) / BM, num_n = N / BP;
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
  // staging: thread idx -> row = idx/4, q = idx%4 (coalesced 64B row
  // segments). Global rows clamp to P-1 / N-1 for the tail block (one-time
  // setup cost, none in the K loop).
  const uint16_t* ga[AN];
  const uint16_t* gb[BN];
  uint32_t saoff[AN], sboff[BN];
#pragma unroll
  for (int j = 0; j < AN; ++j) {
    const int idx = tid + j * NT;
    const int row = idx / (KST / 8), q = idx % (KST / 8);
    ga[j] = X + (size_t)min(m0 + row, P - 1) * K + q * 8;
    saoff[j] = (row * LSA + q * 8) * 2;
  }
#pragma unroll
  for (int j = 0; j < BN; ++j) {
    const int idx = tid + j * NT;
    const int row = idx / (KST / 8), q = idx % (KST / 8);
    gb[j] = W + (size_t)min(n0 + row, N - 1) * K + q * 8;
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
