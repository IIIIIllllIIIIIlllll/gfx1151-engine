// Concurrent serve (QWENOX_PARALLEL > 1): thrown by kv_reserve / kv_cow when the
// shared KV pool cannot back this sequence's rows (the later request loses),
// and by the serve yield hook when another request aborted this one. Always
// raised before any device write of the step, so the slot can simply reset.
struct KvOverflow : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct GpuModel {
  const Checkpoint& ckpt;
  bool ple_on = true;
  int maxctx;
  int maxbatch;  // prefill chunk size: batch buffers are allocated at
                 // maxbatch_cap (maxbatch + tail-merge slack)
  int maxbatch_cap;
  int pos = 0;
  NgramMod ngram;
  // Adaptive ngram verify chunk rows, shared by both chain loops and kept
  // across requests (the content mix is a workload property, so warmup should
  // not repeat per request). 0 = the QWENOX_NGRAM_CHUNK env is still unparsed.
  int ng_cs = 0;
  bool ng_cs_fixed = false;
  bool ng_legacy = false;  // QWENOX_NGRAM_CHUNK=0: legacy single-pass verify
  void ng_chunk_init() {
    if (ng_cs) return;
    const char* e = getenv("QWENOX_NGRAM_CHUNK");
    const int v = e ? atoi(e) : -1;
    ng_legacy = v == 0;
    ng_cs_fixed = v > 0;
    ng_cs = ng_cs_fixed ? std::max(1, std::min(64, v)) : 16;
  }
  // Adaptive ngram gate (chain drafters): the ngram leg only pays on
  // repetitive content. Track the draft-token accept ratio with an EMA over
  // ngram rounds, close the leg for 32 rounds, then probe again. Like ng_cs
  // the state is a Model member shared by both chain loops and kept across
  // requests (the content mix is a workload property). QWENOX_NGRAM_GATE=0
  // pins the leg open.
  struct NgGate {
    double ema = 1.0;  // decayed draft-token accept ratio (starts optimistic)
    int cd = 0;        // rounds the leg stays closed (probes when it hits 0)
    int rounds = 0;
    static bool disabled() {
      static const int v = [] {
        const char* e = getenv("QWENOX_NGRAM_GATE");
        return e ? atoi(e) : 1;
      }();
      return v == 0;
    }
    bool open() {
      if (disabled()) return true;
      if (cd > 0) { --cd; return false; }
      return true;
    }
    // A probe round measures the content with a short capped draft; a good
    // result reopens the leg to full-length drafts on the next round.
    bool probing() const { return !disabled() && rounds >= 4 && ema < 0.7; }
    void update(int keep, int proposed) {
      ++rounds;
      ema += ((double)keep / std::max(1, proposed) - ema) * 0.25;
      if (rounds >= 4 && ema < 0.7 && cd == 0) cd = 32;
    }
  };
  NgGate ng_gate;

  // device state
  float *d_R, *d_Rhat, *d_x, *d_y, *d_emb;
  float *d_t320, *d_G, *d_w4;
  float *d_qkv, *d_g48, *d_beta48, *d_z;  // GDN scratch (z also used by QSA gate)
  float* d_S = nullptr;                   // (gdn_layers, 48, 128, 128)
  float* d_convst = nullptr;              // (gdn_layers, 3, 10240)
  float* d_convw = nullptr;               // (gdn_layers, 10240, 4) fp32
  float *d_Alog, *d_dtbias;               // (gdn_layers, 48)
  float* d_gdnnorm = nullptr;             // (gdn_layers, 128)
  float *d_kc = nullptr, *d_vc = nullptr;  // (qsa_layers, maxctx, 512), FP32 mode
  uint16_t *d_kcb = nullptr, *d_vcb = nullptr;  // BF16 KV cache (default mode)
  uint16_t *d_vcbt = nullptr;  // optional BF16 V cache: [block][kvh][dim][slot]
  float *d_qg, *d_qs, *d_gs, *d_kb, *d_vb, *d_attn;
  float *d_pacc = nullptr, *d_pml = nullptr;  // QSA decode flash-decoding partials
  float *d_iproj = nullptr, *d_iq = nullptr, *d_ik = nullptr;
  float *d_iraw = nullptr, *d_inorm = nullptr, *d_iscores = nullptr;
  float* d_iw = nullptr;
  // C1 f16 indexer: f16 twins of d_iq / d_ik / d_ikg（RNE 自同一份 fp32 值），
  // 打分走 k_index_scores_wmma16<false>，score 复用 d_iscores 内存按 f16 存，
  // select 走 ST=__half 变体。原型（logs/c1_f16_proto1.log）：重合率
  // 0.99886-0.99992 ≥ 0.995 门，x1.93-2.00 vs t64。QWENOX_INDEX_F16=0 退出；
  // 其余 index 旧开关（OLDSEL/SCORE64=0/SEL2P=0）任一出现即整路回 fp32。
  uint16_t *d_iqh = nullptr, *d_ikh = nullptr, *d_ikgh = nullptr;
  bool index_f16() const {
    return d_iqh != nullptr;
  }
  // Block ranking is sensitive to projection rounding; retain FP32 here.
  // aggr 10-05: bf16 indexer projection by default (the fp32 SGEMM ran at
  // ~2.4 TFLOPS: 264 ms per 16K chunk; bf16 gemm path 16K +2.6%, ids equal).
  // QWENOX_INDEX_F32=1 (or QWENOX_INDEX_BF16=0) restores the fp32 projection.
  bool index_f32 = getenv("QWENOX_INDEX_F32") != nullptr || env_zero("QWENOX_INDEX_BF16");
  bool index_fused2 = getenv("QWENOX_INDEX_FUSED2") != nullptr;
  bool index_stream_select = getenv("QWENOX_INDEX_STREAM_SELECT") != nullptr;
  // Fused-path fast kernels (09_kernels_index.inc), bit-exact, default on:
  // QWENOX_INDEX_SCORE64=0 -> old k_index_scores_tiled; QWENOX_INDEX_SEL2P=0 ->
  // old selector for n > 8192 (stream / rocPRIM per QWENOX_INDEX_STREAM_SELECT).
  bool index_score64 = !env_zero("QWENOX_INDEX_SCORE64");
  bool index_sel2p = !env_zero("QWENOX_INDEX_SEL2P");
  static bool env_zero(const char* k) {
    const char* v = getenv(k);
    return v && v[0] == '0' && v[1] == 0;
  }
  // B3 split-K 门控（d_skx 分配与 gemm() dispatch 共用）：k1 (2560,6144)
  // 在 P>=8192 时把 X 重排成 2 个连续 K=3072 切片跑两次 + slice1 原子加。
  // proto（tools/b3_splitk_proto.cu，logs/b3_splitk3.log）：16K 下
  // 14.5+0.5 ms vs 16.4 ms（x1.13）；收益来自 X 连续切片，strided 版 x0.84
  // 已否决。默认开，QWENOX_GEMM_SPLITK=0 退出。
  // B3 split-K: Windows +1.3% (10-03); Linux 16K −1.6% (aggr 10-05: unsplit
  // k1 16.35 ms vs split+repack ~18.8) -> Linux default off, QWENOX_GEMM_SPLITK=1 on.
  static bool gemm_splitk_on() {
#ifdef _WIN32
    static const bool want = !env_zero("QWENOX_GEMM_SPLITK");
#else
    static const bool want = getenv("QWENOX_GEMM_SPLITK") && !env_zero("QWENOX_GEMM_SPLITK");
#endif
    static const bool on = getenv("QWENOX_GEMM_WMMA") != nullptr &&
                           getenv("QWENOX_GEMM_NO_K64") == nullptr &&
                           getenv("QWENOX_GEMM_RB") == nullptr && want;
    return on;
  }
  int* d_iselected = nullptr;
  int index_blocks = 0;
  // Draft lm_head override (default on, QWENOX_DRAFT_LM_Q4=0 opts out): when an
  // overlay upgraded lm_head to a wider dtype, drafting reads the base file's
  // 4-bit copy instead (~0.5x bytes per draft step). Verify/output exactness
  // is unchanged -- only the proposal distribution shifts slightly.
  WView w_dlm{};
  bool dlm_on = false;
  // Paged QSA KV (memory model Phase A1). The trunk FP32 KV cache (d_kc/d_vc)
  // and indexer keys (d_ik) are addressed through a logical->physical page
  // table (KV_PAGE = 256 tokens = 64 indexer blocks; one table shared by all
  // QSA layers). QWENOX_KV_PAGED=1: identity table, =2: reversed table
  // (self-test: output must stay bitwise identical to unpaged). Unset/0: legacy
  // addressing (ptab == nullptr in every kernel). A1c: the BF16 cache
  // (d_kcb/d_vcb and the transposed d_vcbt) is paged too when the sparse
  // prefill kernel is k_qsa_wmma (no WMMA6/UNION). B0: the MTP layer's KV and
  // indexer keys are layer nqsa of the same arrays (kv_nl = nqsa + 1), so
  // they share the page table, the pool, checkpoint pins and COW; d_mkcf/
  // d_mvcf/d_mkcb/d_mvcb/d_mik are views of that slice (no transposed V).
  // A2: pages are allocated on demand from a pool (KvPagePool, 10_ple_io.inc)
  // as the live sequence grows (kv_reserve) and returned when it shrinks or
  // resets (kv_trim). QWENOX_KV_PAGED=1: lowest-free-first allocation, =2:
  // scrambled order + FIFO reuse (self-test). Physical page kv_pool.guard()
  // backs every unmapped logical page.
  // A3: RAM checkpoints (49_rckpt.inc) pin their prefix pages by refcount
  // (kv_pin / kv_unpin) and are restored by remapping (kv_adopt) instead of
  // requiring their tokens to be a prefix of the live stream. A write into a
  // page still shared with a checkpoint copies it first (kv_prepare -> kv_cow).
  // When the pool runs dry, kv_evict_fn drops the LRU checkpoint.
  bool kv_paged = false;
  int kv_lpages = 0;            // logical pages per sequence = ceil(maxctx / 256)
  int kv_pool_pages = 0;        // allocatable physical pages (kv_pool_pages_for)
  int kv_mode = 0;              // QWENOX_KV_PAGED value (2 = self-test logging)
  bool (*kv_evict_fn)(GpuModel&) = nullptr;  // A3: drop one pinned checkpoint
  // Concurrent serve: called once no checkpoint is left to evict. Frees pages
  // held by other sequence slots (true -> retry) or throws KvOverflow.
  bool (*kv_starve_fn)(GpuModel&) = nullptr;
  // Concurrent serve: GPU hand-off point between sequence slots (nullptr =
  // single sequence). Called only where no round-local state is live: spec
  // round tops, between prompt prefill chunks, between serial decode steps.
  void (*yield_fn)(GpuModel&) = nullptr;
  void yield_point() {
    if (yield_fn) yield_fn(*this);
  }
  // D1b mid-chunk GPU lending (serve mode, nslots > 1): while a long prompt
  // chunk is enqueued layer by layer, queued decode workers may borrow the
  // GPU for light rounds (chunks of at most conc_rmax_ rows) running on a
  // second, small copy of the P-row workspaces. All callbacks null -> inert.
  bool (*borrow_query_fn)() = nullptr;      // someone queued on the turnstile?
  void (*borrow_fn)(GpuModel&) = nullptr;   // release+reacquire under g_borrow_light
  void (*borrow_skip_fn)(GpuModel&) = nullptr;  // heavy chunk during borrow: requeue
  int conc_rmax_ = 32;                      // borrow round row cap (= small ws rows)
  float conc_quantum_ms_ = 300;             // min GPU-busy ms between lends
  int kv_nl = 0;                // layers in d_kc/d_vc/d_kcb/d_vcb/d_ik (+1 = MTP)
  size_t kv_layer_rows = 0;     // per-layer KV row stride (== maxctx unpaged)
  size_t ik_layer_blocks = 0;   // per-layer d_ik block stride (== index_blocks unpaged)
  int* d_ptab = nullptr;        // [kv_lpages] live logical page -> physical page
  KvPagePool kv_pool;
  KvSeqTable kv_seq;            // live sequence; kv_seq.tab is d_ptab's host mirror
  float* d_ikg = nullptr;       // [index_blocks][128] gathered keys (paged prefill)
  const int* kvpt() const { return kv_paged ? d_ptab : nullptr; }
  bool qsa_dense = getenv("QWENOX_QSA_DENSE") != nullptr;
  // BF16 KV cache: measured parity at 32K (QSA is latency-bound, not DRAM-bound)
  // and flips the 8192 oracle -> opt-in experiment, default stays FP32.
  bool qsa_kv_fp32 = getenv("QWENOX_QSA_KV_BF16") == nullptr;  // default: FP32 cache
  // WMMA sparse prefill attention (k_qsa_wmma): opt-in on top of BF16 KV mode.
  // bf16 P + WMMA accumulation is not bitwise fp32 — FP32 kernels stay default.
  bool qsa_wmma = getenv("QWENOX_QSA_WMMA") != nullptr;
  // Keep the established row-major cache for decode, MTP, snapshots, and
  // the default WMMA path; this second cache is a prefill-only opt-in.
  bool qsa_wmma_btv = qsa_wmma && !qsa_kv_fp32 &&
                      getenv("QWENOX_QSA_WMMA_BTV") != nullptr;
  // QWENOX_QSA_WMMA6=1: dim-split 128-thread rewrite (k_qsa_wmma6); layered on
  // the transposed-V path since it reads B fragments from that cache.
  bool qsa_wmma6 = qsa_wmma_btv && getenv("QWENOX_QSA_WMMA6") != nullptr;
  // QWENOX_QSA_UNION=1: 4-query 块并集 WMMA prefill（k_qsa_union_merge +
  // k_qsa_wmma_u），叠加在转置 V 路径上；组边缘回退 k_qsa_wmma<true>。
  // aggr 10-05: default ON (paged support + tail-edge sel-offset fix added;
  // 16K +0.9% paged, +2.2% unpaged; NOT bit-identical — key order — KLD
  // pending). QWENOX_QSA_UNION=0 restores the per-token kernel.
  bool qsa_union = qsa_wmma_btv && !env_zero("QWENOX_QSA_UNION");
  uint16_t* d_ublk = nullptr;  // [ngroups][QSA_UCAP] 并集块号
  uint16_t* d_umask = nullptr; // [ngroups][QSA_UCAP] 4 query x 4 slot 掩码
  int* d_ucount = nullptr;     // [ngroups] 并集块数（含 pad，4 的倍数）
  size_t qsa_vt_blocks = 0;
  // Per-QSA-layer base of the transposed V cache (nullptr unless BTV).
  uint16_t* vct_layer(int qi) const {
    return qsa_wmma_btv ? d_vcbt + (size_t)qi * qsa_vt_blocks * 2 * 256 * 4 : nullptr;
  }
  bool prefill_fused = getenv("QWENOX_PREFILL_UNFUSED") == nullptr;
  bool gr_bf16 = getenv("QWENOX_GR_BF16") != nullptr && prefill_fused;
  bool gr_scatter_norm = getenv("QWENOX_GR_SCATTER_NORM") != nullptr || gr_bf16;
  // Independent of QWENOX_QSA_DENSE: dense used to force atomic MoE (and so lose
  // MoE_LT), which made it 2x slower in MoE and not a clean attention-only A/B.
  bool moe_deterministic = getenv("QWENOX_MOE_ATOMIC") == nullptr;
  float* d_router;                        // 512
  int* d_topids;                          // 16
  float* d_topw;                          // 16
  float *d_guv, *d_hid, *d_ey, *d_moeacc, *d_sg;
  float *d_e, *d_keys, *d_v, *d_kn, *d_qn, *d_U, *d_ring, *d_convout;
  float* d_ple_convw;                     // 10240*4
  float *d_ple_nk, *d_ple_nq, *d_ple_nc;  // 10240 each
  float* d_logits;
  float* d_logits_b = nullptr;  // [65][vocab] speculative verify lm_head scratch
  int* d_argmax;
  float* d_hnorm_mixer;  // 10240

  // ---- prefill batch state (Phase 3a), sized maxctx ----
  rocblas_handle rbh = nullptr;
  hipblasLtHandle_t lth = nullptr;   // prefill GEMMs (Phase 3e)
  void* d_ltws = nullptr;            // hipBLASLt workspace
  size_t lt_ws_bytes = (size_t)64 << 20;
  // (N,K,P[,flags]) -> full LT descriptor set. Descriptors are shape-implied
  // and self-contained, so cache them for process lifetime instead of
  // create/destroy per call (~55 ms per 8-token chunk of pure host time).
  struct LtEntry {
    hipblasLtMatmulAlgo_t algo;
    hipblasLtMatmulDesc_t opd = nullptr;
    hipblasLtMatrixLayout_t Al = nullptr, Bl = nullptr, Cl = nullptr,
                          Dl = nullptr;
  };
  std::unordered_map<uint64_t, LtEntry> lt_algos;
  int* d_posarr;   // [maxctx] d_posarr[i] = i
  int* d_ringarr;  // [maxctx] d_ringarr[i] = i % 9
  int* d_tokarr;   // [maxctx] prefill token ids
  float2* d_ropecs = nullptr;  // [maxbatch][32] per-token rope cos/sin table
  int ropecs_base_ = -1, ropecs_P_ = -1;  // cache key of the last k_rope_cs fill
  unsigned ropecs_gen_ = 0;  // mrope state generation, part of the cache key
  // ---- M-RoPE (vision) request state ----
  // Empty mrope_pos_ = plain text request: rope is untouched (device-table
  // fill, decode delta 0) and bitwise identical to the pre-mrope engine.
  bool mrope_active_ = false;
  int rope_delta_ = 0;                       // host mirror of d_rdelta
  unsigned mrope_gen_ = 0;                   // bumps on set/clear
  std::vector<std::array<int, 3>> mrope_pos_;  // per-prompt-token [T,H,W]
  std::vector<std::array<int, 3>> mrope_grids_;  // request grids (kvsnap meta)
  std::vector<float2> h_ropecs_;             // host fill staging
  int* d_rdelta = nullptr;                   // device scalar: decode rope delta
  float2* d_cs1 = nullptr;                   // dump scratch (32 cos/sin)
  int* d_rpos1 = nullptr;                    // dump scratch (rope position)
  FILE* mrope_dump_ = nullptr;               // QWENOX_MROPE_DUMP sink
  bool mrope_dump_meta_ = false;
  // ---- Vision tower (stage 4) ----
  // vision_on only via load_vision (--vision-tower). Per-request state lives
  // and dies with the mrope request lifecycle (clear_mrope drops it): the
  // stitched bf16 embeds and the (src row, prompt pos) injection map.
  VisionTower vt;
  bool vision_on = false;
  uint16_t* d_img_emb = nullptr;   // [img_emb_rows, 2560] bf16
  size_t img_emb_rows = 0, img_emb_cap = 0;
  std::vector<int2> img_inj;       // per image token: (embed row, prompt pos)
  uint64_t img_hash_ = 0;          // FNV-1a of the request's patch bytes (cont guard)
  int2* d_img_inj = nullptr;       // [maxbatch] chunk-local upload scratch
  std::vector<int2> h_img_inj;     // host staging for the chunk slice
  float *d_Rb, *d_Rhatb, *d_xb, *d_yb, *d_Gb, *d_t320b, *d_w4b;
  // HC inject fusion (gr_write_read_b): double-buffered [BP][4][4] partials of
  // the site injp_for's w4, valid while injp_cur >= 0 (see inj_fuse_on()).
  float* d_injp[2] = {nullptr, nullptr};
  int injp_cur = -1;
  std::string injp_for;
  // B2-1: site whose HC write was already applied by the moe reduce
  // (k_*_hc); the next gr_write_read_b on it runs y=nullptr (norm only).
  std::string hcw_site_;
  std::unordered_map<std::string, uint16_t*> inj_wcache_;  // bf16 inject W per site
  float *d_qkvb, *d_zb, *d_g48b, *d_beta48b;
  float *d_qgb, *d_qsb, *d_gsb, *d_kbb, *d_vbb, *d_attnb;
  float* d_routerb;
  int* d_topidsb;
  float* d_topwb;
  int* d_slot_order = nullptr;  // verify MoE 槽序（k_moe_sort_slots，按专家排序）
  int moe_dump_base_ = -1;      // QWENOX_MOE_TOPIDS_DUMP：记录用的当前 chunk base
  float *d_guvb, *d_hidb, *d_moeaccb, *d_eyb, *d_sgb;
  float* d_eb;              // [maxctx, 2560] PLE embeddings
  float* ple_eb = nullptr;  // pinned staging [maxctx, 2560]
  uint8_t* ple_eb_dev = nullptr;  // device alias of ple_eb (zero-copy fp8 staging)
  uint16_t* d_wbf16;        // dequant workspace (64 MiB)
  uint16_t* d_xbf16;        // [maxctx, 10240] bf16 activations
  uint16_t* d_Rhatbf16 = nullptr;  // survives intermediate projection conversions
  float* d_hcsc = nullptr;  // B2-2: [BP,4] rms inv scales（Rhat 不落盘时由 GEMM 现场归一化）
  // Non-cacheable gemm inputs (d_hidb, d_t320b, attn/conv outs) convert here so
  // they don't evict the d_xb/d_Rhatb cache entries in d_xbf16.
  uint16_t* d_xbf16b = nullptr;  // [maxctx, 6144] bf16
  // A3 dense-LUT dispatch (§11.3): single-"expert" 64-slot tiles scratch for
  // the q4cp dense d3/d9 diversion in gemm() (tiny; (BP/64+2) tiles).
  MoeTile* d_dlut_tiles = nullptr;
  int* d_dlut_nt = nullptr;
  // B3 split-K: [BP, 6144] bf16，k1 (2560,6144) 的 X 重排成 2 个连续 K=3072
  // 切片。层边界即死（仅单次 gemm 内活跃），与 d_wbf16 同理性不入 WS 小拷贝。
  uint16_t* d_skx = nullptr;
  // QWENOX_MOE_LT buffers (allocated only when enabled): gathered bf16 pair rows
  // and per-group dequant scratch.
  __hip_bfloat16* d_moexg = nullptr;   // [BP*topk, d]
  __hip_bfloat16* d_moewup = nullptr;  // [MOE_LT_GRP(x2 if OVL), 2*moe_mid, d]
  __hip_bfloat16* d_moewdn = nullptr;  // [MOE_LT_GRP(x2 if OVL), d, moe_mid]
  uint16_t* d_hidb2 = nullptr;  // QWENOX_MOE_LT_CMP: fused-reference packed hid
  float* d_pairs2 = nullptr;    // QWENOX_MOE_LT_CMP: fused-reference weighted pairs
  int* d_cmpi = nullptr;        // QWENOX_MOE_LT_CMP: [max|diff|, max|ref|] ordered int
  // bf16 conversion cache: d_xb/d_Rhatb are converted once per layer section
  // instead of once per gemm; invalidated where those buffers are written
  const float* xf16_src = nullptr;
  uint64_t xf16_xs = 0;
  int xf16_P = 0, xf16_K = 0;
  // Expert-bucketed MoE with stable GPU routing and independently scheduled tiles.
  int* d_tokidx;       // [maxctx*topk] expert-sorted token indices
  float* d_pweight;    // [maxctx*topk] expert-sorted router weights
  int* d_eoff;         // [experts+1] expert pair offsets
  int* d_pairids = nullptr;  // [maxctx*topk] pair rows in ascending expert order
  GpuMoeRouting moe_routing;
  MoeTile* d_moetiles = nullptr;
  int* d_moentiles = nullptr;
  // GGUF routed experts (gg_on()): P == 1 scratch used by the decode graph;
  // batches reuse the prefill workspaces (see moe_gguf).
  __half* d_gg_x16 = nullptr;   // [d] F16 x
  __half* d_gg_hid = nullptr;   // [topk][moe_mid] F16 up*silu(gate)
  float* d_gg_pairs = nullptr;  // [topk][d] unweighted expert outputs
  float* d_gg_pw = nullptr;     // [topk]
  int *d_gg_tok = nullptr, *d_gg_pid = nullptr, *d_gg_nt = nullptr;
  MoeTile* d_gg_tiles = nullptr;  // [topk]
  int* d_gg_uq = nullptr;         // expert-dedup metadata (moegg::kUqInts), 2 <= P <= 8
  // GDN chunked prefill (Phase 3b)
  float* d_convb;    // [maxctx, 10240] post-conv qkv (conv is not in-place)
  float* d_gdn_ws;   // [48, 2, 64, 128] per-head chunk scratch
  float* d_gdn_split_ws = nullptr;  // [nchunks, 48] intra->inter staging (80.25 KiB each)
  // PLE batched prefill (Phase 3d-1)
  float* d_keysb;     // [maxctx, 10240] key_proj out, normed in place
  float* d_plevb;     // [maxctx, 2560] value_proj out
  float* d_qnb;       // [maxctx, 10240] normed R rows
  float* d_Ub;        // [maxctx, 10240] gated values (pre-norm)
  float* d_Unb;       // [maxctx, 10240] normed U (conv input / ring source)
  float* d_convoutb;  // [maxctx, 10240] conv outputs

  // PLE host state
  std::vector<int64_t> chist;
  int64_t ple_mult[3], ple_vsz[16], ple_voff[16];
  const Tensor* ple_table = nullptr;
  float ple_tscale = 1.f;
  bool ple_iq4 = false;  // table is the GGUF IQ4_NL view (dtype 11, g_ple_rb 90)
  int ple_ringpos = 0;           // host mirror for direct mode
  int* d_token = nullptr;        // device scalars for graph-replayable kernels
  int* d_pos = nullptr;
  int* d_ringpos = nullptr;
  int* p_token = nullptr;        // pinned staging for token id
  int* d_vbase = nullptr;        // P3-A: device mirror of the batch base position
  int* p_vbase = nullptr;        // pinned staging for d_vbase
  int pad_id = 248044;

  // PLE chunk-ahead prefetch: row addresses are pure functions of the last
  // two token ids, so the next chunk's disk gather runs on host threads while
  // this chunk's layers occupy the GPU (the gather otherwise leaves the GPU
  // idle behind embed+L0). One inflight flight, double pinned staging.
  float* ple_eb2 = nullptr;        // second pinned staging [maxbatch_cap, 2560]
  uint8_t* ple_eb_dev2 = nullptr;  // its device alias
  hipEvent_t ple_stg_ev[2] = {};   // recorded after each staging consume
  bool ple_stg_ev_ok[2] = {};
  int ple_par = 0;                 // staging buffer the next gather writes
  struct PleFlight {
    std::vector<std::thread> pool;
    std::vector<uint64_t> addrs, keys;
    std::vector<int> toks;       // the chunk's tokens (cat() window)
    std::atomic<int> barrier{0};
    std::atomic<int> workers{0};
    int64_t pre0 = 0, pre1 = 0;  // tokens before the window (pad_id at start)
    uint8_t* fp8 = nullptr;      // staging this flight fills
    int base = -1, P = 0, buf = 0;
    bool valid = false;
    std::chrono::steady_clock::time_point start, done;
  } ple_fl;

  // ---- MTP speculative draft head (Phase A) ----
  // Single QSA+MoE layer on top of the trunk's 48-layer stream; predicts
  // t_{p+2} from (trunk h_p, emb(t_{p+1})) (DeepSeek-style wiring). Owns its
  // stream/KV/indexer state; reuses trunk scratch (strictly serial use).
  bool mtp_avail = false;      // checkpoint has mtp.* tensors and env is default
  // mtp_avail's condition, known before the KV pool is sized (B0: the MTP
  // layer is pool layer nqsa).
  bool mtp_moe_default() const {
    return moe_deterministic && !moe_ordered && !moe_host_route && !moe_untiled &&
           !moe_fp32_io;
  }
  bool mtp_kv_wanted() const {
    return ckpt.find("mtp.fc_hidden.weight") && mtp_moe_default();
  }
  bool ngram_verify_active = false;
  bool ngram_legacy_gdn = getenv("QWENOX_NGRAM_LEGACY_GDN") != nullptr;
  bool ngram_legacy_final = getenv("QWENOX_NGRAM_LEGACY_FINAL") != nullptr;
  bool mtp_tap_capture = false;  // prefill_chunk copies d_Rb rows into d_mtap
  bool bench_quiet = false;      // offline bench: suppress per-layer/per-run diagnostics
  int mtp_norm_mode = 0;       // 0 = full-10240 RMSNorm (H1), 1 = 4x2560 (H2)
  int mtp_hagg_mode = 0;       // 0 = sum branches + repeat4, 1 = per-branch + bcast e
  int mtp_pos = 0;             // host mirror of the MTP stream position
  float* d_mR = nullptr;       // MTP hyper stream (10240)
  float* d_mRhat = nullptr;    // MTP normed stream (10240)
  float* d_mh = nullptr;       // input h-path accumulator (2560)
  float* d_me = nullptr;       // input e-path normed embed (2560)
  float* d_meo = nullptr;      // input e-path fc_embedding out (2560)
  float* d_mt = nullptr;       // input per-branch gemv tmp (2560)
  float* d_mnorm_e = nullptr;  // mtp.pre_fc_norm_embedding [2560] fp32
  float* d_mnorm_h = nullptr;  // mtp.pre_fc_norm_hidden [10240] fp32
  int* d_mtok = nullptr;       // device scalar: draft input token
  int* d_mpos = nullptr;       // device scalar: MTP position
  // MTP KV / indexer keys: views of pool layer nqsa (B0), not allocations
  float* d_mkcf = nullptr;     // FP32 mode (kv_layer_rows, 512)
  float* d_mvcf = nullptr;
  uint16_t* d_mkcb = nullptr;  // BF16 mode
  uint16_t* d_mvcb = nullptr;
  float* d_mik = nullptr;      // pooled keys (ik_layer_blocks, 128)
  float* d_miraw = nullptr;    // MTP indexer raw ring (4, 128)
  float* d_minorm = nullptr;   // MTP indexer norms q[128] + k[128]
  float* d_miw = nullptr;      // MTP indexer fp32 proj (640, 2560)
  float* d_mRb = nullptr;      // batch stream (maxbatch, 10240)
  float* d_mRhatb = nullptr;   // batch normed stream
  float* d_mtap = nullptr;     // captured trunk taps (maxbatch, 10240)
  // Offline test hook (--spec-gen + QWENOX_SPEC_FORCE=<ids file>): greedy
  // spec_loop verifies drafts against these ids instead of the trunk argmax,
  // so every γ policy commits the same trajectory (paired γ A/B without greedy
  // near-tie divergence; verify cost unchanged). Empty = off.
  std::vector<int> spec_force;
  // Stats for the most recent speculative request, emitted on the terminal
  // serve D line so clients can distinguish MTP from serial decode.
  int last_spec_rounds = 0;
  int last_spec_commit = 0;
  int last_spec_rollbacks = 0;
  int last_spec_proposed = 0;  // sum of drafts offered (D line trailing field)
  // Live progress reporting (2026-09-13): 1s-cadence prefill/decode stats on
  // stderr. Prefill is armed by serve around the prompt ingest only (never
  // for spec-verify batches) and uses NON-BLOCKING event probes — no stream
  // sync, so the pipeline is undisturbed. Decode reads the live spec
  // counters the loops bump every round.
  int prog_total = 0;         // prompt tokens to prefill this request (0=off)
  int64_t prog_base = 0;      // prompt tokens completed before current chunk
  int64_t prog_last_est = 0;  // est tokens at the last report (interval rate)
  int64_t prog_req = -1;      // request id stamped on progress lines
  std::chrono::steady_clock::time_point prog_t0, prog_last;
  hipEvent_t prog_ev[64] = {};  // per-layer completion probes (DisableTiming)
  int live_spec_rounds = 0;
  int live_spec_commits = 0;
  int live_spec_proposed = 0;
  // Trunk rollback checkpoint for verify: S + convst + PLE ring + trunk iraw,
  // plus a separate tiny slot for the MTP indexer ring (saved post-draft-1).
  float* d_ckpt = nullptr;
  size_t ckpt_S = 0, ckpt_convst = 0, ckpt_ring = 0, ckpt_iraw = 0;
  float* d_mring_ckpt = nullptr;  // 512 floats
  int* d_argmax_arr = nullptr;    // per-row verify argmax (<= 65 rows)
  unsigned long long* d_amx_cand = nullptr;  // two-stage argmax scratch
  // GPU sampling prep (spec sampling fast path): correction lists, radix
  // top-k scratch and per-row candidate blocks. Rows are capped at 65 (ngram
  // <= 64 + 1); the dense path stays as fallback (top_k unset/>64, candidate
  // overflow, QWENOX_SMS_GPU=0).
  static constexpr int SMS_ROWS = 65, SMS_TCAP = 512, SMS_PCAP = 64;
  static constexpr int SMS_BIAS_CAP = 20480, SMS_HIST_CAP = 120000;
  int* d_sm_ctrl = nullptr;        // [SMS_ROWS][8]
  unsigned* d_sm_hist = nullptr;   // [SMS_ROWS][4096]
  int* d_sm_cand_ids = nullptr;    // [SMS_ROWS][SMS_TCAP]
  float* d_sm_cand_vals = nullptr; // [SMS_ROWS][SMS_TCAP]
  int* d_sm_cand_n = nullptr;      // [SMS_ROWS]
  int* d_sm_bids = nullptr;        // bias list (shared per round)
  float* d_sm_bvals = nullptr;
  int* d_sm_hids = nullptr;        // penalty list from sampler hist (shared)
  float* d_sm_hvals = nullptr;
  int* d_sm_pids = nullptr;        // per-row draft-prefix penalties
  float* d_sm_pvals = nullptr;     // [SMS_ROWS][SMS_PCAP]
  int* d_sm_pn = nullptr;          // [SMS_ROWS]
  float* d_sm_scratch = nullptr;   // corrected-logits copy (lazy, corr only)
  int sm_nb = 0, sm_nh = 0;        // uploaded shared correction counts
  std::vector<int> h_sm_ids;       // D2H landing + host truncation staging
  std::vector<float> h_sm_vals;
  std::vector<int> h_sm_ci;        // shared-correction upload staging
  std::vector<float> h_sm_cv;
  std::vector<int> h_sm_pi;        // prefix upload staging (separate: the
  std::vector<float> h_sm_pv;      //   shared H2D may still be in flight)
  std::vector<int> sms_out_ids;    // truncated top-k results [rows][k]
  std::vector<float> sms_out_vals;
  // Rollback snapshots (spec verify): S/convst per accepted depth, indexer
  // proj rows per QSA layer. Written only when spec_snap is set.
  float* d_Ssnap = nullptr;    // [maxbatch+1][ngdn][48*128*128]
  float* d_csnap = nullptr;    // [maxbatch+1][ngdn][3*10240]
  float* d_iproj_snap = nullptr;  // [nqsa][maxbatch+1][128]
  bool spec_snap = false;         // set around spec verify chunks only
  // aggr 10-05 in-pass snapshot cut: the serve loop used to split the prompt
  // at the frontend snapshot point cc (prefill [0,cc), checkpoint, then a tiny
  // [cc,N) pass ~70 ms of extra TTFT). With cut_abs = cc the single pass
  // assembles the recurrent state at cc into shadow buffers instead (GDN S by
  // splitting the fused scan at the cut row, conv / PLE ring / indexer rings
  // by replaying their update with the first r rows, d_R = stream row cc-1,
  // MTP ring at cc-1); with_cut_state() swaps them in for rckpt / kvsnap.
  int cut_abs = -1;   // absolute cut position, -1 = off
  int cut_r_ = 0;     // chunk-local cut row this chunk (0 = none)
  bool cut_hit = false, cut_mtp_hit = false;
  float *d_cS = nullptr, *d_cconvst = nullptr, *d_cring = nullptr, *d_ciraw = nullptr,
        *d_cR = nullptr, *d_cmiraw = nullptr;
  bool cut_busy = false;  // NOT per slot: one shadow set, one owner at a time
  // Greedy draft chain without per-step D2H: step j reads step j-1's argmax
  // via D2D, one blocking D2H collects the chain. 0 restores per-step sync.
  const bool draft_desync =
      !getenv("QWENOX_DRAFT_DESYNC") || atoi(getenv("QWENOX_DRAFT_DESYNC")) != 0;
  // Draft-only vocab subset (first N rows of lm_head, FR-Spec style): greedy
  // draft argmax over the subset; verify is untouched, so committed output is
  // unchanged even when the true argmax lies outside. 0 = full vocab.
  // Greedy argmax path only; sampling drafts keep the full distribution.
  const int draft_vocab = [] {
    const char* e = getenv("QWENOX_DRAFT_VOCAB");
    return e ? atoi(e) : 0;
  }();
  // Dump top-2 verify logits for rows near absolute position N (draft
  // dependence forensics: tiny margin at a fork = fp tie-flip, large = leak).
  const int fork_dump = [] {
    const char* e = getenv("QWENOX_FORK_DUMP");
    return e ? atoi(e) : 0;
  }();

  explicit GpuModel(const Checkpoint& c, int mc) : ckpt(c), maxctx(mc) {
    if (qsa_union && maxctx > 262144) {  // block ids uint16: 65536 blocks max
      fprintf(stderr, "qsa: disabling QWENOX_QSA_UNION above 256K because its "
                      "block ids are uint16\n");
      qsa_union = false;
    }
    ensure_rope_tab(g_cfg.rope_theta, g_cfg.rotary_dim);
    CK(hipStreamSynchronize(nullptr));
    // Prefill workspace is sized by maxbatch, NOT maxctx: long prompts are
    // processed in maxbatch-token chunks via the same continuation path as
    // KV reuse, so a 256K context costs chunk-sized workspace (~4.5 GB at
    // 8192) instead of ~541 KB/token of context.
    maxbatch = eff_maxbatch(maxctx);
    maxbatch_cap = std::min(maxctx, maxbatch + eff_tail_slack());
    int rb = g_cfg.branches * g_cfg.d;
    DALLOC(&d_R, rb * 4);
    DALLOC(&d_Rhat, rb * 4);
    DALLOC(&d_x, rb * 4);
    DALLOC(&d_y, rb * 4);
    DALLOC(&d_emb, g_cfg.d * 4);
    DALLOC(&d_t320, g_cfg.gr_rank * 4);
    DALLOC(&d_G, rb * 4);
    DALLOC(&d_w4, 16);
    DALLOC(&d_qkv, 10240 * 4);
    DALLOC(&d_g48, 48 * 4);
    DALLOC(&d_beta48, 48 * 4);
    DALLOC(&d_z, 6144 * 4);
    int ngdn = g_cfg.layers - g_cfg.layers / 4;  // 36
    DALLOC(&d_S, (size_t)ngdn * 48 * 128 * 128 * 4);
    CK(hipMemset(d_S, 0, (size_t)ngdn * 48 * 128 * 128 * 4));
    DALLOC(&d_convst, (size_t)ngdn * 3 * 10240 * 4);
    CK(hipMemset(d_convst, 0, (size_t)ngdn * 3 * 10240 * 4));
    DALLOC(&d_convw, (size_t)ngdn * 10240 * 4 * 4);
    DALLOC(&d_Alog, (size_t)ngdn * 48 * 4);
    DALLOC(&d_dtbias, (size_t)ngdn * 48 * 4);
    DALLOC(&d_gdnnorm, (size_t)ngdn * 128 * 4);
    // ---- prefill batch buffers (Phase 3a), allocated once at maxctx ----
    // 提到最前（紧随权重 arena）：批缓冲的 gemm 速度随 arena 内绝对地址升高
    // 单调变差（139264 pads 1→12 GiB 实测稳态 1310→1240 tok/s、gdn:proj
    // 2257→2639 ms，无周期）；而 maxctx 尺寸的 KV/ik/vcbt/iscores 对位置
    // 不敏感（qsa:flash/idx 在两位形间不变）。固定低位摆放同时让布局与
    // maxctx 无关——此前 262144 仅因这些缓冲被 maxctx 增长顶高 5.21 GiB
    // 就掉 14%（1464.5→1253.8 tok/s @128k，E1 用 QWENOX_PAD_BATCH 复现）。
    // QWENOX_PAD_BATCH (bytes)：调试开关——哑分配整体推移批缓冲的 arena 偏移。
    {
      const char* pbs = getenv("QWENOX_PAD_BATCH");
      if (pbs && atoll(pbs) > 0) {
        void* pad_dummy = nullptr;
        DALLOC(&pad_dummy, (size_t)atoll(pbs));
      }
    }
    {
      size_t BP = (size_t)maxbatch_cap;
      DALLOC(&d_Rb, BP * 10240 * 4);
      DALLOC(&d_Rhatb, BP * 10240 * 4);
      DALLOC(&d_xb, BP * 2560 * 4);
      DALLOC(&d_yb, BP * 2560 * 4);
      DALLOC(&d_Gb, BP * 10240 * 4);
      DALLOC(&d_t320b, BP * 320 * 4);
      DALLOC(&d_w4b, BP * 4 * 4);
      DALLOC(&d_injp[0], BP * 16 * 4);
      DALLOC(&d_injp[1], BP * 16 * 4);
      DALLOC(&d_qkvb, BP * 10240 * 4);
      DALLOC(&d_zb, BP * 6144 * 4);
      DALLOC(&d_g48b, BP * 48 * 4);
      DALLOC(&d_beta48b, BP * 48 * 4);
      DALLOC(&d_qgb, BP * 12288 * 4);
      DALLOC(&d_qsb, BP * 6144 * 4);
      DALLOC(&d_gsb, BP * 6144 * 4);
      DALLOC(&d_kbb, BP * 512 * 4);
      DALLOC(&d_vbb, BP * 512 * 4);
      DALLOC(&d_ropecs, BP * 32 * sizeof(float2));
      DALLOC(&d_attnb, BP * 6144 * 4);
      DALLOC(&d_routerb, BP * 512 * 4);
      DALLOC(&d_topidsb, BP * 16 * 4);
      DALLOC(&d_topwb, BP * 16 * 4);
      DALLOC(&d_slot_order, 256 * 4);  // k_moe_sort_slots 输出（P*k ≤ 256）
      // Fused down reuses the otherwise idle gate/up workspace for pair outputs.
      // B1：v2+w4b pairs 全 f16（guvb_f16_env，与 30_host_util 估算同步）后按
      // f16 尺寸分配，省一半；任一回退 env 在位则保持 f32 全尺寸。
      DALLOC(&d_guvb,
             BP * g_cfg.topk * (moe_deterministic ? 2560 : 1280) *
                 (guvb_f16_env() ? 2 : 4));
      DALLOC(&d_hidb, BP * g_cfg.topk * 640 * 4);
      DALLOC(&d_moeaccb, BP * 2560 * 4);
      DALLOC(&d_eyb, BP * 2560 * 4);
      DALLOC(&d_sgb, BP * 4);
      DALLOC(&d_eb, BP * 2560 * 4);
      CK(hipHostMalloc((void**)&ple_eb, BP * 2560 * 4, hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ple_eb_dev, ple_eb, 0));
      CK(hipHostMalloc((void**)&ple_eb2, BP * 2560 * 4, hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ple_eb_dev2, ple_eb2, 0));
      CK(hipEventCreateWithFlags(&ple_stg_ev[0], hipEventDisableTiming));
      CK(hipEventCreateWithFlags(&ple_stg_ev[1], hipEventDisableTiming));
      DALLOC(&d_wbf16, (size_t)64 << 20);  // fits q_proj 12288x2560 bf16
      DALLOC(&d_xbf16, BP * 10240 * 2);
      DALLOC(&d_xbf16b, BP * 6144 * 2);  // largest non-cacheable K = 6144
      if (gemm_splitk_on()) DALLOC(&d_skx, BP * 6144 * 2);
      if (prefill_fused) DALLOC(&d_Rhatbf16, BP * 10240 * 2);
      if (prefill_fused) DALLOC(&d_hcsc, BP * 4 * 4);
      DALLOC(&d_tokidx, BP * g_cfg.topk * 4);
      DALLOC(&d_pweight, BP * g_cfg.topk * 4);
      DALLOC(&d_eoff, (g_cfg.experts + 1) * 4);
      if (moe_deterministic) DALLOC(&d_pairids, BP * g_cfg.topk * 4);
      if (!moe_host_route && !moe_ordered) moe_routing.init(BP * g_cfg.topk, g_cfg.experts);
      if (!moe_untiled && !moe_ordered) {
        DALLOC(&d_moetiles, ((BP * g_cfg.topk + 63) / 64 + g_cfg.experts) * sizeof(MoeTile));
        DALLOC(&d_moentiles, sizeof(int));
      }
      if (moe_lt && g_cfg.topk == 10 && moe_deterministic && !moe_host_route && !moe_ordered) {
        // 4-expert dequant groups: scratch stays MALL-resident between dequant
        // and its GEMMs (measured in tools/moe_lt_bench.cu). QWENOX_MOE_LT_OVL
        // doubles it for the A/B slot rotation of the overlap pipeline.
        DALLOC(&d_moexg, (size_t)BP * g_cfg.topk * g_cfg.d * 2);
        DALLOC(&d_moewup, (size_t)4 * 2 * g_cfg.moe_mid * g_cfg.d * 2 * (moe_lt_ovl ? 2 : 1));
        DALLOC(&d_moewdn, (size_t)4 * g_cfg.d * g_cfg.moe_mid * 2 * (moe_lt_ovl ? 2 : 1));
        if (moe_lt_cmp) {
          DALLOC(&d_hidb2, (size_t)BP * g_cfg.topk * g_cfg.moe_mid * 2);
          DALLOC(&d_pairs2, (size_t)BP * g_cfg.topk * g_cfg.d * 4);
          DALLOC(&d_cmpi, 8);
        }
      }
      DALLOC(&d_convb, BP * 10240 * 4);
      DALLOC(&d_gdn_ws, (size_t)48 * 2 * 64 * 128 * 4);
      if (!gdn_nosplit && !gdn_loop && !gdn_fused)  // fused needs no ws; streaming caps it at window chunks (15.05 MiB @4); pipeline double-buffers the slots
        DALLOC(&d_gdn_split_ws,
                     (size_t)(gdn_stream ? std::min<size_t>(gdn_window_chunks, (BP + 63) / 64)
                                         : (BP + 63) / 64) * 48 * GDN_SPLIT_WS_FLOATS * 4 *
                         (gdn_pipe ? 2 : 1));
      DALLOC(&d_keysb, BP * 10240 * 4);
      DALLOC(&d_plevb, BP * 2560 * 4);
      DALLOC(&d_qnb, BP * 10240 * 4);
      DALLOC(&d_Ub, BP * 10240 * 4);
      DALLOC(&d_Unb, BP * 10240 * 4);
      DALLOC(&d_convoutb, BP * 10240 * 4);
      // d_posarr/d_ringarr are indexed by ABSOLUTE position (qsa_loop debug,
      // ple_gpu_t): they stay maxctx-sized. d_tokarr only ever holds one
      // prefill chunk.
      DALLOC(&d_posarr, (size_t)maxctx * 4);
      DALLOC(&d_ringarr, (size_t)maxctx * 4);
      DALLOC(&d_tokarr, BP * 4);
      DALLOC(&d_img_inj, BP * sizeof(int2));  // vision injection map
      std::vector<int> h(maxctx);
      for (int i = 0; i < maxctx; i++) h[i] = i;
      CK(hipMemcpy(d_posarr, h.data(), (size_t)maxctx * 4, hipMemcpyHostToDevice));
      for (int i = 0; i < maxctx; i++) h[i] = i % 9;
      CK(hipMemcpy(d_ringarr, h.data(), (size_t)maxctx * 4, hipMemcpyHostToDevice));
      if (rocblas_create_handle(&rbh) != rocblas_status_success) {
        fprintf(stderr, "rocblas_create_handle failed\n");
        exit(1);
      }
      if (hipblasLtCreate(&lth) != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "hipblasLtCreate failed\n");
        exit(1);
      }
      DALLOC(&d_ltws, lt_ws_bytes);
      // dlut 小分配一律放 WS 末尾：插在中间会把其后所有缓冲推开（256 B 对齐后
      // 768 B），gdn/gr 工作区地址敏感（§13.8：P=2048 hc +20 µs/tok 的成因）。
      DALLOC(&d_dlut_tiles, (BP / 64 + 2) * sizeof(MoeTile));
      DALLOC(&d_dlut_nt, sizeof(int));
    }
    int nqsa = g_cfg.layers / 4;  // 12
    kv_nl = nqsa + (mtp_kv_wanted() ? 1 : 0);
    {
      const char* pe = getenv("QWENOX_KV_PAGED");
      int mode = pe ? atoi(pe) : 0;
      // A1c: the BF16 cache is paged on the k_qsa_flash (dense / decode) and
      // k_qsa_wmma<*, true> (sparse prefill, with or without BTV) kernels.
      // The union / wmma6 / k_qsa_flash_bf16 sparse kernels are not paged.
      const bool bf16_ok = qsa_dense || (qsa_wmma && !qsa_wmma6);  // union: paged since aggr
      if (mode && !qsa_kv_fp32 && !bf16_ok) {
        fprintf(stderr, "[kvpage] QWENOX_KV_PAGED ignored: with QWENOX_QSA_KV_BF16, "
                        "paging needs QWENOX_QSA_WMMA (with or without _BTV) and no "
                        "QWENOX_QSA_WMMA6 / QWENOX_QSA_UNION\n");
        mode = 0;
      }
      kv_paged = mode != 0;
      kv_mode = mode;
      kv_lpages = (maxctx + KV_PAGE - 1) / KV_PAGE;
      // A3: at least one full-length sequence; pinned checkpoints share the
      // pool (QWENOX_KV_POOL_TOKENS enlarges it); +1 physical guard page.
      kv_pool_pages = kv_pool_pages_for(maxctx);
      kv_layer_rows =
          kv_paged ? (size_t)(kv_pool_pages + 1) * KV_PAGE : (size_t)maxctx;
      if (kv_paged) {
        kv_pool.init(kv_pool_pages, mode == 2 ? 2 : 1);
        kv_seq.init(kv_lpages, kv_pool.guard());
        DALLOC(&d_ptab, (size_t)kv_lpages * 4);
        CK(hipMemcpy(d_ptab, kv_seq.tab.data(), (size_t)kv_lpages * 4,
                     hipMemcpyHostToDevice));
        fprintf(stderr,
                "[kvpage] paged QSA KV on: %d pages x %d tokens (+1 guard), "
                "on-demand alloc=%s, kv=%s\n",
                kv_pool_pages, KV_PAGE,
                mode == 2 ? "scrambled+fifo(self-test)" : "lowest-first",
                qsa_kv_fp32 ? "fp32" : (qsa_wmma_btv ? "bf16+btv" : "bf16"));
        if (kv_pool_pages > kv_lpages)
          fprintf(stderr, "[kvpage] pool enlarged: %d pages = %.2fx one sequence "
                          "(QWENOX_KV_POOL_TOKENS)\n",
                  kv_pool_pages, (double)kv_pool_pages / kv_lpages);
      }
    }
    if (!qsa_dense) {
      index_blocks = (maxctx + 3) / 4;
      ik_layer_blocks = kv_paged ? (size_t)(kv_pool_pages + 1) * KV_PAGE_BLOCKS
                                 : (size_t)index_blocks;
      // A1b: per-query indexer buffers are batch-local (row = abs - base), so
      // they are sized by one prefill chunk (maxbatch_cap), not by maxctx.
      // At 256K ctx / 16K chunk this is ~1.6 GB less. d_ik (per-block keys
      // of the whole history) stays maxctx-sized.
      DALLOC(&d_iproj, (size_t)maxbatch_cap * 640 * 4);
      DALLOC(&d_iq, (size_t)maxbatch_cap * 512 * 4);
      DALLOC(&d_ik, (size_t)kv_nl * ik_layer_blocks * 128 * 4);
      if (kv_paged) DALLOC(&d_ikg, (size_t)index_blocks * 128 * 4);
      // C1 f16 twins（门控与 30_host_util.inc 的 arena 估算同步）
      if (!env_zero("QWENOX_INDEX_F16") && index_score64 && index_sel2p &&
          getenv("QWENOX_INDEX_OLDSEL") == nullptr) {
        DALLOC(&d_iqh, (size_t)maxbatch_cap * 512 * 2);
        DALLOC(&d_ikh, (size_t)kv_nl * ik_layer_blocks * 128 * 2);
        if (kv_paged) DALLOC(&d_ikgh, (size_t)index_blocks * 128 * 2);
      }
      DALLOC(&d_iraw, (size_t)nqsa * 4 * 128 * 4);
      CK(hipMemset(d_iraw, 0, (size_t)nqsa * 4 * 128 * 4));
      DALLOC(&d_inorm, (size_t)nqsa * 256 * 4);
      if (index_f32) DALLOC(&d_iw, (size_t)nqsa * 640 * 2560 * 4);
      DALLOC(&d_iscores, (size_t)std::min(maxctx, INDEX_BATCH) * 4 * index_blocks * 4);
      DALLOC(&d_iselected, (size_t)maxbatch_cap * INDEX_BUDGET * 4);
    }
    if (qsa_kv_fp32) {
      DALLOC(&d_kc, (size_t)kv_nl * kv_layer_rows * 512 * 4);
      DALLOC(&d_vc, (size_t)kv_nl * kv_layer_rows * 512 * 4);
    } else {
      DALLOC(&d_kcb, (size_t)kv_nl * kv_layer_rows * 512 * 2);
      DALLOC(&d_vcb, (size_t)kv_nl * kv_layer_rows * 512 * 2);
      if (qsa_wmma_btv) {
        // paged: kv_layer_rows = (pool + 1) * 256 -> (pool + 1) * 64 blocks,
        // and physical page p owns physical blocks [p*64, p*64+64)
        qsa_vt_blocks = (kv_layer_rows + 3) / 4;
        DALLOC(&d_vcbt, (size_t)nqsa * qsa_vt_blocks * 2 * 256 * 4 * 2);
      }
      if (qsa_union) {
        // 并集工作区按单 chunk 的组数分配（maxbatch_cap/4 组，每组 ≤ 2052 块）
        const size_t ng = (size_t)maxbatch_cap / 4 + 1;
        DALLOC(&d_ublk, ng * QSA_UCAP * 2);
        DALLOC(&d_umask, ng * QSA_UCAP * 2);
        DALLOC(&d_ucount, ng * 4);
      }
    }
    if (kv_paged)  // A2: the guard page must read as finite data (zeros)
      kv_guard_regions([&](void* p, size_t n) { CK(hipMemset(p, 0, n)); });
    DALLOC(&d_qg, 12288 * 4);
    DALLOC(&d_qs, 24 * 256 * 4);
    DALLOC(&d_gs, 24 * 256 * 4);
    DALLOC(&d_kb, 512 * 4);
    DALLOC(&d_vb, 512 * 4);
    DALLOC(&d_attn, 24 * 256 * 4);
    DALLOC(&d_pacc, (size_t)2 * QSA_DEC_NSPLIT * 12 * 256 * 4);
    DALLOC(&d_pml, (size_t)2 * QSA_DEC_NSPLIT * 12 * 2 * 4);
    DALLOC(&d_router, 512 * 4);
    DALLOC(&d_topids, 16 * 4);
    DALLOC(&d_topw, 16 * 4);
    DALLOC(&d_guv, 16 * 1280 * 4);
    DALLOC(&d_hid, 16 * 640 * 4);
    DALLOC(&d_ey, g_cfg.d * 4);
    DALLOC(&d_moeacc, g_cfg.d * 4);
    if (gg_on()) {
      const int k = g_cfg.topk;
      if (k != 10 || !moe_deterministic || moe_host_route || moe_ordered || moe_untiled)
        throw std::runtime_error(
            "QWENOX_GGUF needs top-10 and the default MoE routing (no QWENOX_MOE_ATOMIC / "
            "HOST_ROUTE / ORDERED / UNTILED)");
      DALLOC(&d_gg_x16, (size_t)g_cfg.d * 2);
      DALLOC(&d_gg_hid, (size_t)k * g_cfg.moe_mid * 2);
      DALLOC(&d_gg_pairs, (size_t)k * g_cfg.d * 4);
      DALLOC(&d_gg_pw, k * 4);
      DALLOC(&d_gg_tok, k * 4);
      DALLOC(&d_gg_pid, k * 4);
      DALLOC(&d_gg_nt, 4);
      DALLOC(&d_gg_tiles, k * sizeof(MoeTile));
      DALLOC(&d_gg_uq, moegg::kUqInts * 4);
    }
    DALLOC(&d_sg, 4);
    DALLOC(&d_e, 2560 * 4);
    CK(hipHostMalloc((void**)&ple_e, 2560 * 4, hipHostMallocDefault));
    DALLOC(&d_token, 4);
    DALLOC(&d_pos, 4);
    DALLOC(&d_ringpos, 4);
    DALLOC(&d_rdelta, 4);
    DALLOC(&d_vbase, 4);
    CK(hipMemset(d_pos, 0, 4));
    CK(hipMemset(d_ringpos, 0, 4));
    CK(hipMemset(d_rdelta, 0, 4));
    if (getenv("QWENOX_MROPE_DUMP")) {
      DALLOC(&d_cs1, 32 * sizeof(float2));
      DALLOC(&d_rpos1, 4);
    }
    CK(hipHostMalloc((void**)&p_token, 4, hipHostMallocDefault));
    CK(hipHostMalloc((void**)&p_vbase, 4 * kVbRing, hipHostMallocDefault));
    DALLOC(&d_keys, 10240 * 4);
    DALLOC(&d_v, 2560 * 4);
    DALLOC(&d_kn, 10240 * 4);
    DALLOC(&d_qn, 10240 * 4);
    DALLOC(&d_U, 10240 * 4);
    DALLOC(&d_ring, (size_t)9 * 10240 * 4);
    CK(hipMemset(d_ring, 0, (size_t)9 * 10240 * 4));
    DALLOC(&d_convout, 10240 * 4);
    DALLOC(&d_ple_convw, 10240 * 4 * 4);
    DALLOC(&d_ple_nk, 10240 * 3 * 4);
    DALLOC(&d_logits, (size_t)g_cfg.vocab * 4);
    // Ngram verifies up to 64 drafts plus the current token; MTP needs nine
    // rows at most. Keep logits independent of the much larger prefill chunk.
    DALLOC(&d_logits_b, (size_t)g_cfg.vocab * 4 * 65);
    DALLOC(&d_argmax_arr, 65 * sizeof(int));
    DALLOC(&d_amx_cand, 65 * ARGMAX_B1 * sizeof(unsigned long long));
    DALLOC(&d_sm_ctrl, SMS_ROWS * 8 * sizeof(int));
    DALLOC(&d_sm_hist, (size_t)SMS_ROWS * 4096 * sizeof(unsigned));
    DALLOC(&d_sm_cand_ids, (size_t)SMS_ROWS * SMS_TCAP * sizeof(int));
    DALLOC(&d_sm_cand_vals, (size_t)SMS_ROWS * SMS_TCAP * sizeof(float));
    DALLOC(&d_sm_cand_n, SMS_ROWS * sizeof(int));
    DALLOC(&d_sm_bids, SMS_BIAS_CAP * sizeof(int));
    DALLOC(&d_sm_bvals, SMS_BIAS_CAP * sizeof(float));
    DALLOC(&d_sm_hids, SMS_HIST_CAP * sizeof(int));
    DALLOC(&d_sm_hvals, SMS_HIST_CAP * sizeof(float));
    DALLOC(&d_sm_pids, (size_t)SMS_ROWS * SMS_PCAP * sizeof(int));
    DALLOC(&d_sm_pvals, (size_t)SMS_ROWS * SMS_PCAP * sizeof(float));
    DALLOC(&d_sm_pn, SMS_ROWS * sizeof(int));
    h_sm_ids.resize((size_t)SMS_ROWS * SMS_TCAP);
    h_sm_vals.resize((size_t)SMS_ROWS * SMS_TCAP);
    // Trunk checkpoint belongs to the verifier, independently of MTP weights.
    {
      int ngdn = g_cfg.layers - g_cfg.layers / 4;
      ckpt_S = (size_t)ngdn * 48 * 128 * 128 * 4;
      ckpt_convst = (size_t)ngdn * 3 * 10240 * 4;
      ckpt_ring = (size_t)9 * 10240 * 4;
      ckpt_iraw = qsa_dense ? 0 : (size_t)(g_cfg.layers / 4) * 4 * 128 * 4;
      DALLOC(&d_ckpt, ckpt_S + ckpt_convst + ckpt_ring + ckpt_iraw);
    }
    DALLOC(&d_argmax, 4);
    DALLOC(&d_hnorm_mixer, 10240 * 4);

    // ---- upload small fp32 weights (norms, conv, A_log, dt_bias) ----
    char buf[128];
    std::vector<float> tmp;
    auto upload = [&](const char* name, float* dst, size_t n) {
      const Tensor& t = ckpt.at(name);
      if (t.dtype == 16 || t.dtype == 24) {  // hgn v2 (fp32 indexer projection): decode on GPU
        uint16_t* bf;
        CK(hipMalloc(&bf, n * 2));
        v2_deq_bf16(wview(t), bf, g_str);
        k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(bf, dst, n);
        CK(hipStreamSynchronize(g_str));
        CK(hipFree(bf));
        return;
      }
      tmp.resize(n);
      ckpt.dequant(t, tmp.data());
      CK(hipMemcpy(dst, tmp.data(), n * 4, hipMemcpyHostToDevice));
    };
    int gi = 0;
    for (int l = 0; l < g_cfg.layers; l++) {
      if (is_qsa(l)) {
        if (!qsa_dense) {
          snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.q_layernorm.weight", l);
          upload(buf, d_inorm + (l / 4) * 256, 128);
          snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.k_layernorm.weight", l);
          upload(buf, d_inorm + (l / 4) * 256 + 128, 128);
          if (index_f32) {
            snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
            upload(buf, d_iw + (size_t)(l / 4) * 640 * 2560, 640 * 2560);
          }
        }
        continue;
      }
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.conv1d.weight", l);
      upload(buf, d_convw + (size_t)gi * 10240 * 4, 10240 * 4);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.A_log", l);
      upload(buf, d_Alog + (size_t)gi * 48, 48);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.dt_bias", l);
      upload(buf, d_dtbias + (size_t)gi * 48, 48);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.norm.weight", l);
      upload(buf, d_gdnnorm + (size_t)gi * 128, 128);
      gi++;
    }
    upload("hyper_connection_mixer.hc_norm.weight", d_hnorm_mixer, 10240);
    upload("layers.1.ple.conv1d.weight", d_ple_convw, 10240 * 4);
    upload("layers.1.ple.norm_key.weight", d_ple_nk, 10240);
    upload("layers.1.ple.norm_query.weight", d_ple_nk + 10240, 10240);
    upload("layers.1.ple.norm_conv.weight", d_ple_nk + 20480, 10240);

    // PLE config tensors
    {
      const Tensor& mt = ckpt.at("layers.1.ple.ple_embedding.layer_multipliers");
      memcpy(ple_mult, mt.data, 24);
      const Tensor& vs = ckpt.at("layers.1.ple.ple_embedding.ngram_heads_vocab_sizes");
      memcpy(ple_vsz, vs.data, 128);
      const Tensor& vo = ckpt.at("layers.1.ple.ple_embedding.ngram_heads_offsets");
      memcpy(ple_voff, vo.data, 128);
      ple_table = ckpt.find("layers.1.ple.ngram_embedding.weight");
      if (!ple_table)
        throw std::runtime_error(
            "PLE n-gram table (layers.1.ple.ngram_embedding.weight) not found: hgn v2 keeps it "
            "in a separate file — set NGRAM_FILE (qwenox: pass that .hgn as an overlay)");
      ple_iq4 = ple_table->dtype == 11;
      if (ple_iq4) {
        // GGUF IQ4_NL table: 90 B rows, no global scale. The batch-read fd is
        // re-pointed at the shard that holds it (mmapped from offset 0).
        g_ple_rb = 90;
#ifndef _WIN32
        if (g_gguf)
          for (const auto& mp : g_gguf->mappings())
            if (ple_table->data >= mp.base && ple_table->data < mp.base + mp.len) {
              ple_uring_open(mp.path.c_str());
              g_ple_foff = (uint64_t)(ple_table->data - mp.base);
              g_ple_off_ok = g_ple_fd >= 0;
            }
#endif
      } else {
      memcpy(&ple_tscale, ple_table->data + ple_table->numel(), 4);
      // io_uring / IOCP path: file offset of the PLE table in the file that
      // holds it. open_base opened the base .hgn; a separate n-gram file
      // (NGRAM_FILE, passed as an overlay — hgn v2 ships the table on its
      // own) re-points the reader at that file. Every mapping maps its file
      // from offset 0.
      if (const auto* mp = ckpt.mapping_of(ple_table->data)) {
        if (!ckpt.mappings().empty() && mp != &ckpt.mappings()[0]) {
          ple_uring_open(mp->path.c_str());
          fprintf(stderr, "ple: n-gram table from %s\n", mp->path.c_str());
        }
#ifdef _WIN32
        if (g_ple_win.fh != INVALID_HANDLE_VALUE) {
#else
        if (g_ple_fd >= 0) {
#endif
          g_ple_foff = (uint64_t)(ple_table->data - mp->base);
          g_ple_off_ok = true;
        }
      }
      }
    }

    // ---- MTP draft head: detect weights, allocate own state ----
    if (ckpt.find("mtp.fc_hidden.weight")) {
      // moe_lt is compatible: it only activates at P >= moe_lt_min (4096) in
      // the trunk batch MoE, while MTP draft/ingest/verify run their own
      // fixed kernels at P <= gamma+1 <= 9. gr_bf16 taps are converted
      // bf16->f32 at capture/rollback/ingest (see spec_taps/restore_dR_row).
      bool moe_default = mtp_moe_default();
      if (!moe_default) {
        fprintf(stderr, "mtp: disabled (moe_default=%d)\n", (int)moe_default);
      } else {
        // Validated wiring (mtp-test 0.855): branch-grouped norm + per-branch
        // fc_hidden + broadcast e. Env flips back for debugging only.
        const char* nm = getenv("QWENOX_MTP_NORM");
        mtp_norm_mode = (nm && !strcmp(nm, "full")) ? 0 : 1;
        const char* hg = getenv("QWENOX_MTP_HAGG");
        mtp_hagg_mode = (hg && !strcmp(hg, "sum")) ? 0 : 1;
        DALLOC(&d_mR, rb * 4);
        DALLOC(&d_mRhat, rb * 4);
        DALLOC(&d_mh, g_cfg.d * 4);
        DALLOC(&d_me, g_cfg.d * 4);
        DALLOC(&d_meo, g_cfg.d * 4);
        DALLOC(&d_mt, g_cfg.d * 4);
        DALLOC(&d_mtok, 4);
        DALLOC(&d_mpos, 4);
        DALLOC(&d_mnorm_e, g_cfg.d * 4);
        DALLOC(&d_mnorm_h, rb * 4);
        upload("mtp.pre_fc_norm_embedding.weight", d_mnorm_e, g_cfg.d);
        upload("mtp.pre_fc_norm_hidden.weight", d_mnorm_h, rb);
        // B0: pool layer nqsa (allocated above since kv_nl == nqsa + 1)
        const size_t moff = (size_t)nqsa * kv_layer_rows * 512;
        if (qsa_kv_fp32) {
          d_mkcf = d_kc + moff;
          d_mvcf = d_vc + moff;
        } else {
          d_mkcb = d_kcb + moff;
          d_mvcb = d_vcb + moff;
        }
        DALLOC(&d_mRb, (size_t)maxbatch_cap * 10240 * 4);
        DALLOC(&d_mRhatb, (size_t)maxbatch_cap * 10240 * 4);
        DALLOC(&d_mtap, (size_t)maxbatch_cap * 10240 * 4);
        {
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          DALLOC(&d_mring_ckpt, 512 * 4);
          // Rollback snapshots: per-row GDN S / convst / indexer-proj state
          // captured during a spec verify chunk, indexed by accepted depth.
          // Spec chunks are gamma+1 <= 9 rows (gamma <= 8), not maxbatch.
          DALLOC(&d_Ssnap, (size_t)9 * ngdn * 48 * 128 * 128 * 4);
          DALLOC(&d_csnap, (size_t)9 * ngdn * 3 * 10240 * 4);
          DALLOC(&d_iproj_snap, (size_t)(g_cfg.layers / 4) * 9 * 128 * 4);
        }
        if (!qsa_dense) {
          d_mik = d_ik + (size_t)nqsa * ik_layer_blocks * 128;
          DALLOC(&d_miraw, 4 * 128 * 4);
          CK(hipMemset(d_miraw, 0, 4 * 128 * 4));
          DALLOC(&d_minorm, 256 * 4);
          upload("mtp.layers.0.self_attn.indexer.q_layernorm.weight", d_minorm, 128);
          upload("mtp.layers.0.self_attn.indexer.k_layernorm.weight", d_minorm + 128,
                 128);
          if (index_f32) {
            DALLOC(&d_miw, (size_t)640 * 2560 * 4);
            upload("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", d_miw,
                   (size_t)640 * 2560);
          }
        }
        mtp_avail = true;
        fprintf(stderr,
                "mtp: draft head enabled (norm=%s hagg=%s kv=%s fc_hidden dtype=%d)\n",
                mtp_norm_mode ? "branch" : "full",
                mtp_hagg_mode ? "branch" : "sum",
                qsa_kv_fp32 ? "fp32" : "bf16",
                (int)ckpt.at("mtp.fc_hidden.weight").dtype);
      }
    }
    fprintf(stderr, "qwenox: state allocated (maxctx=%d)\n", maxctx);
  }

  ~GpuModel() {
    if (rbh) rocblas_destroy_handle(rbh);
    if (lth) hipblasLtDestroy(lth);
#ifndef _WIN32
    // Windows 下 d_Rhatbf16 来自 devarena bump 分配，不能单独 hipFree
    if (d_Rhatbf16) CK(hipFree(d_Rhatbf16));
#endif
  }

  // GR read: R -> Rhat -> t320 -> G -> x. prefix like
  // "layers.3.attn_hyper_connection" or "hyper_connection_mixer".
  void gr_read(const std::string& prefix, float* x_out) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.hc_norm.weight", prefix.c_str());
    // hc_norm: BF16 10240 — needs fp32; we did NOT upload per-layer norms!
    // per-layer hc_norm weights are uploaded lazily on first use (see ensure_hcnorm)
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_R, hw, d_Rhat, 2560, g_cfg.norm_eps, 4);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_Rhat, d_t320);
    k_silu_scale<<<(320 + 255) / 256, 256, 0, g_str>>>(d_t320, 0.25f, 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_t320, d_G);
    k_sigmoid_inplace<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_G, 10240);
    k_gr_combine<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_G, d_Rhat, x_out, g_cfg.d,
                                              g_cfg.branches);
  }

  void gr_write(const std::string& prefix, const float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_Rhat, d_w4);
    k_gr_write<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_w4, d_R, y, g_cfg.d, g_cfg.branches);
  }

  // lazy cache of per-layer hc_norm + q/k_norm fp32 uploads
  std::unordered_map<std::string, float*> hcnorm_cache_;
  float* ensure_hcnorm(const std::string& prefix) {
    auto it = hcnorm_cache_.find(prefix);
    if (it != hcnorm_cache_.end()) return it->second;
    char buf[128];
    snprintf(buf, sizeof buf, "%s.hc_norm.weight", prefix.c_str());
    float* p;
    CK(hipMalloc(&p, 10240 * 4));
    std::vector<float> tmp(10240);
    ckpt.dequant(ckpt.at(buf), tmp.data());
    CK(hipMemcpy(p, tmp.data(), 10240 * 4, hipMemcpyHostToDevice));
    hcnorm_cache_[prefix] = p;
    return p;
  }
  float* ensure_norm256(const std::string& name) {
    auto it = hcnorm_cache_.find(name);
    if (it != hcnorm_cache_.end()) return it->second;
    float* p;
    CK(hipMalloc(&p, 256 * 4));
    std::vector<float> tmp(256);
    ckpt.dequant(ckpt.at(name), tmp.data());
    CK(hipMemcpy(p, tmp.data(), 256 * 4, hipMemcpyHostToDevice));
    hcnorm_cache_[name] = p;
    return p;
  }

  // Draft lm_head override: keep a private device copy of the BASE file's
  // q4cp lm_head when an overlay replaced it with a wider dtype. Drafting
  // reads it instead (~0.5x bytes per draft step); verify and serial decode
  // still use the production lm_head, so committed outputs are unchanged.
  void setup_draft_lmhead(const Tensor* base_lm) {
    if (env_zero("QWENOX_DRAFT_LM_Q4") || !base_lm || base_lm->dtype != 5) return;
    const Tensor& cur = ckpt.at("lm_head.weight");
    if (cur.data == base_lm->data) return;  // no overlay override
    uint64_t cols = base_lm->dims[base_lm->ndims - 1];
    uint64_t rows = base_lm->numel() / cols;
    if (rows != (uint64_t)g_cfg.vocab || cols != (uint64_t)g_cfg.d) return;
    uint8_t* p = nullptr;
    DALLOC(&p, base_lm->data_size);
    CK(hipMemcpy(p, base_lm->data, base_lm->data_size, hipMemcpyHostToDevice));
    w_dlm.dtype = base_lm->dtype;
    w_dlm.data = p;
    w_dlm.rows = rows;
    w_dlm.cols = cols;
    w_dlm.scale_stride = ((cols / 32 * 2) + 15) & ~15ull;
    dlm_on = true;
    fprintf(stderr,
            "draft lm_head: base q4cp %.0f MiB (overlay dtype %d skipped for drafts)\n",
            base_lm->data_size / 1048576.0, (int)cur.dtype);
  }

  // ---- M-RoPE (vision) ----
  // set_mrope expands the request's image grid_thw triples into per-token
  // 3-row positions, replicating reference get_rope_index (tools/mrope_ref.py):
  // text runs take arange+current_pos on all 3 rows; an image run consumes
  // one grid (t, h, w) -> t*(h/2)*(w/2) placeholder tokens on a 3D grid
  // (t outer, h mid, w inner; each row offset by the run's start), and
  // current_pos advances only max(h, w)//2. rope_delta = max(pos)+1-N is the
  // decode offset (reference modeling_qwen4_exp.py:2121,2237-2241).
  bool set_mrope(const std::vector<int>& ids,
                 const std::vector<std::array<int, 3>>& grids) {
    clear_mrope();
    const int IMG = 248056, VID = 248057, MERGE = 2;
    const int N = (int)ids.size();
    mrope_pos_.assign(N, {0, 0, 0});
    auto type = [&](int id) { return id == IMG ? 1 : id == VID ? 2 : 0; };
    size_t gi = 0;
    int cur = 0;
    long long pmax = -1;
    for (int i = 0; i < N;) {
      int tp = type(ids[i]), j = i + 1;
      while (j < N && type(ids[j]) == tp) j++;
      if (tp == 0) {
        for (int k = i; k < j; k++) {
          int p = cur + (k - i);
          mrope_pos_[k] = {p, p, p};
        }
        pmax = std::max<long long>(pmax, cur + (j - i) - 1);
        cur += j - i;
      } else {
        if (gi >= grids.size()) return false;
        auto g = grids[gi++];
        if (g[0] < 1 || g[1] <= 0 || g[2] <= 0 || g[1] % MERGE || g[2] % MERGE)
          return false;
        int t = g[0], h = g[1] / MERGE, w = g[2] / MERGE;
        if ((long long)t * h * w != j - i) return false;
        int k = i;
        for (int tt = 0; tt < t; tt++)
          for (int hh = 0; hh < h; hh++)
            for (int ww = 0; ww < w; ww++, k++)
              mrope_pos_[k] = {cur + tt, cur + hh, cur + ww};
        pmax = std::max<long long>(pmax, cur + std::max({t - 1, h - 1, w - 1}));
        cur += std::max(g[1], g[2]) / MERGE;
      }
      i = j;
    }
    if (gi != grids.size()) return false;  // unconsumed grids
    rope_delta_ = (int)(pmax + 1 - N);
    mrope_active_ = true;
    mrope_grids_ = grids;  // kvsnap meta: pixel-agnostic placeholders need this
    mrope_gen_++;
    CK(hipMemcpy(d_rdelta, &rope_delta_, 4, hipMemcpyHostToDevice));
    return true;
  }

  void clear_mrope() {
    img_inj.clear();  // vision per-request state dies with the mrope request
    img_emb_rows = 0;
    img_hash_ = 0;
    mrope_grids_.clear();
    if (!mrope_active_ && rope_delta_ == 0) return;
    mrope_active_ = false;
    rope_delta_ = 0;
    mrope_pos_.clear();
    mrope_gen_++;
    CK(hipMemcpy(d_rdelta, &rope_delta_, 4, hipMemcpyHostToDevice));
  }

  // ---- Vision tower wiring (stage 4) ----
  void load_vision(const char* path) {
    vt.lth = lth;
    vt.rbh = rbh;
    vt.d_ltws = d_ltws;
    vt.lt_ws = lt_ws_bytes;
    vt.load(path);
    vision_on = true;
  }

  // Run the ViT over each image's patches and build the injection map.
  // grids/ids were already validated by set_mrope (same run-walking logic
  // here); images are t==1 only at this stage. Returns false on any
  // patches/grid mismatch (stderr carries the position).
  bool vision_embeds(const std::vector<int>& ids,
                     const std::vector<std::array<int, 3>>& grids,
                     const std::vector<std::vector<float>>& patches,
                     int cached = 0) {
    img_inj.clear();
    img_emb_rows = 0;
    const int IMG = 248056, VID = 248057, MERGE = 2;
    const int N = (int)ids.size();
    auto type = [&](int id) { return id == IMG ? 1 : id == VID ? 2 : 0; };
    size_t gi = 0, emb_off = 0;
    for (int i = 0; i < N;) {
      int tp = type(ids[i]), j = i + 1;
      while (j < N && type(ids[j]) == tp) j++;
      if (tp == 0) {
        i = j;
        continue;
      }
      if (tp != 1 || gi >= grids.size() || gi >= patches.size()) {
        fprintf(stderr, "vision: unsupported run at prompt pos %d\n", i);
        return false;
      }
      auto g = grids[gi];
      const std::vector<float>& px = patches[gi];
      gi++;
      int gh = g[1], gw = g[2];
      if (g[0] != 1 || gh <= 0 || gw <= 0 || gh % MERGE || gw % MERGE) {
        fprintf(stderr, "vision: bad grid t=%d h=%d w=%d at prompt pos %d\n",
                g[0], g[1], g[2], i);
        return false;
      }
      int P = gh * gw, M = P / 4;
      if ((int)px.size() != P * 1536 || j - i != M) {
        fprintf(stderr,
                "vision: patches/grid mismatch at prompt pos %d "
                "(patches=%zu floats, grid P=%d, run=%d tokens)\n",
                i, px.size(), P, j - i);
        return false;
      }
      if (j <= cached) {
        // Fully inside the reused live prefix: the KV already carries this
        // image's injected embeds. Shape validation above still applies;
        // skip the forward (and the injection rows, which would all fall
        // below the chunk base and be filtered out anyway).
        fprintf(stderr, "vision: image %zu grid=(%d,%d,%d) cached, forward "
                "skipped\n", gi, g[0], gh, gw);
        i = j;
        continue;
      }
      if (emb_off + (size_t)M > img_emb_cap) {
        // Grow WITHOUT losing the rows already written for earlier images of
        // this request: they are only consumed at prefill (k_img_inject), so
        // the old free-then-malloc dropped images 1..k-1 whenever image k
        // overflowed (first multi-image request per slot past 4096 rows).
        size_t ncap = std::max<size_t>(emb_off + M, 4096);
        uint16_t* nbuf = nullptr;
        CK(hipMalloc(&nbuf, ncap * 2560 * 2));
        if (d_img_emb && emb_off)
          CK(hipMemcpy(nbuf, d_img_emb, emb_off * 2560 * 2,
                       hipMemcpyDeviceToDevice));
        if (d_img_emb) (void)hipFree(d_img_emb);
        d_img_emb = nbuf;
        img_emb_cap = ncap;
      }
      auto t0 = std::chrono::steady_clock::now();
      // Debug hook (QWENOX_VISION_EMBEDS_NPY=<f4 (M,2560) npy>): inject
      // precomputed embeds (host bf16 RNE) instead of running the ViT.
      // Isolates LLM/injection behavior from ViT numerics in A/B probes.
      const char* dbg_emb = getenv("QWENOX_VISION_EMBEDS_NPY");
      if (dbg_emb) {
        VitNpy e = vit_npy_read(dbg_emb);
        if (e.shape.size() != 2 || e.shape[0] != (uint64_t)M ||
            e.shape[1] != 2560) {
          fprintf(stderr, "vision: embeds npy shape mismatch (want (%d,2560))\n", M);
          return false;
        }
        const float* ef = (const float*)e.data.data();
        std::vector<uint16_t> bf((size_t)M * 2560);
        for (size_t k = 0; k < bf.size(); k++) {
          uint32_t b;
          memcpy(&b, ef + k, 4);
          b += 0x7FFFu + ((b >> 16) & 1u);
          bf[k] = (uint16_t)(b >> 16);
        }
        CK(hipMemcpy(d_img_emb + emb_off * 2560, bf.data(),
                     (size_t)M * 2560 * 2, hipMemcpyHostToDevice));
      } else {
      int Mr = vt.forward(px.data(), P, gh, gw);
      if (Mr != M) return false;
      // forward only enqueues; the stitch copy must observe the finished
      // embeds (and the log line should report real forward time).
      CK(hipStreamSynchronize(g_str));
      CK(hipMemcpy(d_img_emb + emb_off * 2560, vt.d_emb_bf16,
                   (size_t)M * 2560 * 2, hipMemcpyDeviceToDevice));
      }
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
      fprintf(stderr, "vision: image %zu grid=(%d,%d,%d) P=%d M=%d "
              "forward %.0f ms\n", gi, g[0], gh, gw, P, M, ms);
      for (int k = 0; k < M; k++)
        img_inj.push_back(make_int2((int)emb_off + k, i + k));
      emb_off += M;
      i = j;
    }
    if (gi != grids.size() || gi != patches.size()) {
      fprintf(stderr, "vision: %zu grids/patches for unconsumed runs\n",
              grids.size() - gi);
      return false;
    }
    img_emb_rows = emb_off;
    img_hash_ = fnv1a_patches(patches);
    return true;
  }


  // ---- M-RoPE debug dump (QWENOX_MROPE_DUMP=<file>; syncs the stream) ----
  // Format: "META rope_delta=<d>", then per table fill one line per row:
  //   ROW <base> <row> <pT> <pH> <pW> <cos0> <sin0> ... <cos31> <sin31>
  // and per decode step:
  //   DEC <tag> <rpos> <rdelta> <cos0> <sin0> ... <cos31> <sin31>
  // cos/sin as C %a hexfloats (exact binary value of the fp32 table entry).
  FILE* mrope_dump() {
    if (mrope_dump_ || !getenv("QWENOX_MROPE_DUMP")) return mrope_dump_;
    mrope_dump_ = fopen(getenv("QWENOX_MROPE_DUMP"), "w");
    if (mrope_dump_) fprintf(mrope_dump_, "# qwenox-mrope-dump v1\n");
    return mrope_dump_;
  }

  void dump_rope_cs(int base, int P) {
    FILE* f = mrope_dump();
    if (!f) return;
    if (!mrope_dump_meta_) {
      fprintf(f, "META rope_delta=%d\n", rope_delta_);
      mrope_dump_meta_ = true;
    }
    std::vector<float2> h((size_t)P * 32);
    CK(hipMemcpy(h.data(), d_ropecs, h.size() * sizeof(float2),
                 hipMemcpyDeviceToHost));
    for (int r = 0; r < P; r++) {
      int pT = base + r, pH = base + r, pW = base + r;
      if (mrope_active_ && base + r < (int)mrope_pos_.size()) {
        pT = mrope_pos_[base + r][0];
        pH = mrope_pos_[base + r][1];
        pW = mrope_pos_[base + r][2];
      } else if (mrope_active_) {
        pT = pH = pW = base + r + rope_delta_;
      }
      fprintf(f, "ROW %d %d %d %d %d", base, r, pT, pH, pW);
      for (int i = 0; i < 32; i++)
        fprintf(f, " %a %a", (double)h[(size_t)r * 32 + i].x,
                (double)h[(size_t)r * 32 + i].y);
      fprintf(f, "\n");
    }
    fflush(f);
  }

  void dump_decode_step(int tag) {
    FILE* f = mrope_dump();
    if (!f || !d_cs1) return;
    if (!mrope_dump_meta_) {
      fprintf(f, "META rope_delta=%d\n", rope_delta_);
      mrope_dump_meta_ = true;
    }
    k_rope_dump<<<1, 32, 0, g_str>>>(d_cs1, d_rpos1, d_pos, d_rdelta,
                                     g_cfg.rope_theta);
    float2 h[32];
    int rpos = -1;
    CK(hipMemcpy(h, d_cs1, sizeof h, hipMemcpyDeviceToHost));
    CK(hipMemcpy(&rpos, d_rpos1, 4, hipMemcpyDeviceToHost));
    fprintf(f, "DEC %d %d %d", tag, rpos, rope_delta_);
    for (int i = 0; i < 32; i++)
      fprintf(f, " %a %a", (double)h[i].x, (double)h[i].y);
    fprintf(f, "\n");
    fflush(f);
  }

  // P3-A: mirror the batch base position to device so batch-path kernels read
  // it from memory (a fixed pointer arg replays correctly under graph capture;
  // a baked scalar would freeze the capture-time base). Same value, same math
  // -> bit-identical. Stream-ordered: every consumer's public entry re-writes
  // it before launching. QWENOX_VBASE=0 falls back to scalar args everywhere.
  static bool vbase_on() { return !env_zero("QWENOX_VBASE"); }
  const int* vbptr() const { return vbase_on() ? d_vbase : nullptr; }
  // Staging is a small pinned ring, not one slot: a pinned async copy reads
  // host memory when the stream reaches it, so rewriting a single slot before
  // the GPU got there would hand an earlier chunk the later chunk's base.
  // Each slot's copy is fenced by an event and waited on before reuse (in
  // practice never blocks). NOTE (stage 2 / graph capture): hipEventSynchronize
  // is illegal while capturing -- a captured round needs its own staging slot
  // written before each replay, with the per-round readback as the fence.
  static constexpr int kVbRing = 4;
  hipEvent_t vb_ev[kVbRing] = {};
  int vb_slot = 0;
  void set_vbase(int base) {
    if (!vbase_on()) return;
    const int s = vb_slot;
    vb_slot = (vb_slot + 1) % kVbRing;
    if (vb_ev[s]) CK(hipEventSynchronize(vb_ev[s]));
    else CK(hipEventCreateWithFlags(&vb_ev[s], hipEventDisableTiming));
    p_vbase[s] = base;
    CK(hipMemcpyAsync(d_vbase, p_vbase + s, 4, hipMemcpyHostToDevice, g_str));
    CK(hipEventRecord(vb_ev[s], g_str));
  }

  // Refill the rope cos/sin table only when the chunk's (base, P) changes:
  // one compute per chunk, shared by all 12 QSA layers (stream-ordered).
  // M-RoPE: host fill with the request's per-token 3-row positions,
  // slot-interleaved; same fp64 expression as k_rope_cs (pow + libm cos/sin,
  // cast to float). Text requests keep the device-kernel fill untouched.
  void ensure_rope_cs(int base, int P) {
    if (base == ropecs_base_ && P == ropecs_P_ && mrope_gen_ == ropecs_gen_)
      return;
    if (mrope_active_) {
      double tab[32];
      for (int i = 0; i < 32; i++)
        tab[i] = rope_inv_freq(g_cfg.rope, g_cfg.rope_theta, 64, i);
      h_ropecs_.resize((size_t)P * 32);
      const int mropeN = (int)mrope_pos_.size();
      for (int r = 0; r < P; r++)
        for (int i = 0; i < 32; i++) {
          // Spec verify runs with base=q+1 >= N: mrope_pos_ only holds the
          // prompt's N rows. Verified positions keep advancing from the last
          // mrope row by +1 (same linear sequence mrope_step_ produces), so
          // fall back to rope_delta_ + absolute position out of range.
          int p = (base + r < mropeN)
                      ? mrope_pos_[base + r][mrope_slot_row(i)]
                      : (base + r + rope_delta_);
          double ang = (double)p * tab[i];
          h_ropecs_[(size_t)r * 32 + i] =
              make_float2((float)cos(ang), (float)sin(ang));
        }
      CK(hipMemcpy(d_ropecs, h_ropecs_.data(),
                   (size_t)P * 32 * sizeof(float2), hipMemcpyHostToDevice));
    } else {
      set_vbase(base);
      k_rope_cs<<<(P * 32 + 255) / 256, 256, 0, g_str>>>(d_ropecs, base, P,
                                                         vbptr());
    }
    ropecs_base_ = base;
    ropecs_P_ = P;
    ropecs_gen_ = mrope_gen_;
    dump_rope_cs(base, P);
  }

  void gdn(int l, int gi, const float* x, float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_qkv.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_qkv);
    // conv + silu (state per layer)
    float* convst = d_convst + (size_t)gi * 3 * 10240;
    const float* cw = d_convw + (size_t)gi * 10240 * 4;
    k_gdn_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_qkv, cw, convst, d_qkv, 10240);
    // note: in-place (cur==out) is safe: read cur[ch] before write out[ch]
    // split: q [0,2048) k [2048,4096) v [4096,10240)
    float* qp = d_qkv;
    float* kp = d_qkv + 2048;
    float* vp = d_qkv + 4096;
    k_l2norm_qk<<<16, 128, 0, g_str>>>(qp, kp, 128, 1.f / sqrtf(128.f));
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_a.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_g48 /*tmp*/);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_b.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_beta48 /*tmp: raw b*/);
    k_gdn_gates<<<1, 64, 0, g_str>>>(d_g48, d_beta48, d_Alog + (size_t)gi * 48,
                           d_dtbias + (size_t)gi * 48, d_g48, d_beta48, 48, 48);
    k_gdn_step<<<48, 512, 0, g_str>>>(qp, kp, vp, d_g48, d_beta48,
                            d_S + (size_t)gi * 48 * 128 * 128, vp /*out in place*/,
                            128, 128, 48, 16);
    // out written over vp is safe: k_gdn_step reads vh before writing out
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_z);
    k_gdn_gatednorm<<<dim3(48, 1), 128, 0, g_str>>>(vp, d_z, d_gdnnorm + (size_t)gi * 128,
                                                    vp, 128, g_cfg.norm_eps, 10240,
                                                    6144);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
    gemv(wview(ckpt.at(buf)), vp, y);
  }

  // One decode step of QSA attention over the KV cache: flash-decoding split
  // over QSA_DEC_NSPLIT position slices plus a merge, for every token. Dense
  // tokens (< 2051) take the same path (ntok = token+1, identity source map);
  // the old 3-pass k_qsa_step dense kernel cost ~0.7ms->5.4ms/layer growing
  // linearly with pos (rocprofv3: 55.8% of all decode kernel time), while the
  // flash split stays flat. QWENOX_QSA_DENSE takes the same split kernel with no
  // selection list (all visible positions). k_qsa_step used to be its fallback,
  // but its pass 3 recomputes every q.k per output dim: 1170 ms/token at 8K.
  template <typename KVT>
  void qsa_attn(const KVT* kc, const KVT* vc, const int* ptab = nullptr) {
    k_qsa_flash<true, false, KVT, true><<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
        d_qs, kc, vc, d_attn, 1, 256, 2, 24, qsa_dense ? nullptr : d_iselected, 0,
        d_pos, d_pacc, d_pml, 0, ptab);
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, d_attn, 12,
                                                       256, d_pos);
  }

  void qsa(int l, int qi, const float* x, float* y) {
    char buf[128];
    ensure_rope_tab(g_cfg.rope_theta, g_cfg.rotary_dim);
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      float* keys = d_ik + (size_t)qi * ik_layer_blocks * 128;
      uint16_t* keys16 =
          d_ikh ? d_ikh + (size_t)qi * ik_layer_blocks * 128 : nullptr;
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256, d_iq,
                                              1, g_cfg.norm_eps, g_cfg.rope_theta, d_pos,
                                              0, nullptr, d_rdelta);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256 + 128,
                                           d_iraw + qi * 512, keys, d_pos,
                                           g_cfg.norm_eps, g_cfg.rope_theta,
                                           d_rdelta, kvpt(), keys16);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(d_iq, keys, d_iscores,
                                                               index_blocks, d_pos,
                                                               kvpt());
      index_select(d_iscores, index_blocks, d_iselected, 0, 1, d_pos);
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_qg);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_kb);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.v_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_vb);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_norm.weight", l);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_norm.weight", l);
    float* knw = ensure_norm256(buf);
    // per-head: q[h] = zcnorm(qg[h*512 .. +256]); gate[h] = qg[h*512+256 ..]
    k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(d_qg, qnw, d_qs, d_gs, 256,
                                                 g_cfg.norm_eps);
    k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(d_kb, knw, d_kb, 256, g_cfg.norm_eps, 1);
    k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(d_qs, 24, 256, 64, d_pos,
                                                     g_cfg.rope_theta, d_rdelta);
    k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(d_kb, 2, 256, 64, d_pos,
                                                    g_cfg.rope_theta, d_rdelta);
    // append to cache at *d_pos
    if (qsa_kv_fp32) {
      float* kc = d_kc + (size_t)qi * kv_layer_rows * 512;
      float* vc = d_vc + (size_t)qi * kv_layer_rows * 512;
      k_store_kv<<<2, 256, 0, g_str>>>(d_kb, d_vb, kc, vc, d_pos, kvpt());
      qsa_attn(kc, vc, kvpt());
    } else {
      uint16_t* kc = d_kcb + (size_t)qi * kv_layer_rows * 512;
      uint16_t* vc = d_vcb + (size_t)qi * kv_layer_rows * 512;
      // BTV: keep the transposed prefill V cache complete for decoded
      // positions (read later by sparse prefill/verify via k_qsa_wmma<true>)
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(d_kb, d_vb, kc, vc, d_pos, vct_layer(qi),
                                            kvpt());
      qsa_attn(kc, vc, kvpt());
    }
    k_sigmoid_gate<<<(24 * 256 + 255) / 256, 256, 0, g_str>>>(d_attn, d_gs, d_attn, 24 * 256);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.o_proj.weight", l);
    gemv(wview(ckpt.at(buf)), d_attn, y);
  }


  // ---- 专家权重直读(零拷贝,gfx1151 统一内存;唯一形态,全量 arena 已移除)
  // load_arena 已把 .mlp.experts. 张量登记为 host mmap 指针;kernel 直读
  // 页缓存实测 224 GB/s(tools/tgv2_probe,与 GTT 同一片 DRAM),gather
  // kernel 按专家 id 索引全量张量,读到的字节与 arena 逐位一致 → 数值
  // 不变、kernel 零改动。decode 无 D2H routing 同步、无 H2D staging,
  // 指针全程静态,图捕获照常。
  // (2026-09-12 staging 方案曾记"裸指针 Memory Fault",与今日实测矛盾;
  //  当时 fault 应另有原因,本路径以 oracle 验证为准。)
  // Single-token routed down projection (trunk decode + MTP draft) into
  // d_moeacc. The atomic kernel sums the 10 expert partials in wave-schedule
  // order, so a decode step was not bit-reproducible: timing changes (another
  // slot's work in between, a debug sync) flipped near-tie tokens thousands of
  // steps later. The top-k kernel keeps the partials in registers and sums in
  // expert order, overwriting every row (no memset). QWENOX_MOE_DOWN_ATOMIC /
  // QWENOX_MOE_ATOMIC restore the old path.
  bool moe_down_direct(int k, const WView& dn) const {
    return moe_down_topk && k == 10 && dn.cols == 640;
  }
  // Routed-expert tensor -> operands of the q4cp-style grouped kernels. q4cp
  // (dtype 5): 64 B codebook header, one f16 scale per 32 cols (GSH 0). hgn v2
  // i4r (dtype 23): codes at 0, linear -8..7 book, one scale per 128 (GSH 2).
  struct XCodes {
    const uint8_t* codes;
    const uint8_t* scales;
    const float* cb;
  };
  static XCodes xcodes(const WView& w) {
    if (w.dtype == 23) return {w.data, w.data + w.rows * w.cols / 2, g_i4r_cb};
    return {w.data + 64, w.data + 64 + w.rows * w.cols / 2, (const float*)w.data};
  }
  template <int GSH>
  void moe_down1_t(int k, const WView& dn, const int* topids) {
    const XCodes c = xcodes(dn);
    if (moe_down_direct(k, dn)) {
      const uint64_t row_pairs = (g_cfg.d + 1) / 2, warps_per_block = 128 / 32;
      k_q4cp_gemv_gd_topk_h16<10, GSH><<<(unsigned)((row_pairs + warps_per_block - 1) / warps_per_block),
                                    128, 0, g_str>>>(
          c.codes, c.scales, c.cb, d_hid,
          d_moeacc, topids, d_topw, (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
          (uint64_t)g_cfg.moe_mid, 1, 16 /*id_stride*/, (uint64_t)g_cfg.d);
      return;
    }
    // v3 kernel: 2 rows per warp -> k * (d/2) warp-pairs
    uint64_t pairs = (uint64_t)k * ((g_cfg.d + 1) / 2);
    k_q4cp_gemv_gd<GSH><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
        c.codes, c.scales, c.cb, d_hid,
        d_moeacc, topids, d_topw, (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
        (uint64_t)g_cfg.moe_mid, k, 16 /*id_stride*/, 0 /*acc_stride*/, k);
  }
  void moe_down1(int k, const WView& dn, const int* topids) {
    if (dn.dtype == 23)
      moe_down1_t<2>(k, dn, topids);
    else
      moe_down1_t<0>(k, dn, topids);
  }
  // hgn v2 rotation helpers (k_v2_rot): P rows of n values, f32 in / f32 or
  // f16 out; pre = input plane (rot_in), post = output plane (rot_out).
  void v2_rot(const float* x, uint64_t xs, void* y, uint64_t ys, const float* pre,
              const float* post, int n, int P, bool outh = false) {
    const unsigned waves = (unsigned)P * (n / 128);
    if (outh)
      k_v2_rot<float, true><<<(waves + 7) / 8, 256, 0, g_str>>>(x, xs, y, ys, pre, post, n, P);
    else
      k_v2_rot<float, false><<<(waves + 7) / 8, 256, 0, g_str>>>(x, xs, y, ys, pre, post, n, P);
  }
  // i4r routed experts for one token (decode / MTP-free trunk): x -> d_moeacc.
  //   rot_in(x, su) -> gate_up (rotated) -> k_i4r_mid (rot_out, silu*u,
  //   rot_in(down.suh)) -> down (rotated, weighted sum over experts) ->
  //   rot_out(acc, down.svh). svh is shared by all experts of a layer, so the
  //   output rotation runs once on the weighted sum.
  void moe_v2_1(int k, const WView& gu, const WView& dn, const float* x, const int* topids) {
    const int d = g_cfg.d, mid = g_cfg.moe_mid;
    if (!moe_down_direct(k, dn)) CK(hipMemsetAsync(d_moeacc, 0, d * 4, g_str));
    v2_rot(x, 0, d_ey, 0, gu.suh, nullptr, d, 1);  // d_ey: free until the shared expert
    const XCodes c = xcodes(gu);
    const uint64_t pairs = (uint64_t)k * mid;
    k_q4cp_gemv_gg<2><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
        c.codes, c.scales, c.cb, d_ey, d_guv, topids, (uint64_t)2 * mid, gu.cols,
        gu.scale_stride, k, 0 /*x_stride*/, 16 /*id_stride*/, k);
    k_i4r_mid<float, false><<<k, 2 * mid / 128 * 32, 0, g_str>>>(d_guv, 2 * mid, d_hid, mid,
                                                                 gu.svh, dn.suh, mid);
    moe_down1(k, dn, topids);
    v2_rot(d_moeacc, 0, d_moeacc, 0, nullptr, dn.svh, d, 1);
  }
  // GGUF routed experts g (trunk layer g_ggx[l] or the MTP layer g_ggx_mtp;
  // needs gg_on() for the scratch buffers): x f32 [P][d] with top-k
  // rows (16 wide) -> acc f32 [P][d], every element overwritten (no memset).
  // P == 1 (decode graph): sort-free per-pair tiles on private scratch.
  // P > 1: stable GPU bucketing + 64-slot tiles on the prefill workspaces
  // (F16 x in d_xbf16b, which is never cached; hid F16 in d_hidb; unweighted
  // pairs in d_guvb, dead again once the weighted reduce has run).
  void moe_gguf(const GgExperts& g, const float* x, const int* topids, const float* topw,
                float* acc, int P) {
    const int k = g_cfg.topk, E = g_cfg.experts, d = g_cfg.d, mid = g_cfg.moe_mid;
    // Small P (decode, spec verify): raw-block GEMVs, FP32 hid, weighted sum in
    // expert-rank order written straight to acc (26_kernels_moe_gguf.inc).
    // QWENOX_GG_GEMV=<max P> (default 8, 0 = always WMMA).
    static const int gemv_max = [] {
      const char* s = getenv("QWENOX_GG_GEMV");
      return s && *s ? atoi(s) : 8;
    }();
    // 2 <= P <= 8: experts shared between the P tokens (spec verify) are read
    // once (bit-identical to the plain path); unweighted pairs go to d_guvb.
    // QWENOX_GG_UQ=0 disables.
    static const bool uq_on = [] {
      const char* s = getenv("QWENOX_GG_UQ");
      return !(s && *s == '0');
    }();
    const bool uq = uq_on && P >= 2;
    if (P <= gemv_max && P <= maxbatch_cap &&
        moe_gg_gemv(g.tgu, g.tdn, g.gate, g.up, g.down, x, topids, topw,
                    P == 1 ? d_hid : d_hidb, acc, P, d, mid, 16, g_str,
                    uq ? d_gg_uq : nullptr, uq ? d_guvb : nullptr))
      return;
    int *tokidx, *pairids, *ntiles;
    float *pw, *pairs;
    MoeTile* tiles;
    __half *x16, *hid;
    int max_tiles;
    if (P == 1) {
      k_moe_gg_route1<<<1, 64, 0, g_str>>>(topids, topw, 1, k, 16, d_gg_tok, d_gg_pw,
                                           d_gg_pid, d_gg_tiles, d_gg_nt);
      tokidx = d_gg_tok, pairids = d_gg_pid, ntiles = d_gg_nt, pw = d_gg_pw;
      tiles = d_gg_tiles, x16 = d_gg_x16, hid = d_gg_hid, pairs = d_gg_pairs;
      max_tiles = k;
    } else {
      if (P > maxbatch_cap) throw std::runtime_error("moe_gguf: P exceeds the batch buffers");
      moe_routing.run(topids, topw, d_tokidx, d_pweight, d_eoff, d_pairids, P, k, E);
      k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 64);
      tokidx = d_tokidx, pairids = d_pairids, ntiles = d_moentiles, pw = d_pweight;
      tiles = d_moetiles, x16 = (__half*)d_xbf16b, hid = (__half*)d_hidb, pairs = d_guvb;
      max_tiles = (P * k + 63) / 64 + E;
    }
    const size_t n4 = (size_t)P * d / 4;
    k_f32_to_f16_v4<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(x, x16, n4);
    if (!moe_gg_up(g.tgu, g.gate, g.up, x16, tokidx, tiles, ntiles, max_tiles, hid, mid, d,
                   g_str) ||
        !moe_gg_down(g.tdn, g.down, hid, tiles, ntiles, max_tiles, pairs, d, mid, g_str))
      throw std::runtime_error("moe_gguf: unsupported expert type");
    k_moe_reduce_pw<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(pairs, pairids, pw, acc,
                                                                     P, d);
  }

  // hgn q4cp routed experts through the LUT-decode WMMA kernels (QWENOX_MOE_Q4W):
  // same prefill workspaces / dataflow as moe_gguf's P > 1 branch; acc f32
  // [P][d] fully overwritten. Returns false (nothing launched) if unsupported.
  // Whether moe_q4w_run(gu, dn, ..., P) will take the LUT path (no launches).
  bool moe_q4w_ok(const WView& gu, const WView& dn, int P) const {
    const int k = g_cfg.topk, E = g_cfg.experts, d = g_cfg.d, mid = g_cfg.moe_mid;
    if (!moe_q4w || k != 10 || !moe_deterministic || moe_host_route || moe_ordered ||
        moe_untiled || P > maxbatch_cap || gu.dtype != 5 || dn.dtype != 5 ||
        gu.cols != (uint64_t)d || gu.rows != (uint64_t)E * 2 * mid || dn.cols != (uint64_t)mid ||
        dn.rows != (uint64_t)E * d)
      return false;
    const moelut::LutW pgu = lut_q4cp_view(gu.data, gu.rows, gu.cols, gu.scale_stride, 2 * mid, mid);
    const moelut::LutW pdn = lut_q4cp_view(dn.data, dn.rows, dn.cols, dn.scale_stride, d, 0);
    return !(mid % 64 || d % 128 || !moelut::lut_ok(moelut::kQ4CP, pgu, true, d) ||
             !moelut::lut_ok(moelut::kQ4CP, pdn, false, mid));
  }
  // ey/sg (optional): shared-expert output + gate logit, added in the reduce
  // (k_moe_reduce_pw_sg) instead of by k_axpy_sg afterwards.
  // w4hc/Rhc (optional, requires ey): the reduce fuses the HC write-back
  // (k_moe_reduce_pw_sg_hc) and acc is left untouched.
  bool moe_q4w_run(const WView& gu, const WView& dn, const float* x, const int* topids,
                   const float* topw, float* acc, int P, const float* ey = nullptr,
                   const float* sg = nullptr, const float* w4hc = nullptr,
                   uint16_t* Rhc = nullptr) {
    const int k = g_cfg.topk, E = g_cfg.experts, d = g_cfg.d, mid = g_cfg.moe_mid;
    if (!moe_q4w_ok(gu, dn, P)) return false;
    const moelut::LutW pgu = lut_q4cp_view(gu.data, gu.rows, gu.cols, gu.scale_stride, 2 * mid, mid);
    const moelut::LutW pdn = lut_q4cp_view(dn.data, dn.rows, dn.cols, dn.scale_stride, d, 0);
    static bool logged = false;
    if (!logged) {
      logged = true;
      fprintf(stderr, "moe prefill: q4cp LUT WMMA (QWENOX_MOE_Q4W=0 disables)\n");
    }
    KP("moe:route");
    moe_routing.run(topids, topw, d_tokidx, d_pweight, d_eoff, d_pairids, P, k, E);
    // §11.5 mt（q4cp 站）：同专家双 64-tile 合并 BN=128，每个 K stage 的权重
    // 解码一次供两个 tile 半用，逐 bit 不变；默认开，QWENOX_MOE_MT=0 回退。
    // 门槛同 v2：平均每专家 slot >= 80（P=2048 档 v2 实测 −5.2%，小 P 走原
    // 路径）。w4b 与 v2 不同：moe 在关键路径上（segsum==wall，见 §11.5）。
    static const bool moe_mt_env = !env_zero("QWENOX_MOE_MT");
    const bool mt = moe_mt_env && (int64_t)P * k >= 80 * E;
    k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, mt ? 128 : 64);
    const int max_tiles = mt ? (P * k + 127) / 128 + E : (P * k + 63) / 64 + E;
    __half* x16 = (__half*)d_xbf16b;
    __half* hid = (__half*)d_hidb;
    const size_t n4 = (size_t)P * d / 4;
    k_f32_to_f16_v4<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(x, x16, n4);
    KP("moe:up");
    if (mt)
      moe_lut_up_mt_q4cp(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, mid, d,
                         g_str);
    else
      moe_lut_up(moelut::kQ4CP, pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, mid, d,
                 g_str);
    KP("moe:down");
    // B1 w4b f16 pairs（与 v2 QWENOX_V2_PAIRS_F16 同款）：down 写 f16、reduce 读
    // f16（__half2float 精确拓宽，fp32 累加不变），reduce 读流量减半。默认开
    // （KLD 过线后翻转，=0 回退）。w4b_pairs_f16_env 与分配门控
    // guvb_f16_env 语义同步。
    static const bool pairs_f16 = w4b_pairs_f16_env();
    if (mt)
      moe_lut_down_mt_q4cp(pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid, g_str,
                           pairs_f16);
    else
      moe_lut_down(moelut::kQ4CP, pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid,
                   g_str, pairs_f16);
    KP("moe:reduce");
    if (ey)
      if (Rhc)
        if (pairs_f16)
          k_moe_reduce_pw_sg_hc<10, __half><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
              (const __half*)d_guvb, d_pairids, d_pweight, ey, sg, w4hc, Rhc, P, d);
        else
          k_moe_reduce_pw_sg_hc<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
              d_guvb, d_pairids, d_pweight, ey, sg, w4hc, Rhc, P, d);
      else if (pairs_f16)
        k_moe_reduce_pw_sg<10, __half><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
            (const __half*)d_guvb, d_pairids, d_pweight, ey, sg, acc, P, d);
      else
        k_moe_reduce_pw_sg<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
            d_guvb, d_pairids, d_pweight, ey, sg, acc, P, d);
    else if (pairs_f16)
      k_moe_reduce_pw<10, __half><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
          (const __half*)d_guvb, d_pairids, d_pweight, acc, P, d);
    else
      k_moe_reduce_pw<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(d_guvb, d_pairids,
                                                                       d_pweight, acc, P, d);
    return true;
  }

  void moe(int l, const float* x, float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.mlp.gate.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_router);
    int k = g_cfg.topk;
    k_router_topk<<<1, 512, 0, g_str>>>(d_router, d_topids, d_topw, g_cfg.experts, k);
    if (dbg) {
      int hi[16];
      float hw[16];
      CK(hipMemcpy(hi, d_topids, k * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(hw, d_topw, k * 4, hipMemcpyDeviceToHost));
      fprintf(stderr, "gpu L%02d moe topk:", l);
      for (int i = 0; i < k; i++) fprintf(stderr, " %d:%.5f", hi[i], (double)hw[i]);
      fprintf(stderr, "\n");
    }

    if (gg_on()) {
      moe_gguf(g_ggx.at(l), x, d_topids, d_topw, d_moeacc, 1);
    } else {
      snprintf(buf, sizeof buf, "layers.%d.mlp.experts.gate_up_proj.weight", l);
      WView gu = wview(ckpt.at(buf));
      snprintf(buf, sizeof buf, "layers.%d.mlp.experts.down_proj.weight", l);
      WView dn = wview(ckpt.at(buf));
      if (gu.dtype == 23) {
        moe_v2_1(k, gu, dn, x, d_topids);
      } else {
      if (!moe_down_direct(k, dn)) CK(hipMemsetAsync(d_moeacc, 0, g_cfg.d * 4, g_str));
      const int* topids = d_topids;  // 直读:ids 即专家号,免 D2H 同步
      // fused experts: one grouped launch each for gate_up / silu / down+accum
      {
        // v3 kernel: 2 rows per warp -> k * moe_mid warp-pairs
        uint64_t pairs = (uint64_t)k * g_cfg.moe_mid;
        k_q4cp_gemv_gg<0><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
            gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data, x,
            d_guv, topids, (uint64_t)2 * g_cfg.moe_mid, gu.cols, gu.scale_stride, k,
            0 /*x_stride*/, 16 /*id_stride*/, k);
      }
      k_silu_mul_g<<<(k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(d_guv, d_hid,
                                                                       g_cfg.moe_mid, k);
      moe_down1(k, dn, topids);
      }
    }
    // shared expert
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.gate_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_guv);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.up_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_guv + 640);
    k_silu_mul<<<(640 + 255) / 256, 256, 0, g_str>>>(d_guv, d_guv + 640, d_hid, 640);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.down_proj.weight", l);
    gemv(wview(ckpt.at(buf)), d_hid, d_ey);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert_gate.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_sg);
    k_axpy_sg<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_moeacc, d_ey, d_sg, g_cfg.d,
                                                     g_cfg.d);
    CK(hipMemcpyAsync(y, d_moeacc, g_cfg.d * 4, hipMemcpyDeviceToDevice, g_str));
  }

  // PLE host phase: hash + read 16 rows into pinned staging (disk latency — overlapped).
  float* ple_e = nullptr;  // pinned
  // one raw table row (g_ple_rb bytes) -> 160 f32
  void ple_deq(const uint8_t* p, float* dst) const {
    if (ple_iq4) {
      gguf::dequant_row(gguf::IQ4_NL, p, dst, 160);
      return;
    }
    for (int i = 0; i < 160; i++) dst[i] = hgn::fp8e4m3_to_f32(p[i]) * ple_tscale;
  }
  // reads the 16 PLE rows for chist position t into dst (pure read of chist;
  // safe to call concurrently for distinct t/dst).
  void ple_rows(int t, float* dst) {
    auto cat = [&](int back) -> int64_t {
      int j = t - back;
      return j < 0 ? pad_id : chist[j];
    };
    const uint8_t* addrs[16];
    // pass 1: compute rows, kick async readahead on all 16 pages (concurrent I/O)
    for (int h = 0; h < 16; h++) {
      int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
      if (h >= 8) mix ^= cat(2) * ple_mult[2];
      uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
      const uint8_t* p = ple_table->data + row * g_ple_rb;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      uint8_t stg[16 * 160];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, stg, (uint64_t)(uintptr_t)ple_table->data, 16);
      for (int h = 0; h < 16; h++) ple_deq(stg + h * g_ple_rb, dst + h * 160);
      return;
    }
#endif
    // pass 2: read (pages already in flight)
    for (int h = 0; h < 16; h++) ple_deq(addrs[h], dst + h * 160);
  }
  // raw variant: staging keeps the g_ple_rb row bytes per head (fp8 codes or
  // IQ4_NL blocks); dequant happens on GPU (k_ple_fp8_dequant /
  // k_ple_iq4nl_dequant). ~4x less host work per token.
  void ple_rows_fp8(int t, uint8_t* dst) {
    auto cat = [&](int back) -> int64_t {
      int j = t - back;
      return j < 0 ? pad_id : chist[j];
    };
    const uint8_t* addrs[16];
    for (int h = 0; h < 16; h++) {
      int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
      if (h >= 8) mix ^= cat(2) * ple_mult[2];
      uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
      const uint8_t* p = ple_table->data + row * g_ple_rb;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, dst, (uint64_t)(uintptr_t)ple_table->data, 16);
      return;
    }
#endif
    for (int h = 0; h < 16; h++) memcpy(dst + h * g_ple_rb, addrs[h], g_ple_rb);
  }
  void ple_host(int token, float* dst) {
    chist.push_back(token);
    ple_rows((int)chist.size() - 1, dst);
  }

  // PLE device phase: upload + gate/conv kernels (must be enqueued after L0).
  void ple_gpu() {
    CK(hipMemcpyAsync(d_e, ple_e, 2560 * 4, hipMemcpyHostToDevice, g_str));
    gemv(wview(ckpt.at("layers.1.ple.key_proj.weight")), d_e, d_keys);
    gemv(wview(ckpt.at("layers.1.ple.value_proj.weight")), d_e, d_v);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_keys, d_ple_nk, d_kn, 2560,
                                                g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_R, d_ple_nk + 10240, d_qn, 2560,
                                                g_cfg.norm_eps, 4);
    k_ple_gate<<<4, 1024, 0, g_str>>>(d_kn, d_qn, d_v, d_U, 2560, 4);
    // ring[ringpos] = norm_conv(U)
    k_rmsnorm_zc_ring<<<4, 1024, 0, g_str>>>(d_U, d_ple_nk + 20480, d_ring, d_ringpos,
                                             2560, g_cfg.norm_eps);
    k_ple_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_ring, d_ringpos, d_ple_convw,
                                                       d_convout, 10240);
    k_add2<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_R, d_U, d_convout, 10240);
    // device ringpos advanced by k_step_incr (graph) or host upload (direct)
  }

  // one token; returns true + writes argmax id when want_token
  bool dbg = getenv("QWENOX_DEBUG") != nullptr;
  bool perf = getenv("QWENOX_PERF") != nullptr;
  bool qsa_loop = getenv("QWENOX_QSA_LOOP") != nullptr;  // prefill QSA fallback
  bool qsa_global_v = getenv("QWENOX_QSA_GLOBAL_V") != nullptr;
  bool moe_naive = getenv("QWENOX_MOE_NAIVE") != nullptr;  // prefill MoE fallback
  // Small batches: per-pair naive gemv reads ~the same expert bytes as the
  // packed path (few shared experts) but at ~2x its effective bandwidth
  // (measured: P=8 chunk 0.19s -> 0.14s). 0 disables the auto switch.
  int moe_naive_max = [] {
    const char* e = getenv("QWENOX_MOE_NAIVE_MAX");
    return e ? atoi(e) : 16;
  }();
  // Default small-P down path; QWENOX_MOE_DOWN_ATOMIC restores the old
  // per-expert atomic accumulation for numerical/performance comparisons.
  bool moe_down_topk =
      moe_deterministic && getenv("QWENOX_MOE_DOWN_ATOMIC") == nullptr;
  bool moe_ordered = getenv("QWENOX_MOE_ORDERED") != nullptr;  // diagnostic: expert-order scatter
  bool moe_host_route = getenv("QWENOX_MOE_HOST_ROUTE") != nullptr;
  bool moe_untiled = getenv("QWENOX_MOE_UNTILED") != nullptr;
  bool moe_fp32_io = getenv("QWENOX_MOE_FP32_IO") != nullptr;
  bool moe_up_legacy = getenv("QWENOX_MOE_UP_LEGACY") != nullptr;
  bool moe_up_no_table = getenv("QWENOX_MOE_UP_NO_TABLE") != nullptr;
  bool moe_down_no_table = getenv("QWENOX_MOE_DOWN_NO_TABLE") != nullptr;
  bool moe_reduce_old = getenv("QWENOX_MOE_REDUCE_OLD") != nullptr;
  // bf16 pair rows from the deterministic down kernel; the router weight is
  // applied in fp32 by k_moe_reduce_pw_bf16 instead of pre-multiplied in fp32.
  bool moe_pairs_bf16 = getenv("QWENOX_MOE_PAIRS_BF16") != nullptr;
  // hgn q4cp prefill MoE (P > moe_naive_max) via the LUT-decode F16 WMMA kernels
  // (27_kernels_moe_lut.inc, same dataflow as moe_gguf). Default on,
  // QWENOX_MOE_Q4W=0 restores QWENOX_MOE_LT / tiled k_moe_w4. When on, QWENOX_MOE_LT is
  // ignored (no LT buffers); decode (P <= moe_naive_max) is untouched.
  bool moe_q4w = moe_q4w_env();
  bool moe_lt = getenv("QWENOX_MOE_LT") != nullptr && !moe_q4w;  // prefill MoE: dequant + per-expert hipBLASLt
  // Validated PP dataflow changes. Unset each switch to restore its baseline.
  // Keep the P<=8 direct GEMV path's input precision and scratch lifetime.
  bool pp_moe_out = getenv("QWENOX_PP_MOE_OUT") != nullptr;
  bool moe_lt_bf16 = getenv("QWENOX_MOE_LT_BF16") != nullptr;  // bf16 GEMM outputs (LT path)
  bool moe_lt_cmp = getenv("QWENOX_MOE_LT_CMP") != nullptr;  // debug: in-run fused reference diff
  bool moe_lt_ovl = getenv("QWENOX_MOE_LT_OVL") != nullptr;  // overlap next group's dequant with current GEMMs (LT path)
  // LT loses below ~4K tokens (512 tiny per-expert launches + the d_eoff sync
  // cost more than the GEMM wins; measured 2026-09-10: +0.7s @2K, wash @8K).
  int moe_lt_min = getenv("QWENOX_MOE_LT_MIN") ? atoi(getenv("QWENOX_MOE_LT_MIN")) : 4096;
  bool gdn_loop = getenv("QWENOX_GDN_LOOP") != nullptr;  // prefill GDN fallback
  bool gdn_nosplit = getenv("QWENOX_GDN_NOSPLIT") != nullptr;  // single-kernel GDN fallback
  bool gdn_nostrip = getenv("QWENOX_GDN_NOSTRIP") != nullptr;  // old inter (no strip)
  bool gdn_stream = getenv("QWENOX_GDN_STREAM") != nullptr;  // four-chunk workspace window
  int gdn_window_chunks = [] {
    const char* e = getenv("QWENOX_GDN_WINDOW_CHUNKS");
    return e ? std::max(1, atoi(e)) : 4;
  }();
  bool gdn_wave = getenv("QWENOX_GDN_WAVE") != nullptr;  // column-local triangular solve
  bool gdn_pipe = getenv("QWENOX_GDN_PIPE") != nullptr;  // intra/strip two-stream window pipeline (strip path)
  bool gdn_pipe2 = getenv("QWENOX_GDN_PIPE2") != nullptr;  // fp16-resident WMMA intra (chunk-looped)
  bool gdn_fused = getenv("QWENOX_GDN_FUSED") != nullptr;  // intra+strip fused, block-per-head persistent (no ws)
  bool gr_scat4 = getenv("QWENOX_GR_SCAT4") != nullptr;  // one-block-per-token GR scatter+norm
  bool gdn_ut5 = getenv("QWENOX_GDN_UT5") != nullptr;  // blocked triangular inverse
  bool ple_loop = getenv("QWENOX_PLE_LOOP") != nullptr;  // prefill PLE fallback
  bool prof = getenv("QWENOX_PROF") != nullptr;  // host-side phase timing
  double p_topk_wait = 0, p_bucket = 0, p_gemm = 0, p_pleh = 0,
         p_ple_wait = 0;
  // perf accumulators (ms): embed, ple, attn, moe, head
  double t_acc[5] = {0, 0, 0, 0, 0};
  double t_enq = 0, t_sync = 0;
  int t_n = 0;
  std::chrono::steady_clock::time_point tick() {
    if (!perf) return std::chrono::steady_clock::time_point{};
    CK(hipDeviceSynchronize());
    return std::chrono::steady_clock::now();
  }
  void tock(int slot, std::chrono::steady_clock::time_point t0) {
    if (!perf) return;
    CK(hipDeviceSynchronize());
    t_acc[slot] +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
  }
  void h_rms(const char* tag, int l, const char* sub, const float* p, int n) {
    if (!dbg) return;
    std::vector<float> h(n);
    CK(hipMemcpy(h.data(), (void*)p, n * 4, hipMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)h[i] * h[i];
    fprintf(stderr, "%s L%02d %s rms=%.4f [0..3]=%.5f %.5f %.5f %.5f\n", tag, l, sub,
            sqrt(s / n), h[0], h[1], h[2], h[3]);
  }
  // enqueue helpers — shared by direct execution and graph capture
  void enq_embed() {
    embed_rows(wview(ckpt.at("embed_tokens.weight")), d_token, 1, d_emb, 0);
    k_repeat4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_emb, d_R, g_cfg.d);
    h_rms("gpu", -1, "emb", d_emb, g_cfg.d);
  }
  void enq_layer(int l) {
    char prefix[128];
    snprintf(prefix, sizeof prefix, "layers.%d.attn_hyper_connection", l);
    auto t2 = tick();
    gr_read(prefix, d_x);
    if (is_qsa(l))
      qsa(l, l / 4, d_x, d_y);
    else
      gdn(l, l - (l + 1) / 4, d_x, d_y);
    h_rms("gpu", l, "attn-x", d_x, 2560);
    h_rms("gpu", l, "attn-y", d_y, 2560);
    gr_write(prefix, d_y);
    tock(2, t2);
    snprintf(prefix, sizeof prefix, "layers.%d.mlp_hyper_connection", l);
    auto t3 = tick();
    gr_read(prefix, d_x);
    moe(l, d_x, d_y);
    h_rms("gpu", l, "mlp-x", d_x, 2560);
    h_rms("gpu", l, "mlp-y", d_y, 2560);
    gr_write(prefix, d_y);
    tock(3, t3);
  }
  void enq_final() {
    gr_read("hyper_connection_mixer", d_x /*h*/);
    gemv(wview(ckpt.at("lm_head.weight")), d_x, d_logits);
    argmax_rows(d_logits, g_cfg.vocab, 1, d_amx_cand, d_argmax, g_str);
  }

  // =================== prefill batch (Phase 3a) ==============================
  // Batched GEMM: Y[P,N] = X[P,K] . W[N,K]^T. W dequanted to bf16 workspace,
  // X converted to bf16; rocBLAS col-major mapping: C_cm[N,P] with
  // A = W (transa=T, lda=K), B = X (transb=N, ldb=K), C = Y (ldc=N).
  // W -> bf16 for the LT / WMMA paths: dense bf16 is used in place, the
  // quantized dtypes dequant into the shared d_wbf16 scratch (<= 64 MiB).
  const uint16_t* gemm_wbf16(const WView& w) {
    uint64_t N = w.rows, K = w.cols;
    const uint16_t* wp = nullptr;
    if (w.dtype == 0) {
      wp = (const uint16_t*)w.data;
    } else if (w.dtype == 5) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 32;
      KP_SUB("ht_deq");
      k_dequant_q4cp_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          w.data + 64, w.data + 64 + N * K / 2, (const float*)w.data, d_wbf16, N, K,
          w.scale_stride);
      KP_SUB_END();
      wp = d_wbf16;
    } else if (w.dtype == 7) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 64;
      KP_SUB("ht_deq");
      k_dequant_q8g64_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          w.data, d_wbf16, N, K);
      KP_SUB_END();
      wp = d_wbf16;
    } else if (w.dtype == 8) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 32;
      KP_SUB("ht_deq");
      k_dequant_q8g32_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          (const int8_t*)w.data, (const __half*)(w.data + N * K), d_wbf16, N, K);
      KP_SUB_END();
      wp = d_wbf16;
    } else if (w.dtype == 16 || w.dtype == 24) {  // hgn v2 ht / q6g64
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      KP_SUB("ht_deq");
      v2_deq_bf16(w, d_wbf16, g_str);
      KP_SUB_END();
      wp = d_wbf16;
    } else {
      throw std::runtime_error("gemm: unsupported dtype");
    }
    return wp;
  }

  void gemm(const WView& w, const float* x, float* y, int P, uint64_t xstride = 0,
            const uint16_t* xbf = nullptr) {
    auto pt0 = std::chrono::steady_clock::now();
    uint64_t N = w.rows, K = w.cols;
    if (!xstride) xstride = K;
    // Small P: direct multi-row q4 gemv reads the weight once (~0.56x bytes
    // of the dequant->bf16->LT path's traffic) and skips 3 launches/gemm.
    // xbf inputs are bf16 with row stride K (the LT B layout).
    bool mr_done =
        xbf ? gemv_multi_q4cp(w, xbf, y, P, K, true)
            : gemv_multi_q4cp(w, x, y, P, xstride);
    if (mr_done) {
      if (prof)
        p_gemm += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pt0)
                      .count();
      return;
    }
    // A3 (§11.3): dense q4cp direct-read LUT for the d3/d9/dz/dq projection
    // shapes at P in [MINP, MAXP]. Harness ratios (deq+gemm_wmma)/lut with the
    // L2-hot dequant lower bound: d3 (2560,6144) 1.25/1.13 at P=1024/2048,
    // crossover ~4000; d9 (10240,2560) 1.11 at P=1024, crossover ~1500
    // (logs/a3_dense_lut.md); dz (6144,2560) 1.18@1024 crossover ~2000, dq
    // (12288,2560) 1.12@1024 crossover ~1800. In-engine (32k@1024 KPROF):
    // ht_deq -24 ms/chunk, gdn:proj -15, qsa:proj -4, qsa:oproj ~0 — the
    // engine's k1 WMMA config for (2560,6144) is ~1.3x faster than the
    // harness's old-d3 reference, so the true d3 ratio vs the OFF path is
    // ~1.0, not 1.25; end-to-end +1.8% < the 3% default-on bar, so MAXP
    // defaults are 0 (off). Opt in with QWENOX_DENSE_LUT_MAXP_{D3,D9,DZ,DQ}
    // (80%-of-crossover values 3200/1200/1600/1400). The weight runs as a
    // single-"expert" moe_lut_down with plain 64-slot tiles — the MoE-mt
    // BN=128 path (v2 MoE station) is not involved; the activation enters as
    // f16 (xbf calls: exact bf16->f16 of the live gated/normed input — the
    // f32 x there is the stale pre-gate/norm pre-image; f32 calls:
    // k_f32_to_f16_v4 rounding as in the MoE LUT station) — NOT bit-identical
    // vs the bf16 path, non-bit-exact by design, KLD-gated (c8192@4096:
    // mean 0.121637 vs off 0.121246, +0.0004 << 0.002). P<1024 keeps the Lt
    // fallback (MINP); =0 disables.
    static const int dlut_minp = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MINP");
      return s && *s ? atoi(s) : 1024;
    }();
    static const int dlut_d3 = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MAXP_D3");
      return s && *s ? atoi(s) : 0;
    }();
    static const int dlut_d9 = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MAXP_D9");
      return s && *s ? atoi(s) : 0;
    }();
    static const int dlut_dz = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MAXP_DZ");
      return s && *s ? atoi(s) : 0;
    }();
    static const int dlut_dq = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MAXP_DQ");
      return s && *s ? atoi(s) : 0;
    }();
    if (w.dtype == 5 && P >= dlut_minp && P <= maxbatch_cap && !(N % 128) &&
        K % 4 == 0 && xstride % 4 == 0 && (((uintptr_t)x & 15) == 0) &&
        (!xbf || ((((uintptr_t)xbf & 15) == 0) && K % 8 == 0))) {
      const int dlut_maxp = (N == 2560 && K == 6144) ? dlut_d3
                          : (N == 10240 && K == 2560) ? dlut_d9
                          : (N == 6144 && K == 2560) ? dlut_dz
                          : (N == 12288 && K == 2560) ? dlut_dq : 0;
      const moelut::LutW pdn =
          lut_q4cp_view(w.data, N, K, w.scale_stride, (int)N, 0);
      if (dlut_maxp && P <= dlut_maxp &&
          moelut::lut_ok(moelut::kQ4CP, pdn, false, (int)K)) {
        static bool dllog = false;
        if (!dllog) {
          dllog = true;
          fprintf(stderr,
                  "dense lut: q4cp d3<=%d d9<=%d dz<=%d dq<=%d minp=%d "
                  "(QWENOX_DENSE_LUT_MAXP_D3/D9/DZ/DQ, QWENOX_DENSE_LUT_MINP)\n",
                  dlut_d3, dlut_d9, dlut_dz, dlut_dq, dlut_minp);
        }
        const int nt = (P + 63) / 64;
        k_dense_tiles<<<1, 1024, 0, g_str>>>(d_dlut_tiles, d_dlut_nt, P);
        // x16 借用 d_xbf16b。xbf 调用点（qsa:oproj、fused gdn:oproj）的 f32 x
        // 是 gate/norm 前的旧值——活输入是 xbf（bf16, packed [P][K]），原地
        // bf16→f16（值精确）；非 xbf 调用点 x 即活 f32 输入，直接转。
        __half* x16 = (__half*)d_xbf16b;
        if (xbf) {
          const size_t n8 = (size_t)P * K / 8;
          k_bf16_to_f16_v8<<<(unsigned)((n8 + 255) / 256), 256, 0, g_str>>>(
              xbf, x16, n8);
        } else if (xstride == (uint64_t)K) {
          const size_t n4 = (size_t)P * K / 4;
          k_f32_to_f16_v4<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(
              x, x16, n4);
        } else {
          const size_t n4 = (size_t)P * (K / 4);
          k_f32_to_f16_v4s<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(
              x, x16, xstride, P, (int)K);
        }
        moe_lut_down(moelut::kQ4CP, pdn, x16, d_dlut_tiles, d_dlut_nt, nt, y,
                     (int)N, (int)K, g_str);
        if (prof)
          p_gemm += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pt0)
                        .count();
        return;
      }
    }
    static const bool gemm_dbg = getenv("QWENOX_GEMM_DBG") != nullptr;
    if (gemm_dbg)
      fprintf(stderr, "gemm-LT: dtype=%d N=%llu K=%llu P=%d xbf=%d xs=%llu\n",
              w.dtype, (unsigned long long)N, (unsigned long long)K, P,
              (int)(xbf != nullptr), (unsigned long long)xstride);
    const uint16_t* wp = gemm_wbf16(w);
    uint64_t tot = (uint64_t)P * K;
    bool residual_ready =
        !xbf && prefill_fused && x == d_Rhatb && K == 10240 && xstride == K;
    bool cacheable = (x == d_xb || x == d_Rhatb);
    // Non-cacheable inputs convert into d_xbf16b, leaving the d_xb/d_Rhatb
    // cache entries (xf16_*) untouched; xbf bypasses conversion entirely.
    uint16_t* dst = cacheable ? d_xbf16 : d_xbf16b;
    const uint16_t* xp = xbf ? xbf : residual_ready ? d_Rhatbf16 : dst;
    if (!xbf && !residual_ready &&
        !(cacheable && xf16_src == x && xf16_xs == xstride && xf16_P == P &&
          xf16_K == (int)K)) {
      if (K % 4 == 0 && xstride % 4 == 0 &&
          (((uintptr_t)x | (uintptr_t)dst) & 15) == 0)
        k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
            x, dst, xstride, P, (int)K);
      else
        k_f32_to_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
            x, dst, xstride, P, (int)K);
      if (cacheable) {
        xf16_src = x;
        xf16_xs = xstride;
        xf16_P = P;
        xf16_K = (int)K;
      }
    }
    float one = 1.f, zero = 0.f;
    if (getenv("QWENOX_GEMM_RB")) {  // fallback: rocBLAS default algo
      rocblas_set_stream(rbh, g_str);
      rocblas_status st = rocblas_gemm_ex(
          rbh, rocblas_operation_transpose, rocblas_operation_none, (rocblas_int)N,
          (rocblas_int)P, (rocblas_int)K, &one, wp, rocblas_datatype_bf16_r,
          (rocblas_int)K, xp, rocblas_datatype_bf16_r, (rocblas_int)K, &zero, y,
          rocblas_datatype_f32_r, (rocblas_int)N, y, rocblas_datatype_f32_r,
          (rocblas_int)N, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
      if (st != rocblas_status_success) {
        fprintf(stderr, "rocblas_gemm_ex failed: %d (N=%llu K=%llu P=%d)\n", (int)st,
                (unsigned long long)N, (unsigned long long)K, P);
        exit(1);
      }
      if (prof)
        p_gemm += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pt0)
                      .count();
      return;
    }
    // Phase 3g: self-written bf16 WMMA GEMM (QWENOX_GEMM_WMMA=1). Diverts the
    // winning projection shapes at P >= 1024 to k_gemm_wmma — d3 config
    // (512t w2x8 f4x2, gm=16) for N=2560 K=6144, d9 (256t w2x4 f4x4) for
    // N=6144/10240/12288 K=2560 (gm=4) and the narrow-K/N winners
    // N=10240 K=320 / N=512 K=2560 / N=2560 K=640 (gm=2, 1.25-2.0x vs Lt in
    // tools/gemm_lt_bench.cu); every other shape and the default flag-off
    // state fall through to hipBLASLt unchanged. Purely async. The kernel
    // requires K % 32 == 0; P and N tails are handled inside the kernel
    // (staging row clamps + predicated epilogue; the grid must be ceiled to
    // match). (640,2560) was measured and stays on Lt — 1.08-1.13x, under
    // the 1.15x bar. KST=64 family (128B per row per staging step, the
    // tools/gr320_proto.cu lesson; re-swept 2026-09-24 with
    // tools/gemm_lt_bench.cu at P=8192/16384, gm=1 best for both):
    //  - (320,10240) GR mix: h1 BM=64 BP=160 (num_n=2 exact), 3.9 ms vs
    //    5.3 ms Lt idx4 (x1.35); LDS 65,152B of 65,536 — any layout growth
    //    in the kernel breaks it.
    //  - (2560,6144) out-proj: k1 BM=64 BP=128, 16.4 ms vs 21.1-21.6 ms for
    //    the old d3 512t config and Lt (x1.3).
    // The K=2560 d9 shapes and (2560,640) stay on d9 (KST=64 configs lose
    // there). QWENOX_GEMM_NO_K64 restores the old routing (Lt / d3) for A/B.
    // QWENOX_GEMM_RB above takes precedence if both are set.
    static const bool gemm_wmma = getenv("QWENOX_GEMM_WMMA") != nullptr;
    static const bool gemm_k64 = getenv("QWENOX_GEMM_NO_K64") == nullptr;
    if (gemm_dbg)
      fprintf(stderr, "gemm-ptr: N=%llu K=%llu P=%d wp=%p xp=%p y=%p\n",
              (unsigned long long)N, (unsigned long long)K, P,
              (const void*)wp, (const void*)xp, (void*)y);
    if (gemm_wmma && P >= 1024) {
      if (gemm_k64 && ((N == 320 && K == 10240) || (N == 2560 && K == 6144))) {
        if (N == 320)
          k_gemm_wmma<64, 160, 2, 2, 2, 5, 64, 128>
              <<<(unsigned)(((P + 63) / 64) * 2), 128, 0, g_str>>>(
                  xp, wp, y, P, (int)K, (int)N, 1);
        else if (gemm_splitk_on() && P >= 8192 && d_skx) {
          // B3 split-K：X 重排成 2 个连续 K=3072 切片（W 维持原布局 strided
          // 读），slice0 写 Y，slice1 Epi=3 原子加。16K 下单 GEMM 省 ~1.5 ms
          //（proto x1.13 扣除 repack 0.23 ms）。P<8192 收益消失（x1.01）。
          const int ks = 3072;
          const unsigned grid = (unsigned)(((P + 63) / 64) * 20);
          k_splitk_repack2<<<1024, 256, 0, g_str>>>(xp, d_skx, P, (int)K);
          k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
              <<<grid, 128, 0, g_str>>>(d_skx, wp, y, P, ks, (int)N, 1,
                                        nullptr, nullptr, nullptr, nullptr, 0,
                                        (unsigned)K);
          k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128, 3>
              <<<grid, 128, 0, g_str>>>(d_skx + (size_t)P * ks, wp + ks, y, P,
                                        ks, (int)N, 1, nullptr, nullptr,
                                        nullptr, nullptr, 0, (unsigned)K);
        } else
          k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
              <<<(unsigned)(((P + 63) / 64) * 20), 128, 0, g_str>>>(
                  xp, wp, y, P, (int)K, (int)N, 1);
        if (prof)
          p_gemm += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pt0)
                        .count();
        return;
      }
      const bool gw_d3 = (N == 2560 && K == 6144);
      static const bool d9_old0 = getenv("QWENOX_GEMM_D9_OLD") != nullptr;
      // aggr: N=640 (shared gate/up, indexer proj; was hipBLASLt ~1.7 ms @16K)
      // and N=512 join the 128x128 tile route (1.33 / 1.06 ms @16K).
      const bool gw_d9 = (K == 2560 && (N == 6144 || N == 10240 || N == 12288 ||
                                        (!d9_old0 && (N == 640 || N == 512))));
      const bool gw_d9n = !gw_d9 && ((N == 10240 && K == 320) ||
                                     (N == 512 && K == 2560) || (N == 2560 && K == 640));
      if (gw_d3 || gw_d9 || gw_d9n) {
        const unsigned gw_grid =
            (unsigned)(((P + 127) / 128) * (int)((N + 255) / 256));
        // aggr 10-05 (tools/gemm_tune d9 @16384): 128x128 w2x2 f4x4 128-thread
        // tile beats the 128x256 d9 one — qkv N=10240 22.0 vs 22.8 ms (gm=4),
        // z N=6144 12.8 vs 13.8 ms (gm=1). QWENOX_GEMM_D9_OLD=1 restores.
        static const bool d9_old = getenv("QWENOX_GEMM_D9_OLD") != nullptr;
        if (gw_d9 && !d9_old) {
          const unsigned g2 = (unsigned)(((P + 127) / 128) * (int)((N + 127) / 128));
          k_gemm_wmma<128, 128, 2, 2, 4, 4, 32, 128>
              <<<g2, 128, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N,
                                      (N == 10240 || N == 12288) ? 4 : 1);
        } else if (gw_d3)
          k_gemm_wmma<128, 256, 2, 8, 4, 2, 32, 512>
              <<<gw_grid, 512, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N, 16);
        else if (gw_d9)
          k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256>
              <<<gw_grid, 256, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N, 4);
        else
          k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256>
              <<<gw_grid, 256, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N, 2);
        if (prof)
          p_gemm += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pt0)
                        .count();
        return;
      }
    }
    // hipBLASLt path: ~30 TFLOPS vs ~5.6 for the rocBLAS default algo on
    // gfx1151. The heuristic's rank order is misleading — index 4 is the fast
    // tensile kernel for every projection shape measured (5-6x over index 0),
    // so use it after a validation run; fall back to a timing scan if the
    // list is short or index 4 fails. Full descriptor set cached per (N,K,P).
    uint64_t key = (N << 40) ^ (K << 20) ^ (uint64_t)P;
    auto it = lt_algos.find(key);
    if (it == lt_algos.end()) {
      LtEntry e;
      hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
      hipblasLtMatrixLayoutCreate(&e.Al, HIP_R_16BF, K, N, K);
      hipblasLtMatrixLayoutCreate(&e.Bl, HIP_R_16BF, K, P, K);
      hipblasLtMatrixLayoutCreate(&e.Cl, HIP_R_32F, N, P, N);
      hipblasLtMatrixLayoutCreate(&e.Dl, HIP_R_32F, N, P, N);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
          sizeof(lt_ws_bytes));
      hipblasLtMatmulHeuristicResult_t hres[8];
      int nres = 0;
      hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, e.Bl, e.Cl, e.Dl, pref,
                                      8, hres, &nres);
      auto try_run = [&](int i) {
        hipblasLtMatmul(lth, e.opd, &one, wp, e.Al, xp, e.Bl, &zero, y, e.Cl, y,
                        e.Dl, &hres[i].algo, d_ltws, lt_ws_bytes, g_str);
        if (hipStreamSynchronize(g_str) != hipSuccess) {
          (void)hipGetLastError();
          return false;
        }
        return true;
      };
      int bi = -1;
      if (nres > 4 && hres[4].state == HIPBLAS_STATUS_SUCCESS &&
          hres[4].workspaceSize <= lt_ws_bytes && try_run(4))
        bi = 4;
      else {  // fallback: time all candidates
        double best = 1e30;
        for (int i = 0; i < nres; i++) {
          if (hres[i].state != HIPBLAS_STATUS_SUCCESS) continue;
          if (hres[i].workspaceSize > lt_ws_bytes) continue;
          if (!try_run(i)) continue;
          auto t0 = std::chrono::steady_clock::now();
          for (int r = 0; r < 3; r++) try_run(i);
          double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
          if (ms < best) {
            best = ms;
            bi = i;
          }
        }
      }
      if (bi < 0) {
        fprintf(stderr, "hipBLASLt: no algo for N=%llu K=%llu P=%d\n",
                (unsigned long long)N, (unsigned long long)K, P);
        exit(1);
      }
      e.algo = hres[bi].algo;
      hipblasLtMatmulPreferenceDestroy(pref);
      it = lt_algos.emplace(key, e).first;
    }
    const LtEntry& e = it->second;
    hipblasStatus_t lst =
        hipblasLtMatmul(lth, e.opd, &one, wp, e.Al, xp, e.Bl, &zero, y, e.Cl, y,
                        e.Dl, &e.algo, d_ltws, lt_ws_bytes, g_str);
    if (lst != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "hipblasLtMatmul failed: %d (N=%llu K=%llu P=%d)\n", (int)lst,
              (unsigned long long)N, (unsigned long long)K, P);
      exit(1);
    }
    if (prof)
      p_gemm += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - pt0)
                    .count();
  }

  // gemm variant for inputs that are already bf16 was used by the 3a' grouped
  // MoE path; removed with it (3d-1 fused W4 kernels replace it).

  // bf16 A (weights, [N][K] k-contiguous -> op T) x bf16 B ([P][K] k-contiguous
  // -> op N) -> C[N,P] (n-contiguous), fp32 or bf16 out. Used by QWENOX_MOE_LT
  // per-expert GEMMs. Algo cached per (N,K,P,outbf16); on miss trust heuristic
  // index 4 (the fast Tensile kernel for every projection shape measured in
  // tools/moe_lt_bench.cu scan mode; per-M validation syncs would stall the
  // per-layer expert loop). Bit 62 keeps keys disjoint from dense gemm's.
  void gemm_bf16(const __hip_bfloat16* A, const __hip_bfloat16* B, void* C,
                 int N, int K, int P, bool outbf16) {
    float one = 1.f, zero = 0.f;
    uint64_t key = (1ULL << 62) ^ ((uint64_t)outbf16 << 63) ^ ((uint64_t)N << 40) ^
                   ((uint64_t)K << 20) ^ (uint64_t)P;
    auto it = lt_algos.find(key);
    hipDataType dtout = outbf16 ? HIP_R_16BF : HIP_R_32F;
    if (it == lt_algos.end()) {
      LtEntry e;
      hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
      hipblasLtMatrixLayoutCreate(&e.Al, HIP_R_16BF, K, N, K);
      hipblasLtMatrixLayoutCreate(&e.Bl, HIP_R_16BF, K, P, K);
      hipblasLtMatrixLayoutCreate(&e.Cl, dtout, N, P, N);
      hipblasLtMatrixLayoutCreate(&e.Dl, dtout, N, P, N);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
          sizeof(lt_ws_bytes));
      hipblasLtMatmulHeuristicResult_t hres[8];
      int nres = 0;
      hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, e.Bl, e.Cl, e.Dl, pref,
                                      8, hres, &nres);
      if (nres <= 0) {
        fprintf(stderr, "moe_lt: no algo for N=%d K=%d P=%d\n", N, K, P);
        exit(1);
      }
      int bi = (nres > 4 && hres[4].state == HIPBLAS_STATUS_SUCCESS &&
                hres[4].workspaceSize <= lt_ws_bytes)
                   ? 4
                   : 0;
      e.algo = hres[bi].algo;
      hipblasLtMatmulPreferenceDestroy(pref);
      it = lt_algos.emplace(key, e).first;
    }
    const LtEntry& e = it->second;
    hipblasStatus_t lst =
        hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl, &zero, C, e.Cl, C,
                        e.Dl, &e.algo, d_ltws, lt_ws_bytes, g_str);
    if (lst != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "moe_lt: hipblasLtMatmul failed: %d (N=%d K=%d P=%d)\n",
              (int)lst, N, K, P);
      exit(1);
    }
  }

  // #51 HC gate fusion: the mix-up GEMM (N=10240 K=320) with the
  // k_gr_combine_b_hc_bf16 epilogue folded in (k_gemm_wmma Epi=1, W rows
  // permuted at LDS staging so one lane holds the 4 branch gates of its
  // channel). Same X and W bf16 prep as the gw_d9n route in gemm(); the
  // tile / column->block assignment changes but not the K order -> G and x are
  // bit-identical, and the P x 10240 fp32 G never goes to DRAM. Writes d_xb
  // (fp32) + d_xbf16 exactly like the unfused pair. Returns false (caller
  // runs the unfused pair) unless that exact route would have been taken.
  // Opt-out QWENOX_HC_FUSE=0 (tools/hcmix_verify.sh).
  // B2-2 (QWENOX_HC_NORHAT=1 显式开启，默认关): sc 非空时 Rh 实参应是未归一化
  // 的 d_Rb，epilogue 现场乘 (R*sc)*(1+Hw)（见 k_gemm_wmma Epi=1 注释），Rhat
  // 不落盘。终判否决（2026-10-03，128k@16384 背靠背 1175.7 vs 1216.4 = -3.3%，
  // KPROF hc+hc2 +293ms/chunk）：Epi=2 staging 归一化的 VALU + Sc/Hw 额外读
  // 超过省下的 Rhat 落盘带宽；正确性全过（c8192 0.08129 vs 0.08249），代码留存。
  bool hc_norhat(int P) const {
    static const bool on = getenv("QWENOX_HC_NORHAT") != nullptr &&
                           !env_zero("QWENOX_HC_NORHAT") &&
                           !env_zero("QWENOX_HC_FUSE") &&
                           getenv("QWENOX_GEMM_WMMA") != nullptr &&
                           getenv("QWENOX_GEMM_RB") == nullptr &&
                           getenv("QWENOX_GEMM_NO_K64") == nullptr;
    return on && prefill_fused && gr_bf16 && P >= 1024 && g_cfg.branches == 4 &&
           (uint64_t)g_cfg.d * 4 == 10240 &&
           ((((uintptr_t)d_t320b | (uintptr_t)d_xbf16b) & 15) == 0);
  }
  bool hc_up_fused(const WView& w, int P, const float* sc = nullptr,
                   const float* hw = nullptr, bool x_ready = false) {
    static const bool on = !env_zero("QWENOX_HC_FUSE") &&
                           getenv("QWENOX_GEMM_WMMA") != nullptr &&
                           getenv("QWENOX_GEMM_RB") == nullptr;
    const uint64_t N = w.rows, K = w.cols;
    if (!on || P < 1024 || N != 10240 || K != 320 || g_cfg.branches != 4 ||
        (uint64_t)g_cfg.d * 4 != N)
      return false;
    if ((((uintptr_t)d_t320b | (uintptr_t)d_xbf16b) & 15) != 0) return false;
    auto pt0 = std::chrono::steady_clock::now();
    const uint16_t* wp = gemm_wbf16(w);
    const uint64_t tot = (uint64_t)P * K;
    if (!x_ready)  // aggr: k_hc_mixcomb already wrote the bf16 X into d_xbf16b
      k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
          d_t320b, d_xbf16b, K, P, (int)K);
    // 128x128 w4x2 f2x4 (tools/hcmix_proto.cu tile sweep): the fused epilogue
    // favours the narrower tile — 2.33 vs 2.80 ms at P=8192 (pair 4.92 ms),
    // still bit-identical (K accumulation order is tile-independent).
    const unsigned grid = (unsigned)(((P + 127) / 128) * (int)(N / 128));
    k_gemm_wmma<128, 128, 4, 2, 2, 4, 32, 256, 1><<<grid, 256, 0, g_str>>>(
        d_xbf16b, wp, d_xb, P, (int)K, (int)N, 2,
        sc ? (const uint16_t*)d_Rb : d_Rhatbf16, d_xbf16, sc, hw);
    if (prof)
      p_gemm += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - pt0)
                    .count();
    return true;
  }

  void gr_mix_b(const std::string& prefix, int P, float* hw = nullptr,
                bool norhat = false) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    static const bool mixsplit = !env_zero("QWENOX_HC_MIXSPLIT");
    const WView wdn = wview(ckpt.at(buf));
    if (norhat && mixsplit && wdn.dtype != 0 && wdn.rows == 320 && wdn.cols == 10240) {
      // aggr 10-05: branch-split mix-down. W' = W*(1+hw) (dequant scratch,
      // scaled in place), 4 K=2560 slice GEMMs in one launch on the raw R'
      // (blockIdx.y = branch, partials in the fused path's idle d_Gb), then
      // t = silu(0.25 * sum_b sc_b * part_b) straight to bf16 for the up GEMM.
      // Not bit-identical to the Rhat path (bf16 W' vs bf16 Rhat rounding).
      uint16_t* wp = (uint16_t*)gemm_wbf16(wdn);
      k_scale_cols_bf16<<<640, 256, 0, g_str>>>(wp, hw, 320, 10240);
      float* parts = d_Gb;
      k_gemm_wmma<64, 160, 2, 2, 2, 5, 64, 128>
          <<<dim3((unsigned)(((P + 63) / 64) * 2), 4), 128, 0, g_str>>>(
              (const uint16_t*)d_Rb, wp, parts, P, 2560, 320, 1, nullptr, nullptr,
              nullptr, nullptr, 10240, 10240, 2560, 2560, (size_t)P * 320);
      k_hc_mixcomb<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(parts, d_hcsc,
                                                            d_xbf16b, P, 320);
      const WView wup = wview(ckpt.at(std::string(prefix) + ".input_mix_weight_up.weight"));
      if (!hc_up_fused(wup, P, d_hcsc, hw, true)) {
        fprintf(stderr, "hc_norhat: up-fused route fell back with Rhat unwritten\n");
        exit(1);
      }
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
      return;
    }
    if (norhat) {
      // B2-2: mix-down 的 A = 未归一化的 d_Rb，Epi=2 在 staging 现场归一化
      //（与 rmsnorm 同一表达式 -> A tile 与被取代的 Rhat 逐 bit）。路线条件与
      // gemm() 的 (320,10240) k64 分流一致（hc_norhat 已核）。
      auto pt0 = std::chrono::steady_clock::now();
      const uint16_t* wp = gemm_wbf16(wview(ckpt.at(buf)));
      k_gemm_wmma<64, 160, 2, 2, 2, 5, 64, 128, 2>
          <<<(unsigned)(((P + 63) / 64) * 2), 128, 0, g_str>>>(
              (const uint16_t*)d_Rb, wp, d_t320b, P, 10240, 320, 1, nullptr,
              nullptr, d_hcsc, hw);
      if (prof)
        p_gemm += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pt0)
                      .count();
    } else {
      gemm(wview(ckpt.at(buf)), d_Rhatb, d_t320b, P, 0,
           gr_bf16 ? d_Rhatbf16 : nullptr);
    }
    k_silu_scale<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(d_t320b, 0.25f, P * 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    const WView wup = wview(ckpt.at(buf));
    if (norhat) {
      // hc_norhat 与 hc_up_fused 条件逐一对应；仍失败说明 Rhat 未落盘而
      // 后续读者存在——不能静默走旧路（会读 stale Rhat），直接 fatal。
      if (!hc_up_fused(wup, P, d_hcsc, hw)) {
        fprintf(stderr, "hc_norhat: up-fused route fell back with Rhat unwritten\n");
        exit(1);
      }
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
      return;
    }
    if (prefill_fused && gr_bf16 && hc_up_fused(wup, P)) {
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
      return;
    }
    gemm(wup, d_t320b, d_Gb, P);
    if (prefill_fused) {
      if (gr_bf16) {
        // verify sizes: 4 column slices over P*4 blocks (per-column arithmetic
        // single-thread, unchanged). QWENOX_COMBINE_CS=0 keeps P blocks.
        static const bool comb_cs = !env_zero("QWENOX_COMBINE_CS");
        if (comb_cs && P <= 8)
          k_gr_combine_b_hc_bf16<<<dim3(P, 4), 256, 0, g_str>>>(
              d_Gb, d_Rhatbf16, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16,
              g_cfg.d / 4);
        else
          k_gr_combine_b_hc_bf16<<<P, 256, 0, g_str>>>(
              d_Gb, d_Rhatbf16, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16,
              g_cfg.d);
      } else
        k_gr_combine_b<true><<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
            d_Gb, d_Rhatb, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16);
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
    } else {
      k_sigmoid_inplace<<<(P * 10240 + 255) / 256, 256, 0, g_str>>>(d_Gb, P * 10240);
      k_gr_combine_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_Gb, d_Rhatb, d_xb, g_cfg.d, g_cfg.branches, P);
      if (xf16_src == d_xb) xf16_src = nullptr;
    }
  }

  void gr_read_b(const std::string& prefix, int P) {
    float* hw = ensure_hcnorm(prefix);
    if (gr_bf16)
      k_rmsnorm_zc_grouped_bf16<<<4 * P, 1024, 0, g_str>>>(
          (const uint16_t*)d_Rb, hw, d_Rhatbf16, 2560, g_cfg.norm_eps, 4);
    else if (prefill_fused)
      k_rmsnorm_zc_grouped<true><<<4 * P, 1024, 0, g_str>>>(
          d_Rb, hw, d_Rhatb, 2560, g_cfg.norm_eps, 4, d_Rhatbf16);
    else
      k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(
          d_Rb, hw, d_Rhatb, 2560, g_cfg.norm_eps, 4);
    if (xf16_src == d_Rhatb) xf16_src = nullptr;  // d_Rhatb overwritten
    injp_cur = -1;  // new Rhat without inject partials
    gr_mix_b(prefix, P);
  }

  // HC inject fusion (k_gr_scatter_norm_inj_bf16): the scatter+norm that
  // produces a site's Rhat also emits that site's inject projection as
  // per-branch partials, so the later N=4 inject GEMM (which re-read the
  // whole P x 10240 bf16 Rhat, ~0.97 ms at P=8192) is skipped. R / Rhat are
  // bit-identical; w4 differs in summation order only (fp32, err/sum|terms|
  // ~1e-8 vs ~1e-7 for any sequential fp32 dot; tools/injfuse_proto.cu).
  // Opt-out QWENOX_HC_INJ_FUSE=0 (tools/injfuse_verify.sh: KLD + speed).
  bool inj_fuse_on() const {
    static const bool on = !env_zero("QWENOX_HC_INJ_FUSE");
    return on && gr_bf16 && prefill_fused && !gr_scat4 && g_cfg.d <= 2560 &&
           g_cfg.branches == 4;
  }
  // Private bf16 copy of a site's inject weight (4 x 4d). gemm_wbf16 dequants
  // quantized weights into the shared d_wbf16 scratch, which later GEMMs
  // overwrite, so quantized ones are copied out once (~80 KB per site).
  const uint16_t* inj_wbf16(const std::string& prefix) {
    auto it = inj_wcache_.find(prefix);
    if (it != inj_wcache_.end()) return it->second;
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    const auto* ct = ckpt.find(buf);
    if (!ct) return inj_wcache_[prefix] = nullptr;
    const WView w = wview(*ct);
    if ((uint64_t)w.rows != 4 || (uint64_t)w.cols != (uint64_t)4 * g_cfg.d ||
        !(w.dtype == 0 || w.dtype == 5 || w.dtype == 7 || w.dtype == 8 || w.dtype == 24))
      return inj_wcache_[prefix] = nullptr;
    const uint16_t* src = gemm_wbf16(w);
    uint16_t* dst = (uint16_t*)src;
    if (w.dtype != 0) {
      CK(hipMalloc(&dst, (size_t)w.rows * w.cols * 2));
      CK(hipMemcpyAsync(dst, src, (size_t)w.rows * w.cols * 2,
                        hipMemcpyDeviceToDevice, g_str));
    }
    return inj_wcache_[prefix] = dst;
  }

  // First half of gr_write_b: make d_w4b for prefix (inject partials psum or
  // the N=4 GEMM). Bit-identical to what gr_write_b would compute; injp_cur
  // is left untouched so a non-fused fallback recomputes the same d_w4b.
  void gr_write_prep_b(const std::string& prefix, int P) {
    char buf[128];
    if (injp_cur >= 0 && injp_for == prefix) {
      k_inj_psum<<<(P * 4 + 255) / 256, 256, 0, g_str>>>(d_injp[injp_cur], d_w4b, P);
    } else {
      snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
      gemm(wview(ckpt.at(buf)), d_Rhatb, d_w4b, P, 0,
           gr_bf16 ? d_Rhatbf16 : nullptr);
    }
  }

  void gr_write_b(const std::string& prefix, int P) {
    const bool had_p = injp_cur >= 0 && injp_for == prefix;
    gr_write_prep_b(prefix, P);
    if (had_p) injp_cur = -1;
    if (gr_bf16)
      k_gr_write_b_hc_bf16<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_w4b, (uint16_t*)d_Rb, d_yb, g_cfg.d, g_cfg.branches, P);
    else
      k_gr_write_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, g_cfg.d, g_cfg.branches, P);
  }

  void gr_write_read_b(const std::string& write_prefix,
                       const std::string& read_prefix, int P) {
    char buf[128];
    // B2-1: the moe reduce already applied this site's write (R is
    // post-write); the scatter below only norms. pfull/pin go unused.
    const bool write_done = !hcw_site_.empty() && hcw_site_ == write_prefix;
    if (write_done) hcw_site_.clear();
    const bool have_p = injp_cur >= 0 && injp_for == write_prefix;
    if (!have_p && !write_done) {
      snprintf(buf, sizeof buf, "%s.block_inject_weight.weight",
               write_prefix.c_str());
      gemm(wview(ckpt.at(buf)), d_Rhatb, d_w4b, P, 0,
           gr_bf16 ? d_Rhatbf16 : nullptr);
    }
    float* hw = ensure_hcnorm(read_prefix);
    const uint16_t* wn = inj_fuse_on() ? inj_wbf16(read_prefix) : nullptr;
    if (!wn && write_done)  // R is already post-write; a writing scatter would double-apply
      throw std::runtime_error("gr_write_read_b: fused HC write but read site can't fuse");
    if (!wn && have_p && !write_done)  // partials exist but this site can't fuse (defensive)
      k_inj_psum<<<(P * 4 + 255) / 256, 256, 0, g_str>>>(d_injp[injp_cur], d_w4b, P);
    if (wn) {
      static bool said = false;
      if (!said) { said = true; fprintf(stderr, "[hc-inj-fuse] on (inject GEMM folded into scatter_norm)\n"); }
      const int nxt = injp_cur == 0 ? 1 : 0;  // pin != pout
      const float* pfull = (have_p || write_done) ? nullptr : d_w4b;
      const float* pin = (have_p && !write_done) ? d_injp[injp_cur] : nullptr;
      const float* y = write_done ? nullptr : d_yb;
      // B2-2: Rhat 不落盘，scatter 只写 P×4 inv scales（gr_read_b 站除外——
      // 那里的 Rhat 还要喂后续 inject GEMM，维持旧路，见 gr_read_b）。
      const bool nr = hc_norhat(P);
      uint16_t* const rh = nr ? nullptr : d_Rhatbf16;
      float* const sc = nr ? d_hcsc : nullptr;
      // Verify sizes (P<=8): 1 token/block -> 4P blocks instead of 4*ceil(P/4).
      // Per-(t,b) arithmetic is T-invariant (proto: R/Rhat bit-identical), the
      // 4-token serial loop was pure latency at 4 blocks. QWENOX_SCAT_T1=0: T=4.
      static const bool scat_t1 = !env_zero("QWENOX_SCAT_T1");
      // v2 (coalesced + pipelined): kernel −5% only (this op is at ~190 GB/s,
      // bandwidth-bound) and not bit-identical -> opt-in QWENOX_GR_SCAT_V2=1.
      static const bool scat_v2 = getenv("QWENOX_GR_SCAT_V2") && !env_zero("QWENOX_GR_SCAT_V2");
      static const bool scat_t8 = getenv("QWENOX_GR_SCAT_T8") != nullptr;
      if (scat_v2 && scat_t8 && g_cfg.d == 2560 && P > 8) {
        if (write_done)
          k_gr_scatter_norm_inj_bf16_v2<8, false><<<4 * ((P + 7) / 8), 256, 0, g_str>>>(
              nullptr, nullptr, (uint16_t*)d_Rb, nullptr, hw, rh, wn, d_injp[nxt], P,
              g_cfg.norm_eps, sc);
        else
          k_gr_scatter_norm_inj_bf16_v2<8><<<4 * ((P + 7) / 8), 256, 0, g_str>>>(
              pfull, pin, (uint16_t*)d_Rb, y, hw, rh, wn, d_injp[nxt], P,
              g_cfg.norm_eps, sc);
      } else if (scat_v2 && g_cfg.d == 2560 && P > 8) {
        if (write_done)
          k_gr_scatter_norm_inj_bf16_v2<4, false><<<4 * ((P + 3) / 4), 256, 0, g_str>>>(
              nullptr, nullptr, (uint16_t*)d_Rb, nullptr, hw, rh, wn, d_injp[nxt], P,
              g_cfg.norm_eps, sc);
        else
          k_gr_scatter_norm_inj_bf16_v2<4><<<4 * ((P + 3) / 4), 256, 0, g_str>>>(
              pfull, pin, (uint16_t*)d_Rb, y, hw, rh, wn, d_injp[nxt], P,
              g_cfg.norm_eps, sc);
      } else if (scat_t1 && P <= 8 && !write_done)
        k_gr_scatter_norm_inj_bf16<1><<<4 * P, 256, 0, g_str>>>(
            pfull, pin, (uint16_t*)d_Rb, y, hw, rh, wn, d_injp[nxt],
            g_cfg.d, P, g_cfg.norm_eps, sc);
      else if (write_done)
        k_gr_scatter_norm_inj_bf16<4, false><<<4 * ((P + 3) / 4), 256, 0, g_str>>>(
            nullptr, nullptr, (uint16_t*)d_Rb, nullptr, hw, rh, wn,
            d_injp[nxt], g_cfg.d, P, g_cfg.norm_eps, sc);
      else
        k_gr_scatter_norm_inj_bf16<4><<<4 * ((P + 3) / 4), 256, 0, g_str>>>(
            pfull, pin, (uint16_t*)d_Rb, y, hw, rh, wn, d_injp[nxt],
            g_cfg.d, P, g_cfg.norm_eps, sc);
      injp_cur = nxt;
      injp_for = read_prefix;
    } else if (gr_bf16)
      if (gr_scat4 && g_cfg.d <= 2560)
        k_gr_scatter_norm_b_hc_bf16_f4<<<P, 256, 0, g_str>>>(
            d_w4b, (uint16_t*)d_Rb, d_yb, hw, d_Rhatbf16, g_cfg.d,
            g_cfg.branches, P, g_cfg.norm_eps);
      else
        k_gr_scatter_norm_b_hc_bf16<<<4 * P, 256, 0, g_str>>>(
            d_w4b, (uint16_t*)d_Rb, d_yb, hw, d_Rhatbf16, g_cfg.d,
            g_cfg.branches, P, g_cfg.norm_eps);
    else if (prefill_fused)
      k_gr_scatter_norm_b<true><<<4 * P, 1024, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, hw, d_Rhatb, g_cfg.d, g_cfg.branches, P,
          g_cfg.norm_eps, d_Rhatbf16);
    else
      k_gr_scatter_norm_b<<<4 * P, 1024, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, hw, d_Rhatb, g_cfg.d, g_cfg.branches, P,
          g_cfg.norm_eps);
    if (!wn) injp_cur = -1;
    if (xf16_src == d_Rhatb) xf16_src = nullptr;
    gr_mix_b(read_prefix, P, hw, wn && hc_norhat(P));
  }

  // GDN: projections batched. Chunked path: conv/l2norm/gates batched, then
  // k_gdn_chunk (chunk=64, fp32) advances S and writes all outputs at once.
  // QWENOX_GDN_LOOP=1 falls back to the per-token recurrence loop.
  void gdn_b(int l, int gi, int P) {
    char buf[128];
    KP_SUB("gdn:proj");
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_qkv.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_qkvb, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_a.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_g48b, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_b.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_beta48b, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_zb, P);
    KP_SUB_END();
    float* convst = d_convst + (size_t)gi * 3 * 10240;
    const float* cw = d_convw + (size_t)gi * 10240 * 4;
    const float* Alog = d_Alog + (size_t)gi * 48;
    const float* dtb = d_dtbias + (size_t)gi * 48;
    float* S = d_S + (size_t)gi * 48 * 128 * 128;
    const float* gnw = d_gdnnorm + (size_t)gi * 128;
    float qscale = 1.f / sqrtf(128.f);
    if (!gdn_loop) {
      KP_SUB("gdn:conv");
      // conv+silu and the q/k l2norm in one pass (k_gdn_conv_l2n_b, bit-exact
      // vs the pair, tools/convl2_verify.sh); opt-out QWENOX_GDN_CONVL2=0.
      static const bool convl2 = !env_zero("QWENOX_GDN_CONVL2");
      const bool cl2 = convl2 && P <= 65535;
      int64_t tot = (int64_t)P * 10240 / 4;  // 4 channels per thread (float4)
      if (cl2)
        k_gdn_conv_l2n_b<<<dim3(10, (unsigned)P), 256, 0, g_str>>>(
            d_qkvb, cw, convst, d_convb, P, 10240, 16, qscale);
      else
        k_gdn_conv_b<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
            d_qkvb, cw, convst, d_convb, P, 10240);
      if (spec_snap) {
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        k_convst_snap<<<P, 512, 0, g_str>>>(
            d_qkvb, convst, d_csnap + (size_t)gi * 3 * 10240, P, 10240,
            (size_t)ngdn * 3 * 10240);
      }
      if (cut_r_)  // in-pass cut: conv state after the first r rows
        k_convst_update<<<(10240 + 255) / 256, 256, 0, g_str>>>(
            d_cconvst + (size_t)gi * 3 * 10240, d_qkvb, cut_r_, 10240);
      k_convst_update<<<(10240 + 255) / 256, 256, 0, g_str>>>(convst, d_qkvb, P,
                                                              10240);
      if (!cl2)
        k_l2norm_qk_b<<<dim3(16, P), 128, 0, g_str>>>(d_convb, 128, qscale, 10240, 16);
      k_gdn_gates<<<P, 64, 0, g_str>>>(d_g48b, d_beta48b, Alog, dtb, d_g48b, d_beta48b,
                                       48, 48);
      KP_SUB_END();
      if (spec_snap) {
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        k_gdn_snap<<<48, 512, 0, g_str>>>(
            d_convb, d_g48b, d_beta48b, S, d_Ssnap + (size_t)gi * 48 * 128 * 128,
            P, 10240, (size_t)ngdn * 48 * 128 * 128);
      }
      KP_SUB("gdn:scan");
      if (ngram_verify_active && !ngram_legacy_gdn && P <= 65) {
        k_gdn_verify<<<48, 512, 0, g_str>>>(
            d_convb, d_g48b, d_beta48b, S, d_convb, P, 10240);
      } else if (gdn_nosplit) {
        k_gdn_chunk<<<48, GDN_NT, 0, g_str>>>(d_convb, d_g48b, d_beta48b, S, d_convb,
                                              d_gdn_ws, P, 10240);
      } else if (gdn_fused) {
        // one launch for all of P: no window, no workspace; S in/out via global
        const int r = cut_r_;
        float* const cS = d_cS ? d_cS + (size_t)gi * 48 * 128 * 128 : nullptr;
        if (r > 0 && r < P) {
          // in-pass cut: scan [0, r), keep S(cut), scan [r, P) (chunk grid
          // restarts at r -> not bit-identical to the single launch)
          k_gdn_fused<<<48, 1024, 0, g_str>>>(d_convb, d_g48b, d_beta48b, S,
                                              d_convb, r, 10240);
          CK(hipMemcpyAsync(cS, S, (size_t)48 * 128 * 128 * 4, hipMemcpyDeviceToDevice, g_str));
          k_gdn_fused<<<48, 1024, 0, g_str>>>(
              d_convb + (size_t)r * 10240, d_g48b + (size_t)r * 48, d_beta48b + (size_t)r * 48,
              S, d_convb + (size_t)r * 10240, P - r, 10240);
        } else {
          k_gdn_fused<<<48, 1024, 0, g_str>>>(d_convb, d_g48b, d_beta48b, S,
                                              d_convb, P, 10240);
          if (r == P)
            CK(hipMemcpyAsync(cS, S, (size_t)48 * 128 * 128 * 4, hipMemcpyDeviceToDevice, g_str));
        }
      } else {
        // Keep producer and consumer close; S carries the exact recurrence
        // across windows and the last window alone can contain a padded chunk.
        int window = gdn_stream ? 64 * gdn_window_chunks : P;
        // QWENOX_GDN_PIPE (strip path): intra stays on g_str, strips run on a
        // side stream. ws is double-buffered by window parity: intra(w) waits
        // for strip(w-2) (its ws slot's previous reader), strip(w) waits for
        // intra(w) and stays stream-serial (the S recurrence). Event objects
        // are reused by parity; host enqueue order matches dependency order.
        const bool pipe = gdn_pipe && gdn_stream && !gdn_nostrip;
        static hipStream_t pipe_str = nullptr;
        static hipEvent_t ev_i[2] = {nullptr, nullptr}, ev_s[2] = {nullptr, nullptr};
        if (pipe && !pipe_str) {
          CK(hipStreamCreateWithFlags(&pipe_str, hipStreamNonBlocking));
          for (int i = 0; i < 2; i++) {
            CK(hipEventCreateWithFlags(&ev_i[i], hipEventDisableTiming));
            CK(hipEventCreateWithFlags(&ev_s[i], hipEventDisableTiming));
          }
        }
        const size_t ws_slot =
            (size_t)std::min<size_t>(gdn_window_chunks, (maxbatch_cap + 63) / 64) *
            48 * GDN_SPLIT_WS_FLOATS;
        int w = 0, lastw = -1;
        for (int first = 0; first < P; first += window, w++) {
          int count = std::min(window, P - first), nchunks = (count + 63) / 64;
          float* conv = d_convb + (size_t)first * 10240;
          float* ws = d_gdn_split_ws + (pipe ? (w & 1) * ws_slot : 0);
          if (pipe && w >= 2) CK(hipStreamWaitEvent(g_str, ev_s[w & 1], 0));
          if (gdn_pipe2)
            k_gdn_intra_p2<<<dim3((nchunks + GDN_PIPE2_NCPB - 1) / GDN_PIPE2_NCPB, 48),
                             1024, 0, g_str>>>(
                conv, d_g48b + first * 48, d_beta48b + first * 48,
                ws, count, 10240);
          else if (gdn_wave)
            if (gdn_ut5)
              k_gdn_intra<1024, true, true><<<dim3(nchunks, 48), 1024, 0, g_str>>>(
                  conv, d_g48b + first * 48, d_beta48b + first * 48,
                  ws, count, 10240);
            else
              k_gdn_intra<1024, true><<<dim3(nchunks, 48), 1024, 0, g_str>>>(
                  conv, d_g48b + first * 48, d_beta48b + first * 48,
                  ws, count, 10240);
          else
            k_gdn_intra<GDN_NT_INTRA><<<dim3(nchunks, 48), GDN_NT_INTRA, 0, g_str>>>(
                conv, d_g48b + first * 48, d_beta48b + first * 48,
                ws, count, 10240);
          if (!pipe) {
            if (gdn_nostrip)
              k_gdn_inter<GDN_NT_INTER><<<48, GDN_NT_INTER, 0, g_str>>>(
                  conv, S, conv, ws, count, 10240);
            else
              k_gdn_inter_strip<<<dim3(4, 48), GDN_NT_STRIP, 0, g_str>>>(
                  conv, S, conv, ws, count, 10240);
          } else {
            CK(hipEventRecord(ev_i[w & 1], g_str));
            CK(hipStreamWaitEvent(pipe_str, ev_i[w & 1], 0));
            k_gdn_inter_strip<<<dim3(4, 48), GDN_NT_STRIP, 0, pipe_str>>>(
                conv, S, conv, ws, count, 10240);
            CK(hipEventRecord(ev_s[w & 1], pipe_str));
            lastw = w;
          }
        }
        if (pipe) CK(hipStreamWaitEvent(g_str, ev_s[lastw & 1], 0));
      }
      KP_SUB_END();
      KP_SUB("gdn:norm");
      k_gdn_gatednorm_b<true><<<dim3(48, (P + 31) / 32), 128, 0, g_str>>>(
          d_convb + 4096, d_zb, gnw, nullptr, g_cfg.norm_eps, 10240, 6144,
          d_xbf16b, P);
      KP_SUB_END();
      KP_SUB("gdn:oproj");
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
      gemm(wview(ckpt.at(buf)), d_convb + 4096, d_yb, P, 10240 /*row stride*/,
           d_xbf16b);
      KP_SUB_END();
      return;
    }
    for (int t = 0; t < P; t++) {
      float* qkv = d_qkvb + (size_t)t * 10240;
      k_gdn_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(qkv, cw, convst, qkv, 10240);
      k_l2norm_qk<<<16, 128, 0, g_str>>>(qkv, qkv + 2048, 128, qscale);
      k_gdn_gates<<<1, 64, 0, g_str>>>(d_g48b + t * 48, d_beta48b + t * 48, Alog, dtb,
                                       d_g48b + t * 48, d_beta48b + t * 48, 48, 48);
      k_gdn_step<<<48, 512, 0, g_str>>>(qkv, qkv + 2048, qkv + 4096, d_g48b + t * 48,
                                        d_beta48b + t * 48, S, qkv + 4096, 128, 128, 48,
                                        16);
      k_gdn_gatednorm<<<dim3(48, 1), 128, 0, g_str>>>(qkv + 4096,
                                                      d_zb + (size_t)t * 6144, gnw,
                                                      qkv + 4096, 128, g_cfg.norm_eps,
                                                      10240, 6144);
    }
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_qkvb + 4096, d_yb, P, 10240 /*row stride*/);
  }

  // Debug (QWENOX_ATTN_HASH): per prefill chunk and QSA layer, hashes of the
  // attention inputs (K / V / transposed V over valid rows [0, base+P), the
  // transposed-V tail slots [base+P, roundup4), q, selections) and the output.
  void dbg_attn_hash(int qi, int base, int P, const uint16_t* kc, const uint16_t* vc,
                     const uint16_t* vct) {
    static unsigned long long* d_h = nullptr;
    if (!d_h) CK(hipMalloc(&d_h, 8 * sizeof(unsigned long long)));
    CK(hipMemsetAsync(d_h, 0, 8 * sizeof(unsigned long long), g_str));
    const int r1 = base + P, r2 = (r1 + 3) & ~3, first = std::max(base, 2051);
    auto g = [](uint64_t n) { return (unsigned)((n + 255) / 256); };
    k_dbg_hash_rows<<<g((uint64_t)r1 * 512), 256, 0, g_str>>>(kc, nullptr, kvpt(), 0, r1, d_h + 0);
    k_dbg_hash_rows<<<g((uint64_t)r1 * 512), 256, 0, g_str>>>(vc, nullptr, kvpt(), 0, r1, d_h + 1);
    if (vct) {
      k_dbg_hash_rows<<<g((uint64_t)r1 * 512), 256, 0, g_str>>>(nullptr, vct, kvpt(), 0, r1, d_h + 2);
      if (r2 > r1)
        k_dbg_hash_rows<<<g((uint64_t)(r2 - r1) * 512), 256, 0, g_str>>>(nullptr, vct, kvpt(), r1, r2,
                                                                         d_h + 3);
    }
    k_dbg_hash_f32<<<g((uint64_t)P * 6144), 256, 0, g_str>>>(d_qsb, (uint64_t)P * 6144, d_h + 4);
    if (base + P > first)
      k_dbg_hash_i32<<<g((uint64_t)(base + P - first) * 512), 256, 0, g_str>>>(
          d_iselected + (size_t)(first - base) * 512, (uint64_t)(base + P - first) * 512, d_h + 5);
    k_dbg_hash_f32<<<g((uint64_t)P * 6144), 256, 0, g_str>>>(d_attnb, (uint64_t)P * 6144, d_h + 6);
    unsigned long long h[8];
    CK(hipStreamSynchronize(g_str));
    CK(hipMemcpy(h, d_h, sizeof h, hipMemcpyDeviceToHost));
    fprintf(stderr,
            "[ahash] base=%d P=%d qsa%d K=%016llx V=%016llx Vt=%016llx tail=%016llx "
            "q=%016llx sel=%016llx out=%016llx\n",
            base, P, qi, h[0], h[1], h[2], h[3], h[4], h[5], h[6]);
  }

  // QSA prefill flash over the KV cache: dense prefix plus the sparse tail.
  // base = absolute position of batch row 0 (0 for a fresh prefill; > 0 when
  // continuing from live KV/state). Dense rows are the batch-local prefix,
  // sparse queries are the absolute positions >= 2051 within the batch.
  template <typename KVT>
  void qsa_flash_b(const KVT* kc, const KVT* vc, int P, int dense_P, int base = 0,
                   const uint16_t* vc_transposed = nullptr,
                   const int* ptab = nullptr, const int* dvb = nullptr) {
    // dvb (P3-B graph capture): the kernels read base from *dvb and recompute
    // first_abs / ga / ge on device. Requires base >= 2051 (the verify graph
    // gate), so first_abs == base, nsparse == P and sel is the batch-local
    // row-0 pointer; grids are baked worst-case (edges: 3 tokens, union
    // groups: max over base mod 4) and masked in-kernel.
    // ptab (paged KV) is honoured by k_qsa_flash (all KV types) and, for the
    // BF16 cache, by k_qsa_wmma<*, true> (A1c). The union / wmma6 /
    // k_qsa_flash_bf16 paths are not paged: the constructor refuses paging
    // for those configurations.
    if (dense_P > 0) {
      // F1 (§11.1)：bf16 KV + qsa_wmma 时稠密前缀也走 k_qsa_wmma 的 Dense 变体
      // （grid=(dense_P,2)，first=kbase=base，selected 不用）。fp32 KV、
      // QWENOX_QSA_DENSE（dense_P=P 全稠密）保持原 fp32 标量路径。
      bool dense_launched = false;
      if constexpr (sizeof(KVT) == 2) {
        if (qsa_wmma && !qsa_dense) {
          dense_launched = true;
          if (vc_transposed) {
            if (ptab)
              k_qsa_wmma<true, true, true><<<dim3(dense_P, 2), 256, 0, g_str>>>(
                  d_qsb, kc, vc, vc_transposed, d_attnb, dense_P, 256, 2, 24,
                  nullptr, base, base, ptab, dvb);
            else
              k_qsa_wmma<true, false, true><<<dim3(dense_P, 2), 256, 0, g_str>>>(
                  d_qsb, kc, vc, vc_transposed, d_attnb, dense_P, 256, 2, 24,
                  nullptr, base, base, nullptr, dvb);
          } else {
            if (ptab)
              k_qsa_wmma<false, true, true><<<dim3(dense_P, 2), 256, 0, g_str>>>(
                  d_qsb, kc, vc, nullptr, d_attnb, dense_P, 256, 2, 24, nullptr,
                  base, base, ptab, dvb);
            else
              k_qsa_wmma<false, false, true><<<dim3(dense_P, 2), 256, 0, g_str>>>(
                  d_qsb, kc, vc, nullptr, d_attnb, dense_P, 256, 2, 24, nullptr,
                  base, base, nullptr, dvb);
          }
        }
      }
      if (!dense_launched)
        k_qsa_flash<false, false, KVT><<<dim3((dense_P + 15) / 16, 24), 128, 0, g_str>>>(
            d_qsb, kc, vc, d_attnb, dense_P, 256, 2, 24, nullptr, 0, nullptr, nullptr,
            nullptr, base, ptab, dvb);
    }
    int first_abs = std::max(base, 2051), nsparse = base + P - first_abs;
    if (dvb) nsparse = P;  // gate: base >= 2051 -> first_abs == base
    if (!qsa_dense && nsparse > 0) {
      // d_iselected rows are batch-local (row = abs - base; A1b)
      const int* sel = dvb ? d_iselected
                           : d_iselected + (size_t)(first_abs - base) * 512;
      if (qsa_global_v)
        k_qsa_flash<true, false, KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
            d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, nullptr, nullptr,
            nullptr, base, ptab, dvb);
      else if constexpr (sizeof(KVT) == 2) {
        if (qsa_union && vc_transposed) {
          // 并集路径：[ga, ge) 内的 4 对齐完整组走 merge + k_qsa_wmma_u，
          // 两侧 ragged 边缘回退逐 token 的 k_qsa_wmma<true>
          const int ga = (first_abs + 3) & ~3;
          const int ge = (base + P) & ~3;
          int ngroups = (ge - ga) >> 2;
          if (dvb) {  // baked worst case over base mod 4; in-kernel ge guard
            ngroups = 0;
            for (int r = 0; r < 4; r++)
              ngroups = std::max(ngroups, (((r + P) & ~3) - ((r + 3) & ~3)) >> 2);
          }
          if (ngroups > 0) {
            k_qsa_union_merge<<<ngroups, 128, 0, g_str>>>(
                d_iselected, ga, d_ublk, d_umask, d_ucount, QSA_UCAP, base, P,
                dvb);
            if (ptab)
              k_qsa_wmma_u<true><<<dim3(ngroups, 2), 256, 0, g_str>>>(
                  d_qsb, kc, vc_transposed, d_attnb, 256, 2, 24, d_ublk, d_umask,
                  d_ucount, QSA_UCAP, ga, base, ptab, P, dvb);
            else
              k_qsa_wmma_u<<<dim3(ngroups, 2), 256, 0, g_str>>>(
                  d_qsb, kc, vc_transposed, d_attnb, 256, 2, 24, d_ublk, d_umask,
                  d_ucount, QSA_UCAP, ga, base, nullptr, P, dvb);
          }
          // ragged edges: per-token kernel (selected rows relative to its
          // own first token -> offset sel)
          auto edge = [&](int f, int n) {
            const int* es = sel + (size_t)(f - first_abs) * 512;
            if (ptab)
              k_qsa_wmma<true, true><<<dim3(n, 2), 256, 0, g_str>>>(
                  d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, es, f, base, ptab);
            else
              k_qsa_wmma<true><<<dim3(n, 2), 256, 0, g_str>>>(
                  d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, es, f, base);
          };
          if (dvb) {
            // baked 3-token grids for both edges; kernels mask by *dvb
            // (dvb_edge 2 = left [first_abs, ga), 1 = right [ge, base+P))
            for (int ek = 2; ek >= 1; ek--) {
              if (ptab)
                k_qsa_wmma<true, true><<<dim3(3, 2), 256, 0, g_str>>>(
                    d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24,
                    d_iselected, 0, 0, ptab, dvb, ek);
              else
                k_qsa_wmma<true><<<dim3(3, 2), 256, 0, g_str>>>(
                    d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24,
                    d_iselected, 0, 0, nullptr, dvb, ek);
            }
          } else {
            if (ga > first_abs) edge(first_abs, std::min(ga, base + P) - first_abs);
            if (base + P > ge && ge >= ga) edge(ge, base + P - ge);
          }
        }
        else if (qsa_wmma6 && vc_transposed)
          k_qsa_wmma6<<<dim3(nsparse, 2), 128, 0, g_str>>>(
              d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel,
              first_abs, base);
        else if (qsa_wmma_btv && vc_transposed) {
          if (ptab)  // A1c paged instantiation (never taken unpaged)
            k_qsa_wmma<true, true><<<dim3(nsparse, 2), 256, 0, g_str>>>(
                d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel, first_abs,
                base, ptab, dvb);
          else
            k_qsa_wmma<true><<<dim3(nsparse, 2), 256, 0, g_str>>>(
                d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel, first_abs,
                base, nullptr, dvb);
        } else if (qsa_wmma) {
          // QWENOX_QSA_WMMA=1: WMMA bf16 sparse flash (256 thr), see k_qsa_wmma.
          if (ptab)
            k_qsa_wmma<false, true><<<dim3(nsparse, 2), 256, 0, g_str>>>(
                d_qsb, kc, vc, nullptr, d_attnb, P, 256, 2, 24, sel, first_abs, base,
                ptab, dvb);
          else
            k_qsa_wmma<false><<<dim3(nsparse, 2), 256, 0, g_str>>>(
                d_qsb, kc, vc, nullptr, d_attnb, P, 256, 2, 24, sel, first_abs, base,
                nullptr, dvb);
        } else
          // BF16 KV mode (!qsa_kv_fp32): bf16 LDS tiles + fdot2 (-61% at 32K).
          k_qsa_flash_bf16<KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
              d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, base, dvb);
      }
      else
        k_qsa_flash<true, true, KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
            d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, nullptr, nullptr,
            nullptr, base, ptab, dvb);
    }
  }

  // Copy P rows (512 floats) for absolute positions [base, base+P) from a
  // contiguous batch buffer into one layer of the FP32 KV cache. Unpaged: one
  // bulk copy (legacy behaviour). Paged: one copy per physically contiguous
  // run of pages (a single copy for the identity table).
  void kv_rows_store(float* layer, const float* src, int base, int P) {
    kv_runs(base, base + P, KV_PAGE, [&](int lr, size_t pr, int n) {
      CK(hipMemcpyAsync(layer + pr * 512, src + (size_t)(lr - base) * 512,
                        (size_t)n * 512 * 4, hipMemcpyDeviceToDevice, g_str));
    });
  }

  // Visit logical rows [r0, r1) of one paged per-layer region as physically
  // contiguous runs: fn(first_logical_row, first_physical_row, nrows).
  // per_page = KV_PAGE for KV rows, KV_PAGE_BLOCKS for indexer key blocks.
  // Unpaged: a single identity run. Paged: requires r1 <= mapped pages *
  // per_page (A2: rows beyond the live mapping have no backing page).
  template <typename F>
  void kv_runs(int r0, int r1, int per_page, F&& fn) const {
    if (r1 <= r0) return;
    if (!kv_paged) {
      fn(r0, (size_t)r0, r1 - r0);
      return;
    }
    if ((size_t)r1 > (size_t)kv_seq.mapped * per_page) {
      fprintf(stderr, "[kvpage] FATAL: kv_runs [%d, %d) x%d beyond the %d mapped pages\n",
              r0, r1, per_page, kv_seq.mapped);
      abort();
    }
    const std::vector<int>& h_ptab = kv_seq.tab;
    for (int t = r0; t < r1;) {
      const int lp = t / per_page, pp = h_ptab[lp];
      int q = lp + 1;
      while ((size_t)q * per_page < (size_t)r1 && h_ptab[q] == pp + (q - lp)) q++;
      const int re = (int)std::min<size_t>((size_t)r1, (size_t)q * per_page);
      fn(t, (size_t)pp * per_page + (size_t)(t % per_page), re - t);
      t = re;
    }
  }

  // ---- A2: on-demand page mapping of the live sequence ----
  // Every device write to the trunk QSA KV / indexer keys goes through
  // prefill_chunk (rows [base, base+P)) or forward (row pos), and both call
  // kv_prepare first; the only other writer is kvsnap::restore, which
  // reserves fresh rows itself. New entries were guard entries that no
  // in-flight kernel reads, so the table upload needs no extra sync.
  // A3: drop the LRU pinned checkpoint; false when there is none left.
  bool kv_evict_one() {
    return (kv_evict_fn && kv_evict_fn(*this)) || (kv_starve_fn && kv_starve_fn(*this));
  }
  [[noreturn]] void kv_fatal(const char* what, int end) {
    fprintf(stderr,
            "[kvpage] FATAL: %s (%d tokens; pool %d pages, %d free, table %d "
            "pages, mapped %d)\n",
            what, end, kv_pool_pages, kv_pool.nfree(), kv_lpages, kv_seq.mapped);
    exit(1);
  }
  void kv_reserve(int end) {
    if (!kv_paged || end <= kv_seq.rows()) return;
    if ((end + KV_PAGE - 1) / KV_PAGE > kv_lpages) kv_fatal("beyond maxctx", end);
    const int m0 = kv_seq.mapped;
    // pool dry: evict pinned checkpoints (LRU first) until the rows fit
    while (!kv_seq.reserve(kv_pool, end))
      if (!kv_evict_one()) kv_fatal("pool exhausted", end);
    CK(hipMemcpy(d_ptab + m0, kv_seq.tab.data() + m0,
                 (size_t)(kv_seq.mapped - m0) * 4, hipMemcpyHostToDevice));
    if (kv_zero_dbg)  // debug: fresh pages read as zeros, like a zeroed unpaged cache
      for (int lp = m0; lp < kv_seq.mapped; lp++)
        kv_page_regions_k(kv_seq.tab[(size_t)lp], [&](void* q, size_t n, char c) {
          if (kv_zero_has(c)) CK(hipMemset(q, 0, n));
        });
    if (kv_mode == 2 && m0 == 0)
      fprintf(stderr, "[kvpage] seq map: logical 0 -> phys %d (free %d/%d)\n",
              kv_seq.tab[0], kv_pool.nfree(), kv_pool_pages);
  }
  // A3: rows [start, end) are about to be written. Pages there that a pinned
  // checkpoint shares are copied first (in practice only the partial page
  // holding `start`), then the tail is mapped. Cheap host check otherwise.
  void kv_prepare(int start, int end) {
    if (!kv_paged) return;
    const int lp1 = std::min(kv_seq.mapped, (end + KV_PAGE - 1) / KV_PAGE);
    for (int lp = std::max(0, start) / KV_PAGE; lp < lp1; lp++)
      if (kv_pool.ref[(size_t)kv_seq.tab[(size_t)lp]] > 1) kv_cow(lp);
    kv_reserve(end);
  }
  void kv_cow(int lp) {
    const int old = kv_seq.tab[(size_t)lp];
    int p;
    while ((p = kv_pool.alloc()) < 0) {
      if (!kv_evict_one()) kv_fatal("pool exhausted (copy-on-write)", lp * KV_PAGE);
      if (kv_pool.ref[(size_t)old] == 1) return;  // the other holder was evicted
    }
    CK(hipDeviceSynchronize());  // no in-flight kernel may still write `old`
    std::vector<std::pair<void*, size_t>> src, dst;
    kv_page_regions(old, [&](void* q, size_t n) { src.emplace_back(q, n); });
    kv_page_regions(p, [&](void* q, size_t n) { dst.emplace_back(q, n); });
    for (size_t i = 0; i < src.size(); i++)
      CK(hipMemcpy(dst[i].first, src[i].first, src[i].second, hipMemcpyDeviceToDevice));
    kv_seq.replace(kv_pool, lp, p);
    CK(hipMemcpy(d_ptab + lp, &p, 4, hipMemcpyHostToDevice));
    kv_cow_count++;
    if (kv_mode == 2)
      fprintf(stderr, "[kvpage] cow logical %d: phys %d -> %d (free %d/%d)\n", lp,
              old, p, kv_pool.nfree(), kv_pool_pages);
  }
  long long kv_cow_count = 0;
  // QWENOX_KV_ZERO=1 (debug, A5 hunt): no row a sequence has not written holds
  // another sequence's data -- unpaged: the whole trunk KV/ik cache is zeroed
  // on reset_state; paged: every freshly mapped page is zeroed. If outputs
  // then agree across modes, some kernel reads rows >= pos and uses them.
  // debug: QWENOX_KV_ZERO=1 zeroes all of K/V/Vt/ik; or a subset of letters
  // k v t i (e.g. QWENOX_KV_ZERO=i), for both the unpaged and the paged zeroing.
  bool kv_zero_dbg = getenv("QWENOX_KV_ZERO") != nullptr;
  bool kv_zero_has(char c) {
    const char* e = getenv("QWENOX_KV_ZERO");
    return e && (!strcmp(e, "1") || strchr(e, c));
  }
  void kv_zero_unpaged() {
    const size_t nqsa = (size_t)(g_cfg.layers / 4), rows = (size_t)kv_nl * kv_layer_rows;
    if (qsa_kv_fp32) {
      if (kv_zero_has('k')) CK(hipMemset(d_kc, 0, rows * 512 * 4));
      if (kv_zero_has('v')) CK(hipMemset(d_vc, 0, rows * 512 * 4));
    } else {
      if (kv_zero_has('k')) CK(hipMemset(d_kcb, 0, rows * 512 * 2));
      if (kv_zero_has('v')) CK(hipMemset(d_vcb, 0, rows * 512 * 2));
      if (qsa_wmma_btv && kv_zero_has('t'))
        CK(hipMemset(d_vcbt, 0, nqsa * qsa_vt_blocks * 2 * 256 * 4 * 2));
    }
    if (d_ik && kv_zero_has('i')) CK(hipMemset(d_ik, 0, (size_t)kv_nl * ik_layer_blocks * 128 * 4));
    if (d_ikh && kv_zero_has('i')) CK(hipMemset(d_ikh, 0, (size_t)kv_nl * ik_layer_blocks * 128 * 2));
  }
  // A3: pin / unpin / adopt the live prefix for RAM checkpoints.
  std::vector<int> kv_pin(int pos) {
    if (!kv_paged) return {};
    if ((pos + KV_PAGE - 1) / KV_PAGE > kv_seq.mapped) kv_fatal("pin beyond mapped", pos);
    std::vector<int> v = kv_seq.pin(kv_pool, pos);
    if (kv_mode == 2)
      fprintf(stderr, "[kvpage] pin %zu pages for %d tokens (free %d/%d)\n", v.size(),
              pos, kv_pool.nfree(), kv_pool_pages);
    return v;
  }
  void kv_unpin(const std::vector<int>& pages) {
    if (!kv_paged || pages.empty()) return;
    KvSeqTable::unpin(kv_pool, pages);
    if (kv_mode == 2)
      fprintf(stderr, "[kvpage] unpin %zu pages (free %d/%d)\n", pages.size(),
              kv_pool.nfree(), kv_pool_pages);
  }
  void kv_adopt(const std::vector<int>& pages) {
    if (!kv_paged) return;
    CK(hipDeviceSynchronize());  // entries about to change may still be read
    const int m1 = kv_seq.mapped;
    kv_seq.adopt(kv_pool, pages);
    const int n = std::max(m1, kv_seq.mapped);
    if (n > 0)
      CK(hipMemcpy(d_ptab, kv_seq.tab.data(), (size_t)n * 4, hipMemcpyHostToDevice));
    if (kv_mode == 2) {
      fprintf(stderr, "[kvpage] adopt %d pages (was %d mapped, free %d/%d)\n",
              kv_seq.mapped, m1, kv_pool.nfree(), kv_pool_pages);
      if (kv_guard_dirty())
        fprintf(stderr, "[kvpage] ERROR: GUARD PAGE WRITTEN — some kernel wrote "
                        "through an unmapped logical page\n");
    }
  }
  // Release the pages that hold no row < keep_pos (reset_state: 0; RAM/SSD
  // checkpoint restores: the restored length). Waits for the device first:
  // in-flight work may still read the entries being reset to the guard.
  void kv_trim(int keep_pos) {
    if (!kv_paged) return;
    const int m1 = kv_seq.mapped;
    if (kv_seq.trim(kv_pool, keep_pos) == 0) return;
    const int m0 = kv_seq.mapped;
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(d_ptab + m0, kv_seq.tab.data() + m0, (size_t)(m1 - m0) * 4,
                 hipMemcpyHostToDevice));
    if (kv_mode == 2) {
      fprintf(stderr, "[kvpage] released %d pages (mapped %d, free %d/%d)\n",
              m1 - m0, m0, kv_pool.nfree(), kv_pool_pages);
      if (kv_guard_dirty())
        fprintf(stderr, "[kvpage] ERROR: GUARD PAGE WRITTEN — some kernel wrote "
                        "through an unmapped logical page\n");
    }
  }
  // The guard page's slice of every paged buffer: fn(device ptr, bytes).
  template <typename F>
  void kv_guard_regions(F&& fn) {
    kv_page_regions(kv_pool.guard(), fn);
  }
  // Physical page `page`'s slice of every paged buffer, in a fixed order.
  template <typename F>
  void kv_page_regions(int page, F&& fn) {
    kv_page_regions_k(page, [&](void* q, size_t n, char) { fn(q, n); });
  }
  // Same, fn(device ptr, bytes, kind) with kind k / v / t (transposed V) / i.
  // Layers 0..kv_nl-1: the trunk QSA layers, then the MTP layer (no t).
  template <typename F>
  void kv_page_regions_k(int page, F&& fn) {
    const int nqsa = g_cfg.layers / 4;
    const size_t g = (size_t)page;
    for (int qi = 0; qi < kv_nl; qi++) {
      const size_t row0 = (size_t)qi * kv_layer_rows + g * KV_PAGE;
      if (qsa_kv_fp32) {
        fn(d_kc + row0 * 512, (size_t)KV_PAGE * 512 * 4, 'k');
        fn(d_vc + row0 * 512, (size_t)KV_PAGE * 512 * 4, 'v');
      } else {
        fn(d_kcb + row0 * 512, (size_t)KV_PAGE * 512 * 2, 'k');
        fn(d_vcb + row0 * 512, (size_t)KV_PAGE * 512 * 2, 'v');
        if (qsa_wmma_btv && qi < nqsa)
          fn(vct_layer(qi) + g * KV_PAGE_BLOCKS * 2 * 256 * 4,
             (size_t)KV_PAGE_BLOCKS * 2 * 256 * 4 * 2, 't');
      }
      if (d_ik)
        fn(d_ik + ((size_t)qi * ik_layer_blocks + g * KV_PAGE_BLOCKS) * 128,
           (size_t)KV_PAGE_BLOCKS * 128 * 4, 'i');
      if (d_ikh)  // C1 f16 twin: COW/zero/guard treat it like the fp32 page
        fn(d_ikh + ((size_t)qi * ik_layer_blocks + g * KV_PAGE_BLOCKS) * 128,
           (size_t)KV_PAGE_BLOCKS * 128 * 2, 'i');
    }
  }
  bool kv_guard_dirty() {
    std::vector<uint8_t> h;
    bool dirty = false;
    kv_guard_regions([&](void* p, size_t n) {
      if (dirty) return;
      h.resize(n);
      CK(hipMemcpy(h.data(), p, n, hipMemcpyDeviceToHost));
      for (uint8_t b : h)
        if (b) {
          dirty = true;
          break;
        }
    });
    return dirty;
  }

  // FP32 indexer projection y[640,P] = w[640,2560] . x[P,2560]^T (index_f32).
  // rocblas_sgemm's own pick for this shape is a MT32x32x8 kernel at ~2.7
  // TFLOPS; the rocBLAS solution list for the same fp32 problem (hipBLASLt-
  // backed, negative indices) holds kernels at 6.2-7.4 TFLOPS (x2.35-2.7 for
  // P >= 1024, slower at P=53). tools/sgemm_iproj_check.cu: deterministic
  // run-to-run, maxrel ~7e-6 vs sgemm — the same order as sgemm's own error vs
  // fp64 (~4.5e-6); still fp32 in/accumulate, only the K order differs. For
  // P >= 1024 the fastest listed solution is timed once (first such call,
  // one ~0.1 s sync stall) and reused for every P; it is only adopted if it
  // beats sgemm, and any non-success launch falls back to sgemm.
  // QWENOX_IPROJ_SGEMM=1 keeps plain rocblas_sgemm (A/B).
  int iproj_sol = INT_MIN;  // INT_MIN = not picked yet, INT_MAX = use sgemm
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"  // beta API
  void iproj_sgemm(const float* w, const float* x, float* y, int P) {
    const float one = 1.f, zero = 0.f;
    static const bool plain = getenv("QWENOX_IPROJ_SGEMM") != nullptr;
    // decode-P (<=8) multi-row fp32 GEMV in place of rocblas_sgemm's
    // MT32x32x8 (20 blocks, ~110 us at P=4 vs ~36 us streaming the 6.5 MB
    // weight once). NOT bit-exact vs sgemm (lane-split reorder); ktest
    // maxrel 1e-5 vs fp64 (same order as sgemm's own fp32 error) and
    // 8K/32K MTP+chain ids matched base exactly. QWENOX_IPROJ_GEMV=0 opt-out.
    static const bool use_gemv = !env_zero("QWENOX_IPROJ_GEMV");
    if (use_gemv && !plain && P >= 1 && P <= 8) {
      const unsigned blocks = (640 + 15) / 16;  // 16 warps (512 thr) per block
      switch (P) {
#define QWENOX_IPROJ_GV(PP)                                                \
  case PP:                                                               \
    k_f32_gemv_mr<PP><<<blocks, 512, 0, g_str>>>(w, x, y, 640, 2560);    \
    break;
        QWENOX_IPROJ_GV(1)
        QWENOX_IPROJ_GV(2)
        QWENOX_IPROJ_GV(3)
        QWENOX_IPROJ_GV(4)
        QWENOX_IPROJ_GV(5)
        QWENOX_IPROJ_GV(6)
        QWENOX_IPROJ_GV(7)
        QWENOX_IPROJ_GV(8)
#undef QWENOX_IPROJ_GV
      }
      return;
    }
    auto sol_run = [&](int idx) {
      return rocblas_gemm_ex(
          rbh, rocblas_operation_transpose, rocblas_operation_none, 640, P,
          2560, &one, w, rocblas_datatype_f32_r, 2560, x,
          rocblas_datatype_f32_r, 2560, &zero, y, rocblas_datatype_f32_r, 640,
          y, rocblas_datatype_f32_r, 640, rocblas_datatype_f32_r,
          rocblas_gemm_algo_solution_index, idx, 0);
    };
    auto sgemm_run = [&]() {
      return rocblas_sgemm(rbh, rocblas_operation_transpose,
                           rocblas_operation_none, 640, P, 2560, &one, w, 2560,
                           x, 2560, &zero, y, 640);
    };
    if (!plain && P >= 1024) {
      if (iproj_sol == INT_MIN) {
        iproj_sol = INT_MAX;
        rocblas_int n = 0;
        rocblas_gemm_ex_get_solutions(
            rbh, rocblas_operation_transpose, rocblas_operation_none, 640, P,
            2560, &one, w, rocblas_datatype_f32_r, 2560, x,
            rocblas_datatype_f32_r, 2560, &zero, y, rocblas_datatype_f32_r,
            640, y, rocblas_datatype_f32_r, 640, rocblas_datatype_f32_r,
            rocblas_gemm_algo_solution_index, 0, nullptr, &n);
        std::vector<rocblas_int> sols(std::max<rocblas_int>(n, 1));
        if (n > 0)
          rocblas_gemm_ex_get_solutions(
              rbh, rocblas_operation_transpose, rocblas_operation_none, 640, P,
              2560, &one, w, rocblas_datatype_f32_r, 2560, x,
              rocblas_datatype_f32_r, 2560, &zero, y, rocblas_datatype_f32_r,
              640, y, rocblas_datatype_f32_r, 640, rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index, 0, sols.data(), &n);
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        auto time2 = [&](auto&& fn) {  // validate once, then time 2 runs
          if (fn() != rocblas_status_success ||
              hipStreamSynchronize(g_str) != hipSuccess) {
            (void)hipGetLastError();
            return 1e30f;
          }
          float ms = 0;
          CK(hipEventRecord(e0, g_str));
          fn();
          fn();
          CK(hipEventRecord(e1, g_str));
          CK(hipEventSynchronize(e1));
          CK(hipEventElapsedTime(&ms, e0, e1));
          return ms;
        };
        const float t_sgemm = time2(sgemm_run);
        float best = t_sgemm;
        for (int i = 0; i < (int)n; i++) {
          const float t = time2([&] { return sol_run(sols[i]); });
          if (t < best * 0.95f) {
            best = t;
            iproj_sol = sols[i];
          }
        }
        CK(hipEventDestroy(e0));
        CK(hipEventDestroy(e1));
        fprintf(stderr, "iproj: fp32 indexer projection P=%d sgemm %.2f ms, %s\n",
                P, t_sgemm / 2,
                iproj_sol == INT_MAX
                    ? "keeping sgemm"
                    : ("solution " + std::to_string(iproj_sol) + " " +
                       std::to_string(best / 2) + " ms")
                          .c_str());
      }
      if (iproj_sol != INT_MAX && sol_run(iproj_sol) == rocblas_status_success)
        return;
    }
    if (sgemm_run() != rocblas_status_success) {
      fprintf(stderr, "indexer fp32 projection failed (P=%d)\n", P);
      exit(1);
    }
  }
#pragma clang diagnostic pop

  // QSA: projections batched. Flash path: qsplit/norm/rope batched, KV rows
  // bulk-copied to cache slots 0..P-1, one causal flash kernel over all
  // positions. QWENOX_QSA_LOOP=1 falls back to the per-token loop (debug).
  // Indexer + score/select phase of qsa_b: host-driven, base-dependent grids,
  // so verify graph capture (P3-B) runs it eagerly between graph segments.
  void qsa_idx_b(int l, int qi, int P, int base) {
    if (qsa_dense) return;
    char buf[128];
    {
      KP_SUB("qsa:idx");
      // rope cos/sin table for the indexer kernels below (idempotent; the QSA
      // prep path re-ensures the same (base, P) later and hits the cache)
      ensure_rope_tab(g_cfg.rope_theta, 64);
      ensure_rope_cs(base, P);
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      if (getenv("QWENOX_VGDUMP")) {  // P3-B: xf16 cache state at the idx gemm
        fprintf(stderr, "[xc%d] P=%d hit=%d xf16P=%d xf16K=%d\n", l, P,
                (int)(xf16_src == d_xb && xf16_xs == (size_t)g_cfg.d &&
                      xf16_P == P && xf16_K == g_cfg.d),
                xf16_P, xf16_K);
        char tag[12];
        snprintf(tag, sizeof tag, "xb%d", l);
        vg_dump_buf(tag, P, d_xbf16, (size_t)P * 2560 * 2);
      }
      if (index_f32) {
        iproj_sgemm(d_iw + (size_t)qi * 640 * 2560, d_xb, d_iproj, P);
      } else {
        gemm(wview(ckpt.at(buf)), d_xb, d_iproj, P);
      }
      float* keys = d_ik + (size_t)qi * ik_layer_blocks * 128;
      uint16_t* keys16 =
          d_ikh ? d_ikh + (size_t)qi * ik_layer_blocks * 128 : nullptr;
      k_index_q<<<dim3(P, 4), 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256, d_iq,
                                              P, g_cfg.norm_eps, g_cfg.rope_theta,
                                              nullptr, base, d_ropecs, nullptr,
                                              vbptr(), d_iqh);
      // absolute 4-token blocks [base/4, (base+P)/4); a left-straddling block
      // (base%4 != 0) is completed with history rows from the raw ring
      int nb = (base + P) / 4, b0 = base / 4;
      if (getenv("QWENOX_VGDUMP") && P >= 2 && P <= 16 && !vg_capturing_) {
        char tag[12];
        snprintf(tag, sizeof tag, "ir0_%d", l);
        vg_dump_buf(tag, P, d_iraw + (size_t)qi * 512, 512 * 4);
        int vbv = -1;
        if (vbptr()) {
          CK(hipStreamSynchronize(g_str));
          CK(hipMemcpy(&vbv, d_vbase, 4, hipMemcpyDeviceToHost));
        }
        fprintf(stderr, "[vb%d] P=%d base=%d vbase=%d\n", l, P, base, vbv);
      }
      if (nb > b0)
        k_index_pool<<<nb - b0, 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256 + 128,
                                                 keys, g_cfg.norm_eps,
                                                 g_cfg.rope_theta, base,
                                                 d_iraw + qi * 512, d_ropecs, kvpt(),
                                                 vbptr(), keys16);
      k_index_ring<<<min(P, 4), 128, 0, g_str>>>(d_iproj, d_iraw + qi * 512, P, base,
                                                 vbptr());
      if (cut_r_)  // in-pass cut: raw ring after the first r rows
        k_index_ring<<<min(cut_r_, 4), 128, 0, g_str>>>(d_iproj, d_ciraw + qi * 512,
                                                         cut_r_, base, vbptr());
      if (spec_snap)
        k_iproj_snap<<<P, 128, 0, g_str>>>(
            d_iproj, d_iproj_snap + (size_t)qi * 9 * 128, P);
      // Paged: the score sgemm / tiled kernel read keys [0, nb) contiguously;
      // gather them into d_ikg first (same values, same sgemm shape -> the
      // scores and selections stay bit-identical to the unpaged layout).
      const float* skeys = keys;
      const uint16_t* skeys16 = keys16;
      if (kv_paged && nb > 0 && std::max(2051, base) < base + P) {
        k_ik_gather<<<nb, 128, 0, g_str>>>(keys, d_ptab, d_ikg, nb);
        skeys = d_ikg;
        if (keys16) {
          k_ik_gather16<<<nb, 128, 0, g_str>>>(keys16, d_ptab, d_ikgh, nb);
          skeys16 = d_ikgh;
        }
      }
      const float one = 1.f, zero = 0.f;
      // NOTE: nb (m/ldc) is the absolute pooled-block count, matching the
      // shape a from-scratch prefill of the same total length would use —
      // the sgemm shape stays numerically locked across request splits.
      // Chunk starts are additionally aligned to the from-scratch grid
      // (2051 + k*INDEX_BATCH): with the same (m, n, k) every full chunk
      // replays bitwise-identical scores and selections, so a continuation
      // prefill can only diverge inside the leading ragged chunk [a0, grid).
      const int a0 = std::max(2051, base);
      const int grid =
          2051 + (a0 - 2051 + INDEX_BATCH - 1) / INDEX_BATCH * INDEX_BATCH;
      if (getenv("QWENOX_VGDUMP")) {  // P3-B bisect: selection inputs
        char tag[12];
        // split keys hash: prefix blocks vs last two blocks (freshly pooled)
        snprintf(tag, sizeof tag, "ikp%d", l);
        vg_dump_buf(tag, P, skeys, (size_t)(nb > 2 ? nb - 2 : 0) * 128 * 4);
        snprintf(tag, sizeof tag, "ikl%d", l);
        vg_dump_buf(tag, P, skeys + (size_t)(nb > 2 ? nb - 2 : 0) * 128,
                    (size_t)(nb > 2 ? 2 : nb) * 128 * 4);
        if (skeys16) {
          snprintf(tag, sizeof tag, "ih%d", l);
          vg_dump_buf(tag, P, skeys16, (size_t)nb * 128 * 2);
        }
        snprintf(tag, sizeof tag, "ip%d", l);
        vg_dump_buf(tag, P, d_iproj, (size_t)P * 640 * 4);
        if (P >= 2 && P <= 16) {  // q-half vs k-half split
          CK(hipStreamSynchronize(g_str));
          std::vector<float> ip((size_t)P * 640);
          CK(hipMemcpy(ip.data(), d_iproj, (size_t)P * 640 * 4,
                       hipMemcpyDeviceToHost));
          uint64_t hq = 1469598103934665603ULL, hk = 1469598103934665603ULL;
          for (int t = 0; t < P; t++)
            for (int d = 0; d < 640; d++) {
              uint32_t u;
              memcpy(&u, &ip[(size_t)t * 640 + d], 4);
              if (d < 512) { hq ^= u; hq *= 1099511628211ULL; }
              else { hk ^= u; hk *= 1099511628211ULL; }
            }
          fprintf(stderr, "[ips%d] P=%d q=%016llx k=%016llx\n", l, P,
                  (unsigned long long)hq, (unsigned long long)hk);
        }
        snprintf(tag, sizeof tag, "ir%d", l);
        vg_dump_buf(tag, P, d_iraw + (size_t)qi * 512, 512 * 4);
        snprintf(tag, sizeof tag, "pt%d", l);
        vg_dump_buf(tag, P, d_ptab, (size_t)kv_lpages * 4);
      }
      for (int first = a0; first < base + P;) {
        int nxt = first < grid ? min(grid, base + P)
                               : min(first + INDEX_BATCH, base + P);
        int count = nxt - first;
        if (index_fused2 && count >= 16 && getenv("QWENOX_INDEX_OLDSEL") == nullptr) {
          int* const selrow = d_iselected + (size_t)(first - base) * 512;  // batch-local
          if (index_f16()) {
            // C1: f16 WMMA 打分 + f16 score（复用 d_iscores 内存）+ f16 select
            uint16_t* const sc16 = (uint16_t*)d_iscores;
            k_index_scores_wmma16<false>
                <<<index_wmma16_grid(nb, count), 256, 0, g_str>>>(
                    d_iqh + (size_t)(first - base) * 512, skeys16, sc16, nb,
                    count, first);
            if ((first + count) / 4 <= 8192)
              k_index_select_rs<true, __half><<<count, 256, 0, g_str>>>(
                  (const __half*)sc16, nb, selrow, first);
            else
              k_index_select_2p<__half><<<count, 256, 0, g_str>>>(
                  (const __half*)sc16, nb, selrow, first);
            first = nxt;
            continue;
          }
          if (index_score64)
            k_index_scores_t64<<<index_t64_grid(nb, count), 256, 0, g_str>>>(
                d_iq + (size_t)(first - base) * 512, skeys, d_iscores, nb, count, first);
          else
            k_index_scores_tiled<<<dim3((nb + 31) / 32, (count + 15) / 16), 256, 0, g_str>>>(
                d_iq + (size_t)(first - base) * 512, skeys, d_iscores, nb, count, first);
          if ((first + count) / 4 <= 8192)
            k_index_select_rs<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          else if (index_sel2p)
            k_index_select_2p<<<count, 256, 0, g_str>>>(d_iscores, nb, selrow, first);
          else if (index_stream_select)
            k_index_select_stream<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          else
            k_index_select<16, true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          first = nxt;
          continue;
        }
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose, rocblas_operation_none,
                                    nb, count * 4, 128, &one, skeys, 128,
                                    d_iq + (size_t)(first - base) * 512, 128, &zero,
                                    d_iscores, nb);
        if (status != rocblas_status_success) {
          fprintf(stderr, "indexer sgemm failed: %d\n", (int)status);
          exit(1);
        }
        index_select(d_iscores, nb, d_iselected + (size_t)(first - base) * 512, first,
                     count);
        first = nxt;
      }
      if (getenv("QWENOX_VGDUMP")) {
        char tag[12];
        snprintf(tag, sizeof tag, "is%d", l);
        vg_dump_buf(tag, P, d_iselected, (size_t)P * 512 * 4);
        snprintf(tag, sizeof tag, "iq%d", l);
        vg_dump_buf(tag, P, d_iq, (size_t)P * 512 * 4);
        snprintf(tag, sizeof tag, "rc%d", l);
        vg_dump_buf(tag, P, d_ropecs, (size_t)P * 32 * 8);
      }
      KP_SUB_END();
    }
  }

  void qsa_b(int l, int qi, int P, int base = 0, const int* dvb = nullptr) {
    qsa_idx_b(l, qi, P, base);
    qsa_main_b(l, qi, P, base, dvb);
  }

  // Graphable remainder of qsa_b (P3-B): dvb = device base mirror (graph
  // capture) or nullptr (direct execution, identical scalar path).
  void qsa_main_b(int l, int qi, int P, int base, const int* dvb) {
    char buf[128];
    const uint16_t* attn_bf = nullptr;  // gated o_proj input already in bf16
    KP_SUB("qsa:proj");
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_qgb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_kbb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.v_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_vbb, P);
    KP_SUB_END();
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_norm.weight", l);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_norm.weight", l);
    float* knw = ensure_norm256(buf);
    // kv_layer_rows == maxctx unless paged (FP32 or BF16 cache, A1c)
    const size_t qsa_off = (size_t)qi * kv_layer_rows * 512;
    if (!qsa_loop) {
      ensure_rope_tab(g_cfg.rope_theta, 64);
      ensure_rope_cs(base, P);
      // fused: qsplit+rope (K1) and knorm+rope[+bf16 store] (K2), bit-exact
      // vs the previous qsplit / rmsnorm_zc / rope_b x2 / convert chain
      // P > 8: o_proj takes the dequant->bf16->GEMM path (P <= 8 would run
      // the multi-row GEMV on fp32 x), so the output gate + bf16 convert fuse
      // into k_sigmoid_gate_bf16_v4, which reads the gate half of d_qgb
      // directly (qsplit skips its gs copy). QWENOX_QSA_GATE_BF16=0 disables.
      static const bool gate_bf16_on = !env_zero("QWENOX_QSA_GATE_BF16");
      const bool gate_bf16 = gate_bf16_on && P > 8;
      KP_SUB("qsa:prep");
      k_qsa_qsplit<true><<<dim3(24, P), 256, 0, g_str>>>(
          d_qgb, qnw, d_qsb, gate_bf16 ? nullptr : d_gsb, 256, g_cfg.norm_eps, base,
          d_ropecs);
      int dense_P = qsa_dense ? P : min(P, std::max(0, 2051 - base));
      if (qsa_kv_fp32) {
        float* kc = d_kc + qsa_off;
        float* vc = d_vc + qsa_off;
        k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(d_kbb, knw, d_kbb,
                                                          nullptr, g_cfg.norm_eps,
                                                          base, d_ropecs);
        // absolute positions base..base+P-1 are logically contiguous: one bulk
        // copy (unpaged) or one copy per physically contiguous page run
        if (dvb) {  // P3-B graph capture: device-side base + page mapping
          k_kv_rows_store_pd<<<P, 128, 0, g_str>>>(kc, d_kbb, kvpt(), P, dvb);
          k_kv_rows_store_pd<<<P, 128, 0, g_str>>>(vc, d_vbb, kvpt(), P, dvb);
        } else {
          kv_rows_store(kc, d_kbb, base, P);
          kv_rows_store(vc, d_vbb, base, P);
        }
        KP_SUB_END();
        KP_SUB("qsa:flash");
        qsa_flash_b(kc, vc, P, dense_P, base, nullptr, kvpt(), dvb);
        KP_SUB_END();
      } else {
        uint16_t* kc = d_kcb + qsa_off;
        uint16_t* vc = d_vcb + qsa_off;
        uint16_t* vct = vct_layer(qi);
        if (vct && base > 0 && getenv("QWENOX_VCT_CHECK")) {  // debug only
          static unsigned* d_chk = nullptr;
          if (!d_chk) CK(hipMalloc(&d_chk, 3 * sizeof(unsigned)));
          const unsigned init[3] = {0u, ~0u, 0u};
          unsigned h[3];
          CK(hipMemcpy(d_chk, init, sizeof init, hipMemcpyHostToDevice));
          k_vct_check<<<(unsigned)(((size_t)base * 512 + 255) / 256), 256, 0, g_str>>>(
              vc, vct, kvpt(), base, d_chk);
          CK(hipStreamSynchronize(g_str));
          CK(hipMemcpy(h, d_chk, sizeof h, hipMemcpyDeviceToHost));
          if (h[0] || qi == 0)
            fprintf(stderr, "[vctchk] qsa%d base=%d P=%d mismatch=%u first=%d last=%d\n",
                    qi, base, P, h[0], h[0] ? (int)h[1] : -1, h[0] ? (int)h[2] : -1);
          const uint64_t nraw = (uint64_t)qsa_vt_blocks * 2 * 256 * 4;
          CK(hipMemcpy(d_chk, init, sizeof init, hipMemcpyHostToDevice));
          k_vct_nonfinite<<<(unsigned)((nraw + 255) / 256), 256, 0, g_str>>>(vct, nraw,
                                                                            d_chk);
          CK(hipStreamSynchronize(g_str));
          CK(hipMemcpy(h, d_chk, sizeof h, hipMemcpyDeviceToHost));
          if (h[0] || qi == 0)
            fprintf(stderr, "[vctnan] qsa%d base=%d nonfinite=%u phys rows %d..%d\n", qi,
                    base, h[0], h[0] ? (int)h[1] : -1, h[0] ? (int)h[2] : -1);
        }
        if (!kv_paged) {
          k_qsa_kprep<true><<<dim3(2, P), 256, 0, g_str>>>(
              d_kbb, knw, d_kbb, dvb ? kc : kc + (size_t)base * 512,
              g_cfg.norm_eps, base, d_ropecs, dvb);
          // absolute positions base..base+P-1 are contiguous: one bulk convert
          k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256, 0,
                             g_str>>>(d_vbb, dvb ? vc : vc + (size_t)base * 512,
                                      512, P, 512, dvb);
          if (vct) {
            // dvb: baked worst-case grid (base mod 4 unknown at capture)
            const int btb = dvb ? (P + 6) / 4 + 1 : (base + P + 3) / 4 - base / 4;
            k_f32_to_bf16_v4_bt<<<(unsigned)(((size_t)btb * 512 + 255) / 256),
                                 256, 0, g_str>>>(d_vbb, vct, 512, P, 512, base, dvb);
          }
        } else {
          // A1c paged: kprep writes the fp32 K rows only (kprep<true> stores
          // f2bf of exactly those values, so converting afterwards is the same
          // bits), then K / V / transposed V are converted per physically
          // contiguous page run. The transposed-V kernel is called with the
          // run's physical base: a page holds whole 4-token blocks, so the
          // slot (pos & 3) and the partial-edge-block masking are unchanged and
          // block pr/4 is the physical block ik_blk(lr/4).
          k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(
              d_kbb, knw, d_kbb, nullptr, g_cfg.norm_eps, base, d_ropecs);
          if (dvb) {  // P3-B graph capture: one launch per output, device-side
                      // base + page mapping (same bits as the per-run loop)
            const unsigned g = (unsigned)(((size_t)P * 128 + 255) / 256);
            k_f32_to_bf16_v4_pd<<<g, 256, 0, g_str>>>(d_kbb, kc, kvpt(), P, 512,
                                                      dvb);
            k_f32_to_bf16_v4_pd<<<g, 256, 0, g_str>>>(d_vbb, vc, kvpt(), P, 512,
                                                      dvb);
            if (vct)  // baked worst-case grid; the kernel masks by *dvb
              k_f32_to_bf16_v4_bt_pd<<<(unsigned)(((((size_t)P + 6) / 4 + 1) *
                                                   512 + 255) /
                                                  256),
                                    256, 0, g_str>>>(d_vbb, vct, kvpt(), P, 512,
                                                     dvb);
          } else {
            kv_runs(base, base + P, KV_PAGE, [&](int lr, size_t prz, int n) {
              const int pr = (int)prz;
              const size_t so = (size_t)(lr - base) * 512;
              const unsigned g = (unsigned)(((size_t)n * 128 + 255) / 256);
              k_f32_to_bf16_v4<<<g, 256, 0, g_str>>>(d_kbb + so, kc + prz * 512,
                                                     512, n, 512);
              k_f32_to_bf16_v4<<<g, 256, 0, g_str>>>(d_vbb + so, vc + prz * 512,
                                                     512, n, 512);
              if (vct)
                k_f32_to_bf16_v4_bt<<<(unsigned)(((size_t)((pr + n + 3) / 4 - pr / 4) *
                                                  512 + 255) /
                                                     256),
                                     256, 0, g_str>>>(d_vbb + so, vct, 512, n, 512, pr);
            });
          }
        }
        if (vct && getenv("QWENOX_VCT_TAILZERO") && ((base + P) & 3)) {  // debug only
          const int r0 = base + P, r1 = (r0 + 3) & ~3;
          const char* tv = getenv("QWENOX_VCT_TAILVAL");  // hex bf16 poison (debug)
          k_vct_zero_slots<<<(unsigned)(((size_t)(r1 - r0) * 512 + 255) / 256), 256, 0,
                             g_str>>>(vct, kvpt(), r0, r1,
                                      tv ? (uint16_t)strtoul(tv, nullptr, 16) : 0);
        }
        KP_SUB_END();
        KP_SUB("qsa:flash");
        qsa_flash_b(kc, vc, P, dense_P, base, vct, kvpt(), dvb);
        KP_SUB_END();
        if (base > 0 && getenv("QWENOX_ATTN_HASH")) dbg_attn_hash(qi, base, P, kc, vc, vct);
      }
      KP_SUB("qsa:gate");
      if (gate_bf16) {  // see qsplit above
        const uint64_t n4 = (uint64_t)P * 6144 / 4;
        k_sigmoid_gate_bf16_v4<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(
            d_attnb, d_qgb, d_xbf16b, n4);
        attn_bf = d_xbf16b;
      } else {
        k_sigmoid_gate<<<(P * 6144 + 255) / 256, 256, 0, g_str>>>(d_attnb, d_gsb,
                                                                  d_attnb, P * 6144);
      }
      KP_SUB_END();
    } else {
      for (int t = 0; t < P; t++) {
        const int* pabs = d_posarr + base + t;  // absolute position
        const float* qg = d_qgb + (size_t)t * 12288;
        float* qs = d_qsb + (size_t)t * 6144;
        float* gs = d_gsb + (size_t)t * 6144;
        float* kb = d_kbb + (size_t)t * 512;
        float* vb = d_vbb + (size_t)t * 512;
        float* att = d_attnb + (size_t)t * 6144;
        k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(qg, qnw, qs, gs, 256,
                                                     g_cfg.norm_eps);
        k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(kb, knw, kb, 256, g_cfg.norm_eps, 1);
        k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(qs, 24, 256, 64, pabs,
                                                         g_cfg.rope_theta);
        k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(kb, 2, 256, 64, pabs,
                                                        g_cfg.rope_theta);
        if (qsa_kv_fp32) {
          k_store_kv<<<2, 256, 0, g_str>>>(kb, vb, d_kc + qsa_off, d_vc + qsa_off,
                                           pabs, kvpt());
          k_qsa_step<float><<<24, 256, 0, g_str>>>(
              qs, d_kc + qsa_off, d_vc + qsa_off, att, pabs, 256, 2, 24,
              qsa_dense ? nullptr : d_iselected + (size_t)t * 512, false,
              kvpt());
        } else {
          k_store_kv_bf16<<<2, 256, 0, g_str>>>(kb, vb, d_kcb + qsa_off,
                                                d_vcb + qsa_off, pabs,
                                                vct_layer(qi), kvpt());
          k_qsa_step<uint16_t><<<24, 256, 0, g_str>>>(
              qs, d_kcb + qsa_off, d_vcb + qsa_off, att, pabs, 256, 2, 24,
              qsa_dense ? nullptr : d_iselected + (size_t)t * 512, false,
              kvpt());
        }
        k_sigmoid_gate<<<(6144 + 255) / 256, 256, 0, g_str>>>(att, gs, att, 6144);
      }
    }
    KP_SUB("qsa:oproj");
    snprintf(buf, sizeof buf, "layers.%d.self_attn.o_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_attnb, d_yb, P, 0, attn_bf);
    KP_SUB_END();
  }

  // MoE: router batched; expert GEMVs grouped over P*k slots (naive: full
  // weight traffic per token — correct but slow; optimized in a later phase).
  // Returns true when the HC write-back of the moe output was fused into the
  // reduce (B2-1; only on the LUT sg_fuse paths with try_hcw set): R is
  // updated, acc/d_yb are not written and the caller skips gr_write_b.
  // ---- in-pass snapshot cut (see cut_abs) ----
  bool cut_snap_supported() const {
    static const bool off = env_zero("QWENOX_CUT_INPASS");
    return !off && gdn_fused && !gdn_loop && !gdn_nosplit && gr_bf16 && !ple_loop;
  }
  void cut_alloc() {
    if (d_cS) return;
    const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
    CK(hipMalloc(&d_cS, (size_t)ngdn * 48 * 128 * 128 * 4));
    CK(hipMalloc(&d_cconvst, (size_t)ngdn * 3 * 10240 * 4));
    CK(hipMalloc(&d_cring, (size_t)9 * 10240 * 4));
    CK(hipMalloc(&d_ciraw, (size_t)nqsa * 4 * 128 * 4));
    CK(hipMalloc(&d_cR, (size_t)g_cfg.branches * g_cfg.d * 4));
    CK(hipMalloc(&d_cmiraw, (size_t)4 * 128 * 4));
  }
  // chunk start: r = chunk-local cut row in (0, P], else 0. Copies the
  // pre-chunk rings / conv state; the per-site replays below update them.
  void cut_begin_chunk(int base, int P) {
    cut_r_ = (cut_abs > base && cut_abs <= base + P) ? cut_abs - base : 0;
    if (!cut_r_) return;
    const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
    CK(hipMemcpyAsync(d_cconvst, d_convst, (size_t)ngdn * 3 * 10240 * 4,
                      hipMemcpyDeviceToDevice, g_str));
    CK(hipMemcpyAsync(d_cring, d_ring, (size_t)9 * 10240 * 4, hipMemcpyDeviceToDevice, g_str));
    if (d_iraw)
      CK(hipMemcpyAsync(d_ciraw, d_iraw, (size_t)nqsa * 4 * 128 * 4, hipMemcpyDeviceToDevice,
                        g_str));
  }
  // Run fn with the cut-point state swapped in (pos = cut_abs). mtp: the MTP
  // ring snapshot was taken, so the lockstep invariant mtp_pos == pos-1 holds.
  template <class F>
  void with_cut_state(bool mtp, F&& fn) {
    const int sv_pos = pos, sv_ring = ple_ringpos, sv_mpos = mtp_pos;
    auto swp = [&] {
      std::swap(d_S, d_cS);
      std::swap(d_convst, d_cconvst);
      std::swap(d_ring, d_cring);
      if (d_iraw) std::swap(d_iraw, d_ciraw);
      std::swap(d_R, d_cR);
      if (mtp && d_miraw) std::swap(d_miraw, d_cmiraw);
    };
    swp();
    pos = cut_abs;
    ple_ringpos = cut_abs % 9;
    if (mtp) mtp_pos = cut_abs - 1;
    fn();
    swp();
    pos = sv_pos;
    ple_ringpos = sv_ring;
    mtp_pos = sv_mpos;
  }

  // d_routerb [P][512] -> d_topidsb/d_topwb. wave-per-token kernel (bit-
  // identical, aggr 10-05); QWENOX_ROUTER_W32=0 restores the block-per-token one.
  void router_topk_b(int P, int k) {
    static const bool w32 = !env_zero("QWENOX_ROUTER_W32");
    if (w32 && g_cfg.experts == 512 && k == 10)
      k_router_topk_w32<10><<<(P + 7) / 8, 256, 0, g_str>>>(d_routerb, d_topidsb, d_topwb, P);
    else
      k_router_topk<<<P, 512, 0, g_str>>>(d_routerb, d_topidsb, d_topwb, g_cfg.experts, k);
  }
  bool moe_b(int l, int P, bool try_hcw = false) {
    char buf[128];
    const bool direct_output = pp_moe_out && P > 8;
    float* const acc = direct_output ? d_yb : d_moeaccb;
    snprintf(buf, sizeof buf, "layers.%d.mlp.gate.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_routerb, P);
    int k = g_cfg.topk;
    router_topk_b(P, k);
    // QWENOX_MOE_UNION_LOG=1: distinct experts across the P rows (P<=8, i.e.
    // verify); decides whether naive-MoE expert dedup is worth it.
    static const bool moe_union_log = getenv("QWENOX_MOE_UNION_LOG") != nullptr;
    if (moe_union_log && P <= 8) {
      struct UStat {
        long long n = 0, uni = 0, slots = 0;
        ~UStat() {
          if (n)
            fprintf(stderr,
                    "moe-union: layers=%lld avg_union=%.2f avg_slots=%.2f "
                    "union/slots=%.3f\n",
                    n, (double)uni / n, (double)slots / n,
                    (double)uni / slots);
        }
      };
      static UStat S;
      std::vector<int> hids((size_t)P * 16);
      CK(hipMemcpy(hids.data(), d_topidsb, (size_t)P * 16 * 4,
                   hipMemcpyDeviceToHost));
      std::vector<char> seen(g_cfg.experts, 0);
      int u = 0;
      for (int t = 0; t < P; t++)
        for (int s = 0; s < k; s++) {
          int e = hids[(size_t)t * 16 + s];
          if (!seen[e]) { seen[e] = 1; u++; }
        }
      S.n++; S.uni += u; S.slots += (long long)P * k;
    }
    // QWENOX_MOE_TOPIDS_DUMP=<file>: 把每次小 P 调用（2<=P<=32，即 verify/合批
    // 场景）的 topids 追加到文件，供 tools/c0_union.py 做跨序列专家并集分析。
    // 行格式：L <layer> B <base> P <P> 后接 P*k 个专家 id。默认零开销。
    static FILE* moe_dump = [] {
      const char* f = getenv("QWENOX_MOE_TOPIDS_DUMP");
      return f ? fopen(f, "w") : nullptr;
    }();
    if (moe_dump && P >= 2 && P <= 32) {
      std::vector<int> hids((size_t)P * 16);
      CK(hipMemcpy(hids.data(), d_topidsb, (size_t)P * 16 * 4,
                   hipMemcpyDeviceToHost));
      fprintf(moe_dump, "L %d B %d P %d", l, moe_dump_base_, P);
      for (int t = 0; t < P; t++)
        for (int s = 0; s < k; s++)
          fprintf(moe_dump, " %d", hids[(size_t)t * 16 + s]);
      fputc('\n', moe_dump);
      fflush(moe_dump);
    }
    if (gg_on()) {
      moe_gguf(g_ggx.at(l), d_xb, d_topidsb, d_topwb, acc, P);
      moe_b_shared(l, P, acc, direct_output);
      return false;
    }
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.gate_up_proj.weight", l);
    WView gu = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.down_proj.weight", l);
    WView dn = wview(ckpt.at(buf));
    if (gu.dtype == 23) {  // hgn v2 i4r: own paths (q4cp fast paths do not apply)
      return moe_b_v2(l, gu, dn, acc, P, direct_output, try_hcw ? d_w4b : nullptr,
                      try_hcw ? (uint16_t*)d_Rb : nullptr);
    }
    // Shared expert first (its scratch d_guvb/d_hidb/d_xbf16b is dead before
    // the routed experts reuse it), so its add folds into the routed reduce.
    // QWENOX_MOE_SG_FUSE=0: old order + k_axpy_sg.
    static const bool sg_fuse = !env_zero("QWENOX_MOE_SG_FUSE");
    if (!moe_naive && P > moe_naive_max && sg_fuse && moe_q4w_ok(gu, dn, P)) {
      KP("moe:shared");
      moe_b_shared_ey(l, P);
      moe_q4w_run(gu, dn, d_xb, d_topidsb, d_topwb, acc, P, d_eyb, d_sgb,
                  try_hcw ? d_w4b : nullptr, try_hcw ? (uint16_t*)d_Rb : nullptr);
      if (try_hcw) return true;
      if (!direct_output)
        CK(hipMemcpyAsync(d_yb, acc, (size_t)P * g_cfg.d * 4, hipMemcpyDeviceToDevice,
                          g_str));
      return false;
    }
    if (!moe_naive && P > moe_naive_max &&
        moe_q4w_run(gu, dn, d_xb, d_topidsb, d_topwb, acc, P)) {
      KP("moe:shared");
      moe_b_shared(l, P, acc, direct_output);
      return false;
    }
    // fused kernels require exact tiling (guards would spill accumulators)
    bool fused_ok = g_cfg.moe_mid % MOE_CT == 0 && g_cfg.d % MOE_CT == 0 &&
                    g_cfg.d % FD_CT_DN == 0 &&
                    gu.cols % MOE_KC == 0 && dn.cols % MOE_KC == 0;
    bool naive_path = moe_naive || !fused_ok || P <= moe_naive_max;
    bool direct_down = naive_path && moe_down_topk && k == 10 && dn.cols == 640;
    // Direct-down and deterministic reductions overwrite every output element.
    // Atomic/ordered accumulation still requires a zero destination.
    if (!direct_down && (!direct_output || naive_path || moe_ordered || !moe_deterministic))
      CK(hipMemsetAsync(acc, 0, (size_t)P * g_cfg.d * 4, g_str));
    const int* topidsb = d_topidsb;
    if (naive_path) {
      // 同专家槽相邻化（k_moe_sort_slots 排序 + gg 查表，纯 warp 调度排列，逐 bit
      // 不变）。09-28 HQ γ=4 实测：verify gg 351→338µs/层，减去 sort 7.3µs/层后
      // 净 +0.26 ms/轮（0.3%）——Cache 层级已接住绝大部分重复专家读（未排序 gg 等效
      // 262 GB/s，已超 DRAM 冷流率），单序列 P=5 收益低于门槛。
      // 默认关，QWENOX_MOE_SORT=1 启用（留待 D2 合批 R≥16、工作集远超 Cache 的场景）。
      // P=1（串行/draft）单 token topk 专家天然不重复，跳过。
      static const bool moe_sort =
          getenv("QWENOX_MOE_SORT") != nullptr && !env_zero("QWENOX_MOE_SORT");
      const int* slot_order = nullptr;
      if (moe_sort && P >= 2 && P * k <= 256) {
        k_moe_sort_slots<<<1, 128, 0, g_str>>>(topidsb, P, k, 16, d_slot_order);
        slot_order = d_slot_order;
      }
      // v3 kernels: 2 rows per warp -> halved warp counts
      uint64_t pairs = (uint64_t)P * k * g_cfg.moe_mid;
      k_q4cp_gemv_gg<0><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
          d_xb, d_guvb, topidsb, (uint64_t)2 * g_cfg.moe_mid, gu.cols,
          gu.scale_stride, P * k, (uint64_t)g_cfg.d, 16, k, slot_order);
      k_silu_mul_g<<<(P * k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(
          d_guvb, d_hidb, g_cfg.moe_mid, P * k);
      if (direct_down) {
        uint64_t row_pairs = (g_cfg.d + 1) / 2;
        constexpr uint64_t warps_per_block = 128 / 32;
        k_q4cp_gemv_gd_topk_h16<10, 0><<<
            (unsigned)(((uint64_t)P * row_pairs + warps_per_block - 1) /
                       warps_per_block),
            128, 0, g_str>>>(
            dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
            (const float*)dn.data, d_hidb, acc, topidsb, d_topwb,
            (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
            (uint64_t)g_cfg.moe_mid, P, 16, (uint64_t)g_cfg.d);
      } else {
        uint64_t pairs2 = (uint64_t)P * k * ((g_cfg.d + 1) / 2);
        k_q4cp_gemv_gd<0><<<(unsigned)((pairs2 + 15) / 16), 512, 0, g_str>>>(
            dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
            (const float*)dn.data, d_hidb, acc, topidsb, d_topwb,
            (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
            (uint64_t)g_cfg.moe_mid, P * k, 16, (uint64_t)g_cfg.d, k);
      }
    } else {
      // Stable expert bucketing preserves token order and deterministic reduction.
      int E = g_cfg.experts, npairs = P * k;
      if (!moe_host_route && !moe_ordered) {
        moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff,
                        moe_deterministic ? d_pairids : nullptr, P, k, E);
      } else {
        std::vector<int> hids(P * 16);  // topids rows are 16 wide, first k valid
        std::vector<float> hws(P * 16);
        auto pt1 = std::chrono::steady_clock::now();
        CK(hipMemcpy(hids.data(), d_topidsb, (size_t)P * 16 * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hws.data(), d_topwb, (size_t)P * 16 * 4, hipMemcpyDeviceToHost));
        auto pt2 = std::chrono::steady_clock::now();
        std::vector<int> off(E + 1, 0);
        for (int t = 0; t < P; t++)
          for (int s = 0; s < k; s++) off[hids[t * 16 + s] + 1]++;
        for (int e = 0; e < E; e++) off[e + 1] += off[e];
        std::vector<int> cur(off.begin(), off.end() - 1);
        std::vector<int> tidx(npairs);
        std::vector<float> pw(npairs);
        for (int t = 0; t < P; t++)
          for (int s = 0; s < k; s++) {
            int slot = cur[hids[t * 16 + s]]++;
            tidx[slot] = t;
            pw[slot] = hws[t * 16 + s];
          }
        CK(hipMemcpy(d_tokidx, tidx.data(), npairs * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_pweight, pw.data(), npairs * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_eoff, off.data(), (E + 1) * 4, hipMemcpyHostToDevice));
        if (moe_deterministic && !moe_ordered) {
          std::vector<int> pairids(npairs), next(P, 0);
          for (int p = 0; p < npairs; p++) {
            int t = tidx[p];
            pairids[t * k + next[t]++] = p;
          }
          CK(hipMemcpy(d_pairids, pairids.data(), npairs * 4, hipMemcpyHostToDevice));
        }
        if (prof) {
          auto pt3 = std::chrono::steady_clock::now();
          p_topk_wait +=
              std::chrono::duration<double, std::milli>(pt2 - pt1).count();
          p_bucket += std::chrono::duration<double, std::milli>(pt3 - pt2).count();
        }
      }
      bool tiled = !moe_untiled && !moe_ordered;
      // Packed bf16 x/hid no longer requires the tiled grid: the untiled
      // expert-major grid resolves rows from eoff and is bit-identical.
      bool packed = !moe_ordered && moe_deterministic && !moe_fp32_io;
      int tasks = (npairs + 63) / 64 + E;
      bool lt_ran = false;
      if (moe_lt && k == 10 && moe_deterministic && !moe_host_route && !moe_ordered &&
          P >= moe_lt_min) {
        lt_ran = true;
        // Dequant + per-expert hipBLASLt path (see kernels' banner comment).
        // One host readback per layer to get expert offsets for the launch
        // loop. Async into a pinned buffer + event, issued BEFORE the
        // convert+gather launches: the host waits only for the 2KB copy
        // while the GPU works through gather (~16 ms), instead of the GPU
        // draining idle before a blocking sync (~5.5 ms/layer of GPU gap).
        const int mid = g_cfg.moe_mid, dd = g_cfg.d;
        static int* hoff_pin = nullptr;
        static hipEvent_t ev_eoff = nullptr;
        if (!hoff_pin) {
          CK(hipHostMalloc((void**)&hoff_pin, (size_t)(E + 1) * 4,
                           hipHostMallocDefault));
          CK(hipEventCreateWithFlags(&ev_eoff, hipEventDisableTiming));
        }
        CK(hipMemcpyAsync(hoff_pin, d_eoff, (E + 1) * 4, hipMemcpyDeviceToHost,
                          g_str));
        CK(hipEventRecord(ev_eoff, g_str));
        if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
          k_f32_to_bf16_v4<<<((size_t)P * dd / 4 + 255) / 256, 256, 0, g_str>>>(
              d_xb, d_xbf16, dd, P, dd);
          xf16_src = d_xb;
          xf16_xs = xf16_K = dd;
          xf16_P = P;
        }
        k_moe_gather_bf16<<<npairs, 256, 0, g_str>>>(
            d_moexg, (const __hip_bfloat16*)d_xbf16, d_tokidx, dd);
        CK(hipEventSynchronize(ev_eoff));
        const int* hoff = hoff_pin;
        bool b16 = moe_lt_bf16;
        __hip_bfloat16* guv16 = (__hip_bfloat16*)d_guvb;
        // up phase: group dequant -> per-expert GEMM -> segment silu
        // QWENOX_MOE_LT_OVL pipeline: scratch is double-buffered (slots gi&1);
        // group gi+1's dequant runs on ovl_str while g_str runs group gi's
        // GEMMs. ovl_deq[b] = deq into slot b done; ovl_gemm[b] = GEMMs that
        // read slot b done. deq(gi+1) waits on the group gi-1 GEMMs (the
        // previous reader of its slot). g_str never touches slot b before
        // waiting ovl_deq[b], and by loop end has waited on every ovl_str
        // deq, so the next phase/layer (whose ovl_str work chains behind
        // these deqs and g_str's later GEMM event records) cannot race the
        // slots either. Prefill only runs outside graph capture.
        static hipStream_t ovl_str = nullptr;
        static hipEvent_t ovl_deq[2] = {nullptr, nullptr};
        static hipEvent_t ovl_gemm[2] = {nullptr, nullptr};
        if (moe_lt_ovl && !ovl_str) {
          // Non-blocking: g_str is the legacy default stream outside graph
          // capture, and a blocking side stream would implicitly serialize
          // with it (measured: -16% at 8K, no overlap at all).
          CK(hipStreamCreateWithFlags(&ovl_str, hipStreamNonBlocking));
          for (int i = 0; i < 2; i++) {
            CK(hipEventCreateWithFlags(&ovl_deq[i], hipEventDisableTiming));
            CK(hipEventCreateWithFlags(&ovl_gemm[i], hipEventDisableTiming));
          }
        }
        const int ngrp = (E + 3) / 4;
        auto deq_up = [&](int gi, hipStream_t st) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          int64_t total = (int64_t)g * 2 * mid * (dd / 32);
          k_moe_deq_bf16<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
              gu.data + 64 + (size_t)e0 * 2 * mid * (dd / 2),
              gu.data + 64 + gu.rows * gu.cols / 2 +
                  (size_t)e0 * 2 * mid * gu.scale_stride,
              (const float*)gu.data,
              d_moewup + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * 2 * mid * dd,
              g * 2 * mid, dd, gu.scale_stride);
        };
        auto deq_dn = [&](int gi, hipStream_t st) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          int64_t total = (int64_t)g * dd * (mid / 32);
          k_moe_deq_bf16<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
              dn.data + 64 + (size_t)e0 * dd * (mid / 2),
              dn.data + 64 + dn.rows * dn.cols / 2 +
                  (size_t)e0 * dd * dn.scale_stride,
              (const float*)dn.data,
              d_moewdn + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * dd * mid,
              g * dd, mid, dn.scale_stride);
        };
        if (moe_lt_ovl) {
          deq_up(0, ovl_str);
          CK(hipEventRecord(ovl_deq[0], ovl_str));
        }
        for (int gi = 0; gi < ngrp; gi++) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          __hip_bfloat16* wup =
              d_moewup + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * 2 * mid * dd;
          if (moe_lt_ovl) {
            if (gi + 1 < ngrp) {
              if (gi >= 1)
                CK(hipStreamWaitEvent(ovl_str, ovl_gemm[(gi + 1) & 1], 0));
              deq_up(gi + 1, ovl_str);
              CK(hipEventRecord(ovl_deq[(gi + 1) & 1], ovl_str));
            }
            CK(hipStreamWaitEvent(g_str, ovl_deq[gi & 1], 0));
          } else {
            deq_up(gi, g_str);
          }
          for (int e = e0; e < e0 + g; e++) {
            int n0 = hoff[e], m = hoff[e + 1] - n0;
            if (!m) continue;
            void* cup = b16 ? (void*)(guv16 + (size_t)n0 * 2 * mid)
                            : (void*)(d_guvb + (size_t)n0 * 2 * mid);
            gemm_bf16(wup + (size_t)(e - e0) * 2 * mid * dd,
                      d_moexg + (size_t)n0 * dd, cup, 2 * mid, dd, m, b16);
            int64_t seg = (int64_t)m * mid;
            if (b16)
              k_moe_silu_bf16<<<(unsigned)((seg + 255) / 256), 256, 0, g_str>>>(
                  guv16 + (size_t)n0 * 2 * mid,
                  (__hip_bfloat16*)d_hidb + (size_t)n0 * mid, mid, m);
            else
              k_moe_silu_f32_bf16<<<(unsigned)((seg + 255) / 256), 256, 0, g_str>>>(
                  d_guvb + (size_t)n0 * 2 * mid,
                  (__hip_bfloat16*)d_hidb + (size_t)n0 * mid, mid, m);
          }
          if (moe_lt_ovl) CK(hipEventRecord(ovl_gemm[gi & 1], g_str));
        }
        if (moe_lt_cmp && l == 0) {
          // Reference: fused packed up on identical inputs -> d_hidb2.
          k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 64);
          k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0,
                                       g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2,
              (const float*)gu.data, d_xb, d_tokidx, d_eoff, nullptr, 2 * mid,
              gu.cols, mid, gu.scale_stride, d_moetiles, d_moentiles, d_xbf16,
              (uint16_t*)d_hidb2);
          CK(hipMemsetAsync(d_cmpi, 0, 8, g_str));
          int64_t nh = (int64_t)npairs * mid;
          k_cmp_absmax_bf16<<<(unsigned)std::min<int64_t>(1024, (nh + 255) / 256),
                              256, 0, g_str>>>(
              (const __hip_bfloat16*)d_hidb, (const __hip_bfloat16*)d_hidb2, nh,
              d_cmpi);
          int hc[2];
          CK(hipMemcpy(hc, d_cmpi, 8, hipMemcpyDeviceToHost));
          float fd, fr;
          memcpy(&fd, &hc[0], 4);
          memcpy(&fr, &hc[1], 4);
          fprintf(stderr, "moe_lt cmp up-hid: max|d|=%g max|ref|=%g\n", fd, fr);
        }
        // down phase: d_guvb is dead after silu; pair outputs alias it.
        if (moe_lt_ovl) {
          deq_dn(0, ovl_str);
          CK(hipEventRecord(ovl_deq[0], ovl_str));
        }
        for (int gi = 0; gi < ngrp; gi++) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          __hip_bfloat16* wdn =
              d_moewdn + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * dd * mid;
          if (moe_lt_ovl) {
            if (gi + 1 < ngrp) {
              if (gi >= 1)
                CK(hipStreamWaitEvent(ovl_str, ovl_gemm[(gi + 1) & 1], 0));
              deq_dn(gi + 1, ovl_str);
              CK(hipEventRecord(ovl_deq[(gi + 1) & 1], ovl_str));
            }
            CK(hipStreamWaitEvent(g_str, ovl_deq[gi & 1], 0));
          } else {
            deq_dn(gi, g_str);
          }
          for (int e = e0; e < e0 + g; e++) {
            int n0 = hoff[e], m = hoff[e + 1] - n0;
            if (!m) continue;
            void* cdn = b16 ? (void*)(guv16 + (size_t)n0 * dd)
                            : (void*)(d_guvb + (size_t)n0 * dd);
            gemm_bf16(wdn + (size_t)(e - e0) * dd * mid,
                      (const __hip_bfloat16*)d_hidb + (size_t)n0 * mid, cdn, dd,
                      mid, m, b16);
          }
          if (moe_lt_ovl) CK(hipEventRecord(ovl_gemm[gi & 1], g_str));
        }
        if (moe_lt_cmp && l == 0 && b16) {
          // Reference: fused packed down on LT's hid -> weighted fp32 pairs.
          k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks),
                                             256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
              (const float*)dn.data, nullptr, d_tokidx, d_pweight, d_eoff,
              d_pairs2, dd, dn.cols, dn.scale_stride, d_moetiles, d_moentiles,
              (const uint16_t*)d_hidb);
          CK(hipMemsetAsync(d_cmpi, 0, 8, g_str));
          int64_t npr = (int64_t)npairs * dd;
          k_cmp_absmax_pw<<<(unsigned)std::min<int64_t>(1024, (npr + 255) / 256),
                            256, 0, g_str>>>(
              d_pairs2, (const __hip_bfloat16*)d_guvb, d_pweight, npairs, dd,
              d_cmpi);
          int hc[2];
          CK(hipMemcpy(hc, d_cmpi, 8, hipMemcpyDeviceToHost));
          float fd, fr;
          memcpy(&fd, &hc[0], 4);
          memcpy(&fr, &hc[1], 4);
          fprintf(stderr, "moe_lt cmp down-pairs: max|d|=%g max|ref|=%g\n", fd,
                  fr);
        }
        if (b16)
          k_moe_reduce_pw_bf16<10><<<dim3((dd + 255) / 256, P), 256, 0, g_str>>>(
              (const __hip_bfloat16*)d_guvb, d_pairids, d_pweight, acc, P, dd);
        else
          k_moe_reduce_pw<10><<<dim3((dd + 255) / 256, P), 256, 0, g_str>>>(
              d_guvb, d_pairids, d_pweight, acc, P, dd);
      } else if (tiled) {
        k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 64);
        if (packed) {
          if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
            k_f32_to_bf16_v4<<<((size_t)P * g_cfg.d / 4 + 255) / 256, 256, 0, g_str>>>(
                d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
            xf16_src = d_xb;
            xf16_xs = xf16_K = g_cfg.d;
            xf16_P = P;
          }
          // Hidden BF16 occupies the existing workspace until the shared expert rewrites it.
          if (moe_up_legacy)
            k_moe_w4_up<true, true, false, false><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
          else if (moe_up_no_table)
            k_moe_w4_up<true, true, true, false><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
          else
            k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
        } else {
          k_moe_w4_up<true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, d_hidb, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride, d_moetiles, d_moentiles);
        }
      } else {
        if (packed) {
          if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
            k_f32_to_bf16_v4<<<((size_t)P * g_cfg.d / 4 + 255) / 256, 256, 0, g_str>>>(
                d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
            xf16_src = d_xb;
            xf16_xs = xf16_K = g_cfg.d;
            xf16_P = P;
          }
          k_moe_w4_up<false, true><<<dim3(g_cfg.moe_mid / MOE_CT, E), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride, nullptr, nullptr, d_xbf16, (uint16_t*)d_hidb);
        } else {
          k_moe_w4_up<<<dim3(g_cfg.moe_mid / MOE_CT, E), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, d_hidb, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride);
        }
      }
      if (lt_ran) {
        // LT branch already produced acc (down + weighted reduce).
      } else if (moe_ordered) {
        for (int e = 0; e < E; e++)
          k_moe_w4_down<<<dim3(g_cfg.d / FD_CT_DN, 1), 256, 0, g_str>>>(
              dn.data + 64 + (size_t)e * g_cfg.d * dn.cols / 2,
              dn.data + 64 + dn.rows * dn.cols / 2 + (size_t)e * g_cfg.d * dn.scale_stride,
              (const float*)dn.data, d_hidb, d_tokidx, d_pweight, d_eoff + e,
              acc, g_cfg.d, dn.cols, dn.scale_stride);
      } else if (moe_deterministic) {
        // QWENOX_MOE_PAIRS_BF16: down writes unweighted bf16 pair rows; the
        // router weight moves to the reduce (fp32 multiply). Requires the
        // fixed top-10 reduce below.
        bool pairs_bf16 = moe_pairs_bf16 && k == 10 && !moe_reduce_old;
        if (packed) {
          if (tiled) {
            if (moe_down_no_table) {
              if (pairs_bf16)
                k_moe_w4_down<false, true, true, false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
              else
                k_moe_w4_down<false, true, true, false><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
            } else {
              if (pairs_bf16)
                k_moe_w4_down<false, true, true, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
              else
                k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
            }
          } else {
            if (pairs_bf16)
              k_moe_w4_down<false, false, true, true, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                  dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                  nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                  dn.scale_stride, nullptr, nullptr, (const uint16_t*)d_hidb);
            else
              k_moe_w4_down<false, false, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                  dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                  nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                  dn.scale_stride, nullptr, nullptr, (const uint16_t*)d_hidb);
          }
        } else if (tiled) {
          if (pairs_bf16)
            k_moe_w4_down<false, true, false, false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride, d_moetiles, d_moentiles);
          else
            k_moe_w4_down<false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride, d_moetiles, d_moentiles);
        } else {
          if (pairs_bf16)
            k_moe_w4_down<false, false, false, false, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride);
          else
            k_moe_w4_down<false><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride);
        }
        if (pairs_bf16)
          k_moe_reduce_pw_bf16<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
              (const __hip_bfloat16*)d_guvb, d_pairids, d_pweight, acc, P, g_cfg.d);
        else if (k == 10 && !moe_reduce_old)
          k_moe_reduce_fast<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
              d_guvb, d_pairids, acc, P, g_cfg.d);
        else
          k_moe_reduce<<<(P * g_cfg.d + 255) / 256, 256, 0, g_str>>>(
              d_guvb, d_pairids, acc, P, k, g_cfg.d);
      } else {
        if (tiled) {
          k_moe_w4_down<true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
              d_hidb, d_tokidx, d_pweight, d_eoff, acc, g_cfg.d, dn.cols,
              dn.scale_stride, d_moetiles, d_moentiles);
        } else {
          k_moe_w4_down<<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
              d_hidb, d_tokidx, d_pweight, d_eoff, acc, g_cfg.d, dn.cols,
              dn.scale_stride);
        }
      }
    }
    moe_b_shared(l, P, acc, direct_output);
    return false;
  }

  // hgn v2 i4r MoE of trunk layer l for P rows of d_xb (topk in d_topidsb /
  // d_topwb): routed + shared expert -> acc f32 [P][d] (then d_yb unless
  // direct_output). Same dataflow as moe_v2_1; small P (verify) runs the
  // grouped GEMVs, larger P the kI4R LUT WMMA kernels on the moe_q4w_run
  // workspaces (moe_lut_up_mid writes the F16 down input straight to d_hidb;
  // d_guvb takes the down pairs). LUT path: the
  // shared expert runs first (its d_guvb / d_hidb scratch is dead by then) and
  // k_v2_reduce_rot folds reduce + rot_out + shared add into one pass
  // (QWENOX_MOE_SG_FUSE=0: shared expert afterwards via k_axpy_sg).
  // w4hc/Rhc: fuse the HC write-back into k_v2_reduce_rot_hc (B2-1) and
  // return true; only on the LUT sg_fuse path.
  bool moe_b_v2(int l, const WView& gu, const WView& dn, float* acc, int P,
                bool direct_output, const float* w4hc = nullptr, uint16_t* Rhc = nullptr) {
    const int k = g_cfg.topk, E = g_cfg.experts, d = g_cfg.d, mid = g_cfg.moe_mid;
    if (gu.cols != (uint64_t)d || gu.rows != (uint64_t)E * 2 * mid || dn.dtype != 23 ||
        dn.cols != (uint64_t)mid || dn.rows != (uint64_t)E * d || mid % 128 || d % 128 ||
        2 * mid > 2048)
      throw std::runtime_error("moe_b_v2: unexpected i4r expert shapes");
    const unsigned mid_threads = 2 * mid / 128 * 32;
    const moelut::LutW pgu = lut_i4r_view(gu.data, gu.rows, gu.cols, g_i4r_cb, 2 * mid, mid);
    const moelut::LutW pdn = lut_i4r_view(dn.data, dn.rows, dn.cols, g_i4r_cb, d, 0);
    const bool lut = moe_q4w && !moe_naive && P > moe_naive_max && k == 10 && moe_deterministic &&
                     !moe_host_route && !moe_ordered && !moe_untiled && P <= maxbatch_cap &&
                     mid % 64 == 0 &&
                     moelut::lut_ok(moelut::kI4R, pgu, true, d) &&
                     moelut::lut_ok(moelut::kI4R, pdn, false, mid);
    if (lut) {
      static bool logged = false;
      if (!logged) {
        logged = true;
        fprintf(stderr, "moe prefill: hgn v2 i4r LUT WMMA\n");
      }
      static const bool sg_fuse = !env_zero("QWENOX_MOE_SG_FUSE");
      KP("moe:shared");
      if (sg_fuse) moe_b_shared_ey(l, P);
      KP("moe:route");
      moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff, d_pairids, P, k, E);
      // §11.5 多 tile 共享解码（mt）：tile 按 128 slot 合并（同专家两个 64-slot
      // tile 一个 CTA 解码一遍），逐 bit 不变；默认开，QWENOX_MOE_MT=0 回退。
      // microbench：P=2048 亏 5%、4096 +1.8%、8192 +7.5%、16384 +5.5%——按平均
      // 每专家 slot 数 >= 80 设门槛，小 P（tail chunk / 短 prompt）走原路径。
      static const bool moe_mt_env = !env_zero("QWENOX_MOE_MT");
      const bool mt = moe_mt_env && (int64_t)P * k >= 80 * E;
      k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, mt ? 128 : 64);
      const int max_tiles = mt ? (P * k + 127) / 128 + E : (P * k + 63) / 64 + E;
      __half* x16 = (__half*)d_xbf16b;
      __half* hid = (__half*)d_hidb;
      KP("moe:rot_in");
      v2_rot(d_xb, d, x16, d, gu.suh, nullptr, d, P, true);
      KP("moe:up");
      // up + mid fused (方向 3): hid = fwht(silu(g)*u*suh) 直接落盘，
      // 省掉 guv16 的写读往返和 k_i4r_mid 那次 launch，数学逐位一致。
      if (mt)
        moe_lut_up_mid_mt(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, gu.svh,
                          dn.suh, mid, d, g_str);
      else
        moe_lut_up_mid(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, gu.svh, dn.suh,
                       mid, d, g_str);
      KP("moe:mid");
      KP("moe:down");
      // f16 pairs（down 写/reduce 读，v2-only，省一半 pairs 往返流量）。
      // 实测（Windows, 2026-10-02）：reduce −148 ms/16K，down 几乎不变，整体
      // +1.2~1.5%。终判（10-03，Linux BF16 基准 data/kld/bf16_c512.kld 64×512）：
      // mean_kld 0.106993(关) → 0.107705(开)，门槛 0.10699+0.002=0.10899 过线，
      // 默认开；QWENOX_V2_PAIRS_F16=0 回退 fp32 pairs。
      static const bool pairs_f16 = !env_zero("QWENOX_V2_PAIRS_F16");
      if (mt)
        moe_lut_down_mt(pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid, g_str,
                        pairs_f16);
      else
        moe_lut_down(moelut::kI4R, pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid,
                     g_str, pairs_f16);
      KP("moe:reduce");
      const unsigned waves = (unsigned)P * (d / 128);
      if (sg_fuse && Rhc) {
        if (pairs_f16)
          k_v2_reduce_rot_hc<10, __half><<<(waves + 7) / 8, 256, 0, g_str>>>(
              (const __half*)d_guvb, d_pairids, d_pweight, dn.svh, d_eyb, d_sgb, w4hc,
              Rhc, P, d);
        else
          k_v2_reduce_rot_hc<10><<<(waves + 7) / 8, 256, 0, g_str>>>(
              d_guvb, d_pairids, d_pweight, dn.svh, d_eyb, d_sgb, w4hc, Rhc, P, d);
        KP("moe:tail");
        return true;
      }
      if (pairs_f16)
        k_v2_reduce_rot<10, __half><<<(waves + 7) / 8, 256, 0, g_str>>>(
            (const __half*)d_guvb, d_pairids, d_pweight, dn.svh, sg_fuse ? d_eyb : nullptr,
            d_sgb, acc, P, d);
      else
        k_v2_reduce_rot<10><<<(waves + 7) / 8, 256, 0, g_str>>>(
            d_guvb, d_pairids, d_pweight, dn.svh, sg_fuse ? d_eyb : nullptr, d_sgb, acc, P, d);
      KP("moe:tail");
      if (sg_fuse) {
        if (!direct_output)
          CK(hipMemcpyAsync(d_yb, acc, (size_t)P * d * 4, hipMemcpyDeviceToDevice, g_str));
      } else {
        moe_b_shared(l, P, acc, direct_output);
      }
      return false;
    } else {
      const XCodes cg = xcodes(gu), cd = xcodes(dn);
      const bool direct_down = moe_down_topk && k == 10;
      if (!direct_down) CK(hipMemsetAsync(acc, 0, (size_t)P * d * 4, g_str));
      v2_rot(d_xb, d, d_eyb, d, gu.suh, nullptr, d, P);  // d_eyb: free until the shared expert
      const uint64_t pairs = (uint64_t)P * k * mid;
      k_q4cp_gemv_gg<2><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          cg.codes, cg.scales, cg.cb, d_eyb, d_guvb, d_topidsb, (uint64_t)2 * mid, gu.cols,
          gu.scale_stride, P * k, (uint64_t)d, 16, k, nullptr);
      k_i4r_mid<float, false><<<P * k, mid_threads, 0, g_str>>>(d_guvb, 2 * mid, d_hidb, mid,
                                                                gu.svh, dn.suh, mid);
      if (direct_down) {
        const uint64_t row_pairs = (d + 1) / 2, warps_per_block = 128 / 32;
        k_q4cp_gemv_gd_topk_h16<10, 2><<<
            (unsigned)(((uint64_t)P * row_pairs + warps_per_block - 1) / warps_per_block), 128, 0,
            g_str>>>(cd.codes, cd.scales, cd.cb, d_hidb, acc, d_topidsb, d_topwb, (uint64_t)d,
                     dn.cols, dn.scale_stride, (uint64_t)mid, P, 16, (uint64_t)d);
      } else {
        const uint64_t pairs2 = (uint64_t)P * k * ((d + 1) / 2);
        k_q4cp_gemv_gd<2><<<(unsigned)((pairs2 + 15) / 16), 512, 0, g_str>>>(
            cd.codes, cd.scales, cd.cb, d_hidb, acc, d_topidsb, d_topwb, (uint64_t)d, dn.cols,
            dn.scale_stride, (uint64_t)mid, P * k, 16, (uint64_t)d, k);
      }
    }
    v2_rot(acc, d, acc, d, nullptr, dn.svh, d, P);
    moe_b_shared(l, P, acc, direct_output);
    return false;
  }

  // Shared expert of trunk layer l added into acc (routed sum), then acc -> d_yb.
  void moe_b_shared(int l, int P, float* acc, bool direct_output) {
    moe_b_shared_ey(l, P);
    k_axpy_sg<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(acc, d_eyb, d_sgb,
                                                         g_cfg.d, P * g_cfg.d);
    if (!direct_output)
      CK(hipMemcpyAsync(d_yb, acc, (size_t)P * g_cfg.d * 4, hipMemcpyDeviceToDevice,
                        g_str));
  }
  // Shared expert output d_eyb [P][d] and gate logit d_sgb [P].
  void moe_b_shared_ey(int l, int P) {
    char buf[128];
    // shared expert (gate/up scratch in d_guvb head, silu out in d_hidb head)
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.gate_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_guvb, P);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.up_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_guvb + (size_t)P * 640, P);
    k_silu_mul<<<(P * 640 + 255) / 256, 256, 0, g_str>>>(d_guvb, d_guvb + (size_t)P * 640,
                                                         d_hidb, P * 640);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.down_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_hidb, d_eyb, P);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert_gate.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_sgb, P);
  }

  // PLE device phase for one prefill token (embeddings already in d_eb).
  void ple_gpu_t(int t, int base = 0) {
    const float* e = d_eb + (size_t)t * 2560;
    gemv(wview(ckpt.at("layers.1.ple.key_proj.weight")), e, d_keys);
    gemv(wview(ckpt.at("layers.1.ple.value_proj.weight")), e, d_v);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_keys, d_ple_nk, d_kn, 2560,
                                                g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_Rb + (size_t)t * 10240,
                                                d_ple_nk + 10240, d_qn, 2560,
                                                g_cfg.norm_eps, 4);
    k_ple_gate<<<4, 1024, 0, g_str>>>(d_kn, d_qn, d_v, d_U, 2560, 4);
    k_rmsnorm_zc_ring<<<4, 1024, 0, g_str>>>(d_U, d_ple_nk + 20480, d_ring,
                                             d_ringarr + base + t, 2560,
                                             g_cfg.norm_eps);
    k_ple_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_ring, d_ringarr + base + t,
                                                       d_ple_convw, d_convout, 10240);
    k_add2<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_Rb + (size_t)t * 10240, d_U,
                                                   d_convout, 10240);
  }

  // Batched PLE over all P positions (embeddings already in d_eb). Bitwise
  // matches the ple_gpu_t loop: same per-block reductions, ring semantics of
  // k_ple_conv (tap back-9 aliases the current slot) folded into k_ple_conv_b.
  void ple_gpu_b(int P, int base = 0) {
    gemm(wview(ckpt.at("layers.1.ple.key_proj.weight")), d_eb, d_keysb, P);
    gemm(wview(ckpt.at("layers.1.ple.value_proj.weight")), d_eb, d_plevb, P);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_keysb, d_ple_nk, d_keysb, 2560,
                                                    g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_Rb, d_ple_nk + 10240, d_qnb,
                                                    2560, g_cfg.norm_eps, 4);
    k_ple_gate_b<<<dim3(4, P), 1024, 0, g_str>>>(d_keysb, d_qnb, d_plevb, d_Ub, 2560,
                                                 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_Ub, d_ple_nk + 20480, d_Unb,
                                                    2560, g_cfg.norm_eps, 4);
    int64_t tot = (int64_t)P * 10240;
    k_ple_conv_b<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
        d_Unb, d_ple_convw, d_convoutb, P, 10240, base, d_ring, vbptr());
    k_add2<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(d_Rb, d_Ub, d_convoutb,
                                                             (int)tot);
    k_ple_ring_wr<<<9, 256, 0, g_str>>>(d_Unb, d_ring, P, 10240, base, vbptr());
    if (cut_r_)  // in-pass cut: PLE ring after the first r rows
      k_ple_ring_wr<<<9, 256, 0, g_str>>>(d_Unb, d_cring, cut_r_, 10240, base, vbptr());
  }

  // P3-B: the layer's HC read + attention norm as a standalone step. A
  // verify-graph segment boundary runs this eagerly ahead of qsa_idx_b,
  // because the indexer reads d_xb, which this norm writes (it also carries
  // layer l-1's deferred MLP write-back under gr_scatter_norm).
  void enq_layer_norm_b(int l, int P, bool fuse_prev_mlp) {
    char attn_prefix[128], mlp_prefix[128];
    snprintf(attn_prefix, sizeof attn_prefix,
             "layers.%d.attn_hyper_connection", l);
    KP("hc");
    if (fuse_prev_mlp) {
      snprintf(mlp_prefix, sizeof mlp_prefix,
               "layers.%d.mlp_hyper_connection", l - 1);
      gr_write_read_b(mlp_prefix, attn_prefix, P);
    } else {
      gr_read_b(attn_prefix, P);
    }
    if (getenv("QWENOX_VGDUMP")) {
      char tag[12];
      snprintf(tag, sizeof tag, "nx%d", l);
      vg_dump_buf(tag, P, d_xb, (size_t)P * 2560 * 4);
      snprintf(tag, sizeof tag, "xn%d", l);
      vg_dump_buf(tag, P, d_xbf16, (size_t)P * 2560 * 2);
      snprintf(tag, sizeof tag, "rn%d", l);
      vg_dump_buf(tag, P, d_Rhatbf16, (size_t)P * 10240 * 2);
      snprintf(tag, sizeof tag, "gb%d", l);
      vg_dump_buf(tag, P, d_Gb, (size_t)P * 10240 * 4);
    }
  }

  void enq_layer_b(int l, int P, bool fuse_prev_mlp = false,
                   bool flush_mlp = true, int base = 0,
                   const int* dvb = nullptr, bool skip_qsa_idx = false) {
    char attn_prefix[128], mlp_prefix[128];
    snprintf(attn_prefix, sizeof attn_prefix,
             "layers.%d.attn_hyper_connection", l);
    // skip_qsa_idx (P3-B): norm + indexer ran eagerly between graph segments
    // (enq_layer_norm_b + qsa_idx_b); start at the attention projections.
    // Callers pass it only for QSA layers.
    if (!skip_qsa_idx) enq_layer_norm_b(l, P, fuse_prev_mlp);
    KP(is_qsa(l) ? "qsa" : "gdn");
    if (is_qsa(l)) {
      if (skip_qsa_idx)  // P3-B: indexer ran eagerly between graph segments
        qsa_main_b(l, l / 4, P, base, dvb);
      else
        qsa_b(l, l / 4, P, base, dvb);
    } else
      gdn_b(l, l - (l + 1) / 4, P);
    if (dbg) {
      h_rms("gpu", l, "attn-x", d_xb + (size_t)(P - 1) * 2560, 2560);
      h_rms("gpu", l, "attn-y", d_yb + (size_t)(P - 1) * 2560, 2560);
    }
    snprintf(mlp_prefix, sizeof mlp_prefix,
             "layers.%d.mlp_hyper_connection", l);
    KP("hc2");
    if (gr_scatter_norm)
      gr_write_read_b(attn_prefix, mlp_prefix, P);
    else {
      gr_write_b(attn_prefix, P);
      gr_read_b(mlp_prefix, P);
    }
    KP("moe");
    // B2-1 (QWENOX_HC_FUSE_WRITE, 默认关 — KPROF 实测回退，见 HANDOFF-PREFILL-ALL
    // §4 B2-1 状态段): fold the mlp site's HC write-back into the moe reduce
    // (saves the d_yb f32 write + re-read; R bit-identical). Requires the
    // gr_bf16 write path; the deferred (flush_mlp=false) consumer additionally
    // needs the inj-fused scatter's Write=false mode. moe_b fuses only on the
    // LUT sg_fuse paths and reports back; otherwise the precomputed d_w4b is
    // recomputed bit-identically by the normal write path.
    static const bool hcw_env =
        getenv("QWENOX_HC_FUSE_WRITE") != nullptr && !env_zero("QWENOX_HC_FUSE_WRITE");
    const bool can_hcw = hcw_env && !dbg && gr_bf16 &&
                         (flush_mlp || (gr_scatter_norm && inj_fuse_on()));
    bool hcw = false;
    if (can_hcw) {
      gr_write_prep_b(mlp_prefix, P);
      hcw = moe_b(l, P, true);
    } else {
      moe_b(l, P);
    }
    if (dbg) {
      h_rms("gpu", l, "mlp-x", d_xb + (size_t)(P - 1) * 2560, 2560);
      h_rms("gpu", l, "mlp-y", d_yb + (size_t)(P - 1) * 2560, 2560);
    }
    KP("hc3");
    if (flush_mlp) {
      if (!hcw) gr_write_b(mlp_prefix, P);
    } else if (hcw) {
      hcw_site_ = mlp_prefix;
    }
    if (!dbg && !bench_quiet && isatty(2)) {  // per-layer \r is terminal-only; logs use prefill-live
      fprintf(stderr, "\rprefill L%02d/%d", l + 1, g_cfg.layers);
      fflush(stderr);
    }
  }

  // Batched prefill: returns first generated token (argmax at last position).
  // Batched prefill: returns first generated token (argmax at last position).
  // base = absolute position of tokens[0] (KV-continuation); state handoff:
  // d_R = last row, pos = base+P, d_pos/d_ringpos absolute, chist/ring/KV/GDN
  // state fully advanced.
  // Chunked prefill: prompts longer than maxbatch are processed in
  // maxbatch-token continuations (the same path KV reuse uses), so batch
  // workspace stays chunk-sized regardless of maxctx.
  // 1s-cadence prefill progress: record a completion probe after layer l's
  // enqueue, and once a second report the highest COMPLETED layer as an
  // estimated token count. hipEventQuery is non-blocking (hipErrorNotReady
  // is the expected answer for in-flight layers), so this never serializes
  // the pipeline the way the QWENOX_PHASE sync profiling does.
  void prog_print(int P) {
    auto now = std::chrono::steady_clock::now();
    double iv = std::chrono::duration<double>(now - prog_last).count();
    if (iv < 1.0) return;
    int done = -1;
    for (int k = g_cfg.layers - 1; k >= 0; --k)
      if (prog_ev[k] && hipEventQuery(prog_ev[k]) == hipSuccess) {
        done = k;
        break;
      }
    // done+1 of g_cfg.layers layers finished (L0 counts as one layer).
    double frac = double(done + 1) / g_cfg.layers;
    int64_t est = prog_base + (int64_t)(P * frac);
    double el = std::chrono::duration<double>(now - prog_t0).count();
    fprintf(stderr,
            "%s req %lld | prompt processing | n_tokens = %lld/%d (%.0f%%) "
            "| t = %.1f s | %.0f tok/s (avg %.0f)\n",
            log_ts(), (long long)prog_req, (long long)est, prog_total,
            100.0 * est / prog_total, el, (est - prog_last_est) / iv,
            el > 0 ? est / el : 0.0);
    perfsnap::write_prefill((uint64_t)prog_req, (est - prog_last_est) / iv,
                            (uint64_t)est, (uint64_t)prog_total);
    prog_last = now;
    prog_last_est = est;
  }
  void prog_probe(int l, int P) {
    if (l < 0 || l >= 64) return;
    if (!prog_ev[l])
      CK(hipEventCreateWithFlags(&prog_ev[l], hipEventDisableTiming));
    CK(hipEventRecord(prog_ev[l], g_str));
    prog_print(P);
  }
  // The layer enqueue is fully asynchronous: every probe above fires within
  // the first fraction of a second of host time, then the host blocks on the
  // final readback. Real progress is therefore reported while WAITING for
  // the graph to drain — poll the last layer's event here at 1 s cadence.
  void prog_wait(int P) {
    if (!prog_ev[g_cfg.layers - 1]) return;
    // 1 ms poll (was 20 ms: up to 20 ms of TTFT per served chunk end, i.e. at
    // the snapshot cut and again at the prompt end). prog_print self-limits
    // to its 1 s cadence.
    while (hipEventQuery(prog_ev[g_cfg.layers - 1]) == hipErrorNotReady) {
      prog_print(P);
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
  }

  // ---- PLE gather flights ---------------------------------------------------
  // cat() window for a flight: token at window position t-back, falling back
  // to the two history tokens carried with the flight (pad_id at sequence
  // start). Matches the chist[ple_base0 + t - back] reads of the old inline
  // gather exactly.
  static int64_t ple_cat(const PleFlight& fl, int t, int back) {
    int j = t - back;
    return j >= 0 ? fl.toks[j] : (j == -1 ? fl.pre0 : fl.pre1);
  }
  // per-token NOSORT fallback, window variant of ple_rows_fp8.
  void ple_rows_fp8_win(const PleFlight& fl, int t, uint8_t* dst) {
    const uint8_t* addrs[16];
    for (int h = 0; h < 16; h++) {
      int64_t mix = ple_cat(fl, t, 0) * ple_mult[0] ^ (ple_cat(fl, t, 1) * ple_mult[1]);
      if (h >= 8) mix ^= ple_cat(fl, t, 2) * ple_mult[2];
      uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
      const uint8_t* p = ple_table->data + row * g_ple_rb;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, dst, (uint64_t)(uintptr_t)ple_table->data, 16);
      return;
    }
#endif
    for (int h = 0; h < 16; h++) memcpy(dst + h * g_ple_rb, addrs[h], g_ple_rb);
  }
  // Spawn the disk gather for toks (global position base) into staging fp8.
  // Same three pipelines as the old inline gather (IOCP/io_uring, madvise
  // two-phase, NOSORT per-token); the flight owns all worker-visible state so
  // it can be joined from a later prefill_chunk call.
  void ple_spawn(std::vector<int>&& toks, int base, int64_t pre0, int64_t pre1,
                 uint8_t* fp8, int buf) {
    PleFlight& fl = ple_fl;
    fl.toks = std::move(toks);
    fl.base = base;
    fl.P = (int)fl.toks.size();
    fl.pre0 = pre0;
    fl.pre1 = pre1;
    fl.fp8 = fp8;
    fl.buf = buf;
    fl.start = std::chrono::steady_clock::now();
    if (ple_stg_ev_ok[buf]) {
      // Staging was last consumed by an earlier chunk's dequant; in steady
      // state that kernel finished two chunks ago, so this never blocks.
      CK(hipEventSynchronize(ple_stg_ev[buf]));
    }
    const int P = fl.P;
    PleFlight* f = &fl;
    constexpr int nt = 16;
    fl.workers.store(nt);
    fl.pool.clear();
    fl.pool.reserve(nt);
    if (!getenv("QWENOX_PLE_NOSORT")) {
      fl.barrier.store(0, std::memory_order_relaxed);
      if (ple_uring_ready()) {
        // IOCP/io_uring batch-pread gather: phase A is the same 16-thread row
        // address computation; the pump threads then push all 16*P rows in
        // token order through the deep rings straight into pinned staging
        // (same multi-ring split as the inline gather; tiny P keeps one ring).
        fl.addrs.resize((size_t)P * 16);
        const int npump = P >= 64 ? std::min(nt, ple_pump_count()) : 1;
        fl.workers.store(npump);  // only the pump threads account for fl.done
        for (int w = 0; w < nt; w++)
          fl.pool.emplace_back([this, f, w, npump] {
            const int t0 = (int)((size_t)w * f->P / nt),
                      t1 = (int)((size_t)(w + 1) * f->P / nt);
            for (int t = t0; t < t1; t++)
              for (int h = 0; h < 16; h++) {
                int64_t mix = ple_cat(*f, t, 0) * ple_mult[0] ^
                              (ple_cat(*f, t, 1) * ple_mult[1]);
                if (h >= 8) mix ^= ple_cat(*f, t, 2) * ple_mult[2];
                uint64_t row =
                    (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
                f->addrs[(size_t)t * 16 + h] =
                    (uint64_t)(uintptr_t)(ple_table->data + row * g_ple_rb);
              }
            f->barrier.fetch_add(1, std::memory_order_acq_rel);
            if (w >= npump) return;
            while (f->barrier.load(std::memory_order_acquire) < nt)
              std::this_thread::yield();
            const size_t nrow = (size_t)f->P * 16;
            const size_t r0 = nrow * w / npump, r1 = nrow * (w + 1) / npump;
            PLE_PUMP_I(w).run(f->addrs.data() + r0, f->fp8 + r0 * g_ple_rb,
                              (uint64_t)(uintptr_t)ple_table->data, r1 - r0);
            if (f->workers.fetch_sub(1) == 1)
              f->done = std::chrono::steady_clock::now();
          });
      } else {
        // Two-phase gather: phase A computes all 16*P row addresses, sorts
        // each shard by page and issues madvise in ascending order; phase B
        // copies rows in token order with pages already in flight. Output
        // staging bytes are identical to the per-token pipeline.
        fl.addrs.resize((size_t)P * 16);
        fl.keys.resize((size_t)P * 16);
        for (int w = 0; w < nt; w++)
          fl.pool.emplace_back([this, f, w] {
            const int t0 = (int)((size_t)w * f->P / nt),
                      t1 = (int)((size_t)(w + 1) * f->P / nt);
            for (int t = t0; t < t1; t++)
              for (int h = 0; h < 16; h++) {
                int64_t mix = ple_cat(*f, t, 0) * ple_mult[0] ^
                              (ple_cat(*f, t, 1) * ple_mult[1]);
                if (h >= 8) mix ^= ple_cat(*f, t, 2) * ple_mult[2];
                uint64_t row =
                    (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
                f->addrs[(size_t)t * 16 + h] =
                    (uint64_t)(uintptr_t)(ple_table->data + row * g_ple_rb);
              }
            uint64_t* kb = f->keys.data() + (size_t)t0 * 16;
            std::memcpy(kb, f->addrs.data() + (size_t)t0 * 16,
                        (size_t)(t1 - t0) * 16 * 8);
            std::sort(kb, kb + (size_t)(t1 - t0) * 16);
            uint64_t prev = ~0ull;
            for (int64_t i = 0; i < (int64_t)(t1 - t0) * 16; i++) {
              uint64_t pg = kb[i] & ~4095ull;
              if (pg != prev) {
                madvise((void*)(uintptr_t)pg, 4096, MADV_WILLNEED);
                prev = pg;
              }
            }
            f->barrier.fetch_add(1, std::memory_order_acq_rel);
            while (f->barrier.load(std::memory_order_acquire) < nt)
              std::this_thread::yield();
            for (int t = t0; t < t1; t++) {
              uint8_t* dst = f->fp8 + (size_t)t * 16 * g_ple_rb;
              const uint64_t* ap = f->addrs.data() + (size_t)t * 16;
              for (int h = 0; h < 16; h++)
                std::memcpy(dst + h * g_ple_rb, (const void*)(uintptr_t)ap[h],
                            g_ple_rb);
            }
            if (f->workers.fetch_sub(1) == 1)
              f->done = std::chrono::steady_clock::now();
          });
      }
    } else {
      for (int w = 0; w < nt; w++)
        fl.pool.emplace_back([this, f, w] {
          for (int t = w; t < f->P; t += nt)
            ple_rows_fp8_win(*f, t, f->fp8 + (size_t)t * 16 * g_ple_rb);
          if (f->workers.fetch_sub(1) == 1)
            f->done = std::chrono::steady_clock::now();
        });
    }
    fl.valid = true;
  }
  void ple_join() {
    for (auto& th : ple_fl.pool) th.join();
    ple_fl.pool.clear();
  }

  // QWENOX_VGDUMP (debug): sync + D2H an arbitrary device buffer, FNV-1a hash.
  void vg_dump_buf(const char* tag, int P, const void* p, size_t bytes) {
    if (vg_capturing_ || P < 2 || P > 16 || !getenv("QWENOX_VGDUMP")) return;
    CK(hipStreamSynchronize(g_str));
    std::vector<uint8_t> h(bytes);
    CK(hipMemcpy(h.data(), p, bytes, hipMemcpyDeviceToHost));
    uint64_t fnv = 1469598103934665603ULL;
    for (uint8_t b : h) { fnv ^= b; fnv *= 1099511628211ULL; }
    fprintf(stderr, "[%s] P=%d bufhash=%016llx\n", tag, P,
            (unsigned long long)fnv);
  }

  // QWENOX_VGDUMP (debug): sync + D2H d_Rb rows [0,P), print an FNV-1a hash of
  // the raw row bytes (bf16 or f32 depending on gr_bf16) — sensitive
  // round-state comparator for verify-graph bisection.
  void vg_dump_row(const char* tag, int P, int base) {
    if (vg_capturing_ || P > 16 || !getenv("QWENOX_VGDUMP")) return;
    CK(hipStreamSynchronize(g_str));
    const size_t row = (size_t)10240 * (gr_bf16 ? 2 : 4);
    std::vector<uint8_t> h((size_t)P * row);
    CK(hipMemcpy(h.data(), (const uint8_t*)d_Rb, (size_t)P * row,
                 hipMemcpyDeviceToHost));
    uint64_t fnv = 1469598103934665603ULL;
    for (uint8_t b : h) { fnv ^= b; fnv *= 1099511628211ULL; }
    fprintf(stderr, "[%s] P=%d base=%d rowhash=%016llx\n", tag, P, base,
            (unsigned long long)fnv);
  }

  // embed all tokens: gather rows then repeat to 4 branches (in-place safe).
  // src is the H2D source: pageable in direct prefill, the pinned staging
  // p_vgtok under verify graph capture (P3-B).
  void enq_embed_body(const int* src, int P) {
    CK(hipMemcpyAsync(d_tokarr, src, P * 4, hipMemcpyHostToDevice, g_str));
    WView et = wview(ckpt.at("embed_tokens.weight"));
    if (gr_bf16)
      embed_rows(et, d_tokarr, P, d_Rb, 0, true);
    else {
      embed_rows(et, d_tokarr, P, d_Rb, 10240);
      k_repeat4_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_Rb, d_Rb, g_cfg.d);
    }
    if (!vg_capturing_ && getenv("QWENOX_VGDUMP")) {
      CK(hipStreamSynchronize(g_str));
      int tk[16] = {};
      int nd = P < 16 ? P : 16;
      CK(hipMemcpy(tk, d_tokarr, (size_t)nd * 4, hipMemcpyDeviceToHost));
      fprintf(stderr, "[vgd] P=%d tok=%d,%d,%d,%d\n", P, tk[0],
              P > 1 ? tk[1] : -1, P > 2 ? tk[2] : -1, P > 3 ? tk[3] : -1);
      vg_dump_row("vge", P, -1);
    }
  }

  // PLE device phase on a gathered staging buffer: dequant -> conv/ring ->
  // add. Under verify graph capture (P3-B) the staging-free event is skipped:
  // the round's closing blocking D2H fences the staging reuse instead.
  void enq_ple_device_b(int P, int base, int buf) {
    uint8_t* stg_dev = buf ? ple_eb_dev2 : ple_eb_dev;
    // dequant staging -> f32 d_eb on GPU (zero-copy read of pinned staging)
    if (ple_iq4) {
      size_t nb = (size_t)P * 2560 / 32;
      k_ple_iq4nl_dequant<<<(unsigned)((nb + 255) / 256), 256, 0, g_str>>>(
          stg_dev, d_eb, nb);
    } else {
      size_t n16 = (size_t)P * 2560 / 16;
      k_ple_fp8_dequant<<<(unsigned)((n16 + 255) / 256), 256, 0, g_str>>>(
          stg_dev, d_eb, ple_tscale, n16);
    }
    // staging is free once the dequant ran; gates the next gather into buf.
    if (!vg_capturing_) {
      CK(hipEventRecord(ple_stg_ev[buf], g_str));
      ple_stg_ev_ok[buf] = true;
    }
    ple_par = buf ^ 1;  // this buffer consumed; the next gather writes the other
    float* hc_storage = d_Rb;
    if (gr_bf16) {
      uint64_t n = (uint64_t)P * 10240;
      k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
          (const uint16_t*)hc_storage, d_Rhatb, n);
      d_Rb = d_Rhatb;
    }
    if (ple_loop)
      for (int t = 0; t < P; t++) ple_gpu_t(t, base);
    else
      ple_gpu_b(P, base);
    if (gr_bf16) {
      uint64_t n = (uint64_t)P * 10240;
      k_f32_to_bf16_v4<<<(unsigned)((n / 4 + 255) / 256), 256, 0, g_str>>>(
          d_Rb, (uint16_t*)hc_storage, 10240, P, 10240);
      d_Rb = hc_storage;
    }
  }

  int prefill_batch(const std::vector<int>& tokens, int base = 0, bool final = true,
                    bool quiet = false) {
    int N = (int)tokens.size();
    int id = -1;
    for (int off = 0; off < N;) {
      int rem = N - off;
      // Absorb a small tail into this chunk instead of running a tiny final
      // chunk whose per-chunk fixed cost (~0.9 s) dwarfs its token count.
      int n = ple_prefetch::chunk_size(rem, maxbatch, maxbatch_cap);
      // D1b: while the GPU is lent mid-chunk, only light rounds may start;
      // a heavy chunk requeues behind the owner and retries.
      while (n > conc_rmax_ && borrow_skip_fn &&
             g_borrow_light.load(std::memory_order_acquire))
        borrow_skip_fn(*this);
      // Next chunk (same sizing rule) feeds the PLE prefetch flight: its
      // gather runs on host threads under this chunk's layer compute.
      std::vector<int> nxt;
      if (off + n < N) {
        int rem2 = N - (off + n);
        int n2 = ple_prefetch::chunk_size(rem2, maxbatch, maxbatch_cap);
        nxt.assign(tokens.begin() + off + n, tokens.begin() + off + n + n2);
      }
      id = prefill_chunk(
          std::vector<int>(tokens.begin() + off, tokens.begin() + off + n),
          base + off, final, quiet, nxt.empty() ? nullptr : &nxt);
      if (mtp_tap_capture) {
        // Ingest this chunk's taps (chunk-local rows in d_mtap) into the MTP
        // layer before the next chunk overwrites them. MTP position i takes
        // token i+1, so the very last prompt row is left to the spec loop.
        int c0 = base + off;
        int count = std::min(n, base + N - 1 - c0);
        if (count > 0) {
          std::vector<int> shifted(tokens.begin() + off + 1,
                                   tokens.begin() + off + 1 + count);
          std::chrono::steady_clock::time_point mt0;
          if (prof) {
            CK(hipStreamSynchronize(g_str));
            mt0 = std::chrono::steady_clock::now();
          }
          mtp_ingest_b(d_mtap, shifted, count, c0);
          if (prof) {
            CK(hipStreamSynchronize(g_str));
            fprintf(stderr, "prof-mtp: P=%d ingest=%.1fms\n", count,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - mt0).count());
          }
        }
      }
      off += n;
      // Concurrent serve: prompt prefill hands the GPU over between chunks
      // (only persistent slot state is live here; spec verify never yields).
      if (prog_total > 0 && off < N) yield_point();
    }
    return id;
  }

  int prefill_chunk(const std::vector<int>& tokens, int base = 0, bool final = true,
                    bool quiet = false, const std::vector<int>* nxt = nullptr) {
    int P = (int)tokens.size();
    if (P < 1 || base + P > maxctx || base < 0 || P > maxbatch_cap) {
      fprintf(stderr, "prefill_chunk: base=%d P=%d out of range "
              "(maxctx=%d maxbatch_cap=%d)\n", base, P, maxctx, maxbatch_cap);
      exit(1);
    }
    moe_dump_base_ = base;
    kv_prepare(base, base + P);  // A2/A3: back (and unshare) rows [base, base+P)
    set_vbase(base);  // P3-A: all base-reading batch kernels this chunk read d_vbase
    cut_begin_chunk(base, P);
    static const bool phase_prof = getenv("QWENOX_PHASE") != nullptr;
    auto phase = [&](const char* tag, std::chrono::steady_clock::time_point& tp) {
      if (!phase_prof) return;
      CK(hipStreamSynchronize(g_str));
      auto now = std::chrono::steady_clock::now();
      fprintf(stderr, "phase %-10s %8.1f ms\n", tag,
              std::chrono::duration<double, std::milli>(now - tp).count());
      tp = now;
    };
    xf16_src = nullptr;  // a new batch invalidates the bf16 conversion cache
#ifdef QWENOX_PHASE_PROF
    {
      static const unsigned long long zeros[16] = {};
      CK(hipMemcpyToSymbol(g_ph, zeros, sizeof(zeros)));
    }
#endif
    auto T0 = std::chrono::steady_clock::now();
    auto TP = T0;
    // PLE rows depend only on token history. This chunk's gather may already
    // be inflight — prefetched at the end of the previous chunk so the disk
    // pump overlaps that chunk's layer compute; otherwise gather now (first
    // chunk, quiet re-ingest, or a flight mismatch). Either way the gather
    // fills staging buffer ple_par and its GPU consumer stays ordered after L0.
    if (ple_on) {
      if (!(ple_fl.valid && ple_fl.base == base && ple_fl.P == P)) {
        if (ple_fl.valid) {  // stale flight for another chunk: finish and drop
          ple_join();
          ple_fl.valid = false;
        }
        const int64_t pre0 = chist.empty() ? pad_id : chist.back();
        const int64_t pre1 = chist.size() < 2 ? pad_id : chist[chist.size() - 2];
        ple_spawn(std::vector<int>(tokens), base, pre0, pre1,
                  (uint8_t*)(ple_par ? ple_eb2 : ple_eb), ple_par);
      }
      for (int t = 0; t < P; t++) chist.push_back(tokens[t]);
    }
    KP("embed_ple");
    enq_embed_body(tokens.data(), P);
    if (dbg && !gr_bf16)
      h_rms("gpu", -1, "emb", d_Rb + (size_t)(P - 1) * 10240, g_cfg.d);
    // Vision (stage 4): masked_scatter the ViT embeds over this chunk's image
    // placeholder rows (HF inputs_embeds semantics; bf16-quantized source).
    if (mrope_active_ && !img_inj.empty()) {
      h_img_inj.clear();
      for (const int2& e : img_inj)
        if (e.y >= base && e.y < base + P)
          h_img_inj.push_back(make_int2(e.x, e.y - base));
      if (!h_img_inj.empty()) {
        CK(hipMemcpyAsync(d_img_inj, h_img_inj.data(),
                          h_img_inj.size() * sizeof(int2),
                          hipMemcpyHostToDevice, g_str));
        if (gr_bf16)
          k_img_inject_bf16<<<(unsigned)h_img_inj.size(), 256, 0, g_str>>>(
              d_img_emb, d_img_inj, (uint16_t*)d_Rb);
        else
          k_img_inject_f32<<<(unsigned)h_img_inj.size(), 256, 0, g_str>>>(
              d_img_emb, d_img_inj, d_Rb);
      }
    }
    enq_layer_b(0, P, false, true, base);
    vg_dump_row("vg0", P, base);
    if (prog_total > 0) prog_probe(0, P);
    phase("embed+L0", TP);
    if (ple_on) {
      auto wait_start = std::chrono::steady_clock::now();
      ple_join();
      auto wait_done = std::chrono::steady_clock::now();
      if (prof) {
        p_pleh += std::chrono::duration<double, std::milli>(ple_fl.done - ple_fl.start)
                      .count();
        p_ple_wait +=
            std::chrono::duration<double, std::milli>(wait_done - wait_start).count();
      }
      const int buf = ple_fl.buf;
      ple_fl.valid = false;
      enq_ple_device_b(P, base, buf);
      vg_dump_row("vgp", P, base);
    }
    phase("ple", TP);
    if (ws_ok_) CK(hipEventRecord(conc_ev_a_, g_str));  // D1b quantum clock
    for (int l = 1; l < g_cfg.layers; l++) {
      bool fuse_prev = gr_scatter_norm && l > 1;
      bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
      enq_layer_b(l, P, fuse_prev, flush_mlp, base);
      {
        char tag[8];
        snprintf(tag, sizeof tag, "pcL%d", l);
        vg_dump_row(tag, P, base);
      }
      if (prog_total > 0) prog_probe(l, P);
      conc_borrow_yield(base, P, l);  // D1b: no-op unless armed and contended
      if (l == 24) phase("L01-24", TP);
      if (phase_prof && getenv("QWENOX_PHASE")[0] == '2') {
        char lbuf[16];
        snprintf(lbuf, sizeof lbuf, "L%02d", l);
        phase(lbuf, TP);
      }
    }
    phase("L25-47", TP);
    // PLE chunk-ahead prefetch: gather the next chunk's rows on host threads
    // while the GPU drains this chunk's layer queue below (row addresses are
    // pure functions of token ids, all known now; staging is double-buffered).
    // QWENOX_PLE_NOPREFETCH disables for A/B.
    static const bool ple_noprefetch = getenv("QWENOX_PLE_NOPREFETCH") != nullptr;
    if (ple_on && nxt && !nxt->empty() && !ple_fl.valid && !ple_noprefetch) {
      const int64_t pre0 = chist.empty() ? pad_id : chist.back();
      const int64_t pre1 = chist.size() < 2 ? pad_id : chist[chist.size() - 2];
      ple_spawn(std::vector<int>(*nxt), base + P, pre0, pre1,
                (uint8_t*)(ple_par ? ple_eb2 : ple_eb), ple_par);
    }
    KP("tail");
    // MTP tap capture: d_Rb now holds the final 48-layer stream for every row.
    if (mtp_tap_capture) {
      // Taps are stored at chunk-local rows (0..P) regardless of the absolute
      // base; prefill_batch ingests each chunk right after it completes, so
      // prompts longer than maxbatch work.
      if (P > maxbatch_cap) {
        fprintf(stderr, "mtp tap capture: unsupported (P=%d maxbatch_cap=%d)\n", P,
                maxbatch_cap);
        exit(1);
      }
      if (gr_bf16) {  // d_Rb rows are bf16; d_mtap stays fp32 for ingest
        uint64_t n = (uint64_t)P * 10240;
        k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
            (const uint16_t*)d_Rb, d_mtap, n);
      } else {
        CK(hipMemcpyAsync(d_mtap, d_Rb, (size_t)P * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
      }
    }
    if (cut_r_) {  // in-pass cut: d_R(cut) = final stream row cut-1
      if (gr_bf16)
        k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
            (const uint16_t*)d_Rb + (size_t)(cut_r_ - 1) * 10240, d_cR, 10240);
      else
        CK(hipMemcpyAsync(d_cR, d_Rb + (size_t)(cut_r_ - 1) * 10240, 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
      cut_hit = true;
      cut_r_ = 0;
    }
    // hand off last row to the single-token decode state
    if (gr_bf16)
      k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
          (const uint16_t*)d_Rb + (size_t)(P - 1) * 10240, d_R, 10240);
    else
      CK(hipMemcpyAsync(d_R, d_Rb + (size_t)(P - 1) * 10240, 10240 * 4,
                        hipMemcpyDeviceToDevice, g_str));
    pos = base + P;
    ple_ringpos = (base + P) % 9;
    CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
    // Verify consumes batch logits; do not compute and synchronously read
    // the last row's lm_head a second time.
    if (final) enq_final();
    phase("final", TP);
    int id = -1;
    if (final && prog_total > 0) prog_wait(P);
    if (final) CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    phase("argmax_d2h", TP);
    KP("end");
    if (g_kprof_on) g_kprof_ring.flush(base, P, quiet);
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
    // quiet: spec-verify re-ingest chunks (<= gamma+1 tokens each round).
    // Serve-mode prompt chunks use the live progress and request summary logs;
    // keep this legacy line only for offline prefill benchmarks.
    if (!quiet && prog_total <= 0)
      fprintf(stderr, "\n%s prefill: %d tokens in %.2f s = %.1f tok/s\n", log_ts(),
              P, sec, P / sec);
    if (prog_total > 0) prog_base += P;  // live progress: chunk complete
    if (prof)
      fprintf(stderr,
              "prof: gemm_host=%.0fms topk_wait=%.0fms bucket+h2d=%.0fms "
              "ple_host=%.0fms ple_wait=%.0fms\n",
              p_gemm, p_topk_wait, p_bucket, p_pleh, p_ple_wait);
    if (prof && P >= 64) {  // per-chunk view: zero the accumulators (aggr 10-05)
      fprintf(stderr, "prof-chunk: P=%d wall=%.1fms\n", P, sec * 1e3);
      p_gemm = p_topk_wait = p_bucket = p_pleh = p_ple_wait = 0;
    }
#ifdef QWENOX_PHASE_PROF
    if (getenv("QWENOX_PHASE_PROF")) {
      CK(hipStreamSynchronize(g_str));
      unsigned long long h[16];
      CK(hipMemcpyFromSymbol(h, g_ph, sizeof(h)));
      hipDeviceProp_t dp;
      CK(hipGetDeviceProperties(&dp, 0));
      double cyc2ms = 1.0 / (dp.clockRate * 1e3);  // clockRate is kHz
      const char* names[11] = {"in_staging", "in_dots",   "in_solve", "in_vpkcd",
                               "in_attn2",   "st_Sstage", "st_vnew",  "st_out",
                               "st_Supd",    "in_TOTAL",  "st_TOTAL"};
      for (int s = 0; s < 11; ++s)
        fprintf(stderr, "ph %-10s %12llu cyc  %9.1f CU-ms\n", names[s], h[s],
                h[s] * cyc2ms);
    }
#endif
    return id;
  }

  // =================== MTP draft head (Phase A) ==============================
  // Wiring: MTP stream position p takes (trunk h_p, emb(t_{p+1})) and predicts
  // t_{p+2}. Input: h = sum_b fc_hidden(norm_h(tap)[b]), e =
  // fc_embedding(norm_e(embed(tok))), stream = repeat4(h + e). Norm mode is
  // QWENOX_MTP_NORM=full (one 10240 RMSNorm, default) or =branch (4x2560).
  void mtp_input_1(const float* tap) {
    embed_rows(wview(ckpt.at("embed_tokens.weight")), d_mtok, 1, d_emb, 0);
    k_rmsnorm_zc_grouped<<<1, 1024, 0, g_str>>>(d_emb, d_mnorm_e, d_me, 2560,
                                                g_cfg.norm_eps, 1);
    gemv(wview(ckpt.at("mtp.fc_embedding.weight")), d_me, d_meo);
    if (mtp_norm_mode == 0)
      k_rmsnorm_zc_grouped<<<1, 1024, 0, g_str>>>(tap, d_mnorm_h, d_mRhat, 10240,
                                                  g_cfg.norm_eps, 1);
    else
      k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(tap, d_mnorm_h, d_mRhat, 2560,
                                                  g_cfg.norm_eps, 4);
    if (mtp_hagg_mode == 0) {
      // H-agg=sum: x = sum_b fc_hidden(norm_h[b]) + e, then repeat4.
      gemv(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat, d_mh);
      for (int b = 1; b < 4; b++) {
        gemv(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat + (size_t)b * 2560,
             d_mt);
        k_acc<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_mt, 2560);
      }
      k_acc<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_meo, 2560);
      k_repeat4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_mR, g_cfg.d);
    } else {
      // H-agg=branch: stream[b] = fc_hidden(norm_h[b]) + e (branch identity
      // kept). d_mRhat IS a contiguous [4][2560] batch of branch rows, so
      // one 4-row gemm streams fc_hidden once instead of once per branch.
      gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat, d_mR, 4, 2560);
      k_acc_broadcast4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_meo, d_mR, 2560);
    }
  }

  // GR read/write on the MTP stream (mirrors gr_read/gr_write with d_mR).
  void mtp_gr_read(const std::string& prefix, float* x_out) {
    char buf[128];
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_mR, hw, d_mRhat, 2560,
                                                g_cfg.norm_eps, 4);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_mRhat, d_t320);
    k_silu_scale<<<(320 + 255) / 256, 256, 0, g_str>>>(d_t320, 0.25f, 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_t320, d_G);
    k_sigmoid_inplace<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_G, 10240);
    k_gr_combine<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_G, d_mRhat, x_out,
                                                        g_cfg.d, g_cfg.branches);
  }
  void mtp_gr_write(const std::string& prefix, const float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_mRhat, d_w4);
    k_gr_write<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_w4, d_mR, y, g_cfg.d,
                                                      g_cfg.branches);
  }

  template <typename KVT>
  void mtp_qsa_attn(const KVT* kc, const KVT* vc) {
    k_qsa_flash<true, false, KVT, true><<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
        d_qs, kc, vc, d_attn, 1, 256, 2, 24, qsa_dense ? nullptr : d_iselected, 0,
        d_mpos, d_pacc, d_pml, 0, kvpt());
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, d_attn, 12,
                                                       256, d_mpos);
  }

  // One QSA attention step for the MTP layer (mirrors qsa()).
  void mtp_qsa(const float* x, float* y) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    ensure_rope_tab(g_cfg.rope_theta, g_cfg.rotary_dim);
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, 1,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               d_mpos, 0, nullptr, d_rdelta);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_minorm + 128, d_miraw,
                                           d_mik, d_mpos, g_cfg.norm_eps,
                                           g_cfg.rope_theta, d_rdelta, kvpt(),
                                           d_ikh ? d_ikh + (size_t)(g_cfg.layers / 4) *
                                                       ik_layer_blocks * 128
                                                 : nullptr);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(
          d_iq, d_mik, d_iscores, index_blocks, d_mpos, kvpt());
      index_select(d_iscores, index_blocks, d_iselected, 0, 1, d_mpos);
    }
    snprintf(buf, sizeof buf, "%s.q_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_qg);
    snprintf(buf, sizeof buf, "%s.k_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_kb);
    snprintf(buf, sizeof buf, "%s.v_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_vb);
    snprintf(buf, sizeof buf, "%s.q_norm.weight", pfx);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "%s.k_norm.weight", pfx);
    float* knw = ensure_norm256(buf);
    k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(d_qg, qnw, d_qs, d_gs, 256,
                                                 g_cfg.norm_eps);
    k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(d_kb, knw, d_kb, 256,
                                               g_cfg.norm_eps, 1);
    k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(d_qs, 24, 256, 64, d_mpos,
                                                     g_cfg.rope_theta,
                                                     d_rdelta);
    k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(d_kb, 2, 256, 64, d_mpos,
                                                    g_cfg.rope_theta,
                                                    d_rdelta);
    if (qsa_kv_fp32) {
      k_store_kv<<<2, 256, 0, g_str>>>(d_kb, d_vb, d_mkcf, d_mvcf, d_mpos, kvpt());
      mtp_qsa_attn(d_mkcf, d_mvcf);
    } else {
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(d_kb, d_vb, d_mkcb, d_mvcb, d_mpos,
                                            nullptr, kvpt());
      mtp_qsa_attn(d_mkcb, d_mvcb);
    }
    k_sigmoid_gate<<<(24 * 256 + 255) / 256, 256, 0, g_str>>>(d_attn, d_gs,
                                                              d_attn, 24 * 256);
    snprintf(buf, sizeof buf, "%s.o_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), d_attn, y);
  }

  // MTP MoE step (mirrors moe()).
  void mtp_moe(const float* x, float* y) {
    const char* pfx = "mtp.layers.0.mlp";
    char buf[128];
    snprintf(buf, sizeof buf, "%s.gate.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_router);
    int k = g_cfg.topk;
    k_router_topk<<<1, 512, 0, g_str>>>(d_router, d_topids, d_topw, g_cfg.experts,
                                        k);
    if (gg_mtp_on()) {  // GGUF sidecar experts (Q8_0)
      moe_gguf(g_ggx_mtp, x, d_topids, d_topw, d_moeacc, 1);
    } else {
      snprintf(buf, sizeof buf, "%s.experts.gate_up_proj.weight", pfx);
      WView gu = wview(ckpt.at(buf));
      snprintf(buf, sizeof buf, "%s.experts.down_proj.weight", pfx);
      WView dn = wview(ckpt.at(buf));
      if (!moe_down_direct(k, dn)) CK(hipMemsetAsync(d_moeacc, 0, g_cfg.d * 4, g_str));
      const int* topids = d_topids;  // 直读:ids 即专家号,免 D2H 同步
      uint64_t pairs = (uint64_t)k * g_cfg.moe_mid;
      k_q4cp_gemv_gg<0><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
          x, d_guv, topids, (uint64_t)2 * g_cfg.moe_mid, gu.cols, gu.scale_stride,
          k, 0 /*x_stride*/, 16 /*id_stride*/, k);
      k_silu_mul_g<<<(k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(
          d_guv, d_hid, g_cfg.moe_mid, k);
      moe_down1(k, dn, topids);
    }
    snprintf(buf, sizeof buf, "%s.shared_expert.gate_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_guv);
    snprintf(buf, sizeof buf, "%s.shared_expert.up_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_guv + 640);
    k_silu_mul<<<(640 + 255) / 256, 256, 0, g_str>>>(d_guv, d_guv + 640, d_hid,
                                                     640);
    snprintf(buf, sizeof buf, "%s.shared_expert.down_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), d_hid, d_ey);
    snprintf(buf, sizeof buf, "%s.shared_expert_gate.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_sg);
    k_axpy_sg<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_moeacc, d_ey, d_sg,
                                                     g_cfg.d, g_cfg.d);
    CK(hipMemcpyAsync(y, d_moeacc, g_cfg.d * 4, hipMemcpyDeviceToDevice, g_str));
  }

  // One full MTP draft step: appends (tap, tok) at mtp_pos. Greedy callers get
  // the argmax; sampling callers request the full logits instead.
  // d_tok_in: read the input token from this device pointer instead of the
  // host `tok`. d_amx_out: write argmax here on-device and skip the D2H.
  int mtp_step(const float* tap, int tok,
               std::vector<float>* host_logits = nullptr,
               bool argmax_out = true, const int* d_tok_in = nullptr,
               int* d_amx_out = nullptr) {
    // B0: row mtp_pos is in the shared pool; drafts run ahead of the trunk
    // (rows pos-1 .. pos+gamma-2), so map / un-share the page first
    kv_prepare(mtp_pos, mtp_pos + 1);
    if (d_tok_in)
      CK(hipMemcpyAsync(d_mtok, d_tok_in, 4, hipMemcpyDeviceToDevice, g_str));
    else
      CK(hipMemcpyAsync(d_mtok, &tok, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
    mtp_input_1(tap);
    mtp_gr_read("mtp.layers.0.attn_hyper_connection", d_x);
    mtp_qsa(d_x, d_y);
    mtp_gr_write("mtp.layers.0.attn_hyper_connection", d_y);
    mtp_gr_read("mtp.layers.0.mlp_hyper_connection", d_x);
    mtp_moe(d_x, d_y);
    mtp_gr_write("mtp.layers.0.mlp_hyper_connection", d_y);
    mtp_gr_read("mtp.hyper_connection_mixer", d_x);
    WView lm = dlm_on ? w_dlm : wview(ckpt.at("lm_head.weight"));
    const int dvn =
        (draft_vocab > 0 && !host_logits) ? std::min(draft_vocab, g_cfg.vocab)
                                          : 0;
    if (dvn)  // q4cp scales sit behind a codes region sized by lm.rows; nrows
      gemv(lm, d_x, d_logits, 0, (uint64_t)dvn);  // keeps that base intact
    else
      gemv(lm, d_x, d_logits);
    int id = -1;
    if (host_logits) {
      host_logits->resize(g_cfg.vocab);
      CK(hipMemcpy(host_logits->data(), d_logits, (size_t)g_cfg.vocab * 4,
                   hipMemcpyDeviceToHost));
    } else if (argmax_out) {
      int* dst = d_amx_out ? d_amx_out : d_argmax;
      argmax_rows(d_logits, dvn ? dvn : g_cfg.vocab, 1, d_amx_cand, dst,
                  g_str);
      if (!d_amx_out) CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    }
    mtp_pos++;
    return id;
  }

  // ---- batch ingest: build MTP KV over a token range (prompt/accepted) ----
  // Row j takes (tap[j] = trunk h_{base+j}, toks[j] = t_{base+j+1}) at MTP
  // position base+j. No mixer/lm_head: only the layer state (KV) is built.
  void mtp_input_b(const float* taps, int P) {
    embed_rows(wview(ckpt.at("embed_tokens.weight")), d_tokarr, P, d_eb, 2560);
    k_rmsnorm_zc_grouped<<<P, 1024, 0, g_str>>>(d_eb, d_mnorm_e, d_eb, 2560,
                                                g_cfg.norm_eps, 1);
    gemm(wview(ckpt.at("mtp.fc_embedding.weight")), d_eb, d_qnb, P);
    if (mtp_norm_mode == 0)
      k_rmsnorm_zc_grouped<<<P, 1024, 0, g_str>>>(taps, d_mnorm_h, d_mRhatb,
                                                  10240, g_cfg.norm_eps, 1);
    else
      k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(taps, d_mnorm_h, d_mRhatb,
                                                      2560, g_cfg.norm_eps, 4);
    if (mtp_hagg_mode == 0) {
      gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhatb, d_Unb, P,
           10240 /*xstride*/);
      for (int b = 1; b < 4; b++) {
        gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhatb + (size_t)b * 2560,
             d_convoutb, P, 10240 /*xstride*/);
        k_acc<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
            d_Unb, d_convoutb, P * 2560);
      }
      k_acc<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
          d_Unb, d_qnb, P * 2560);
      k_row_head_copy<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0,
                        g_str>>>(d_Unb, d_mRb, 2560);
      k_repeat4_b<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
          d_mRb, d_mRb, g_cfg.d);
    } else {
      // H-agg=branch, branch-batched: d_mRhatb/d_mRb rows are [p*4+b], so
      // fc_hidden runs as one R=4P-row pass (chunked <=8-row direct reads for
      // spec-depth R, a single LT gemm past that) instead of 4 full passes +
      // 4 row copies.
      const WView& hw = wview(ckpt.at("mtp.fc_hidden.weight"));
      const int R = 4 * P;
      bool done = false;
      if (R <= 32) {
        done = true;
        for (int off = 0; off < R && done; off += 8)
          done = gemv_multi_q4cp(hw, d_mRhatb + (size_t)off * 2560,
                                 d_mRb + (size_t)off * 2560,
                                 std::min(8, R - off), 2560);
      }
      if (!done) {
        // d_xbf16b (BP*6144) overflows past P > 0.6*BP, so convert into the
        // 10240-wide d_xbf16 ourselves (mtp_gr_read_b pattern).
        uint64_t tot = (uint64_t)R * 2560;
        k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
            d_mRhatb, d_xbf16, 2560, R, 2560);
        gemm(hw, d_mRhatb, d_mRb, R, 2560, d_xbf16);
        if (xf16_src == d_xb) xf16_src = nullptr;  // d_xbf16 clobbered
      }
      k_acc_broadcast4<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0,
                         g_str>>>(d_qnb, d_mRb, 2560);
    }
  }

  void mtp_gr_read_b(const std::string& prefix, int P) {
    char buf[128];
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_mRb, hw, d_mRhatb, 2560,
                                                    g_cfg.norm_eps, 4);
    // K=10240 gemms would overflow d_xbf16b (maxbatch*6144); convert into the
    // 10240-wide d_xbf16 ourselves and pass it as xbf. (d_xbf16 is clobbered
    // by later d_xb gemms, so gr_write_b reconverts.)
    uint64_t tot = (uint64_t)P * 10240;
    k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
        d_mRhatb, d_xbf16, 10240, P, 10240);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_mRhatb, d_t320b, P, 0, d_xbf16);
    k_silu_scale<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(d_t320b, 0.25f,
                                                           P * 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_t320b, d_Gb, P);
    k_sigmoid_inplace<<<(unsigned)(((size_t)P * 10240 + 255) / 256), 256, 0,
                        g_str>>>(d_Gb, P * 10240);
    k_gr_combine_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_Gb, d_mRhatb, d_xb, g_cfg.d, g_cfg.branches, P);
    if (xf16_src == d_xb) xf16_src = nullptr;
  }
  void mtp_gr_write_b(const std::string& prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    uint64_t tot = (uint64_t)P * 10240;
    k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
        d_mRhatb, d_xbf16, 10240, P, 10240);
    gemm(wview(ckpt.at(buf)), d_mRhatb, d_w4b, P, 0, d_xbf16);
    k_gr_write_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_w4b, d_mRb, d_yb,
                                                            g_cfg.d,
                                                            g_cfg.branches, P);
  }

  // Batch QSA for the MTP layer (mirrors qsa_b flash path).
  // kv_only (aggr 10-05): ingest needs only the persistent state this layer
  // leaves behind — indexer keys + raw ring and the K/V rows. The query path,
  // selection, flash, gate and o_proj outputs are never read by any ingest
  // caller, so they are skipped (identical KV -> identical drafts).
  void mtp_qsa_b(int P, int base, bool kv_only = false) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    // Table-driven rope (M-RoPE requests carry their 3-row positions in the
    // table; text gets the same absolute-position table, bit-identical to
    // the inline fp64 path it replaces).
    ensure_rope_tab(g_cfg.rope_theta, 64);
    ensure_rope_cs(base, P);
    if (!qsa_dense) {
      if (index_f32) {
        iproj_sgemm(d_miw, d_xb, d_iproj, P);
      } else {
        snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
        gemm(wview(ckpt.at(buf)), d_xb, d_iproj, P);
      }
      if (!kv_only)
      k_index_q<<<dim3(P, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, P,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               nullptr, base, d_ropecs, nullptr,
                                               nullptr, d_iqh);
      uint16_t* const mikh =
          d_ikh ? d_ikh + (size_t)(g_cfg.layers / 4) * ik_layer_blocks * 128
                : nullptr;
      int nb = (base + P) / 4, b0 = base / 4;
      if (nb > b0)
        k_index_pool<<<nb - b0, 128, 0, g_str>>>(d_iproj, d_minorm + 128, d_mik,
                                                 g_cfg.norm_eps,
                                                 g_cfg.rope_theta, base, d_miraw,
                                                 d_ropecs, kvpt(), nullptr, mikh);
      if (cut_abs > 0 && d_cmiraw && cut_abs - 1 >= base && cut_abs - 1 <= base + P) {
        // in-pass cut: MTP lockstep at the cut = rows [0, cut-1) ingested
        CK(hipMemcpyAsync(d_cmiraw, d_miraw, 4 * 128 * 4, hipMemcpyDeviceToDevice, g_str));
        const int rr = cut_abs - 1 - base;
        if (rr > 0)
          k_index_ring<<<min(rr, 4), 128, 0, g_str>>>(d_iproj, d_cmiraw, rr, base);
        cut_mtp_hit = true;
      }
      k_index_ring<<<min(P, 4), 128, 0, g_str>>>(d_iproj, d_miraw, P, base);
      // paged: gather the keys contiguous for the scores (as trunk qsa_b)
      const float* skeys = d_mik;
      const uint16_t* skeys16 = mikh;
      if (!kv_only && kv_paged && nb > 0 && std::max(2051, base) < base + P) {
        k_ik_gather<<<nb, 128, 0, g_str>>>(d_mik, d_ptab, d_ikg, nb);
        skeys = d_ikg;
        if (mikh) {
          k_ik_gather16<<<nb, 128, 0, g_str>>>(mikh, d_ptab, d_ikgh, nb);
          skeys16 = d_ikgh;
        }
      }
      const float one = 1.f, zero = 0.f;
      const int a0 = std::max(2051, base);
      const int grid =
          2051 + (a0 - 2051 + INDEX_BATCH - 1) / INDEX_BATCH * INDEX_BATCH;
      for (int first = kv_only ? base + P : a0; first < base + P;) {
        int nxt = first < grid ? min(grid, base + P)
                               : min(first + INDEX_BATCH, base + P);
        int count = nxt - first;
        // Same fused tiled scores + select as the trunk qsa_b path (the MTP
        // sgemm alone was ~0.23 s per 32K prompt, the tiled kernel ~0.03 s).
        // Only the MTP draft path sees these selections.
        // QWENOX_MTP_INDEX_SGEMM=1 keeps the old sgemm + index_select.
        static const bool mtp_sgemm = getenv("QWENOX_MTP_INDEX_SGEMM") != nullptr;
        if (index_fused2 && !mtp_sgemm && count >= 16 &&
            getenv("QWENOX_INDEX_OLDSEL") == nullptr) {
          int* const selrow = d_iselected + (size_t)(first - base) * 512;
          if (index_f16()) {
            uint16_t* const sc16 = (uint16_t*)d_iscores;
            k_index_scores_wmma16<false>
                <<<index_wmma16_grid(nb, count), 256, 0, g_str>>>(
                    d_iqh + (size_t)(first - base) * 512, skeys16, sc16, nb,
                    count, first);
            if ((first + count) / 4 <= 8192)
              k_index_select_rs<true, __half><<<count, 256, 0, g_str>>>(
                  (const __half*)sc16, nb, selrow, first);
            else
              k_index_select_2p<__half><<<count, 256, 0, g_str>>>(
                  (const __half*)sc16, nb, selrow, first);
            first = nxt;
            continue;
          }
          if (index_score64)
            k_index_scores_t64<<<index_t64_grid(nb, count), 256, 0, g_str>>>(
                d_iq + (size_t)(first - base) * 512, skeys, d_iscores, nb, count, first);
          else
            k_index_scores_tiled<<<dim3((nb + 31) / 32, (count + 15) / 16), 256, 0, g_str>>>(
                d_iq + (size_t)(first - base) * 512, skeys, d_iscores, nb, count, first);
          if ((first + count) / 4 <= 8192)
            k_index_select_rs<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          else if (index_sel2p)
            k_index_select_2p<<<count, 256, 0, g_str>>>(d_iscores, nb, selrow, first);
          else if (index_stream_select)
            k_index_select_stream<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          else
            k_index_select<16, true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          first = nxt;
          continue;
        }
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose,
                                    rocblas_operation_none, nb, count * 4, 128,
                                    &one, skeys, 128,
                                    d_iq + (size_t)(first - base) * 512, 128,
                                    &zero, d_iscores, nb);
        if (status != rocblas_status_success) {
          fprintf(stderr, "mtp indexer sgemm failed: %d\n", (int)status);
          exit(1);
        }
        index_select(d_iscores, nb, d_iselected + (size_t)(first - base) * 512,
                     first, count);
        first = nxt;
      }
    }
    if (!kv_only) {
      snprintf(buf, sizeof buf, "%s.q_proj.weight", pfx);
      gemm(wview(ckpt.at(buf)), d_xb, d_qgb, P);
    }
    snprintf(buf, sizeof buf, "%s.k_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_kbb, P);
    snprintf(buf, sizeof buf, "%s.v_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_vbb, P);
    snprintf(buf, sizeof buf, "%s.q_norm.weight", pfx);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "%s.k_norm.weight", pfx);
    float* knw = ensure_norm256(buf);
    if (!kv_only)
      k_qsa_qsplit<true><<<dim3(24, P), 256, 0, g_str>>>(d_qgb, qnw, d_qsb, d_gsb,
                                                         256, g_cfg.norm_eps, base,
                                                         d_ropecs);
    k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(d_kbb, knw, d_kbb,
                                                      nullptr, g_cfg.norm_eps,
                                                      base, d_ropecs);
    int dense_P = qsa_dense ? P : min(P, std::max(0, 2051 - base));
    // rows [base, base+P) of pool layer nqsa: one copy / convert per
    // physically contiguous page run (a single one unpaged)
    if (qsa_kv_fp32) {
      kv_rows_store(d_mkcf, d_kbb, base, P);
      kv_rows_store(d_mvcf, d_vbb, base, P);
      if (!kv_only) qsa_flash_b(d_mkcf, d_mvcf, P, dense_P, base, nullptr, kvpt());
    } else {
      kv_runs(base, base + P, KV_PAGE, [&](int lr, size_t pr, int n) {
        const size_t so = (size_t)(lr - base) * 512;
        const unsigned g = (unsigned)(((size_t)n * 128 + 255) / 256);
        k_f32_to_bf16_v4<<<g, 256, 0, g_str>>>(d_kbb + so, d_mkcb + pr * 512, 512, n,
                                               512);
        k_f32_to_bf16_v4<<<g, 256, 0, g_str>>>(d_vbb + so, d_mvcb + pr * 512, 512, n,
                                               512);
      });
      if (!kv_only) qsa_flash_b(d_mkcb, d_mvcb, P, dense_P, base, nullptr, kvpt());
    }
    if (kv_only) return;
    k_sigmoid_gate<<<(P * 6144 + 255) / 256, 256, 0, g_str>>>(d_attnb, d_gsb,
                                                              d_attnb, P * 6144);
    snprintf(buf, sizeof buf, "%s.o_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_attnb, d_yb, P);
  }

  // Batch MoE for the MTP layer: default deterministic packed path only
  // (mtp_init refuses non-default MoE flags).
  void mtp_moe_b(int P) {
    const char* pfx = "mtp.layers.0.mlp";
    char buf[128];
    snprintf(buf, sizeof buf, "%s.gate.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_routerb, P);
    int k = g_cfg.topk, E = g_cfg.experts, npairs = P * k;
    router_topk_b(P, k);
    if (gg_mtp_on()) {  // GGUF sidecar experts; overwrites d_moeaccb
      moe_gguf(g_ggx_mtp, d_xb, d_topidsb, d_topwb, d_moeaccb, P);
      mtp_moe_b_shared(P);
      return;
    }
    snprintf(buf, sizeof buf, "%s.experts.gate_up_proj.weight", pfx);
    const Tensor& gu_t = ckpt.at(buf);
    WView gu = wview(gu_t);
    snprintf(buf, sizeof buf, "%s.experts.down_proj.weight", pfx);
    const Tensor& dn_t = ckpt.at(buf);
    WView dn = wview(dn_t);
    if (P > moe_naive_max && moe_q4w_run(gu, dn, d_xb, d_topidsb, d_topwb, d_moeaccb, P)) {
      mtp_moe_b_shared(P);
      return;
    }
    CK(hipMemsetAsync(d_moeaccb, 0, (size_t)P * g_cfg.d * 4, g_str));
    moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff, d_pairids,
                    P, k, E);
    int tasks = (npairs + 63) / 64 + E;
    k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 64);
    k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * g_cfg.d / 4 + 255) / 256), 256, 0,
                       g_str>>>(d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
    k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0,
                               g_str>>>(
        gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2,
        (const float*)gu.data, d_xb, d_tokidx, d_eoff, nullptr,
        2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid, gu.scale_stride, d_moetiles,
        d_moentiles, d_xbf16, (uint16_t*)d_hidb);
    k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0,
                                        g_str>>>(
        dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
        (const float*)dn.data, nullptr, d_tokidx, d_pweight, d_eoff, d_guvb,
        g_cfg.d, dn.cols, dn.scale_stride, d_moetiles, d_moentiles,
        (const uint16_t*)d_hidb);
    k_moe_reduce_fast<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
        d_guvb, d_pairids, d_moeaccb, P, g_cfg.d);
    mtp_moe_b_shared(P);
  }
  // shared expert + gate into d_moeaccb, then d_yb = d_moeaccb
  void mtp_moe_b_shared(int P) {
    const char* pfx = "mtp.layers.0.mlp";
    char buf[128];
    snprintf(buf, sizeof buf, "%s.shared_expert.gate_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_guvb, P);
    snprintf(buf, sizeof buf, "%s.shared_expert.up_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_guvb + (size_t)P * 640, P);
    k_silu_mul<<<(P * 640 + 255) / 256, 256, 0, g_str>>>(
        d_guvb, d_guvb + (size_t)P * 640, d_hidb, P * 640);
    snprintf(buf, sizeof buf, "%s.shared_expert.down_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_hidb, d_eyb, P);
    snprintf(buf, sizeof buf, "%s.shared_expert_gate.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_sgb, P);
    k_axpy_sg<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_moeaccb, d_eyb, d_sgb,
                                                         g_cfg.d, P * g_cfg.d);
    CK(hipMemcpyAsync(d_yb, d_moeaccb, (size_t)P * g_cfg.d * 4,
                      hipMemcpyDeviceToDevice, g_str));
  }

  void mtp_ingest_b(const float* taps, const std::vector<int>& toks, int P,
                    int base) {
    if (P < 1) return;
    kv_prepare(base, base + P);  // B0: MTP rows live in the shared pool
    CK(hipMemcpyAsync(d_tokarr, toks.data(), P * 4, hipMemcpyHostToDevice,
                      g_str));
    mtp_input_b(taps, P);
    xf16_src = nullptr;
    mtp_gr_read_b("mtp.layers.0.attn_hyper_connection", P);
    // aggr 10-05: ingest = KV/indexer build only (see mtp_qsa_b kv_only). The
    // attention output, the MLP-site HC and the MoE of ingested rows were
    // never consumed (~80% of a 16K prompt's 425 ms MTP ingest).
    // QWENOX_MTP_INGEST_FULL=1 runs the full layer as before.
    static const bool full = getenv("QWENOX_MTP_INGEST_FULL") != nullptr;
    static const bool kvo_qsa = !full && !env_zero("QWENOX_MTP_KVO_QSA");
    static const bool kvo_rest = !full && !env_zero("QWENOX_MTP_KVO_REST");
    mtp_qsa_b(P, base, kvo_qsa);
    static const int rest_lvl = [] {  // bisect: how many of the 4 rest steps run
      const char* e = getenv("QWENOX_MTP_REST_LVL");
      return e ? atoi(e) : -1;
    }();
    const int lvl = rest_lvl >= 0 ? rest_lvl : (kvo_rest ? 0 : 4);
    if (lvl >= 1) mtp_gr_write_b("mtp.layers.0.attn_hyper_connection", P);
    if (lvl >= 2) mtp_gr_read_b("mtp.layers.0.mlp_hyper_connection", P);
    if (lvl >= 3) mtp_moe_b(P);
    if (lvl >= 4) mtp_gr_write_b("mtp.layers.0.mlp_hyper_connection", P);
    if (getenv("QWENOX_MTP_HASH")) {  // debug: hash the MTP persistent state rows
      CK(hipStreamSynchronize(g_str));
      auto fnv = [](uint64_t h, const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ULL;
        return h;
      };
      uint64_t hk = 1469598103934665603ULL, hv = hk, hi = hk, hr = hk;
      std::vector<uint8_t> tmp;
      kv_runs(base, base + P, KV_PAGE, [&](int, size_t pr, int n) {
        const size_t bytes = (size_t)n * 512 * (qsa_kv_fp32 ? 4 : 2);
        tmp.resize(bytes);
        CK(hipMemcpy(tmp.data(), qsa_kv_fp32 ? (void*)(d_mkcf + pr * 512) : (void*)(d_mkcb + pr * 512),
                     bytes, hipMemcpyDeviceToHost));
        hk = fnv(hk, tmp.data(), bytes);
        CK(hipMemcpy(tmp.data(), qsa_kv_fp32 ? (void*)(d_mvcf + pr * 512) : (void*)(d_mvcb + pr * 512),
                     bytes, hipMemcpyDeviceToHost));
        hv = fnv(hv, tmp.data(), bytes);
      });
      const int nb = (base + P) / 4, b0 = base / 4;
      if (!qsa_dense && nb > b0)
        kv_runs(b0, nb, KV_PAGE_BLOCKS, [&](int, size_t pb, int n) {
          tmp.resize((size_t)n * 128 * 4);
          CK(hipMemcpy(tmp.data(), d_mik + pb * 128, tmp.size(), hipMemcpyDeviceToHost));
          hi = fnv(hi, tmp.data(), tmp.size());
        });
      if (d_miraw) {
        tmp.resize(512 * 4);
        CK(hipMemcpy(tmp.data(), d_miraw, tmp.size(), hipMemcpyDeviceToHost));
        hr = fnv(hr, tmp.data(), tmp.size());
      }
      fprintf(stderr, "[mtp-hash] base=%d P=%d K=%016llx V=%016llx ik=%016llx ring=%016llx\n",
              base, P, (unsigned long long)hk, (unsigned long long)hv,
              (unsigned long long)hi, (unsigned long long)hr);
    }
    // The MTP layer's d_xb rows live in the trunk's shared batch buffers: never
    // leave a bf16-conversion cache entry for them behind (with the KV-only
    // ingest a later same-shape gemm(d_xb) reused the MTP input — bisected
    // 10-05: output diverged from plain greedy until this reset).
    xf16_src = nullptr;
    mtp_pos = base + P;
    CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
  }

  // Acceptance-rate harness: after prefill_batch(prompt) with tap capture,
  // ingest the prompt into the MTP layer, then walk K trunk steps; at each,
  // the draft (from the frozen pre-step d_R) must match the trunk's next
  // argmax. Prints top-1 acceptance.
  void mtp_accept_test(const std::vector<int>& prompt, int first_id, int K) {
    int P = (int)prompt.size();
    if (P > maxbatch) {
      fprintf(stderr, "mtp-test: prompt %d > maxbatch %d\n", P, maxbatch);
      exit(1);
    }
    std::vector<int> shifted(prompt.begin() + 1, prompt.end());
    mtp_ingest_b(d_mtap, shifted, P - 1, 0);
    int cur = first_id, hits = 0, n = 0;
    auto T0 = std::chrono::steady_clock::now();
    for (int g = 0; g < K; g++) {
      int draft = mtp_step(d_R, cur);  // predicts the token after next
      int nxt = forward(cur);          // trunk commits cur
      hits += (draft == nxt);
      n++;
      cur = nxt;
      if (g % 32 == 31 || g == K - 1) {
        double sec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - T0)
                         .count();
        fprintf(stderr, "\rmtp-test %d/%d acc=%.4f (%.1f steps/s)", n, K,
                (double)hits / n, n / sec);
        fflush(stderr);
      }
    }
    fprintf(stderr, "\nmtp-test: top-1 acceptance %d/%d = %.4f (norm=%s hagg=%s)\n",
            hits, n, (double)hits / n, mtp_norm_mode ? "branch" : "full",
            mtp_hagg_mode ? "branch" : "sum");
  }

  // =================== MTP speculative loop (Phase B) ========================
  // gr_bf16 shims: d_Rb rows are bf16 under QWENOX_GR_BF16, but the MTP ingest
  // path and the decode state d_R are fp32. spec_taps stages rows 0..P-1 as
  // fp32 in d_Rhatb (scratch outside prefill; consumed by mtp_input_b before
  // anything else touches it).
  const float* spec_taps(int P) {
    if (!gr_bf16) return d_Rb;
    uint64_t n = (uint64_t)P * 10240;
    k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
        (const uint16_t*)d_Rb, d_Rhatb, n);
    return d_Rhatb;
  }
  void restore_dR_row(int d) {
    if (gr_bf16)
      k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
          (const uint16_t*)d_Rb + (size_t)d * 10240, d_R, 10240);
    else
      CK(hipMemcpyAsync(d_R, d_Rb + (size_t)d * 10240, 10240 * 4,
                        hipMemcpyDeviceToDevice, g_str));
  }
  // Spec checkpoints: only the position-mod rings (PLE ring + QSA indexer raw
  // ring) are needed — GDN S / convst are restored from per-row snapshots.
  void mtp_ckpt_save() {
    float* p = d_ckpt + (ckpt_S + ckpt_convst) / 4;
    CK(hipMemcpyAsync(p, d_ring, ckpt_ring, hipMemcpyDeviceToDevice, g_str));
    p += ckpt_ring / 4;
    if (ckpt_iraw)
      CK(hipMemcpyAsync(p, d_iraw, ckpt_iraw, hipMemcpyDeviceToDevice, g_str));
  }

  // Produce logits for every batch row's final stream. Rows remain live in
  // d_Rb after prefill_batch, and the multi-row lm_head streams weights once.
  void batch_logits_device(int P) {
    gr_read_b("hyper_connection_mixer", P);
    const WView& lw = wview(ckpt.at("lm_head.weight"));
    // The multi-row kernel is specialized for up to eight rows. Chunk larger
    // ngram proposals into groups of eight so lm_head weights are streamed
    // once per group instead of once per row.
    for (int off = 0; off < P; ) {
      const int q = std::min(8, P - off);
      const bool ok = gemv_multi_q4cp(
          lw, d_xb + (size_t)off * 2560,
          d_logits_b + (size_t)off * g_cfg.vocab, q, 2560);
      if (!ok) {
        for (int r = 0; r < q; r++) {
          gemv(lw, d_xb + (size_t)(off + r) * 2560,
               d_logits_b + (size_t)(off + r) * g_cfg.vocab);
        }
      }
      off += q;
    }
  }

  void batch_logits(int P, std::vector<float>& host) {
    batch_logits_device(P);
    host.resize((size_t)P * g_cfg.vocab);
    CK(hipMemcpy(host.data(), d_logits_b, host.size() * sizeof(float),
                 hipMemcpyDeviceToHost));
  }

  void batch_nll(int P, int first, int scored, const int* targets,
                 std::vector<float>& losses) {
    losses.resize(scored);
    if (!scored) return;
    gr_read_b("hyper_connection_mixer", P);
    const WView& lw = wview(ckpt.at("lm_head.weight"));
    float* d_losses = d_logits_b + (size_t)64 * g_cfg.vocab;
    for (int off = first; off < first + scored; off += 64) {
      int rows = std::min(64, first + scored - off);
      for (int group = 0; group < rows; group += 8) {
        int count = std::min(8, rows - group);
        if (!gemv_multi_q4cp(lw, d_xb + (size_t)(off + group) * 2560,
                              d_logits_b + (size_t)group * g_cfg.vocab,
                              count, 2560)) {
          for (int row = 0; row < count; row++)
            gemv(lw, d_xb + (size_t)(off + group + row) * 2560,
                 d_logits_b + (size_t)(group + row) * g_cfg.vocab);
        }
      }
      CK(hipMemcpyAsync(d_argmax_arr, targets + off - first, rows * sizeof(int),
                        hipMemcpyHostToDevice, g_str));
      k_batch_nll<<<rows, 256, 0, g_str>>>(d_logits_b, g_cfg.vocab,
                                           d_argmax_arr, d_losses);
      CK(hipMemcpy(losses.data() + off - first, d_losses, rows * sizeof(float),
                   hipMemcpyDeviceToHost));
    }
  }

  // Full logits of rows [first, first+n) of the last prefill chunk (P rows),
  // row-major [n][vocab] on the host; same 64-row lm_head passes as batch_nll.
  void batch_logits_rows(int P, int first, int n, std::vector<float>& host) {
    host.resize((size_t)n * g_cfg.vocab);
    if (!n) return;
    gr_read_b("hyper_connection_mixer", P);
    const WView& lw = wview(ckpt.at("lm_head.weight"));
    for (int off = first; off < first + n; off += 64) {
      int rows = std::min(64, first + n - off);
      for (int group = 0; group < rows; group += 8) {
        int count = std::min(8, rows - group);
        if (!gemv_multi_q4cp(lw, d_xb + (size_t)(off + group) * 2560,
                              d_logits_b + (size_t)group * g_cfg.vocab,
                              count, 2560)) {
          for (int row = 0; row < count; row++)
            gemv(lw, d_xb + (size_t)(off + group + row) * 2560,
                 d_logits_b + (size_t)(group + row) * g_cfg.vocab);
        }
      }
      CK(hipMemcpy(host.data() + (size_t)(off - first) * g_cfg.vocab, d_logits_b,
                   (size_t)rows * g_cfg.vocab * sizeof(float), hipMemcpyDeviceToHost));
    }
  }

  // Greedy reduction of the same batched logits.
  void batch_argmax(int P, int* host) {
    batch_logits_device(P);
    argmax_rows(d_logits_b, g_cfg.vocab, P, d_amx_cand, d_argmax_arr, g_str);
    CK(hipMemcpy(host, d_argmax_arr, P * 4, hipMemcpyDeviceToHost));
  }

  // ---- GPU sampling prep (spec sampling fast path) ----
  // Corrections (bias/penalties) and the top-k pre-filter of
  // HostSampler::prepare run on device; only k (id, logit) pairs per row come
  // back and the host replays the remaining filter chain on them. The
  // distribution is unchanged; candidate order is (value desc, id asc)
  // instead of nth_element's arbitrary order, so same-seed draws can differ
  // from the dense path (allowed: nth_element never promised an order).
  template <typename Sampler>
  bool sms_gpu_active(const Sampler& s) const {
    static const bool off =
        getenv("QWENOX_SMS_GPU") && atoi(getenv("QWENOX_SMS_GPU")) == 0;
    if (off) return false;
    if (s.top_k <= 0 || s.top_k > 64 || s.top_k >= g_cfg.vocab) return false;
    if ((int)s.bias.size() > SMS_BIAS_CAP) return false;
    if (s.bias.size() > 1) {
      // The GPU bias phase applies entries in parallel; duplicate ids would
      // race instead of adding sequentially, so those requests stay dense.
      std::vector<int> ids;
      ids.reserve(s.bias.size());
      for (auto& b : s.bias) ids.push_back(b.first);
      std::sort(ids.begin(), ids.end());
      for (size_t i = 1; i < ids.size(); i++)
        if (ids[i] == ids[i - 1]) return false;
    }
    return true;
  }

  // Upload the round-shared correction lists (bias, then hist penalties).
  // False when the hist overflows the buffer: caller uses the dense path.
  template <typename Sampler>
  bool sms_round_shared(const Sampler& s) {
    if ((int)sms_out_ids.size() < SMS_ROWS * s.top_k) {
      sms_out_ids.resize((size_t)SMS_ROWS * s.top_k);
      sms_out_vals.resize((size_t)SMS_ROWS * s.top_k);
    }
    sm_nb = (int)s.bias.size();
    sm_nh = 0;
    const bool pen = (s.presence != 0.0f || s.frequency != 0.0f);
    if (pen) {
      if ((int)s.hist.size() > SMS_HIST_CAP) return false;
      sm_nh = (int)s.hist.size();
    }
    if (!sm_nb && !sm_nh) return true;
    h_sm_ci.clear();
    h_sm_cv.clear();
    for (auto& b : s.bias) {
      h_sm_ci.push_back(b.first);
      h_sm_cv.push_back(b.second);
    }
    if (sm_nb) {
      CK(hipMemcpyAsync(d_sm_bids, h_sm_ci.data(), sm_nb * 4,
                        hipMemcpyHostToDevice, g_str));
      CK(hipMemcpyAsync(d_sm_bvals, h_sm_cv.data(), sm_nb * 4,
                        hipMemcpyHostToDevice, g_str));
    }
    if (sm_nh) {
      for (auto& h : s.hist) {
        h_sm_ci.push_back(h.first);
        h_sm_cv.push_back(s.presence + s.frequency * (float)h.second);
      }
      CK(hipMemcpyAsync(d_sm_hids, h_sm_ci.data() + sm_nb, sm_nh * 4,
                        hipMemcpyHostToDevice, g_str));
      CK(hipMemcpyAsync(d_sm_hvals, h_sm_cv.data() + sm_nb, sm_nh * 4,
                        hipMemcpyHostToDevice, g_str));
    }
    return true;
  }

  // Per-row draft-prefix penalties. Walk row a (flat draft index base_off+a)
  // is only prepared when drafts[0..base_off+a) were all drawn before it, so
  // its penalty hist counts include exactly those. An id already in the base
  // hist gains only frequency * extra (presence came with the shared list);
  // a new id costs presence + frequency * count. The shared+prefix split
  // rounds differently from prepare()'s single merged subtraction by ~1 ulp;
  // the sampling path is distribution-gated, not bit-gated.
  // Returns whether any prefix list is nonempty.
  template <typename Sampler>
  bool sms_prefix_upload(const Sampler& s, const int* drafts, int base_off,
                         int rows) {
    const bool pen = (s.presence != 0.0f || s.frequency != 0.0f);
    if (!pen) return false;
    std::vector<int> pn(rows, 0);
    h_sm_pi.assign((size_t)rows * SMS_PCAP, 0);
    h_sm_pv.assign((size_t)rows * SMS_PCAP, 0.0f);
    bool any = false;
    for (int a = 0; a < rows; a++) {
      const int nd = std::min(base_off + a, SMS_PCAP);
      int used = 0;
      int* rids = h_sm_pi.data() + (size_t)a * SMS_PCAP;
      float* rvals = h_sm_pv.data() + (size_t)a * SMS_PCAP;
      for (int j = 0; j < nd; j++) {
        const int id = drafts[j];
        int slot = -1;
        for (int u = 0; u < used; u++)
          if (rids[u] == id) { slot = u; break; }
        if (slot < 0) {
          slot = used++;
          rids[slot] = id;
          rvals[slot] = s.hist.count(id) ? 0.0f : s.presence;
        }
        rvals[slot] += s.frequency;
      }
      pn[a] = used;
      if (used) any = true;
    }
    if (!any) return false;
    CK(hipMemcpyAsync(d_sm_pids, h_sm_pi.data(), (size_t)rows * SMS_PCAP * 4,
                      hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_sm_pvals, h_sm_pv.data(), (size_t)rows * SMS_PCAP * 4,
                      hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_sm_pn, pn.data(), rows * 4, hipMemcpyHostToDevice,
                      g_str));
    return true;
  }

  // Corrections + two-pass radix top-k + collect for `rows` logit rows
  // (stride floats each), then D2H and an exact top-k truncation ordered by
  // (value desc, id asc). On success out_ids/out_vals hold rows*k pairs.
  // Returns false on candidate overflow: the caller re-runs the dense path
  // (logits may already carry corrections, so the dense path must regenerate
  // or read a pristine copy -- hence the scratch bounce when corrections
  // exist).
  bool sms_topk_rows(float* x, int stride, int rows, int k, bool use_pfx,
                     int* out_ids, float* out_vals) {
    float* src = x;
    if (sm_nb > 0 || sm_nh > 0 || use_pfx) {
      if (!d_sm_scratch)
        DALLOC(&d_sm_scratch, (size_t)SMS_ROWS * g_cfg.vocab * 4);
      CK(hipMemcpyAsync(d_sm_scratch, x, (size_t)rows * stride * 4,
                        hipMemcpyDeviceToDevice, g_str));
      k_corr_apply<<<dim3(1, rows), 256, 0, g_str>>>(
          d_sm_scratch, stride, d_sm_bids, d_sm_bvals, sm_nb, d_sm_hids,
          d_sm_hvals, sm_nh, d_sm_pids, d_sm_pvals,
          use_pfx ? d_sm_pn : nullptr, SMS_PCAP);
      src = d_sm_scratch;
    }
    k_tk_init<<<rows, 32, 0, g_str>>>(d_sm_ctrl, k);
    CK(hipMemsetAsync(d_sm_hist, 0, (size_t)rows * 4096 * sizeof(unsigned),
                      g_str));
    k_tk_hist<0><<<dim3(8, rows), 256, 0, g_str>>>(src, stride, stride,
                                                   d_sm_ctrl, d_sm_hist);
    k_tk_scan<0><<<dim3(1, rows), 256, 0, g_str>>>(d_sm_hist, d_sm_ctrl,
                                                   d_sm_cand_n);
    CK(hipMemsetAsync(d_sm_hist, 0, (size_t)rows * 4096 * sizeof(unsigned),
                      g_str));
    k_tk_hist<1><<<dim3(8, rows), 256, 0, g_str>>>(src, stride, stride,
                                                   d_sm_ctrl, d_sm_hist);
    k_tk_scan<1><<<dim3(1, rows), 256, 0, g_str>>>(d_sm_hist, d_sm_ctrl,
                                                   d_sm_cand_n);
    k_tk_collect<<<dim3(8, rows), 256, 0, g_str>>>(
        src, stride, stride, d_sm_ctrl, d_sm_cand_ids, d_sm_cand_vals,
        d_sm_cand_n, SMS_TCAP);
    std::vector<int> cn(rows);
    CK(hipMemcpy(cn.data(), d_sm_cand_n, rows * 4, hipMemcpyDeviceToHost));
    for (int r = 0; r < rows; r++)
      if (cn[r] < k || cn[r] > SMS_TCAP) return false;
    CK(hipMemcpy(h_sm_ids.data(), d_sm_cand_ids, (size_t)rows * SMS_TCAP * 4,
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(h_sm_vals.data(), d_sm_cand_vals,
                 (size_t)rows * SMS_TCAP * 4, hipMemcpyDeviceToHost));
    std::vector<int> ord;
    for (int r = 0; r < rows; r++) {
      const int m = cn[r];
      const int* ids = h_sm_ids.data() + (size_t)r * SMS_TCAP;
      const float* vals = h_sm_vals.data() + (size_t)r * SMS_TCAP;
      ord.resize(m);
      for (int j = 0; j < m; j++) ord[j] = j;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (vals[a] != vals[b]) return vals[a] > vals[b];
        return ids[a] < ids[b];
      });
      for (int j = 0; j < k; j++) {
        out_ids[(size_t)r * k + j] = ids[ord[j]];
        out_vals[(size_t)r * k + j] = vals[ord[j]];
      }
    }
    return true;
  }

  // prepare_sparse on candidate row `row` (draft: row 0 of the d_logits
  // pass). QWENOX_SMS_CHECK=1 also replays the dense prepare() on the raw
  // logits row and compares the two distributions (support + probabilities).
  template <typename Sampler>
  void sms_prep(Sampler& s, int row, bool draft,
                typename Sampler::SDist& out) {
    const int k = s.top_k;
    s.prepare_sparse(sms_out_ids.data() + (size_t)row * k,
                     sms_out_vals.data() + (size_t)row * k, k, out);
    static const bool chk = getenv("QWENOX_SMS_CHECK") != nullptr;
    if (!chk) return;
    static long n = 0, bad = 0;
    static double maxd = 0, maxout = 0;
    std::vector<float> v(g_cfg.vocab);
    const float* src = draft ? d_logits : d_logits_b + (size_t)row * g_cfg.vocab;
    CK(hipMemcpy(v.data(), src, (size_t)g_cfg.vocab * 4, hipMemcpyDeviceToHost));
    Sampler c = s;
    c.prepare(v);
    double d = 0, in = 0;
    for (size_t j = 0; j < out.ids.size(); j++) {
      d = std::max(d, (double)fabsf(v[out.ids[j]] - out.pr[j]));
      in += v[out.ids[j]];
    }
    int dn = 0;
    for (int i = 0; i < g_cfg.vocab; i++) dn += v[i] > 0.0f;
    const double o = fabs(1.0 - in);
    n++;
    maxd = std::max(maxd, d);
    maxout = std::max(maxout, o);
    const bool b = d > 1e-5 || o > 1e-5 || dn != (int)out.ids.size();
    if (b) {
      bad++;
      fprintf(stderr, "[sms-check] MISMATCH %s row %d: maxdiff %.3g outside %.3g support dense %d sparse %zu\n",
              draft ? "draft" : "verify", row, d, o, dn, out.ids.size());
    }
    if (b || n % 32 == 0)
      fprintf(stderr, "[sms-check] n=%ld bad=%ld maxdiff=%.3g maxoutside=%.3g\n", n, bad, maxd, maxout);
  }

  std::vector<int> spec_loop_ngram(int cur, int gen, int gamma,
      const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, gamma), maxbatch - 1});
    int accepted = 0, fallback = 0;
    double verify_ms = 0, replay_ms = 0;
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto step = [&](int tok) {
      int nx = forward(tok);
      if (!ple_on) chist.push_back(tok);
      ple_ringpos = pos % 9;  // graph advances the device scalar, not the host mirror
      return nx;
    };
    auto verify = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    while ((int)out.size() < gen && pos < maxctx) {
      yield_point();  // round top: only persistent slot state is live
      if (chist.size() != (size_t)pos)
        throw std::runtime_error("ngram: token history and trunk position disagree");
      ngram.observe(chist);
      // Leave one slot for the target-model continuation. Never verify past
      // the generation budget or context boundary, including the final round.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      std::vector<int> draft = want > 0 ? ngram.draft(chist, cur, want) : std::vector<int>{};
      if (draft.empty()) {
        int next = step(cur);
        ++fallback;
        if (emit && !emit(next)) break;
        out.push_back(next);
        cur = next;
        continue;
      }
      const int base = pos;
      const size_t history_size = chist.size();
      ngram_ckpt_save();
      std::vector<int> tokens{cur};
      tokens.insert(tokens.end(), draft.begin(), draft.end());
      auto t = std::chrono::steady_clock::now();
      verify(tokens, base);
      std::vector<int> targets(tokens.size());
      batch_argmax((int)tokens.size(), targets.data());
      verify_ms += elapsed(t);
      int a = 0;
      while (a < (int)draft.size() && draft[a] == targets[a]) ++a;
      int keep = a;
      bool stop = false;
      for (int j = 0; j < a; ++j) {
        if (emit && !emit(draft[j])) { keep = j; stop = true; break; }
        out.push_back(draft[j]);
      }
      if (!stop) {
        if (emit && !emit(targets[a])) stop = true;
        else out.push_back(targets[a]);
      }
      if (keep != (int)draft.size()) {
        // Restore every overwritten position-mod state, then replay exactly
        // [cur, accepted drafts]. Causal KV/indexer entries are overwritten
        // before they are visible. Replaying a batch preserves verify math.
        t = std::chrono::steady_clock::now();
        ngram_ckpt_restore();
        chist.resize(history_size);
        pos = base;
        ple_ringpos = base % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
        tokens.resize(keep + 1);
        verify(tokens, base);
        CK(hipStreamSynchronize(g_str));
        replay_ms += elapsed(t);
        ++last_spec_rollbacks;
      }
      cur = targets[keep];
      accepted += keep;
      ++last_spec_rounds;
      last_spec_proposed += draft.size();
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = accepted + last_spec_rounds;
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    last_spec_commit = (int)out.size();
    fprintf(stderr, "%s ngram: tokens=%d rounds=%d accepted=%d proposed=%d "
                    "fallback=%d rollbacks=%d verify=%.1fms replay=%.1fms\n",
            log_ts(), last_spec_commit, last_spec_rounds, accepted,
            last_spec_proposed, fallback, last_spec_rollbacks, verify_ms,
            replay_ms);
    return out;
  }

  void ngram_ckpt_save() {
    CK(hipMemcpyAsync(d_ckpt,d_S,ckpt_S,hipMemcpyDeviceToDevice,g_str));
    CK(hipMemcpyAsync(d_ckpt+ckpt_S/4,d_convst,ckpt_convst,hipMemcpyDeviceToDevice,g_str));
    float* p=d_ckpt+(ckpt_S+ckpt_convst)/4; CK(hipMemcpyAsync(p,d_ring,ckpt_ring,hipMemcpyDeviceToDevice,g_str)); p+=ckpt_ring/4;
    if(ckpt_iraw) CK(hipMemcpyAsync(p,d_iraw,ckpt_iraw,hipMemcpyDeviceToDevice,g_str));
  }
  void ngram_ckpt_restore() {
    CK(hipMemcpyAsync(d_S,d_ckpt,ckpt_S,hipMemcpyDeviceToDevice,g_str));
    CK(hipMemcpyAsync(d_convst,d_ckpt+ckpt_S/4,ckpt_convst,hipMemcpyDeviceToDevice,g_str));
    const float* p=d_ckpt+(ckpt_S+ckpt_convst)/4; CK(hipMemcpyAsync(d_ring,p,ckpt_ring,hipMemcpyDeviceToDevice,g_str)); p+=ckpt_ring/4;
    if(ckpt_iraw) CK(hipMemcpyAsync(d_iraw,p,ckpt_iraw,hipMemcpyDeviceToDevice,g_str));
  }

  // Adaptive MTP draft length, active when a spec_loop* is entered with
  // gamma == 0 (API: sampling with QWENOX_SPEC_GAMMA unset; offline: --gamma 0).
  // Tracks an EMA of the per-depth conditional acceptance p_j = P(draft j
  // accepted | drafts 0..j-1 accepted) and after each round picks γ in
  // [kmin, gmax] maximizing E[commit]/cost, cost(γ) = 1 + r·γ.
  // Two states: p is kept separately for rounds that follow a full accept (F)
  // and rounds that follow a reject (R) — 09-29 γ7 traces, sampling real
  // text: F .93/.86/.83 vs R .77/.71/.73. A deep γ makes full accepts rare,
  // so the effective acceptance falls with γ; a single-state estimate misses
  // that and drifts deep. With A_s = Π_{j<γ} p_s,j and E_s the per-state
  // expected commit, the stationary P(F) = A_R / (1 − A_F + A_R) and
  // E(γ) = π_F·E_F + (1 − π_F)·E_R (predicts fixed γ3 commit/round 3.03 vs
  // measured 2.98; single-state 2.82). Round wall ms ≈ 42 + 10.7·γ → r ≈ 0.25;
  // default 0.3 absorbs the model's remaining optimism for deep γ. kmin = 3:
  // on-GPU sampling runs stuck at γ2 were ~20% slower than γ3 (the replay
  // can't see that). tools/gamma_sim.py MODEL=markov: real text = fixed γ3,
  // code/repetitive text ≈ best fixed. On GPU (tools/gamma_sample_check.sh,
  // 18 samples): real text 1.004× γ3 (single-state 0.940×, fixed γ4 0.953×),
  // code/repetitive 1.040× γ3 (fixed γ4 1.032×). Depths >= the current γ
  // are unobserved and extrapolated from the deepest probed one. gmax = 7:
  // γ=8 means P=9 verify rows, past the P<=8 fast paths (cliff). Sampling
  // stays exact: γ is fixed before the round draws anything. Greedy ids can
  // differ from a fixed γ only through fp near-tie flips (as any γ change).
  // A gamma > 0 is used as-is (clamped to 8); adapt stays off.
  struct GammaCtl {
    static constexpr float kPrior = 0.8f, kPriorN = 4.f;
    bool adapt = false;
    int cur = 4, kmin = 3, gmax = 7, switches = 0, full = 0;
    float r = 0.3f, hyst = 0.02f, alpha = 0.05f;
    float p[2][8], n[2][8];  // [0] after a reject, [1] after a full accept
    int hist[9] = {0};
    static float env_f(const char* k, float def) {
      const char* e = getenv(k);
      return e ? (float)atof(e) : def;
    }
    GammaCtl(int gamma, bool sample) {
      for (int s = 0; s < 2; s++)
        for (int j = 0; j < 8; j++) p[s][j] = kPrior, n[s][j] = kPriorN;
      if (gamma > 0) { cur = std::min(8, gamma); return; }
      adapt = true;
      cur = sample ? 3 : 4;  // previous fixed auto defaults
      static const float env_r = env_f("QWENOX_SPEC_ADAPT_R", 0.3f);
      static const float env_alpha = env_f("QWENOX_SPEC_ADAPT_ALPHA", 0.05f);
      static const float env_hyst = env_f("QWENOX_SPEC_ADAPT_HYST", 0.02f);
      static const int env_min = (int)env_f("QWENOX_SPEC_ADAPT_MIN", 3.f);
      static const int env_max = (int)env_f("QWENOX_SPEC_ADAPT_MAX", 7.f);
      r = std::max(0.01f, env_r);
      alpha = std::max(0.005f, std::min(1.f, env_alpha));
      hyst = std::max(0.f, env_hyst);
      kmin = std::max(1, std::min(8, env_min));
      gmax = std::max(kmin, std::min(8, env_max));
      cur = std::max(kmin, std::min(cur, gmax));
    }
    double score(int G) const {
      double e[2], a[2];
      for (int s = 0; s < 2; s++) {
        double pr = 1.0;
        e[s] = 1.0;
        for (int k = 0; k < G; k++) {
          pr *= p[s][std::min(k, cur - 1)];
          e[s] += pr;
        }
        a[s] = pr;
      }
      double pf = a[0] / std::max(1e-9, 1.0 - a[1] + a[0]);
      return (pf * e[1] + (1.0 - pf) * e[0]) / (1.0 + r * G);
    }
    // QWENOX_SPEC_TRACE=1: record every round (γ, accepted, wall ms since the
    // previous update) and print it in report() — offline controller replay
    // (tools/gamma_sim.py). Observation only; fixed and adaptive γ alike.
    static bool trace_on() {
      static const bool on = [] {
        const char* e = getenv("QWENOX_SPEC_TRACE");
        return e && e[0] == '1';
      }();
      return on;
    }
    std::vector<std::array<float, 3>> trace;
    std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();
    // g = drafts proposed this round, a = accepted prefix length.
    void update(int g, int a) {
      if (trace_on()) {
        auto now = std::chrono::steady_clock::now();
        trace.push_back({(float)g, (float)a,
                         (float)std::chrono::duration<double, std::milli>(now - t_last).count()});
        t_last = now;
      }
      hist[std::min(8, g)]++;
      const int s = full;
      full = a >= g;
      if (!adapt || g < cur) return;  // tail round cut short by gen/ctx
      for (int j = 0; j <= a && j < g; j++) {
        n[s][j] += 1.f;
        p[s][j] += std::max(alpha, 1.f / n[s][j]) * ((j < a ? 1.f : 0.f) - p[s][j]);
      }
      int best = cur;
      double sb = score(cur);
      for (int G = kmin; G <= gmax; G++) {
        double s = score(G);
        if (s > sb * (1.0 + hyst) || (best != cur && s > sb)) best = G, sb = s;
      }
      if (best == cur) return;
      for (int t = 0; t < 2; t++)
        for (int j = cur; j < best; j++) p[t][j] = p[t][cur - 1], n[t][j] = kPriorN;
      cur = best;
      switches++;
    }
    void report(const char* tag) const {
      if (trace_on()) {
        fprintf(stderr, "%s gamma-trace:", tag);
        for (auto& t : trace) fprintf(stderr, " %d,%d,%.1f", (int)t[0], (int)t[1], t[2]);
        fprintf(stderr, "\n");
      }
      if (!adapt) return;
      fprintf(stderr, "%s gamma-adapt: rounds per γ", tag);
      for (int G = 1; G <= 8; G++)
        if (hist[G]) fprintf(stderr, " %d:%d", G, hist[G]);
      fprintf(stderr, " | switches=%d final γ=%d r=%.2f p(after full):", switches, cur, r);
      for (int j = 0; j < cur; j++) fprintf(stderr, " %.2f", p[1][j]);
      fprintf(stderr, " p(after reject):");
      for (int j = 0; j < cur; j++) fprintf(stderr, " %.2f", p[0][j]);
      fprintf(stderr, "\n");
    }
  };

  // Speculative decode driver. cur = emitted token for position pos (trunk
  // state processed through pos-1, MTP KV covers 0..pos-2). Greedy accept:
  // draft j committed iff it equals the trunk argmax at its verify row; the
  // correction (or full-accept continuation) comes from the trunk.
  // emit (optional): called with each committed token as it lands; returning
  // false aborts the loop early (serve: eos/cancel/streaming).
  std::vector<int> spec_loop(int cur, int gen, int gamma,
                             const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(gen + 1);
    GammaCtl gctl(gamma, false);
    std::vector<int> drafts(8), v(9);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int rounds = 0, rollbacks = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, t_ingest = 0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen) {
      yield_point();  // round top: mtp_pos == pos-1, d_R == h_{pos-1}
      int q = pos - 1;  // trunk processed through q
      if (d_iraw) vg_dump_buf("irT", 2, d_iraw, 512 * 4);  // P3-B bisect
      int g = std::min(gctl.cur, gen - (int)out.size());
      size_t chist_sz = chist.size();
      mtp_ckpt_save();
      if (d_iraw) vg_dump_buf("irC", 2, d_iraw, 512 * 4);  // P3-B bisect
      auto t0 = std::chrono::steady_clock::now();
      // draft 1 used the committed (h_q, cur): its indexer ring rows are
      // truth. Later drafts pollute the ring; save/restore around them.
      if (draft_desync) {
        mtp_step(d_R, cur, nullptr, true, nullptr, d_argmax_arr);
        CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        for (int j = 1; j < g; j++)
          mtp_step(d_mR, 0, nullptr, true, d_argmax_arr + j - 1,
                   d_argmax_arr + j);
        CK(hipMemcpy(drafts.data(), d_argmax_arr, (size_t)g * sizeof(int),
                     hipMemcpyDeviceToHost));
      } else {
        drafts[0] = mtp_step(d_R, cur);
        CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        for (int j = 1; j < g; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
      }
      auto t1 = std::chrono::steady_clock::now();
      if (d_iraw) vg_dump_buf("irD", 2, d_iraw, 512 * 4);  // P3-B bisect
      std::vector<int> vt;
      vt.reserve(g + 1);
      vt.push_back(cur);
      for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
      spec_snap = true;  // capture per-row rollback snapshots during verify
      if (!verify_graph_round_try(vt, q + 1, v.data())) {
        prefill_batch(vt, q + 1, false, true);
        spec_snap = false;
        batch_argmax(g + 1, v.data());
      } else
        spec_snap = false;
      if (getenv("QWENOX_VGDUMP")) {  // P3-B bisect: whole-logits round hash
        CK(hipStreamSynchronize(g_str));
        const size_t nb = (size_t)(g + 1) * g_cfg.vocab * 4;
        std::vector<uint8_t> lh(nb);
        CK(hipMemcpy(lh.data(), d_logits_b, nb, hipMemcpyDeviceToHost));
        uint64_t fnv = 1469598103934665603ULL;
        for (uint8_t b : lh) { fnv ^= b; fnv *= 1099511628211ULL; }
        fprintf(stderr, "[vgl] base=%d logithash=%016llx v=%d,%d,%d,%d\n",
                q + 1, (unsigned long long)fnv, v[0], v[1], v[2], v[3]);
      }
      if (!spec_force.empty())  // v[j] predicts out[out.size() + j]
        for (int j = 0; j <= g; j++)
          if (out.size() + j < spec_force.size()) v[j] = spec_force[out.size() + j];
      if (fork_dump) {
        for (int j = 0; j <= g; j++) {
          int apos = q + 1 + j;
          if (apos < fork_dump - 4 || apos > fork_dump + 4) continue;
          std::vector<float> L(g_cfg.vocab);
          CK(hipMemcpy(L.data(), d_logits_b + (size_t)j * g_cfg.vocab,
                       (size_t)g_cfg.vocab * 4, hipMemcpyDeviceToHost));
          int t1 = 0, t2 = 1;
          for (int i = 1; i < g_cfg.vocab; i++)
            if (L[i] > L[t1]) { t2 = t1; t1 = i; }
            else if (t2 == t1 || L[i] > L[t2]) t2 = i;
          fprintf(stderr,
                  "[fork] pos=%d row=%d top1=%d %.4f top2=%d %.4f margin=%.6f\n",
                  apos, j, t1, L[t1], t2, L[t2], L[t1] - L[t2]);
        }
      }
      auto t2 = std::chrono::steady_clock::now();
      int a = 0;
      while (a < g && drafts[a] == v[a]) a++;
      for (int j = 0; j < g; j++) {
        depth_tot[j]++;
        if (j < a) depth_hits[j]++;
      }
      gctl.update(g, a);
      rounds++;
      live_spec_rounds = rounds;
      live_spec_proposed += g;
      live_spec_commits = (int)out.size();  // pre-emit; one round of lag is fine
      // emit() streams each committed token as it lands; returning false
      // stops the round and leaves THIS token out of the trunk state (eos,
      // or cancel after it was streamed — the next turn re-ingests it via
      // ordinary KV-reuse continuation). d = commit depth: keep
      // drafts[0..d-1]; keep_v = also keep the correction/continuation v[a].
      int d = a;
      bool stop = false, keep_v = true;
      if (emit) {
        for (int j = 0; j < a; j++)
          if (!emit(drafts[j])) { d = j; stop = true; keep_v = false; break; }
        if (!stop && !emit(v[a])) { stop = true; keep_v = false; }
      }
      for (int j = 0; j < d; j++) out.push_back(drafts[j]);
      if (keep_v) out.push_back(v[a]);
      auto t3 = std::chrono::steady_clock::now();
      if (!stop && d == g) {
        cur = v[g];  // full accept; state already at q+1+g
      } else {
        rollbacks++;
        // Install the state at commit depth d from the verify snapshots —
        // no re-prefill. KV pool entries and pooled indexer keys for the
        // accepted positions were written causally by the verify pass (and
        // rejected positions are rewritten before they can be read), so only
        // the position-mod ring slots polluted by rejected rows are restored
        // from the checkpoint. d_Rb rows 0..d hold the accepted tokens'
        // streams for the MTP ingest below.
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        const size_t sL = 48 * 128 * 128;
        for (int gi = 0; gi < ngdn; gi++)
          CK(hipMemcpyAsync(d_S + (size_t)gi * sL,
                            d_Ssnap + ((size_t)d * ngdn + gi) * sL, sL * 4,
                            hipMemcpyDeviceToDevice, g_str));
        CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)d * ngdn * 3 * 10240,
                          (size_t)ngdn * 3 * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (ckpt_iraw)
          k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
              d_iproj_snap, d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
              d_iraw, q + 1, q + d + 1, 9);
        const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
        for (int p = q + d + 2; p <= q + g + 1; p++)
          CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                            ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        chist.resize(chist_sz + d + 1);
        pos = q + d + 2;
        ple_ringpos = pos % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                          g_str));
        restore_dR_row(d);
        cur = v[d];
      }
      auto t4 = std::chrono::steady_clock::now();
      // Rebuild MTP KV over committed positions q+1..q+d with trunk taps
      // (d_Rb rows of the state-advancing pass) and committed tokens.
      CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4, hipMemcpyDeviceToDevice,
                        g_str));
      if (d > 0) {
        std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
        mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
      } else {
        mtp_pos = q + 1;
        CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
      }
      if (stop) break;
      auto t5 = std::chrono::steady_clock::now();
      t_draft += ms(t0, t1);
      t_verify += ms(t1, t2);
      t_roll += ms(t3, t4);
      t_ingest += ms(t4, t5);
      if (d_iraw) vg_dump_buf("irE", 2, d_iraw, 512 * 4);  // P3-B bisect
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    if (!bench_quiet) {
      fprintf(stderr,
              "\nspec: %d tokens in %.2f s = %.1f tok/s | rounds=%d commit/round=%.2f "
              "rollbacks=%d\n",
              (int)out.size(), sec, out.size() / sec, rounds,
              (double)out.size() / rounds, rollbacks);
      fprintf(stderr, "spec depth acc:");
      for (int j = 0; j < 8 && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr,
              "\nspec time: draft=%.0fms verify=%.0fms rollback+reprefill=%.0fms "
              "ingest=%.0fms\n",
              t_draft, t_verify, t_roll, t_ingest);
    }
    gctl.report("spec");
    last_spec_rounds = rounds;
    last_spec_commit = (int)out.size();
    last_spec_proposed = live_spec_proposed;
    last_spec_rollbacks = rollbacks;
    return out;
  }

  // Chain speculative driver (llama.cpp common/speculative.cpp style): every
  // round asks the ngram table first; an empty draft hands the round to MTP.
  // Both drafters stay live — ngram observes every round, and ngram rounds
  // patch the MTP KV over the committed positions — so either can draft the
  // next round. Round-start invariant (same as spec_loop): mtp_pos = pos-1,
  // MTP KV covers 0..pos-2, d_R = h_{pos-1} pending pairing with cur.
  std::vector<int> spec_loop_chain(int cur, int gen, int gamma,
      const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks =
        last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, ngram.n_max), maxbatch - 1});
    GammaCtl gctl(std::max(0, gamma), false);  // 0 = adaptive MTP rounds
    std::vector<int> drafts(8), v(9);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int accepted = 0, ng_rounds = 0, mtp_rounds = 0, ng_roll = 0, mtp_roll = 0;
    double verify_ms = 0, replay_ms = 0;
    // ngram verify chunking: QWENOX_NGRAM_CHUNK=0 keeps the legacy single-pass
    // 65-row verify; N pins the chunk size; unset adapts within [8,64]
    // (init 16), doubling after fully-accepted rounds, halving after rejects.
    // The level is a Model member shared with the sampling chain and kept
    // across requests (workload property, not per-request).
    ng_chunk_init();
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto verify_ng = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen && pos < maxctx) {
      yield_point();  // round top: only persistent slot state is live
      if (chist.size() != (size_t)pos)
        throw std::runtime_error(
            "chain: token history and trunk position disagree");
      ngram.observe(chist);
      // Same budget math as the pure-ngram loop: one slot stays for the
      // target-model continuation, never verify past gen or maxctx.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      const bool ng_open = ng_gate.open();
      if (ng_open && ng_gate.probing())
        want = std::min(want, std::max(8, ngram.n_min));
      std::vector<int> draft =
          (want > 0 && ng_open)
              ? ngram.draft(chist, cur, want)
              : std::vector<int>{};
      bool stop = false;
      if (!draft.empty() && ng_legacy) {
        // ---- ngram round: long draft, whole-state checkpoint + replay ----
        const int base = pos;
        const size_t history_size = chist.size();
        // Stash h_{base-1} (d_R) for the MTP patch below: prefill_batch
        // overwrites d_R with its last row. d_mtap is idle outside prefill.
        CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        ngram_ckpt_save();
        std::vector<int> tokens{cur};
        tokens.insert(tokens.end(), draft.begin(), draft.end());
        auto t = std::chrono::steady_clock::now();
        verify_ng(tokens, base);
        std::vector<int> targets(tokens.size());
        batch_argmax((int)tokens.size(), targets.data());
        verify_ms += elapsed(t);
        int a = 0;
        while (a < (int)draft.size() && draft[a] == targets[a]) ++a;
        int keep = a;
        for (int j = 0; j < a; ++j) {
          if (emit && !emit(draft[j])) { keep = j; stop = true; break; }
          out.push_back(draft[j]);
        }
        if (!stop) {
          if (emit && !emit(targets[a])) stop = true;
          else out.push_back(targets[a]);
        }
        if (keep != (int)draft.size()) {
          // Restore every overwritten position-mod state, then replay exactly
          // [cur, accepted drafts] (same as spec_loop_ngram).
          t = std::chrono::steady_clock::now();
          ngram_ckpt_restore();
          chist.resize(history_size);
          pos = base;
          ple_ringpos = base % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          tokens.resize(keep + 1);
          verify_ng(tokens, base);
          CK(hipStreamSynchronize(g_str));
          replay_ms += elapsed(t);
          ++ng_roll;
        }
        cur = targets[keep];
        accepted += keep;
        ++ng_rounds;
        ng_gate.update(keep, (int)draft.size());
        last_spec_proposed += (int)draft.size();
        // Patch the MTP KV over committed positions base-1..base+keep-1: tap
        // row 0 = stashed h_{base-1} paired with cur; rows 1..keep = d_Rb
        // rows 0..keep-1 (the committed tokens' trunk streams — the replay
        // re-generated them on partial accept). This restores the round-start
        // invariant (mtp_pos = pos-1), keeping MTP drafts live for later
        // rounds. d_R itself already holds h_{pos-1} (verify/replay leave the
        // last processed row's stream there; mtp_ingest_b does not touch it).
        if (keep > 0)
          CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                            (size_t)keep * 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        mtp_ingest_b(d_mtap,
                     std::vector<int>(tokens.begin(),
                                      tokens.begin() + keep + 1),
                     keep + 1, base - 1);
      } else if (!draft.empty()) {
        // ---- ngram round: chunked verify, tokens emit as each chunk ----
        // accepts. A rejection restores only this chunk's checkpoint and
        // replays its committed prefix (bounded by ng_cs), replacing the
        // legacy 65-row verify + keep-sized replay. ng_cs doubles after a
        // fully-accepted round (hot tables return to single-pass 65-row
        // throughput) and halves after a rejection. MTP KV is patched per
        // chunk while its d_Rb rows are still live; d_mtap row 0 carries the
        // h_{base-1} stash for the chunk about to verify (verify/replay never
        // touch d_mtap — mtp_tap_capture is off outside prefill).
        CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        int cur_c = cur, keep_round = 0;
        bool round_full = true;
        int di = 0;
        while (di < (int)draft.size()) {
          const int cs = std::min(ng_cs, (int)draft.size() - di);
          const int base = pos;
          const size_t history_size = chist.size();
          ngram_ckpt_save();
          std::vector<int> tokens{cur_c};
          tokens.insert(tokens.end(), draft.begin() + di,
                        draft.begin() + di + cs);
          auto t = std::chrono::steady_clock::now();
          verify_ng(tokens, base);
          std::vector<int> targets(tokens.size());
          batch_argmax((int)tokens.size(), targets.data());
          verify_ms += elapsed(t);
          int a = 0;
          while (a < cs && draft[di + a] == targets[a]) ++a;
          int keep = a;
          for (int j = 0; j < a; ++j) {
            if (emit && !emit(draft[di + j])) { keep = j; stop = true; break; }
            out.push_back(draft[di + j]);
          }
          if (!stop) {
            if (emit && !emit(targets[a])) stop = true;
            else out.push_back(targets[a]);
          }
          if (keep != cs) {
            // Mid-chunk rejection (or consumer stop): rows keep+1..cs
            // polluted the state — restore this chunk's checkpoint and
            // replay exactly its committed prefix. The round ends here.
            t = std::chrono::steady_clock::now();
            ngram_ckpt_restore();
            chist.resize(history_size);
            pos = base;
            ple_ringpos = base % 9;
            CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
            CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4,
                              hipMemcpyHostToDevice, g_str));
            tokens.resize(keep + 1);
            verify_ng(tokens, base);
            CK(hipStreamSynchronize(g_str));
            replay_ms += elapsed(t);
            ++ng_roll;
            round_full = false;
          }
          // MTP patch for this chunk's committed positions (same tap layout
          // as the legacy round: row 0 = stash, rows 1..keep = d_Rb rows).
          if (keep > 0)
            CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                              (size_t)keep * 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          mtp_ingest_b(d_mtap,
                       std::vector<int>(tokens.begin(),
                                        tokens.begin() + keep + 1),
                       keep + 1, base - 1);
          keep_round += keep;
          cur_c = targets[keep];
          if (!round_full || stop) break;
          // Chunk fully accepted: the continuation targets[cs] fills the
          // position the next draft token proposes. Chain into the next
          // chunk only when it equals that draft token (a mismatch is a
          // boundary rejection: the correction is already committed, the
          // state is clean, and the round ends without a replay).
          if (di + cs >= (int)draft.size()) break;  // bonus row: draft done
          if (targets[cs] != draft[di + cs]) { round_full = false; break; }
          ++keep_round;  // the splice committed draft[di+cs] as a draft
          di += cs + 1;
          // Stash h_{pos-1} (d_R = the accepted chunk's last row) as the next
          // chunk's tap pair before its verify overwrites d_R.
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        }
        cur = cur_c;
        accepted += keep_round;
        ++ng_rounds;
        ng_gate.update(keep_round, (int)draft.size());
        last_spec_proposed += (int)draft.size();
        if (!ng_cs_fixed) {
          if (round_full && (int)draft.size() > ng_cs)
            ng_cs = std::min(64, ng_cs * 2);
          else if (!round_full)
            ng_cs = std::max(8, ng_cs / 2);
        }
      } else {
        // ---- MTP round (ngram table missed): short draft, snapshot rollback
        // ---- (same machinery as spec_loop)
        int q = pos - 1;
        int g = std::min({gctl.cur, gen - (int)out.size(), maxctx - pos - 1});
        if (g < 1) break;  // context edge: no room for even one draft
        size_t chist_sz = chist.size();
        mtp_ckpt_save();
        auto t0 = std::chrono::steady_clock::now();
        // draft 1's indexer ring rows are truth; later drafts pollute them.
        if (draft_desync) {
          mtp_step(d_R, cur, nullptr, true, nullptr, d_argmax_arr);
          CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          for (int j = 1; j < g; j++)
            mtp_step(d_mR, 0, nullptr, true, d_argmax_arr + j - 1,
                     d_argmax_arr + j);
          CK(hipMemcpy(drafts.data(), d_argmax_arr, (size_t)g * sizeof(int),
                       hipMemcpyDeviceToHost));
        } else {
          drafts[0] = mtp_step(d_R, cur);
          CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          for (int j = 1; j < g; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
        }
        std::vector<int> vt;
        vt.reserve(g + 1);
        vt.push_back(cur);
        for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
        spec_snap = true;  // capture per-row rollback snapshots during verify
        if (!verify_graph_round_try(vt, q + 1, v.data())) {
          prefill_batch(vt, q + 1, false, true);
          spec_snap = false;
          batch_argmax(g + 1, v.data());
        } else
          spec_snap = false;
        verify_ms += elapsed(t0);
        int a = 0;
        while (a < g && drafts[a] == v[a]) a++;
        for (int j = 0; j < g; j++) {
          depth_tot[j]++;
          if (j < a) depth_hits[j]++;
        }
        gctl.update(g, a);
        ++mtp_rounds;
        last_spec_proposed += g;
        int d = a;
        bool keep_v = true;
        if (emit) {
          for (int j = 0; j < a; j++)
            if (!emit(drafts[j])) { d = j; stop = true; keep_v = false; break; }
          if (!stop && !emit(v[a])) { stop = true; keep_v = false; }
        }
        for (int j = 0; j < d; j++) out.push_back(drafts[j]);
        if (keep_v) out.push_back(v[a]);
        accepted += d;
        if (!stop && d == g) {
          cur = v[g];  // full accept; state already at q+1+g
        } else {
          ++mtp_roll;
          // Install the state at commit depth d from the verify snapshots —
          // identical to spec_loop's rollback.
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          const size_t sL = 48 * 128 * 128;
          for (int gi = 0; gi < ngdn; gi++)
            CK(hipMemcpyAsync(d_S + (size_t)gi * sL,
                              d_Ssnap + ((size_t)d * ngdn + gi) * sL, sL * 4,
                              hipMemcpyDeviceToDevice, g_str));
          CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)d * ngdn * 3 * 10240,
                            (size_t)ngdn * 3 * 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          if (ckpt_iraw)
            k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
                d_iproj_snap, d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
                d_iraw, q + 1, q + d + 1, 9);
          const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
          for (int p = q + d + 2; p <= q + g + 1; p++)
            CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                              ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          pos = q + d + 2;
          ple_ringpos = pos % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          restore_dR_row(d);
          cur = v[d];
        }
        // Committed-token history: rebuild explicitly so it is right whether
        // or not ple_on made the verify append (spec_loop relies on ple_on).
        chist.resize(chist_sz);
        chist.insert(chist.end(), vt.begin(), vt.begin() + d + 1);
        // MTP self-repair over committed positions q+1..q+d (as spec_loop).
        CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (d > 0) {
          std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
          mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
        } else {
          mtp_pos = q + 1;
          CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
        }
      }
      ++last_spec_rounds;
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = (int)out.size();
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    last_spec_commit = (int)out.size();
    last_spec_rollbacks = ng_roll + mtp_roll;
    fprintf(stderr,
            "%s chain: %d tokens in %.2f s = %.1f tok/s | rounds=%d (ngram=%d "
            "mtp=%d) commit/round=%.2f accepted=%d proposed=%d rollbacks=%d\n",
            log_ts(), (int)out.size(), sec, sec > 0 ? out.size() / sec : 0.0,
            last_spec_rounds, ng_rounds, mtp_rounds,
            last_spec_rounds ? (double)out.size() / last_spec_rounds : 0.0,
            accepted, last_spec_proposed, last_spec_rollbacks);
    if (mtp_rounds) {
      fprintf(stderr, "chain mtp depth acc:");
      for (int j = 0; j < 8 && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr, "\n");
      gctl.report("chain");
    }
    return out;
  }

  // Sampling speculative decode. The sampler supplies normalized filtered
  // distributions, independent draft/accept RNG streams, and residual draws.
  // Accepted draft tokens use p/q rejection sampling; a rejection draws from
  // normalized max(p-q, 0), while full acceptance draws the verify continuation.
  template <typename Sampler>
  std::vector<int> spec_loop_sample(
      int cur, int gen, int gamma, Sampler& sampler,
      const std::function<bool(int, float)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(gen + 1);
    GammaCtl gctl(gamma, true);
    std::vector<int> drafts(8);
    std::vector<float> draft_lps(8), verify_logits, p_row(g_cfg.vocab);
    std::vector<std::vector<float>> q_rows(8);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int rounds = 0, rollbacks = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, t_ingest = 0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen) {
      yield_point();  // round top: mtp_pos == pos-1, d_R == h_{pos-1}
      int q = pos - 1;
      int g = std::min(gctl.cur, gen - (int)out.size());
      size_t chist_sz = chist.size();
      mtp_ckpt_save();
      Sampler draft_sampler = sampler.fork_stream(
          0xd1b54a32d192ed03ull ^ (uint64_t)(q + 1));
      Sampler accept_sampler = sampler.fork_stream(
          0x3c6ef372fe94f82aull ^ (uint64_t)(q + 1));
      auto t0 = std::chrono::steady_clock::now();
      // GPU sampling prep (direction A): corrections + top-k on device, host
      // replays the filter chain on <= top_k candidates per row. The dense
      // path remains as fallback (QWENOX_SMS_GPU=0, top_k unset/>64, overflow).
      const bool smg = sms_gpu_active(sampler) && sms_round_shared(sampler);
      std::vector<typename Sampler::SDist> q_sd(smg ? g : 0);
      std::vector<char> q_sparse(g, 0);
      for (int j = 0; j < g; j++) {
        bool dense = true;
        if (smg) {
          mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, nullptr, false);
          if (sms_topk_rows(d_logits, g_cfg.vocab, 1, sampler.top_k, false,
                            sms_out_ids.data(), sms_out_vals.data())) {
            dense = false;
          } else {
            // Candidate overflow (never expected): corrections touched the
            // logits in place, so regenerate the raw row before going dense.
            gemv(dlm_on ? w_dlm : wview(ckpt.at("lm_head.weight")), d_x,
                 d_logits);
            q_rows[j].resize(g_cfg.vocab);
            CK(hipMemcpy(q_rows[j].data(), d_logits, (size_t)g_cfg.vocab * 4,
                         hipMemcpyDeviceToHost));
          }
        } else {
          mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, &q_rows[j]);
        }
        float qlp = 0.0f;
        // Penalties are defined over committed/emitted tokens only. Provisional
        // draft tokens therefore do not enter the proposal sampler's history;
        // q_rows[j]/q_sd[j] records the distribution that produced drafts[j].
        if (dense) {
          draft_sampler.prepare(q_rows[j]);
          drafts[j] = draft_sampler.draw_prepared(q_rows[j], &qlp, false);
        } else {
          sms_prep(draft_sampler, 0, true, q_sd[j]);
          drafts[j] = draft_sampler.draw_sparse(q_sd[j], &qlp, false);
          q_sparse[j] = 1;
        }
        if (j == 0) {
          // MTP indexer state after consuming cur is causal truth. Later
          // speculative draft steps are discarded before committed ingest.
          CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      std::vector<int> vt;
      vt.reserve(g + 1);
      vt.push_back(cur);
      for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
      spec_snap = true;
      prefill_batch(vt, q + 1, true, true);
      spec_snap = false;
      bool smg_v = false;
      if (smg) {
        const bool pfx = sms_prefix_upload(sampler, drafts.data(), 0, g + 1);
        batch_logits_device(g + 1);
        smg_v = sms_topk_rows(d_logits_b, g_cfg.vocab, g + 1, sampler.top_k,
                              pfx, sms_out_ids.data(), sms_out_vals.data());
      }
      if (!smg_v) batch_logits(g + 1, verify_logits);
      const int smk = sampler.top_k;

      Sampler target_view = sampler;
      std::vector<typename Sampler::SDist> p_sd(smg_v ? g + 1 : 0);
      int a = 0;
      // q_dense[a]: q_rows[a] holds a real dense distribution. The sparse
      // draft path leaves q_rows[a] stale, so scatter q_sd[a] when the two
      // sides disagree (rare per-row fallback mixing).
      auto q_dense = [&](int r) -> std::vector<float>& {
        if (q_sparse[r]) {
          q_sparse[r] = 0;
          q_rows[r].assign(g_cfg.vocab, 0.0f);
          for (size_t u = 0; u < q_sd[r].ids.size(); u++)
            q_rows[r][q_sd[r].ids[u]] = q_sd[r].pr[u];
        }
        return q_rows[r];
      };
      for (; a < g; a++) {
        float px, qx;
        const bool q_sp = q_sparse[a];
        if (smg_v && q_sp) {
          sms_prep(target_view, a, false, p_sd[a]);
          px = Sampler::sdist_prob(p_sd[a], drafts[a]);
          qx = Sampler::sdist_prob(q_sd[a], drafts[a]);
        } else {
          if (smg_v) {
            // Sparse verify but dense draft row: materialize p densely.
            sms_prep(target_view, a, false, p_sd[a]);
            p_row.assign(g_cfg.vocab, 0.0f);
            for (size_t u = 0; u < p_sd[a].ids.size(); u++)
              p_row[p_sd[a].ids[u]] = p_sd[a].pr[u];
          } else {
            std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                        g_cfg.vocab, p_row.data());
            target_view.prepare(p_row);
          }
          px = p_row[drafts[a]];
          qx = q_dense(a)[drafts[a]];
        }
        draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
        double threshold = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
        if (accept_sampler.uniform() >= threshold) break;
        target_view.remember(drafts[a]);
      }

      int next;
      float next_lp = 0.0f;
      target_view.rng = sampler.rng;
      if (a < g) {
        next = (smg_v && q_sparse[a])
                   ? target_view.sample_residual_sparse(p_sd[a], q_sd[a],
                                                        &next_lp)
                   : target_view.sample_residual(p_row, q_dense(a),
                                                 &next_lp);
      } else if (smg_v) {
        sms_prep(target_view, g, false, p_sd[g]);
        next = target_view.draw_sparse(p_sd[g], &next_lp, false);
      } else {
        std::copy_n(verify_logits.data() + (size_t)g * g_cfg.vocab,
                    g_cfg.vocab, p_row.data());
        target_view.prepare(p_row);
        next = target_view.draw_prepared(p_row, &next_lp, false);
      }
      sampler.rng = target_view.rng;
      auto t2 = std::chrono::steady_clock::now();
      for (int j = 0; j < g; j++) {
        depth_tot[j]++;
        if (j < a) depth_hits[j]++;
      }
      gctl.update(g, a);
      rounds++;
      live_spec_rounds = rounds;
      live_spec_proposed += g;
      live_spec_commits = (int)out.size();  // pre-emit; one round of lag is fine

      int d = a, stopped_token = -1;
      bool stop = false, keep_next = true;
      if (emit) {
        for (int j = 0; j < a; j++)
          if (!emit(drafts[j], draft_lps[j])) {
            d = j;
            stopped_token = drafts[j];
            stop = true;
            keep_next = false;
            break;
          }
        if (!stop && !emit(next, next_lp)) {
          stop = true;
          keep_next = false;
        }
      }
      for (int j = 0; j < d; j++) {
        out.push_back(drafts[j]);
        sampler.remember(drafts[j]);
      }
      if (keep_next) {
        out.push_back(next);
        sampler.remember(next);
      }

      auto t3 = std::chrono::steady_clock::now();
      if (!stop && d == g) {
        cur = next;
      } else {
        rollbacks++;
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        const size_t sL = 48 * 128 * 128;
        for (int gi = 0; gi < ngdn; gi++)
          CK(hipMemcpyAsync(d_S + (size_t)gi * sL,
                            d_Ssnap + ((size_t)d * ngdn + gi) * sL, sL * 4,
                            hipMemcpyDeviceToDevice, g_str));
        CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)d * ngdn * 3 * 10240,
                          (size_t)ngdn * 3 * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (ckpt_iraw)
          k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
              d_iproj_snap, d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
              d_iraw, q + 1, q + d + 1, 9);
        const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
        for (int p = q + d + 2; p <= q + g + 1; p++)
          CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                            ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        chist.resize(chist_sz + d + 1);
        pos = q + d + 2;
        ple_ringpos = pos % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                          g_str));
        restore_dR_row(d);
        cur = stopped_token >= 0 ? stopped_token : next;
      }
      auto t4 = std::chrono::steady_clock::now();
      CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                        hipMemcpyDeviceToDevice, g_str));
      if (d > 0) {
        std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
        mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
      } else {
        mtp_pos = q + 1;
        CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
      }
      if (stop) break;
      auto t5 = std::chrono::steady_clock::now();
      t_draft += ms(t0, t1);
      t_verify += ms(t1, t2);
      t_roll += ms(t3, t4);
      t_ingest += ms(t4, t5);
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    fprintf(stderr,
            "\nspec-sample: %d tokens in %.2f s = %.1f tok/s | rounds=%d "
            "commit/round=%.2f rollbacks=%d\n",
            (int)out.size(), sec, out.size() / sec, rounds,
            rounds ? (double)out.size() / rounds : 0.0, rollbacks);
    fprintf(stderr, "spec-sample depth acc:");
    for (int j = 0; j < 8 && depth_tot[j]; j++)
      fprintf(stderr, " %d:%.3f", j + 1,
              (double)depth_hits[j] / depth_tot[j]);
    fprintf(stderr,
            "\nspec-sample time: draft=%.0fms verify=%.0fms "
            "rollback=%.0fms ingest=%.0fms\n",
            t_draft, t_verify, t_roll, t_ingest);
    gctl.report("spec-sample");
    last_spec_rounds = rounds;
    last_spec_commit = (int)out.size();
    last_spec_proposed = live_spec_proposed;
    last_spec_rollbacks = rollbacks;
    return out;
  }

  // Sampling chain (llama.cpp ngram semantics + rejection-sampled MTP leg).
  // ngram round: a draft token is committed only when it equals a fresh draw
  // from the target distribution at its verify row; the first mismatching
  // draw becomes the correction token. Every emitted token is therefore an
  // exact target-distribution sample — the draft only decides how many
  // verify rows stay usable. Rollback/replay is the greedy ngram round's
  // checkpoint machinery; the MTP patch over committed positions is the
  // greedy chain's. The MTP round is spec_loop_sample's round body with the
  // greedy chain's explicit chist rebuild. mtp_ok=false degrades empty-draft
  // rounds to plain serial sampling steps (MTP KV unavailable).
  template <typename Sampler>
  std::vector<int> spec_loop_chain_sample(
      int cur, int gen, int gamma, Sampler& sampler, bool mtp_ok,
      const std::function<bool(int, float)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks =
        last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, ngram.n_max), maxbatch - 1});
    GammaCtl gctl(std::max(0, gamma), true);  // 0 = adaptive MTP rounds
    std::vector<int> drafts(8);
    std::vector<float> draft_lps(8), verify_logits, p_row(g_cfg.vocab);
    std::vector<std::vector<float>> q_rows(8);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int accepted = 0, ng_rounds = 0, mtp_rounds = 0, ng_roll = 0,
        mtp_roll = 0, fallback = 0;
    double verify_ms = 0, replay_ms = 0;
    // ngram verify chunking: same env semantics as the greedy chain; the
    // level lives in the shared Model member (see ng_chunk_init).
    ng_chunk_init();
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto verify_ng = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen && pos < maxctx) {
      yield_point();  // round top: only persistent slot state is live
      if (chist.size() != (size_t)pos)
        throw std::runtime_error(
            "chain-sample: token history and trunk position disagree");
      ngram.observe(chist);
      // Same budget math as the greedy chain: one slot stays for the
      // target-model continuation, never verify past gen or maxctx.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      const bool ng_open = ng_gate.open();
      if (ng_open && ng_gate.probing())
        want = std::min(want, std::max(8, ngram.n_min));
      std::vector<int> draft =
          (want > 0 && ng_open)
              ? ngram.draft(chist, cur, want)
              : std::vector<int>{};
      bool stop = false;
      // GPU sampling prep (direction A) applies to all three legs below.
      const bool smg = sms_gpu_active(sampler) && sms_round_shared(sampler);
      if (!draft.empty() && ng_legacy) {
        // ---- ngram round: long draft, whole-state checkpoint + replay ----
        const int base = pos;
        const size_t history_size = chist.size();
        if (mtp_ok)
          // Stash h_{base-1} (d_R) for the MTP patch below (as spec_loop_chain).
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        ngram_ckpt_save();
        std::vector<int> tokens{cur};
        tokens.insert(tokens.end(), draft.begin(), draft.end());
        auto t = std::chrono::steady_clock::now();
        verify_ng(tokens, base);
        const int ngP = (int)tokens.size();
        bool smg_v = false;
        if (smg) {
          const bool pfx = sms_prefix_upload(sampler, draft.data(), 0, ngP);
          batch_logits_device(ngP);
          smg_v = sms_topk_rows(d_logits_b, g_cfg.vocab, ngP, sampler.top_k,
                                pfx, sms_out_ids.data(), sms_out_vals.data());
        }
        if (!smg_v) batch_logits(ngP, verify_logits);
        const int smk = sampler.top_k;
        verify_ms += elapsed(t);
        // Walk the draft rows with a scratch sampler view: accept draft[a]
        // iff it equals this round's draw from row a, else the draw is the
        // correction. keep = committed draft count (emit may stop early).
        Sampler view = sampler;
        const size_t committed0 = out.size();
        std::vector<typename Sampler::SDist> p_sd(smg_v ? ngP : 0);
        int a = 0, next = -1;
        float next_lp = 0.0f;
        for (; a < (int)draft.size(); a++) {
          int s;
          float lp_a = 0.0f;
          if (smg_v) {
            sms_prep(view, a, false, p_sd[a]);
            s = view.draw_sparse(p_sd[a], &lp_a, true);
          } else {
            std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                        g_cfg.vocab, p_row.data());
            s = view.sample(p_row, &lp_a);
          }
          if (s != draft[a]) {
            next = s;
            next_lp = lp_a;
            break;
          }
          if (emit && !emit(draft[a], lp_a)) { stop = true; break; }
          out.push_back(draft[a]);
        }
        const int keep = a;
        if (!stop) {
          if (next < 0) {
            // Full accept: the bonus row continues the sequence.
            if (smg_v) {
              sms_prep(view, draft.size(), false, p_sd[draft.size()]);
              next = view.draw_sparse(p_sd[draft.size()], &next_lp, true);
            } else {
              std::copy_n(verify_logits.data() +
                              (size_t)draft.size() * g_cfg.vocab,
                          g_cfg.vocab, p_row.data());
              next = view.sample(p_row, &next_lp);
            }
          }
          if (emit && !emit(next, next_lp)) stop = true;
          else out.push_back(next);
        }
        sampler.rng = view.rng;
        for (size_t j = committed0; j < out.size(); j++)
          sampler.remember(out[j]);
        if (keep != (int)draft.size()) {
          // Restore every overwritten position-mod state, then replay exactly
          // [cur, accepted drafts] (same as the greedy ngram round).
          t = std::chrono::steady_clock::now();
          ngram_ckpt_restore();
          chist.resize(history_size);
          pos = base;
          ple_ringpos = base % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          tokens.resize(keep + 1);
          verify_ng(tokens, base);
          CK(hipStreamSynchronize(g_str));
          replay_ms += elapsed(t);
          ++ng_roll;
        }
        if (next >= 0) cur = next;
        accepted += keep;
        ++ng_rounds;
        ng_gate.update(keep, (int)draft.size());
        last_spec_proposed += (int)draft.size();
        if (mtp_ok) {
          // Patch the MTP KV over committed positions (as spec_loop_chain):
          // tap row 0 = stashed h_{base-1} paired with cur; rows 1..keep the
          // committed tokens' trunk streams. Restores mtp_pos = pos-1.
          if (keep > 0)
            CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                              (size_t)keep * 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          mtp_ingest_b(d_mtap,
                       std::vector<int>(tokens.begin(),
                                        tokens.begin() + keep + 1),
                       keep + 1, base - 1);
        }
      } else if (!draft.empty()) {
        // ---- chunked ngram round, sampling semantics: the Sampler view ----
        // walks rows in draft order across chunks, so the draw stream matches
        // the single-pass form; tokens emit per chunk (see the greedy chain).
        if (mtp_ok)
          // Stash h_{base-1} (d_R) for the per-chunk MTP patch below.
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        Sampler view = sampler;
        const size_t committed0 = out.size();
        int cur_c = cur, keep_round = 0;
        bool round_full = true;
        int cont = -1;
        float cont_lp = 0.0f;
        int di = 0;
        while (di < (int)draft.size()) {
          const int cs = std::min(ng_cs, (int)draft.size() - di);
          const int base = pos;
          const size_t history_size = chist.size();
          ngram_ckpt_save();
          std::vector<int> tokens{cur_c};
          tokens.insert(tokens.end(), draft.begin() + di,
                        draft.begin() + di + cs);
          auto t = std::chrono::steady_clock::now();
          verify_ng(tokens, base);
          const int ngP = (int)tokens.size();
          bool smg_v = false;
          if (smg) {
            // Row a of this chunk is prepared with draft[0..di+a) in the
            // penalty history (the walk only reaches it via matching draws).
            const bool pfx = sms_prefix_upload(sampler, draft.data(), di, ngP);
            batch_logits_device(ngP);
            smg_v =
                sms_topk_rows(d_logits_b, g_cfg.vocab, ngP, sampler.top_k,
                              pfx, sms_out_ids.data(), sms_out_vals.data());
          }
          if (!smg_v) batch_logits(ngP, verify_logits);
          const int smk = sampler.top_k;
          verify_ms += elapsed(t);
          std::vector<typename Sampler::SDist> p_sd(smg_v ? ngP : 0);
          cont = -1;
          int a = 0;
          for (; a < cs; a++) {
            int s;
            float lp_a = 0.0f;
            if (smg_v) {
              sms_prep(view, a, false, p_sd[a]);
              s = view.draw_sparse(p_sd[a], &lp_a, true);
            } else {
              std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                          g_cfg.vocab, p_row.data());
              s = view.sample(p_row, &lp_a);
            }
            if (s != draft[di + a]) { cont = s; cont_lp = lp_a; break; }
            if (emit && !emit(draft[di + a], lp_a)) { stop = true; break; }
            out.push_back(draft[di + a]);
          }
          const int keep = a;
          if (!stop) {
            if (cont < 0) {
              // Full chunk accept: the bonus row continues the sequence.
              if (smg_v) {
                sms_prep(view, cs, false, p_sd[cs]);
                cont = view.draw_sparse(p_sd[cs], &cont_lp, true);
              } else {
                std::copy_n(verify_logits.data() + (size_t)cs * g_cfg.vocab,
                            g_cfg.vocab, p_row.data());
                cont = view.sample(p_row, &cont_lp);
              }
            }
            if (emit && !emit(cont, cont_lp)) stop = true;
            else out.push_back(cont);
          }
          if (keep != cs) {
            // Mid-chunk rejection: restore + replay only this chunk's
            // committed prefix (the greedy block has the full commentary).
            t = std::chrono::steady_clock::now();
            ngram_ckpt_restore();
            chist.resize(history_size);
            pos = base;
            ple_ringpos = base % 9;
            CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
            CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4,
                              hipMemcpyHostToDevice, g_str));
            tokens.resize(keep + 1);
            verify_ng(tokens, base);
            CK(hipStreamSynchronize(g_str));
            replay_ms += elapsed(t);
            ++ng_roll;
            round_full = false;
          }
          if (mtp_ok) {
            if (keep > 0)
              CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                                (size_t)keep * 10240 * 4,
                                hipMemcpyDeviceToDevice, g_str));
            mtp_ingest_b(d_mtap,
                         std::vector<int>(tokens.begin(),
                                          tokens.begin() + keep + 1),
                         keep + 1, base - 1);
          }
          keep_round += keep;
          if (cont >= 0) cur_c = cont;
          if (!round_full || stop) break;
          // Full chunk accept: chain into the next chunk only when the drawn
          // continuation equals the next draft token (see the greedy block);
          // a mismatch is a clean boundary rejection — no replay needed.
          if (di + cs >= (int)draft.size()) break;  // bonus row: draft done
          if (cont != draft[di + cs]) { round_full = false; break; }
          ++keep_round;  // the splice committed draft[di+cs] as a draft
          di += cs + 1;
          if (mtp_ok)
            CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                              g_str));
        }
        sampler.rng = view.rng;
        for (size_t j = committed0; j < out.size(); j++)
          sampler.remember(out[j]);
        if (cont >= 0) cur = cur_c;
        accepted += keep_round;
        ++ng_rounds;
        ng_gate.update(keep_round, (int)draft.size());
        last_spec_proposed += (int)draft.size();
        if (!ng_cs_fixed) {
          if (round_full && (int)draft.size() > ng_cs)
            ng_cs = std::min(64, ng_cs * 2);
          else if (!round_full)
            ng_cs = std::max(8, ng_cs / 2);
        }
      } else if (mtp_ok) {
        // ---- MTP round (ngram table missed): spec_loop_sample's round ----
        int q = pos - 1;
        int g = std::min({gctl.cur, gen - (int)out.size(), maxctx - pos - 1});
        if (g < 1) break;  // context edge: no room for even one draft
        size_t chist_sz = chist.size();
        mtp_ckpt_save();
        Sampler draft_sampler = sampler.fork_stream(
            0xd1b54a32d192ed03ull ^ (uint64_t)(q + 1));
        Sampler accept_sampler = sampler.fork_stream(
            0x3c6ef372fe94f82aull ^ (uint64_t)(q + 1));
        auto t = std::chrono::steady_clock::now();
        // GPU sampling prep (direction A), as in spec_loop_sample.
        std::vector<typename Sampler::SDist> q_sd(smg ? g : 0);
        std::vector<char> q_sparse(g, 0);
        for (int j = 0; j < g; j++) {
          bool dense = true;
          if (smg) {
            mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, nullptr, false);
            if (sms_topk_rows(d_logits, g_cfg.vocab, 1, sampler.top_k, false,
                              sms_out_ids.data(), sms_out_vals.data())) {
              dense = false;
            } else {
              // Candidate overflow: regenerate the raw row, then go dense.
              gemv(dlm_on ? w_dlm : wview(ckpt.at("lm_head.weight")), d_x,
                   d_logits);
              q_rows[j].resize(g_cfg.vocab);
              CK(hipMemcpy(q_rows[j].data(), d_logits,
                           (size_t)g_cfg.vocab * 4, hipMemcpyDeviceToHost));
            }
          } else {
            mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, &q_rows[j]);
          }
          // Provisional drafts stay out of the penalty history (as
          // spec_loop_sample): q_rows[j]/q_sd[j] records the distribution.
          if (dense) {
            draft_sampler.prepare(q_rows[j]);
            drafts[j] =
                draft_sampler.draw_prepared(q_rows[j], &draft_lps[j], false);
          } else {
            sms_prep(draft_sampler, 0, true, q_sd[j]);
            drafts[j] =
                draft_sampler.draw_sparse(q_sd[j], &draft_lps[j], false);
            q_sparse[j] = 1;
          }
          if (j == 0) {
            // Draft 1's indexer ring rows are truth; later drafts pollute.
            CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          }
        }
        std::vector<int> vt;
        vt.reserve(g + 1);
        vt.push_back(cur);
        for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
        spec_snap = true;  // capture per-row rollback snapshots during verify
        prefill_batch(vt, q + 1, true, true);
        spec_snap = false;
        bool smg_v = false;
        if (smg) {
          const bool pfx = sms_prefix_upload(sampler, drafts.data(), 0, g + 1);
          batch_logits_device(g + 1);
          smg_v = sms_topk_rows(d_logits_b, g_cfg.vocab, g + 1, sampler.top_k,
                                pfx, sms_out_ids.data(), sms_out_vals.data());
        }
        if (!smg_v) batch_logits(g + 1, verify_logits);
        const int smk = sampler.top_k;
        verify_ms += elapsed(t);
        Sampler target_view = sampler;
        std::vector<typename Sampler::SDist> p_sd(smg_v ? g + 1 : 0);
        int a = 0;
        auto q_dense = [&](int r) -> std::vector<float>& {
          if (q_sparse[r]) {
            q_sparse[r] = 0;
            q_rows[r].assign(g_cfg.vocab, 0.0f);
            for (size_t u = 0; u < q_sd[r].ids.size(); u++)
              q_rows[r][q_sd[r].ids[u]] = q_sd[r].pr[u];
          }
          return q_rows[r];
        };
        for (; a < g; a++) {
          float px, qx;
          const bool q_sp = q_sparse[a];
          if (smg_v && q_sp) {
            sms_prep(target_view, a, false, p_sd[a]);
            px = Sampler::sdist_prob(p_sd[a], drafts[a]);
            qx = Sampler::sdist_prob(q_sd[a], drafts[a]);
          } else {
            if (smg_v) {
              // Sparse verify but dense draft row: materialize p densely.
              sms_prep(target_view, a, false, p_sd[a]);
              p_row.assign(g_cfg.vocab, 0.0f);
              for (size_t u = 0; u < p_sd[a].ids.size(); u++)
                p_row[p_sd[a].ids[u]] = p_sd[a].pr[u];
            } else {
              std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                          g_cfg.vocab, p_row.data());
              target_view.prepare(p_row);
            }
            px = p_row[drafts[a]];
            qx = q_dense(a)[drafts[a]];
          }
          draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
          double threshold = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
          if (accept_sampler.uniform() >= threshold) break;
          target_view.remember(drafts[a]);
        }
        int next;
        float next_lp = 0.0f;
        target_view.rng = sampler.rng;
        if (a < g) {
          next = (smg_v && q_sparse[a])
                     ? target_view.sample_residual_sparse(p_sd[a], q_sd[a],
                                                          &next_lp)
                     : target_view.sample_residual(p_row, q_dense(a),
                                                   &next_lp);
        } else if (smg_v) {
          sms_prep(target_view, g, false, p_sd[g]);
          next = target_view.draw_sparse(p_sd[g], &next_lp, false);
        } else {
          std::copy_n(verify_logits.data() + (size_t)g * g_cfg.vocab,
                      g_cfg.vocab, p_row.data());
          target_view.prepare(p_row);
          next = target_view.draw_prepared(p_row, &next_lp, false);
        }
        sampler.rng = target_view.rng;
        for (int j = 0; j < g; j++) {
          depth_tot[j]++;
          if (j < a) depth_hits[j]++;
        }
        gctl.update(g, a);
        ++mtp_rounds;
        last_spec_proposed += g;
        int d = a, stopped_token = -1;
        bool keep_next = true;
        if (emit) {
          for (int j = 0; j < a; j++)
            if (!emit(drafts[j], draft_lps[j])) {
              d = j;
              stopped_token = drafts[j];
              stop = true;
              keep_next = false;
              break;
            }
          if (!stop && !emit(next, next_lp)) { stop = true; keep_next = false; }
        }
        for (int j = 0; j < d; j++) {
          out.push_back(drafts[j]);
          sampler.remember(drafts[j]);
        }
        if (keep_next) {
          out.push_back(next);
          sampler.remember(next);
        }
        accepted += d;
        if (!stop && d == g) {
          cur = next;  // full accept; state already at q+1+g
        } else {
          ++mtp_roll;
          // Install the state at commit depth d from the verify snapshots —
          // identical to spec_loop_sample's rollback.
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          const size_t sL = 48 * 128 * 128;
          for (int gi = 0; gi < ngdn; gi++)
            CK(hipMemcpyAsync(d_S + (size_t)gi * sL,
                              d_Ssnap + ((size_t)d * ngdn + gi) * sL, sL * 4,
                              hipMemcpyDeviceToDevice, g_str));
          CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)d * ngdn * 3 * 10240,
                            (size_t)ngdn * 3 * 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          if (ckpt_iraw)
            k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
                d_iproj_snap, d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
                d_iraw, q + 1, q + d + 1, 9);
          const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
          for (int p = q + d + 2; p <= q + g + 1; p++)
            CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                              ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          pos = q + d + 2;
          ple_ringpos = pos % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          restore_dR_row(d);
          cur = stopped_token >= 0 ? stopped_token : next;
        }
        // Committed-token history: rebuild explicitly so it is right whether
        // or not ple_on made the verify append (as spec_loop_chain).
        chist.resize(chist_sz);
        chist.insert(chist.end(), vt.begin(), vt.begin() + d + 1);
        // MTP self-repair over committed positions q+1..q+d.
        CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (d > 0) {
          std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
          mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
        } else {
          mtp_pos = q + 1;
          CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
        }
      } else {
        // ---- no draft, no MTP: plain serial sampling step ----
        ++fallback;
        forward(cur);
        if (!ple_on) chist.push_back(cur);
        ple_ringpos = pos % 9;  // graph advances the device scalar
        get_logits(p_row);
        float lp_s = 0.0f;
        int nx = sampler.sample(p_row, &lp_s);
        if (emit && !emit(nx, lp_s)) break;
        out.push_back(nx);
        cur = nx;
        continue;  // serial steps are not speculation rounds
      }
      ++last_spec_rounds;
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = (int)out.size();
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    last_spec_commit = (int)out.size();
    last_spec_rollbacks = ng_roll + mtp_roll;
    fprintf(stderr,
            "%s chain-sample: %d tokens in %.2f s = %.1f tok/s | rounds=%d "
            "(ngram=%d mtp=%d serial=%d) commit/round=%.2f accepted=%d "
            "proposed=%d rollbacks=%d\n",
            log_ts(), (int)out.size(), sec, sec > 0 ? out.size() / sec : 0.0,
            last_spec_rounds, ng_rounds, mtp_rounds, fallback,
            last_spec_rounds ? (double)out.size() / last_spec_rounds : 0.0,
            accepted, last_spec_proposed, last_spec_rollbacks);
    if (mtp_rounds) {
      fprintf(stderr, "chain-sample mtp depth acc:");
      for (int j = 0; j < 8 && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1,
                (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr, "\n");
      gctl.report("chain-sample");
    }
    return out;
  }

  // --- decode graphs: G1 = token upload + embed + L0; G2 = [PLE] + L1..end --
  hipGraphExec_t g1_exec = nullptr, g2_exec = nullptr;
  bool graphs_ok = false;
  void build_graphs() {
    if (dbg || perf || getenv("QWENOX_NOGRAPH")) return;
    // pre-populate lazy caches (hipMalloc/hipMemcpy are illegal during capture)
    for (int l = 0; l < g_cfg.layers; l++) {
      char p[128];
      snprintf(p, sizeof p, "layers.%d.attn_hyper_connection", l);
      ensure_hcnorm(p);
      snprintf(p, sizeof p, "layers.%d.mlp_hyper_connection", l);
      ensure_hcnorm(p);
      if (is_qsa(l)) {
        snprintf(p, sizeof p, "layers.%d.self_attn.q_norm.weight", l);
        ensure_norm256(p);
        snprintf(p, sizeof p, "layers.%d.self_attn.k_norm.weight", l);
        ensure_norm256(p);
      }
    }
    ensure_hcnorm("hyper_connection_mixer");
    hipStream_t cap;
    CK(hipStreamCreate(&cap));
    g_str = cap;
    g_kprof_mute = true;  // hipEventRecord 进捕获流会把 event 录进图里
    hipGraph_t g1, g2;
    CK(hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal));
    CK(hipMemcpyAsync(d_token, p_token, 4, hipMemcpyHostToDevice, cap));
    enq_embed();
    enq_layer(0);
    CK(hipStreamEndCapture(cap, &g1));
    CK(hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal));
    if (ple_on) ple_gpu();
    for (int l = 1; l < g_cfg.layers; l++) enq_layer(l);
    enq_final();
    k_step_incr<<<1, 1, 0, cap>>>(d_pos, d_ringpos);
    CK(hipStreamEndCapture(cap, &g2));
    g_str = nullptr;
    g_kprof_mute = false;
    CK(hipGraphInstantiate(&g1_exec, g1, nullptr, nullptr, 0));
    CK(hipGraphInstantiate(&g2_exec, g2, nullptr, nullptr, 0));
    CK(hipStreamDestroy(cap));
    graphs_ok = true;
    fprintf(stderr, "decode graphs captured (ple=%d)\n", (int)ple_on);
  }

  // ==================== P3-B: spec-verify segment graphs ====================
  // One graph set per batch size P (= gamma+1): seg 0 = pinned token upload +
  // embed + L0 + PLE device phase + L1..L2; then per QSA layer lq an eager
  // indexer (qsa_idx_b: base-dependent host grids) followed by a captured
  // segment with lq's layer body and the dense layers up to the next QSA; the
  // last segment also carries the d_R handoff, the HC mixer read, the batched
  // lm_head and argmax_rows. Every base-dependent write inside a captured
  // segment reads *d_vbase (P3-A/P3-B device base), so a replay with a new
  // base is bit-identical to the eager path: same kernels, same inputs, same
  // order — only the base scalar moved to device memory. The PLE gather, KV
  // page backing, indexer and the pos/argmax readback stay eager per round.
  static constexpr int kVGMaxP = 16;   // graphs for P in [2, 16]
  static constexpr int kVGMaxSeg = 16; // 1 + layers/4 QSA segments
  struct VGraph {
    hipGraphExec_t seg[kVGMaxSeg] = {};
    int nseg = 0;  // 0 = not built, -1 = capture failed (permanent fallback)
    // Host-side HC inject-partials pipeline state as the captured sequence
    // left it after each segment (graph replay runs no host code, so
    // vgraph_run restores this after each launch; otherwise the eager
    // inter-segment norm would misjudge have_p and recompute the inject
    // GEMM — same math, different summation order, different bits).
    int bnd_injp_cur[kVGMaxSeg] = {};
    std::string bnd_injp_for[kVGMaxSeg];
  };
  VGraph vg_[kVGMaxP - 1];
  int vg_warm_[kVGMaxP - 1] = {};  // eager rounds per P (descriptor warmup)
  bool vg_capturing_ = false;      // suppresses capture-illegal side ops
  int* p_vgtok = nullptr;          // pinned token staging for seg 0's H2D node
  int* p_vgbase = nullptr;         // pinned base staging for the d_vbase write

  bool vgraph_ok(int P, int base) {
    static const bool off = env_zero("QWENOX_VGRAPH");
    if (off || !vbase_on()) return false;
    if ((dbg && !getenv("QWENOX_VGRAPH_DBG")) || perf || g_kprof_on)
      return false;
    if (P < 2 || P > kVGMaxP || base < 2051 || base + P > maxctx) return false;
    if (mrope_active_ || cut_abs >= 0 || qsa_dense || qsa_loop || qsa_wmma6)
      return false;
    if (ple_on && ple_loop) return false;
    if (ngram_verify_active || ws_lent_ || prog_total > 0) return false;
    // graphs bake this slot's pointers (d_ptab, d_R, ...): single-slot only
    if (nslots > 1) return false;
    if (injp_cur >= 0 || !hcw_site_.empty()) return false;
    if (getenv("QWENOX_VCT_CHECK") || getenv("QWENOX_ATTN_HASH") ||
        getenv("QWENOX_VCT_TAILZERO"))
      return false;
    // D1b mid-chunk GPU lending is not part of the graphs (production has
    // moe_lt on, which already makes the yield a no-op).
    if (ws_ok_ && borrow_query_fn && !moe_lt && P > conc_rmax_) return false;
    return true;
  }

  // Capture the segments for P (kernels do not execute while capturing; the
  // interleaved eager indexer runs harmlessly on scratch). kv_prepare runs
  // first so the eager indexer sees a backed page table. On any capture error
  // the set is discarded and the caller falls back to the eager round.
  bool vgraph_capture(int P, int base) {
    VGraph& G = vg_[P - 2];
    if (!p_vgtok) {
      if (hipHostMalloc((void**)&p_vgtok, kVGMaxP * 4, hipHostMallocDefault) !=
              hipSuccess ||
          hipHostMalloc((void**)&p_vgbase, 4, hipHostMallocDefault) != hipSuccess)
        return false;
    }
    kv_prepare(base, base + P);
    // The eager boundary ops below (norm + indexer) execute for real during
    // capture, on stale inputs. Give them the same device base the replay
    // will use so their pooled-block/ring write footprint matches the
    // replay's exactly (the replay then rewrites all of it)...
    set_vbase(base);
    // ...except the raw indexer ring: its 4 rows are fully overwritten by
    // the capture-phase indexer, while the replay's pooling reads a history
    // row (base%4 != 0 straddle) written before this round. Shield it.
    static float* d_iraw_sav = nullptr;
    static size_t iraw_sav_sz = 0;
    const size_t iraw_sz = (size_t)(g_cfg.layers / 4) * 4 * 128 * 4;
    if (d_iraw) {
      if (iraw_sav_sz < iraw_sz) {
        if (d_iraw_sav) (void)hipFree(d_iraw_sav);
        if (hipMalloc(&d_iraw_sav, iraw_sz) != hipSuccess) return false;
        iraw_sav_sz = iraw_sz;
      }
      CK(hipMemcpyAsync(d_iraw_sav, d_iraw, iraw_sz, hipMemcpyDeviceToDevice,
                        g_str));
    }
    hipStream_t cap = nullptr;
    if (hipStreamCreate(&cap) != hipSuccess) return false;
    bool ok = true;
    hipStream_t save = g_str;
    g_str = cap;
    g_kprof_mute = true;
    vg_capturing_ = true;
    auto end_seg = [&]() {
      hipGraph_t gr = nullptr;
      if (hipStreamEndCapture(cap, &gr) != hipSuccess || !gr) return false;
      hipGraphExec_t ex = nullptr;
      const bool iok =
          hipGraphInstantiate(&ex, gr, nullptr, nullptr, 0) == hipSuccess;
      (void)hipGraphDestroy(gr);
      if (!iok) return false;
      G.seg[G.nseg++] = ex;
      return true;
    };
    if (hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal) != hipSuccess)
      ok = false;
    if (ok) {  // seg 0: token upload + embed + L0 + PLE + L1..first QSA-1
      // QWENOX_VGRAPH_NOPLE / QWENOX_VGRAPH_NOGDN / QWENOX_VGRAPH_NOEMB (debug
      // bisect): run that part eagerly in vgraph_run instead of capturing it.
      static const bool s0_ple = !getenv("QWENOX_VGRAPH_NOPLE");
      static const bool s0_gdn = !getenv("QWENOX_VGRAPH_NOGDN");
      static const bool s0_emb = !getenv("QWENOX_VGRAPH_NOEMB");
      if (s0_emb)
        enq_embed_body(p_vgtok, P);
      else
        CK(hipMemcpyAsync(d_tokarr, p_vgtok, P * 4, hipMemcpyHostToDevice,
                          g_str));
      if (s0_gdn) enq_layer_b(0, P, false, true, base, d_vbase);
      if (ple_on && s0_ple) enq_ple_device_b(P, base, 0);
      if (s0_gdn)
        for (int l = 1; l < 3 && l < g_cfg.layers; l++) {
          const bool fuse_prev = gr_scatter_norm && l > 1;
          const bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
          enq_layer_b(l, P, fuse_prev, flush_mlp, base, d_vbase);
        }
      ok = end_seg();
      if (ok) {  // host-side injp pipeline state this segment's capture left
        G.bnd_injp_cur[G.nseg - 1] = injp_cur;
        G.bnd_injp_for[G.nseg - 1] = injp_for;
      }
    }
    // QWENOX_VGRAPH_SEGS=k (debug bisect): capture only the first k segments;
    // vgraph_run finishes the remaining layers eagerly (host base).
    static const int seg_limit =
        getenv("QWENOX_VGRAPH_SEGS") ? atoi(getenv("QWENOX_VGRAPH_SEGS")) : 999;
    for (int lq = 3; ok && lq < g_cfg.layers && G.nseg < seg_limit; lq += 4) {
      // The indexer reads d_xb, written by the layer's attention norm: run
      // the norm eagerly too, ahead of the indexer, outside the graph.
      enq_layer_norm_b(lq, P, gr_scatter_norm && lq > 1);
      qsa_idx_b(lq, lq / 4, P, base);
      if (hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal) != hipSuccess) {
        ok = false;
        break;
      }
      const int lend = std::min(lq + 4, (int)g_cfg.layers);
      for (int l = lq; l < lend; l++) {
        const bool fuse_prev = gr_scatter_norm && l > 1;
        const bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
        enq_layer_b(l, P, fuse_prev, flush_mlp, base, d_vbase, l == lq);
      }
      if (lend == g_cfg.layers) {  // final segment: handoff + logits + argmax
        if (gr_bf16)
          k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, cap>>>(
              (const uint16_t*)d_Rb + (size_t)(P - 1) * 10240, d_R, 10240);
        else
          CK(hipMemcpyAsync(d_R, d_Rb + (size_t)(P - 1) * 10240, 10240 * 4,
                            hipMemcpyDeviceToDevice, cap));
        batch_logits_device(P);
        argmax_rows(d_logits_b, g_cfg.vocab, P, d_amx_cand, d_argmax_arr, cap);
      }
      ok = end_seg();
      if (ok) {  // host-side injp pipeline state this segment's capture left
        G.bnd_injp_cur[G.nseg - 1] = injp_cur;
        G.bnd_injp_for[G.nseg - 1] = injp_for;
      }
    }
    g_str = save;
    g_kprof_mute = false;
    vg_capturing_ = false;
    // capture-phase eager boundary writes landed on cap; order the restore
    (void)hipStreamSynchronize(cap);
    (void)hipStreamDestroy(cap);
    if (d_iraw && iraw_sav_sz)
      CK(hipMemcpyAsync(d_iraw, d_iraw_sav, iraw_sav_sz, hipMemcpyDeviceToDevice,
                        g_str));
    if (!ok || G.nseg == 0) {
      for (int s = 0; s < G.nseg; s++) (void)hipGraphExecDestroy(G.seg[s]);
      G.nseg = 0;
      (void)hipGetLastError();  // clear any capture error
      return false;
    }
    return true;
  }

  // One spec-verify round on the graphs: eager head (page backing, device
  // base, PLE gather, token staging), segment replays interleaved with the
  // eager per-QSA-layer indexer, eager tail (position mirrors + blocking
  // argmax readback). Matches prefill_batch(final=false) + batch_argmax.
  void vgraph_run(const std::vector<int>& tokens, int base, int* vout) {
    const int P = (int)tokens.size();
    VGraph& G = vg_[P - 2];
    if (d_iraw) vg_dump_buf("irH0", 2, d_iraw, 512 * 4);  // P3-B bisect
    kv_prepare(base, base + P);
    *p_vgbase = base;
    CK(hipMemcpyAsync(d_vbase, p_vgbase, 4, hipMemcpyHostToDevice, g_str));
    xf16_src = nullptr;  // a new batch invalidates the bf16 conversion cache
    if (ple_on) {
      if (ple_fl.valid) {  // stale flight for another chunk: finish and drop
        ple_join();
        ple_fl.valid = false;
      }
      const int64_t pre0 = chist.empty() ? pad_id : chist.back();
      const int64_t pre1 = chist.size() < 2 ? pad_id : chist[chist.size() - 2];
      // fixed staging buffer 0: seg 0's dequant reads ple_eb_dev
      ple_spawn(std::vector<int>(tokens), base, pre0, pre1, (uint8_t*)ple_eb, 0);
      for (int t : tokens) chist.push_back(t);
      ple_join();
      ple_fl.valid = false;
      // mirror the host-side buffer flip enq_ple_device_b does outside
      // capture: replay consumed staging buffer 0, the next gather writes 1
      ple_par = 1;
    }
    memcpy(p_vgtok, tokens.data(), P * 4);
    if (d_iraw) vg_dump_buf("irH1", 2, d_iraw, 512 * 4);  // P3-B bisect
    CK(hipGraphLaunch(G.seg[0], g_str));
    // Graph replay runs no host code: restore the HC inject-partials pipeline
    // state the capture left after this segment, so the eager code below
    // branches exactly as the fully-eager reference would.
    injp_cur = G.bnd_injp_cur[0];
    injp_for = G.bnd_injp_for[0];
    if (d_iraw) vg_dump_buf("irH2", 2, d_iraw, 512 * 4);  // P3-B bisect
    if (getenv("QWENOX_VGDUMP")) {
      CK(hipStreamSynchronize(g_str));
      int tk[16] = {};
      int nd = P < 16 ? P : 16;
      CK(hipMemcpy(tk, d_tokarr, (size_t)nd * 4, hipMemcpyDeviceToHost));
      fprintf(stderr, "[vgr] P=%d base=%d tok=%d,%d,%d,%d\n", P, base, tk[0],
              P > 1 ? tk[1] : -1, P > 2 ? tk[2] : -1, P > 3 ? tk[3] : -1);
      vg_dump_row("vgs", P, base);
    }
    {  // eager fallback for parts excluded from seg 0 (bisect envs)
      static const bool s0_ple = !getenv("QWENOX_VGRAPH_NOPLE");
      static const bool s0_gdn = !getenv("QWENOX_VGRAPH_NOGDN");
      static const bool s0_emb = !getenv("QWENOX_VGRAPH_NOEMB");
      if (!s0_emb) enq_embed_body(tokens.data(), P);  // H2D repeats, same value
      if (!s0_gdn) enq_layer_b(0, P, false, true, base);
      if (ple_on && !s0_ple) enq_ple_device_b(P, base, 0);
      if (!s0_gdn)
        for (int l = 1; l < 3 && l < g_cfg.layers; l++) {
          const bool fuse_prev = gr_scatter_norm && l > 1;
          const bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
          enq_layer_b(l, P, fuse_prev, flush_mlp, base);
        }
    }
    int s = 1, lq = 3;
    for (; lq < g_cfg.layers && s < G.nseg; lq += 4) {
      enq_layer_norm_b(lq, P, gr_scatter_norm && lq > 1);
      qsa_idx_b(lq, lq / 4, P, base);
      CK(hipGraphLaunch(G.seg[s], g_str));
      injp_cur = G.bnd_injp_cur[s];  // see the restore after seg[0]
      injp_for = G.bnd_injp_for[s];
      s++;
      if (getenv("QWENOX_VGDUMP")) {
        char tag[8];
        snprintf(tag, sizeof tag, "vgS%d", s - 1);
        vg_dump_row(tag, P, base);
      }
    }
    if (lq < g_cfg.layers) {  // QWENOX_VGRAPH_SEGS bisect: eager remainder
      for (int l = lq; l < g_cfg.layers; l++) {
        if (is_qsa(l)) {
          enq_layer_norm_b(l, P, gr_scatter_norm && l > 1);
          qsa_idx_b(l, l / 4, P, base);
        }
        const bool fuse_prev = gr_scatter_norm && l > 1;
        const bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
        enq_layer_b(l, P, fuse_prev, flush_mlp, base, nullptr, is_qsa(l));
        {
          char tag[8];
          snprintf(tag, sizeof tag, "rL%d", l);
          vg_dump_row(tag, P, base);
        }
      }
      if (gr_bf16)
        k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
            (const uint16_t*)d_Rb + (size_t)(P - 1) * 10240, d_R, 10240);
      else
        CK(hipMemcpyAsync(d_R, d_Rb + (size_t)(P - 1) * 10240, 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
      batch_logits_device(P);
      argmax_rows(d_logits_b, g_cfg.vocab, P, d_amx_cand, d_argmax_arr, g_str);
    }
    pos = base + P;
    ple_ringpos = (base + P) % 9;
    CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpy(vout, d_argmax_arr, (size_t)P * 4, hipMemcpyDeviceToHost));
  }

  // verify_graph_round_try: graph fast path for MTP verify; false -> caller
  // runs the legacy eager sequence. The first qualifying round per P stays
  // eager (warms the P-shaped GEMM/LT descriptors), the second captures.
  bool verify_graph_round_try(const std::vector<int>& tokens, int base,
                              int* vout) {
    const int P = (int)tokens.size();
    if (!vgraph_ok(P, base)) return false;
    VGraph& G = vg_[P - 2];
    if (G.nseg < 0) return false;
    if (!G.nseg) {
      if (vg_warm_[P - 2]++ < 1) return false;
      if (!vgraph_capture(P, base)) {
        G.nseg = -1;
        fprintf(stderr, "verify graphs: capture failed for P=%d, staying eager\n",
                P);
        return false;
      }
      fprintf(stderr, "verify graphs captured (P=%d, %d segments)\n", P, G.nseg);
      // QWENOX_VGRAPH_NOLAUNCH (debug): capture once, then stay eager forever —
      // isolates capture-time side effects from replay effects.
      if (getenv("QWENOX_VGRAPH_NOLAUNCH")) {
        for (int s = 0; s < G.nseg; s++) (void)hipGraphExecDestroy(G.seg[s]);
        G.nseg = -1;
        return false;
      }
    }
    vgraph_run(tokens, base, vout);
    return true;
  }

  // ---- Sequence slots (concurrent serve, QWENOX_PARALLEL) ----
  // Everything one sequence carries from step to step: host mirrors, GDN /
  // conv / PLE ring / indexer ring state, the page table (the KV pages live
  // in the shared pool, MTP layer included), the MTP indexer ring and
  // position, per-slot decode graphs (they bake these
  // pointers) and the request-scoped flags/stats. Weights, prefill/decode
  // scratch, logits, verify checkpoints and the KV pool are shared: slots
  // time-slice the GPU and hand it over only at yield_point(), where none of
  // the shared buffers holds live data. bind(s) swaps slot s into the model
  // members, so every kernel path runs unchanged on the bound slot.
  // mrope_gen_ is NOT per slot: it is a global generation for the ropecs
  // cache key (bind also drops that cache).
#define QWENOX_SLOT_FIELDS(X)                                                    \
  X(pos) X(ple_ringpos) X(chist) X(ngram) X(mtp_pos) X(kv_seq) X(d_ptab)      \
  X(mrope_active_) X(rope_delta_) X(mrope_pos_) X(mrope_grids_) X(d_rdelta)   \
  X(img_inj) X(img_hash_) X(d_img_emb) X(img_emb_rows) X(img_emb_cap)         \
  X(mtp_tap_capture) X(spec_snap) X(ngram_verify_active)                      \
  X(cut_abs) X(cut_r_) X(cut_hit) X(cut_mtp_hit)                              \
  X(prog_total) X(prog_base) X(prog_last_est) X(prog_req) X(prog_t0)          \
  X(prog_last) X(last_spec_rounds) X(last_spec_commit) X(last_spec_rollbacks) \
  X(last_spec_proposed) X(live_spec_rounds) X(live_spec_commits)              \
  X(live_spec_proposed) X(g1_exec) X(g2_exec)                                 \
  X(d_S) X(d_convst) X(d_ring) X(d_iraw) X(d_R) X(d_pos) X(d_ringpos)         \
  X(d_miraw) X(d_mpos)
  struct SeqState {
#define QWENOX_SLOT_DECL(f) decltype(GpuModel::f) f{};
    QWENOX_SLOT_FIELDS(QWENOX_SLOT_DECL)
#undef QWENOX_SLOT_DECL
  };
  // Invariant: the bound slot's state is in the members and slots_[bound_]
  // holds a placeholder; every other slots_[i] holds slot i's state.
  std::vector<SeqState> slots_;
  int nslots = 1, bound_ = 0;
  void swap_slot(SeqState& s) {
    using std::swap;
#define QWENOX_SLOT_SWAP(f) swap(f, s.f);
    QWENOX_SLOT_FIELDS(QWENOX_SLOT_SWAP)
#undef QWENOX_SLOT_SWAP
  }
  void bind(int s) {
    if (s == bound_) return;
    CK(hipDeviceSynchronize());  // nothing of the outgoing slot still in flight
    swap_slot(slots_[(size_t)bound_]);  // members <- placeholder
    swap_slot(slots_[(size_t)s]);       // members <- slot s
    bound_ = s;
    ropecs_base_ = -1;   // rope cos/sin table belongs to the previous slot
    xf16_src = nullptr;  // so does the bf16 conversion cache
  }

  // ---- D1b borrow workspace ----------------------------------------------
  // Second, small copy of every maxbatch_cap-sized row buffer a borrower can
  // reach: spec verify runs the full prefill_chunk path at P <= conc_rmax_
  // and mtp_step touches row 0 of d_mRb/d_xbf16/d_eb/d_xb/d_Gb/d_t320b/
  // d_qnb/d_Unb/d_convoutb/d_tokarr; serial decode stays on the P=1 buffers
  // and its graphs bake only those (never BP-sized ones). Buffers provably
  // dead at a layer boundary stay shared: d_wbf16, d_gdn_ws, d_gdn_split_ws,
  // d_logits_b, d_eoff, d_gg_uq, d_iscores tiles are row-windowed... (see
  // HANDOFF-CONCURRENCY §5.2 audit), d_posarr/d_ringarr (constant).
  // d_moetiles/d_moentiles/moe_routing/moe_lt/qsa_union paths are never taken
  // at P <= conc_rmax_; conc_borrow_yield refuses to lend when they are on.
#define QWENOX_WS_FIELDS(X)                                                     \
  X(d_Rb) X(d_Rhatb) X(d_Rhatbf16) X(d_hcsc) X(d_xb) X(d_yb) X(d_Gb)          \
  X(d_t320b)                                                                  \
  X(d_w4b) X(d_qkvb) X(d_zb) X(d_g48b) X(d_beta48b) X(d_qgb) X(d_qsb)        \
  X(d_gsb) X(d_kbb) X(d_vbb) X(d_ropecs) X(d_attnb) X(d_routerb)             \
  X(d_topidsb) X(d_topwb) X(d_slot_order) X(d_guvb) X(d_hidb) X(d_moeaccb)   \
  X(d_eyb) X(d_sgb) X(d_eb) X(d_xbf16) X(d_xbf16b) X(d_tokidx) X(d_pweight)  \
  X(d_pairids) X(d_convb) X(d_keysb) X(d_plevb) X(d_qnb) X(d_Ub) X(d_Unb)    \
  X(d_convoutb) X(d_tokarr) X(d_img_inj) X(d_iproj) X(d_iq) X(d_iqh) X(d_iselected)   \
  X(d_iscores) X(d_mRb) X(d_mRhatb) X(d_mtap)
  struct WsPtrs {
#define QWENOX_WS_DECL(f) decltype(GpuModel::f) f = nullptr;
    QWENOX_WS_FIELDS(QWENOX_WS_DECL)
#undef QWENOX_WS_DECL
    float *d_injp0 = nullptr, *d_injp1 = nullptr;
    float* ple_eb = nullptr;
    uint8_t* ple_eb_dev = nullptr;
  };
  WsPtrs ws_small_, ws_saved_;
  bool ws_lent_ = false, ws_ok_ = false;
  hipEvent_t conc_ev_a_ = nullptr, conc_ev_b_ = nullptr;
  long conc_lends_ = 0;  // D1b diagnostics: total lends since init

  // Mirror of the constructor's BP-sized block at R rows. Each field is
  // allocated only when the big counterpart exists (same feature flags), so
  // "small == nullptr" always implies "big == nullptr" and the lend/takeback
  // guards below never drop a live buffer.
  void ws_alloc_small(int R) {
    if (ws_ok_ || R < 1) return;
    const size_t B = (size_t)R;
#define QWENOX_WS_DUP(p, bytes) \
    if (p) DALLOC(&ws_small_.p, (bytes))
    QWENOX_WS_DUP(d_Rb, B * 10240 * 4);
    QWENOX_WS_DUP(d_Rhatb, B * 10240 * 4);
    QWENOX_WS_DUP(d_Rhatbf16, B * 10240 * 2);
    QWENOX_WS_DUP(d_hcsc, B * 4 * 4);
    QWENOX_WS_DUP(d_xb, B * 2560 * 4);
    QWENOX_WS_DUP(d_yb, B * 2560 * 4);
    QWENOX_WS_DUP(d_Gb, B * 10240 * 4);
    QWENOX_WS_DUP(d_t320b, B * 320 * 4);
    QWENOX_WS_DUP(d_w4b, B * 4 * 4);
    QWENOX_WS_DUP(d_qkvb, B * 10240 * 4);
    QWENOX_WS_DUP(d_zb, B * 6144 * 4);
    QWENOX_WS_DUP(d_g48b, B * 48 * 4);
    QWENOX_WS_DUP(d_beta48b, B * 48 * 4);
    QWENOX_WS_DUP(d_qgb, B * 12288 * 4);
    QWENOX_WS_DUP(d_qsb, B * 6144 * 4);
    QWENOX_WS_DUP(d_gsb, B * 6144 * 4);
    QWENOX_WS_DUP(d_kbb, B * 512 * 4);
    QWENOX_WS_DUP(d_vbb, B * 512 * 4);
    QWENOX_WS_DUP(d_ropecs, B * 32 * sizeof(float2));
    QWENOX_WS_DUP(d_attnb, B * 6144 * 4);
    QWENOX_WS_DUP(d_routerb, B * 512 * 4);
    QWENOX_WS_DUP(d_topidsb, B * 16 * 4);
    QWENOX_WS_DUP(d_topwb, B * 16 * 4);
    QWENOX_WS_DUP(d_slot_order, 256 * 4);
    QWENOX_WS_DUP(d_guvb, B * g_cfg.topk * (moe_deterministic ? 2560 : 1280) * 4);
    QWENOX_WS_DUP(d_hidb, B * g_cfg.topk * 640 * 4);
    QWENOX_WS_DUP(d_moeaccb, B * 2560 * 4);
    QWENOX_WS_DUP(d_eyb, B * 2560 * 4);
    QWENOX_WS_DUP(d_sgb, B * 4);
    QWENOX_WS_DUP(d_eb, B * 2560 * 4);
    QWENOX_WS_DUP(d_xbf16, B * 10240 * 2);
    QWENOX_WS_DUP(d_xbf16b, B * 6144 * 2);
    QWENOX_WS_DUP(d_tokidx, B * g_cfg.topk * 4);
    QWENOX_WS_DUP(d_pweight, B * g_cfg.topk * 4);
    QWENOX_WS_DUP(d_pairids, B * g_cfg.topk * 4);
    QWENOX_WS_DUP(d_convb, B * 10240 * 4);
    QWENOX_WS_DUP(d_keysb, B * 10240 * 4);
    QWENOX_WS_DUP(d_plevb, B * 2560 * 4);
    QWENOX_WS_DUP(d_qnb, B * 10240 * 4);
    QWENOX_WS_DUP(d_Ub, B * 10240 * 4);
    QWENOX_WS_DUP(d_Unb, B * 10240 * 4);
    QWENOX_WS_DUP(d_convoutb, B * 10240 * 4);
    QWENOX_WS_DUP(d_tokarr, B * 4);
    QWENOX_WS_DUP(d_img_inj, B * sizeof(int2));
    if (d_iproj) {  // !qsa_dense indexer block
      DALLOC(&ws_small_.d_iproj, B * 640 * 4);
      DALLOC(&ws_small_.d_iq, B * 512 * 4);
      if (d_iqh) DALLOC(&ws_small_.d_iqh, B * 512 * 2);
      DALLOC(&ws_small_.d_iselected, B * INDEX_BUDGET * 4);
      DALLOC(&ws_small_.d_iscores,
             (size_t)std::min(R, INDEX_BATCH) * 4 * index_blocks * 4);
    }
    QWENOX_WS_DUP(d_mRb, B * 10240 * 4);
    QWENOX_WS_DUP(d_mRhatb, B * 10240 * 4);
    QWENOX_WS_DUP(d_mtap, B * 10240 * 4);
#undef QWENOX_WS_DUP
    if (d_injp[0]) {
      DALLOC(&ws_small_.d_injp0, B * 16 * 4);
      DALLOC(&ws_small_.d_injp1, B * 16 * 4);
    }
    if (ple_eb) {
      CK(hipHostMalloc((void**)&ws_small_.ple_eb, B * 2560 * 4,
                       hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ws_small_.ple_eb_dev,
                                 ws_small_.ple_eb, 0));
    }
    CK(hipEventCreate(&conc_ev_a_));  // timing events: quantum clock
    CK(hipEventCreate(&conc_ev_b_));
    ws_ok_ = true;
  }

  void ws_lend() {  // members -> small set; big pointers saved in ws_saved_
#define QWENOX_WS_SAVE(f) \
    ws_saved_.f = f;    \
    if (ws_small_.f) f = ws_small_.f;
    QWENOX_WS_FIELDS(QWENOX_WS_SAVE)
#undef QWENOX_WS_SAVE
    ws_saved_.d_injp0 = d_injp[0];
    ws_saved_.d_injp1 = d_injp[1];
    if (ws_small_.d_injp0) {
      d_injp[0] = ws_small_.d_injp0;
      d_injp[1] = ws_small_.d_injp1;
    }
    ws_saved_.ple_eb = ple_eb;
    ws_saved_.ple_eb_dev = ple_eb_dev;
    if (ws_small_.ple_eb) {
      ple_eb = ws_small_.ple_eb;
      ple_eb_dev = ws_small_.ple_eb_dev;
    }
    ws_lent_ = true;
  }
  void ws_takeback() {  // members -> big set (exact inverse of ws_lend)
#define QWENOX_WS_REST(f) \
    if (ws_small_.f) f = ws_saved_.f;
    QWENOX_WS_FIELDS(QWENOX_WS_REST)
#undef QWENOX_WS_REST
    if (ws_small_.d_injp0) {
      d_injp[0] = ws_saved_.d_injp0;
      d_injp[1] = ws_saved_.d_injp1;
    }
    if (ws_small_.ple_eb) {
      ple_eb = ws_saved_.ple_eb;
      ple_eb_dev = ws_saved_.ple_eb_dev;
    }
    ws_lent_ = false;
  }

  // Layer-boundary yield inside a big prefill chunk. Nobody queued: one
  // atomic read. Contended: drain the layers enqueued so far (event sync),
  // and if the GPU-busy quantum elapsed, lend the small workspace and block
  // in borrow_fn until the turnstile comes back. The chunk's live state
  // (residual stream in the big set, per-slot fields, KV pages) is untouched
  // by the borrower; chunk-level member state is restored explicitly.
  void conc_borrow_yield(int base, int P, int layer) {
    if (!ws_ok_ || !borrow_query_fn || ws_lent_ || P <= conc_rmax_) return;
    // moe_lt: unduplicated scratch live across layers. qsa_union's
    // d_ublk/d_umask/d_ucount are written and consumed inside one qsa_flash_b
    // call -> dead at this layer boundary, so a borrower may reuse them
    // (aggr 10-05: union is default-on; refusing here would disable D1b).
    if (moe_lt) return;
    if (!borrow_query_fn()) return;
    CK(hipEventRecord(conc_ev_b_, g_str));
    CK(hipEventSynchronize(conc_ev_b_));  // GPU drained through this layer
    float gpu_ms = 0;
    CK(hipEventElapsedTime(&gpu_ms, conc_ev_a_, conc_ev_b_));
    if (gpu_ms < conc_quantum_ms_) return;
    fprintf(stderr, "%s [conc] prefill lends GPU at L%d (base=%d, %.0f ms since last)\n",
            log_ts(), layer, base, (double)gpu_ms);
    conc_lends_++;
    // HC inject partials pipeline (injp_cur/injp_for -> big d_injp[]) is live
    // across this layer boundary: park it, hand the borrower an empty
    // pipeline, restore on return. The partial sums themselves sit in the
    // big d_injp buffers, which the borrower never touches (small set).
    const int saved_injp_cur = injp_cur;
    const std::string saved_injp_for = injp_for;
    injp_cur = -1;
    injp_for.clear();
    ws_lend();
    try {
      borrow_fn(*this);  // flag up; release; light borrowers run; reacquire; bind
    } catch (...) {
      ws_takeback();  // aborted mid-chunk: never leave members on the small set
      injp_cur = saved_injp_cur;
      injp_for = saved_injp_for;
      throw;
    }
    ws_takeback();
    injp_cur = saved_injp_cur;
    injp_for = saved_injp_for;
    set_vbase(base);    // the borrower's set_vbase rotated the ring
    ropecs_base_ = -1;  // rope table and bf16 cache below are the borrower's
    xf16_src = nullptr;
    CK(hipEventRecord(conc_ev_a_, g_str));  // restart the quantum clock
  }

  // Allocate slots 1..n-1 (slot 0 is the constructor's state) and capture
  // their decode graphs. Paged KV only: all slots map pages from one pool.
  void slots_init(int n) {
    if (n <= 1) return;
    if (!kv_paged) {
      fprintf(stderr, "qwenox: QWENOX_PARALLEL=%d needs paged KV (QWENOX_KV_PAGED); "
                      "running 1 slot\n", n);
      return;
    }
    const size_t rb = (size_t)g_cfg.branches * g_cfg.d;
    const size_t ngdn = g_cfg.layers - g_cfg.layers / 4, nqsa = g_cfg.layers / 4;
    const size_t sS = ngdn * 48 * 128 * 128 * 4, sC = ngdn * 3 * 10240 * 4,
                 sRing = (size_t)9 * 10240 * 4, sIraw = nqsa * 4 * 128 * 4;
    slots_.resize((size_t)n);
    for (int i = 1; i < n; i++) {
      SeqState& t = slots_[(size_t)i];
      DALLOC(&t.d_R, rb * 4);
      DALLOC(&t.d_S, sS);
      DALLOC(&t.d_convst, sC);
      DALLOC(&t.d_ring, sRing);
      DALLOC(&t.d_pos, 4);
      DALLOC(&t.d_ringpos, 4);
      DALLOC(&t.d_rdelta, 4);
      CK(hipMemset(t.d_R, 0, rb * 4));
      CK(hipMemset(t.d_S, 0, sS));
      CK(hipMemset(t.d_convst, 0, sC));
      CK(hipMemset(t.d_ring, 0, sRing));
      CK(hipMemset(t.d_pos, 0, 4));
      CK(hipMemset(t.d_ringpos, 0, 4));
      CK(hipMemset(t.d_rdelta, 0, 4));
      if (d_iraw) {
        DALLOC(&t.d_iraw, sIraw);
        CK(hipMemset(t.d_iraw, 0, sIraw));
      }
      t.kv_seq.init(kv_lpages, kv_pool.guard());
      DALLOC(&t.d_ptab, (size_t)kv_lpages * 4);
      CK(hipMemcpy(t.d_ptab, t.kv_seq.tab.data(), (size_t)kv_lpages * 4,
                   hipMemcpyHostToDevice));
      if (mtp_avail) {
        DALLOC(&t.d_mpos, 4);
        CK(hipMemset(t.d_mpos, 0, 4));
        // the MTP KV / keys live in the shared pool (layer nqsa, B0)
        if (d_miraw) {
          DALLOC(&t.d_miraw, 4 * 128 * 4);
          CK(hipMemset(t.d_miraw, 0, 4 * 128 * 4));
        }
      }
      t.prog_req = -1;
      if (graphs_ok) {  // per-slot decode graphs (they bake the slot pointers)
        bind(i);
        build_graphs();
        bind(0);
      }
    }
    nslots = n;
    if (!env_zero("QWENOX_CONC_PREFILL")) {  // D1b: arm the borrow workspace
      if (const char* v = getenv("QWENOX_CONC_PREFILL_QUANTUM_MS"))
        conc_quantum_ms_ = (float)atof(v);
      if (const char* v = getenv("QWENOX_CONC_RMAX")) conc_rmax_ = atoi(v);
      ws_alloc_small(conc_rmax_);
      if (ws_ok_)
        fprintf(stderr, "qwenox: prefill borrow armed (R<=%d rows, quantum %.0f ms)\n",
                conc_rmax_, (double)conc_quantum_ms_);
    }
    fprintf(stderr, "qwenox: %d sequence slots (shared KV pool %d pages = %d tokens)\n",
            n, kv_pool_pages, kv_pool_pages * KV_PAGE);
  }
  // Pages the slot's live sequence maps (bound slot: the members).
  int slot_pages(int s) const {
    return s == bound_ ? kv_seq.mapped : slots_[(size_t)s].kv_seq.mapped;
  }
  // Drop slot s's sequence (returns its pages to the pool) while `back`
  // stays the slot the caller runs on.
  void slot_reset(int s, int back) {
    bind(s);
    reset_state();
    mtp_tap_capture = spec_snap = ngram_verify_active = false;
    prog_total = 0;
    bind(back);
  }

  // Zero all cross-request state so the next request behaves like a fresh
  // process (the constructor memsets d_S/d_convst; hipMalloc's fresh zero
  // pages cover d_ring — reproduce both here).
  void reset_state() {
    pos = 0;
    ple_ringpos = 0;
    chist.clear();
    xf16_src = nullptr;  // invalidate the prefill bf16 conversion cache
    clear_mrope();       // drop any M-RoPE request state (d_rdelta <- 0)
    int ngdn = g_cfg.layers - g_cfg.layers / 4;
    CK(hipMemsetAsync(d_pos, 0, 4, 0));
    CK(hipMemsetAsync(d_ringpos, 0, 4, 0));
    CK(hipMemsetAsync(d_S, 0, (size_t)ngdn * 48 * 128 * 128 * 4, 0));
    CK(hipMemsetAsync(d_convst, 0, (size_t)ngdn * 3 * 10240 * 4, 0));
    CK(hipMemsetAsync(d_ring, 0, (size_t)9 * 10240 * 4, 0));
    CK(hipDeviceSynchronize());
    kv_trim(0);  // A2: the next sequence maps its pages from scratch
    if (kv_zero_dbg && !kv_paged) kv_zero_unpaged();
  }

  int forward(int token) {
    kv_prepare(pos, pos + 1);  // A2/A3: back (and unshare) row pos, both paths
    if (graphs_ok) {
      *p_token = token;
      auto g0 = std::chrono::steady_clock::now();
      CK(hipGraphLaunch(g1_exec, 0));
      dump_decode_step(pos);  // debug only (QWENOX_MROPE_DUMP); syncs the stream
      double g1_ms;
      if (ple_on) {
        ple_host(token, ple_e);  // host hash+read overlaps G1 on GPU
        g1_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - g0)
                    .count();
      } else {
        g1_ms = 0;
      }
      CK(hipGraphLaunch(g2_exec, 0));
      pos++;
      int id;
      auto sync0 = std::chrono::steady_clock::now();
      CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
      double sync_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync0)
              .count();
      if (getenv("QWENOX_TIME")) {
        t_enq += g1_ms;
        t_sync += sync_ms;
        if (t_n % 16 == 15)
          fprintf(stderr, "TIME: ple_host=%.2f ms/tok g2_sync=%.2f ms/tok\n",
                  t_enq / (t_n + 1), t_sync / (t_n + 1));
      }
      t_n++;
      return id;
    }
    // direct path (debug/perf): upload scalars, enqueue everything
    CK(hipMemcpyAsync(d_token, &token, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
    auto T0 = tick();
    enq_embed();
    tock(0, T0);
    // layer 0 enqueued first so PLE host reads overlap with GPU work
    enq_layer(0);
    dump_decode_step(pos);  // debug only (QWENOX_MROPE_DUMP); syncs the stream
    if (ple_on) {
      auto t1 = tick();
      ple_host(token, ple_e);
      tock(1, t1);
      ple_gpu();
      ple_ringpos = (ple_ringpos + 1) % 9;
    }
    for (int l = 1; l < g_cfg.layers; l++) enq_layer(l);
    auto t4 = tick();
    auto enq0 = std::chrono::steady_clock::now();
    enq_final();
    pos++;
    int id;
    double enq_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - enq0)
            .count();
    auto sync0 = std::chrono::steady_clock::now();
    CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    double sync_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync0)
            .count();
    if (getenv("QWENOX_TIME")) {
      t_enq += enq_ms;
      t_sync += sync_ms;
      if (t_n % 16 == 15)
        fprintf(stderr, "TIME: enq=%.2f ms/tok sync=%.2f ms/tok\n", t_enq / t_n,
                t_sync / t_n);
    }
    tock(4, t4);
    t_n++;
    return id;
  }

  void perf_report() {
    if (!perf || !t_n) return;
    fprintf(stderr,
            "PERF ms/token: embed=%.2f ple=%.2f attn=%.2f moe=%.2f head=%.2f total=%.2f "
            "(%d tokens)\n",
            t_acc[0] / t_n, t_acc[1] / t_n, t_acc[2] / t_n, t_acc[3] / t_n, t_acc[4] / t_n,
            (t_acc[0] + t_acc[1] + t_acc[2] + t_acc[3] + t_acc[4]) / t_n, t_n);
  }

  void get_logits(std::vector<float>& out) {
    out.resize(g_cfg.vocab);
    CK(hipMemcpy(out.data(), d_logits, (size_t)g_cfg.vocab * 4,
                 hipMemcpyDeviceToHost));
  }
};

// --- QSA q/gate split + per-head zc norm: qg (24, 512) -> qs (24,256), gs (24,256)
// Batched: grid.y = token row (qg row stride 24*2*dh, qs/gs row stride 24*dh).
// Rope=true additionally applies the rotary (64 dims, pairs i/i+32) in-kernel,
// bit-exact vs the k_qsa_qsplit + k_rope_b sequence: the normed value is staged
// in LDS (store/reload of a float is exact) and the cos/sin come from the
// k_rope_cs table (bit-identical to k_rope_b's inline fp64 math, but computed
// once per (token, i) instead of once per (head, token, i) — gfx1151 fp64 is
// too weak to pay 24x redundant transcendentals inside this kernel).
// Requires dh == 256, rotary == 64 (all QSA callers).
template <bool Rope>
__global__ void k_qsa_qsplit(const float* __restrict__ qg, const float* __restrict__ qnw,
                             float* __restrict__ qs, float* __restrict__ gs, int dh,
                             float eps, int rope_base, const float2* __restrict__ ropecs) {
  int h = blockIdx.x;
  int t = threadIdx.x;
  qg += (size_t)blockIdx.y * 48 * dh;
  qs += (size_t)blockIdx.y * 24 * dh;
  gs += (size_t)blockIdx.y * 24 * dh;
  __shared__ float sm[256];
  float v = (t < dh) ? qg[h * 2 * dh + t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / dh + eps);
  if (gs && t < dh) gs[h * dh + t] = qg[h * 2 * dh + dh + t];  // null: gate read from qg
  if (!Rope) {
    if (t < dh) qs[h * dh + t] = v * inv * (1.f + qnw[t]);
    return;
  }
  float* q = qs + h * dh;
  if (t < 32) {
    // recompute the pair's normed values inline (same expressions, same bits;
    // the extra qg load is L2-hot) instead of staging them through LDS —
    // saves two __syncthreads and the sm round-trip per block.
    float2 cs = ropecs[(size_t)blockIdx.y * 32 + t];
    if (g_rope_mscale != 1.f) cs.x *= g_rope_mscale, cs.y *= g_rope_mscale;
    float v1 = qg[h * 2 * dh + t + 32];
    float x0 = v * inv * (1.f + qnw[t]);
    float x1 = v1 * inv * (1.f + qnw[t + 32]);
    // contraction pinned to k_rope_b's compiled forms (verified bit-identical
    // at -O2 and -O3): r0 = fma(x0,cs,-(x1*sn)), r1 = fma(x1,cs,x0*sn)
    q[t] = fmaf(x0, cs.x, -(x1 * cs.y));
    q[t + 32] = fmaf(x1, cs.x, x0 * cs.y);
  } else if (t >= 64 && t < dh) {
    q[t] = v * inv * (1.f + qnw[t]);
  }
  (void)rope_base;
}

// --- Fused QSA K prepare, one block per (kv head, token): zc rmsnorm over the
// 256-dim head (same tree, weight per head == k_rmsnorm_zc_grouped with
// ngroups=2 addressing) + rotary (cos/sin from the k_rope_cs table, bit-
// identical to k_rope_b) + optional bf16 store into the compact KV cache
// (same f2bf as k_f32_to_bf16_v4). Bit-exact vs the rmsnorm + k_rope_b
// [+ k_f32_to_bf16_v4] sequence.
// grid = dim3(2, P); kc (when Bf16) already includes the base*512 offset —
// or, with dvb set (P3-B graph capture), kc is the layer base and the kernel
// adds (*dvb)*512 itself.
template <bool Bf16>
__global__ void k_qsa_kprep(const float* __restrict__ kb, const float* __restrict__ knw,
                            float* __restrict__ kout, uint16_t* __restrict__ kc,
                            float eps, int rope_base,
                            const float2* __restrict__ ropecs,
                            const int* __restrict__ dvb) {
  if (Bf16 && dvb) kc += (size_t)(*dvb) * 512;
  int h = blockIdx.x;
  int t = threadIdx.x;
  const float* xg = kb + ((size_t)blockIdx.y * 2 + h) * 256;
  const float* wg = knw;  // both K heads share the single 256-weight (knfix)
  float* og = kout + ((size_t)blockIdx.y * 2 + h) * 256;
  __shared__ float sm[256];
  float v = xg[t];
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / 256 + eps);
  if (t < 32) {
    // inline recompute instead of LDS staging (see k_qsa_qsplit)
    float2 cs = ropecs[(size_t)blockIdx.y * 32 + t];
    if (g_rope_mscale != 1.f) cs.x *= g_rope_mscale, cs.y *= g_rope_mscale;
    float v1 = xg[t + 32];
    float x0 = v * inv * (1.f + wg[t]);
    float x1 = v1 * inv * (1.f + wg[t + 32]);
    // contraction pinned to k_rope_b's compiled forms (see k_qsa_qsplit)
    float r0 = fmaf(x0, cs.x, -(x1 * cs.y)), r1 = fmaf(x1, cs.x, x0 * cs.y);
    og[t] = r0;
    og[t + 32] = r1;
    if (Bf16) {
      uint16_t* dst = kc + (size_t)blockIdx.y * 512 + h * 256;
      dst[t] = f2bf(r0);
      dst[t + 32] = f2bf(r1);
    }
  } else if (t >= 64) {
    float o = v * inv * (1.f + wg[t]);
    og[t] = o;
    if (Bf16) kc[(size_t)blockIdx.y * 512 + h * 256 + t] = f2bf(o);
  }
  (void)rope_base;
}

// --- acc[r] += sigmoid(sg[i/n]) * y[i]; flat over total = P*n
__global__ void k_axpy_sg(float* __restrict__ acc, const float* __restrict__ y,
                          const float* __restrict__ sg, int n, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    float s = 1.f / (1.f + expf(-sg[i / n]));
    acc[i] += s * y[i];
  }
}
