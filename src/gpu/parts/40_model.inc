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
  // not repeat per request). 0 = the GDEC_NGRAM_CHUNK env is still unparsed.
  int ng_cs = 0;
  bool ng_cs_fixed = false;
  bool ng_legacy = false;  // GDEC_NGRAM_CHUNK=0: legacy single-pass verify
  void ng_chunk_init() {
    if (ng_cs) return;
    const char* e = getenv("GDEC_NGRAM_CHUNK");
    const int v = e ? atoi(e) : -1;
    ng_legacy = v == 0;
    ng_cs_fixed = v > 0;
    ng_cs = ng_cs_fixed ? std::max(1, std::min(64, v)) : 16;
  }

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
  // Block ranking is sensitive to projection rounding; retain FP32 here.
  bool index_f32 = getenv("GDEC_INDEX_BF16") == nullptr;
  bool index_fused2 = getenv("GDEC_INDEX_FUSED2") != nullptr;
  bool index_stream_select = getenv("GDEC_INDEX_STREAM_SELECT") != nullptr;
  int* d_iselected = nullptr;
  int index_blocks = 0;
  bool qsa_dense = getenv("GDEC_QSA_DENSE") != nullptr;
  // BF16 KV cache: measured parity at 32K (QSA is latency-bound, not DRAM-bound)
  // and flips the 8192 oracle -> opt-in experiment, default stays FP32.
  bool qsa_kv_fp32 = getenv("GDEC_QSA_KV_BF16") == nullptr;  // default: FP32 cache
  // WMMA sparse prefill attention (k_qsa_wmma): opt-in on top of BF16 KV mode.
  // bf16 P + WMMA accumulation is not bitwise fp32 — FP32 kernels stay default.
  bool qsa_wmma = getenv("GDEC_QSA_WMMA") != nullptr;
  // Keep the established row-major cache for decode, MTP, snapshots, and
  // the default WMMA path; this second cache is a prefill-only opt-in.
  bool qsa_wmma_btv = qsa_wmma && !qsa_kv_fp32 &&
                      getenv("GDEC_QSA_WMMA_BTV") != nullptr;
  // GDEC_QSA_WMMA6=1: dim-split 128-thread rewrite (k_qsa_wmma6); layered on
  // the transposed-V path since it reads B fragments from that cache.
  bool qsa_wmma6 = qsa_wmma_btv && getenv("GDEC_QSA_WMMA6") != nullptr;
  // GDEC_QSA_UNION=1: 4-query 块并集 WMMA prefill（k_qsa_union_merge +
  // k_qsa_wmma_u），叠加在转置 V 路径上；组边缘回退 k_qsa_wmma<true>。
  bool qsa_union = qsa_wmma_btv && getenv("GDEC_QSA_UNION") != nullptr;
  uint16_t* d_ublk = nullptr;  // [ngroups][QSA_UCAP] 并集块号
  uint16_t* d_umask = nullptr; // [ngroups][QSA_UCAP] 4 query x 4 slot 掩码
  int* d_ucount = nullptr;     // [ngroups] 并集块数（含 pad，4 的倍数）
  size_t qsa_vt_blocks = 0;
  bool prefill_fused = getenv("GDEC_PREFILL_UNFUSED") == nullptr;
  bool gr_bf16 = getenv("GDEC_GR_BF16") != nullptr && prefill_fused;
  bool gr_scatter_norm = getenv("GDEC_GR_SCATTER_NORM") != nullptr || gr_bf16;
  bool moe_deterministic = !qsa_dense && getenv("GDEC_MOE_ATOMIC") == nullptr;
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
  FILE* mrope_dump_ = nullptr;               // GDEC_MROPE_DUMP sink
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
  float *d_qkvb, *d_zb, *d_g48b, *d_beta48b;
  float *d_qgb, *d_qsb, *d_gsb, *d_kbb, *d_vbb, *d_attnb;
  float* d_routerb;
  int* d_topidsb;
  float* d_topwb;
  float *d_guvb, *d_hidb, *d_moeaccb, *d_eyb, *d_sgb;
  float* d_eb;              // [maxctx, 2560] PLE embeddings
  float* ple_eb = nullptr;  // pinned staging [maxctx, 2560]
  uint8_t* ple_eb_dev = nullptr;  // device alias of ple_eb (zero-copy fp8 staging)
  uint16_t* d_wbf16;        // dequant workspace (64 MiB)
  uint16_t* d_xbf16;        // [maxctx, 10240] bf16 activations
  uint16_t* d_Rhatbf16 = nullptr;  // survives intermediate projection conversions
  // Non-cacheable gemm inputs (d_hidb, d_t320b, attn/conv outs) convert here so
  // they don't evict the d_xb/d_Rhatb cache entries in d_xbf16.
  uint16_t* d_xbf16b = nullptr;  // [maxctx, 6144] bf16
  // GDEC_MOE_LT buffers (allocated only when enabled): gathered bf16 pair rows
  // and per-group dequant scratch.
  __hip_bfloat16* d_moexg = nullptr;   // [BP*topk, d]
  __hip_bfloat16* d_moewup = nullptr;  // [MOE_LT_GRP(x2 if OVL), 2*moe_mid, d]
  __hip_bfloat16* d_moewdn = nullptr;  // [MOE_LT_GRP(x2 if OVL), d, moe_mid]
  uint16_t* d_hidb2 = nullptr;  // GDEC_MOE_LT_CMP: fused-reference packed hid
  float* d_pairs2 = nullptr;    // GDEC_MOE_LT_CMP: fused-reference weighted pairs
  int* d_cmpi = nullptr;        // GDEC_MOE_LT_CMP: [max|diff|, max|ref|] ordered int
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
  int ple_ringpos = 0;           // host mirror for direct mode
  int* d_token = nullptr;        // device scalars for graph-replayable kernels
  int* d_pos = nullptr;
  int* d_ringpos = nullptr;
  int* p_token = nullptr;        // pinned staging for token id
  int pad_id = 248044;

  // ---- MTP speculative draft head (Phase A) ----
  // Single QSA+MoE layer on top of the trunk's 48-layer stream; predicts
  // t_{p+2} from (trunk h_p, emb(t_{p+1})) (DeepSeek-style wiring). Owns its
  // stream/KV/indexer state; reuses trunk scratch (strictly serial use).
  bool mtp_avail = false;      // checkpoint has mtp.* tensors and env is default
  bool ngram_verify_active = false;
  bool ngram_legacy_gdn = getenv("GDEC_NGRAM_LEGACY_GDN") != nullptr;
  bool ngram_legacy_final = getenv("GDEC_NGRAM_LEGACY_FINAL") != nullptr;
  bool mtp_tap_capture = false;  // prefill_chunk copies d_Rb rows into d_mtap
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
  float* d_mkcf = nullptr;     // MTP KV, FP32 mode (maxctx, 512)
  float* d_mvcf = nullptr;
  uint16_t* d_mkcb = nullptr;  // MTP KV, BF16 mode
  uint16_t* d_mvcb = nullptr;
  float* d_mik = nullptr;      // MTP indexer pooled keys (index_blocks, 128)
  float* d_miraw = nullptr;    // MTP indexer raw ring (4, 128)
  float* d_minorm = nullptr;   // MTP indexer norms q[128] + k[128]
  float* d_miw = nullptr;      // MTP indexer fp32 proj (640, 2560)
  float* d_mRb = nullptr;      // batch stream (maxbatch, 10240)
  float* d_mRhatb = nullptr;   // batch normed stream
  float* d_mtap = nullptr;     // captured trunk taps (maxbatch, 10240)
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
  // Rollback snapshots (spec verify): S/convst per accepted depth, indexer
  // proj rows per QSA layer. Written only when spec_snap is set.
  float* d_Ssnap = nullptr;    // [maxbatch+1][ngdn][48*128*128]
  float* d_csnap = nullptr;    // [maxbatch+1][ngdn][3*10240]
  float* d_iproj_snap = nullptr;  // [nqsa][maxbatch+1][128]
  bool spec_snap = false;         // set around spec verify chunks only

  explicit GpuModel(const Checkpoint& c, int mc) : ckpt(c), maxctx(mc) {
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
    int nqsa = g_cfg.layers / 4;  // 12
    if (!qsa_dense) {
      index_blocks = (maxctx + 3) / 4;
      DALLOC(&d_iproj, (size_t)maxctx * 640 * 4);
      DALLOC(&d_iq, (size_t)maxctx * 512 * 4);
      DALLOC(&d_ik, (size_t)nqsa * index_blocks * 128 * 4);
      DALLOC(&d_iraw, (size_t)nqsa * 4 * 128 * 4);
      CK(hipMemset(d_iraw, 0, (size_t)nqsa * 4 * 128 * 4));
      DALLOC(&d_inorm, (size_t)nqsa * 256 * 4);
      if (index_f32) DALLOC(&d_iw, (size_t)nqsa * 640 * 2560 * 4);
      DALLOC(&d_iscores, (size_t)std::min(maxctx, INDEX_BATCH) * 4 * index_blocks * 4);
      DALLOC(&d_iselected, (size_t)maxctx * INDEX_BUDGET * 4);
    }
    if (qsa_kv_fp32) {
      DALLOC(&d_kc, (size_t)nqsa * maxctx * 512 * 4);
      DALLOC(&d_vc, (size_t)nqsa * maxctx * 512 * 4);
    } else {
      DALLOC(&d_kcb, (size_t)nqsa * maxctx * 512 * 2);
      DALLOC(&d_vcb, (size_t)nqsa * maxctx * 512 * 2);
      if (qsa_wmma_btv) {
        qsa_vt_blocks = (size_t)(maxctx + 3) / 4;
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
    DALLOC(&d_sg, 4);
    DALLOC(&d_e, 2560 * 4);
    CK(hipHostMalloc((void**)&ple_e, 2560 * 4, hipHostMallocDefault));
    DALLOC(&d_token, 4);
    DALLOC(&d_pos, 4);
    DALLOC(&d_ringpos, 4);
    DALLOC(&d_rdelta, 4);
    CK(hipMemset(d_pos, 0, 4));
    CK(hipMemset(d_ringpos, 0, 4));
    CK(hipMemset(d_rdelta, 0, 4));
    if (getenv("GDEC_MROPE_DUMP")) {
      DALLOC(&d_cs1, 32 * sizeof(float2));
      DALLOC(&d_rpos1, 4);
    }
    CK(hipHostMalloc((void**)&p_token, 4, hipHostMallocDefault));
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

    // ---- prefill batch buffers (Phase 3a), allocated once at maxctx ----
    {
      size_t BP = (size_t)maxbatch_cap;
      DALLOC(&d_Rb, BP * 10240 * 4);
      DALLOC(&d_Rhatb, BP * 10240 * 4);
      DALLOC(&d_xb, BP * 2560 * 4);
      DALLOC(&d_yb, BP * 2560 * 4);
      DALLOC(&d_Gb, BP * 10240 * 4);
      DALLOC(&d_t320b, BP * 320 * 4);
      DALLOC(&d_w4b, BP * 4 * 4);
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
      // Fused down reuses the otherwise idle gate/up workspace for pair outputs.
      DALLOC(&d_guvb, BP * g_cfg.topk * (moe_deterministic ? 2560 : 1280) * 4);
      DALLOC(&d_hidb, BP * g_cfg.topk * 640 * 4);
      DALLOC(&d_moeaccb, BP * 2560 * 4);
      DALLOC(&d_eyb, BP * 2560 * 4);
      DALLOC(&d_sgb, BP * 4);
      DALLOC(&d_eb, BP * 2560 * 4);
      CK(hipHostMalloc((void**)&ple_eb, BP * 2560 * 4, hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ple_eb_dev, ple_eb, 0));
      DALLOC(&d_wbf16, (size_t)64 << 20);  // fits q_proj 12288x2560 bf16
      DALLOC(&d_xbf16, BP * 10240 * 2);
      DALLOC(&d_xbf16b, BP * 6144 * 2);  // largest non-cacheable K = 6144
      if (prefill_fused) DALLOC(&d_Rhatbf16, BP * 10240 * 2);
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
        // and its GEMMs (measured in tools/moe_lt_bench.cu). GDEC_MOE_LT_OVL
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
    }

    // ---- upload small fp32 weights (norms, conv, A_log, dt_bias) ----
    char buf[128];
    std::vector<float> tmp;
    auto upload = [&](const char* name, float* dst, size_t n) {
      tmp.resize(n);
      ckpt.dequant(ckpt.at(name), tmp.data());
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
      ple_table = &ckpt.at("layers.1.ple.ngram_embedding.weight");
      memcpy(&ple_tscale, ple_table->data + ple_table->numel(), 4);
      // io_uring path: file offset of the PLE table in the base .hgn. The fd
      // is the base file, so only the base mapping (maps from file offset 0)
      // is meaningful — if an overlay shadows the table, fall back.
#ifdef _WIN32
      if (g_ple_win.fh != INVALID_HANDLE_VALUE && !ckpt.mappings().empty()) {
#else
      if (g_ple_fd >= 0 && !ckpt.mappings().empty()) {
#endif
        const auto& mp0 = ckpt.mappings()[0];
        if (ple_table->data >= mp0.base && ple_table->data < mp0.base + mp0.len) {
          g_ple_foff = (uint64_t)(ple_table->data - mp0.base);
          g_ple_off_ok = true;
        }
      }
    }

    // ---- MTP draft head: detect weights, allocate own state ----
    if (ckpt.find("mtp.fc_hidden.weight")) {
      // moe_lt is compatible: it only activates at P >= moe_lt_min (4096) in
      // the trunk batch MoE, while MTP draft/ingest/verify run their own
      // fixed kernels at P <= gamma+1 <= 9. gr_bf16 taps are converted
      // bf16->f32 at capture/rollback/ingest (see spec_taps/restore_dR_row).
      bool moe_default = moe_deterministic && !moe_ordered && !moe_host_route &&
                         !moe_untiled && !moe_fp32_io;
      if (!moe_default) {
        fprintf(stderr, "mtp: disabled (moe_default=%d)\n", (int)moe_default);
      } else {
        // Validated wiring (mtp-test 0.855): branch-grouped norm + per-branch
        // fc_hidden + broadcast e. Env flips back for debugging only.
        const char* nm = getenv("GDEC_MTP_NORM");
        mtp_norm_mode = (nm && !strcmp(nm, "full")) ? 0 : 1;
        const char* hg = getenv("GDEC_MTP_HAGG");
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
        if (qsa_kv_fp32) {
          DALLOC(&d_mkcf, (size_t)maxctx * 512 * 4);
          DALLOC(&d_mvcf, (size_t)maxctx * 512 * 4);
        } else {
          DALLOC(&d_mkcb, (size_t)maxctx * 512 * 2);
          DALLOC(&d_mvcb, (size_t)maxctx * 512 * 2);
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
          DALLOC(&d_mik, (size_t)index_blocks * 128 * 4);
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
    fprintf(stderr, "gdec: state allocated (maxctx=%d)\n", maxctx);
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
        size_t ncap = std::max<size_t>(emb_off + M, 4096);
        if (d_img_emb) (void)hipFree(d_img_emb);
        CK(hipMalloc(&d_img_emb, ncap * 2560 * 2));
        img_emb_cap = ncap;
      }
      auto t0 = std::chrono::steady_clock::now();
      // Debug hook (GDEC_VISION_EMBEDS_NPY=<f4 (M,2560) npy>): inject
      // precomputed embeds (host bf16 RNE) instead of running the ViT.
      // Isolates LLM/injection behavior from ViT numerics in A/B probes.
      const char* dbg_emb = getenv("GDEC_VISION_EMBEDS_NPY");
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


  // ---- M-RoPE debug dump (GDEC_MROPE_DUMP=<file>; syncs the stream) ----
  // Format: "META rope_delta=<d>", then per table fill one line per row:
  //   ROW <base> <row> <pT> <pH> <pW> <cos0> <sin0> ... <cos31> <sin31>
  // and per decode step:
  //   DEC <tag> <rpos> <rdelta> <cos0> <sin0> ... <cos31> <sin31>
  // cos/sin as C %a hexfloats (exact binary value of the fp32 table entry).
  FILE* mrope_dump() {
    if (mrope_dump_ || !getenv("GDEC_MROPE_DUMP")) return mrope_dump_;
    mrope_dump_ = fopen(getenv("GDEC_MROPE_DUMP"), "w");
    if (mrope_dump_) fprintf(mrope_dump_, "# gdec-mrope-dump v1\n");
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
        tab[i] = pow(g_cfg.rope_theta, -2.0 * i / 64.0);
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
      k_rope_cs<<<(P * 32 + 255) / 256, 256, 0, g_str>>>(d_ropecs, base, P);
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
  // flash split stays flat. GDEC_QSA_DENSE keeps k_qsa_step as a debug fallback.
  template <typename KVT>
  void qsa_attn(const KVT* kc, const KVT* vc) {
    if (qsa_dense) {
      k_qsa_step<KVT><<<24, 256, 0, g_str>>>(d_qs, kc, vc, d_attn, d_pos, 256, 2, 24);
      return;
    }
    k_qsa_flash<true, false, KVT, true><<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
        d_qs, kc, vc, d_attn, 1, 256, 2, 24, d_iselected, 0, d_pos, d_pacc, d_pml);
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, d_attn, 12,
                                                       256, d_pos);
  }

  void qsa(int l, int qi, const float* x, float* y) {
    char buf[128];
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      float* keys = d_ik + (size_t)qi * index_blocks * 128;
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256, d_iq,
                                              1, g_cfg.norm_eps, g_cfg.rope_theta, d_pos,
                                              0, nullptr, d_rdelta);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256 + 128,
                                           d_iraw + qi * 512, keys, d_pos,
                                           g_cfg.norm_eps, g_cfg.rope_theta,
                                           d_rdelta);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(d_iq, keys, d_iscores,
                                                               index_blocks, d_pos);
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
      float* kc = d_kc + (size_t)qi * maxctx * 512;
      float* vc = d_vc + (size_t)qi * maxctx * 512;
      k_store_kv<<<2, 256, 0, g_str>>>(d_kb, d_vb, kc, vc, d_pos);
      qsa_attn(kc, vc);
    } else {
      uint16_t* kc = d_kcb + (size_t)qi * maxctx * 512;
      uint16_t* vc = d_vcb + (size_t)qi * maxctx * 512;
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(d_kb, d_vb, kc, vc, d_pos);
      qsa_attn(kc, vc);
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

    CK(hipMemsetAsync(d_moeacc, 0, g_cfg.d * 4, g_str));
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.gate_up_proj.weight", l);
    WView gu = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.down_proj.weight", l);
    WView dn = wview(ckpt.at(buf));
    const int* topids = d_topids;  // 直读:ids 即专家号,免 D2H 同步
    // fused experts: one grouped launch each for gate_up / silu / down+accum
    {
      // v3 kernel: 2 rows per warp -> k * moe_mid warp-pairs
      uint64_t pairs = (uint64_t)k * g_cfg.moe_mid;
      k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data, x,
          d_guv, topids, (uint64_t)2 * g_cfg.moe_mid, gu.cols, gu.scale_stride, k,
          0 /*x_stride*/, 16 /*id_stride*/, k);
    }
    k_silu_mul_g<<<(k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(d_guv, d_hid, g_cfg.moe_mid, k);
    {
      // v3 kernel: 2 rows per warp -> k * (d/2) warp-pairs
      uint64_t pairs = (uint64_t)k * ((g_cfg.d + 1) / 2);
      k_q4cp_gemv_gd<<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data, d_hid,
          d_moeacc, topids, d_topw, (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
          (uint64_t)g_cfg.moe_mid, k, 16 /*id_stride*/, 0 /*acc_stride*/, k);
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
      const uint8_t* p = ple_table->data + row * 160;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      uint8_t stg[16 * 160];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, stg, (uint64_t)(uintptr_t)ple_table->data, 16);
      for (int i = 0; i < 16 * 160; i++)
        dst[i] = hgn::fp8e4m3_to_f32(stg[i]) * ple_tscale;
      return;
    }
#endif
    // pass 2: read (pages already in flight)
    for (int h = 0; h < 16; h++) {
      const uint8_t* p = addrs[h];
      for (int i = 0; i < 160; i++)
        dst[h * 160 + i] = hgn::fp8e4m3_to_f32(p[i]) * ple_tscale;
    }
  }
  // raw fp8 variant: staging keeps the 160 code bytes per head; dequant+scale
  // happens on GPU (k_ple_fp8_dequant). ~4x less host work per token.
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
      const uint8_t* p = ple_table->data + row * 160;
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
    for (int h = 0; h < 16; h++) memcpy(dst + h * 160, addrs[h], 160);
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
  bool dbg = getenv("GDEC_DEBUG") != nullptr;
  bool perf = getenv("GDEC_PERF") != nullptr;
  bool qsa_loop = getenv("GDEC_QSA_LOOP") != nullptr;  // prefill QSA fallback
  bool qsa_global_v = getenv("GDEC_QSA_GLOBAL_V") != nullptr;
  bool moe_naive = getenv("GDEC_MOE_NAIVE") != nullptr;  // prefill MoE fallback
  // Small batches: per-pair naive gemv reads ~the same expert bytes as the
  // packed path (few shared experts) but at ~2x its effective bandwidth
  // (measured: P=8 chunk 0.19s -> 0.14s). 0 disables the auto switch.
  int moe_naive_max = [] {
    const char* e = getenv("GDEC_MOE_NAIVE_MAX");
    return e ? atoi(e) : 16;
  }();
  // Default small-P down path; GDEC_MOE_DOWN_ATOMIC restores the old
  // per-expert atomic accumulation for numerical/performance comparisons.
  bool moe_down_topk =
      moe_deterministic && getenv("GDEC_MOE_DOWN_ATOMIC") == nullptr;
  bool moe_ordered = getenv("GDEC_MOE_ORDERED") != nullptr;  // diagnostic: expert-order scatter
  bool moe_host_route = getenv("GDEC_MOE_HOST_ROUTE") != nullptr;
  bool moe_untiled = getenv("GDEC_MOE_UNTILED") != nullptr;
  bool moe_fp32_io = getenv("GDEC_MOE_FP32_IO") != nullptr;
  bool moe_up_legacy = getenv("GDEC_MOE_UP_LEGACY") != nullptr;
  bool moe_up_no_table = getenv("GDEC_MOE_UP_NO_TABLE") != nullptr;
  bool moe_down_no_table = getenv("GDEC_MOE_DOWN_NO_TABLE") != nullptr;
  bool moe_reduce_old = getenv("GDEC_MOE_REDUCE_OLD") != nullptr;
  // bf16 pair rows from the deterministic down kernel; the router weight is
  // applied in fp32 by k_moe_reduce_pw_bf16 instead of pre-multiplied in fp32.
  bool moe_pairs_bf16 = getenv("GDEC_MOE_PAIRS_BF16") != nullptr;
  bool moe_lt = getenv("GDEC_MOE_LT") != nullptr;  // prefill MoE: dequant + per-expert hipBLASLt
  // Validated PP dataflow changes. Unset each switch to restore its baseline.
  // Keep the P<=8 direct GEMV path's input precision and scratch lifetime.
  bool pp_moe_out = getenv("GDEC_PP_MOE_OUT") != nullptr;
  bool moe_lt_bf16 = getenv("GDEC_MOE_LT_BF16") != nullptr;  // bf16 GEMM outputs (LT path)
  bool moe_lt_cmp = getenv("GDEC_MOE_LT_CMP") != nullptr;  // debug: in-run fused reference diff
  bool moe_lt_ovl = getenv("GDEC_MOE_LT_OVL") != nullptr;  // overlap next group's dequant with current GEMMs (LT path)
  // LT loses below ~4K tokens (512 tiny per-expert launches + the d_eoff sync
  // cost more than the GEMM wins; measured 2026-09-10: +0.7s @2K, wash @8K).
  int moe_lt_min = getenv("GDEC_MOE_LT_MIN") ? atoi(getenv("GDEC_MOE_LT_MIN")) : 4096;
  bool gdn_loop = getenv("GDEC_GDN_LOOP") != nullptr;  // prefill GDN fallback
  bool gdn_nosplit = getenv("GDEC_GDN_NOSPLIT") != nullptr;  // single-kernel GDN fallback
  bool gdn_nostrip = getenv("GDEC_GDN_NOSTRIP") != nullptr;  // old inter (no strip)
  bool gdn_stream = getenv("GDEC_GDN_STREAM") != nullptr;  // four-chunk workspace window
  int gdn_window_chunks = [] {
    const char* e = getenv("GDEC_GDN_WINDOW_CHUNKS");
    return e ? std::max(1, atoi(e)) : 4;
  }();
  bool gdn_wave = getenv("GDEC_GDN_WAVE") != nullptr;  // column-local triangular solve
  bool gdn_pipe = getenv("GDEC_GDN_PIPE") != nullptr;  // intra/strip two-stream window pipeline (strip path)
  bool gdn_pipe2 = getenv("GDEC_GDN_PIPE2") != nullptr;  // fp16-resident WMMA intra (chunk-looped)
  bool gdn_fused = getenv("GDEC_GDN_FUSED") != nullptr;  // intra+strip fused, block-per-head persistent (no ws)
  bool gr_scat4 = getenv("GDEC_GR_SCAT4") != nullptr;  // one-block-per-token GR scatter+norm
  bool gdn_ut5 = getenv("GDEC_GDN_UT5") != nullptr;  // blocked triangular inverse
  bool ple_loop = getenv("GDEC_PLE_LOOP") != nullptr;  // prefill PLE fallback
  bool prof = getenv("GDEC_PROF") != nullptr;  // host-side phase timing
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
    WView et = wview(ckpt.at("embed_tokens.weight"));
    const uint8_t* codes = et.data + 64;
    const uint8_t* scales = et.data + 64 + et.rows * et.cols / 2;
    k_q4cp_row<<<1, 128, 0, g_str>>>(codes, scales, (const float*)et.data, d_token,
                                     et.cols, et.scale_stride, d_emb);
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
    k_argmax<<<1, 1024, 0, g_str>>>(d_logits, g_cfg.vocab, d_argmax);
  }

  // =================== prefill batch (Phase 3a) ==============================
  // Batched GEMM: Y[P,N] = X[P,K] . W[N,K]^T. W dequanted to bf16 workspace,
  // X converted to bf16; rocBLAS col-major mapping: C_cm[N,P] with
  // A = W (transa=T, lda=K), B = X (transb=N, ldb=K), C = Y (ldc=N).
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
    static const bool gemm_dbg = getenv("GDEC_GEMM_DBG") != nullptr;
    if (gemm_dbg)
      fprintf(stderr, "gemm-LT: dtype=%d N=%llu K=%llu P=%d xbf=%d xs=%llu\n",
              w.dtype, (unsigned long long)N, (unsigned long long)K, P,
              (int)(xbf != nullptr), (unsigned long long)xstride);
    const uint16_t* wp = nullptr;
    if (w.dtype == 0) {
      wp = (const uint16_t*)w.data;
    } else if (w.dtype == 5) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 32;
      k_dequant_q4cp_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          w.data + 64, w.data + 64 + N * K / 2, (const float*)w.data, d_wbf16, N, K,
          w.scale_stride);
      wp = d_wbf16;
    } else if (w.dtype == 7) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 64;
      k_dequant_q8g64_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          w.data, d_wbf16, N, K);
      wp = d_wbf16;
    } else {
      throw std::runtime_error("gemm: unsupported dtype");
    }
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
    if (getenv("GDEC_GEMM_RB")) {  // fallback: rocBLAS default algo
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
    // Phase 3g: self-written bf16 WMMA GEMM (GDEC_GEMM_WMMA=1). Diverts the
    // winning projection shapes at P >= 1024 to k_gemm_wmma — d3 config
    // (512t w2x8 f4x2, gm=16) for N=2560 K=6144, d9 (256t w2x4 f4x4) for
    // N=6144/10240/12288 K=2560 (gm=4) and the narrow-K/N winners
    // N=10240 K=320 / N=512 K=2560 / N=2560 K=640 (gm=2, 1.25-2.0x vs Lt in
    // tools/gemm_lt_bench.cu); every other shape and the default flag-off
    // state fall through to hipBLASLt unchanged. Purely async. The kernel
    // requires K % 32 == 0; P and N tails are handled inside the kernel
    // (staging row clamps + predicated epilogue; the grid must be ceiled to
    // match). (320,10240) and (640,2560) were measured and stay on Lt —
    // 0.94x/1.08x, under the 1.15x bar. GDEC_GEMM_RB above takes precedence
    // if both are set.
    static const bool gemm_wmma = getenv("GDEC_GEMM_WMMA") != nullptr;
    if (gemm_wmma && P >= 1024) {
      const bool gw_d3 = (N == 2560 && K == 6144);
      const bool gw_d9 = (K == 2560 && (N == 6144 || N == 10240 || N == 12288));
      const bool gw_d9n = (N == 10240 && K == 320) ||
                          (N == 512 && K == 2560) || (N == 2560 && K == 640);
      if (gw_d3 || gw_d9 || gw_d9n) {
        const unsigned gw_grid =
            (unsigned)(((P + 127) / 128) * (int)((N + 255) / 256));
        if (gw_d3)
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
  // -> op N) -> C[N,P] (n-contiguous), fp32 or bf16 out. Used by GDEC_MOE_LT
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

  void gr_mix_b(const std::string& prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_Rhatb, d_t320b, P, 0,
         gr_bf16 ? d_Rhatbf16 : nullptr);
    k_silu_scale<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(d_t320b, 0.25f, P * 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_t320b, d_Gb, P);
    if (prefill_fused) {
      if (gr_bf16)
        k_gr_combine_b_hc_bf16<<<P, 256, 0, g_str>>>(
            d_Gb, d_Rhatbf16, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16);
      else
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
    gr_mix_b(prefix, P);
  }

  void gr_write_b(const std::string& prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_Rhatb, d_w4b, P, 0,
         gr_bf16 ? d_Rhatbf16 : nullptr);
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
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight",
             write_prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_Rhatb, d_w4b, P, 0,
         gr_bf16 ? d_Rhatbf16 : nullptr);
    float* hw = ensure_hcnorm(read_prefix);
    if (gr_bf16)
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
    if (xf16_src == d_Rhatb) xf16_src = nullptr;
    gr_mix_b(read_prefix, P);
  }

  // GDN: projections batched. Chunked path: conv/l2norm/gates batched, then
  // k_gdn_chunk (chunk=64, fp32) advances S and writes all outputs at once.
  // GDEC_GDN_LOOP=1 falls back to the per-token recurrence loop.
  void gdn_b(int l, int gi, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_qkv.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_qkvb, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_a.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_g48b, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_b.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_beta48b, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_zb, P);
    float* convst = d_convst + (size_t)gi * 3 * 10240;
    const float* cw = d_convw + (size_t)gi * 10240 * 4;
    const float* Alog = d_Alog + (size_t)gi * 48;
    const float* dtb = d_dtbias + (size_t)gi * 48;
    float* S = d_S + (size_t)gi * 48 * 128 * 128;
    const float* gnw = d_gdnnorm + (size_t)gi * 128;
    float qscale = 1.f / sqrtf(128.f);
    if (!gdn_loop) {
      int64_t tot = (int64_t)P * 10240 / 4;  // 4 channels per thread (float4)
      k_gdn_conv_b<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          d_qkvb, cw, convst, d_convb, P, 10240);
      if (spec_snap) {
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        k_convst_snap<<<P, 512, 0, g_str>>>(
            d_qkvb, convst, d_csnap + (size_t)gi * 3 * 10240, P, 10240,
            (size_t)ngdn * 3 * 10240);
      }
      k_convst_update<<<(10240 + 255) / 256, 256, 0, g_str>>>(convst, d_qkvb, P,
                                                              10240);
      k_l2norm_qk_b<<<dim3(16, P), 128, 0, g_str>>>(d_convb, 128, qscale, 10240, 16);
      k_gdn_gates<<<P, 64, 0, g_str>>>(d_g48b, d_beta48b, Alog, dtb, d_g48b, d_beta48b,
                                       48, 48);
      if (spec_snap) {
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        k_gdn_snap<<<48, 512, 0, g_str>>>(
            d_convb, d_g48b, d_beta48b, S, d_Ssnap + (size_t)gi * 48 * 128 * 128,
            P, 10240, (size_t)ngdn * 48 * 128 * 128);
      }
      if (ngram_verify_active && !ngram_legacy_gdn && P <= 65) {
        k_gdn_verify<<<48, 512, 0, g_str>>>(
            d_convb, d_g48b, d_beta48b, S, d_convb, P, 10240);
      } else if (gdn_nosplit) {
        k_gdn_chunk<<<48, GDN_NT, 0, g_str>>>(d_convb, d_g48b, d_beta48b, S, d_convb,
                                              d_gdn_ws, P, 10240);
      } else if (gdn_fused) {
        // one launch for all of P: no window, no workspace; S in/out via global
        k_gdn_fused<<<48, 1024, 0, g_str>>>(d_convb, d_g48b, d_beta48b, S,
                                            d_convb, P, 10240);
      } else {
        // Keep producer and consumer close; S carries the exact recurrence
        // across windows and the last window alone can contain a padded chunk.
        int window = gdn_stream ? 64 * gdn_window_chunks : P;
        // GDEC_GDN_PIPE (strip path): intra stays on g_str, strips run on a
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
      k_gdn_gatednorm_b<true><<<dim3(48, (P + 31) / 32), 128, 0, g_str>>>(
          d_convb + 4096, d_zb, gnw, nullptr, g_cfg.norm_eps, 10240, 6144,
          d_xbf16b, P);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
      gemm(wview(ckpt.at(buf)), d_convb + 4096, d_yb, P, 10240 /*row stride*/,
           d_xbf16b);
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

  // QSA prefill flash over the KV cache: dense prefix plus the sparse tail.
  // base = absolute position of batch row 0 (0 for a fresh prefill; > 0 when
  // continuing from live KV/state). Dense rows are the batch-local prefix,
  // sparse queries are the absolute positions >= 2051 within the batch.
  template <typename KVT>
  void qsa_flash_b(const KVT* kc, const KVT* vc, int P, int dense_P, int base = 0,
                   const uint16_t* vc_transposed = nullptr) {
    if (dense_P > 0)
      k_qsa_flash<false, false, KVT><<<dim3((dense_P + 15) / 16, 24), 128, 0, g_str>>>(
          d_qsb, kc, vc, d_attnb, dense_P, 256, 2, 24, nullptr, 0, nullptr, nullptr,
          nullptr, base);
    int first_abs = std::max(base, 2051), nsparse = base + P - first_abs;
    if (!qsa_dense && nsparse > 0) {
      const int* sel = d_iselected + (size_t)first_abs * 512;
      if (qsa_global_v)
        k_qsa_flash<true, false, KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
            d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, nullptr, nullptr,
            nullptr, base);
      else if constexpr (sizeof(KVT) == 2) {
        if (qsa_union && vc_transposed) {
          // 并集路径：[ga, ge) 内的 4 对齐完整组走 merge + k_qsa_wmma_u，
          // 两侧 ragged 边缘回退逐 token 的 k_qsa_wmma<true>
          const int ga = (first_abs + 3) & ~3;
          const int ge = (base + P) & ~3;
          if (ge - ga >= 4) {
            const int ngroups = (ge - ga) >> 2;
            k_qsa_union_merge<<<ngroups, 128, 0, g_str>>>(
                d_iselected, ga, d_ublk, d_umask, d_ucount, QSA_UCAP);
            k_qsa_wmma_u<<<dim3(ngroups, 2), 256, 0, g_str>>>(
                d_qsb, kc, vc_transposed, d_attnb, 256, 2, 24, d_ublk, d_umask,
                d_ucount, QSA_UCAP, ga, base);
          }
          if (ga > first_abs)
            k_qsa_wmma<true><<<dim3(ga - first_abs, 2), 256, 0, g_str>>>(
                d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel,
                first_abs, base);
          if (base + P > ge)
            k_qsa_wmma<true><<<dim3(base + P - ge, 2), 256, 0, g_str>>>(
                d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel, ge,
                base);
        }
        else if (qsa_wmma6 && vc_transposed)
          k_qsa_wmma6<<<dim3(nsparse, 2), 128, 0, g_str>>>(
              d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel,
              first_abs, base);
        else if (qsa_wmma_btv && vc_transposed)
          k_qsa_wmma<true><<<dim3(nsparse, 2), 256, 0, g_str>>>(
              d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel, first_abs,
              base);
        else if (qsa_wmma)
          // GDEC_QSA_WMMA=1: WMMA bf16 sparse flash (256 thr), see k_qsa_wmma.
          k_qsa_wmma<false><<<dim3(nsparse, 2), 256, 0, g_str>>>(
              d_qsb, kc, vc, nullptr, d_attnb, P, 256, 2, 24, sel, first_abs, base);
        else
          // BF16 KV mode (!qsa_kv_fp32): bf16 LDS tiles + fdot2 (-61% at 32K).
          k_qsa_flash_bf16<KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
              d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, base);
      }
      else
        k_qsa_flash<true, true, KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
            d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, nullptr, nullptr,
            nullptr, base);
    }
  }

  // QSA: projections batched. Flash path: qsplit/norm/rope batched, KV rows
  // bulk-copied to cache slots 0..P-1, one causal flash kernel over all
  // positions. GDEC_QSA_LOOP=1 falls back to the per-token loop (debug).
  void qsa_b(int l, int qi, int P, int base = 0) {
    char buf[128];
    if (!qsa_dense) {
      // rope cos/sin table for the indexer kernels below (idempotent; the QSA
      // prep path re-ensures the same (base, P) later and hits the cache)
      ensure_rope_tab(g_cfg.rope_theta, 64);
      ensure_rope_cs(base, P);
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      if (index_f32) {
        float one = 1, zero = 0;
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose, rocblas_operation_none,
                                    640, P, 2560, &one, d_iw + (size_t)qi * 640 * 2560,
                                    2560, d_xb, 2560, &zero, d_iproj, 640);
        if (status != rocblas_status_success) {
          fprintf(stderr, "indexer fp32 projection failed: %d\n", (int)status);
          exit(1);
        }
      } else {
        gemm(wview(ckpt.at(buf)), d_xb, d_iproj, P);
      }
      float* keys = d_ik + (size_t)qi * index_blocks * 128;
      k_index_q<<<dim3(P, 4), 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256, d_iq,
                                              P, g_cfg.norm_eps, g_cfg.rope_theta,
                                              nullptr, base, d_ropecs);
      // absolute 4-token blocks [base/4, (base+P)/4); a left-straddling block
      // (base%4 != 0) is completed with history rows from the raw ring
      int nb = (base + P) / 4, b0 = base / 4;
      if (nb > b0)
        k_index_pool<<<nb - b0, 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256 + 128,
                                                 keys, g_cfg.norm_eps,
                                                 g_cfg.rope_theta, base,
                                                 d_iraw + qi * 512, d_ropecs);
      k_index_ring<<<min(P, 4), 128, 0, g_str>>>(d_iproj, d_iraw + qi * 512, P, base);
      if (spec_snap)
        k_iproj_snap<<<P, 128, 0, g_str>>>(
            d_iproj, d_iproj_snap + (size_t)qi * 9 * 128, P);
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
      for (int first = a0; first < base + P;) {
        int nxt = first < grid ? min(grid, base + P)
                               : min(first + INDEX_BATCH, base + P);
        int count = nxt - first;
        if (index_fused2 && count >= 16 && getenv("GDEC_INDEX_OLDSEL") == nullptr) {
          k_index_scores_tiled<<<dim3((nb + 31) / 32, (count + 15) / 16), 256, 0, g_str>>>(
              d_iq + (size_t)(first - base) * 512, keys, d_iscores, nb, count, first);
          if ((first + count) / 4 <= 8192)
            k_index_select_rs<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, d_iselected + (size_t)first * 512, first);
          else if (index_stream_select)
            k_index_select_stream<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, d_iselected + (size_t)first * 512, first);
          else
            k_index_select<16, true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, d_iselected + (size_t)first * 512, first);
          first = nxt;
          continue;
        }
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose, rocblas_operation_none,
                                    nb, count * 4, 128, &one, keys, 128,
                                    d_iq + (size_t)(first - base) * 512, 128, &zero,
                                    d_iscores, nb);
        if (status != rocblas_status_success) {
          fprintf(stderr, "indexer sgemm failed: %d\n", (int)status);
          exit(1);
        }
        index_select(d_iscores, nb, d_iselected + (size_t)first * 512, first, count);
        first = nxt;
      }
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_qgb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_kbb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.v_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_vbb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_norm.weight", l);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_norm.weight", l);
    float* knw = ensure_norm256(buf);
    const size_t qsa_off = (size_t)qi * maxctx * 512;
    if (!qsa_loop) {
      ensure_rope_tab(g_cfg.rope_theta, 64);
      ensure_rope_cs(base, P);
      // fused: qsplit+rope (K1) and knorm+rope[+bf16 store] (K2), bit-exact
      // vs the previous qsplit / rmsnorm_zc / rope_b x2 / convert chain
      k_qsa_qsplit<true><<<dim3(24, P), 256, 0, g_str>>>(d_qgb, qnw, d_qsb, d_gsb,
                                                         256, g_cfg.norm_eps, base,
                                                         d_ropecs);
      int dense_P = qsa_dense ? P : min(P, std::max(0, 2051 - base));
      if (qsa_kv_fp32) {
        float* kc = d_kc + qsa_off;
        float* vc = d_vc + qsa_off;
        k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(d_kbb, knw, d_kbb,
                                                          nullptr, g_cfg.norm_eps,
                                                          base, d_ropecs);
        // absolute positions base..base+P-1 are contiguous: one bulk copy
        CK(hipMemcpyAsync(kc + (size_t)base * 512, d_kbb, (size_t)P * 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        CK(hipMemcpyAsync(vc + (size_t)base * 512, d_vbb, (size_t)P * 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        qsa_flash_b(kc, vc, P, dense_P, base);
      } else {
        uint16_t* kc = d_kcb + qsa_off;
        uint16_t* vc = d_vcb + qsa_off;
        k_qsa_kprep<true><<<dim3(2, P), 256, 0, g_str>>>(
            d_kbb, knw, d_kbb, kc + (size_t)base * 512, g_cfg.norm_eps, base,
            d_ropecs);
        // absolute positions base..base+P-1 are contiguous: one bulk convert
        k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256, 0,
                           g_str>>>(d_vbb, vc + (size_t)base * 512, 512, P, 512);
        const size_t bt_layer = qsa_vt_blocks * 2 * 256 * 4;
        uint16_t* vct = qsa_wmma_btv
                            ? d_vcbt + (size_t)qi * bt_layer
                            : nullptr;
        if (vct)
          k_f32_to_bf16_v4_bt<<<(unsigned)(((size_t)((base + P + 3) / 4 - base / 4) *
                                            512 + 255) /
                                               256),
                               256, 0, g_str>>>(d_vbb, vct, 512, P, 512, base);
        qsa_flash_b(kc, vc, P, dense_P, base, vct);
      }
      k_sigmoid_gate<<<(P * 6144 + 255) / 256, 256, 0, g_str>>>(d_attnb, d_gsb,
                                                                d_attnb, P * 6144);
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
                                           pabs);
          k_qsa_step<float><<<24, 256, 0, g_str>>>(
              qs, d_kc + qsa_off, d_vc + qsa_off, att, pabs, 256, 2, 24,
              qsa_dense ? nullptr : d_iselected + (size_t)(base + t) * 512);
        } else {
          k_store_kv_bf16<<<2, 256, 0, g_str>>>(kb, vb, d_kcb + qsa_off,
                                                d_vcb + qsa_off, pabs);
          k_qsa_step<uint16_t><<<24, 256, 0, g_str>>>(
              qs, d_kcb + qsa_off, d_vcb + qsa_off, att, pabs, 256, 2, 24,
              qsa_dense ? nullptr : d_iselected + (size_t)(base + t) * 512);
        }
        k_sigmoid_gate<<<(6144 + 255) / 256, 256, 0, g_str>>>(att, gs, att, 6144);
      }
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.o_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_attnb, d_yb, P);
  }

  // MoE: router batched; expert GEMVs grouped over P*k slots (naive: full
  // weight traffic per token — correct but slow; optimized in a later phase).
  void moe_b(int l, int P) {
    char buf[128];
    const bool direct_output = pp_moe_out && P > 8;
    float* const acc = direct_output ? d_yb : d_moeaccb;
    snprintf(buf, sizeof buf, "layers.%d.mlp.gate.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_routerb, P);
    int k = g_cfg.topk;
    k_router_topk<<<P, 512, 0, g_str>>>(d_routerb, d_topidsb, d_topwb, g_cfg.experts, k);
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.gate_up_proj.weight", l);
    WView gu = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.down_proj.weight", l);
    WView dn = wview(ckpt.at(buf));
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
      // v3 kernels: 2 rows per warp -> halved warp counts
      uint64_t pairs = (uint64_t)P * k * g_cfg.moe_mid;
      k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
          d_xb, d_guvb, topidsb, (uint64_t)2 * g_cfg.moe_mid, gu.cols,
          gu.scale_stride, P * k, (uint64_t)g_cfg.d, 16, k);
      k_silu_mul_g<<<(P * k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(
          d_guvb, d_hidb, g_cfg.moe_mid, P * k);
      if (direct_down) {
        uint64_t row_pairs = (g_cfg.d + 1) / 2;
        constexpr uint64_t warps_per_block = 128 / 32;
        k_q4cp_gemv_gd_topk_h16<10><<<
            (unsigned)(((uint64_t)P * row_pairs + warps_per_block - 1) /
                       warps_per_block),
            128, 0, g_str>>>(
            dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
            (const float*)dn.data, d_hidb, acc, topidsb, d_topwb,
            (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
            (uint64_t)g_cfg.moe_mid, P, 16, (uint64_t)g_cfg.d);
      } else {
        uint64_t pairs2 = (uint64_t)P * k * ((g_cfg.d + 1) / 2);
        k_q4cp_gemv_gd<<<(unsigned)((pairs2 + 15) / 16), 512, 0, g_str>>>(
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
        // GDEC_MOE_LT_OVL pipeline: scratch is double-buffered (slots gi&1);
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
          k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E);
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
        k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E);
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
        // GDEC_MOE_PAIRS_BF16: down writes unweighted bf16 pair rows; the
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
    k_axpy_sg<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(acc, d_eyb, d_sgb,
                                                         g_cfg.d, P * g_cfg.d);
    if (!direct_output)
      CK(hipMemcpyAsync(d_yb, acc, (size_t)P * g_cfg.d * 4, hipMemcpyDeviceToDevice,
                        g_str));
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
        d_Unb, d_ple_convw, d_convoutb, P, 10240, base, d_ring);
    k_add2<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(d_Rb, d_Ub, d_convoutb,
                                                             (int)tot);
    k_ple_ring_wr<<<9, 256, 0, g_str>>>(d_Unb, d_ring, P, 10240, base);
  }

  void enq_layer_b(int l, int P, bool fuse_prev_mlp = false,
                   bool flush_mlp = true, int base = 0) {
    char attn_prefix[128], mlp_prefix[128];
    snprintf(attn_prefix, sizeof attn_prefix,
             "layers.%d.attn_hyper_connection", l);
    if (fuse_prev_mlp) {
      snprintf(mlp_prefix, sizeof mlp_prefix,
               "layers.%d.mlp_hyper_connection", l - 1);
      gr_write_read_b(mlp_prefix, attn_prefix, P);
    } else {
      gr_read_b(attn_prefix, P);
    }
    if (is_qsa(l))
      qsa_b(l, l / 4, P, base);
    else
      gdn_b(l, l - (l + 1) / 4, P);
    if (dbg) {
      h_rms("gpu", l, "attn-x", d_xb + (size_t)(P - 1) * 2560, 2560);
      h_rms("gpu", l, "attn-y", d_yb + (size_t)(P - 1) * 2560, 2560);
    }
    snprintf(mlp_prefix, sizeof mlp_prefix,
             "layers.%d.mlp_hyper_connection", l);
    if (gr_scatter_norm)
      gr_write_read_b(attn_prefix, mlp_prefix, P);
    else {
      gr_write_b(attn_prefix, P);
      gr_read_b(mlp_prefix, P);
    }
    moe_b(l, P);
    if (dbg) {
      h_rms("gpu", l, "mlp-x", d_xb + (size_t)(P - 1) * 2560, 2560);
      h_rms("gpu", l, "mlp-y", d_yb + (size_t)(P - 1) * 2560, 2560);
    }
    if (flush_mlp) gr_write_b(mlp_prefix, P);
    if (!dbg && isatty(2)) {  // per-layer \r is terminal-only; logs use prefill-live
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
  // the pipeline the way the GDEC_PHASE sync profiling does.
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
    while (hipEventQuery(prog_ev[g_cfg.layers - 1]) == hipErrorNotReady) {
      prog_print(P);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  int prefill_batch(const std::vector<int>& tokens, int base = 0, bool final = true,
                    bool quiet = false) {
    int N = (int)tokens.size();
    int id = -1;
    for (int off = 0; off < N;) {
      int rem = N - off;
      int n = std::min(maxbatch, rem);
      // Absorb a small tail into this chunk instead of running a tiny final
      // chunk whose per-chunk fixed cost (~0.9 s) dwarfs its token count.
      if (rem > maxbatch && rem <= maxbatch_cap) n = rem;
      id = prefill_chunk(
          std::vector<int>(tokens.begin() + off, tokens.begin() + off + n),
          base + off, final, quiet);
      if (mtp_tap_capture) {
        // Ingest this chunk's taps (chunk-local rows in d_mtap) into the MTP
        // layer before the next chunk overwrites them. MTP position i takes
        // token i+1, so the very last prompt row is left to the spec loop.
        int c0 = base + off;
        int count = std::min(n, base + N - 1 - c0);
        if (count > 0) {
          std::vector<int> shifted(tokens.begin() + off + 1,
                                   tokens.begin() + off + 1 + count);
          mtp_ingest_b(d_mtap, shifted, count, c0);
        }
      }
      off += n;
    }
    return id;
  }

  int prefill_chunk(const std::vector<int>& tokens, int base = 0, bool final = true,
                    bool quiet = false) {
    int P = (int)tokens.size();
    if (P < 1 || base + P > maxctx || base < 0 || P > maxbatch_cap) {
      fprintf(stderr, "prefill_chunk: base=%d P=%d out of range "
              "(maxctx=%d maxbatch_cap=%d)\n", base, P, maxctx, maxbatch_cap);
      exit(1);
    }
    static const bool phase_prof = getenv("GDEC_PHASE") != nullptr;
    auto phase = [&](const char* tag, std::chrono::steady_clock::time_point& tp) {
      if (!phase_prof) return;
      CK(hipStreamSynchronize(g_str));
      auto now = std::chrono::steady_clock::now();
      fprintf(stderr, "phase %-10s %8.1f ms\n", tag,
              std::chrono::duration<double, std::milli>(now - tp).count());
      tp = now;
    };
    xf16_src = nullptr;  // a new batch invalidates the bf16 conversion cache
#ifdef GDEC_PHASE_PROF
    {
      static const unsigned long long zeros[16] = {};
      CK(hipMemcpyToSymbol(g_ph, zeros, sizeof(zeros)));
    }
#endif
    auto T0 = std::chrono::steady_clock::now();
    auto TP = T0;
    // PLE rows depend only on token history, so start the existing page-fault
    // pipeline before L0. Its GPU consumer remains ordered after L0 below.
    std::vector<std::thread> ple_pool;
    std::chrono::steady_clock::time_point ple_start, ple_done;
    std::atomic<int> ple_workers{0};
    int ple_base0 = 0;
    uint8_t* ple_fp8 = nullptr;
    // two-phase PLE gather state (must outlive the async workers — every
    // variable the pool lambdas capture by reference lives at this scope;
    // declaring aliases/sizes inside the gather block is
    // stack-use-after-scope once the workers outlive that block. ASan-proven
    // in plegs v1/v2; the pre-plegs code had the same latent bug via
    // block-scope reference aliases — plfix removes them).
    std::vector<uint64_t> ple_addrs, ple_keys;
    std::atomic<int> ple_barrier{0};
    if (ple_on) {
      ple_start = std::chrono::steady_clock::now();
      for (int t = 0; t < P; t++) chist.push_back(tokens[t]);
      ple_base0 = (int)chist.size() - P;
      ple_fp8 = (uint8_t*)ple_eb;
      constexpr int nt = 16;
      ple_workers.store(nt);
      ple_pool.reserve(nt);
      if (!getenv("GDEC_PLE_NOSORT")) {
        ple_barrier.store(0, std::memory_order_relaxed);
        if (ple_uring_ready()) {
          // io_uring batch-pread gather: phase A is the same 16-thread row
          // address computation; worker 0 then pumps all 16*P rows in token
          // order through the 256-deep ring straight into pinned staging.
          ple_addrs.resize((size_t)P * 16);
          ple_workers.store(1);  // only the pump thread accounts for ple_done
          for (int w = 0; w < nt; w++)
            ple_pool.emplace_back([&, w] {
              const int t0 = (int)((size_t)w * P / nt), t1 = (int)((size_t)(w + 1) * P / nt);
              for (int t = t0; t < t1; t++) {
                const int at = ple_base0 + t;
                auto cat = [&](int back) -> int64_t {
                  int j = at - back;
                  return j < 0 ? pad_id : chist[j];
                };
                for (int h = 0; h < 16; h++) {
                  int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
                  if (h >= 8) mix ^= cat(2) * ple_mult[2];
                  uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
                  ple_addrs[(size_t)t * 16 + h] = (uint64_t)(uintptr_t)(ple_table->data + row * 160);
                }
              }
              ple_barrier.fetch_add(1, std::memory_order_acq_rel);
              if (w != 0) return;
              while (ple_barrier.load(std::memory_order_acquire) < nt)
                std::this_thread::yield();
              PLE_PUMP.run(ple_addrs.data(), ple_fp8,
                           (uint64_t)(uintptr_t)ple_table->data, (size_t)P * 16);
              if (ple_workers.fetch_sub(1) == 1)
                ple_done = std::chrono::steady_clock::now();
            });
        } else {
        // Two-phase gather: phase A computes all 16*P row addresses, sorts
        // each shard by page and issues madvise in ascending order (random
        // 4K reads become quasi-sequential for the disk scheduler, adjacent
        // pages dedup); phase B copies rows in token order with pages
        // already in flight. Output staging bytes are identical to the
        // per-token madvise+gather pipeline. The PLE table is mostly
        // page-cache-cold in steady state (124GB file > RAM), so submission
        // order is the cost driver (measured: 512K random madvise ~2.5s
        // serial-equivalent, warm gather ~0.01s).
        ple_addrs.resize((size_t)P * 16);
        ple_keys.resize((size_t)P * 16);
        for (int w = 0; w < nt; w++)
          ple_pool.emplace_back([&, w] {
            const int t0 = (int)((size_t)w * P / nt), t1 = (int)((size_t)(w + 1) * P / nt);
            for (int t = t0; t < t1; t++) {
              const int at = ple_base0 + t;
              auto cat = [&](int back) -> int64_t {
                int j = at - back;
                return j < 0 ? pad_id : chist[j];
              };
              for (int h = 0; h < 16; h++) {
                int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
                if (h >= 8) mix ^= cat(2) * ple_mult[2];
                uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
                ple_addrs[(size_t)t * 16 + h] = (uint64_t)(uintptr_t)(ple_table->data + row * 160);
              }
            }
            uint64_t* kb = ple_keys.data() + (size_t)t0 * 16;
            std::memcpy(kb, ple_addrs.data() + (size_t)t0 * 16, (size_t)(t1 - t0) * 16 * 8);
            std::sort(kb, kb + (size_t)(t1 - t0) * 16);
            uint64_t prev = ~0ull;
            for (int64_t i = 0; i < (int64_t)(t1 - t0) * 16; i++) {
              uint64_t pg = kb[i] & ~4095ull;
              if (pg != prev) {
                madvise((void*)(uintptr_t)pg, 4096, MADV_WILLNEED);
                prev = pg;
              }
            }
            ple_barrier.fetch_add(1, std::memory_order_acq_rel);
            while (ple_barrier.load(std::memory_order_acquire) < nt)
              std::this_thread::yield();
            for (int t = t0; t < t1; t++) {
              uint8_t* dst = ple_fp8 + (size_t)t * 2560;
              const uint64_t* ap = ple_addrs.data() + (size_t)t * 16;
              for (int h = 0; h < 16; h++)
                std::memcpy(dst + h * 160, (const void*)(uintptr_t)ap[h], 160);
            }
            if (ple_workers.fetch_sub(1) == 1)
              ple_done = std::chrono::steady_clock::now();
          });
        }
      } else {
        for (int w = 0; w < nt; w++)
          ple_pool.emplace_back([&, w] {
            for (int t = w; t < P; t += nt)
              ple_rows_fp8(ple_base0 + t, ple_fp8 + (size_t)t * 2560);
            if (ple_workers.fetch_sub(1) == 1)
              ple_done = std::chrono::steady_clock::now();
          });
      }
    }
    CK(hipMemcpyAsync(d_tokarr, tokens.data(), P * 4, hipMemcpyHostToDevice, g_str));
    // embed all tokens: gather rows then repeat to 4 branches (in-place safe)
    WView et = wview(ckpt.at("embed_tokens.weight"));
    const uint8_t* codes = et.data + 64;
    const uint8_t* scales = et.data + 64 + et.rows * et.cols / 2;
    if (gr_bf16)
      k_q4cp_row_b_hc_bf16<<<P, 128, 0, g_str>>>(
          codes, scales, (const float*)et.data, d_tokarr, et.cols,
          et.scale_stride, (uint16_t*)d_Rb);
    else {
      k_q4cp_row_b<<<P, 128, 0, g_str>>>(codes, scales, (const float*)et.data,
                                         d_tokarr, et.cols, et.scale_stride, d_Rb,
                                         10240);
      k_repeat4_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_Rb, d_Rb, g_cfg.d);
    }
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
    if (prog_total > 0) prog_probe(0, P);
    phase("embed+L0", TP);
    if (ple_on) {
      auto wait_start = std::chrono::steady_clock::now();
      for (auto& th : ple_pool) th.join();
      auto wait_done = std::chrono::steady_clock::now();
      if (prof) {
        p_pleh +=
            std::chrono::duration<double, std::milli>(ple_done - ple_start).count();
        p_ple_wait +=
            std::chrono::duration<double, std::milli>(wait_done - wait_start).count();
      }
      // dequant fp8 -> f32 d_eb on GPU (zero-copy read of pinned staging)
      size_t n16 = (size_t)P * 2560 / 16;
      k_ple_fp8_dequant<<<(unsigned)((n16 + 255) / 256), 256, 0, g_str>>>(
          ple_eb_dev, d_eb, ple_tscale, n16);
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
    phase("ple", TP);
    for (int l = 1; l < g_cfg.layers; l++) {
      bool fuse_prev = gr_scatter_norm && l > 1;
      bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
      enq_layer_b(l, P, fuse_prev, flush_mlp, base);
      if (prog_total > 0) prog_probe(l, P);
      if (l == 24) phase("L01-24", TP);
      if (phase_prof && getenv("GDEC_PHASE")[0] == '2') {
        char lbuf[16];
        snprintf(lbuf, sizeof lbuf, "L%02d", l);
        phase(lbuf, TP);
      }
    }
    phase("L25-47", TP);
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
#ifdef GDEC_PHASE_PROF
    if (getenv("GDEC_PHASE_PROF")) {
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
  // GDEC_MTP_NORM=full (one 10240 RMSNorm, default) or =branch (4x2560).
  void mtp_input_1(const float* tap) {
    WView et = wview(ckpt.at("embed_tokens.weight"));
    k_q4cp_row<<<1, 128, 0, g_str>>>(et.data + 64,
                                     et.data + 64 + et.rows * et.cols / 2,
                                     (const float*)et.data, d_mtok, et.cols,
                                     et.scale_stride, d_emb);
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
      // H-agg=branch: stream[b] = fc_hidden(norm_h[b]) + e (branch identity kept).
      for (int b = 0; b < 4; b++)
        gemv(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat + (size_t)b * 2560,
             d_mR + (size_t)b * 2560);
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
    if (qsa_dense) {
      k_qsa_step<KVT><<<24, 256, 0, g_str>>>(d_qs, kc, vc, d_attn, d_mpos, 256, 2,
                                             24);
      return;
    }
    k_qsa_flash<true, false, KVT, true><<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
        d_qs, kc, vc, d_attn, 1, 256, 2, 24, d_iselected, 0, d_mpos, d_pacc,
        d_pml);
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, d_attn, 12,
                                                       256, d_mpos);
  }

  // One QSA attention step for the MTP layer (mirrors qsa()).
  void mtp_qsa(const float* x, float* y) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, 1,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               d_mpos, 0, nullptr, d_rdelta);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_minorm + 128, d_miraw,
                                           d_mik, d_mpos, g_cfg.norm_eps,
                                           g_cfg.rope_theta, d_rdelta);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(
          d_iq, d_mik, d_iscores, index_blocks, d_mpos);
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
      k_store_kv<<<2, 256, 0, g_str>>>(d_kb, d_vb, d_mkcf, d_mvcf, d_mpos);
      mtp_qsa_attn(d_mkcf, d_mvcf);
    } else {
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(d_kb, d_vb, d_mkcb, d_mvcb, d_mpos);
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
    CK(hipMemsetAsync(d_moeacc, 0, g_cfg.d * 4, g_str));
    snprintf(buf, sizeof buf, "%s.experts.gate_up_proj.weight", pfx);
    WView gu = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "%s.experts.down_proj.weight", pfx);
    WView dn = wview(ckpt.at(buf));
    const int* topids = d_topids;  // 直读:ids 即专家号,免 D2H 同步
    uint64_t pairs = (uint64_t)k * g_cfg.moe_mid;
    k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
        gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
        x, d_guv, topids, (uint64_t)2 * g_cfg.moe_mid, gu.cols, gu.scale_stride,
        k, 0 /*x_stride*/, 16 /*id_stride*/, k);
    k_silu_mul_g<<<(k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(
        d_guv, d_hid, g_cfg.moe_mid, k);
    uint64_t pairs2 = (uint64_t)k * ((g_cfg.d + 1) / 2);
    k_q4cp_gemv_gd<<<(unsigned)((pairs2 + 15) / 16), 512, 0, g_str>>>(
        dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
        d_hid, d_moeacc, topids, d_topw, (uint64_t)g_cfg.d, dn.cols,
        dn.scale_stride, (uint64_t)g_cfg.moe_mid, k, 16 /*id_stride*/,
        0 /*acc_stride*/, k);
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
  int mtp_step(const float* tap, int tok,
               std::vector<float>* host_logits = nullptr) {
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
    gemv(wview(ckpt.at("lm_head.weight")), d_x, d_logits);
    int id = -1;
    if (host_logits) {
      host_logits->resize(g_cfg.vocab);
      CK(hipMemcpy(host_logits->data(), d_logits, (size_t)g_cfg.vocab * 4,
                   hipMemcpyDeviceToHost));
    } else {
      k_argmax<<<1, 1024, 0, g_str>>>(d_logits, g_cfg.vocab, d_argmax);
      CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    }
    mtp_pos++;
    return id;
  }

  // ---- batch ingest: build MTP KV over a token range (prompt/accepted) ----
  // Row j takes (tap[j] = trunk h_{base+j}, toks[j] = t_{base+j+1}) at MTP
  // position base+j. No mixer/lm_head: only the layer state (KV) is built.
  void mtp_input_b(const float* taps, int P) {
    WView et = wview(ckpt.at("embed_tokens.weight"));
    k_q4cp_row_b<<<P, 128, 0, g_str>>>(et.data + 64,
                                       et.data + 64 + et.rows * et.cols / 2,
                                       (const float*)et.data, d_tokarr, et.cols,
                                       et.scale_stride, d_eb, 2560);
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
      for (int b = 0; b < 4; b++) {
        gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhatb + (size_t)b * 2560,
             d_convoutb, P, 10240 /*xstride*/);
        k_row_branch_copy<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0,
                            g_str>>>(d_convoutb, d_mRb, 2560, b);
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
  void mtp_qsa_b(int P, int base) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    // Table-driven rope (M-RoPE requests carry their 3-row positions in the
    // table; text gets the same absolute-position table, bit-identical to
    // the inline fp64 path it replaces).
    ensure_rope_tab(g_cfg.rope_theta, 64);
    ensure_rope_cs(base, P);
    if (!qsa_dense) {
      if (index_f32) {
        float one = 1, zero = 0;
        auto status =
            rocblas_sgemm(rbh, rocblas_operation_transpose,
                          rocblas_operation_none, 640, P, 2560, &one, d_miw,
                          2560, d_xb, 2560, &zero, d_iproj, 640);
        if (status != rocblas_status_success) {
          fprintf(stderr, "mtp indexer fp32 projection failed: %d\n", (int)status);
          exit(1);
        }
      } else {
        snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
        gemm(wview(ckpt.at(buf)), d_xb, d_iproj, P);
      }
      k_index_q<<<dim3(P, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, P,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               nullptr, base, d_ropecs);
      int nb = (base + P) / 4, b0 = base / 4;
      if (nb > b0)
        k_index_pool<<<nb - b0, 128, 0, g_str>>>(d_iproj, d_minorm + 128, d_mik,
                                                 g_cfg.norm_eps,
                                                 g_cfg.rope_theta, base, d_miraw,
                                                 d_ropecs);
      k_index_ring<<<min(P, 4), 128, 0, g_str>>>(d_iproj, d_miraw, P, base);
      const float one = 1.f, zero = 0.f;
      const int a0 = std::max(2051, base);
      const int grid =
          2051 + (a0 - 2051 + INDEX_BATCH - 1) / INDEX_BATCH * INDEX_BATCH;
      for (int first = a0; first < base + P;) {
        int nxt = first < grid ? min(grid, base + P)
                               : min(first + INDEX_BATCH, base + P);
        int count = nxt - first;
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose,
                                    rocblas_operation_none, nb, count * 4, 128,
                                    &one, d_mik, 128,
                                    d_iq + (size_t)(first - base) * 512, 128,
                                    &zero, d_iscores, nb);
        if (status != rocblas_status_success) {
          fprintf(stderr, "mtp indexer sgemm failed: %d\n", (int)status);
          exit(1);
        }
        index_select(d_iscores, nb, d_iselected + (size_t)first * 512, first,
                     count);
        first = nxt;
      }
    }
    snprintf(buf, sizeof buf, "%s.q_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_qgb, P);
    snprintf(buf, sizeof buf, "%s.k_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_kbb, P);
    snprintf(buf, sizeof buf, "%s.v_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_vbb, P);
    snprintf(buf, sizeof buf, "%s.q_norm.weight", pfx);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "%s.k_norm.weight", pfx);
    float* knw = ensure_norm256(buf);
    k_qsa_qsplit<true><<<dim3(24, P), 256, 0, g_str>>>(d_qgb, qnw, d_qsb, d_gsb,
                                                       256, g_cfg.norm_eps, base,
                                                       d_ropecs);
    k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(d_kbb, knw, d_kbb,
                                                      nullptr, g_cfg.norm_eps,
                                                      base, d_ropecs);
    int dense_P = qsa_dense ? P : min(P, std::max(0, 2051 - base));
    if (qsa_kv_fp32) {
      CK(hipMemcpyAsync(d_mkcf + (size_t)base * 512, d_kbb, (size_t)P * 512 * 4,
                        hipMemcpyDeviceToDevice, g_str));
      CK(hipMemcpyAsync(d_mvcf + (size_t)base * 512, d_vbb, (size_t)P * 512 * 4,
                        hipMemcpyDeviceToDevice, g_str));
      qsa_flash_b(d_mkcf, d_mvcf, P, dense_P, base);
    } else {
      k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256, 0,
                         g_str>>>(d_kbb, d_mkcb + (size_t)base * 512, 512, P,
                                  512);
      k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256, 0,
                         g_str>>>(d_vbb, d_mvcb + (size_t)base * 512, 512, P,
                                  512);
      qsa_flash_b(d_mkcb, d_mvcb, P, dense_P, base);
    }
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
    k_router_topk<<<P, 512, 0, g_str>>>(d_routerb, d_topidsb, d_topwb,
                                        g_cfg.experts, k);
    CK(hipMemsetAsync(d_moeaccb, 0, (size_t)P * g_cfg.d * 4, g_str));
    snprintf(buf, sizeof buf, "%s.experts.gate_up_proj.weight", pfx);
    const Tensor& gu_t = ckpt.at(buf);
    WView gu = wview(gu_t);
    snprintf(buf, sizeof buf, "%s.experts.down_proj.weight", pfx);
    const Tensor& dn_t = ckpt.at(buf);
    WView dn = wview(dn_t);
    moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff, d_pairids,
                    P, k, E);
    int tasks = (npairs + 63) / 64 + E;
    k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E);
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
    CK(hipMemcpyAsync(d_tokarr, toks.data(), P * 4, hipMemcpyHostToDevice,
                      g_str));
    mtp_input_b(taps, P);
    xf16_src = nullptr;
    mtp_gr_read_b("mtp.layers.0.attn_hyper_connection", P);
    mtp_qsa_b(P, base);
    mtp_gr_write_b("mtp.layers.0.attn_hyper_connection", P);
    mtp_gr_read_b("mtp.layers.0.mlp_hyper_connection", P);
    mtp_moe_b(P);
    mtp_gr_write_b("mtp.layers.0.mlp_hyper_connection", P);
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
  // gr_bf16 shims: d_Rb rows are bf16 under GDEC_GR_BF16, but the MTP ingest
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

  // Greedy reduction of the same batched logits.
  void batch_argmax(int P, int* host) {
    batch_logits_device(P);
    for (int r = 0; r < P; r++)
      k_argmax<<<1, 1024, 0, g_str>>>(d_logits_b + (size_t)r * g_cfg.vocab,
                                      g_cfg.vocab, d_argmax_arr + r);
    CK(hipMemcpy(host, d_argmax_arr, P * 4, hipMemcpyDeviceToHost));
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
    std::vector<int> drafts(gamma), v(gamma + 1);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int rounds = 0, rollbacks = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, t_ingest = 0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen) {
      int q = pos - 1;  // trunk processed through q
      int g = std::min(gamma, gen - (int)out.size());
      size_t chist_sz = chist.size();
      mtp_ckpt_save();
      auto t0 = std::chrono::steady_clock::now();
      drafts[0] = mtp_step(d_R, cur);
      // draft 1 used the committed (h_q, cur): its indexer ring rows are
      // truth. Later drafts pollute the ring; save/restore around them.
      CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                        g_str));
      for (int j = 1; j < g; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
      auto t1 = std::chrono::steady_clock::now();
      std::vector<int> vt;
      vt.reserve(g + 1);
      vt.push_back(cur);
      for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
      spec_snap = true;  // capture per-row rollback snapshots during verify
      prefill_batch(vt, q + 1, true, true);
      spec_snap = false;
      batch_argmax(g + 1, v.data());
      auto t2 = std::chrono::steady_clock::now();
      int a = 0;
      while (a < g && drafts[a] == v[a]) a++;
      for (int j = 0; j < g; j++) {
        depth_tot[j]++;
        if (j < a) depth_hits[j]++;
      }
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
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    fprintf(stderr,
            "\nspec: %d tokens in %.2f s = %.1f tok/s | rounds=%d commit/round=%.2f "
            "rollbacks=%d\n",
            (int)out.size(), sec, out.size() / sec, rounds,
            (double)out.size() / rounds, rollbacks);
    fprintf(stderr, "spec depth acc:");
    for (int j = 0; j < gamma && depth_tot[j]; j++)
      fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
    fprintf(stderr,
            "\nspec time: draft=%.0fms verify=%.0fms rollback+reprefill=%.0fms "
            "ingest=%.0fms\n",
            t_draft, t_verify, t_roll, t_ingest);
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
    gamma = std::max(1, std::min(8, gamma));
    std::vector<int> drafts(8), v(9);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int accepted = 0, ng_rounds = 0, mtp_rounds = 0, ng_roll = 0, mtp_roll = 0;
    double verify_ms = 0, replay_ms = 0;
    // ngram verify chunking: GDEC_NGRAM_CHUNK=0 keeps the legacy single-pass
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
      if (chist.size() != (size_t)pos)
        throw std::runtime_error(
            "chain: token history and trunk position disagree");
      ngram.observe(chist);
      // Same budget math as the pure-ngram loop: one slot stays for the
      // target-model continuation, never verify past gen or maxctx.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      std::vector<int> draft =
          want > 0 ? ngram.draft(chist, cur, want) : std::vector<int>{};
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
        int g = std::min({gamma, gen - (int)out.size(), maxctx - pos - 1});
        if (g < 1) break;  // context edge: no room for even one draft
        size_t chist_sz = chist.size();
        mtp_ckpt_save();
        auto t0 = std::chrono::steady_clock::now();
        drafts[0] = mtp_step(d_R, cur);
        // draft 1's indexer ring rows are truth; later drafts pollute them.
        CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        for (int j = 1; j < g; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
        std::vector<int> vt;
        vt.reserve(g + 1);
        vt.push_back(cur);
        for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
        spec_snap = true;  // capture per-row rollback snapshots during verify
        prefill_batch(vt, q + 1, true, true);
        spec_snap = false;
        batch_argmax(g + 1, v.data());
        verify_ms += elapsed(t0);
        int a = 0;
        while (a < g && drafts[a] == v[a]) a++;
        for (int j = 0; j < g; j++) {
          depth_tot[j]++;
          if (j < a) depth_hits[j]++;
        }
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
      for (int j = 0; j < gamma && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr, "\n");
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
    std::vector<int> drafts(gamma);
    std::vector<float> draft_lps(gamma), verify_logits, p_row(g_cfg.vocab);
    std::vector<std::vector<float>> q_rows(gamma);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int rounds = 0, rollbacks = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, t_ingest = 0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen) {
      int q = pos - 1;
      int g = std::min(gamma, gen - (int)out.size());
      size_t chist_sz = chist.size();
      mtp_ckpt_save();
      Sampler draft_sampler = sampler.fork_stream(
          0xd1b54a32d192ed03ull ^ (uint64_t)(q + 1));
      Sampler accept_sampler = sampler.fork_stream(
          0x3c6ef372fe94f82aull ^ (uint64_t)(q + 1));
      auto t0 = std::chrono::steady_clock::now();
      for (int j = 0; j < g; j++) {
        mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, &q_rows[j]);
        draft_sampler.prepare(q_rows[j]);
        float qlp = 0.0f;
        // Penalties are defined over committed/emitted tokens only. Provisional
        // draft tokens therefore do not enter the proposal sampler's history;
        // q_rows[j] records exactly the distribution that produced drafts[j].
        drafts[j] = draft_sampler.draw_prepared(q_rows[j], &qlp, false);
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
      batch_logits(g + 1, verify_logits);

      Sampler target_view = sampler;
      int a = 0;
      for (; a < g; a++) {
        std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                    g_cfg.vocab, p_row.data());
        target_view.prepare(p_row);
        float px = p_row[drafts[a]];
        float qx = q_rows[a][drafts[a]];
        draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
        double threshold = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
        if (accept_sampler.uniform() >= threshold) break;
        target_view.remember(drafts[a]);
      }

      int next;
      float next_lp = 0.0f;
      target_view.rng = sampler.rng;
      if (a < g) {
        next = target_view.sample_residual(p_row, q_rows[a], &next_lp);
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
    for (int j = 0; j < gamma && depth_tot[j]; j++)
      fprintf(stderr, " %d:%.3f", j + 1,
              (double)depth_hits[j] / depth_tot[j]);
    fprintf(stderr,
            "\nspec-sample time: draft=%.0fms verify=%.0fms "
            "rollback=%.0fms ingest=%.0fms\n",
            t_draft, t_verify, t_roll, t_ingest);
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
    gamma = std::max(1, std::min(8, gamma));
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
      if (chist.size() != (size_t)pos)
        throw std::runtime_error(
            "chain-sample: token history and trunk position disagree");
      ngram.observe(chist);
      // Same budget math as the greedy chain: one slot stays for the
      // target-model continuation, never verify past gen or maxctx.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      std::vector<int> draft =
          want > 0 ? ngram.draft(chist, cur, want) : std::vector<int>{};
      bool stop = false;
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
        batch_logits((int)tokens.size(), verify_logits);
        verify_ms += elapsed(t);
        // Walk the draft rows with a scratch sampler view: accept draft[a]
        // iff it equals this round's draw from row a, else the draw is the
        // correction. keep = committed draft count (emit may stop early).
        Sampler view = sampler;
        const size_t committed0 = out.size();
        int a = 0, next = -1;
        float next_lp = 0.0f;
        for (; a < (int)draft.size(); a++) {
          std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                      g_cfg.vocab, p_row.data());
          float lp_a = 0.0f;
          int s = view.sample(p_row, &lp_a);
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
            std::copy_n(verify_logits.data() +
                            (size_t)draft.size() * g_cfg.vocab,
                        g_cfg.vocab, p_row.data());
            next = view.sample(p_row, &next_lp);
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
          batch_logits((int)tokens.size(), verify_logits);
          verify_ms += elapsed(t);
          cont = -1;
          int a = 0;
          for (; a < cs; a++) {
            std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                        g_cfg.vocab, p_row.data());
            float lp_a = 0.0f;
            int s = view.sample(p_row, &lp_a);
            if (s != draft[di + a]) { cont = s; cont_lp = lp_a; break; }
            if (emit && !emit(draft[di + a], lp_a)) { stop = true; break; }
            out.push_back(draft[di + a]);
          }
          const int keep = a;
          if (!stop) {
            if (cont < 0) {
              // Full chunk accept: the bonus row continues the sequence.
              std::copy_n(verify_logits.data() + (size_t)cs * g_cfg.vocab,
                          g_cfg.vocab, p_row.data());
              cont = view.sample(p_row, &cont_lp);
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
        int g = std::min({gamma, gen - (int)out.size(), maxctx - pos - 1});
        if (g < 1) break;  // context edge: no room for even one draft
        size_t chist_sz = chist.size();
        mtp_ckpt_save();
        Sampler draft_sampler = sampler.fork_stream(
            0xd1b54a32d192ed03ull ^ (uint64_t)(q + 1));
        Sampler accept_sampler = sampler.fork_stream(
            0x3c6ef372fe94f82aull ^ (uint64_t)(q + 1));
        auto t = std::chrono::steady_clock::now();
        for (int j = 0; j < g; j++) {
          mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, &q_rows[j]);
          draft_sampler.prepare(q_rows[j]);
          // Provisional drafts stay out of the penalty history (as
          // spec_loop_sample): q_rows[j] records the producing distribution.
          drafts[j] =
              draft_sampler.draw_prepared(q_rows[j], &draft_lps[j], false);
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
        batch_logits(g + 1, verify_logits);
        verify_ms += elapsed(t);
        Sampler target_view = sampler;
        int a = 0;
        for (; a < g; a++) {
          std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                      g_cfg.vocab, p_row.data());
          target_view.prepare(p_row);
          float px = p_row[drafts[a]];
          float qx = q_rows[a][drafts[a]];
          draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
          double threshold = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
          if (accept_sampler.uniform() >= threshold) break;
          target_view.remember(drafts[a]);
        }
        int next;
        float next_lp = 0.0f;
        target_view.rng = sampler.rng;
        if (a < g) {
          next = target_view.sample_residual(p_row, q_rows[a], &next_lp);
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
      for (int j = 0; j < gamma && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1,
                (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr, "\n");
    }
    return out;
  }

  // --- decode graphs: G1 = token upload + embed + L0; G2 = [PLE] + L1..end --
  hipGraphExec_t g1_exec = nullptr, g2_exec = nullptr;
  bool graphs_ok = false;
  void build_graphs() {
    if (dbg || perf || getenv("GDEC_NOGRAPH")) return;
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
    CK(hipGraphInstantiate(&g1_exec, g1, nullptr, nullptr, 0));
    CK(hipGraphInstantiate(&g2_exec, g2, nullptr, nullptr, 0));
    CK(hipStreamDestroy(cap));
    graphs_ok = true;
    fprintf(stderr, "decode graphs captured (ple=%d)\n", (int)ple_on);
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
  }

  int forward(int token) {
    if (graphs_ok) {
      *p_token = token;
      auto g0 = std::chrono::steady_clock::now();
      CK(hipGraphLaunch(g1_exec, 0));
      dump_decode_step(pos);  // debug only (GDEC_MROPE_DUMP); syncs the stream
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
      if (getenv("GDEC_TIME")) {
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
    dump_decode_step(pos);  // debug only (GDEC_MROPE_DUMP); syncs the stream
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
    if (getenv("GDEC_TIME")) {
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
  if (t < dh) gs[h * dh + t] = qg[h * 2 * dh + dh + t];
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
// grid = dim3(2, P); kc (when Bf16) already includes the base*512 offset.
template <bool Bf16>
__global__ void k_qsa_kprep(const float* __restrict__ kb, const float* __restrict__ knw,
                            float* __restrict__ kout, uint16_t* __restrict__ kc,
                            float eps, int rope_base,
                            const float2* __restrict__ ropecs) {
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
