// ---------------- main -------------------------------------------------------
// Pure GGUF start (G3.3): `qwenox <first shard>.gguf ...` needs no .hgn. Implies
// QWENOX_GGUF=<base> + QWENOX_GGUF_DENSE=1; the MTP sidecar defaults to the single
// mtp-*.gguf next to the shards (QWENOX_GGUF_MTP= empty = none).
static bool is_gguf_path(const std::string& p) {
  return p.size() > 5 && p.compare(p.size() - 5, 5, ".gguf") == 0;
}
static void gguf_base_setup(const std::string& base) {
#ifdef _WIN32
  _putenv_s("QWENOX_GGUF", base.c_str());
  _putenv_s("QWENOX_GGUF_DENSE", "1");
#else
  setenv("QWENOX_GGUF", base.c_str(), 1);
  setenv("QWENOX_GGUF_DENSE", "1", 1);
  if (!getenv("QWENOX_GGUF_MTP")) {
    const size_t sl = base.find_last_of('/');
    const std::string dir = sl == std::string::npos ? "." : base.substr(0, sl);
    std::string hit;
    int n = 0;
    if (DIR* d = opendir(dir.c_str())) {
      while (dirent* e = readdir(d)) {
        const std::string nm = e->d_name;
        if (nm.compare(0, 4, "mtp-") == 0 && is_gguf_path(nm)) hit = dir + "/" + nm, n++;
      }
      closedir(d);
    }
    if (n == 1) setenv("QWENOX_GGUF_MTP", hit.c_str(), 1);
    else fprintf(stderr, "gguf: %s MTP sidecar (mtp-*.gguf) in %s; set QWENOX_GGUF_MTP\n",
                 n ? "several" : "no", dir.c_str());
  }
#endif
  fprintf(stderr, "gguf: pure GGUF start from %s (MTP %s)\n", base.c_str(),
          getenv("QWENOX_GGUF_MTP") && *getenv("QWENOX_GGUF_MTP") ? getenv("QWENOX_GGUF_MTP") : "none");
}
// Base checkpoint: the .hgn file, or empty when the base is a GGUF (tensors
// come from gguf_dense_apply).
static std::unique_ptr<Checkpoint> open_base(const std::string& base) {
  if (is_gguf_path(base)) return std::make_unique<Checkpoint>();
  auto ck = std::make_unique<Checkpoint>(base.c_str());
  ple_uring_open(base.c_str());
  return ck;
}

// Base-file lm_head record, captured before overlays are applied (the draft
// lm_head override in GpuModel::setup_draft_lmhead). The record points into
// the base file's mmap, which outlives the model (ckbase is main-scoped).
static Tensor g_lmhead_base;
static bool g_lmhead_base_ok = false;
static void lmhead_base_capture(const Checkpoint& ck) {
  if (const Tensor* t = ck.find("lm_head.weight")) {
    g_lmhead_base = *t;
    g_lmhead_base_ok = true;
  }
}

#if !defined(QWENOX_NO_MAIN)
int main(int argc, char** argv) {
#ifdef _WIN32
  if (!wsa_init()) {
    fprintf(stderr, "WSAStartup failed\n");
    return 1;
  }
#endif
  if (argc < 2) {
    fprintf(stderr,
            "usage: qwenox <base.hgn> [overlay.hgn ...] --tokens 1,2,3 [--gen N] [--dump] "
            "[--no-ple] [--maxctx N]\n"
            "       qwenox <base.hgn> [overlay.hgn ...] --tokens-file IDS --ppl "
            "[--ppl-window N --ppl-stride N] [--maxctx N]\n"
            "       qwenox <base.hgn> [overlay.hgn ...] --kld-base REF.kld "
            "[--kld-chunks N] [--kld-save OUT]   (llama.cpp KLD file)\n"
            "       qwenox <base.hgn> [overlay.hgn ...] --tokens-file IDS --kld-save OUT "
            "[--kld-ctx 512] [--kld-chunks N]\n"
            "       qwenox <base.hgn> [overlay.hgn ...] --serve [--host IPv4] [--port N] [--maxctx N] "
            "[--no-ple]\n"
            "       Serve defaults: --host 127.0.0.1 --port 8730 (IPv4)\n"
            "       M-RoPE: [--mrope-grid t,h,w ...] with --tokens; "
            "--mrope-test K dumps rope tables/decode positions without weights "
            "(QWENOX_MROPE_DUMP=<file>)\n"
            "       YaRN: [--rope-factor F] [--rope-original-ctx N] "
            "[--rope-beta-fast F] [--rope-beta-slow F] [--rope-attn-scale F]\n"
            "       Vision: --vision-tower <sidecar.hgn> (serve: GEN suffix "
            "VIMG k + binary patch frames); --vision-test PREFIX "
            "--vision-tower S --patches x.npy --mrope-grid t,h,w (offline "
            "ViT parity dumps, no main checkpoint)\n");
    return 2;
  }
  std::string base = argv[1];
  std::vector<std::string> overlays;  // applied in order; later files win
  std::vector<int> tokens;
  int gen = 8, argi = 2, maxctx = 4096;
  std::vector<int> splits;  // --split K1,K2: KV-continuation self-test
  bool dump = false, ple_on = true, show_top = false, ppl = false;
  bool ppl_window_set = false, ppl_stride_set = false;
  bool serve_mode = false, maxctx_set = false;
  int port = 8730, mtp_test = 0, spec_gen = 0, gamma = 7;
  std::string engine_host = engine_net::kDefaultHost;
  int ppl_window = 0, ppl_stride = 0;
  kld::Opts kld_opts;  // --kld-*: KL divergence vs a llama.cpp logits file
  // M-RoPE: per-image grid triples (repeatable), rope-only test mode.
  std::vector<std::array<int, 3>> mrope_grids;
  int mrope_test = -1;  // -1 off; else number of decode steps to dump
  // Vision (stage 4): sidecar tower weights; offline ViT parity mode.
  std::string vision_tower, vision_test, vision_patches;
  std::string rope_error;
  if (!rope_from_env(g_cfg.rope, rope_error)) {
    fprintf(stderr, "%s\n", rope_error.c_str());
    return 2;
  }
  while (argi < argc && argv[argi][0] != '-') overlays.push_back(argv[argi++]);
  if (is_gguf_path(base)) gguf_base_setup(base);
  for (; argi < argc; argi++) {
    std::string a = argv[argi];
    auto next = [&]() { return argv[++argi]; };
    if (a == "--tokens") {
      char* s = strdup(next());
      for (char* p = strtok(s, ","); p; p = strtok(nullptr, ",")) tokens.push_back(atoi(p));
      free(s);
    } else if (a == "--tokens-file") {
      // one token id per line or comma-separated; for long prompts (ARG_MAX limit)
      std::ifstream f(next());
      std::string line;
      while (std::getline(f, line)) {
        char* s = strdup(line.c_str());
        for (char* p = strtok(s, ", \t"); p; p = strtok(nullptr, ", \t"))
          tokens.push_back(atoi(p));
        free(s);
      }
    } else if (a == "--gen")
      gen = atoi(next());
    else if (a == "--split") {
      char* s = strdup(next());
      for (char* p = strtok(s, ","); p; p = strtok(nullptr, ","))
        splits.push_back(atoi(p));
      free(s);
    } else if (a == "--dump")
      dump = true;
    else if (a == "--ppl")
      ppl = true;
    else if (a == "--ppl-window") {
      ppl_window = atoi(next());
      ppl_window_set = true;
    } else if (a == "--ppl-stride") {
      ppl_stride = atoi(next());
      ppl_stride_set = true;
    } else if (a == "--kld-base")
      kld_opts.base = next();
    else if (a == "--kld-save")
      kld_opts.save = next();
    else if (a == "--kld-chunks")
      kld_opts.chunks = atoi(next());
    else if (a == "--kld-ctx")
      kld_opts.ctx = atoi(next());
    else if (a == "--no-ple")
      ple_on = false;
    else if (a == "--show")
      show_top = true;
    else if (a == "--serve")
      serve_mode = true;
    else if (a == "--host") {
      if (argi + 1 >= argc || argv[argi + 1][0] == '-') {
        fprintf(stderr, "--host requires an IPv4 address\n");
        return 2;
      }
      engine_host = next();
    } else if (a == "--port")
      port = atoi(next());
    else if (a == "--maxctx") {
      maxctx = atoi(next());
      maxctx_set = true;
    } else if (const char* key = rope_cli_key(a)) {
      if (argi + 1 >= argc || !rope_set(g_cfg.rope, key, next())) {
        fprintf(stderr, "%s requires a valid numeric value\n", a.c_str());
        return 2;
      }
    } else if (a == "--mtp-test")
      mtp_test = atoi(next());
    else if (a == "--spec-gen")
      spec_gen = atoi(next());
    else if (a == "--gamma")
      gamma = atoi(next());
    else if (a == "--mrope-grid") {
      char* s = strdup(next());
      std::array<int, 3> g{0, 0, 0};
      int k = 0;
      for (char* p = strtok(s, ","); p && k < 3; p = strtok(nullptr, ","))
        g[k++] = atoi(p);
      free(s);
      if (k != 3) {
        fprintf(stderr, "--mrope-grid expects t,h,w\n");
        return 2;
      }
      mrope_grids.push_back(g);
    } else if (a == "--mrope-test")
      mrope_test = atoi(next());
    else if (a == "--vision-tower")
      vision_tower = next();
    else if (a == "--vision-test")
      vision_test = next();
    else if (a == "--patches")
      vision_patches = next();
  }
  if (serve_mode) {
    sockaddr_in listen_addr{};
    if (!engine_net::address(engine_host, port, &listen_addr)) {
      fprintf(stderr, "invalid engine listen address %s:%d (expected IPv4)\n",
              engine_host.c_str(), port);
      return 2;
    }
  }
  if (!rope_config_valid(g_cfg.rope)) {
    fprintf(stderr, "invalid YaRN config: factor>=1, original_ctx>0, beta_fast>=beta_slow>0, "
                    "attention_scale>=0\n");
    return 2;
  }
  // vLLM-style consistency check: per-sequence context must stay within the
  // position-encoding range. factor>1: hard limit factor x original_ctx;
  // factor==1 beyond native: warn only (works, but unvalidated positions).
  const double rope_cap =
      g_cfg.rope.factor * g_cfg.rope.original_max_position_embeddings;
  if (g_cfg.rope.factor > 1.0 && maxctx > rope_cap) {
    fprintf(stderr, "maxctx %d exceeds YaRN range factor %.6g x original_ctx %d "
                    "= %.0f; lower --maxctx or raise --rope-factor\n",
            maxctx, g_cfg.rope.factor, g_cfg.rope.original_max_position_embeddings,
            rope_cap);
    return 2;
  }
  if (g_cfg.rope.factor == 1.0 &&
      maxctx > g_cfg.rope.original_max_position_embeddings)
    fprintf(stderr, "warning: maxctx %d exceeds native context %d without YaRN; "
                    "positions beyond it are unvalidated (set --rope-factor to "
                    "extend)\n",
            maxctx, g_cfg.rope.original_max_position_embeddings);
  if (g_cfg.rope.factor > 1.0)
    fprintf(stderr, "RoPE: YaRN factor=%.6g original_ctx=%d beta_fast=%.6g "
                    "beta_slow=%.6g attention_scale=%.6g\n",
            g_cfg.rope.factor, g_cfg.rope.original_max_position_embeddings,
            g_cfg.rope.beta_fast, g_cfg.rope.beta_slow,
            rope_attention_factor(g_cfg.rope));
  if (ppl && (serve_mode || !vision_test.empty() || mtp_test > 0 || spec_gen > 0 ||
              !mrope_grids.empty() || dump || !splits.empty())) {
    fprintf(stderr, "--ppl is text-only and cannot be combined with serve, vision, "
                    "MTP, M-RoPE, dump or split modes\n");
    return 2;
  }
  const bool kld_mode = !kld_opts.base.empty() || !kld_opts.save.empty();
  if (kld_mode && (ppl || serve_mode || !vision_test.empty() || mtp_test > 0 ||
                   spec_gen > 0 || !mrope_grids.empty() || dump || !splits.empty() ||
                   mrope_test >= 0)) {
    fprintf(stderr, "--kld-* is text-only and cannot be combined with ppl, serve, "
                    "vision, MTP, M-RoPE, dump or split modes\n");
    return 2;
  }
  if (kld_mode && kld_opts.base.empty() && tokens.empty()) {
    fprintf(stderr, "--kld-save without --kld-base needs --tokens-file\n");
    return 2;
  }
  if ((ppl_window_set || ppl_stride_set) &&
      (!ppl || !ppl_window_set || !ppl_stride_set || ppl_window < 2 ||
       ppl_stride < 1 || ppl_stride >= ppl_window || ppl_window > maxctx)) {
    fprintf(stderr, "--ppl-window and --ppl-stride require --ppl and "
                    "1 <= stride < window <= maxctx\n");
    return 2;
  }
  if (!vision_test.empty()) {
    // Offline ViT parity mode (stage 4a): runs the INTEGRATED vision tower
    // standalone — no main checkpoint, no arena — and dumps vit.cpp-
    // compatible layers under <prefix>.layers/ for tools/vit_cmp.py.
    //   qwenox <ignored-> --vision-test PREFIX --vision-tower S.hgn
    //        --patches x.npy --mrope-grid t,h,w
    if (vision_tower.empty() || vision_patches.empty() ||
        mrope_grids.empty()) {
      fprintf(stderr, "--vision-test needs --vision-tower, --patches and "
              "--mrope-grid t,h,w\n");
      return 2;
    }
    auto g = mrope_grids[0];
    VitNpy patches = vit_npy_read(vision_patches.c_str());
    if (patches.shape.size() != 2 || patches.shape[1] != 1536)
      throw std::runtime_error("patches npy must be <f4 (P,1536)");
    int P = (int)patches.shape[0];
    if (g[0] != 1 || g[1] * g[2] != P) {
      fprintf(stderr, "--vision-test: grid %d,%d,%d vs patches P=%d\n", g[0],
              g[1], g[2], P);
      return 2;
    }
    CK(hipSetDevice(0));
    hipblasLtHandle_t vlth;
    rocblas_handle vrbh;
    hipblasLtCreate(&vlth);
    rocblas_create_handle(&vrbh);
    void* vws = nullptr;
    const size_t vws_bytes = (size_t)64 << 20;
    CK(hipMalloc(&vws, vws_bytes));
    VisionTower vt;
    vt.lth = vlth;
    vt.rbh = vrbh;
    vt.d_ltws = vws;
    vt.lt_ws = vws_bytes;
    auto t0 = std::chrono::steady_clock::now();
    vt.load(vision_tower.c_str());
    int M = vt.forward((const float*)patches.data.data(), P, g[1], g[2],
                       vision_test.c_str());
    CK(hipStreamSynchronize(g_str));
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    if (M < 0) return 1;
    fprintf(stderr, "vision-test: P=%d M=%d total %.0f ms (load+forward, "
            "dumps at %s.layers/)\n", P, M, ms, vision_test.c_str());
    return 0;
  }
  if (mrope_test >= 0) {
    // M-RoPE rope-only verification (stage 1b/1c): drives the cos/sin table
    // fill and the decode rope position directly — no weight arena, no layer
    // forward. A second full engine cannot run next to the live 256K service
    // (the global pinned-page ceiling rejects a second 65 GiB expert
    // registration, tools/probe_reg3.cu),
    // and none of the rope machinery touches weights. Use with
    // QWENOX_MROPE_DUMP=<file> and compare via tools/cmp_mrope.py.
    Checkpoint ckpt(base.c_str());
    for (const auto& o : overlays)
      if (const char* why = ckpt.add_overlay(o.c_str()))
        fprintf(stderr, "overlay %s: %s, skipped\n", o.c_str(), why);
    if (tokens.empty()) {
      fprintf(stderr, "need --tokens\n");
      return 2;
    }
    GpuModel m(ckpt, maxctx);
    if (!mrope_grids.empty() && !m.set_mrope(tokens, mrope_grids)) {
      fprintf(stderr, "mrope: grids do not match the image token runs\n");
      return 1;
    }
    const int N = (int)tokens.size();
    ensure_rope_tab(g_cfg.rope_theta, 64);
    m.ensure_rope_cs(0, N);
    for (int s = 0; s < mrope_test; s++) {
      int p = N + s;
      CK(hipMemcpy(m.d_pos, &p, 4, hipMemcpyHostToDevice));
      m.dump_decode_step(p);
    }
    // k_rope decode kernel probe: a deterministic vector rotated in place at
    // position N (+ rope_delta); bitwise-comparable across binaries.
    if (FILE* f = m.mrope_dump()) {
      std::vector<float> hv(24 * 256);
      for (size_t i = 0; i < hv.size(); i++)
        hv[i] = (float)(int)(((uint32_t)i * 2654435761u) >> 9) / 8388608.f - 1.f;
      float* d_vp = nullptr;
      CK(hipMalloc(&d_vp, hv.size() * 4));
      CK(hipMemcpy(d_vp, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(m.d_pos, &N, 4, hipMemcpyHostToDevice));
      k_rope<<<(24 * 32 + 127) / 128, 128>>>(d_vp, 24, 256, 64, m.d_pos,
                                             g_cfg.rope_theta, m.d_rdelta);
      CK(hipMemcpy(hv.data(), d_vp, hv.size() * 4, hipMemcpyDeviceToHost));
      CK(hipFree(d_vp));
      for (int h = 0; h < 24; h++) {
        fprintf(f, "KROPE %d", h);
        for (int i = 0; i < 64; i++)
          fprintf(f, " %a", (double)hv[(size_t)h * 256 + i]);
        fprintf(f, "\n");
      }
      fflush(f);
    }
    return 0;
  }
  if (serve_mode) {
    if (!maxctx_set) maxctx = 65536;  // service default; CLI one-shot stays 4096
    auto ckbase = open_base(base);
    Checkpoint& ckpt = *ckbase;
    lmhead_base_capture(ckpt);
    for (const auto& o : overlays)
      if (const char* why = ckpt.add_overlay(o.c_str()))
        fprintf(stderr, "overlay %s: %s, skipped\n", o.c_str(), why);
    fprintf(stderr, "loaded %zu tensors (base+%zu overlay)\n", ckpt.tensor_count(),
            overlays.size());
    gguf_dense_apply(ckpt);
    g_cfg.vocab = (int)ckpt.at("lm_head.weight").dims[0];
    memstats_capture_hip_baseline();
    // D1a：并发 serve 用更小的 prefill 分段，缩短 D1b 层间让出的最长停顿。
    // 必须在 devarena_init 之前（arena sizing 与 GpuModel 共用 eff_maxbatch）。
    if (eff_parallel() > 1) {
      if (const char* e = getenv("QWENOX_CONC_PREFILL_CHUNK")) {
        int v = atoi(e);
        if (v > 0) {
          // 只缩不放：大于单路分段时沿用单路值（不放大工作区，也不拉长让出间隔）。
          const int single = eff_maxbatch(maxctx);
          g_maxbatch_override = std::min(v, single);
          fprintf(stderr, "qwenox: parallel %d -> prefill chunk %d (QWENOX_CONC_PREFILL_CHUNK=%d%s)\n",
                  eff_parallel(), eff_maxbatch(maxctx), v,
                  v > single ? ", capped by single-stream chunk" : "");
        }
      }
    }
    devarena_init(ckpt, maxctx);
    load_arena(ckpt);
    v2_init(ckpt);
    GpuModel m(ckpt, maxctx);
    m.setup_draft_lmhead(g_lmhead_base_ok ? &g_lmhead_base : nullptr);
    gguf_dense_release(ckpt);
    m.ple_on = ple_on;
    m.build_graphs();
    if (!vision_tower.empty()) m.load_vision(vision_tower.c_str());
    if (!getenv("QWENOX_NOWARMUP")) {
      // one dummy chunk moves first-call costs (Tensile library load,
      // hipBLASLt algo selection, kernel module loads) out of the first
      // served prefill; reset_state is the KV-split self-test path, proven
      // output-identical to a fresh model.
      std::vector<int> warm((size_t)m.maxbatch, 0);
      m.prefill_batch(warm);
      m.reset_state();
    }
    memstats_log("ready");
    // kvsnap cache identity: the weights (and the vision tower for image
    // prompts) that produced the cached KV
    kvsnap::g_model_files = overlays;
    kvsnap::g_model_files.insert(kvsnap::g_model_files.begin(), base);
    if (g_gguf)  // QWENOX_GGUF experts: every shard is part of the identity
      for (const auto& mp : g_gguf->mappings()) kvsnap::g_model_files.push_back(mp.path);
    if (g_gguf_dense) {  // same files as the G2 hybrid, different weights
      kvsnap::g_model_files.push_back("@gguf-dense");
      if (g_gguf_mtp)
        for (const auto& mp : g_gguf_mtp->mappings()) kvsnap::g_model_files.push_back(mp.path);
    }
    if (gg_mtp_on()) {  // MTP routed experts from the sidecar (MTP KV differs)
      kvsnap::g_model_files.push_back("@gguf-mtp-experts");
      if (!g_gguf_dense)
        for (const auto& mp : g_gguf_mtp->mappings()) kvsnap::g_model_files.push_back(mp.path);
    }
    kvsnap::g_vision_file = vision_tower;
    return serve::run(m, engine_host, port);
  }
  if (!kld_mode && tokens.size() < (ppl ? 2u : 1u)) {
    fprintf(stderr, ppl ? "--ppl needs at least two tokens\n" : "need --tokens\n");
    return 2;
  }
  // --gamma 0 = adaptive (Model::GammaCtl, same as the API auto path)
  if (gamma < 0) gamma = 0;
  if (gamma > 8) gamma = 8;  // spec workspace holds at most 8 draft rows
  int need = ppl ? 0 : mtp_test > 0 ? mtp_test : spec_gen > 0 ? spec_gen + (gamma ? gamma : 8) : gen;
  if (!kld_mode && ppl_window == 0 && tokens.size() + (size_t)need > (size_t)maxctx) {
    fprintf(stderr, "tokens+gen exceed maxctx\n");
    return 2;
  }

  auto ckbase = open_base(base);
  Checkpoint& ckpt = *ckbase;
  lmhead_base_capture(ckpt);
  for (const auto& o : overlays)
    if (const char* why = ckpt.add_overlay(o.c_str()))
      fprintf(stderr, "overlay %s: %s, skipped\n", o.c_str(), why);
  fprintf(stderr, "loaded %zu tensors (base+%zu overlay)\n", ckpt.tensor_count(),
          overlays.size());
  gguf_dense_apply(ckpt);
  g_cfg.vocab = (int)ckpt.at("lm_head.weight").dims[0];
  if (ppl) {
    for (size_t i = 0; i < tokens.size(); i++) {
      if (tokens[i] < 0 || tokens[i] >= g_cfg.vocab) {
        fprintf(stderr, "--ppl token %zu is out of vocabulary: %d\n", i, tokens[i]);
        return 2;
      }
    }
  }

  memstats_capture_hip_baseline();
  devarena_init(ckpt, maxctx);
  load_arena(ckpt);
  v2_init(ckpt);

  GpuModel m(ckpt, maxctx);
  m.setup_draft_lmhead(g_lmhead_base_ok ? &g_lmhead_base : nullptr);
  gguf_dense_release(ckpt);
  m.ple_on = ple_on;
  m.build_graphs();
  if (mtp_test > 0 || spec_gen > 0) {
    if (mtp_test > 0 && !mrope_grids.empty()) {
      fprintf(stderr, "mrope: mtp-test not supported (acceptance harness "
              "measures text only)\n");
      return 1;
    }
    if (!m.mtp_avail) {
      fprintf(stderr, "mtp: no usable mtp.* weights in checkpoint\n");
      return 1;
    }
    m.mtp_tap_capture = true;  // prefill_chunk records per-row trunk streams
  }

  std::vector<int> all = tokens;
  std::vector<float> logits;

  if (!getenv("QWENOX_NOWARMUP") && mtp_test == 0 && spec_gen == 0) {
    // warmup: one dummy chunk prefill, then reset (see serve path above).
    // Skipped for mtp/spec modes: tap capture would ingest warmup rows.
    std::vector<int> warm((size_t)m.maxbatch, 0);
    m.prefill_batch(warm);
    m.reset_state();
  }
  if (!mrope_grids.empty() && !m.set_mrope(tokens, mrope_grids)) {
    fprintf(stderr, "mrope: grids do not match the image token runs\n");
    return 1;
  }

  if (kld_mode) return kld::run(m, tokens, kld_opts);

  if (ppl) {
    double total_nll = 0.0;
    size_t total_scored = 0;
    printf("ppl_token\tpos\tnext_id\tnll\n");
    auto report = [&](size_t pos, double nll) {
      if (!std::isfinite(nll)) {
        fprintf(stderr, "--ppl non-finite NLL at pos %zu\n", pos);
        exit(1);
      }
      total_nll += nll;
      total_scored++;
      printf("ppl_token\t%zu\t%d\t%.9g\n", pos, tokens[pos + 1], nll);
    };
    if (getenv("QWENOX_NOPREFILLBATCH") && ppl_window == 0) {
      for (size_t pos = 0; pos < tokens.size(); pos++) {
        m.forward(tokens[pos]);
        if (pos + 1 == tokens.size()) break;
        std::vector<float> row;
        m.get_logits(row);
        double maximum = *std::max_element(row.begin(), row.end());
        double sum = 0.0;
        for (float logit : row) sum += std::exp((double)logit - maximum);
        report(pos, maximum + std::log(sum) - row[tokens[pos + 1]]);
      }
    } else {
      if (ppl_window && getenv("QWENOX_NOPREFILLBATCH")) {
        fprintf(stderr, "--ppl-window does not support QWENOX_NOPREFILLBATCH\n");
        return 2;
      }
      size_t previous_end = 0;
      size_t end = ppl_window ? std::min(tokens.size(), (size_t)ppl_window)
                              : tokens.size();
      while (previous_end < tokens.size()) {
        size_t begin = ppl_window && end > (size_t)ppl_window
                           ? end - ppl_window : 0;
        if (previous_end) m.reset_state();
        int window_size = (int)(end - begin);
        for (int off = 0; off < window_size;) {
          int remaining = window_size - off;
          int count = std::min(m.maxbatch, remaining);
          if (remaining > m.maxbatch && remaining <= m.maxbatch_cap)
            count = remaining;
          m.prefill_chunk(std::vector<int>(tokens.begin() + begin + off,
                                           tokens.begin() + begin + off + count), off);
          size_t first_target = std::max((size_t)1, previous_end);
          int first = first_target > begin + off + 1
                          ? (int)(first_target - begin - off - 1) : 0;
          int last = std::min(count, window_size - off - 1);
          int scored = std::max(0, last - first);
          if (scored) {
            std::vector<float> losses;
            m.batch_nll(count, first, scored,
                        tokens.data() + begin + off + first + 1, losses);
            for (size_t row = 0; row < losses.size(); row++)
              report(begin + off + first + row, losses[row]);
          }
          off += count;
        }
        previous_end = end;
        if (!ppl_window) break;
        end = std::min(tokens.size(), end + (size_t)ppl_stride);
      }
    }
    if (total_scored != tokens.size() - 1) {
      fprintf(stderr, "--ppl scored %zu of %zu target tokens\n",
              total_scored, tokens.size() - 1);
      return 1;
    }
    double mean_nll = total_nll / total_scored;
    printf("ppl_summary\tcount=%zu\tmean_nll=%.10f\tppl=%.10f\n",
           total_scored, mean_nll, std::exp(mean_nll));
    return 0;
  }

  auto dump_pos = [&](const std::vector<float>& lg, int i, int next_tok) {
    int V = g_cfg.vocab;
    double mx = -1e300;
    for (int j = 0; j < V; j++) mx = std::max(mx, (double)lg[j]);
    double se = 0;
    for (int j = 0; j < V; j++) se += exp(lg[j] - mx);
    double lse = mx + log(se);
    std::vector<int> idx(V);
    for (int j = 0; j < V; j++) idx[j] = j;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    printf("pos\t%d\t", i);
    for (int k = 0; k < 5; k++)
      printf("%s%d,%.5f", k ? ";" : "", idx[k], lg[idx[k]] - (float)lse);
    if (next_tok >= 0) printf("\tnext=%.5f", lg[next_tok] - (float)lse);
    printf("\n");
    fflush(stdout);
  };

  int next_id = -1;
  if (getenv("QWENOX_NOPREFILLBATCH")) {
    for (size_t i = 0; i < tokens.size(); i++) {
      next_id = m.forward(tokens[i]);
      if (dump) {
        m.get_logits(logits);
        dump_pos(logits, (int)i, i + 1 < tokens.size() ? tokens[i + 1] : -1);
      }
      fprintf(stderr, "\rprefill %zu/%zu", i + 1, tokens.size());
      fflush(stderr);
    }
    fprintf(stderr, "\n");
  } else {
    next_id = m.prefill_batch(tokens);
    if (dump) {  // per-position prefill dump unavailable; dump last position
      m.get_logits(logits);
      dump_pos(logits, (int)tokens.size() - 1, -1);
    }
  }

  // QWENOX_VBENCH=P1,P2,...（D0-M2）：prompt prefill 之后，用 prompt 开头的
  // 连续真实 token 片段跑 prefill_batch(P, base=N) 各 20 次取中位数，
  // 量 P 行成本曲线与 P>8 / P>16 悬崖（verify 路径同款参数 final/quiet，
  // 不捕 tap、不做快照）。跑完直接退出。
  // 项写成 PxS（如 8x4）：P 行由 S 段各 P/S 个连续 token 拼成，S 段取自 prompt 里
  // 均匀分开的位置，专家分布近似 S 个不同序列（D2 合批估算用，tools/d2_gate_check.sh）。
  if (const char* vb = getenv("QWENOX_VBENCH")) {
    m.mtp_tap_capture = false;
    std::vector<std::pair<int, int>> Ps;
    for (const char* p = vb;;) {
      char* q = nullptr;
      int P = (int)strtol(p, &q, 10), S = 1;
      if (q && *q == 'x') S = std::max(1, atoi(q + 1));
      Ps.push_back({P, S});
      const char* c = strchr(p, ',');
      if (!c) break;
      p = c + 1;
    }
    for (auto [P, S] : Ps) {
      if (P < 1 || (size_t)P > tokens.size() || P % S) {
        fprintf(stderr, "vbench P=%d segs=%d skip（越界或不整除）\n", P, S);
        continue;
      }
      std::vector<int> slice;
      for (int s = 0; s < S; s++) {
        size_t o = tokens.size() / S * s;
        slice.insert(slice.end(), tokens.begin() + o, tokens.begin() + o + P / S);
      }
      std::vector<double> ts;
      for (int r = 0; r < 20; r++) {
        auto t0 = std::chrono::steady_clock::now();
        m.prefill_batch(slice, (int)tokens.size(), true, true);
        CK(hipDeviceSynchronize());
        ts.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
      }
      std::sort(ts.begin(), ts.end());
      if (S > 1)
        fprintf(stderr, "vbench P=%d segs=%d median=%.2f ms min=%.2f max=%.2f\n", P, S,
                ts[ts.size() / 2], ts.front(), ts.back());
      else
        fprintf(stderr, "vbench P=%d median=%.2f ms min=%.2f max=%.2f\n", P,
                ts[ts.size() / 2], ts.front(), ts.back());
    }
    return 0;
  }

  if (mtp_test > 0) {
    m.mtp_accept_test(tokens, next_id, mtp_test);
    return 0;
  }
  if (spec_gen > 0) {
    m.mtp_tap_capture = false;  // verify/re-prefill must NOT capture taps
    // (MTP prompt ingest already happened per-chunk inside prefill_batch.)
    if (getenv("QWENOX_NOPREFILLBATCH")) {
      fprintf(stderr, "spec-gen: QWENOX_NOPREFILLBATCH unsupported (no tap "
              "capture without prefill_batch)\n");
      return 1;
    }
    // Offline test hooks for the API decode paths:
    //   QWENOX_SPEC_CHAIN=1   chain drafter (ngram first, MTP fallback), the
    //                       API default
    //   QWENOX_SPEC_SAMPLE=temp,top_k,top_p,seed[,presence,frequency]
    //                       rejection-sampled spec
    // The first token (next_id) stays the prefill argmax.
    //   QWENOX_SPEC_FORCE=<file>  greedy MTP only: verify against these ids
    //                       (whitespace-separated; the tokens AFTER the
    //                       prefill argmax, e.g. a plain --gen run's ids
    //                       minus prompt and first token). All γ policies
    //                       then commit the same ids (tools/gamma_force_check.sh).
    const bool chain = getenv("QWENOX_SPEC_CHAIN") && atoi(getenv("QWENOX_SPEC_CHAIN"));
    if (const char* ff = getenv("QWENOX_SPEC_FORCE")) {
      if (chain || getenv("QWENOX_SPEC_SAMPLE")) {
        fprintf(stderr, "QWENOX_SPEC_FORCE: greedy MTP spec only (no CHAIN / SAMPLE)\n");
        return 2;
      }
      FILE* f = fopen(ff, "r");
      if (!f) { fprintf(stderr, "QWENOX_SPEC_FORCE: cannot open %s\n", ff); return 2; }
      for (int t; fscanf(f, "%d", &t) == 1;) m.spec_force.push_back(t);
      fclose(f);
      if ((int)m.spec_force.size() < spec_gen) {
        fprintf(stderr, "QWENOX_SPEC_FORCE: %zu ids < --spec-gen %d\n", m.spec_force.size(), spec_gen);
        return 2;
      }
      fprintf(stderr, "spec-gen force: %zu ids from %s\n", m.spec_force.size(), ff);
    }
    std::vector<int> spec_out;
    if (const char* ss = getenv("QWENOX_SPEC_SAMPLE")) {
      serve::HostSampler smp;
      smp.temp = 1.0f;
      smp.top_k = 20;
      smp.top_p = 0.95f;
      smp.min_p = 0.0f;
      smp.presence = smp.frequency = 0.0f;
      unsigned long long seed = 1;
      if (sscanf(ss, "%f,%d,%f,%llu,%f,%f", &smp.temp, &smp.top_k, &smp.top_p, &seed,
                 &smp.presence, &smp.frequency) < 1) {
        fprintf(stderr, "QWENOX_SPEC_SAMPLE expects temp,top_k,top_p,seed[,presence,frequency]\n");
        return 2;
      }
      smp.rng = seed;
      fprintf(stderr, "spec-gen sample: temp=%g top_k=%d top_p=%g seed=%llu presence=%g "
              "frequency=%g chain=%d\n", smp.temp, smp.top_k, smp.top_p, seed,
              smp.presence, smp.frequency, (int)chain);
      spec_out = chain ? m.spec_loop_chain_sample(next_id, spec_gen, gamma, smp, true)
                       : m.spec_loop_sample(next_id, spec_gen, gamma, smp);
    } else {
      spec_out = chain ? m.spec_loop_chain(next_id, spec_gen, gamma)
                       : m.spec_loop(next_id, spec_gen, gamma);
    }
    printf("ids:");
    for (int t : tokens) printf(" %d", t);
    printf(" %d", next_id);
    for (int t : spec_out) printf(" %d", t);
    printf("\n");
    return 0;
  }

  auto show = [&](const std::vector<float>& lg, int step, int chosen) {
    int V = g_cfg.vocab;
    double mx = -1e300;
    for (int i = 0; i < V; i++) mx = std::max(mx, (double)lg[i]);
    double se = 0;
    for (int i = 0; i < V; i++) se += exp(lg[i] - mx);
    double lse = mx + log(se);
    std::vector<int> idx(V);
    for (int i = 0; i < V; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    printf("step %d chosen=%d top5:", step, chosen);
    for (int i = 0; i < 5; i++) printf(" [%d]=%.4f", idx[i], lg[idx[i]] - (float)lse);
    printf("\n");
    fflush(stdout);
  };

  int cur = next_id;
  auto loop_t0 = std::chrono::steady_clock::now();
  for (int g = 0; g < gen; g++) {
    if (dump) {
      m.get_logits(logits);
      dump_pos(logits, (int)tokens.size() + g, -1);
    } else if (show_top) {
      m.get_logits(logits);
      show(logits, g, cur);
    } else {
      printf("step %d chosen=%d\n", g, cur);
      fflush(stdout);
    }
    all.push_back(cur);
    if (g + 1 < gen) cur = m.forward(cur);
  }
  printf("ids:");
  for (int t : all) printf(" %d", t);
  printf("\n");
  if (gen > 1) {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          loop_t0)
                    .count();
    fprintf(stderr, "decode: %d tokens in %.0f ms = %.1f tok/s (%.1f ms/token)\n", gen - 1,
            ms, (gen - 1) * 1000.0 / ms, ms / (gen - 1));
  }

  // KV-continuation self-test: for each split K, reset, prefill [:K], continue
  // [K:] from live state, then decode — ids must match the one-shot baseline.
  for (int K : splits) {
    if (K <= 0 || K >= (int)tokens.size()) {
      fprintf(stderr, "split %d out of range\n", K);
      continue;
    }
    m.reset_state();
    std::vector<int> head(tokens.begin(), tokens.begin() + K);
    std::vector<int> tail(tokens.begin() + K, tokens.end());
    m.prefill_batch(head, 0);
    int cid = m.prefill_batch(tail, K);
    std::vector<int> sall = tokens;
    sall.push_back(cid);
    for (int g = 1; g < gen; g++) {
      cid = m.forward(cid);
      sall.push_back(cid);
    }
    bool ok = sall == all;
    printf("split %d: %s\n", K, ok ? "MATCH" : "DIFF");
    if (!ok) {
      for (size_t i = tokens.size(); i < all.size() && i < sall.size(); i++)
        if (all[i] != sall[i]) {
          printf("  first diff at gen step %zu: base=%d split=%d\n",
                 i - tokens.size(), all[i], sall[i]);
          break;
        }
    }
    fflush(stdout);
  }
  m.perf_report();
  return 0;
}

#endif  // !defined(QWENOX_NO_MAIN)
