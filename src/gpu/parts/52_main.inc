// ---------------- main -------------------------------------------------------
int main(int argc, char** argv) {
#ifdef _WIN32
  if (!wsa_init()) {
    fprintf(stderr, "WSAStartup failed\n");
    return 1;
  }
#endif
  if (argc < 2) {
    fprintf(stderr,
            "usage: gdec <base.hgn> [overlay.hgn ...] --tokens 1,2,3 [--gen N] [--dump] "
            "[--no-ple] [--maxctx N]\n"
            "       gdec <base.hgn> [overlay.hgn ...] --tokens-file IDS --ppl "
            "[--ppl-window N --ppl-stride N] [--maxctx N]\n"
            "       gdec <base.hgn> [overlay.hgn ...] --kld-base REF.kld "
            "[--kld-chunks N] [--kld-save OUT]   (llama.cpp KLD file)\n"
            "       gdec <base.hgn> [overlay.hgn ...] --tokens-file IDS --kld-save OUT "
            "[--kld-ctx 512] [--kld-chunks N]\n"
            "       gdec <base.hgn> [overlay.hgn ...] --serve [--port N] [--maxctx N] "
            "[--no-ple]\n"
            "       M-RoPE: [--mrope-grid t,h,w ...] with --tokens; "
            "--mrope-test K dumps rope tables/decode positions without weights "
            "(GDEC_MROPE_DUMP=<file>)\n"
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
  int ppl_window = 0, ppl_stride = 0;
  kld::Opts kld_opts;  // --kld-*: KL divergence vs a llama.cpp logits file
  // M-RoPE: per-image grid triples (repeatable), rope-only test mode.
  std::vector<std::array<int, 3>> mrope_grids;
  int mrope_test = -1;  // -1 off; else number of decode steps to dump
  // Vision (stage 4): sidecar tower weights; offline ViT parity mode.
  std::string vision_tower, vision_test, vision_patches;
  while (argi < argc && argv[argi][0] != '-') overlays.push_back(argv[argi++]);
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
    else if (a == "--port")
      port = atoi(next());
    else if (a == "--maxctx") {
      maxctx = atoi(next());
      maxctx_set = true;
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
    //   gdec <ignored-> --vision-test PREFIX --vision-tower S.hgn
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
    // GDEC_MROPE_DUMP=<file> and compare via tools/cmp_mrope.py.
    Checkpoint ckpt(base.c_str());
    for (const auto& o : overlays) ckpt.add_overlay(o.c_str());
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
    Checkpoint ckpt(base.c_str());
    ple_uring_open(base.c_str());
    for (const auto& o : overlays) ckpt.add_overlay(o.c_str());
    fprintf(stderr, "loaded %zu tensors (base+%zu overlay)\n", ckpt.tensor_count(),
            overlays.size());
    gguf_dense_apply(ckpt);
    g_cfg.vocab = (int)ckpt.at("lm_head.weight").dims[0];
    memstats_capture_hip_baseline();
    devarena_init(ckpt, maxctx);
    load_arena(ckpt);
    GpuModel m(ckpt, maxctx);
    gguf_dense_release(ckpt);
    m.ple_on = ple_on;
    m.build_graphs();
    if (!vision_tower.empty()) m.load_vision(vision_tower.c_str());
    if (!getenv("GDEC_NOWARMUP")) {
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
    if (g_gguf)  // GDEC_GGUF experts: every shard is part of the identity
      for (const auto& mp : g_gguf->mappings()) kvsnap::g_model_files.push_back(mp.path);
    if (g_gguf_dense) {  // same files as the G2 hybrid, different weights
      kvsnap::g_model_files.push_back("@gguf-dense");
      if (g_gguf_mtp)
        for (const auto& mp : g_gguf_mtp->mappings()) kvsnap::g_model_files.push_back(mp.path);
    }
    kvsnap::g_vision_file = vision_tower;
    return serve::run(m, port);
  }
  if (!kld_mode && tokens.size() < (ppl ? 2u : 1u)) {
    fprintf(stderr, ppl ? "--ppl needs at least two tokens\n" : "need --tokens\n");
    return 2;
  }
  int need = ppl ? 0 : mtp_test > 0 ? mtp_test : spec_gen > 0 ? spec_gen + gamma : gen;
  if (gamma < 1) gamma = 1;
  if (gamma > 8) gamma = 8;  // spec workspace holds at most 8 draft rows
  if (!kld_mode && ppl_window == 0 && tokens.size() + (size_t)need > (size_t)maxctx) {
    fprintf(stderr, "tokens+gen exceed maxctx\n");
    return 2;
  }

  Checkpoint ckpt(base.c_str());
  ple_uring_open(base.c_str());
  for (const auto& o : overlays) ckpt.add_overlay(o.c_str());
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

  GpuModel m(ckpt, maxctx);
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

  if (!getenv("GDEC_NOWARMUP") && mtp_test == 0 && spec_gen == 0) {
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
    if (getenv("GDEC_NOPREFILLBATCH") && ppl_window == 0) {
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
      if (ppl_window && getenv("GDEC_NOPREFILLBATCH")) {
        fprintf(stderr, "--ppl-window does not support GDEC_NOPREFILLBATCH\n");
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
  if (getenv("GDEC_NOPREFILLBATCH")) {
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

  if (mtp_test > 0) {
    m.mtp_accept_test(tokens, next_id, mtp_test);
    return 0;
  }
  if (spec_gen > 0) {
    m.mtp_tap_capture = false;  // verify/re-prefill must NOT capture taps
    // (MTP prompt ingest already happened per-chunk inside prefill_batch.)
    if (getenv("GDEC_NOPREFILLBATCH")) {
      fprintf(stderr, "spec-gen: GDEC_NOPREFILLBATCH unsupported (no tap "
              "capture without prefill_batch)\n");
      return 1;
    }
    std::vector<int> spec_out = m.spec_loop(next_id, spec_gen, gamma);
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
