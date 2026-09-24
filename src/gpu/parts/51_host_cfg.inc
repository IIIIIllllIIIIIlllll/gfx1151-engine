struct Cfg {
  std::string dir;
  uint64_t cap_bytes = 20ull << 30;
  size_t min_tokens = 4096;
};

static uint64_t fnv1a(const int* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= (uint32_t)p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static bool wr(int fd, const void* data, size_t n) {
  const char* p = (const char*)data;
  while (n) {
    ssize_t w = os_write(fd, p, n);
    if (w <= 0) return false;
    p += w;
    n -= (size_t)w;
  }
  return true;
}

// D2H into staging, then append to fd. Phase timings accumulate for the
// slow-save diagnostic in kvsnap::save (cold-start page-in can stall writes).
static double g_t_d2h = 0, g_t_wr = 0;
static bool d2h_wr(int fd, const void* dptr, void* staging, size_t n) {
  auto a = std::chrono::steady_clock::now();
  if (hipMemcpy(staging, dptr, n, hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  auto b = std::chrono::steady_clock::now();
  const bool ok = wr(fd, staging, n);
  g_t_d2h += std::chrono::duration<double>(b - a).count();
  g_t_wr += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - b)
                .count();
  return ok;
}

static bool h2d_rd(const void* src, void* dptr, size_t n) {
  return hipMemcpy(dptr, src, n, hipMemcpyHostToDevice) == hipSuccess;
}

struct Sizes {
  int ntok, nqsa, ngdn, maxctx, kvb, iblocks, ringpos;
  // M-RoPE extras: placeholder token ids are pixel-agnostic, so an
  // M-RoPE snapshot is only valid for a request with the same grids and
  // the same patch-bytes hash.
  bool mrope = false;
  int rope_delta = 0;
  uint64_t img_hash = 0;
  std::vector<std::array<int, 3>> grids;
  bool legacy = false;  // KVS1/KVS2 snapshot (broken ik layout): never restored
};

// Meta format (KVS3):
//   line 1: "KVS3 ntok nqsa ngdn maxctx kvb index_blocks ringpos mrope"
//   line 2 (mrope == 1 only): "rope_delta img_hash k t0 h0 w0 ..."
// KVS1/KVS2 stored ik.bin as one contiguous slice (only layer 0 correct, see
// KVSNAP-REVIEW.md §1); they are rejected here and dropped by maintain_lru.
static bool read_meta(const std::string& dir, Sizes& s) {
  std::ifstream f(dir + "/meta");
  char tag[8];
  if (!(f >> tag)) return false;
  if (!strcmp(tag, "KVS1") || !strcmp(tag, "KVS2")) {
    s.legacy = true;
    return false;
  }
  if (strcmp(tag, "KVS3")) return false;
  int mrope = 0;
  if (!(f >> s.ntok >> s.nqsa >> s.ngdn >> s.maxctx >> s.kvb >> s.iblocks >>
        s.ringpos >> mrope))
    return false;
  if (!mrope) return true;
  s.mrope = true;
  size_t ng = 0;
  if (!(f >> s.rope_delta >> s.img_hash >> ng) || ng > 64) return false;
  s.grids.resize(ng);
  for (auto& g : s.grids)
    if (!(f >> g[0] >> g[1] >> g[2])) return false;
  return true;
}

// Indexer key blocks stored per QSA layer: logical blocks [0, n), clamped to
// the per-layer capacity (ntok near maxctx would otherwise run past it).
static size_t ik_blocks(int ntok, int iblocks) {
  return std::min((size_t)ntok / 4 + 2, (size_t)std::max(0, iblocks));
}

static uint64_t state_bytes(const Sizes& s) {
  uint64_t fixed = (uint64_t)s.ngdn * 48 * 128 * 128 * 4 +
                   (uint64_t)s.ngdn * 3 * 10240 * 4 + 9 * 10240 * 4 +
                   (uint64_t)s.nqsa * 512 * 4;
  return fixed + (uint64_t)s.nqsa * ik_blocks(s.ntok, s.iblocks) * 128 * 4 +
         2ull * s.nqsa * s.ntok * 512 * s.kvb;
}


// LRU maintenance is intentionally separate from the request path. The queue
// is bounded to one pending job: a newer save supersedes an older maintenance
// pass, so a burst of requests cannot grow an unbounded background queue.
struct MaintJob {
  Cfg cfg;
  std::string protect;
  bool touch_existing = false;
};

static std::mutex g_maint_mtx;
static std::condition_variable g_maint_cv;
static MaintJob g_maint_job;
static bool g_maint_pending = false;
static bool g_maint_started = false;

static void maintain_lru(const MaintJob& job) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const std::string dst = job.cfg.dir + "/" + job.protect;
  if (job.touch_existing && fs::exists(dst, ec))
    fs::last_write_time(dst, fs::file_time_type::clock::now(), ec);

  struct Ent {
    fs::path p;
    fs::file_time_type t;
    uint64_t bytes;
  };
  std::vector<Ent> ents;
  std::vector<fs::path> legacy;
  uint64_t total = 0;
  for (auto& de : fs::directory_iterator(job.cfg.dir, ec)) {
    Sizes s;
    std::string fn = de.path().filename().string();
    if (fn.rfind("s_", 0)) continue;
    if (!read_meta(de.path().string(), s)) {
      if (s.legacy) legacy.push_back(de.path());
      continue;
    }
    uint64_t b = state_bytes(s) + (uint64_t)s.ntok * 4;
    ents.push_back({de.path(), fs::last_write_time(de.path(), ec), b});
    total += b;
  }
  // pre-KVS3 snapshots have an invalid ik layout and are never restored
  for (auto& p : legacy) {
    std::error_code rec;
    fs::remove_all(p, rec);
    if (!rec)
      fprintf(stderr, "kvsnap: removed legacy snapshot %s (pre-KVS3 format)\n",
              p.filename().string().c_str());
  }
  std::sort(ents.begin(), ents.end(),
            [](const Ent& a, const Ent& b) { return a.t < b.t; });
  for (auto& e : ents) {
    if (total <= job.cfg.cap_bytes) break;
    if (e.p.filename().string() == job.protect) continue;
    uint64_t b = e.bytes;
    ec.clear();
    fs::remove_all(e.p, ec);
    if (!ec) {
      total -= b;
      fprintf(stderr, "kvsnap: evicted %s (LRU, cap %.1f GB)\n",
              e.p.filename().string().c_str(),
              job.cfg.cap_bytes / 1073741824.0);
    }
  }
}

static void maintain_worker() {
  for (;;) {
    MaintJob job;
    {
      std::unique_lock<std::mutex> lk(g_maint_mtx);
      g_maint_cv.wait(lk, [] { return g_maint_pending; });
      job = std::move(g_maint_job);
      g_maint_pending = false;
    }
    maintain_lru(job);
  }
}

static void schedule_lru(const Cfg& cfg, const char* protect,
                         bool touch_existing) {
  {
    std::lock_guard<std::mutex> lk(g_maint_mtx);
    if (!g_maint_started) {
      g_maint_started = true;
      std::thread(maintain_worker).detach();
    }
    g_maint_job = MaintJob{cfg, protect ? protect : "", touch_existing};
    g_maint_pending = true;
  }
  g_maint_cv.notify_one();
}
static void save(GpuModel& m, const std::vector<int>& toks, const Cfg& cfg) {
  namespace fs = std::filesystem;
  const int ntok = (int)toks.size();
  const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
  const int kvb = m.qsa_kv_fp32 ? 4 : 2;
  // The device state must describe exactly `toks` (pos == ntok); anything
  // else would publish a snapshot whose KV/GDN state belongs to other tokens.
  if (ntok != m.pos || ntok > m.maxctx) {
    fprintf(stderr, "kvsnap: save skipped: %d tokens but live pos %d\n", ntok, m.pos);
    return;
  }
  char name[64];
  snprintf(name, sizeof name, "s_%d_%016llx", ntok,
           (unsigned long long)fnv1a(toks.data(), toks.size()));
  std::string dst = cfg.dir + "/" + name, tmp = cfg.dir + "/tmp_" + name;
  std::error_code ec;
  if (fs::exists(dst, ec)) {  // same state already cached: just bump LRU time
    schedule_lru(cfg, name, true);
    return;
  }
  fs::remove_all(tmp, ec);
  if (fs::create_directories(tmp, ec) == false && ec) return;
  const size_t s_b = (size_t)ngdn * 48 * 128 * 128 * 4,
               conv_b = (size_t)ngdn * 3 * 10240 * 4, ring_b = 9 * 10240 * 4,
               iraw_b = (size_t)nqsa * 512 * 4,
               ik_n = ik_blocks(ntok, m.index_blocks),
               ik_l = ik_n * 128 * 4,  // one layer's ik slice
               kvl_b = (size_t)ntok * 512 * kvb;
  size_t stage_b = std::max({s_b, ik_l, kvl_b});
  // Persistent staging: a fresh 100+MB malloc per save is mmap/munmap churn,
  // and the first-touch page faults can stall for seconds (observed up to
  // 148s D2H) when the system is under memory pressure. Grow-only, so the
  // steady state faults nothing.
  // NOTE: do NOT pin this buffer — hipHostMalloc'd memory makes hipMemcpy D2H
  // collapse to ~7 MB/s on gfx1151 (measured 160 s for 1 GiB vs ~6 s
  // unpinned); the unified-memory GPU write path hates pinned mappings.
  static void* st = nullptr;
  static size_t st_b = 0;
  if (st_b < stage_b) {
    free(st);
    st = malloc(stage_b);
    st_b = st ? stage_b : 0;
  }
  if (!st) return;
  // one blob file per device state region; any short write aborts the save
  auto blob = [&](const char* fn, const void* dptr, size_t n) {
    int fd = os_open_wr((tmp + "/" + fn).c_str());
    if (fd < 0) return false;
    bool ok = d2h_wr(fd, dptr, st, n);
    if (os_close(fd)) ok = false;
    return ok;
  };
  bool ok = true;
  g_t_d2h = g_t_wr = 0;
  const auto sv0 = std::chrono::steady_clock::now();
  {  // meta
    std::ofstream f(tmp + "/meta");
    f << "KVS3 " << ntok << " " << nqsa << " " << ngdn << " " << m.maxctx << " "
      << kvb << " " << m.index_blocks << " " << m.ple_ringpos << " "
      << (m.mrope_active_ ? 1 : 0) << "\n";
    if (m.mrope_active_) {
      f << m.rope_delta_ << " " << m.img_hash_ << " " << m.mrope_grids_.size();
      for (const auto& g : m.mrope_grids_)
        f << " " << g[0] << " " << g[1] << " " << g[2];
      f << "\n";
    }
  }
  {  // tokens (host data, not a device blob)
    int fd = os_open_wr((tmp + "/tokens").c_str());
    if (fd < 0 || !wr(fd, toks.data(), toks.size() * 4) || os_close(fd)) ok = false;
  }
  ok = ok && blob("s.bin", m.d_S, s_b) && blob("convst.bin", m.d_convst, conv_b) &&
       blob("ring.bin", m.d_ring, ring_b) && blob("iraw.bin", m.d_iraw, iraw_b);
  // Files hold LOGICAL order (independent of the device page table):
  // ik.bin = per QSA layer, blocks [0, ik_n); kv.bin = K half then V half,
  // per QSA layer, rows [0, ntok). Device reads go page-run by page-run.
  if (ok) {
    int fd = os_open_wr((tmp + "/ik.bin").c_str());
    if (fd < 0) ok = false;
    for (int qi = 0; qi < nqsa && ok; qi++) {
      const float* layer = m.d_ik + (size_t)qi * m.ik_layer_blocks * 128;
      m.kv_runs(0, (int)ik_n, KV_PAGE_BLOCKS, [&](int, size_t pb, int n) {
        if (ok) ok = d2h_wr(fd, layer + pb * 128, st, (size_t)n * 128 * 4);
      });
    }
    if (fd >= 0 && os_close(fd)) ok = false;
  }
  if (ok) {
    int fd = os_open_wr((tmp + "/kv.bin").c_str());
    if (fd < 0)
      ok = false;
    const size_t row_b = (size_t)512 * kvb;
    for (int half = 0; half < 2 && ok; half++)
      for (int qi = 0; qi < nqsa && ok; qi++) {
        const char* layer =
            (m.qsa_kv_fp32 ? (const char*)(half ? m.d_vc : m.d_kc)
                           : (const char*)(half ? m.d_vcb : m.d_kcb)) +
            (size_t)qi * m.kv_layer_rows * row_b;
        m.kv_runs(0, ntok, KV_PAGE, [&](int, size_t pr, int n) {
          if (ok) ok = d2h_wr(fd, layer + pr * row_b, st, (size_t)n * row_b);
        });
      }
    if (fd >= 0 && os_close(fd)) ok = false;
  }
  if (ok) {
    fs::rename(tmp, dst, ec);  // atomic publish
    ok = !ec;
  }
  if (!ok) {
    fs::remove_all(tmp, ec);
    fprintf(stderr, "kvsnap: save of %d tokens failed (non-fatal)\n", ntok);
    return;
  }
  Sizes mine{ntok, nqsa, ngdn, m.maxctx, kvb, m.index_blocks, 0};
  fprintf(stderr, "kvsnap: saved %d tokens (%.1f MB) -> %s\n", ntok,
          state_bytes(mine) / 1048576.0, name);
  const double sv = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - sv0)
                        .count();
  if (sv > 1.0)
    fprintf(stderr,
            "%s kvsnap: slow save %.1fs (d2h %.1fs, write %.1fs, other %.1fs)\n",
            log_ts(), sv, g_t_d2h, g_t_wr, sv - g_t_d2h - g_t_wr);
  // Queue LRU accounting/eviction after publishing. Atomic rename remains the
  // snapshot commit point; maintenance cannot make a partial tree visible.
  schedule_lru(cfg, name, false);
}

// map a blob read-only and H2D it into place; expected size must match exactly.
static bool rd_blob(const std::string& path, void* dptr, size_t n) {
  int fd = os_open_rd(path.c_str());
  if (fd < 0) return false;
  int64_t fsz = os_fsize(fd);
  os_close(fd);
  if (fsz < 0 || (uint64_t)fsz != n) return false;
  void* mp = os_map_ro(path.c_str(), n);
  if (!mp) return false;
  bool ok = h2d_rd(mp, dptr, n);
  os_unmap_ro(mp, n);
  return ok;
}

static bool restore(GpuModel& m, const std::string& dir,
                    std::vector<int>& toks_out, const Cfg& cfg) {
  Sizes s;
  const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
  const int kvb = m.qsa_kv_fp32 ? 4 : 2;
  if (!read_meta(dir, s) || s.nqsa != nqsa || s.ngdn != ngdn ||
      s.maxctx != m.maxctx || s.kvb != kvb || s.iblocks != m.index_blocks)
    return false;
  const int ntok = s.ntok;
  if (ntok <= 0 || ntok > m.maxctx) return false;
  {  // tokens
    std::ifstream f(dir + "/tokens", std::ios::binary);
    toks_out.resize(ntok);
    if (!f.read((char*)toks_out.data(), (std::streamsize)ntok * 4)) return false;
  }
  if (!rd_blob(dir + "/s.bin", m.d_S, (size_t)ngdn * 48 * 128 * 128 * 4) ||
      !rd_blob(dir + "/convst.bin", m.d_convst, (size_t)ngdn * 3 * 10240 * 4) ||
      !rd_blob(dir + "/ring.bin", m.d_ring, 9 * 10240 * 4) ||
      !rd_blob(dir + "/iraw.bin", m.d_iraw, (size_t)nqsa * 512 * 4))
    return false;
  // A2: the snapshot replaces the live sequence: drop its pages, then map
  // [0, ntok) on demand (the page-run writes below need them backed). On a
  // failed restore the caller's reset_state() releases them again.
  m.kv_trim(0);
  m.kv_reserve(ntok);
  // Map a logical-order blob and scatter it page-run by page-run (see save()).
  auto rd_layers = [&](const std::string& path, size_t tot, auto&& per_file) {
    int fd = os_open_rd(path.c_str());
    if (fd < 0) return false;
    int64_t fsz = os_fsize(fd);
    os_close(fd);
    if (fsz < 0 || (uint64_t)fsz != tot) return false;
    if (!tot) return true;
    void* mp = os_map_ro(path.c_str(), tot);
    if (!mp) return false;
    bool ok = per_file((const char*)mp);
    os_unmap_ro(mp, tot);
    return ok;
  };
  {  // ik.bin: per QSA layer, logical blocks [0, ik_n)
    const size_t ik_n = ik_blocks(ntok, m.index_blocks), ik_l = ik_n * 128 * 4;
    if (!rd_layers(dir + "/ik.bin", (size_t)nqsa * ik_l, [&](const char* mp) {
          bool ok = true;
          for (int qi = 0; qi < nqsa && ok; qi++) {
            const char* src = mp + (size_t)qi * ik_l;
            float* layer = m.d_ik + (size_t)qi * m.ik_layer_blocks * 128;
            m.kv_runs(0, (int)ik_n, KV_PAGE_BLOCKS, [&](int lb, size_t pb, int n) {
              if (ok)
                ok = h2d_rd(src + (size_t)lb * 128 * 4, layer + pb * 128,
                            (size_t)n * 128 * 4);
            });
          }
          return ok;
        }))
      return false;
  }
  {  // kv.bin: K half then V half, per QSA layer, logical rows [0, ntok)
    const size_t row_b = (size_t)512 * kvb, kvl_b = (size_t)ntok * row_b;
    if (!rd_layers(dir + "/kv.bin", 2 * (size_t)nqsa * kvl_b, [&](const char* mp) {
          bool ok = true;
          for (int half = 0; half < 2 && ok; half++)
            for (int qi = 0; qi < nqsa && ok; qi++) {
              const char* src = mp + ((size_t)half * nqsa + qi) * kvl_b;
              char* layer = (m.qsa_kv_fp32 ? (char*)(half ? m.d_vc : m.d_kc)
                                           : (char*)(half ? m.d_vcb : m.d_kcb)) +
                            (size_t)qi * m.kv_layer_rows * row_b;
              m.kv_runs(0, ntok, KV_PAGE, [&](int lr, size_t pr, int n) {
                if (ok)
                  ok = h2d_rd(src + (size_t)lr * row_b, layer + pr * row_b,
                              (size_t)n * row_b);
              });
            }
          return ok;
        }))
      return false;
  }
  // GDEC_QSA_WMMA_BTV: the transposed prefill V cache is derived state (not
  // in the snapshot); rebuild it for [0, ntok) from the restored row-major V,
  // or sparse prefill after the restore would read stale V for the prefix.
  if (m.qsa_wmma_btv) {
    // Paged (A1c): rebuild per physically contiguous run on physical rows; a
    // page holds whole 4-token blocks, so physical row block pr/4 is exactly
    // the physical vct block of that run (unpaged: one run [0, ntok)).
    for (int qi = 0; qi < nqsa; qi++)
      m.kv_runs(0, ntok, KV_PAGE, [&](int, size_t prz, int n) {
        const int pr = (int)prz;
        const unsigned grid =
            (unsigned)((((size_t)(pr + n + 3) / 4 - (size_t)pr / 4) * 512 + 255) / 256);
        k_bf16_rows_to_bt<<<grid, 256, 0, g_str>>>(
            m.d_vcb + (size_t)qi * m.kv_layer_rows * 512, m.vct_layer(qi), pr, pr + n);
      });
    CK(hipGetLastError());
    CK(hipStreamSynchronize(g_str));
  }
  // host-side and scalar state: pos/ringpos (both mirrors), PLE token history
  m.pos = ntok;
  m.ple_ringpos = s.ringpos;
  CK(hipMemcpy(m.d_pos, &m.pos, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(m.d_ringpos, &m.ple_ringpos, 4, hipMemcpyHostToDevice));
  m.chist.assign(toks_out.begin(), toks_out.end());
  // LRU touch and eviction are coalesced onto the bounded maintenance worker.
  std::string name = std::filesystem::path(dir).filename().string();
  schedule_lru(cfg, name.c_str(), true);
  return true;
}

// Longest snapshot whose token list is a strict prefix of `prompt` (strictly
// shorter: a full-length match has no logits to resume generation from).
// M-RoPE candidates additionally require the request's grids and patch-bytes
// hash to match the snapshot meta (req_grids != nullptr for vision requests).
static std::string best_prefix(const Cfg& cfg, const std::vector<int>& prompt,
                               size_t* ntok_out,
                               const std::vector<std::array<int, 3>>* req_grids =
                                   nullptr,
                               uint64_t req_img_hash = 0) {
  namespace fs = std::filesystem;
  std::error_code ec;
  std::string best;
  size_t bestn = 0;
  for (auto& de : fs::directory_iterator(cfg.dir, ec)) {
    std::string fn = de.path().filename().string();
    if (fn.rfind("s_", 0)) continue;
    Sizes s;
    if (!read_meta(de.path().string(), s) || (size_t)s.ntok >= prompt.size() ||
        (size_t)s.ntok <= bestn)
      continue;
    if (s.mrope && (!req_grids || s.grids != *req_grids ||
                    s.img_hash != req_img_hash))
      continue;  // same-shaped different image: not this snapshot
    std::ifstream f(de.path().string() + "/tokens", std::ios::binary);
    std::vector<int> t(s.ntok);
    if (!f.read((char*)t.data(), (std::streamsize)s.ntok * 4)) continue;
    if (std::equal(t.begin(), t.end(), prompt.begin())) {
      best = de.path().string();
      bestn = (size_t)s.ntok;
    }
  }
  *ntok_out = bestn;
  return best;
}

}  // namespace kvsnap

namespace serve {

struct LineBuf {
  sockfd_t fd;
  std::string buf;
  bool take(std::string& out) {
    size_t nl = buf.find('\n');
    if (nl == std::string::npos) return false;
    out = buf.substr(0, nl);
    buf.erase(0, nl + 1);
    return true;
  }
  // blocking read of one line; false on EOF/error
  bool readline(std::string& out) {
    for (;;) {
      if (take(out)) return true;
      char tmp[65536];
      ssize_t n = sock_recv(fd, tmp, sizeof tmp, false);
      if (n <= 0) return false;
      buf.append(tmp, n);
    }
  }
  // non-blocking: true + line iff a complete line is available right now
  bool try_readline(std::string& out) {
    for (;;) {
      if (take(out)) return true;
      char tmp[65536];
      ssize_t n = sock_recv(fd, tmp, sizeof tmp, true);
      if (n <= 0) return false;  // EAGAIN (or EOF: surfaced by next readline)
      buf.append(tmp, n);
    }
  }
  // Blocking read of exactly n bytes (binary frames after a GEN line).
  // Bytes already swallowed into the line buffer are drained first, so the
  // client may pipeline frame bytes right behind the newline. False on
  // EOF/error/timeout (SO_RCVTIMEO, armed by the caller for bulk reads).
  bool read_full(void* dst, size_t n) {
    char* p = (char*)dst;
    size_t head = std::min(n, buf.size());
    if (head) {
      memcpy(p, buf.data(), head);
      buf.erase(0, head);
      p += head;
      n -= head;
    }
    while (n > 0) {
      ssize_t r = sock_recv(fd, p, n, false);
      if (r <= 0) return false;
      p += r;
      n -= (size_t)r;
    }
    return true;
  }
};

static bool send_all(sockfd_t fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t n = sock_send(fd, s.data() + off, s.size() - off);
    if (n <= 0) return false;
    off += n;
  }
  return true;
}

struct GenReq {
  int64_t req = 0;
  int max_tokens = 0;
  std::vector<int> eos;
  std::vector<int> ids;
  int drafter = -1;  // omitted: engine default; 0 serial; 1 MTP; 3 ngram-mod; 4 chain
  bool reject = false;  // malformed or unsupported optional field
  // Sampler (SAMPLE suffix). sample=true implies temp>0; greedy otherwise.
  bool sample = false;
  float temp = 1.0f, top_p = 0.0f, min_p = 0.0f;  int top_k = 0;
  uint64_t seed = 0;
  // PENALTY (presence, frequency) / BIAS (tid,val pairs) / LOGPROBS: sampler-only.
  float presence = 0.0f, frequency = 0.0f;
  std::vector<std::pair<int, float>> bias;
  bool logprobs = false;
  // SNAP / SNAP2: frontend-hinted prefix lengths worth an SSD snapshot (the
  // next turn repeats them verbatim; see serve_api.py render_prompt).
  // SNAPS: same idea as a counted list (semantic message boundaries); the
  // engine keeps cheap RAM checkpoints (rckpt) at these cuts.
  size_t snap = 0, snap2 = 0;
  std::vector<size_t> snaps;
  // MROPE <k> <t> <h> <w> ...: per-image grid triples, expanded in-engine
  // (stage 1: fresh prefill + serial decode only; cont/spec are rejected).
  std::vector<std::array<int, 3>> mrope_grids;
  // VIMG <k>: k binary patch frames follow the GEN line (one per MROPE grid,
  // same order; stage 4). Frame: u32 magic 0x56494D31, u32 P, u64 nbytes,
  // then P*1536 fp32 LE patches in block-major (C,T,ph,pw) order.
  int vimg = 0;
};



// Host-side sampler: logits come back over unified memory (~1 MB), all filtering
// is O(vocab) or less. Filter order follows HF warpers: penalties/bias on raw
// logits, then temperature, top_k, top_p, min_p. top_p uses a binary-search
// threshold (mass(p >= t) >= top_p) instead of a full sort; ties keep all tied
// tokens, unlike sort-and-cut. RNG is splitmix64: deterministic per
// (seed, emitted sequence).
struct HostSampler {
  float temp, top_p, min_p;
  int top_k;
  uint64_t rng;
  float presence, frequency;
  std::vector<std::pair<int, float>> bias;
  std::unordered_map<int, int> hist;  // emitted token counts (not the prompt)
  std::vector<int> cand;              // scratch: candidate token ids
  std::vector<float> work;            // scratch: top-k preservation / residual

  static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  uint64_t next64() {  // splitmix64
    return mix64(rng += 0x9E3779B97F4A7C15ull);
  }

  double uniform() {
    return (double)(next64() >> 11) * (1.0 / 9007199254740992.0);
  }

  HostSampler fork_stream(uint64_t tag) const {
    HostSampler out = *this;
    out.rng = mix64(rng ^ tag);
    out.cand.clear();
    out.work.clear();
    return out;
  }

  void remember(int id) { hist[id]++; }

  // Convert raw logits in place to the normalized distribution after the
  // engine's penalty/bias -> temperature -> top-k -> top-p -> min-p pipeline.
  void prepare(std::vector<float>& logits) {
    const int n = (int)logits.size();
    for (auto& b : bias)
      if (b.first >= 0 && b.first < n) logits[b.first] += b.second;
    if (presence != 0.0f || frequency != 0.0f)
      for (auto& h : hist)
        if (h.first >= 0 && h.first < n)
          logits[h.first] -= presence + frequency * (float)h.second;
    float lmax = -INFINITY;
    for (int i = 0; i < n; i++) lmax = fmaxf(lmax, logits[i]);
    // unnormalized probs after temperature
    for (int i = 0; i < n; i++) logits[i] = expf((logits[i] - lmax) / temp);
    cand.clear();
    if (top_k > 0 && top_k < n) {
      cand.resize(n);
      for (int i = 0; i < n; i++) cand[i] = i;
      std::nth_element(cand.begin(), cand.begin() + top_k, cand.end(),
                       [&](int a, int b) { return logits[a] > logits[b]; });
      cand.resize(top_k);
      work.resize(cand.size());
      for (size_t j = 0; j < cand.size(); j++) work[j] = logits[cand[j]];
      std::fill(logits.begin(), logits.end(), 0.0f);
      for (size_t j = 0; j < cand.size(); j++) logits[cand[j]] = work[j];
    } else {
      cand.reserve(n);
      for (int i = 0; i < n; i++)
        if (logits[i] > 0.0f) cand.push_back(i);
    }
    if (top_p > 0.0f && top_p < 1.0f && !cand.empty()) {
      // largest threshold t with mass(p >= t) >= top_p * total
      double total = 0;
      for (int i : cand) total += logits[i];
      double lo = 0, hi = 1.0;  // probs are scaled so max == 1
      for (int it = 0; it < 50; it++) {
        double mid = 0.5 * (lo + hi), m = 0;
        for (int i : cand)
          if (logits[i] >= (float)mid) m += logits[i];
        if (m >= top_p * total) lo = mid; else hi = mid;
      }
      std::vector<int> keep;
      keep.reserve(cand.size());
      for (int i : cand)
        if (logits[i] >= (float)lo)
          keep.push_back(i);
        else
          logits[i] = 0.0f;
      cand.swap(keep);
    }
    if (min_p > 0.0f && !cand.empty()) {
      float pmax = 0;
      for (int i : cand) pmax = fmaxf(pmax, logits[i]);
      float thr = min_p * pmax;
      std::vector<int> keep;
      keep.reserve(cand.size());
      for (int i : cand)
        if (logits[i] >= thr) keep.push_back(i);
      if (!keep.empty()) {
        for (int i : cand)
          if (logits[i] < thr) logits[i] = 0.0f;
        cand.swap(keep);  // never empty the candidate set
      }
    }
    double z = 0;
    for (int i : cand) z += logits[i];
    for (int i : cand) logits[i] = (float)(logits[i] / z);
  }

  // Draw from a distribution produced by prepare(). Candidate order is kept
  // stable with the serial sampler so existing seeded behavior is unchanged.
  int draw_prepared(const std::vector<float>& probs, float* lp,
                    bool add_hist) {
    double z = 0;
    for (int i : cand) z += probs[i];
    double u = uniform() * z;
    int sel = cand.back();
    for (int i : cand) {
      u -= probs[i];
      if (u <= 0) { sel = i; break; }
    }
    if (lp) *lp = (float)(log((double)probs[sel]) - log(z));
    if (add_hist) remember(sel);
    return sel;
  }

  int sample(std::vector<float>& logits, float* lp) {
    prepare(logits);
    return draw_prepared(logits, lp, true);
  }

  // Standard speculative correction distribution: normalize max(p-q, 0).
  // The reported logprob remains the target probability p(token), matching
  // serial LOGPROBS semantics rather than the residual draw probability.
  int sample_residual(const std::vector<float>& p,
                      const std::vector<float>& q, float* lp) {
    const int n = (int)p.size();
    work.assign(n, 0.0f);
    cand.clear();
    double z = 0.0;
    for (int i = 0; i < n; i++) {
      float d = fmaxf(0.0f, p[i] - q[i]);
      if (d > 0.0f) {
        work[i] = d;
        cand.push_back(i);
        z += d;
      }
    }
    if (cand.empty() || z <= 0.0) {
      for (int i = 0; i < n; i++)
        if (p[i] > 0.0f) {
          work[i] = p[i];
          cand.push_back(i);
          z += p[i];
        }
    }
    double u = uniform() * z;
    int sel = cand.back();
    for (int i : cand) {
      u -= work[i];
      if (u <= 0.0) {
        sel = i;
        break;
      }
    }
    if (lp) *lp = p[sel] > 0.0f ? logf(p[sel]) : -INFINITY;
    return sel;
  }
};

// Default drafter for GEN lines that omit the field (the HTTP API always
// omits it). Unset: chain — ngram drafts every round it can and MTP
// takes the rounds where the table has nothing, for greedy and sampled
// requests alike (sampled ngram rounds re-draw from the target distribution
// and accept by token-id match, llama.cpp-style; only the pure-ngram
// default stays greedy-only, see parse_gen). GDEC_DRAFTER overrides:
// "serial"/"0" forces serial, "ngram"/"3" pure ngram, "chain"/"4" explicit
// chain, "mtp" the legacy MTP default.
static int env_default_drafter() {
  const char* e = getenv("GDEC_DRAFTER");
  if (!e) return 4;
  if (!strcmp(e, "serial") || !strcmp(e, "0")) return 0;
  if (!strcmp(e, "ngram") || !strcmp(e, "ngram-mod") || !strcmp(e, "3")) return 3;
  if (!strcmp(e, "chain") || !strcmp(e, "4")) return 4;
  return -1;
}

// GEN <req> <max_tokens> <n_eos> <eos...> <n_ids> <ids...> [drafter] [suffixes]
static GenReq parse_gen(std::istringstream& ss) {
  GenReq g;
  int n_eos = 0, n_ids = 0;
  if (!(ss >> g.req >> g.max_tokens >> n_eos) || n_eos < 0 || n_eos > 64 ||
      g.max_tokens <= 0) {
    g.reject = true;
    return g;
  }
  for (int i = 0; i < n_eos; i++) {
    int e;
    if (!(ss >> e)) { g.reject = true; return g; }
    g.eos.push_back(e);
  }
  if (!(ss >> n_ids) || n_ids <= 0) { g.reject = true; return g; }
  g.ids.resize(n_ids);
  for (int i = 0; i < n_ids; i++)
    if (!(ss >> g.ids[i])) { g.reject = true; return g; }
  std::string tok;
  while (ss >> tok) {
    if (tok == "SAMPLE") {
      // SAMPLE <temp> <top_k> <top_p> <min_p> <seed:u64>
      if (!(ss >> g.temp >> g.top_k >> g.top_p >> g.min_p >> g.seed) ||
          g.temp <= 0.0f || g.top_k < 0) {
        g.reject = true;
        return g;
      }
      g.sample = true;
    } else if (tok == "NGRAM") {
      g.drafter = 3;
    } else if (tok == "PENALTY") {
      if (!(ss >> g.presence >> g.frequency)) { g.reject = true; return g; }
    } else if (tok == "BIAS") {
      int nb;
      if (!(ss >> nb) || nb < 0 || nb > 20480) { g.reject = true; return g; }
      g.bias.resize(nb);
      for (int i = 0; i < nb; i++) {
        if (!(ss >> g.bias[i].first >> g.bias[i].second) ||
            g.bias[i].first < 0 || g.bias[i].first >= 248320) {
          g.reject = true;
          return g;
        }
      }
    } else if (tok == "LOGPROBS") {
      g.logprobs = true;
    } else if (tok == "MROPE") {
      int k;
      if (!(ss >> k) || k <= 0 || k > 64) { g.reject = true; return g; }
      for (int i = 0; i < k; i++) {
        std::array<int, 3> g3{0, 0, 0};
        if (!(ss >> g3[0] >> g3[1] >> g3[2])) { g.reject = true; return g; }
        g.mrope_grids.push_back(g3);
      }
    } else if (tok == "VIMG") {
      if (!(ss >> g.vimg) || g.vimg <= 0 || g.vimg > 16) {
        g.reject = true;
        return g;
      }
    } else if (tok == "SNAP" || tok == "SNAP2") {
      long long v;
      if (!(ss >> v) || v < 0) { g.reject = true; return g; }
      (tok == "SNAP" ? g.snap : g.snap2) = (size_t)v;
    } else if (tok == "SNAPS") {
      long long n;
      if (!(ss >> n) || n < 0 || n > 64) { g.reject = true; return g; }
      for (long long i = 0; i < n; i++) {
        long long v;
        if (!(ss >> v) || v < 0) { g.reject = true; return g; }
        g.snaps.push_back((size_t)v);
      }
    } else if (isdigit(tok[0]) || tok[0] == '-') {
      int v = atoi(tok.c_str());
      if (v < 0 || v > 4) g.reject = true;
      else g.drafter = v;
    }
    // anything else: drop (lenient by design)
  }
  // Omitted drafter: resolve the GDEC_DRAFTER default. Pure ngram only
  // drafts greedy requests (sampled ones keep the legacy default); chain
  // covers greedy and sampled alike.
  if (g.drafter < 0) {
    static const int def = env_default_drafter();
    g.drafter = (def == 3 && g.sample) ? -1 : def;
  }
  // PENALTY/BIAS/LOGPROBS are sampler-only (protocol spec): no sampler, no-op
  // fields would silently mislead the client.
  if (!g.sample &&
      (g.presence != 0.0f || g.frequency != 0.0f || !g.bias.empty() || g.logprobs))
    g.reject = true;
  return g;
}

// ---------------------------------------------------------------------------
// Concurrent serving (GDEC_PARALLEL=N, paged KV only; 2026-09-24).
//
// N sequence slots (GpuModel::slots_init) time-share the GPU. Every
// connection gets a reader thread -- PING/INFO/MEM/X answer at once, even
// mid-request -- and each GEN runs on its own worker thread. A FIFO ticket
// lock (Turnstile) owns the GPU; the owner hands it over at yield points
// (spec round tops, prompt prefill chunk boundaries, serial decode tokens)
// when someone is waiting, so sequences interleave but each one computes
// exactly what it would alone (bit-exact per sequence).
//
// All slots map KV pages from ONE pool (default: one maxctx sequence, like
// llama.cpp's unified KV). When the pool runs dry (GpuModel::kv_evict_one):
// RAM checkpoints go first, then idle slots' pages (LRU), then the
// latest-admitted busy request later than the one asking is aborted; if
// the asker is itself the latest, IT fails (KvOverflow). An earlier request
// is never sacrificed for a later one.
//
// N=1 keeps the old batch-1 semantics: live-state reuse is per connection
// (a new connection starts with an empty hist), no hooks are installed.

// FIFO ticket lock around the (single-threaded) GpuModel.
struct Turnstile {
  std::mutex mu;
  std::condition_variable cv;
  uint64_t next = 0, serving = 0;
  void acquire() {
    std::unique_lock<std::mutex> lk(mu);
    const uint64_t t = next++;
    cv.wait(lk, [&] { return serving == t; });
  }
  void release() {
    {
      std::lock_guard<std::mutex> lk(mu);
      serving++;
    }
    cv.notify_all();
  }
  bool contended() {  // someone besides the owner is queued
    std::lock_guard<std::mutex> lk(mu);
    return next - serving > 1;
  }
};

struct Slot {
  bool busy = false;
  bool abort = false;  // pool-overflow victim: fail at the next GPU acquire
  // MTP KV continuity: true after a spec-enabled turn (prefill tap capture
  // + spec decode keep the MTP layer's KV in lockstep with the trunk).
  // False after kvsnap/rckpt restores and any plain (non-speculative) turn.
  bool mtp_live = false;
  // tokens INGESTED by this slot's live state (prompt + every token actually
  // fed to forward). KV reuse: a GEN whose prompt extends hist exactly
  // continues from live state instead of re-prefilling.
  std::vector<int> hist;
  uint64_t seq = 0;       // admission order of the request it runs
  uint64_t last_use = 0;  // LRU among idle slots
  uint64_t owner = 0;     // connection id of the last request (N=1 affinity)
};

static Turnstile g_turn;
static std::mutex g_sm;  // g_slots bookkeeping + admission queue
static std::condition_variable g_scv;
static std::vector<Slot> g_slots;
static std::list<uint64_t> g_wait;  // admission tickets, FIFO
static uint64_t g_ticket = 0, g_seq = 0, g_use = 0;
static std::atomic<uint64_t> g_conn_id{0};
static thread_local int tl_slot = -1;  // slot of the GPU-owning worker

// Idle slot for a prompt: the longest hist that is a strict prefix (live
// continuation), else an empty slot, else the least recently used one whose
// connection is gone (owner 0), else the least recently used one.
static int pick_slot(const std::vector<int>& ids) {
  int best = -1, empty = -1, lru = -1, orphan = -1;
  size_t bestn = 0;
  for (int i = 0; i < (int)g_slots.size(); i++) {
    const Slot& s = g_slots[(size_t)i];
    if (s.busy) continue;
    const size_t n = s.hist.size();
    if (n == 0) {
      if (empty < 0) empty = i;
      continue;
    }
    if (n < ids.size() && n > bestn &&
        std::equal(s.hist.begin(), s.hist.end(), ids.begin())) {
      best = i;
      bestn = n;
    }
    if (lru < 0 || s.last_use < g_slots[(size_t)lru].last_use) lru = i;
    if (s.owner == 0 && (orphan < 0 || s.last_use < g_slots[(size_t)orphan].last_use))
      orphan = i;
  }
  return best >= 0 ? best : empty >= 0 ? empty : orphan >= 0 ? orphan : lru;
}

// Wait (FIFO) for an idle slot and claim it; -1 if cancelled while queued.
static int slot_admit(const std::vector<int>& ids, const std::atomic<bool>& cancel) {
  std::unique_lock<std::mutex> lk(g_sm);
  const uint64_t t = g_ticket++;
  g_wait.push_back(t);
  for (;;) {
    if (g_wait.front() == t) {
      const int s = pick_slot(ids);
      if (s >= 0) {
        g_wait.pop_front();
        Slot& sl = g_slots[(size_t)s];
        sl.busy = true;
        sl.abort = false;
        sl.seq = ++g_seq;
        g_scv.notify_all();
        return s;
      }
    }
    if (cancel) {
      g_wait.remove(t);
      g_scv.notify_all();
      return -1;
    }
    g_scv.wait_for(lk, std::chrono::milliseconds(50));
  }
}

static void slot_leave(int s) {
  {
    std::lock_guard<std::mutex> lk(g_sm);
    g_slots[(size_t)s].busy = false;
    g_slots[(size_t)s].last_use = ++g_use;
  }
  g_scv.notify_all();
}

// GpuModel::yield_fn: hand the GPU to the next queued worker, then fail
// this request if a pool overflow picked it as the victim meanwhile.
static void yield_hook(GpuModel& m) {
  if (tl_slot < 0) return;
  if (g_turn.contended()) {
    g_turn.release();
    g_turn.acquire();
    m.bind(tl_slot);
  }
  bool ab;
  {
    std::lock_guard<std::mutex> lk(g_sm);
    ab = g_slots[(size_t)tl_slot].abort;
  }
  if (ab) throw KvOverflow("aborted: KV pool needed by an earlier request");
}

// GpuModel::kv_starve_fn (pool dry after rckpt eviction), GPU owner only.
static bool starve(GpuModel& m) {
  const int cur = tl_slot;
  if (cur < 0) return false;
  std::lock_guard<std::mutex> lk(g_sm);
  const int n = (int)g_slots.size();
  int v = -1;
  for (int i = 0; i < n; i++) {
    const Slot& s = g_slots[(size_t)i];
    if (i != cur && !s.busy && m.slot_pages(i) > 0 &&
        (v < 0 || s.last_use < g_slots[(size_t)v].last_use))
      v = i;
  }
  if (v >= 0) {
    fprintf(stderr, "%s [kvpage] pool dry: dropped idle slot %d (%zu tokens, %d pages)\n",
            log_ts(), v, g_slots[(size_t)v].hist.size(), m.slot_pages(v));
    m.slot_reset(v, cur);
    g_slots[(size_t)v].hist.clear();
    g_slots[(size_t)v].mtp_live = false;
    return true;
  }
  const uint64_t mine = g_slots[(size_t)cur].seq;
  for (int i = 0; i < n; i++) {
    const Slot& s = g_slots[(size_t)i];
    if (i != cur && s.busy && !s.abort && s.seq > mine && m.slot_pages(i) > 0 &&
        (v < 0 || s.seq > g_slots[(size_t)v].seq))
      v = i;
  }
  if (v >= 0) {
    fprintf(stderr, "%s [kvpage] pool dry: aborting the later request on slot %d "
            "(%d pages) for slot %d\n", log_ts(), v, m.slot_pages(v), cur);
    m.slot_reset(v, cur);
    g_slots[(size_t)v].abort = true;
    return true;
  }
  throw KvOverflow("KV pool exhausted: concurrent contexts exceed the shared pool");
}

struct Conn {
  sockfd_t fd;
  uint64_t id = 0;
  std::mutex wmu;  // reader (PONG/INFO/...) and worker (T/D) both write
  std::atomic<bool> cancel{false}, in_flight{false};
  std::atomic<int64_t> req{0};
  bool send(const std::string& s) {
    std::lock_guard<std::mutex> lk(wmu);
    return send_all(fd, s);
  }
};

// One GEN, on its own worker thread. Protocol lines and semantics are the
// batch-1 ones; only scheduling changed.
static void gen_one(GpuModel& m, Conn& c, GenReq g,
                    std::vector<std::vector<float>> vpatches,
                    const kvsnap::Cfg& kvs) {
  const std::atomic<bool>& cancel = c.cancel;
  char db[192];
  auto err_line = [&]() {
    snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n", (long long)g.req,
             g.ids.size());
    return std::string(db);
  };
  // The request's last line. in_flight drops first so the GEN the client
  // sends the moment it reads D is not taken for a pipelined one.
  auto finish = [&](const std::string& d) {
    c.in_flight = false;
    if (!c.send(d)) sock_shutdown(c.fd);
  };
  // Vision request validation (0.5.8 semantics), before state mutation:
  // patches need a loaded tower and one MROPE grid each; any image
  // placeholder without a grid is rejected (first position logged).
  if (g.vimg > 0) {
    const char* why = nullptr;
    if (!m.vision_on)
      why = "vision tower not loaded (start the engine with "
            "--vision-tower <sidecar.hgn>)";
    else if ((int)g.mrope_grids.size() != g.vimg)
      why = "VIMG frame count != MROPE grid count";
    else
      for (int i = 0; i < g.vimg; i++) {
        auto gr = g.mrope_grids[i];
        if (gr[0] != 1 ||
            (int)vpatches[i].size() != gr[0] * gr[1] * gr[2] * 1536) {
          why = "patches/grid size mismatch (or t != 1)";
          break;
        }
      }
    if (why) {
      fprintf(stderr, "vision: req %lld rejected: %s\n", (long long)g.req, why);
      finish(err_line());
      return;
    }
  }
  if (g.mrope_grids.empty()) {
    size_t bad = g.ids.size();
    for (size_t i = 0; i < g.ids.size(); i++)
      if (g.ids[i] == 248056 || g.ids[i] == 248057) {
        bad = i;
        break;
      }
    if (bad < g.ids.size()) {
      fprintf(stderr, "vision: req %lld rejected: image placeholder at "
              "prompt pos %zu without MROPE/VIMG\n", (long long)g.req, bad);
      finish(err_line());
      return;
    }
  }
  const int s = slot_admit(g.ids, cancel);
  if (s < 0) {
    snprintf(db, sizeof db, "D %lld cancel 0 0 0.0 0.0\n", (long long)g.req);
    finish(db);
    return;
  }
  g_turn.acquire();
  tl_slot = s;
  m.bind(s);
  // Leaving: the slot turns idle while this worker still owns the GPU, so a
  // starving peer never sees a finished request as busy.
  struct Lease {
    int s;
    ~Lease() {
      tl_slot = -1;
      slot_leave(s);
      g_turn.release();
    }
  } lease{s};
  Slot& sl = g_slots[(size_t)s];
  std::vector<int>& hist = sl.hist;
  bool& mtp_live = sl.mtp_live;
  bool victim;
  {
    std::lock_guard<std::mutex> lk(g_sm);
    victim = sl.abort;
    if (m.nslots == 1 && sl.owner != c.id) {  // batch-1: reuse per connection
      hist.clear();
      mtp_live = false;
    }
    sl.owner = c.id;
  }
  if (victim) {  // reset by a pool overflow while queued for the GPU
    fprintf(stderr, "%s serve: req %lld aborted: KV pool needed by an earlier "
            "request\n", log_ts(), (long long)g.req);
    hist.clear();
    mtp_live = false;
    finish(err_line());
    return;
  }
  if (m.nslots > 1)
    fprintf(stderr, "%s req %lld -> slot %d (%zu live tokens)\n", log_ts(),
            (long long)g.req, s, hist.size());
  try {
    // KV reuse: longest common prefix of live state and this prompt.
    // Continue only when hist is a STRICT prefix (mcp == hist.size() and
    // new tokens follow); anything else falls to the SSD snapshot cache
    // and only then to a full recompute.
    size_t mcp = 0;
    while (mcp < hist.size() && mcp < g.ids.size() && hist[mcp] == g.ids[mcp])
      mcp++;
    bool cont = !hist.empty() && mcp == hist.size() && mcp < g.ids.size();
    int n_cached = cont ? (int)mcp : 0;
    // M-RoPE restrictions, decided before any state mutation:
    // KV reuse (cont) IS sound for M-RoPE — prefix positions are
    // suffix-independent and RoPE is applied before K is cached, so the
    // cached prefix is identical to what a recompute would produce —
    // but only when the cached image KV came from the SAME pixels:
    // placeholder ids are pixel-agnostic, so the patch-bytes hash
    // guards cont; a mismatch (or GDEC_VISION_NOCONT) falls back to a
    // full recompute. MTP speculation IS allowed for M-RoPE: verify
    // positions come from the d_ropecs table (ensure_rope_cs handles
    // base=q+1 out-of-range rows with rope_delta_), and the drafter
    // paths rotate with *d_rdelta, so only acceptance rate is at stake.
    if (!g.mrope_grids.empty() && cont &&
        (fnv1a_patches(vpatches) != m.img_hash_ ||
         getenv("GDEC_VISION_NOCONT"))) {
      cont = false;
      n_cached = 0;
    }
    if (!cont && rckpt::enabled()) {
      // RAM checkpoint tier (llama.cpp-style PARTIAL_ONLY): only GDN
      // state is stored; validity requires the checkpoint tokens to be a
      // prefix of the live token stream (m.chist), which guarantees the
      // QSA KV/indexer prefix is still intact (paged: the checkpoint owns
      // its pages, any slot may restore it). MTP KV for the restored
      // prefix is NOT guaranteed -> disable speculation for this turn.
      const bool is_mrope = !g.mrope_grids.empty();
      const uint64_t req_ih = is_mrope ? fnv1a_patches(vpatches) : 0;
      auto r0 = std::chrono::steady_clock::now();
      int ci = rckpt::match(g.ids, m.chist, is_mrope,
                            is_mrope ? &g.mrope_grids : nullptr, req_ih);
      if (ci >= 0 && rckpt::restore(m, ci, &hist)) {
        mcp = hist.size();
        cont = true;
        n_cached = (int)mcp;
        mtp_live = false;
        double rs = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - r0)
                        .count();
        fprintf(stderr, "%s rckpt: restored %zu tokens in %.2fs\n",
                log_ts(), mcp, rs);
      }
    }
    if (!cont && !kvs.dir.empty()) {
      size_t sn = 0;
      // M-RoPE requests: placeholder ids are pixel-agnostic, so a
      // snapshot only qualifies when grids AND patch-bytes hash match
      // (checked inside best_prefix against the KVS3 meta); a mismatch
      // is simply a cache miss and falls to a full recompute.
      const bool is_mrope = !g.mrope_grids.empty();
      const uint64_t req_ih = is_mrope ? fnv1a_patches(vpatches) : 0;
      std::string sp = kvsnap::best_prefix(
          kvs, g.ids, &sn, is_mrope ? &g.mrope_grids : nullptr, req_ih);
      if (!sp.empty()) {
        auto r0 = std::chrono::steady_clock::now();
        if (kvsnap::restore(m, sp, hist, kvs)) {  // hist := snapshot tokens
          mcp = hist.size();
          cont = true;
          n_cached = (int)mcp;
          // the snapshot holds no MTP KV/state: as after an rckpt restore,
          // the MTP layer is stale for the restored prefix
          mtp_live = false;
          double rs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - r0)
                          .count();
          fprintf(stderr, "%s kvsnap: restored %zu tokens from %s in %.1fs\n",
                  log_ts(), mcp, sp.c_str(), rs);
        }
      }
    }
    if (!cont) m.reset_state();
    // Ngram proposals are request-local. Clear learned suffixes so a
    // previous request cannot collide with this prompt's hash history.
    if (g.drafter == 3 || g.drafter == 4) m.ngram.reset();
    // Vision requests expand their grids in-engine (validated against the
    // prompt's image token runs); plain requests drop stale mrope state
    // (a cont turn skips reset_state).
    if (!g.mrope_grids.empty()) {
      if (!m.set_mrope(g.ids, g.mrope_grids)) {
        fprintf(stderr, "vision: req %lld rejected: grids do not cover "
                "the image token runs\n", (long long)g.req);
        finish(err_line());
        return;
      }
      if (g.vimg > 0 &&
          !m.vision_embeds(g.ids, g.mrope_grids, vpatches, n_cached)) {
        finish(err_line());
        return;
      }
      vpatches.clear();
      vpatches.shrink_to_fit();  // ~50 MB/image: release before decode
    } else {
      m.clear_mrope();
    }
    std::unordered_set<int> eos(g.eos.begin(), g.eos.end());
    HostSampler smp;
    std::vector<float> lbuf;
    if (g.sample) {
      smp.temp = g.temp;
      smp.top_p = g.top_p;
      smp.min_p = g.min_p;
      smp.top_k = g.top_k;
      smp.rng = g.seed;
      smp.presence = g.presence;
      smp.frequency = g.frequency;
      smp.bias = g.bias;
      // penalties count GENERATED tokens only: hist starts empty and
      // sample() records each emitted token
    }
    auto t0 = std::chrono::steady_clock::now();
    int cur;
    float cur_lp = 0.0f;
    // MTP speculation is available for greedy and rejection-sampled
    // requests when its KV can be made complete. kvsnap restores lack MTP
    // KV for the prompt, so those requests stay on plain decode.
    static const int spec_gamma = std::max(
        1, std::min(8, getenv("GDEC_SPEC_GAMMA")
                           ? atoi(getenv("GDEC_SPEC_GAMMA"))
                           : 3));
    if (g.drafter == 1 && !m.mtp_avail) {
      finish(err_line());
      return;
    }
    const bool use_ngram = g.drafter == 3 && !g.sample;
    // Chain mode (llama.cpp-style fallback): ngram drafts every round it
    // can, MTP takes the rounds where the table has nothing. Sampled
    // requests run spec_loop_chain_sample (ngram rounds re-draw from the
    // target distribution; GDEC_NOSPEC_SAMPLE forces plain serial, same
    // as use_spec). The MTP leg requires a complete MTP KV: a fresh
    // prefill always has one; a cont/kvsnap turn only if the previous
    // request left MTP live.
    const bool use_chain = g.drafter == 4 &&
                           (!g.sample ||
                            getenv("GDEC_NOSPEC_SAMPLE") == nullptr);
    const bool chain_mtp = use_chain && m.mtp_avail &&
                           (!cont || mtp_live) &&
                           getenv("GDEC_NOSPEC") == nullptr;
    const bool use_spec = !use_ngram && !use_chain && m.mtp_avail &&
                          g.drafter != 0 && g.drafter != 3 &&
                          g.drafter != 4 &&
                          (!cont || mtp_live) &&
                          (!g.sample ||
                           getenv("GDEC_NOSPEC_SAMPLE") == nullptr) &&
                          getenv("GDEC_NOSPEC") == nullptr;
    m.last_spec_rounds = m.last_spec_commit = m.last_spec_rollbacks = 0;
    // Prompt ingest. A failure here (or a KvOverflow anywhere below) lands
    // in the catch at the end: the slot is reset, the client gets D error.
    if (use_spec || chain_mtp) {
      m.mtp_tap_capture = true;  // prefill_batch ingests MTP per chunk
      if (cont) {
        // Boundary: MTP position mcp-1 pairs the live d_R (stream at
        // the last ingested position) with the first new token.
        std::vector<int> bt{g.ids[mcp]};
        m.mtp_ingest_b(m.d_R, bt, 1, (int)mcp - 1);
      }
    }
    // Frontend-hinted snapshot points (SNAPS/SNAP/SNAP2) that fall
    // inside the uncached range: pause the prefill there, checkpoint,
    // resume. The next turn re-renders history as TEXT whose ids match
    // the prompt only up to these boundaries (the assistant opener and
    // generated text retokenize differently), so they are the cuts
    // that hit. Checkpoints go to the RAM tier (rckpt): the SSD tier
    // only takes the prompt-end and request-end states to bound disk
    // churn.
    std::vector<size_t> cuts;
    if (!kvs.dir.empty() || rckpt::enabled()) {
      for (size_t cc : g.snaps)
        if (cc > mcp && cc < g.ids.size() && cc >= kvs.min_tokens)
          cuts.push_back(cc);
      for (size_t cc : {g.snap2, g.snap})
        if (cc > mcp && cc < g.ids.size() && cc >= kvs.min_tokens)
          cuts.push_back(cc);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    if (!cont) hist.clear();
    size_t done_upto = cont ? mcp : 0;
    // Arm the 1s prefill progress reporter for the prompt ingest only
    // (spec-verify batches never see it armed). prog_base accumulates
    // across the kvsnap cut segments inside prefill_chunk.
    m.prog_total = (int)(g.ids.size() - done_upto);
    m.prog_base = 0;
    m.prog_last_est = 0;
    m.prog_req = g.req;
    m.prog_t0 = m.prog_last = std::chrono::steady_clock::now();
    // live spec counters reset per request, or a serial decode after a
    // speculative one would report the previous request's acceptance
    m.live_spec_rounds = m.live_spec_commits = m.live_spec_proposed = 0;
    for (size_t cc : cuts) {
      std::vector<int> seg(g.ids.begin() + done_upto, g.ids.begin() + cc);
      cur = m.prefill_batch(seg, (int)done_upto);
      hist.assign(g.ids.begin(), g.ids.begin() + cc);
      rckpt::save(m, hist);
      done_upto = cc;
      m.yield_point();  // consistent state between segments
    }
    std::vector<int> seg(g.ids.begin() + done_upto, g.ids.end());
    cur = m.prefill_batch(seg, (int)done_upto);
    m.prog_total = 0;
    m.mtp_tap_capture = false;  // decode/verify must not capture taps
    hist = g.ids;  // prompt fully ingested on both paths
    if (use_spec || chain_mtp) mtp_live = true;  // MTP KV complete & in lockstep
    if (g.sample) {  // d_logits still holds the last position's logits
      m.get_logits(lbuf);
      cur = smp.sample(lbuf, &cur_lp);
    }
    double pre_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    // Phase 1: snapshot only at end-of-turn. This removes the first-token
    // stall while the full asynchronous capture path is being designed.
    rckpt::save(m, hist);  // prompt-end RAM checkpoint (cheap, deduped)
    auto d0 = std::chrono::steady_clock::now();
    int ngen = 0;
    bool dead = false;
    std::string reason = "length";
    // 1s-cadence decode stats (2026-09-13): instantaneous + average t/s,
    // with the live MTP acceptance the spec loops bump every round.
    // committed drafts = commits - rounds (each round also emits the
    // verify/bonus token), acceptance = committed / proposed.
    auto dl_t0 = std::chrono::steady_clock::now();
    auto dl_last = dl_t0;
    int dl_last_ngen = 0;
    auto dl_tick = [&]() {
      auto now = std::chrono::steady_clock::now();
      double iv = std::chrono::duration<double>(now - dl_last).count();
      if (iv < 1.0) return;
      double el = std::chrono::duration<double>(now - dl_t0).count();
      if (m.live_spec_rounds > 0) {
        int acc_n = m.live_spec_commits - m.live_spec_rounds;
        fprintf(stderr,
                "%s req %lld | n_decoded = %d | t = %.0f s | tg = %.1f "
                "tok/s (avg %.1f) | accept %.1f%% (%d/%d)\n",
                log_ts(), (long long)g.req, ngen, el,
                (ngen - dl_last_ngen) / iv, el > 0 ? ngen / el : 0.0,
                m.live_spec_proposed
                    ? 100.0 * acc_n / m.live_spec_proposed
                    : 0.0,
                acc_n, m.live_spec_proposed);
      } else {
        fprintf(stderr,
                "%s req %lld | n_decoded = %d | t = %.0f s | tg = %.1f "
                "tok/s (avg %.1f)\n",
                log_ts(), (long long)g.req, ngen, el,
                (ngen - dl_last_ngen) / iv, el > 0 ? ngen / el : 0.0);
      }
      dl_last = now;
      dl_last_ngen = ngen;
    };
    // Token line; the X verb (reader thread) raises `cancel`.
    auto send_tok = [&](int tok, bool with_lp, float lp) -> bool {
      char tb[96];
      int tl = with_lp ? snprintf(tb, sizeof tb, "T %lld %d %.6f\n",
                                  (long long)g.req, tok, (double)lp)
                       : snprintf(tb, sizeof tb, "T %lld %d\n",
                                  (long long)g.req, tok);
      if (!c.send(std::string(tb, tl))) {
        dead = true;
        return false;
      }
      ngen++;
      return true;
    };
    if (use_ngram || use_chain) {
      // chain with a live MTP keeps it live (the loop patches its KV
      // every ngram round); pure/degraded ngram is trunk-only decode and
      // invalidates the previous MTP state.
      const bool chain = use_chain && chain_mtp;
      if (!chain) mtp_live = false;
      if (!m.ple_on) m.chist.assign(g.ids.begin(), g.ids.end());
      if (eos.count(cur)) {
        reason = "done";
      } else {
        send_tok(cur, g.sample && g.logprobs, cur_lp);
      }
      auto emit_ng = [&](int tok) -> bool {
        if (ngen >= g.max_tokens) return false;
        if (eos.count(tok)) { reason = "done"; return false; }
        if (!send_tok(tok, false, 0.0f)) return false;
        dl_tick();
        return !cancel;
      };
      auto emit_ng_smp = [&](int tok, float lp) -> bool {
        if (ngen >= g.max_tokens) return false;
        if (eos.count(tok)) { reason = "done"; return false; }
        if (!send_tok(tok, g.logprobs, lp)) return false;
        dl_tick();
        return !cancel;
      };
      if (!dead && !eos.count(cur)) {
        if (g.sample)
          m.spec_loop_chain_sample(cur, g.max_tokens - 1, spec_gamma, smp,
                                   chain_mtp, emit_ng_smp);
        else if (chain) m.spec_loop_chain(cur, g.max_tokens - 1, spec_gamma, emit_ng);
        else m.spec_loop_ngram(cur, g.max_tokens - 1, m.ngram.n_max, emit_ng);
      }
      hist.assign(m.chist.begin(), m.chist.end());
    } else if (use_spec) {
      // Stream each token as the spec loop commits it. eos/cancel/length
      // return false, rolling the round back so trunk and MTP stop at the
      // last token actually sent to the client.
      if (eos.count(cur)) {
        reason = "done";
      } else if (send_tok(cur, g.sample && g.logprobs, cur_lp)) {
        if (g.sample) {
          auto emit = [&](int tok, float lp) -> bool {
            if (ngen >= g.max_tokens) return false;
            if (eos.count(tok)) {
              reason = "done";
              return false;
            }
            if (!send_tok(tok, g.logprobs, lp)) return false;
            dl_tick();
            return !cancel;
          };
          m.spec_loop_sample(cur, g.max_tokens - 1, spec_gamma, smp, emit);
        } else {
          auto emit = [&](int tok) -> bool {
            if (ngen >= g.max_tokens) return false;
            if (eos.count(tok)) {
              reason = "done";
              return false;
            }
            if (!send_tok(tok, false, 0.0f)) return false;
            dl_tick();
            return !cancel;
          };
          m.spec_loop(cur, g.max_tokens - 1, spec_gamma, emit);
        }
        // Trunk ingested exactly chist tokens; hist must match for the
        // next turn's KV-reuse prefix check.
        hist.assign(m.chist.begin(), m.chist.end());
      }
    } else {
      mtp_live = false;  // plain decode leaves the MTP layer stale
    }
    if (!use_spec && !use_ngram && !use_chain)
      for (int i = 0; i < g.max_tokens; i++) {
        if (eos.count(cur)) {
          reason = "done";
          break;
        }
        if (!send_tok(cur, g.logprobs, cur_lp)) break;
        if (i + 1 < g.max_tokens) {
          int fed = cur;  // the token forward() ingests (pre-sampling)
          cur = m.forward(fed);
          hist.push_back(fed);
          if (g.sample) {
            m.get_logits(lbuf);
            cur = smp.sample(lbuf, &cur_lp);
          }
        }
        dl_tick();
        if (cancel) break;
        m.yield_point();
      }
    if (dead) {  // client gone: no D line, no end-of-turn snapshot
      c.in_flight = false;
      sock_shutdown(c.fd);
      return;
    }
    double dec_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - d0)
                        .count();
    if (cancel)
      snprintf(db, sizeof db, "D %lld cancel 0 0 0.0 0.0\n", (long long)g.req);
    else if ((use_spec || use_ngram || use_chain) && m.last_spec_rounds > 0)
      snprintf(db, sizeof db,
               "D %lld %s %zu %d %.1f %.1f %d %d %d %d %d\n",
               (long long)g.req, reason.c_str(), g.ids.size(), ngen, pre_ms,
               ngen > 1 ? dec_ms : 0.0,
               use_chain ? (chain_mtp ? 4 : 3) : (use_ngram ? 3 : 1),
               m.last_spec_rounds,
               m.last_spec_commit, n_cached, m.last_spec_proposed);
    else
      // trailing "0 0 0 <n_cached> 0": drafter/rounds/commit placeholders
      // so n_cached lands in the protocol's cache column (parts[10]);
      // the last 0 is the proposed-drafts column (parts[11])
      snprintf(db, sizeof db, "D %lld %s %zu %d %.1f %.1f 0 0 0 %d 0\n",
               (long long)g.req, reason.c_str(), g.ids.size(), ngen, pre_ms,
               ngen > 1 ? dec_ms : 0.0, n_cached);
    finish(db);
    {
      // llama.cpp-style per-request timing block (human-facing; the wire
      // D line above stays the machine-readable record).
      size_t n_proc =
          g.ids.size() - (n_cached > 0 ? (size_t)n_cached : 0);
      fprintf(stderr,
              "%s req %lld | prompt eval = %.1f ms / %zu tokens "
              "(%.2f ms/token, %.1f tok/s)%s\n",
              log_ts(), (long long)g.req, pre_ms, n_proc,
              n_proc ? pre_ms / n_proc : 0.0,
              pre_ms > 0 ? n_proc / (pre_ms / 1000.0) : 0.0,
              n_cached ? " | cached" : "");
      if (ngen > 0)
        fprintf(stderr,
                "%s req %lld | decode eval = %.1f ms / %d tokens "
                "(%.2f ms/token, %.1f tok/s) | %s\n",
                log_ts(), (long long)g.req, dec_ms, ngen,
                dec_ms > 0 ? dec_ms / ngen : 0.0,
                dec_ms > 0 ? ngen / (dec_ms / 1000.0) : 0.0,
                cancel ? "cancel" : reason.c_str());
      else
        fprintf(stderr, "%s req %lld | decode eval = no tokens | %s\n",
                log_ts(), (long long)g.req,
                cancel ? "cancel" : reason.c_str());
      if ((use_spec || use_ngram || use_chain) && m.last_spec_rounds > 0) {
        int acc_n = m.live_spec_commits - m.live_spec_rounds;
        fprintf(stderr,
                "%s req %lld | draft acceptance = %.3f (%d accepted / %d "
                "proposed), mean len = %.2f, drafter %s\n",
                log_ts(), (long long)g.req,
                m.live_spec_proposed
                    ? (double)acc_n / m.live_spec_proposed
                    : 0.0,
                acc_n, m.live_spec_proposed,
                m.last_spec_rounds
                    ? (double)m.live_spec_commits / m.last_spec_rounds
                    : 0.0,
                use_chain ? (chain_mtp ? "chain" : "ngram")
                          : (use_ngram ? "ngram" : "mtp"));
      }
    }
    // SSD prompt cache: snapshot the end-of-turn state for future
    // sessions. Cancelled turns are valid too -- hist tracks exactly the
    // ingested tokens, and short states are skipped inside save().
    // M-RoPE states carry the KVS3 grids/img-hash meta (see above).
    if (!kvs.dir.empty() && hist.size() >= kvs.min_tokens)
      kvsnap::save(m, hist, kvs);
  } catch (const std::exception& e) {
    // prefill/decode failure, or KvOverflow (pool exhausted by concurrent
    // requests: this one is the later): the partial state is unknown, so
    // drop it (returning the slot's pages to the pool) and fail the request
    fprintf(stderr, "%s serve: req %lld failed: %s\n", log_ts(),
            (long long)g.req, e.what());
    m.prog_total = 0;
    m.mtp_tap_capture = m.spec_snap = m.ngram_verify_active = false;
    m.reset_state();
    hist.clear();
    mtp_live = false;
    finish(err_line());
  }
}

// Connection reader: control verbs inline, GEN handed to a worker thread.
static void conn_main(GpuModel& m, sockfd_t cfd, const kvsnap::Cfg& kvs) {
  Conn c;
  c.fd = cfd;
  c.id = ++g_conn_id;
  LineBuf lb{cfd, {}};
  std::thread worker;
  std::string line;
  bool alive = true;
  while (alive && lb.readline(line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string verb;
    ss >> verb;
    if (verb == "PING") {
      alive = c.send("PONG\n");
    } else if (verb == "INFO") {
      char ib[160];
      // fields: mtp draft_head ctx spec_rows default drafter_weights dflash2
      //         cache_mb cache_align kv_slots slot_ctx cache_mode sampling
      // The default drafter is the chain (ngram first, MTP fallback) unless
      // GDEC_DRAFTER overrides; draft_head is a separate shortlist readout
      // and is not part of this checkpoint. kv_slots: concurrent sequences.
      snprintf(ib, sizeof ib, "I %d 0 %d 8 %d %d 0 0 0 %d %d 1 1\n",
               m.mtp_avail ? 1 : 0, m.maxctx,
               env_default_drafter() >= 0 ? env_default_drafter()
                                          : (m.mtp_avail ? 1 : 0),
               m.mtp_avail ? 1 : 0, m.nslots, m.maxctx);
      alive = c.send(ib);
    } else if (verb == "MEM") {
      EngineMemStats ms = memstats_snapshot();
      char mb[512];
      // M version device_cur device_peak registered_cur pinned_cur
      //   pinned_peak committed_cur hip_free hip_total hip_delta rss locked
      snprintf(mb, sizeof mb,
               "M 1 %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n",
               (unsigned long long)ms.device_current,
               (unsigned long long)ms.device_peak,
               (unsigned long long)ms.registered_current,
               (unsigned long long)ms.pinned_current,
               (unsigned long long)ms.pinned_peak,
               (unsigned long long)ms.committed_current,
               (unsigned long long)ms.hip_free,
               (unsigned long long)ms.hip_total,
               (unsigned long long)ms.hip_used_since_start,
               (unsigned long long)ms.process_rss,
               (unsigned long long)ms.process_locked);
      alive = c.send(mb);
    } else if (verb == "X") {
      int64_t req = 0;
      ss >> req;
      if (c.in_flight && req == c.req) {
        c.cancel = true;  // the worker ends it with its own D cancel line
      } else {
        char db[96];
        snprintf(db, sizeof db, "D %lld cancel 0 0 0.0 0.0\n", (long long)req);
        alive = c.send(db);
      }
    } else if (verb == "GEN") {
      GenReq g = parse_gen(ss);
      if (g.reject || (int)g.ids.size() + g.max_tokens > m.maxctx) {
        char db[128];
        snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n", (long long)g.req,
                 g.ids.size());
        alive = c.send(db);
        continue;
      }
      // VIMG binary frames follow the GEN line immediately and must be
      // drained before ANY reply path, or their bytes desync the line
      // protocol. 120 s ceiling on the bulk transfer (~50 MB per 1080p
      // image over loopback is sub-second; the ceiling only guards against
      // a stalled client wedging this connection).
      std::vector<std::vector<float>> vpatches;
      if (g.vimg > 0) {
        bool ok = true;
        sock_rcvtimeo(cfd, 120);
        vpatches.resize(g.vimg);
        for (int i = 0; i < g.vimg && ok; i++) {
          struct {
            uint32_t magic, P;
            uint64_t nbytes;
          } fh;
          ok = lb.read_full(&fh, sizeof fh) && fh.magic == 0x56494D31u &&
               fh.P > 0 && fh.P <= 65536 &&
               fh.nbytes == (uint64_t)fh.P * 1536 * 4;
          if (!ok) break;
          vpatches[i].resize((size_t)fh.P * 1536);
          ok = lb.read_full(vpatches[i].data(), fh.nbytes);
        }
        sock_rcvtimeo(cfd, 0);
        if (!ok) {
          fprintf(stderr, "vision: frame read failed for req %lld "
                  "(timeout/short/garbled); dropping the connection\n",
                  (long long)g.req);
          char db[128];
          snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n",
                   (long long)g.req, g.ids.size());
          c.send(db);
          break;  // protocol desync: cannot trust anything after this
        }
      }
      if (c.in_flight) {  // one GEN at a time per connection
        fprintf(stderr, "serve: req %lld rejected: another GEN is in flight "
                "on this connection\n", (long long)g.req);
        char db[128];
        snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n", (long long)g.req,
                 g.ids.size());
        alive = c.send(db);
        continue;
      }
      if (worker.joinable()) worker.join();  // previous request's post-D tail
      c.cancel = false;
      c.req = g.req;
      c.in_flight = true;
      worker = std::thread(gen_one, std::ref(m), std::ref(c), std::move(g),
                           std::move(vpatches), std::cref(kvs));
    }
    // unknown verbs: drop
  }
  c.cancel = true;  // disconnect cancels the request in flight
  if (worker.joinable()) worker.join();
  {  // its slots become the first LRU victims (live conversations keep theirs)
    std::lock_guard<std::mutex> lk(g_sm);
    for (Slot& s : g_slots)
      if (s.owner == c.id) s.owner = 0;
  }
  sock_close(cfd);
  fprintf(stderr, "%s serve: client disconnected\n", log_ts());
}

static int run(GpuModel& m, int port) {
#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);
#endif
  sockfd_t lfd = sock_tcp();
  int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (bind(lfd, (sockaddr*)&addr, sizeof addr) || listen(lfd, 16)) {
    fprintf(stderr, "serve: bind/listen failed on :%d: %s\n", port,
            sock_strerror(sock_errno()));
    return 1;
  }
  m.slots_init(eff_parallel());
  g_slots.assign((size_t)m.nslots, Slot{});
  if (m.nslots > 1) {
    m.kv_starve_fn = &starve;
    m.yield_fn = &yield_hook;
  }
  fprintf(stderr, "serve: listening on :%d (maxctx %d, %d slot%s)\n", port,
          m.maxctx, m.nslots, m.nslots > 1 ? "s" : "");
  static kvsnap::Cfg kvs;  // read-only once serving; workers share it
  kvs.dir = "data/kvsnap";
  if (const char* v = getenv("GDEC_KVSNAP_DIR")) kvs.dir = v;
  if (const char* v = getenv("GDEC_KVSNAP"))
    if (!strcmp(v, "0")) kvs.dir.clear();
  if (const char* v = getenv("GDEC_KVSNAP_MAX_GB"))
    kvs.cap_bytes = (uint64_t)(atof(v) * 1073741824.0);
  if (!kvs.cap_bytes) kvs.dir.clear();
  if (const char* v = getenv("GDEC_KVSNAP_MIN")) kvs.min_tokens = (size_t)atol(v);
  if (!kvs.dir.empty())
    fprintf(stderr, "kvsnap: dir %s, cap %.1f GB, min %zu tokens\n",
            kvs.dir.c_str(), kvs.cap_bytes / 1073741824.0, kvs.min_tokens);
  for (;;) {
    sockfd_t cfd = accept(lfd, nullptr, nullptr);
    if (cfd < 0) continue;
    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
    fprintf(stderr, "%s serve: client connected\n", log_ts());
    std::thread(conn_main, std::ref(m), cfd, std::cref(kvs)).detach();
  }
  return 0;
}

}  // namespace serve
