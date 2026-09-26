struct Cfg {
  std::string dir;
  uint64_t cap_bytes = 20ull << 30;
  size_t min_tokens = 4096;
  uint64_t queue_bytes = 1024ull << 20;  // captured but not yet written
};

// Model identity for the cache fingerprint (set by main before serve::run).
static std::vector<std::string> g_model_files;  // base + overlays
static std::string g_vision_file;

static const int FMT = 1;  // A4 layout version (part of the fingerprint)

// ---- hashing ----------------------------------------------------------------
static inline uint64_t mix64(uint64_t z) {  // splitmix64 finalizer
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static uint64_t hash_toks(uint64_t h, const int* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    h ^= (uint32_t)p[i];
    h *= 1099511628211ULL;
  }
  return mix64(h);
}
static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
// Content checksum: four independent lanes (xxhash-style rounds), so verifying
// a restore runs near memory speed.
static uint64_t sum64(const void* data, size_t n) {
  const char* p = (const char*)data;
  const uint64_t P1 = 0x9E3779B185EBCA87ULL, P2 = 0xC2B2AE3D27D4EB4FULL;
  uint64_t a[4] = {P1, P2, ~P1, ~P2};
  size_t i = 0;
  for (; i + 32 <= n; i += 32)
    for (int k = 0; k < 4; k++) {
      uint64_t w;
      memcpy(&w, p + i + 8 * k, 8);
      a[k] = rotl64(a[k] + w * P2, 31) * P1;
    }
  uint64_t h = mix64(n);
  for (int k = 0; k < 4; k++) h = mix64(h ^ a[k]);
  for (; i < n; i++) h = mix64(h ^ (uint8_t)p[i]);
  return h;
}
// c[i] = key of the prefix of i full pages; c[0] = seed. A page's key covers
// the whole prefix, so equal keys mean equal KV for that page.
static std::vector<uint64_t> chain(uint64_t seed, const int* t, int npages) {
  std::vector<uint64_t> c((size_t)npages + 1);
  c[0] = seed;
  for (int i = 0; i < npages; i++)
    c[(size_t)i + 1] =
        hash_toks(mix64(c[(size_t)i] + 0x6b76636bULL), t + (size_t)i * KV_PAGE, KV_PAGE);
  return c;
}
// MTP row r pairs h_r with token r+1: a page's MTP rows also depend on the
// first token of the next page.
static uint64_t mtp_key(uint64_t page_key, int look) {
  return mix64(page_key ^ mix64(0x6d747073ULL + (uint32_t)look));
}
static uint64_t ck_id(uint64_t seed, const int* t, size_t n) {
  return hash_toks(mix64(seed ^ 0x636b6964ULL), t, n);
}

static uint64_t g_fp = 0, g_vis_id = 0;
static uint64_t file_id(const std::string& path) {
  std::error_code ec;
  uint64_t h = mix64((uint64_t)std::filesystem::file_size(path, ec));
  for (unsigned char ch : std::filesystem::path(path).filename().string()) h = mix64(h ^ ch);
  return h;
}
// Plain requests: the fingerprint. M-RoPE requests: placeholder ids are
// pixel-agnostic, so the vision tower, the patch-bytes hash and the grids are
// folded into the seed (a different image never shares a page key).
static uint64_t seed_for(bool mrope, uint64_t img_hash,
                         const std::vector<std::array<int, 3>>& grids) {
  if (!mrope) return g_fp;
  uint64_t h = mix64(g_fp ^ 0x6d726f7065ULL);
  h = mix64(h ^ g_vis_id);
  h = mix64(h ^ img_hash);
  for (const auto& g : grids)
    for (int v : g) h = mix64(h ^ (uint32_t)v);
  return h;
}
// Seed of a live-state prefix: one without image tokens is text KV (see
// has_image_tok) and is keyed like a text request's, so it is shared.
static uint64_t seed_of(const GpuModel& m, const int* toks, size_t n) {
  if (!m.mrope_active_ || !has_image_tok(toks, n)) return g_fp;
  return seed_for(true, m.img_hash_, m.mrope_grids_);
}

// ---- geometry and file layouts ----------------------------------------------
struct Geo {
  int nqsa, ngdn, kvb, has_ik;
  size_t row_b, s_b, conv_b, ring_b, iraw_b;
};
static Geo geo(const GpuModel& m) {
  Geo G;
  G.nqsa = g_cfg.layers / 4;
  G.ngdn = g_cfg.layers - G.nqsa;
  G.kvb = m.qsa_kv_fp32 ? 4 : 2;
  G.has_ik = m.d_ik ? 1 : 0;
  G.row_b = (size_t)512 * G.kvb;
  G.s_b = (size_t)G.ngdn * 48 * 128 * 128 * 4;
  G.conv_b = (size_t)G.ngdn * 3 * 10240 * 4;
  G.ring_b = 9 * 10240 * 4;
  G.iraw_b = m.d_iraw ? (size_t)G.nqsa * 512 * 4 : 0;
  return G;
}
// one layer of one page: K rows, V rows, indexer key blocks
static size_t page_layer_b(int kvb, int has_ik) {
  return 2 * (size_t)KV_PAGE * 512 * kvb + (has_ik ? (size_t)KV_PAGE_BLOCKS * 512 : 0);
}
static const char MAG_C[8] = "GDKVC1", MAG_M[8] = "GDKVM1", MAG_K[8] = "GDKCK1";
struct ChHdr {  // c_<key>.kvc (trunk page) / m_<key>.kvm (MTP section)
  char magic[8];
  uint64_t key, parent, sum;  // sum: sum64 of the body
  int32_t nl, kvb, has_ik, look;  // look: MTP lookahead token (-1 trunk)
  int32_t toks[KV_PAGE];
};
struct CkHdr {  // k_<n>_<id>.kck
  char magic[8];
  uint64_t seed, base, id, sum;
  int32_t n, nc, nl, kvb, ringpos, mtp, mtp_pos, nik, has_ik, pad;
  uint64_t s_b, conv_b, ring_b, iraw_b, mR_b, miraw_b;
};
static_assert(sizeof(ChHdr) % 8 == 0 && sizeof(CkHdr) % 8 == 0, "header alignment");
static size_t ck_body_b(const CkHdr& h) {
  const size_t nt = (size_t)h.n - (size_t)h.nc * KV_PAGE;
  return (size_t)h.nc * 16 + nt * 4 + h.s_b + h.conv_b + h.ring_b + h.iraw_b + h.mR_b +
         h.miraw_b + (size_t)h.nl * (2 * nt * 512 * h.kvb + (size_t)h.nik * 512);
}
static bool ck_sane(const CkHdr& h) {
  const uint64_t big = 1ull << 36;
  return !memcmp(h.magic, MAG_K, 8) && h.n > 0 && h.n < (1 << 24) &&
         h.nc == (h.n - 1) / KV_PAGE && h.nl > 0 && h.nl <= 64 &&
         (h.kvb == 2 || h.kvb == 4) && h.nik >= 0 && h.nik <= 2 * KV_PAGE_BLOCKS &&
         h.s_b < big && h.conv_b < big && h.ring_b < big && h.iraw_b < big &&
         h.mR_b < big && h.miraw_b < big;
}
// Indexer key blocks past the full pages that a checkpoint carries. Clamped
// to the mapped pages (paged kv_runs aborts past them) and the capacity.
static int tail_ik(const GpuModel& m, int n, int nc) {
  if (!m.d_ik) return 0;
  const int hi = std::min({n / 4 + 2, KV_PAGE_BLOCKS * ((n + KV_PAGE - 1) / KV_PAGE),
                           m.index_blocks});
  return std::max(0, hi - KV_PAGE_BLOCKS * nc);
}

// Device slices (layer nqsa = the MTP layer, B0).
static char* kv_layer(GpuModel& m, int half, int l) {
  char* b = m.qsa_kv_fp32 ? (char*)(half ? m.d_vc : m.d_kc)
                          : (char*)(half ? m.d_vcb : m.d_kcb);
  return b + (size_t)l * m.kv_layer_rows * 512 * (m.qsa_kv_fp32 ? 4 : 2);
}
static char* ik_layer(GpuModel& m, int l) {
  return (char*)(m.d_ik + (size_t)l * m.ik_layer_blocks * 128);
}
// Copy logical rows [r0, r1) of a paged per-layer region to/from a
// row-contiguous host buffer (files hold logical order, layout-agnostic).
static bool xfer(GpuModel& m, bool h2d, char* dev, char* host, int r0, int r1,
                 int per_page, size_t row_b) {
  bool ok = true;
  m.kv_runs(r0, r1, per_page, [&](int lr, size_t pr, int n) {
    if (!ok) return;
    char* d = dev + pr * row_b;
    char* h = host + (size_t)(lr - r0) * row_b;
    ok = hipMemcpy(h2d ? (void*)d : (void*)h, h2d ? (void*)h : (void*)d,
                   (size_t)n * row_b,
                   h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost) == hipSuccess;
  });
  return ok;
}
static bool d2h(void* h, const void* d, size_t n) {
  return !n || hipMemcpy(h, d, n, hipMemcpyDeviceToHost) == hipSuccess;
}
static bool h2d(void* d, const void* h, size_t n) {
  return !n || hipMemcpy(d, h, n, hipMemcpyHostToDevice) == hipSuccess;
}

// ---- host buffers ------------------------------------------------------------
// Pooled (page images are all one size): a fresh 100+ MB malloc per save is
// mmap/munmap churn whose first-touch faults stalled D2H for seconds under
// memory pressure. NOTE: never pinned -- hipHostMalloc'd memory makes D2H
// collapse to ~7 MB/s on gfx1151.
struct HBuf {
  std::unique_ptr<char[]> p;
  size_t cap = 0, n = 0;
  char* data() { return p.get(); }
};
static std::mutex g_pool_mtx;
static std::vector<HBuf> g_pool;
static size_t g_pool_bytes = 0;
static const size_t g_pool_max = 512ull << 20;
static HBuf buf_get(size_t n) {
  {
    std::lock_guard<std::mutex> lk(g_pool_mtx);
    size_t best = g_pool.size();
    for (size_t i = 0; i < g_pool.size(); i++)
      if (g_pool[i].cap >= n && (best == g_pool.size() || g_pool[i].cap < g_pool[best].cap))
        best = i;
    if (best < g_pool.size()) {
      HBuf b = std::move(g_pool[best]);
      g_pool.erase(g_pool.begin() + (long)best);
      g_pool_bytes -= b.cap;
      b.n = n;
      return b;
    }
  }
  HBuf b;
  b.p.reset(new (std::nothrow) char[n ? n : 1]);
  b.cap = b.p ? n : 0;
  b.n = b.p ? n : 0;
  return b;
}
static void buf_put(HBuf&& b) {
  if (!b.p) return;
  std::lock_guard<std::mutex> lk(g_pool_mtx);
  if (g_pool_bytes + b.cap > g_pool_max) return;  // freed on scope exit
  g_pool_bytes += b.cap;
  g_pool.push_back(std::move(b));
}

// ---- files -------------------------------------------------------------------
static std::string g_dir;
static uint64_t g_cap = 0, g_qcap = 0;
static bool g_on = false;
static std::string ch_name(uint64_t k) {
  char b[32];
  snprintf(b, sizeof b, "c_%016llx.kvc", (unsigned long long)k);
  return b;
}
static std::string mt_name(uint64_t k) {
  char b[32];
  snprintf(b, sizeof b, "m_%016llx.kvm", (unsigned long long)k);
  return b;
}
static std::string ck_name(int n, uint64_t id) {
  char b[48];
  snprintf(b, sizeof b, "k_%d_%016llx.kck", n, (unsigned long long)id);
  return b;
}
static std::string path_of(const std::string& name) { return g_dir + "/" + name; }
static int64_t now_t() {
  return (int64_t)std::filesystem::file_time_type::clock::now().time_since_epoch().count();
}
static void rm_file(const std::string& name) {
  std::error_code ec;
  std::filesystem::remove(path_of(name), ec);
}
static bool wr_all(int fd, const char* p, size_t n) {
  while (n) {
    ssize_t w = os_write(fd, p, std::min(n, (size_t)1 << 30));
    if (w <= 0) return false;
    p += w;
    n -= (size_t)w;
  }
  return true;
}
static bool rd_all(int fd, char* p, size_t n) {
  while (n) {
    ssize_t r = os_read(fd, p, std::min(n, (size_t)1 << 30));
    if (r <= 0) return false;
    p += r;
    n -= (size_t)r;
  }
  return true;
}
// tmp_<name>, fdatasync, rename: a crash never publishes a torn file.
static bool write_file(const std::string& name, const char* p, size_t n) {
  const std::string tmp = path_of("tmp_" + name);
  int fd = os_open_wr(tmp.c_str());
  if (fd < 0) return false;
  bool ok = wr_all(fd, p, n) && os_sync(fd) == 0;
  if (ok) os_drop_cache(fd);  // clean now: don't let the page cache keep it
  if (os_close(fd)) ok = false;
  std::error_code ec;
  if (ok) {
    std::filesystem::rename(tmp, path_of(name), ec);
    ok = !ec;
  }
  if (!ok) std::filesystem::remove(tmp, ec);
  return ok;
}
// Whole file into a pooled buffer; `want` = exact expected size (0 = any,
// bounded by `max`).
static bool read_file(const std::string& path, HBuf& b, size_t want, size_t max = 0) {
  int fd = os_open_rd(path.c_str());
  if (fd < 0) return false;
  const int64_t fsz = os_fsize(fd);
  bool ok = fsz > 0 && (want ? (uint64_t)fsz == want : (uint64_t)fsz <= max);
  if (ok) {
    b = buf_get((size_t)fsz);
    ok = b.p && rd_all(fd, b.data(), (size_t)fsz);
  }
  os_drop_cache(fd);
  os_close(fd);
  return ok;
}
static bool read_prefix(const std::string& path, void* dst, size_t n, uint64_t* fsz) {
  int fd = os_open_rd(path.c_str());
  if (fd < 0) return false;
  const int64_t s = os_fsize(fd);
  const bool ok = s >= (int64_t)n && rd_all(fd, (char*)dst, n);
  os_close(fd);
  *fsz = (uint64_t)std::max<int64_t>(s, 0);
  return ok;
}

// ---- in-memory index ---------------------------------------------------------
// Pages and MTP sections are shared by every checkpoint whose prefix covers
// them (refcounted); checkpoints are the LRU unit. All under g_mtx.
enum { ST_PEND, ST_PUB, ST_FAIL };
struct Blk {
  uint64_t bytes = 0;
  int refs = 0;
  int st = ST_PEND;
};
struct CkEnt {
  std::string name;
  uint64_t seed = 0, base = 0, id = 0, bytes = 0;
  int n = 0, nc = 0, nl = 0;
  bool mtp = false, pub = false;
  std::vector<uint64_t> chunks, mtps;  // mtps[i] == 0: not referenced
  std::vector<int> tail;               // tokens [256*nc, n)
  int64_t t = 0;                       // LRU time (file_time clock)
  int busy = 0;                        // restores in progress: not evictable
};
static std::mutex g_mtx;
static std::condition_variable g_cv;
static std::unordered_map<uint64_t, Blk> g_ch, g_mt;
static std::vector<std::shared_ptr<CkEnt>> g_ck;
static uint64_t g_total = 0;  // bytes of published files

static void unref_locked(std::unordered_map<uint64_t, Blk>& mp, uint64_t key, bool mtp) {
  auto it = mp.find(key);
  if (it == mp.end() || --it->second.refs > 0) return;
  if (it->second.st == ST_PEND) return;  // the writer deletes it on publish
  if (it->second.st == ST_PUB) {
    rm_file(mtp ? mt_name(key) : ch_name(key));
    g_total -= it->second.bytes;
  }
  mp.erase(it);
}
static void drop_ck_locked(std::shared_ptr<CkEnt> e, bool del_file) {
  auto it = std::find(g_ck.begin(), g_ck.end(), e);
  if (it == g_ck.end()) return;
  g_ck.erase(it);
  if (e->pub) {
    g_total -= e->bytes;
    if (del_file) rm_file(e->name);
  }
  for (uint64_t k : e->chunks) unref_locked(g_ch, k, false);
  for (uint64_t k : e->mtps)
    if (k) unref_locked(g_mt, k, true);
}
static void evict_locked(const CkEnt* keep) {
  while (g_total > g_cap) {
    std::shared_ptr<CkEnt> v;
    for (auto& e : g_ck)
      if (e->pub && !e->busy && e.get() != keep && (!v || e->t < v->t)) v = e;
    if (!v) break;
    fprintf(stderr, "kvsnap: evicted %s (LRU, cap %.1f GB)\n", v->name.c_str(),
            g_cap / 1073741824.0);
    drop_ck_locked(v, true);
  }
}
static void touch_locked(const std::shared_ptr<CkEnt>& e) {
  e->t = now_t();
  std::error_code ec;  // the file mtime carries the LRU order across restarts
  std::filesystem::last_write_time(path_of(e->name),
                                   std::filesystem::file_time_type::clock::now(), ec);
}

// ---- background writer -------------------------------------------------------
// One detached thread writes captured images in FIFO order: a commit enqueues
// its pages, then its MTP sections, then its checkpoints, so a checkpoint is
// published only after the pages it names (checked again before writing).
struct Item {
  int kind = 0;  // 0 trunk page, 1 MTP section, 2 checkpoint
  uint64_t key = 0;
  HBuf buf;
  std::shared_ptr<CkEnt> ck;
  double cap_s = 0;
  int new_ch = 0, new_mt = 0;
  uint64_t new_b = 0;
};
static std::deque<Item> g_q;
static uint64_t g_q_bytes = 0;  // queued + being written
static bool g_wbusy = false;

static void writer() {
  for (;;) {
    Item it;
    {
      std::unique_lock<std::mutex> lk(g_mtx);
      g_cv.wait(lk, [] { return !g_q.empty(); });
      it = std::move(g_q.front());
      g_q.pop_front();
      g_wbusy = true;
    }
    const size_t len = it.buf.n;
    char* p = it.buf.data();
    std::string name;
    bool go = true;
    if (it.kind == 2) {
      std::lock_guard<std::mutex> lk(g_mtx);
      for (uint64_t k : it.ck->chunks) {
        auto f = g_ch.find(k);
        if (f == g_ch.end() || f->second.st != ST_PUB) go = false;
      }
      name = it.ck->name;
    } else {
      name = it.kind ? mt_name(it.key) : ch_name(it.key);
    }
    const size_t hb = it.kind == 2 ? sizeof(CkHdr) : sizeof(ChHdr);
    const uint64_t sum = sum64(p + hb, len - hb);
    memcpy(p + (it.kind == 2 ? offsetof(CkHdr, sum) : offsetof(ChHdr, sum)), &sum, 8);
    const auto w0 = std::chrono::steady_clock::now();
    const bool ok = go && write_file(name, p, len);
    const double ws =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      g_q_bytes -= len;
      if (it.kind < 2) {
        auto& mp = it.kind ? g_mt : g_ch;
        auto f = mp.find(it.key);
        if (f == mp.end() || f->second.refs == 0) {
          if (ok) rm_file(name);  // every referencing checkpoint is gone
          if (f != mp.end()) mp.erase(f);
        } else if (ok) {
          f->second.st = ST_PUB;
          f->second.bytes = len;
          g_total += len;
        } else {
          f->second.st = ST_FAIL;  // erased with its last reference
          fprintf(stderr, "kvsnap: write of %s failed (non-fatal)\n", name.c_str());
        }
      } else {
        const auto& e = it.ck;
        if (!ok) {
          fprintf(stderr, "kvsnap: save of %d tokens failed (%s, non-fatal)\n", e->n,
                  go ? "write error" : "a page write failed");
          drop_ck_locked(e, false);
        } else {
          // same state saved before without MTP state: this file replaced it
          for (auto& o : g_ck)
            if (o != e && o->name == e->name) {
              drop_ck_locked(o, false);
              break;
            }
          e->pub = true;
          e->bytes = len;
          e->t = now_t();
          g_total += len;
          fprintf(stderr,
                  "%s kvsnap: saved %d tokens -> %s (ckpt %.1f MB + %d new pages%s "
                  "%.1f MB; capture %.2fs, write %.2fs%s; total %.2f GB)\n",
                  log_ts(), e->n, e->name.c_str(), len / 1048576.0, it.new_ch,
                  it.new_mt ? " + mtp" : "", it.new_b / 1048576.0, it.cap_s, ws,
                  e->mtp ? ", mtp" : "", g_total / 1073741824.0);
          evict_locked(e.get());
        }
      }
      g_wbusy = false;
    }
    g_cv.notify_all();
    buf_put(std::move(it.buf));
  }
}
static void enqueue(Item&& it) {
  std::unique_lock<std::mutex> lk(g_mtx);
  // bounded: the GPU thread waits for the disk rather than hoard host memory
  g_cv.wait(lk, [&] { return g_q_bytes == 0 || g_q_bytes + it.buf.n <= g_qcap; });
  g_q_bytes += it.buf.n;
  g_q.push_back(std::move(it));
  lk.unlock();
  g_cv.notify_all();
}

// Requests in flight: a turn commits after its reply went out, so KVSYNC
// also waits for those (gen_one holds a Turn for its whole run).
static int g_turns = 0;
struct Turn {
  Turn() {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_turns++;
  }
  ~Turn() {
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      g_turns--;
    }
    g_cv.notify_all();
  }
};

// KVSYNC: wait for in-flight turns and the writer to drain;
// "K <ckpts> <pages> <mtp> <bytes>".
static std::string sync_line() {
  std::unique_lock<std::mutex> lk(g_mtx);
  g_cv.wait(lk, [] { return g_q.empty() && !g_wbusy && !g_turns; });
  size_t nck = 0;
  for (auto& e : g_ck) nck += e->pub;
  char b[128];
  snprintf(b, sizeof b, "K %zu %zu %zu %llu\n", nck, g_ch.size(), g_mt.size(),
           (unsigned long long)g_total);
  return b;
}

// ---- save --------------------------------------------------------------------
// A checkpoint captured at a cut or at the end of a turn; its trunk pages are
// captured by commit(), once, and only those not already on disk.
struct Pending {
  std::shared_ptr<CkEnt> ck;
  HBuf buf;  // checkpoint file image (the writer fills in the checksum)
  double cap_s = 0;
};

// GPU-owner thread. `toks` must be exactly the live state (m.pos). `mtp`: the
// MTP layer is in lockstep (mtp_pos == pos-1, rows [0, pos-1) ingested); the
// checkpoint then carries d_R/d_miraw/mtp_pos and names MTP sections.
static void capture(GpuModel& m, const std::vector<int>& toks, bool mtp,
                    std::vector<Pending>& out) {
  if (!g_on) return;
  const int n = (int)toks.size();
  if (n <= 0 || n != m.pos) {
    fprintf(stderr, "kvsnap: save skipped: %d tokens but live pos %d\n", n, m.pos);
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const Geo G = geo(m);
  mtp = mtp && m.kv_nl > G.nqsa && m.mtp_pos == m.pos - 1;
  const int nc = (n - 1) / KV_PAGE, nt = n - nc * KV_PAGE;
  auto e = std::make_shared<CkEnt>();
  e->seed = seed_of(m, toks.data(), (size_t)n);
  std::vector<uint64_t> ch = chain(e->seed, toks.data(), nc);
  e->base = ch[(size_t)nc];
  e->id = ck_id(e->seed, toks.data(), (size_t)n);
  e->n = n;
  e->nc = nc;
  e->nl = mtp ? G.nqsa + 1 : G.nqsa;
  e->mtp = mtp;
  e->chunks.assign(ch.begin() + 1, ch.end());
  e->mtps.assign((size_t)nc, 0);
  if (mtp)
    for (int i = 0; i < nc; i++)
      e->mtps[(size_t)i] = mtp_key(ch[(size_t)i + 1], toks[(size_t)(i + 1) * KV_PAGE]);
  e->tail.assign(toks.begin() + (size_t)nc * KV_PAGE, toks.end());
  e->name = ck_name(n, e->id);
  CkHdr h{};
  memcpy(h.magic, MAG_K, 8);
  h.seed = e->seed;
  h.base = e->base;
  h.id = e->id;
  h.n = n;
  h.nc = nc;
  h.nl = e->nl;
  h.kvb = G.kvb;
  h.ringpos = m.ple_ringpos;
  h.mtp = mtp;
  h.mtp_pos = mtp ? m.mtp_pos : 0;
  h.nik = tail_ik(m, n, nc);
  h.has_ik = G.has_ik;
  h.s_b = G.s_b;
  h.conv_b = G.conv_b;
  h.ring_b = G.ring_b;
  h.iraw_b = G.iraw_b;
  h.mR_b = mtp ? (uint64_t)g_cfg.branches * g_cfg.d * 4 : 0;
  h.miraw_b = mtp && m.d_miraw ? 4 * 128 * 4 : 0;
  const size_t len = sizeof h + ck_body_b(h);
  HBuf b = buf_get(len);
  if (!b.p) return;
  char* p = b.data();
  memcpy(p, &h, sizeof h);
  p += sizeof h;
  memcpy(p, e->chunks.data(), (size_t)nc * 8);
  p += (size_t)nc * 8;
  memcpy(p, e->mtps.data(), (size_t)nc * 8);
  p += (size_t)nc * 8;
  memcpy(p, e->tail.data(), (size_t)nt * 4);
  p += (size_t)nt * 4;
  CK(hipStreamSynchronize(g_str));
  bool ok = true;
  auto blob = [&](const void* d, size_t sz) {
    if (ok) ok = d2h(p, d, sz);
    p += sz;
  };
  blob(m.d_S, h.s_b);
  blob(m.d_convst, h.conv_b);
  blob(m.d_ring, h.ring_b);
  blob(m.d_iraw, h.iraw_b);
  blob(m.d_R, h.mR_b);
  blob(m.d_miraw, h.miraw_b);
  const int r0 = nc * KV_PAGE, b0 = nc * KV_PAGE_BLOCKS;
  for (int l = 0; l < h.nl && ok; l++) {
    ok = xfer(m, false, kv_layer(m, 0, l), p, r0, n, KV_PAGE, G.row_b) &&
         xfer(m, false, kv_layer(m, 1, l), p + (size_t)nt * G.row_b, r0, n, KV_PAGE,
              G.row_b) &&
         xfer(m, false, G.has_ik ? ik_layer(m, l) : nullptr,
              p + 2 * (size_t)nt * G.row_b, b0, b0 + h.nik, KV_PAGE_BLOCKS, 512);
    p += 2 * (size_t)nt * G.row_b + (size_t)h.nik * 512;
  }
  if (!ok) {
    fprintf(stderr, "kvsnap: capture of %d tokens failed (non-fatal)\n", n);
    return;
  }
  Pending pd;
  pd.ck = std::move(e);
  pd.buf = std::move(b);
  pd.cap_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.push_back(std::move(pd));
}

// One page's (or MTP section's) file image, read from the live sequence.
static bool page_image(GpuModel& m, const Geo& G, bool mtp, int i, uint64_t key,
                       uint64_t parent, int look, const int* toks, HBuf& b) {
  const int l0 = mtp ? G.nqsa : 0, nl = mtp ? 1 : G.nqsa;
  const size_t lb = page_layer_b(G.kvb, G.has_ik);
  b = buf_get(sizeof(ChHdr) + (size_t)nl * lb);
  if (!b.p) return false;
  ChHdr h{};
  memcpy(h.magic, mtp ? MAG_M : MAG_C, 8);
  h.key = key;
  h.parent = parent;
  h.nl = nl;
  h.kvb = G.kvb;
  h.has_ik = G.has_ik;
  h.look = look;
  memcpy(h.toks, toks, sizeof h.toks);
  memcpy(b.data(), &h, sizeof h);
  char* p = b.data() + sizeof h;
  const int r0 = i * KV_PAGE, k0 = i * KV_PAGE_BLOCKS;
  bool ok = true;
  for (int l = l0; l < l0 + nl && ok; l++, p += lb)
    ok = xfer(m, false, kv_layer(m, 0, l), p, r0, r0 + KV_PAGE, KV_PAGE, G.row_b) &&
         xfer(m, false, kv_layer(m, 1, l), p + (size_t)KV_PAGE * G.row_b, r0,
              r0 + KV_PAGE, KV_PAGE, G.row_b) &&
         (!G.has_ik || xfer(m, false, ik_layer(m, l), p + 2 * (size_t)KV_PAGE * G.row_b,
                            k0, k0 + KV_PAGE_BLOCKS, KV_PAGE_BLOCKS, 512));
  return ok;
}

// End of turn (GPU-owner thread, `hist` == the live state): register the
// pending checkpoints, capture the pages/MTP sections none of the cached
// files hold yet, and hand everything to the writer. Every pending state is
// a prefix of `hist` (cut states of this prompt), so the live KV still holds
// their pages: KV rows and MTP rows below each checkpoint are append-only.
// All of `pend` shares `seed` (commit() groups them).
static void commit_seed(GpuModel& m, const std::vector<int>& hist,
                        std::vector<Pending>& pend, uint64_t seed) {
  if (!g_on || pend.empty() || (int)hist.size() != m.pos) {
    pend.clear();
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const Geo G = geo(m);
  std::sort(pend.begin(), pend.end(),
            [](const Pending& a, const Pending& b) { return a.ck->n < b.ck->n; });
  const int np = (int)((hist.size() - 1) / KV_PAGE);
  const std::vector<uint64_t> ch = chain(seed, hist.data(), np);
  std::vector<char> need_ch((size_t)np, 0), need_mt((size_t)np, 0);
  std::vector<int> owner_ch((size_t)np, -1), owner_mt((size_t)np, -1);
  std::vector<size_t> keep;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    for (size_t pi = 0; pi < pend.size(); pi++) {
      auto& e = pend[pi].ck;
      if (e->seed != seed || e->nc > np || ch[(size_t)e->nc] != e->base ||
          !std::equal(e->tail.begin(), e->tail.end(),
                      hist.begin() + (size_t)e->nc * KV_PAGE))
        continue;  // not a prefix of the final state (should not happen)
      std::shared_ptr<CkEnt> old;
      for (auto& o : g_ck)
        if (o->name == e->name) old = o;
      if (old && (old->mtp || !e->mtp)) {  // already cached (or queued)
        if (old->pub) touch_locked(old);
        continue;
      }
      bool bad = false;  // a failed page write whose references linger
      for (uint64_t k : e->chunks) {
        auto f = g_ch.find(k);
        if (f != g_ch.end() && f->second.st == ST_FAIL) bad = true;
      }
      if (bad) continue;
      for (int i = 0; i < e->nc; i++) {
        auto r = g_ch.try_emplace(e->chunks[(size_t)i]);
        r.first->second.refs++;
        if (r.second) need_ch[(size_t)i] = 1, owner_ch[(size_t)i] = (int)pi;
      }
      for (int i = 0; i < e->nc; i++) {
        uint64_t& k = e->mtps[(size_t)i];
        if (!k) continue;
        auto f = g_mt.find(k);
        if (f != g_mt.end() && f->second.st == ST_FAIL) {
          k = 0;  // restores then find the section missing: no MTP state
          continue;
        }
        auto r = g_mt.try_emplace(k);
        r.first->second.refs++;
        if (r.second) need_mt[(size_t)i] = 1, owner_mt[(size_t)i] = (int)pi;
      }
      g_ck.push_back(e);  // pending (pub = false): dedupes the next commit
      keep.push_back(pi);
    }
  }
  std::vector<int> nch(pend.size(), 0), nmt(pend.size(), 0);
  std::vector<uint64_t> nb(pend.size(), 0);
  auto fail_blk = [&](std::unordered_map<uint64_t, Blk>& mp, uint64_t k) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto f = mp.find(k);
    if (f != mp.end()) f->second.st = ST_FAIL;
  };
  CK(hipStreamSynchronize(g_str));
  for (int pass = 0; pass < 2; pass++)
    for (int i = 0; i < np; i++) {
      if (!(pass ? need_mt : need_ch)[(size_t)i]) continue;
      const int o = (pass ? owner_mt : owner_ch)[(size_t)i];
      Item it;
      it.kind = pass;
      it.key = pass ? pend[(size_t)o].ck->mtps[(size_t)i] : ch[(size_t)i + 1];
      const uint64_t parent = pass ? ch[(size_t)i + 1] : ch[(size_t)i];
      const int look = pass ? hist[(size_t)(i + 1) * KV_PAGE] : -1;
      if (!page_image(m, G, pass == 1, i, it.key, parent, look,
                      hist.data() + (size_t)i * KV_PAGE, it.buf)) {
        fprintf(stderr, "kvsnap: capture of page %d failed (non-fatal)\n", i);
        fail_blk(pass ? g_mt : g_ch, it.key);
        continue;
      }
      (pass ? nmt : nch)[(size_t)o]++;
      nb[(size_t)o] += it.buf.n;
      enqueue(std::move(it));
    }
  const double pages_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  for (size_t k = 0; k < keep.size(); k++) {
    Pending& pd = pend[keep[k]];
    Item it;
    it.kind = 2;
    it.ck = pd.ck;
    it.buf = std::move(pd.buf);
    it.cap_s = pd.cap_s + (k + 1 == keep.size() ? pages_s : 0.0);
    // log the new pages with the checkpoint that first needed them
    it.new_ch = nch[keep[k]];
    it.new_mt = nmt[keep[k]];
    it.new_b = nb[keep[k]];
    enqueue(std::move(it));
  }
  for (auto& pd : pend) buf_put(std::move(pd.buf));  // deduped images
  pend.clear();
}

// A vision request's state before its first image is keyed as text
// (seed_of), later states with its image seed: one pass per seed, since the
// page keys chain from the seed (pages below the first image may then be
// stored under both).
static void commit(GpuModel& m, const std::vector<int>& hist, std::vector<Pending>& pend) {
  std::vector<Pending> img;
  for (size_t i = 0; i < pend.size();) {
    if (pend[i].ck->seed != g_fp) {
      img.push_back(std::move(pend[i]));
      pend.erase(pend.begin() + (std::ptrdiff_t)i);
    } else {
      i++;
    }
  }
  commit_seed(m, hist, pend, g_fp);
  commit_seed(m, hist, img, seed_for(m.mrope_active_, m.img_hash_, m.mrope_grids_));
}

// ---- restore -----------------------------------------------------------------
// Longest published checkpoint whose tokens are a strict prefix of `prompt`
// (strictly shorter: a full-length match has no logits to resume from).
// `seed2` (optional): a second seed to search, e.g. g_fp for the text
// prefix of a vision prompt. The returned entry is marked busy until
// restore() releases it.
static std::shared_ptr<CkEnt> match(const std::vector<int>& prompt, uint64_t seed,
                                    uint64_t seed2 = 0) {
  if (!g_on || prompt.size() < 2) return nullptr;
  if (seed2 == seed) seed2 = 0;
  const int np = (int)((prompt.size() - 1) / KV_PAGE);
  const std::vector<uint64_t> ch = chain(seed, prompt.data(), np);
  const std::vector<uint64_t> ch2 =
      seed2 ? chain(seed2, prompt.data(), np) : std::vector<uint64_t>();
  std::lock_guard<std::mutex> lk(g_mtx);
  std::shared_ptr<CkEnt> best;
  for (auto& e : g_ck) {
    if (!e->pub || (e->seed != seed && (!seed2 || e->seed != seed2))) continue;
    const std::vector<uint64_t>& c = e->seed == seed ? ch : ch2;
    if ((size_t)e->n >= prompt.size() ||
        (best && e->n <= best->n) || e->nc > np || c[(size_t)e->nc] != e->base ||
        !std::equal(e->tail.begin(), e->tail.end(),
                    prompt.begin() + (size_t)e->nc * KV_PAGE))
      continue;
    best = e;
  }
  if (best) best->busy++;
  return best;
}

struct Rd {
  HBuf b;
  bool ok = false;
};
// Read + verify one page / MTP section file (runs on a prefetch thread).
static Rd read_page(const std::string& name, bool mtp, uint64_t key, uint64_t parent,
                    int nl, int kvb, int has_ik, const int* toks, int look) {
  Rd r;
  const size_t len = sizeof(ChHdr) + (size_t)nl * page_layer_b(kvb, has_ik);
  if (!read_file(path_of(name), r.b, len)) return r;
  ChHdr h;
  memcpy(&h, r.b.data(), sizeof h);
  r.ok = !memcmp(h.magic, mtp ? MAG_M : MAG_C, 8) && h.key == key && h.parent == parent &&
         h.nl == nl && h.kvb == kvb && h.has_ik == has_ik && h.look == look &&
         !memcmp(h.toks, toks, sizeof h.toks) &&
         h.sum == sum64(r.b.data() + sizeof h, len - sizeof h);
  return r;
}

enum { RS_OK, RS_CKPT, RS_PAGE, RS_OTHER };
// Pages stream in on a prefetch thread (read + checksum) while the previous
// one is copied to the device. *bad_page: the trunk page that failed.
static int restore_body(GpuModel& m, const CkEnt& e, const std::vector<int>& prompt,
                        bool* mtp_out, uint64_t* bad_page, int* npages) {
  const Geo G = geo(m);
  HBuf cb;
  if (!read_file(path_of(e.name), cb, 0, (size_t)1 << 40)) return RS_CKPT;
  struct Ret {  // pooled buffer back on every exit
    HBuf& b;
    ~Ret() { buf_put(std::move(b)); }
  } ret{cb};
  CkHdr h;
  if (cb.n < sizeof h) return RS_CKPT;
  memcpy(&h, cb.data(), sizeof h);
  if (!ck_sane(h) || cb.n != sizeof h + ck_body_b(h) || h.id != e.id || h.n != e.n ||
      h.seed != e.seed || h.base != e.base ||
      h.sum != sum64(cb.data() + sizeof h, cb.n - sizeof h))
    return RS_CKPT;
  const int n = h.n, nc = h.nc, nt = n - nc * KV_PAGE;
  if (n > m.maxctx || h.kvb != G.kvb || h.has_ik != G.has_ik || h.s_b != G.s_b ||
      h.conv_b != G.conv_b || h.ring_b != G.ring_b || h.iraw_b != G.iraw_b ||
      h.nl < G.nqsa || h.nl > G.nqsa + 1 || (h.nl > G.nqsa && !h.mtp))
    return RS_OTHER;
  const char* p = cb.data() + sizeof h;
  std::vector<uint64_t> chunks((size_t)nc), mtps((size_t)nc);
  memcpy(chunks.data(), p, (size_t)nc * 8);
  memcpy(mtps.data(), p + (size_t)nc * 8, (size_t)nc * 8);
  p += (size_t)nc * 16;
  if (memcmp(p, prompt.data() + (size_t)nc * KV_PAGE, (size_t)nt * 4)) return RS_CKPT;
  p += (size_t)nt * 4;
  bool want_mtp = h.mtp && h.nl == G.nqsa + 1 && m.kv_nl > G.nqsa &&
                  h.mtp_pos == n - 1 &&
                  h.mR_b == (uint64_t)g_cfg.branches * g_cfg.d * 4 &&
                  h.miraw_b == (m.d_miraw ? 4 * 128 * 4u : 0u);
  if (want_mtp) {
    std::lock_guard<std::mutex> lk(g_mtx);
    for (uint64_t k : mtps) {
      auto f = g_mt.find(k);
      if (!k || f == g_mt.end() || f->second.st != ST_PUB) want_mtp = false;
    }
  }
  // the restored state replaces the live sequence: map fresh rows [0, n)
  m.kv_trim(0);
  m.kv_reserve(n);
  const std::vector<uint64_t> ch = chain(e.seed, prompt.data(), nc);
  const int njobs = nc + (want_mtp ? nc : 0);
  auto job = [&](int j) {
    const bool mt = j >= nc;
    const int i = mt ? j - nc : j;
    const int* toks = prompt.data() + (size_t)i * KV_PAGE;
    return std::async(std::launch::async, read_page,
                      mt ? mt_name(mtps[(size_t)i]) : ch_name(chunks[(size_t)i]), mt,
                      mt ? mtps[(size_t)i] : chunks[(size_t)i],
                      mt ? chunks[(size_t)i] : ch[(size_t)i], mt ? 1 : G.nqsa, G.kvb,
                      G.has_ik, toks, mt ? prompt[(size_t)(i + 1) * KV_PAGE] : -1);
  };
  std::future<Rd> fut;
  if (njobs) fut = job(0);
  const size_t lb = page_layer_b(G.kvb, G.has_ik);
  for (int j = 0; j < njobs; j++) {
    Rd r = fut.get();
    if (j + 1 < njobs) fut = job(j + 1);
    const bool mt = j >= nc;
    const int i = mt ? j - nc : j;
    if (!mt && chunks[(size_t)i] != ch[(size_t)i + 1]) r.ok = false;
    if (!r.ok) {
      buf_put(std::move(r.b));
      if (fut.valid()) fut.wait();
      if (!mt) {
        *bad_page = chunks[(size_t)i];
        return RS_PAGE;
      }
      want_mtp = false;  // the trunk is complete; only speculation is lost
      break;
    }
    char* q = r.b.data() + sizeof(ChHdr);
    const int l0 = mt ? G.nqsa : 0, nl = mt ? 1 : G.nqsa;
    const int r0 = i * KV_PAGE, k0 = i * KV_PAGE_BLOCKS;
    bool ok = true;
    for (int l = l0; l < l0 + nl && ok; l++, q += lb)
      ok = xfer(m, true, kv_layer(m, 0, l), q, r0, r0 + KV_PAGE, KV_PAGE, G.row_b) &&
           xfer(m, true, kv_layer(m, 1, l), q + (size_t)KV_PAGE * G.row_b, r0,
                r0 + KV_PAGE, KV_PAGE, G.row_b) &&
           (!G.has_ik || xfer(m, true, ik_layer(m, l), q + 2 * (size_t)KV_PAGE * G.row_b,
                              k0, k0 + KV_PAGE_BLOCKS, KV_PAGE_BLOCKS, 512));
    buf_put(std::move(r.b));
    if (!ok) {
      if (fut.valid()) fut.wait();
      return RS_OTHER;
    }
  }
  *npages = nc;
  // checkpoint: GDN/PLE/indexer state, then the partial page of every layer
  char* src = (char*)p;
  bool ok = true;
  auto blob = [&](void* d, size_t sz, bool use) {
    if (ok && use) ok = h2d(d, src, sz);
    src += sz;
  };
  blob(m.d_S, h.s_b, true);
  blob(m.d_convst, h.conv_b, true);
  blob(m.d_ring, h.ring_b, true);
  blob(m.d_iraw, h.iraw_b, true);
  blob(m.d_R, h.mR_b, want_mtp);
  blob(m.d_miraw, h.miraw_b, want_mtp);
  const int r0 = nc * KV_PAGE, k0 = nc * KV_PAGE_BLOCKS;
  const int nik = std::min(
      h.nik, std::max(0, std::min(KV_PAGE_BLOCKS * ((n + KV_PAGE - 1) / KV_PAGE),
                                  m.index_blocks) - k0));
  const int nl = want_mtp ? h.nl : G.nqsa;
  for (int l = 0; l < nl && ok; l++) {
    ok = xfer(m, true, kv_layer(m, 0, l), src, r0, n, KV_PAGE, G.row_b) &&
         xfer(m, true, kv_layer(m, 1, l), src + (size_t)nt * G.row_b, r0, n, KV_PAGE,
              G.row_b) &&
         (!G.has_ik || xfer(m, true, ik_layer(m, l), src + 2 * (size_t)nt * G.row_b, k0,
                            k0 + nik, KV_PAGE_BLOCKS, 512));
    src += 2 * (size_t)nt * G.row_b + (size_t)h.nik * 512;
  }
  if (!ok) return RS_OTHER;
  if (want_mtp) {
    m.mtp_pos = h.mtp_pos;
    CK(hipMemcpy(m.d_mpos, &m.mtp_pos, 4, hipMemcpyHostToDevice));
  }
  // GDEC_QSA_WMMA_BTV: the transposed prefill V cache is derived state (not
  // saved); rebuild it for [0, n) from the restored row-major V, or sparse
  // prefill after the restore would read stale V for the prefix. Paged:
  // per physically contiguous run; a page holds whole 4-token blocks, so
  // physical row block pr/4 is exactly the physical vct block of that run.
  if (m.qsa_wmma_btv) {
    for (int qi = 0; qi < G.nqsa; qi++)
      m.kv_runs(0, n, KV_PAGE, [&](int, size_t prz, int cnt) {
        const int pr = (int)prz;
        const unsigned grid =
            (unsigned)((((size_t)(pr + cnt + 3) / 4 - (size_t)pr / 4) * 512 + 255) / 256);
        k_bf16_rows_to_bt<<<grid, 256, 0, g_str>>>(
            m.d_vcb + (size_t)qi * m.kv_layer_rows * 512, m.vct_layer(qi), pr, pr + cnt);
      });
    CK(hipGetLastError());
    CK(hipStreamSynchronize(g_str));
  }
  // host-side and scalar state: pos/ringpos (both mirrors), PLE token history
  m.pos = n;
  m.ple_ringpos = h.ringpos;
  CK(hipMemcpy(m.d_pos, &m.pos, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(m.d_ringpos, &m.ple_ringpos, 4, hipMemcpyHostToDevice));
  m.chist.assign(prompt.begin(), prompt.begin() + n);
  *mtp_out = want_mtp;
  return RS_OK;
}

// Restore `e` (from match()) for `prompt`: hist := its tokens. *mtp_out: the
// MTP layer is back in lockstep (speculation may continue). On failure the
// live state is undefined: the caller resets it.
static bool restore(GpuModel& m, const std::shared_ptr<CkEnt>& e,
                    const std::vector<int>& prompt, std::vector<int>& hist,
                    bool* mtp_out) {
  *mtp_out = false;
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t bad = 0;
  int npages = 0;
  const int rc = restore_body(m, *e, prompt, mtp_out, &bad, &npages);
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    e->busy--;
    if (rc == RS_OK) {
      touch_locked(e);
    } else if (rc == RS_CKPT) {
      drop_ck_locked(e, true);
    } else if (rc == RS_PAGE) {
      // a damaged page invalidates every checkpoint built on it
      std::vector<std::shared_ptr<CkEnt>> v;
      for (auto& o : g_ck)
        if (o->pub && !o->busy && std::count(o->chunks.begin(), o->chunks.end(), bad))
          v.push_back(o);
      for (auto& o : v) drop_ck_locked(o, true);
    }
  }
  const double s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (rc != RS_OK) {
    fprintf(stderr, "%s kvsnap: restore of %s failed (%s, non-fatal)\n", log_ts(),
            e->name.c_str(),
            rc == RS_CKPT ? "bad checkpoint" : rc == RS_PAGE ? "bad page" : "device/geometry");
    return false;
  }
  hist.assign(prompt.begin(), prompt.begin() + e->n);
  fprintf(stderr, "%s kvsnap: restored %d tokens from %s in %.2fs (%d pages)%s\n",
          log_ts(), e->n, e->name.c_str(), s, npages, *mtp_out ? " (mtp live)" : "");
  return true;
}

// ---- startup -----------------------------------------------------------------
static uint64_t fingerprint(const GpuModel& m) {
  const Geo G = geo(m);
  uint64_t h = mix64(0x6764656b76736e70ULL + FMT);
  auto add = [&](uint64_t v) { h = mix64(h ^ v); };
  for (const auto& f : g_model_files) add(file_id(f));
  for (uint64_t v : {(uint64_t)m.ple_on, (uint64_t)G.kvb, (uint64_t)G.has_ik,
                     (uint64_t)G.nqsa, (uint64_t)g_cfg.layers, (uint64_t)g_cfg.d,
                     (uint64_t)g_cfg.branches, (uint64_t)KV_PAGE})
    add(v);
  return h;
}
static bool ends_with(const std::string& s, const char* suf) {
  const size_t k = strlen(suf);
  return s.size() >= k && !s.compare(s.size() - k, k, suf);
}
static std::shared_ptr<CkEnt> load_ck(const std::string& fn) {
  CkHdr h;
  uint64_t fsz = 0;
  if (!read_prefix(path_of(fn), &h, sizeof h, &fsz) || !ck_sane(h) ||
      fsz != sizeof h + ck_body_b(h) || fn != ck_name(h.n, h.id))
    return nullptr;
  const size_t nt = (size_t)h.n - (size_t)h.nc * KV_PAGE;
  std::vector<char> v(sizeof h + (size_t)h.nc * 16 + nt * 4);
  if (!read_prefix(path_of(fn), v.data(), v.size(), &fsz)) return nullptr;
  auto e = std::make_shared<CkEnt>();
  e->name = fn;
  e->seed = h.seed;
  e->base = h.base;
  e->id = h.id;
  e->bytes = fsz;
  e->n = h.n;
  e->nc = h.nc;
  e->nl = h.nl;
  e->mtp = h.mtp != 0;
  e->pub = true;
  e->chunks.resize((size_t)h.nc);
  e->mtps.resize((size_t)h.nc);
  e->tail.resize(nt);
  memcpy(e->chunks.data(), v.data() + sizeof h, (size_t)h.nc * 8);
  memcpy(e->mtps.data(), v.data() + sizeof h + (size_t)h.nc * 8, (size_t)h.nc * 8);
  memcpy(e->tail.data(), v.data() + sizeof h + (size_t)h.nc * 16, nt * 4);
  return e;
}

// Index the cache directory (serve::run, before the first request). Torn
// temporaries, pre-A4 snapshot directories, unreadable files and checkpoints
// missing a trunk page are removed; unreferenced pages too.
static void init(GpuModel& m, Cfg& cfg) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(cfg.dir, ec);
  if (!fs::is_directory(cfg.dir, ec)) {
    fprintf(stderr, "kvsnap: cannot use %s: disabled\n", cfg.dir.c_str());
    cfg.dir.clear();
    return;
  }
  g_dir = cfg.dir;
  g_cap = cfg.cap_bytes;
  g_qcap = cfg.queue_bytes;
  g_fp = fingerprint(m);
  g_vis_id = g_vision_file.empty() ? 0 : file_id(g_vision_file);
  std::vector<std::shared_ptr<CkEnt>> cks;
  size_t legacy = 0, bad = 0;
  std::vector<fs::path> ents;
  for (auto& de : fs::directory_iterator(cfg.dir, ec)) ents.push_back(de.path());
  for (const auto& pth : ents) {
    std::error_code e2;
    const std::string fn = pth.filename().string();
    if (!fn.rfind("tmp_", 0)) {
      fs::remove_all(pth, e2);
    } else if (!fn.rfind("s_", 0) && fs::is_directory(pth, e2)) {
      fs::remove_all(pth, e2);  // KVS1-3 whole-state snapshots
      legacy++;
    } else if ((!fn.rfind("c_", 0) && ends_with(fn, ".kvc")) ||
               (!fn.rfind("m_", 0) && ends_with(fn, ".kvm"))) {
      const bool mt = fn[0] == 'm';
      ChHdr h;
      uint64_t fsz = 0;
      if (!read_prefix(pth.string(), &h, sizeof h, &fsz) ||
          memcmp(h.magic, mt ? MAG_M : MAG_C, 8) || h.nl <= 0 || h.nl > 64 ||
          (h.kvb != 2 && h.kvb != 4) ||
          fsz != sizeof h + (size_t)h.nl * page_layer_b(h.kvb, h.has_ik) ||
          fn != (mt ? mt_name(h.key) : ch_name(h.key))) {
        fs::remove(pth, e2);
        bad++;
        continue;
      }
      Blk& b = (mt ? g_mt : g_ch)[h.key];
      b.bytes = fsz;
      b.st = ST_PUB;
    } else if (!fn.rfind("k_", 0) && ends_with(fn, ".kck")) {
      auto e = load_ck(fn);
      if (!e) {
        fs::remove(pth, e2);
        bad++;
        continue;
      }
      e->t = (int64_t)fs::last_write_time(pth, e2).time_since_epoch().count();
      cks.push_back(e);
    }
  }
  std::lock_guard<std::mutex> lk(g_mtx);
  for (auto& e : cks) {
    bool whole = true;
    for (uint64_t k : e->chunks) whole = whole && g_ch.count(k);
    if (!whole) {
      rm_file(e->name);
      bad++;
      continue;
    }
    for (uint64_t k : e->chunks) g_ch[k].refs++;
    for (uint64_t& k : e->mtps) {
      auto f = g_mt.find(k);
      if (f == g_mt.end()) k = 0;
      else f->second.refs++;
    }
    g_total += e->bytes;
    g_ck.push_back(e);
  }
  size_t orphans = 0;
  for (auto* mp : {&g_ch, &g_mt})
    for (auto it = mp->begin(); it != mp->end();) {
      if (it->second.refs == 0) {
        rm_file(mp == &g_mt ? mt_name(it->first) : ch_name(it->first));
        it = mp->erase(it);
        orphans++;
      } else {
        g_total += it->second.bytes;
        ++it;
      }
    }
  if (legacy)
    fprintf(stderr, "kvsnap: removed %zu pre-A4 snapshot director%s (old format)\n",
            legacy, legacy == 1 ? "y" : "ies");
  if (bad || orphans)
    fprintf(stderr, "kvsnap: removed %zu unusable and %zu unreferenced files\n", bad,
            orphans);
  evict_locked(nullptr);
  fprintf(stderr, "kvsnap: index %zu checkpoints, %zu pages, %zu mtp sections, %.2f GB "
          "(cap %.1f GB, queue %.0f MB)\n",
          g_ck.size(), g_ch.size(), g_mt.size(), g_total / 1073741824.0,
          g_cap / 1073741824.0, g_qcap / 1048576.0);
  g_on = true;
  std::thread(writer).detach();
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
  // CKPT <n> <ids...>: generated tokens worth a mid-decode RAM checkpoint
  // (rckpt), e.g. <tool_call> / </think>. The API's token cache can only
  // splice right after such added tokens; without a checkpoint there a
  // retokenization mismatch inside a tool call forces a re-prefill back to
  // the previous prompt-end cut.
  std::vector<int> ckpt;
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
    } else if (tok == "CKPT") {
      int n;
      if (!(ss >> n) || n < 0 || n > 64) { g.reject = true; return g; }
      for (int i = 0; i < n; i++) {
        int v;
        if (!(ss >> v) || v < 0) { g.reject = true; return g; }
        g.ckpt.push_back(v);
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

// Admission miss diagnostic (under g_sm, before claiming `chosen`): when no
// idle slot's live tokens are a strict prefix of the prompt, log the closest
// one and the tokens around its first mismatch. An lcp deep inside an earlier
// reply means the client re-tokenized history differently from what was
// generated; a short lcp is just another conversation.
static void log_prefix_miss(long long req, const std::vector<int>& ids, int chosen) {
  const std::vector<int>& ch = g_slots[(size_t)chosen].hist;
  if (!ch.empty() && ch.size() < ids.size() &&
      std::equal(ch.begin(), ch.end(), ids.begin()))
    return;  // live continuation
  int best = -1;
  size_t bl = 0;
  for (int i = 0; i < (int)g_slots.size(); i++) {
    const Slot& s = g_slots[(size_t)i];
    if (s.busy || s.hist.empty()) continue;
    size_t l = 0;
    while (l < s.hist.size() && l < ids.size() && s.hist[l] == ids[l]) l++;
    if (best < 0 || l > bl) {
      best = i;
      bl = l;
    }
  }
  if (best < 0) return;
  const std::vector<int>& h = g_slots[(size_t)best].hist;
  std::string why;
  if (bl < h.size() && bl < ids.size()) {
    char b[64];
    why = "first mismatch at " + std::to_string(bl) + ": live";
    for (size_t k = bl; k < h.size() && k < bl + 4; k++) {
      snprintf(b, sizeof b, " %d", h[k]);
      why += b;
    }
    why += " vs prompt";
    for (size_t k = bl; k < ids.size() && k < bl + 4; k++) {
      snprintf(b, sizeof b, " %d", ids[k]);
      why += b;
    }
  } else {  // a strict-prefix slot would have been picked
    why = "prompt ends within the live tokens";
  }
  fprintf(stderr, "%s req %lld: no live prefix; closest slot %d matches %zu "
          "of %zu live tokens (prompt %zu), %s\n", log_ts(), req, best, bl,
          h.size(), ids.size(), why.c_str());
}

// Wait (FIFO) for an idle slot and claim it; -1 if cancelled while queued.
static int slot_admit(const std::vector<int>& ids, const std::atomic<bool>& cancel,
                      long long req) {
  std::unique_lock<std::mutex> lk(g_sm);
  const uint64_t t = g_ticket++;
  g_wait.push_back(t);
  for (;;) {
    if (g_wait.front() == t) {
      const int s = pick_slot(ids);
      if (s >= 0) {
        log_prefix_miss(req, ids, s);
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
  kvsnap::Turn kv_turn;  // KVSYNC waits for this turn's SSD commit
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
  const int s = slot_admit(g.ids, cancel, (long long)g.req);
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
    // Only the images the live prefix holds need the same pixels (its
    // request's grids/hash; none for a text prefix, which is text KV
    // whatever this request's images are — see has_image_tok): a
    // follow-up that keeps the earlier images and adds new ones continues.
    const auto& live_g = m.mrope_grids_;
    const bool live_imgs_same =
        live_g.size() <= g.mrope_grids.size() &&
        std::equal(live_g.begin(), live_g.end(), g.mrope_grids.begin()) &&
        fnv1a_patches(vpatches, live_g.size()) == m.img_hash_;
    if (!g.mrope_grids.empty() && cont &&
        ((!live_imgs_same && has_image_tok(hist.data(), mcp)) ||
         getenv("GDEC_VISION_NOCONT"))) {
      fprintf(stderr, "%s vision: req %lld: live prefix %zu not reused "
              "(its images differ)\n", log_ts(), (long long)g.req, mcp);
      cont = false;
      n_cached = 0;
    }
    if (!cont && rckpt::enabled()) {
      // RAM checkpoint tier (llama.cpp-style PARTIAL_ONLY): only GDN
      // state is stored; validity requires the checkpoint tokens to be a
      // prefix of the live token stream (m.chist), which guarantees the
      // QSA KV/indexer prefix is still intact (paged: the checkpoint owns
      // its pages, any slot may restore it). B0: the pinned pages include
      // the MTP layer, so a checkpoint saved with the MTP layer in lockstep
      // restores it too and speculation stays available.
      const bool is_mrope = !g.mrope_grids.empty();
      const uint64_t req_ih = is_mrope ? fnv1a_patches(vpatches) : 0;
      auto r0 = std::chrono::steady_clock::now();
      int ci = rckpt::match(g.ids, m.chist, is_mrope,
                            is_mrope ? &g.mrope_grids : nullptr, req_ih);
      bool ck_mtp = false;
      if (ci >= 0 && rckpt::restore(m, ci, &hist, &ck_mtp)) {
        mcp = hist.size();
        cont = true;
        n_cached = (int)mcp;
        mtp_live = ck_mtp;
        double rs = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - r0)
                        .count();
        fprintf(stderr, "%s rckpt: restored %zu tokens in %.2fs%s\n",
                log_ts(), mcp, rs, ck_mtp ? " (mtp live)" : "");
      }
    }
    if (!cont && !kvs.dir.empty()) {
      // SSD tier. M-RoPE requests: placeholder ids are pixel-agnostic, so
      // the vision tower, grids and patch-bytes hash are folded into the
      // page-key seed; a different image is simply a cache miss. A
      // checkpoint saved with the MTP layer in lockstep restores it too.
      const bool is_mrope = !g.mrope_grids.empty();
      const uint64_t req_ih = is_mrope ? fnv1a_patches(vpatches) : 0;
      auto ck = kvsnap::match(g.ids, kvsnap::seed_for(is_mrope, req_ih, g.mrope_grids),
                              kvsnap::g_fp);  // + the text prefix of a vision prompt
      bool ck_mtp = false;
      if (ck) {
        if (kvsnap::restore(m, ck, g.ids, hist, &ck_mtp)) {  // hist := its tokens
          mcp = hist.size();
          cont = true;
          n_cached = (int)mcp;
          mtp_live = ck_mtp;
        } else {
          hist.clear();  // live state undefined after a failed restore
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
    // CKPT tokens (e.g. <tool_call>, </think>): pause mid-decode right before
    // one commits and checkpoint that position (rckpt), so the next turn --
    // whose token-cache splice can only cut right after such an added token --
    // falls back a few tokens instead of to the previous prompt-end cut.
    // GDEC_CKPT_TOKENS=0 disables; eos membership wins over a ckpt request.
    std::unordered_set<int> ckpt_toks(g.ckpt.begin(), g.ckpt.end());
    for (int e : g.eos) ckpt_toks.erase(e);
    if (!rckpt::enabled()) ckpt_toks.clear();
    {
      const char* e = getenv("GDEC_CKPT_TOKENS");
      if (e && !atoi(e)) ckpt_toks.clear();
    }
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
    // requests when its KV can be made complete: a fresh prefill, or a
    // restore (rckpt/kvsnap) of a state saved with the MTP layer live.
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
    // prefill always has one; a cont/restored turn only if the previous
    // request (or the restored checkpoint) left MTP live.
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
    // that hit. Checkpoints go to the RAM tier (rckpt) and are captured
    // for the SSD tier (kvsnap); the SSD captures are committed at the end
    // of the turn together with the final state, where only the pages no
    // cached file holds yet are written (content-addressed, shared).
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
    std::vector<kvsnap::Pending> pend;  // SSD captures, committed at turn end
    for (size_t cc : cuts) {
      std::vector<int> seg(g.ids.begin() + done_upto, g.ids.begin() + cc);
      cur = m.prefill_batch(seg, (int)done_upto);
      hist.assign(g.ids.begin(), g.ids.begin() + cc);
      // The segment leaves MTP position cc-1 pending (mtp_pos == pos-1,
      // d_R == h_{cc-1}): checkpoint that lockstep state, then pair h_{cc-1}
      // with token cc, which the next segment's ingest would otherwise skip.
      rckpt::save(m, hist, use_spec || chain_mtp);
      if (!kvs.dir.empty()) kvsnap::capture(m, hist, use_spec || chain_mtp, pend);
      if (use_spec || chain_mtp) {
        std::vector<int> bt{g.ids[cc]};
        m.mtp_ingest_b(m.d_R, bt, 1, (int)cc - 1);
      }
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
    // prompt-end RAM checkpoint (cheap, deduped)
    rckpt::save(m, hist, use_spec || chain_mtp);
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
    // CKPT pause/resume. The emit lambdas below return false on a ckpt token
    // BEFORE sending it; the spec loops roll the round back to exclude it
    // (their eos semantics), leaving the trunk/MTP state exactly at that
    // token's position. ckpt_pause checkpoints that position (the next turn's
    // token-cache splice can only cut right after such an added token, so
    // without this a retokenization mismatch inside a tool call re-prefills
    // from the previous prompt-end cut), then ingests the token serially --
    // MTP boundary pairing first, while d_R is still h_{pos-1} -- and the
    // caller re-enters the loop. The loops reset their spec counters at
    // entry, so they are accumulated across pauses and written back.
    int pause_tok = -1;
    float pause_lp = 0.0f;
    int spec_r = 0, spec_c = 0, spec_p = 0;
    auto ckpt_pause = [&](bool mtp_ok) -> bool {
      hist.assign(m.chist.begin(), m.chist.end());
      if (rckpt::save(m, hist, mtp_ok && mtp_live))
        fprintf(stderr, "%s ckpt: saved %zu tokens (before token %d, req %lld)\n",
                log_ts(), hist.size(), pause_tok, (long long)g.req);
      if (!send_tok(pause_tok, g.sample && g.logprobs, pause_lp)) return false;
      if (g.sample) smp.remember(pause_tok);  // the loop's sampler view is gone
      dl_tick();
      if (mtp_ok && mtp_live) {
        std::vector<int> bt{pause_tok};
        m.mtp_ingest_b(m.d_R, bt, 1, (int)hist.size() - 1);
      }
      cur = m.forward(pause_tok);
      hist.push_back(pause_tok);
      if (g.sample) {
        m.get_logits(lbuf);
        cur = smp.sample(lbuf, &cur_lp);
      }
      m.yield_point();
      if (eos.count(cur)) {
        reason = "done";
        return false;
      }
      if (ngen >= g.max_tokens) {  // pause_tok hit the budget: cur unsent
        reason = "length";
        return false;
      }
      // spec-loop entry invariant: cur is already streamed, not yet ingested.
      if (!send_tok(cur, g.sample && g.logprobs, cur_lp)) return false;
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
        if (ckpt_toks.count(tok)) { pause_tok = tok; pause_lp = 0.0f; return false; }
        if (!send_tok(tok, false, 0.0f)) return false;
        dl_tick();
        return !cancel;
      };
      auto emit_ng_smp = [&](int tok, float lp) -> bool {
        if (ngen >= g.max_tokens) return false;
        if (eos.count(tok)) { reason = "done"; return false; }
        if (ckpt_toks.count(tok)) { pause_tok = tok; pause_lp = lp; return false; }
        if (!send_tok(tok, g.logprobs, lp)) return false;
        dl_tick();
        return !cancel;
      };
      if (!dead && !eos.count(cur)) {
        for (;;) {
          pause_tok = -1;
          if (g.sample)
            m.spec_loop_chain_sample(cur, g.max_tokens - 1, spec_gamma, smp,
                                     chain_mtp, emit_ng_smp);
          else if (chain) m.spec_loop_chain(cur, g.max_tokens - 1, spec_gamma, emit_ng);
          else m.spec_loop_ngram(cur, g.max_tokens - 1, m.ngram.n_max, emit_ng);
          spec_r += m.last_spec_rounds;
          spec_c += m.last_spec_commit;
          spec_p += m.last_spec_proposed;
          if (pause_tok < 0 || dead || cancel || reason == "done") break;
          if (!ckpt_pause(chain)) break;
        }
        m.last_spec_rounds = spec_r;
        m.last_spec_commit = spec_c;
        m.last_spec_proposed = spec_p;
      }
      hist.assign(m.chist.begin(), m.chist.end());
    } else if (use_spec) {
      // Stream each token as the spec loop commits it. eos/cancel/length
      // return false, rolling the round back so trunk and MTP stop at the
      // last token actually sent to the client.
      if (eos.count(cur)) {
        reason = "done";
      } else if (send_tok(cur, g.sample && g.logprobs, cur_lp)) {
        for (;;) {
          pause_tok = -1;
          if (g.sample) {
            auto emit = [&](int tok, float lp) -> bool {
              if (ngen >= g.max_tokens) return false;
              if (eos.count(tok)) {
                reason = "done";
                return false;
              }
              if (ckpt_toks.count(tok)) { pause_tok = tok; pause_lp = lp; return false; }
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
              if (ckpt_toks.count(tok)) { pause_tok = tok; pause_lp = 0.0f; return false; }
              if (!send_tok(tok, false, 0.0f)) return false;
              dl_tick();
              return !cancel;
            };
            m.spec_loop(cur, g.max_tokens - 1, spec_gamma, emit);
          }
          spec_r += m.last_spec_rounds;
          spec_c += m.last_spec_commit;
          spec_p += m.last_spec_proposed;
          if (pause_tok < 0 || dead || cancel || reason == "done") break;
          if (!ckpt_pause(true)) break;
        }
        m.last_spec_rounds = spec_r;
        m.last_spec_commit = spec_c;
        m.last_spec_proposed = spec_p;
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
          // Serial decode advances one token at a time: checkpoint right
          // after a ckpt token (mtp is stale on this path).
          if (!ckpt_toks.empty() && ckpt_toks.count(fed) &&
              rckpt::save(m, hist, false))
            fprintf(stderr, "%s ckpt: saved %zu tokens (after token %d, req %lld)\n",
                    log_ts(), hist.size(), fed, (long long)g.req);
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
    // SSD prompt cache: capture the end-of-turn state, then commit it with
    // the cut captures (new pages + MTP sections + checkpoints go to the
    // background writer). Cancelled turns are valid too -- hist tracks
    // exactly the ingested tokens. The MTP layer is saved only when live.
    if (!kvs.dir.empty()) {
      if (hist.size() >= kvs.min_tokens) kvsnap::capture(m, hist, mtp_live, pend);
      kvsnap::commit(m, hist, pend);
    }
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
    } else if (verb == "KVSYNC") {
      // wait until every in-flight turn committed and the SSD writer is
      // idle (tests/ops); replies "K <ckpts> <pages> <mtp> <bytes>"
      alive = c.send(kvs.dir.empty() ? std::string("K 0 0 0 0\n") : kvsnap::sync_line());
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
  if (m.nslots > 1) memstats_log("slots");  // per-slot state allocated
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
  if (const char* v = getenv("GDEC_KVSNAP_QUEUE_MB"))
    kvs.queue_bytes = (uint64_t)std::max(64.0, atof(v)) << 20;
  if (!kvs.dir.empty()) {
    fprintf(stderr, "kvsnap: dir %s, cap %.1f GB, min %zu tokens\n",
            kvs.dir.c_str(), kvs.cap_bytes / 1073741824.0, kvs.min_tokens);
    kvsnap::init(m, kvs);  // may disable it (dir unusable)
  }
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
