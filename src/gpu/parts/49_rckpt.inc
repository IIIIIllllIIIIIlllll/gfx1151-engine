// parts/49_rckpt.inc — RAM checkpoint tier for KV/state reuse.
//
// llama.cpp-style PARTIAL_ONLY checkpoints: only the non-rollbackable
// recurrent (GDN) state is stored. The QSA KV/indexer caches are addressed
// by absolute position and stay valid while the checkpoint's tokens remain
// a prefix of the live token stream (m.chist) — that prefix check is the
// validity guard. kvsnap (51_host_cfg.inc) is the self-contained SSD tier:
// slower and sparse, but survives restarts and KV clobbering.
//
// A3 (GDEC_KV_PAGED): a checkpoint also pins the physical KV pages of its
// prefix (refcount), so it stays valid after the live stream diverges —
// switching back to an earlier conversation remaps its pages instead of
// re-prefilling. Live writes into a shared page copy it first (kv_cow).
// When the page pool runs dry the LRU checkpoint is evicted (evict_one).
//
// B0: the MTP KV/keys are pool layer nqsa, so the pinned pages carry them
// too. A checkpoint taken while the MTP layer is in lockstep (mtp_pos ==
// pos-1, rows [0, pos-1) ingested) also stores the MTP stream state (d_R =
// h_{pos-1}, d_miraw, mtp_pos); restoring it keeps speculation available
// (the cont boundary step pairs d_R with the first new token).

namespace rckpt {

struct Ck {
  std::vector<int> toks;  // tokens covered; restore sets hist/chist to this
  int pos = 0, ringpos = 0;
  std::vector<float> s, convst, ring, iraw;  // host copies, ~118 MB total
  std::vector<int> pages;  // A3: pinned KV pages for rows [0, pos) (paged only)
  bool mtp_ok = false;     // B0: MTP state below is valid (lockstep at save)
  int mtp_pos = 0;
  std::vector<float> mR, miraw;  // d_R (h_{pos-1}), d_miraw
  // M-RoPE guard (KVS2 meta semantics): placeholder ids are pixel-agnostic,
  // so a vision checkpoint only matches the same grids and patch hash.
  bool mrope = false;
  int rope_delta = 0;
  uint64_t img_hash = 0;
  std::vector<std::array<int, 3>> grids;
};

static const size_t g_cap = [] {
  const char* v = getenv("GDEC_RCKPT_MAX");
  return v ? (size_t)std::max(0, atoi(v)) : 8;
}();
static const bool g_on = [] {
  const char* v = getenv("GDEC_RCKPT");
  return g_cap != 0 && (!v || atoi(v) != 0);
}();
static const size_t g_min_tokens = [] {
  const char* v = getenv("GDEC_RCKPT_MIN");
  return v ? (size_t)atol(v) : 4096;
}();
static std::deque<Ck> g_list;  // front = oldest
static GpuModel* g_m = nullptr;  // set on the first save
static bool g_pin = false;       // A3: checkpoints own their KV pages

static void release(Ck& ck) {
  if (g_m && !ck.pages.empty()) g_m->kv_unpin(ck.pages);
  ck.pages.clear();
}
static void drop_front() {
  release(g_list.front());
  g_list.pop_front();
}
// Pool pressure (called from GpuModel::kv_reserve / kv_cow): drop the LRU
// checkpoint. Its GDN state goes too — a checkpoint without KV is useless.
static bool evict_one(GpuModel&) {
  if (g_list.empty()) return false;
  fprintf(stderr, "[kvpage] rckpt evicted (%zu tokens, %zu pages) for pool space\n",
          g_list.front().toks.size(), g_list.front().pages.size());
  drop_front();
  return true;
}

static uint64_t hash_toks(const std::vector<int>& t) {
  uint64_t h = 1469598103934665603ULL;
  for (int v : t) {
    h ^= (uint32_t)v;
    h *= 1099511628211ULL;
  }
  return h;
}

static bool prefix_of(const std::vector<int>& t,
                      const std::vector<int64_t>& full) {
  if (t.size() > full.size()) return false;
  for (size_t i = 0; i < t.size(); i++)
    if (full[i] != t[i]) return false;
  return true;
}

bool enabled() { return g_on; }

// Drop checkpoints whose KV prefix the live state no longer holds. A3: pinned
// checkpoints own their KV, so nothing to prune.
void prune(const std::vector<int64_t>& chist) {
  if (g_pin) return;
  for (auto it = g_list.begin(); it != g_list.end();)
    it = prefix_of(it->toks, chist) ? std::next(it) : g_list.erase(it);
}

// `mtp`: the caller ingested the MTP layer in lockstep with this prefix.
void save(GpuModel& m, const std::vector<int>& toks, bool mtp = false) {
  if (!g_on || toks.size() < g_min_tokens || toks.size() != (size_t)m.pos)
    return;
  if (!g_m) {
    g_m = &m;
    g_pin = m.kv_paged;
    if (g_pin) m.kv_evict_fn = &evict_one;
  }
  const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
  const bool mtp_ok = mtp && m.kv_nl > nqsa && m.mtp_pos == m.pos - 1;
  const uint64_t h = hash_toks(toks);
  for (auto it = g_list.begin(); it != g_list.end(); ++it)
    if (it->toks.size() == toks.size() && hash_toks(it->toks) == h &&
        it->toks == toks) {
      if (it->mtp_ok || !mtp_ok) {  // same state already cached: bump LRU
        Ck mv = std::move(*it);
        g_list.erase(it);
        g_list.push_back(std::move(mv));
        return;
      }
      // The cached copy lacks MTP state and its pinned pages hold stale MTP
      // rows: replace it with a fresh checkpoint.
      release(*it);
      g_list.erase(it);
      break;
    }
  Ck ck;
  ck.toks = toks;
  ck.pos = m.pos;
  ck.ringpos = m.ple_ringpos;
  ck.s.resize((size_t)ngdn * 48 * 128 * 128);
  ck.convst.resize((size_t)ngdn * 3 * 10240);
  ck.ring.resize(9 * 10240);
  ck.iraw.resize((size_t)nqsa * 512);
  const bool ok =
      hipMemcpy(ck.s.data(), m.d_S, ck.s.size() * 4,
                hipMemcpyDeviceToHost) == hipSuccess &&
      hipMemcpy(ck.convst.data(), m.d_convst, ck.convst.size() * 4,
                hipMemcpyDeviceToHost) == hipSuccess &&
      hipMemcpy(ck.ring.data(), m.d_ring, ck.ring.size() * 4,
                hipMemcpyDeviceToHost) == hipSuccess &&
      hipMemcpy(ck.iraw.data(), m.d_iraw, ck.iraw.size() * 4,
                hipMemcpyDeviceToHost) == hipSuccess;
  if (!ok) return;
  if (mtp_ok) {
    ck.mR.resize((size_t)g_cfg.branches * g_cfg.d);
    ck.mtp_ok =
        hipMemcpy(ck.mR.data(), m.d_R, ck.mR.size() * 4,
                  hipMemcpyDeviceToHost) == hipSuccess;
    if (ck.mtp_ok && m.d_miraw) {
      ck.miraw.resize(4 * 128);
      ck.mtp_ok = hipMemcpy(ck.miraw.data(), m.d_miraw, ck.miraw.size() * 4,
                            hipMemcpyDeviceToHost) == hipSuccess;
    }
    ck.mtp_pos = m.mtp_pos;
  }
  ck.mrope = m.mrope_active_;
  ck.rope_delta = m.rope_delta_;
  ck.img_hash = m.img_hash_;
  ck.grids = m.mrope_grids_;
  prune(m.chist);
  if (g_pin) ck.pages = m.kv_pin(m.pos);
  g_list.push_back(std::move(ck));
  while (g_list.size() > g_cap) drop_front();
}

// Longest checkpoint whose tokens are a strict prefix of `prompt` (strictly
// shorter: a full-length match has no logits to resume generation from).
// chist-prefix validity is enforced by prune() inside.
int match(const std::vector<int>& prompt, const std::vector<int64_t>& chist,
          bool req_mrope, const std::vector<std::array<int, 3>>* req_grids,
          uint64_t req_img_hash) {
  if (!g_on) return -1;
  prune(chist);
  int best = -1;
  size_t bestn = 0;
  for (size_t i = 0; i < g_list.size(); i++) {
    const Ck& ck = g_list[i];
    const size_t n = ck.toks.size();
    if (n >= prompt.size() || n <= bestn) continue;
    if (ck.mrope != req_mrope) continue;
    if (ck.mrope && (ck.grids != *req_grids || ck.img_hash != req_img_hash))
      continue;
    if (!std::equal(ck.toks.begin(), ck.toks.end(), prompt.begin())) continue;
    best = (int)i;
    bestn = n;
  }
  return best;
}

// *mtp_out: the MTP layer is back in lockstep (speculation may continue).
bool restore(GpuModel& m, int idx, std::vector<int>* toks_out,
             bool* mtp_out) {
  Ck& ck = g_list[(size_t)idx];
  *mtp_out = false;
  bool ok =
      hipMemcpy(m.d_S, ck.s.data(), ck.s.size() * 4,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(m.d_convst, ck.convst.data(), ck.convst.size() * 4,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(m.d_ring, ck.ring.data(), ck.ring.size() * 4,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(m.d_iraw, ck.iraw.data(), ck.iraw.size() * 4,
                hipMemcpyHostToDevice) == hipSuccess;
  if (!ok) return false;
  if (ck.mtp_ok) {
    ok = hipMemcpy(m.d_R, ck.mR.data(), ck.mR.size() * 4,
                   hipMemcpyHostToDevice) == hipSuccess &&
         (ck.miraw.empty() ||
          hipMemcpy(m.d_miraw, ck.miraw.data(), ck.miraw.size() * 4,
                    hipMemcpyHostToDevice) == hipSuccess);
    if (ok) {
      m.mtp_pos = ck.mtp_pos;
      CK(hipMemcpy(m.d_mpos, &m.mtp_pos, 4, hipMemcpyHostToDevice));
      *mtp_out = true;
    }
  }
  m.pos = ck.pos;
  m.ple_ringpos = ck.ringpos;
  if (g_pin) {
    // A3: map the checkpoint's own pages; the live pages not shared with it
    // return to the pool. The shared partial tail page is copied on the
    // first write past ck.pos (kv_prepare).
    m.kv_adopt(ck.pages);
  } else {
    // A2: the KV prefix [0, ck.pos) is still mapped (prefix_of(chist) guard);
    // pages past it return to the pool and the continuation maps fresh ones.
    m.kv_trim(ck.pos);
  }
  CK(hipMemcpy(m.d_pos, &m.pos, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(m.d_ringpos, &m.ple_ringpos, 4, hipMemcpyHostToDevice));
  m.chist.assign(ck.toks.begin(), ck.toks.end());
  *toks_out = ck.toks;
  // bump LRU
  Ck mv = std::move(ck);
  g_list.erase(g_list.begin() + idx);
  g_list.push_back(std::move(mv));
  return true;
}

}  // namespace rckpt
