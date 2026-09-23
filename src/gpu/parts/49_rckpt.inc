// parts/49_rckpt.inc — RAM checkpoint tier for KV/state reuse.
//
// llama.cpp-style PARTIAL_ONLY checkpoints: only the non-rollbackable
// recurrent (GDN) state is stored. The QSA KV/indexer caches are addressed
// by absolute position and stay valid while the checkpoint's tokens remain
// a prefix of the live token stream (m.chist) — that prefix check is the
// validity guard. kvsnap (51_host_cfg.inc) is the self-contained SSD tier:
// slower and sparse, but survives restarts and KV clobbering.

namespace rckpt {

struct Ck {
  std::vector<int> toks;  // tokens covered; restore sets hist/chist to this
  int pos = 0, ringpos = 0;
  std::vector<float> s, convst, ring, iraw;  // host copies, ~118 MB total
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

// Drop checkpoints whose KV prefix the live state no longer holds.
void prune(const std::vector<int64_t>& chist) {
  for (auto it = g_list.begin(); it != g_list.end();)
    it = prefix_of(it->toks, chist) ? std::next(it) : g_list.erase(it);
}

void save(GpuModel& m, const std::vector<int>& toks) {
  if (!g_on || toks.size() < g_min_tokens || toks.size() != (size_t)m.pos)
    return;
  const uint64_t h = hash_toks(toks);
  for (auto it = g_list.begin(); it != g_list.end(); ++it)
    if (it->toks.size() == toks.size() && hash_toks(it->toks) == h &&
        it->toks == toks) {  // same state already cached: bump LRU
      Ck mv = std::move(*it);
      g_list.erase(it);
      g_list.push_back(std::move(mv));
      return;
    }
  const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
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
  ck.mrope = m.mrope_active_;
  ck.rope_delta = m.rope_delta_;
  ck.img_hash = m.img_hash_;
  ck.grids = m.mrope_grids_;
  prune(m.chist);
  g_list.push_back(std::move(ck));
  while (g_list.size() > g_cap) g_list.pop_front();
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

bool restore(GpuModel& m, int idx, std::vector<int>* toks_out) {
  Ck& ck = g_list[(size_t)idx];
  const bool ok =
      hipMemcpy(m.d_S, ck.s.data(), ck.s.size() * 4,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(m.d_convst, ck.convst.data(), ck.convst.size() * 4,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(m.d_ring, ck.ring.data(), ck.ring.size() * 4,
                hipMemcpyHostToDevice) == hipSuccess &&
      hipMemcpy(m.d_iraw, ck.iraw.data(), ck.iraw.size() * 4,
                hipMemcpyHostToDevice) == hipSuccess;
  if (!ok) return false;
  m.pos = ck.pos;
  m.ple_ringpos = ck.ringpos;
  // A2: the KV prefix [0, ck.pos) is still mapped (prefix_of(chist) guard);
  // pages past it return to the pool and the continuation maps fresh ones.
  m.kv_trim(ck.pos);
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
