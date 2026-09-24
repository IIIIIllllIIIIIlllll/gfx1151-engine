// ---------------- serve mode (gdec wire protocol) ----------------------------
// Batch-1: one connection, one in-flight request. Verbs:
// PING / INFO / MEM / GEN / X / KVSYNC.
// GEN accepts an optional SAMPLE suffix (temp>0, top_k/top_p/min_p/seed) with
// PENALTY/BIAS/LOGPROBS as sampler-only companions; drafter>0 is rejected.
// INFO reports sampling on, prompt cache on, kv_slots=1. MEM reports tracked
// HIP allocations plus driver/process corroborating counters.
// The verb grammars below (parse_gen, INFO fields) are the specification.

// ---------------- KV snapshots: SSD prompt cache (A4) -----------------------
// Content-addressed per page. A KV page (256 tokens, all QSA layers: K, V,
// indexer keys) is a function of the whole token prefix up to its end, so it
// is keyed by a hash chain: c[0] = seed (model fingerprint; M-RoPE prompts
// also fold in the vision tower, grids and patch-bytes hash), c[i+1] =
// H(c[i], tokens of page i). Conversations sharing a prefix share its pages
// on disk. A checkpoint adds what is not per page: the GDN/PLE/indexer-raw
// state at its position and the partial last page. Restoring one continues
// bit-exactly as if the prefix had been prefilled live.
//
// Layout under GDEC_KVSNAP_DIR (default data/kvsnap, relative to CWD):
//   c_<key>.kvc   trunk page: ChHdr (key, parent, 256 tokens, checksum) +
//                 per QSA layer K[256 rows] V[256 rows] ik[64 blocks]
//   m_<key>.kvm   MTP section of a page (B0: the MTP layer is pool layer
//                 nqsa): same body, one layer. MTP row r pairs h_r with token
//                 r+1, so its key also covers the next page's first token.
//   k_<n>_<id>.kck checkpoint: CkHdr, page keys, MTP section keys (0 = none),
//                 tail tokens, GDN d_S + conv, PLE ring, indexer raw, MTP
//                 stream state (d_R, d_miraw; only if saved with the MTP
//                 layer in lockstep), then per layer the tail rows/blocks.
//   All rows in LOGICAL token order (paged device side is copied run by
//   run), so files are layout-agnostic. Every file carries a checksum.
// Saves: the RAM-tier cuts (SNAPS/SNAP/SNAP2) and the end-of-turn state are
// captured (checkpoint image, D2H) on the GPU thread; at the end of the turn
// commit() captures only the pages/MTP sections no cached file holds yet and
// hands everything to one background writer (bounded queue,
// GDEC_KVSNAP_QUEUE_MB, default 1024): tmp_<name> + fdatasync + rename, pages
// before the checkpoints naming them, so a crash never publishes a torn or
// dangling file. KVSYNC waits for the writer (tests).
// Restores: longest checkpoint whose tokens are a strict prefix of the
// prompt; pages are prefetched on a second thread while the previous one is
// copied to the device. A checkpoint saved with the MTP layer live restores
// it too (speculation continues). A damaged file drops the checkpoints that
// need it.
// Eviction: LRU over checkpoints (file mtime, bumped by a restore), capped
// by GDEC_KVSNAP_MAX_GB (default 20); pages are refcounted by checkpoints and
// deleted with the last one. States shorter than GDEC_KVSNAP_MIN (default
// 4096 tokens) are not saved. GDEC_KVSNAP=0 disables. Startup indexes the
// directory, removing temporaries, pre-A4 s_* snapshot directories, and
// orphaned/damaged files. All errors are non-fatal: the cache is an
// accelerator, never a correctness dependency.
namespace kvsnap {
