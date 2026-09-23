// ---------------- serve mode (gdec wire protocol) ----------------------------
// Batch-1: one connection, one in-flight request. Verbs:
// PING / INFO / MEM / GEN / X.
// GEN accepts an optional SAMPLE suffix (temp>0, top_k/top_p/min_p/seed) with
// PENALTY/BIAS/LOGPROBS as sampler-only companions; drafter>0 is rejected.
// INFO reports sampling on, prompt cache on, kv_slots=1. MEM reports tracked
// HIP allocations plus driver/process corroborating counters.
// The verb grammars below (parse_gen, INFO fields) are the specification.

// ---------------- KV snapshots: SSD prompt cache ---------------------------
// A snapshot is the full recurrent state at end-of-context: GDN d_S + conv
// state, PLE ring + chist, QSA KV caches (first ntok rows), indexer keys/raw,
// pos. Restoring one continues bit-exactly as if the prefix had been
// prefilled live (raw state bytes; A/B verified against live prefill).
//
// Layout under GDEC_KVSNAP_DIR (default data/kvsnap, relative to CWD):
//   tmp_<pid>_/  --write files-->  rename -->  s_<ntok>_<hash16>/  (atomic)
//     meta       "KVS3 ntok nqsa ngdn maxctx kvb index_blocks ringpos mrope"
//                (mrope=1: second line "rope_delta img_hash k t0 h0 w0 ..."
//                for M-RoPE states; placeholder ids are pixel-agnostic, so a
//                vision snapshot only restores for the same grids+pixels).
//                KVS1/KVS2 (broken per-layer ik layout) are rejected and
//                removed by the LRU pass.
//     tokens     int32[ntok]
//     s.bin      ngdn*48*128*128 fp32 | convst.bin ngdn*3*10240 fp32
//     ring.bin   9*10240 fp32         | iraw.bin   nqsa*512 fp32
//     ik.bin     per QSA layer, min(ntok/4+2, index_blocks) * 128 fp32
//     kv.bin     K then V, nqsa * ntok * 512 * kvb each (kvb = 2 bf16 / 4 fp32)
//   ik/kv are stored in LOGICAL token order; with GDEC_KV_PAGED the device
//   side is read/written page-run by page-run, so files are layout-agnostic.
// A crashed save never publishes (rename is the commit point). Snapshots are
// self-contained -- no prefix-chain dependencies -- so eviction is
// whole-snapshot LRU by directory mtime (a restore bumps it), capped by
// GDEC_KVSNAP_MAX_GB (default 20). States shorter than GDEC_KVSNAP_MIN
// (default 4096 tokens) are not worth the IO. GDEC_KVSNAP=0 disables.
// All errors are non-fatal: the cache is an accelerator, never a
// correctness dependency.
namespace kvsnap {
