// reqstat: per-request statistics recorder. Appends fixed 64-byte records to
// a single persistent file (default data/reqstat.bin) with a 256-byte header.
//
// File layout (all integers little-endian):
//   header 256B: magic u32 "GRQS" | version u16 | header_size u16 |
//     record_size u32 | file_seq u32 | record_count u64 | first_ts_ms u64 |
//     last_ts_ms u64 | created_ts_ms u64 | rotate_bytes u64 | reserved 200B
//   record 64B x N:
//     ts_ms u64 | req_seq u64 | n_prompt u32 | n_cached u32 | n_gen u32 |
//     proposed u32 | commit u32 | rounds u32 | prefill_us u32 | decode_us u32 |
//     ttft_us u32 | flags u32 | crc16 u16 | pad u16 | reserved u32
//   flags: bits 0-3 finish (1 stop / 2 length / 3 cancel / 4 error),
//          bits 4-7 drafter (0 serial / 1 mtp / 3 ngram / 4 chain), bit 8 vision.
//
// Consistency rules: records are flushed before the header count is updated,
// so after a crash the header may lag; the truth is the file size
// (count = (size - 256) / 64, trailing partial bytes are truncated on open).
// A torn final record is detected by magic+crc16 and skipped by readers.
// Rotation: when the next record would exceed rotate_bytes the current file
// is renamed reqstat-<seq>.bin and a fresh reqstat.bin is started.
//
// Env: GDEC_REQSTAT=0 disables; GDEC_REQSTAT_DIR overrides the directory;
// GDEC_REQSTAT_MAX_MB sets the rotation threshold (default 256, fractional
// values allowed for testing).
#pragma once
#include <cstdint>

namespace reqstat {

struct Entry {
  uint64_t ts_ms = 0;
  uint64_t req_seq = 0;
  uint32_t n_prompt = 0;
  uint32_t n_cached = 0;
  uint32_t n_gen = 0;
  uint32_t proposed = 0;
  uint32_t commit = 0;
  uint32_t rounds = 0;
  uint32_t prefill_us = 0;
  uint32_t decode_us = 0;
  uint32_t ttft_us = 0;
  uint32_t flags = 0;
};

// Record one completed request. Thread-safe; never throws, never blocks the
// caller for more than a buffer flush (one write syscall per 64 records).
void record(const Entry& e);

}  // namespace reqstat
