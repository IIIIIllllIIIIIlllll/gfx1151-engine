// reqstat_read: query-side reader for the request statistics files written
// by reqstat.cpp (file format: see the header comment in reqstat.h).
// Pure C++ — the API serves statistics without invoking Python.
//
// The truth is the file size: trailing partial bytes are ignored and any
// record whose CRC16 mismatches is skipped (the writer may be appending
// while we read). Files are chained by the seq in their headers: rotated
// reqstat-<seq>.bin first, the live reqstat.bin last.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace reqstat {

struct QEntry {
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

struct ScanInfo {
  uint32_t files_total = 0;    // chain files with a valid header
  uint32_t files_scanned = 0;  // files actually read (not window-skipped)
  uint64_t records = 0;        // records passed to the visitor / returned
  uint64_t bad_crc = 0;        // records skipped on CRC mismatch
};

// Directory the recorder writes to: $GDEC_REQSTAT_DIR or "data".
std::string stat_dir();

// Visit records with t0 <= ts_ms <= t1 in chain order (rotated files by seq,
// then the live file). A file whose first and last record ts both fall
// outside the window is skipped after probing just those two records; within
// a file every record is filtered individually (ts is only approximately
// ordered). visit() returning false stops the scan early.
// Returns false only when the directory cannot be enumerated; a missing
// directory simply yields zero records.
bool scan(uint64_t t0, uint64_t t1,
          const std::function<bool(const QEntry&)>& visit, ScanInfo* info,
          std::string* err);

// Last n records in chain order (out is oldest-first). Reads the newest
// files first and stops as soon as n records are collected, so a long
// history does not make tail queries expensive.
bool tail(uint32_t n, std::vector<QEntry>* out, ScanInfo* info,
          std::string* err);

// One page of the chain, paging back from the newest end: offset=0, limit=n
// equals tail(n). Records are addressed by physical slot position in the
// chain (rotated files by seq, then the live file); the page covers slots
// [total-offset-limit, total-offset) clipped to [0, total), returned
// oldest-first. Only files overlapping the window are opened and only the
// overlapping records are read. A CRC-bad slot inside the window is skipped
// (counted in info->bad_crc), so a page can hold fewer than limit records;
// slots past the last complete record are never counted. Concurrent appends
// grow total, which shifts the window toward older records. total_out
// receives the physical record count of the whole chain.
bool page(uint64_t offset, uint32_t limit, std::vector<QEntry>* out,
          uint64_t* total_out, ScanInfo* info, std::string* err);

}  // namespace reqstat
