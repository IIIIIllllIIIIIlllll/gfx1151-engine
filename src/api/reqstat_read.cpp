#include "reqstat_read.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace reqstat {
namespace {

// On-disk constants — keep in sync with reqstat.cpp (the Python reader
// tools/reqstat_dump.py carries the same copies).
constexpr uint32_t kMagic = 0x47525153;  // "GRQS"
constexpr uint16_t kVersion = 1;
constexpr uint16_t kHeaderSize = 256;
constexpr uint32_t kRecordSize = 64;

uint16_t get16(const uint8_t* p) {
  uint16_t v;
  memcpy(&v, p, 2);
  return v;
}
uint32_t get32(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}
uint64_t get64(const uint8_t* p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return v;
}

uint16_t crc16(const uint8_t* p, size_t n) {  // CRC-16/CCITT-FALSE
  uint16_t c = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    c ^= (uint16_t)p[i] << 8;
    for (int b = 0; b < 8; b++) c = (c & 0x8000) ? (c << 1) ^ 0x1021 : c << 1;
  }
  return c;
}

bool seek64(FILE* f, uint64_t off) {
#ifdef _WIN32
  return _fseeki64(f, (long long)off, SEEK_SET) == 0;
#else
  return fseeko(f, (off_t)off, SEEK_SET) == 0;
#endif
}

struct ChainFile {
  std::string path;
  uint32_t seq = 0;
  bool live = false;  // reqstat.bin (the file the writer appends to)
};

bool read_header(const std::string& path, uint32_t* seq) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  uint8_t h[kHeaderSize];
  const bool ok = fread(h, 1, sizeof h, f) == sizeof h &&
                  get32(h + 0) == kMagic && get16(h + 4) == kVersion &&
                  get16(h + 6) == kHeaderSize && get32(h + 8) == kRecordSize;
  if (ok) *seq = get32(h + 12);
  fclose(f);
  return ok;
}

uint64_t record_count(uint64_t file_size) {
  return file_size < kHeaderSize ? 0 : (file_size - kHeaderSize) / kRecordSize;
}

// ts_ms of the record at position index, false when it cannot be read.
bool record_ts(FILE* f, uint64_t index, uint64_t* ts) {
  uint8_t r[8];
  if (!seek64(f, (uint64_t)kHeaderSize + index * kRecordSize)) return false;
  if (fread(r, 1, sizeof r, f) != sizeof r) return false;
  *ts = get64(r);
  return true;
}

bool unpack(const uint8_t* r, QEntry* e) {
  if (crc16(r, 56) != get16(r + 56)) return false;
  e->ts_ms = get64(r + 0);
  e->req_seq = get64(r + 8);
  e->n_prompt = get32(r + 16);
  e->n_cached = get32(r + 20);
  e->n_gen = get32(r + 24);
  e->proposed = get32(r + 28);
  e->commit = get32(r + 32);
  e->rounds = get32(r + 36);
  e->prefill_us = get32(r + 40);
  e->decode_us = get32(r + 44);
  e->ttft_us = get32(r + 48);
  e->flags = get32(r + 52);
  return true;
}

// Chain files with valid headers, oldest first. The live file normally has
// the highest seq (rotation bumps it), but sort live-last anyway so a
// hand-renamed file cannot reorder the chain.
std::vector<ChainFile> chain_files(const std::string& dir, ScanInfo* info,
                                   bool* enum_ok) {
  std::vector<ChainFile> files;
  *enum_ok = true;
  std::error_code ec;
  if (!std::filesystem::exists(dir, ec)) return files;  // no stats yet
  for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
    const std::string name = de.path().filename().string();
    const bool live = name == "reqstat.bin";
    bool rotated = false;
    if (!live && name.size() > 12 && name.compare(0, 8, "reqstat-") == 0 &&
        name.compare(name.size() - 4, 4, ".bin") == 0) {
      rotated = true;
      for (size_t i = 8; i + 4 < name.size(); i++)
        if (name[i] < '0' || name[i] > '9') rotated = false;
    }
    if (!live && !rotated) continue;
    ChainFile cf;
    cf.path = de.path().string();
    cf.live = live;
    if (!read_header(cf.path, &cf.seq)) continue;  // foreign or corrupt
    files.push_back(cf);
  }
  if (ec) *enum_ok = false;
  std::sort(files.begin(), files.end(), [](const ChainFile& a,
                                           const ChainFile& b) {
    return a.seq != b.seq ? a.seq < b.seq : !a.live && b.live;
  });
  if (info) info->files_total = (uint32_t)files.size();
  return files;
}

// Stream records [index, index+count) of one file (pass count ~0ULL for
// "to EOF"); entries outside [t0, t1] are filtered, cb false stops early.
void read_records(const std::string& path, uint64_t index, uint64_t count,
                  uint64_t t0, uint64_t t1,
                  const std::function<bool(const QEntry&)>& cb,
                  ScanInfo* info) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return;
  if (!seek64(f, (uint64_t)kHeaderSize + index * kRecordSize)) {
    fclose(f);
    return;
  }
  uint8_t buf[1024 * kRecordSize];
  while (count > 0) {
    const size_t want = (size_t)std::min<uint64_t>(count, 1024);
    const size_t got = fread(buf, kRecordSize, want, f);
    for (size_t i = 0; i < got; i++) {
      QEntry e;
      if (!unpack(buf + i * kRecordSize, &e)) {
        if (info) info->bad_crc++;
        continue;
      }
      if (e.ts_ms < t0 || e.ts_ms > t1) continue;
      if (info) info->records++;
      if (!cb(e)) {
        fclose(f);
        return;
      }
    }
    count -= want;
    if (got < want) break;  // truncated tail
  }
  fclose(f);
}

}  // namespace

std::string stat_dir() {
  if (const char* v = getenv("QWENOX_REQSTAT_DIR")) return v;
  return "data";
}

bool scan(uint64_t t0, uint64_t t1,
          const std::function<bool(const QEntry&)>& visit, ScanInfo* info,
          std::string* err) {
  if (info) *info = ScanInfo{};
  bool enum_ok;
  const auto files = chain_files(stat_dir(), info, &enum_ok);
  if (!enum_ok) {
    if (err) *err = "cannot enumerate " + stat_dir();
    return false;
  }
  for (const auto& cf : files) {
    std::error_code ec;
    const uint64_t count = record_count(std::filesystem::file_size(cf.path, ec));
    if (ec || count == 0) continue;
    FILE* f = fopen(cf.path.c_str(), "rb");
    if (!f) continue;
    uint64_t first = 0, last = 0;
    const bool probe = record_ts(f, 0, &first) && record_ts(f, count - 1, &last);
    fclose(f);
    if (probe && (last < t0 || first > t1)) continue;  // whole file outside
    if (info) info->files_scanned++;
    read_records(cf.path, 0, ~(uint64_t)0, t0, t1, visit, info);
  }
  return true;
}

bool tail(uint32_t n, std::vector<QEntry>* out, ScanInfo* info,
          std::string* err) {
  if (info) *info = ScanInfo{};
  out->clear();
  if (n == 0) return true;
  bool enum_ok;
  auto files = chain_files(stat_dir(), info, &enum_ok);
  if (!enum_ok) {
    if (err) *err = "cannot enumerate " + stat_dir();
    return false;
  }
  // Newest file first: collect whole files until n records are gathered.
  std::vector<std::vector<QEntry>> parts;
  size_t gathered = 0;
  for (auto it = files.rbegin(); it != files.rend() && gathered < n; ++it) {
    std::error_code ec;
    if (record_count(std::filesystem::file_size(it->path, ec)) == 0 || ec)
      continue;
    if (info) info->files_scanned++;
    parts.emplace_back();
    read_records(it->path, 0, ~(uint64_t)0, 0, ~(uint64_t)0,
              [&](const QEntry& e) {
                parts.back().push_back(e);
                return true;
              },
              info);
    gathered += parts.back().size();
  }
  // Assemble oldest-first and keep the last n.
  std::vector<QEntry> all;
  all.reserve(gathered);
  for (auto it = parts.rbegin(); it != parts.rend(); ++it)
    all.insert(all.end(), it->begin(), it->end());
  if (all.size() > n) all.erase(all.begin(), all.end() - n);
  *out = std::move(all);
  return true;
}

bool page(uint64_t offset, uint32_t limit, std::vector<QEntry>* out,
          uint64_t* total_out, ScanInfo* info, std::string* err) {
  if (info) *info = ScanInfo{};
  out->clear();
  bool enum_ok;
  const auto files = chain_files(stat_dir(), info, &enum_ok);
  if (!enum_ok) {
    if (err) *err = "cannot enumerate " + stat_dir();
    return false;
  }
  // Physical slot ranges: chain position [base, base+count) per file.
  struct Slot {
    const ChainFile* cf;
    uint64_t base, count;
  };
  std::vector<Slot> slots;
  uint64_t total = 0;
  for (const auto& cf : files) {
    std::error_code ec;
    const uint64_t count =
        record_count(std::filesystem::file_size(cf.path, ec));
    if (ec || count == 0) continue;
    slots.push_back({&cf, total, count});
    total += count;
  }
  if (total_out) *total_out = total;
  if (limit == 0 || offset >= total) return true;
  const uint64_t end = total - offset;  // exclusive
  const uint64_t start = end > limit ? end - limit : 0;
  for (const auto& s : slots) {
    if (s.base + s.count <= start || s.base >= end) continue;
    const uint64_t i0 = start > s.base ? start - s.base : 0;
    const uint64_t cnt = std::min(s.count - i0, end - (s.base + i0));
    if (info) info->files_scanned++;
    read_records(s.cf->path, i0, cnt, 0, ~(uint64_t)0,
                 [&](const QEntry& e) {
                   out->push_back(e);
                   return true;
                 },
                 info);
  }
  return true;
}

}  // namespace reqstat
