#include "reqstat.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>

namespace reqstat {
namespace {

constexpr uint32_t kMagic = 0x47525153;  // "GRQS"
constexpr uint16_t kVersion = 1;
constexpr uint16_t kHeaderSize = 256;
constexpr uint32_t kRecordSize = 64;
constexpr size_t kBufRecords = 64;  // flush every 4 KiB
constexpr double kFlushIntervalS = 2.0;

void put16(uint8_t* p, uint16_t v) { memcpy(p, &v, 2); }
void put32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
void put64(uint8_t* p, uint64_t v) { memcpy(p, &v, 8); }

uint16_t crc16(const uint8_t* p, size_t n) {  // CRC-16/CCITT-FALSE
  uint16_t c = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    c ^= (uint16_t)p[i] << 8;
    for (int b = 0; b < 8; b++) c = (c & 0x8000) ? (c << 1) ^ 0x1021 : c << 1;
  }
  return c;
}

struct Header {
  uint64_t record_count = 0;
  uint64_t first_ts = 0;
  uint64_t last_ts = 0;
  uint64_t created_ts = 0;
  uint64_t rotate_bytes = 0;
  uint32_t file_seq = 0;
};

class Recorder;
Recorder& instance();

class Recorder {
 public:
  void record(const Entry& e) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!ready_) init();
    if (!f_) return;
    const uint64_t esz = (uint64_t)kHeaderSize + hdr_.record_count * kRecordSize;
    if (esz + kRecordSize > hdr_.rotate_bytes && hdr_.record_count > 0) rotate();
    uint8_t* r = buf_ + nbuf_ * kRecordSize;
    pack(r, e);
    if (hdr_.record_count == 0) hdr_.first_ts = e.ts_ms;
    hdr_.last_ts = e.ts_ms;
    hdr_.record_count++;
    if (++nbuf_ == kBufRecords || due()) flush();
  }

 private:
  bool ready_ = false;
  bool atexit_registered_ = false;
  FILE* f_ = nullptr;
  std::mutex mtx_;
  Header hdr_;
  uint8_t buf_[kBufRecords * kRecordSize];
  size_t nbuf_ = 0;
  std::chrono::steady_clock::time_point last_flush_ =
      std::chrono::steady_clock::now();
  std::string path_;

  static void pack(uint8_t* r, const Entry& e) {
    memset(r, 0, kRecordSize);
    put64(r + 0, e.ts_ms);
    put64(r + 8, e.req_seq);
    put32(r + 16, e.n_prompt);
    put32(r + 20, e.n_cached);
    put32(r + 24, e.n_gen);
    put32(r + 28, e.proposed);
    put32(r + 32, e.commit);
    put32(r + 36, e.rounds);
    put32(r + 40, e.prefill_us);
    put32(r + 44, e.decode_us);
    put32(r + 48, e.ttft_us);
    put32(r + 52, e.flags);
    put16(r + 56, crc16(r, 56));
  }

  bool due() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - last_flush_).count() >
           kFlushIntervalS;
  }

  void flush() {
    if (nbuf_ > 0) {
      fseek(f_, 0, SEEK_END);
      fwrite(buf_, kRecordSize, nbuf_, f_);
      nbuf_ = 0;
    }
    write_header();
    fflush(f_);
    last_flush_ = std::chrono::steady_clock::now();
  }

  void write_header() {
    uint8_t h[kHeaderSize];
    memset(h, 0, sizeof h);
    put32(h + 0, kMagic);
    put16(h + 4, kVersion);
    put16(h + 6, kHeaderSize);
    put32(h + 8, kRecordSize);
    put32(h + 12, hdr_.file_seq);
    put64(h + 16, hdr_.record_count);
    put64(h + 24, hdr_.first_ts);
    put64(h + 32, hdr_.last_ts);
    put64(h + 40, hdr_.created_ts);
    put64(h + 48, hdr_.rotate_bytes);
    fseek(f_, 0, SEEK_SET);
    fwrite(h, 1, sizeof h, f_);
  }

  void init() {
    ready_ = true;
    const char* dis = getenv("GDEC_REQSTAT");
    if (dis && !strcmp(dis, "0")) return;
    std::string dir = "data";
    if (const char* v = getenv("GDEC_REQSTAT_DIR")) dir = v;
    double max_mb = 256.0;
    if (const char* v = getenv("GDEC_REQSTAT_MAX_MB"))
      if (double x = atof(v); x >= 0.01) max_mb = x;
    hdr_.rotate_bytes = (uint64_t)(max_mb * 1048576.0);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    path_ = dir + "/reqstat.bin";
    if (std::filesystem::exists(path_)) {
      if (!open_existing()) {
        // Corrupt or foreign file: preserve it and start fresh.
        std::string bad =
            dir + "/reqstat-corrupt-" + std::to_string(now_ms()) + ".bin";
        std::filesystem::rename(path_, bad, ec);
        create_fresh(next_seq(dir));
      }
    } else {
      create_fresh(next_seq(dir));
    }
    if (f_)
      fprintf(stderr, "reqstat: %s seq=%u count=%llu cap=%.0fMB\n", path_.c_str(),
              hdr_.file_seq, (unsigned long long)hdr_.record_count, max_mb);
    if (f_ && !atexit_registered_) {
      atexit_registered_ = true;
      std::atexit([] {
        std::lock_guard<std::mutex> lk(instance().mtx_);
        if (instance().f_) instance().flush();
      });
    }
  }

  static uint64_t now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  // Highest seq among reqstat-<seq>.bin in dir (0 when none).
  uint32_t next_seq(const std::string& dir) {
    uint32_t best = 0;
    std::error_code ec;
    for (auto& de : std::filesystem::directory_iterator(dir, ec)) {
      uint64_t v = 0;
      // 经 .string() 取窄字符：MSVC 头下 path::value_type 是 wchar_t，
      // filename().c_str() 不能直接喂 sscanf。本目录文件名均为 ASCII。
      const std::string name = de.path().filename().string();
      if (sscanf(name.c_str(), "reqstat-%llu.bin", (unsigned long long*)&v) == 1 &&
          v > best)
        best = (uint32_t)v;
    }
    return best + 1;
  }

  void create_fresh(uint32_t seq) {
    hdr_.record_count = hdr_.first_ts = hdr_.last_ts = 0;
    hdr_.created_ts = now_ms();
    hdr_.file_seq = seq;
    f_ = fopen(path_.c_str(), "w+b");
    if (f_) write_header();
  }

  bool open_existing() {
    f_ = fopen(path_.c_str(), "r+b");
    if (!f_) return false;
    uint8_t h[kHeaderSize];
    if (fread(h, 1, sizeof h, f_) != sizeof h) return false;
    uint32_t magic, recsz, fseq;
    uint16_t ver, hsz;
    memcpy(&magic, h + 0, 4);
    memcpy(&ver, h + 4, 2);
    memcpy(&hsz, h + 6, 2);
    memcpy(&recsz, h + 8, 4);
    memcpy(&fseq, h + 12, 4);
    if (magic != kMagic || ver != kVersion || hsz != kHeaderSize ||
        recsz != kRecordSize)
      return false;
    std::error_code ec;
    const uint64_t fsz = std::filesystem::file_size(path_, ec);
    if (ec || fsz < kHeaderSize) return false;
    const uint64_t actual = (fsz - kHeaderSize) / kRecordSize;
    if (kHeaderSize + actual * kRecordSize != fsz)
      std::filesystem::resize_file(path_, kHeaderSize + actual * kRecordSize,
                                   ec);  // drop torn tail
    memcpy(&hdr_.record_count, h + 16, 8);
    memcpy(&hdr_.first_ts, h + 24, 8);
    memcpy(&hdr_.last_ts, h + 32, 8);
    memcpy(&hdr_.created_ts, h + 40, 8);
    memcpy(&hdr_.rotate_bytes, h + 48, 8);
    hdr_.file_seq = fseq;
    // Records are flushed before the header is updated, so every complete
    // record on disk is valid: reconcile the count to the file size in both
    // directions (crash mid-flush can leave the header stale either way).
    const bool stale = hdr_.record_count != actual;
    hdr_.record_count = actual;
    if (hdr_.rotate_bytes < kHeaderSize + kRecordSize)
      hdr_.rotate_bytes = 256ull << 20;
    if (stale) write_header();
    return true;
  }

  void rotate() {
    flush();
    fclose(f_);
    f_ = nullptr;
    char name[64];
    snprintf(name, sizeof name, "reqstat-%06u.bin", hdr_.file_seq);
    std::string dir = std::filesystem::path(path_).parent_path().string();
    std::error_code ec;
    std::filesystem::rename(path_, dir + "/" + name, ec);
    if (ec) {  // rename failed: keep appending to the same file
      f_ = fopen(path_.c_str(), "r+b");
      return;
    }
    fprintf(stderr, "reqstat: rotated -> %s (%llu records)\n", name,
            (unsigned long long)hdr_.record_count);
    create_fresh(hdr_.file_seq + 1);
  }
};

Recorder& instance() {
  static Recorder r;
  return r;
}

}  // namespace

void record(const Entry& e) { instance().record(e); }

}  // namespace reqstat
