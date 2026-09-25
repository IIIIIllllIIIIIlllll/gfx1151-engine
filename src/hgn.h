// hgn.h — HGN1 checkpoint loader with mmap + on-the-fly dequant.
// Layouts (verified 2026-09-08, see HANDOVER.md §4):
//   dtype 0  BF16 passthrough
//   dtype 4  u64 array (PLE config)
//   dtype 5  Q4C-P: [64B: 16 x fp32 codebook][rows*cols/2 B: 4-bit codes,
//            row-major flat, low nibble first][scale records: per row
//            cols/32 x fp16, record padded to 16B (e.g. 640-col rows: 40B->48B)];
//            w[r,c] = cb[nib] * scale[r][c/32].  cols = last dim; 3D fused-expert
//            tensors are flat [E*rows, cols] with ONE shared codebook.
//   dtype 7  q8g64 (overlay only): per row of cols: [cols uint8 codes]
//            [cols/64 x (fp16 scale, fp16 min)];  w = code*scale + min
//            (verified 2026-09-08: every group spans exactly [min, min+255*scale])
//   dtype 10 FP8 E4M3 (n-gram table): [numel uint8 codes][4B fp32 global
//            scale at the very end];  w = e4m3(code) * scale
//            (verified 2026-09-08 vs BF16 original rows via range requests)
// Synthetic dtypes (built in memory from GGUF by gguf_map.h, never on disk):
//   dtype 1  F32 passthrough (norms, conv, A_log, dt_bias)
//   dtype 8  q8g32 planar: [rows*cols int8 codes][rows*cols/32 fp16 scales];
//            w[r,c] = code * scale[r*cols/32 + c/32]  (lossless GGUF Q8_0
//            repack; every row's codes start 16B-aligned for cols%16==0)
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <list>
#include <stdexcept>
#include <string>
#ifndef _WIN32
#include <sys/mman.h>
#endif
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include "gpu/os_win32.h"
#endif

namespace hgn {

#pragma pack(push, 1)
struct Header {
  char magic[4];
  uint32_t version, tensor_count, reserved;
  uint64_t records_offset, data_offset, file_size;
  char model_name[64];
};
struct Record {
  char name[96];
  uint32_t dtype, ndims;
  uint64_t dims[4];
  uint64_t data_offset, data_size, extra;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 104 && sizeof(Record) == 160);

struct Tensor {
  std::string name;
  uint32_t dtype = 0, ndims = 0;
  uint64_t dims[4] = {0, 0, 0, 0};
  const uint8_t* data = nullptr;
  uint64_t data_size = 0;
  uint64_t numel() const {
    uint64_t n = 1;
    for (uint32_t i = 0; i < ndims && i < 4; i++) n *= dims[i];
    return n;
  }
};

inline float fp16_to_f32(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v;
  if (e == 0) v = (float)m * 0x1p-24f;
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = ldexpf((float)(1024 + m), (int)e - 25);
  return s ? -v : v;
}

inline float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// FP8 E4M3 (no inf, nan=0x7f)
inline float fp8e4m3_to_f32(uint8_t v) {
  uint32_t s = v >> 7, e = (v >> 3) & 0xf, m = v & 7;
  float r;
  if (e == 0) r = ldexpf((float)m, -9);            // denormal: m * 2^-9
  else if (e == 15 && m == 7) r = NAN;
  else r = ldexpf((float)(8 + m), (int)e - 10);    // (1+m/8) * 2^(e-7)
  return s ? -r : r;
}

class Checkpoint {
public:
  Checkpoint() = default;  // empty; filled via add_synthetic (pure GGUF)
  explicit Checkpoint(const char* path) { map_file(path); }
  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;

  // overlay tensors override base tensors with the same name
  void add_overlay(const char* path) { map_file(path, true); }

  // In-memory tensor (GGUF-derived) owning its bytes; overrides by name like
  // an overlay. t.data / t.data_size are set from buf.
  void add_synthetic(Tensor t, std::vector<uint8_t>&& buf) {
    owned_.push_back(std::move(buf));
    t.data = owned_.back().data();
    t.data_size = owned_.back().size();
    index_[t.name] = t;
    n_synth_++;
  }
  // Borrowed in-memory tensor (e.g. a raw GGUF mmap view); caller keeps it alive.
  void add_view(const Tensor& t) { index_[t.name] = t; n_synth_++; }
  // add_synthetic + add_view calls so far (overrides included)
  size_t synthetic_count() const { return n_synth_; }
  // Drop the host copy of synthetic tensors after they were uploaded (the
  // Tensor records stay: dims/dtype/name remain valid, data becomes null).
  size_t release_owned(bool (*keep)(const Tensor&)) {
    size_t freed = 0;
    for (auto& kv : index_) {
      Tensor& t = kv.second;
      if (!t.data || keep(t)) continue;
      for (auto it = owned_.begin(); it != owned_.end(); ++it)
        if (it->data() == t.data) {
          freed += it->size();
          owned_.erase(it);
          t.data = nullptr;
          break;
        }
    }
    return freed;
  }

  const Tensor* find(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &it->second;
  }
  const Tensor& at(const std::string& name) const {
    const Tensor* t = find(name);
    if (!t) throw std::runtime_error("tensor not found: " + name);
    return *t;
  }

  // Dequant whole tensor to fp32. For big tensors prefer row-wise access.
  void dequant(const Tensor& t, float* out) const {
    uint64_t n = t.numel();
    switch (t.dtype) {
      case 0: {
        const uint16_t* p = (const uint16_t*)t.data;
        for (uint64_t i = 0; i < n; i++) out[i] = bf16_to_f32(p[i]);
        break;
      }
      case 1:
        memcpy(out, t.data, n * 4);
        break;
      case 8: {
        if (n + n / 32 * 2 != t.data_size)
          throw std::runtime_error("q8g32 size mismatch on " + t.name);
        const int8_t* q = (const int8_t*)t.data;
        const uint16_t* s = (const uint16_t*)(t.data + n);
        for (uint64_t i = 0; i < n; i++) out[i] = q[i] * fp16_to_f32(s[i / 32]);
        break;
      }
      case 5: {
        Q4CP q = q4cp_parse(t);
        for (uint64_t r = 0; r < q.rows; r++) q4cp_row(q, r, out + r * q.cols);
        break;
      }
      case 7: {
        // q8g64: per row [cols uint8][cols/64 x (fp16 scale, fp16 min)]
        uint64_t cols = t.dims[t.ndims - 1], rows = t.numel() / cols;
        uint64_t stride = cols + cols / 64 * 4;
        if (rows * stride != t.data_size)
          throw std::runtime_error("q8g64 size mismatch on " + t.name);
        for (uint64_t r = 0; r < rows; r++) {
          const uint8_t* rp = t.data + r * stride;
          const uint16_t* sm = (const uint16_t*)(rp + cols);
          for (uint64_t g = 0; g < cols / 64; g++) {
            float s = fp16_to_f32(sm[g * 2]), m = fp16_to_f32(sm[g * 2 + 1]);
            for (uint64_t j = 0; j < 64; j++)
              out[r * cols + g * 64 + j] = rp[g * 64 + j] * s + m;
          }
        }
        break;
      }
      case 10: {
        // trailing fp32 global scale after the codes
        float scale;
        memcpy(&scale, t.data + n, 4);
        for (uint64_t i = 0; i < n; i++) out[i] = fp8e4m3_to_f32(t.data[i]) * scale;
        break;
      }
      default:
        throw std::runtime_error("dequant: unsupported dtype " + std::to_string(t.dtype));
    }
  }

  // Q4C-P parsed view: [rows, cols] logical matrix (cols = last dim, cols%32==0).
  struct Q4CP {
    const float* cb;       // 16 x fp32
    const uint8_t* codes;  // rows*cols/2 bytes
    const uint8_t* scales; // rows x scale_stride bytes
    uint64_t rows, cols, scale_stride, codes_bytes;
  };
  static Q4CP q4cp_parse(const Tensor& t) {
    Q4CP q;
    q.cols = t.dims[t.ndims - 1];
    q.rows = t.numel() / q.cols;
    q.codes_bytes = q.rows * q.cols / 2;
    q.scale_stride = ((q.cols / 32 * 2) + 15) & ~15ULL;  // 16B-aligned record
    uint64_t expect = 64 + q.codes_bytes + q.rows * q.scale_stride;
    if (t.dtype != 5 || expect != t.data_size)
      throw std::runtime_error("q4cp_parse: size mismatch on " + t.name + " (expect " +
                               std::to_string(expect) + " got " + std::to_string(t.data_size) +
                               ")");
    q.cb = (const float*)t.data;
    q.codes = t.data + 64;
    q.scales = t.data + 64 + q.codes_bytes;
    return q;
  }
  static inline float q4cp_at(const Q4CP& q, uint64_t row, uint64_t col) {
    uint64_t i = row * q.cols + col;
    uint8_t byte = q.codes[i / 2];
    uint8_t nib = (i % 2 == 0) ? (byte & 0xf) : (byte >> 4);
    const uint16_t* sc = (const uint16_t*)(q.scales + row * q.scale_stride);
    return q.cb[nib] * fp16_to_f32(sc[col / 32]);
  }
  // dequant one row into out[cols]
  static void q4cp_row(const Q4CP& q, uint64_t row, float* out) {
    const uint8_t* codes = q.codes + row * (q.cols / 2);
    const uint16_t* sc = (const uint16_t*)(q.scales + row * q.scale_stride);
    for (uint64_t g = 0; g < q.cols / 32; g++) {
      float s = fp16_to_f32(sc[g]);
      for (uint64_t j = 0; j < 32; j++) {
        uint64_t i = g * 32 + j;
        uint8_t byte = codes[i / 2];
        uint8_t nib = (j % 2 == 0) ? (byte & 0xf) : (byte >> 4);
        out[i] = q.cb[nib] * s;
      }
    }
  }

  size_t tensor_count() const { return index_.size(); }

  struct Mapping {
    const uint8_t* base;
    size_t len;
    void* os_handle = nullptr;  // Windows: 保持打开的句柄（OVERLAPPED|NO_BUFFERING，pload 直读用）
  };
  const std::vector<Mapping>& mappings() const { return maps_; }
  const std::unordered_map<std::string, Tensor>& tensors() const { return index_; }

private:
  std::vector<Mapping> maps_;
  std::unordered_map<std::string, Tensor> index_;
  std::list<std::vector<uint8_t>> owned_;  // synthetic tensor storage (stable addresses)
  size_t n_synth_ = 0;

  void map_file(const char* path, bool is_overlay = false) {
#ifdef _WIN32
    // MapViewOfFile 等价 mmap：映射整个文件（64 位 VA，115 GiB 无压力），
    // 页按需从文件调入，不由 pagefile 支撑。os_map_ro 语义与
    // mmap(PROT_READ, MAP_PRIVATE) 一致（对只读场景逐字节相同）。
    // 句柄保持打开并带 OVERLAPPED|NO_BUFFERING：pload 的 ReadFile 直读路径
    // 用它绕过 page cache（映射 memcpy 缺页读在主机 RAM 紧张时退化严重）；
    // NO_BUFFERING 只影响 ReadFile，不影响映射本身。
    HANDLE fh = CreateFileA(path, GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING,
                            FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, nullptr);
    if (fh == INVALID_HANDLE_VALUE)
      throw std::runtime_error(os_last_error((std::string("open ") + path).c_str()));
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) throw std::runtime_error("GetFileSizeEx");
    size_t len = (size_t)sz.QuadPart;
    HANDLE mh =
        CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) throw std::runtime_error("CreateFileMapping");
    const uint8_t* p = (const uint8_t*)MapViewOfFile(mh, FILE_MAP_READ, 0, 0, len);
    CloseHandle(mh);
    if (!p) throw std::runtime_error("MapViewOfFile");
    maps_.push_back({p, len, (void*)fh});
#else
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error(std::string("open ") + path + ": " + strerror(errno));
    struct stat st;
    if (fstat(fd, &st) < 0) throw std::runtime_error("fstat");
    size_t len = (size_t)st.st_size;
    const uint8_t* p =
        (const uint8_t*)mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap");
    // advise random? leave default; page cache shared with the running server
    maps_.push_back({p, len});
#endif

    Header h;
    memcpy(&h, p, sizeof(h));
    if (memcmp(h.magic, "HGN1", 4) != 0) throw std::runtime_error("bad magic");
    if ((uint64_t)h.file_size != len) throw std::runtime_error("size mismatch");

    for (uint32_t i = 0; i < h.tensor_count; i++) {
      Record r;
      memcpy(&r, p + h.records_offset + (uint64_t)i * sizeof(Record), sizeof(r));
      Tensor t;
      t.name.assign(r.name, strnlen(r.name, sizeof(r.name)));
      t.dtype = r.dtype;
      t.ndims = r.ndims;
      for (int k = 0; k < 4; k++) t.dims[k] = r.dims[k];
      t.data = p + r.data_offset;
      t.data_size = r.data_size;
      if (is_overlay)
        index_[t.name] = t;  // override
      else
        index_.emplace(t.name, t);
    }
  }
};

}  // namespace hgn
