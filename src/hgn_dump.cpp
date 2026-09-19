// hgn_dump.cpp — HGN1 checkpoint inspector (Phase 1, step 1)
// Reads header + tensor records per HANDOVER.md §2, lists tensors,
// and pread()s small byte ranges for layout analysis.
// Memory-frugal by design: no whole-file mmap, only targeted pread.
//
// Build: g++ -O2 -o hgn_dump hgn_dump.cpp
// Usage:
//   hgn_dump <file.hgn>                    # header + dtype stats
//   hgn_dump <file.hgn> --list [substr]    # list tensors (optionally filtered)
//   hgn_dump <file.hgn> --hex <name> [off] [len]   # hexdump tensor bytes
//   hgn_dump <file.hgn> --raw <name> <out>         # write tensor bytes to file

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#pragma pack(push, 1)
struct HgnHeader {
  char magic[4];       // "HGN1"
  uint32_t version;
  uint32_t tensor_count;
  uint32_t reserved;
  uint64_t records_offset;
  uint64_t data_offset;
  uint64_t file_size;
  char model_name[64];
};

struct TensorRecord {
  char name[96];
  uint32_t dtype;
  uint32_t ndims;
  uint64_t dims[4];
  uint64_t data_offset;  // absolute file offset
  uint64_t data_size;    // bytes
  uint64_t extra;
};
#pragma pack(pop)

static_assert(sizeof(HgnHeader) == 104, "header size");
static_assert(sizeof(TensorRecord) == 160, "record size");

static void die(const char* msg) {
  perror(msg);
  exit(1);
}

static void pread_exact(int fd, void* buf, size_t n, uint64_t off) {
  char* p = (char*)buf;
  while (n > 0) {
    ssize_t r = pread(fd, p, n, (off_t)off);
    if (r <= 0) die("pread");
    p += r;
    n -= (size_t)r;
    off += (uint64_t)r;
  }
}

static std::string rec_name(const TensorRecord& r) {
  size_t len = strnlen(r.name, sizeof(r.name));
  return std::string(r.name, len);
}

static uint64_t num_elements(const TensorRecord& r) {
  uint64_t n = 1;
  for (uint32_t i = 0; i < r.ndims && i < 4; i++) n *= r.dims[i];
  return n;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: hgn_dump <file.hgn> [--list [substr] | --hex <name> [off] [len] | --raw <name> <out>]\n");
    return 2;
  }
  const char* path = argv[1];
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) die("open");

  HgnHeader h;
  pread_exact(fd, &h, sizeof(h), 0);
  if (memcmp(h.magic, "HGN1", 4) != 0) {
    fprintf(stderr, "bad magic: %.4s\n", h.magic);
    return 1;
  }

  std::vector<TensorRecord> recs(h.tensor_count);
  pread_exact(fd, recs.data(), recs.size() * sizeof(TensorRecord), h.records_offset);

  if (argc == 2 || (argc >= 3 && strcmp(argv[2], "--list") == 0)) {
    printf("magic=%.4s version=%u tensors=%u records_off=0x%lx data_off=0x%lx file_size=%lu model=%.64s\n",
           h.magic, h.version, h.tensor_count, (unsigned long)h.records_offset,
           (unsigned long)h.data_offset, (unsigned long)h.file_size, h.model_name);
    const char* filter = (argc >= 4) ? argv[3] : nullptr;
    uint64_t dtype_count[16] = {0};
    uint64_t shown = 0;
    for (const auto& r : recs) {
      if (r.dtype < 16) dtype_count[r.dtype]++;
      std::string name = rec_name(r);
      if (argc >= 3 && strcmp(argv[2], "--list") == 0) {
        if (!filter || name.find(filter) != std::string::npos) {
          printf("%-72s dtype=%-2u ndims=%u dims=(", name.c_str(), r.dtype, r.ndims);
          for (uint32_t i = 0; i < r.ndims && i < 4; i++)
            printf("%s%lu", i ? "," : "", (unsigned long)r.dims[i]);
          uint64_t nelem = num_elements(r);
          double bpe = nelem ? (double)r.data_size * 8.0 / (double)nelem : 0.0;
          printf(") off=0x%lx size=%lu extra=0x%lx bits/elem=%.3f\n",
                 (unsigned long)r.data_offset, (unsigned long)r.data_size,
                 (unsigned long)r.extra, bpe);
          shown++;
        }
      }
    }
    printf("dtype histogram:");
    for (int i = 0; i < 16; i++)
      if (dtype_count[i]) printf("  [%d]=%lu", i, (unsigned long)dtype_count[i]);
    printf("\n");
    if (argc >= 3) printf("listed %lu tensors\n", (unsigned long)shown);
    return 0;
  }

  // tensor-targeting modes
  if (argc < 4) {
    fprintf(stderr, "missing tensor name\n");
    return 2;
  }
  const char* target = argv[3];
  const TensorRecord* tr = nullptr;
  for (const auto& r : recs) {
    if (rec_name(r) == target) { tr = &r; break; }
  }
  if (!tr) {
    fprintf(stderr, "tensor not found: %s\n", target);
    return 1;
  }

  if (strcmp(argv[2], "--deq") == 0) {
    // Q4C-P assumed layout: [64B: 16 x fp32 codebook][packed 4-bit codes,
    // low nibble first][fp16 scales, one per 32 consecutive elements].
    // Dequantizes and checks self-consistency: per-group max|w| should be
    // ~codebook_max * scale if grouping/nibble order is right.
    if (tr->dtype != 5) {
      fprintf(stderr, "tensor dtype is %u, expected 5 (Q4C-P)\n", tr->dtype);
      return 1;
    }
    uint64_t nelem = num_elements(*tr);
    uint64_t ngroups = nelem / 32;
    uint64_t codes_bytes = (nelem + 1) / 2;
    if (tr->data_size != 64 + codes_bytes + ngroups * 2) {
      fprintf(stderr, "size mismatch: nelem=%lu -> expected %lu, actual %lu\n",
              (unsigned long)nelem,
              (unsigned long)(64 + codes_bytes + ngroups * 2),
              (unsigned long)tr->data_size);
      return 1;
    }
    std::vector<uint8_t> buf(tr->data_size);
    pread_exact(fd, buf.data(), buf.size(), tr->data_offset);

    float codebook[16];
    memcpy(codebook, buf.data(), 64);
    printf("codebook:");
    for (int i = 0; i < 16; i++) printf(" %.6f", codebook[i]);
    printf("\n");
    const uint8_t* codes = buf.data() + 64;
    const uint8_t* scale_raw = buf.data() + 64 + codes_bytes;

    // fp16 -> fp32 (IEEE half)
    auto fp16_to_f32 = [](uint16_t h) -> float {
      uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
      float v;
      if (e == 0) v = (float)m * 0x1p-24f;
      else if (e == 31) v = m ? NAN : INFINITY;
      else v = (float)(1024 + m) * (float)pow(2.0, (int)e - 25);
      return s ? -v : v;
    };

    // decode all, track per-group max|w| and scale
    double sum = 0, sum2 = 0;
    uint64_t bad_groups = 0;  // groups where max|w| deviates from scale*maxcode by >10%
    double max_code = 0;
    for (int i = 0; i < 16; i++) max_code = std::max(max_code, (double)fabsf(codebook[i]));
    double ratio_min = 1e30, ratio_max = 0;
    double smin = 1e30, smax = 0;
    uint32_t hist[16] = {0};
    for (uint64_t g = 0; g < ngroups; g++) {
      float scale = fp16_to_f32(scale_raw[2 * g] | (scale_raw[2 * g + 1] << 8));
      smin = std::min(smin, (double)scale);
      smax = std::max(smax, (double)scale);
      double gmax = 0;
      for (uint64_t j = 0; j < 32; j++) {
        uint64_t idx = g * 32 + j;
        uint8_t byte = codes[idx / 2];
        uint8_t nib = (idx % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
        hist[nib]++;
        float w = codebook[nib] * scale;
        sum += w;
        sum2 += (double)w * w;
        gmax = std::max(gmax, (double)fabsf(w));
      }
      double r = gmax / ((double)scale * max_code);
      ratio_min = std::min(ratio_min, r);
      ratio_max = std::max(ratio_max, r);
      if (r < 0.90 || r > 1.0001) bad_groups++;
    }
    double mean = sum / nelem;
    double std = sqrt(sum2 / nelem - mean * mean);
    printf("nelem=%lu ngroups=%lu scale_range=[%.6f, %.6f]\n",
           (unsigned long)nelem, (unsigned long)ngroups, smin, smax);
    printf("weight mean=%.6f std=%.6f\n", mean, std);
    printf("per-group max|w|/(scale*maxcode): min=%.4f max=%.4f bad(>10%% off)=%lu/%lu\n",
           ratio_min, ratio_max, (unsigned long)bad_groups, (unsigned long)ngroups);
    printf("nibble histogram:");
    for (int i = 0; i < 16; i++) printf(" %u", hist[i]);
    printf("\n");
    return 0;
  }

  if (strcmp(argv[2], "--hex") == 0) {
    uint64_t off = (argc >= 5) ? strtoull(argv[4], nullptr, 0) : 0;
    uint64_t len = (argc >= 6) ? strtoull(argv[5], nullptr, 0) : 256;
    if (off + len > tr->data_size) {
      fprintf(stderr, "range [%lu,+%lu) exceeds tensor size %lu; clamping\n",
              (unsigned long)off, (unsigned long)len, (unsigned long)tr->data_size);
      len = tr->data_size - off;
    }
    std::vector<uint8_t> buf(len);
    pread_exact(fd, buf.data(), len, tr->data_offset + off);
    for (uint64_t i = 0; i < len; i++) {
      if (i % 16 == 0) printf("%08lx  ", (unsigned long)(off + i));
      printf("%02x ", buf[i]);
      if (i % 16 == 15) {
        printf(" |");
        for (uint64_t j = i - 15; j <= i; j++)
          putchar(buf[j] >= 32 && buf[j] < 127 ? (char)buf[j] : '.');
        printf("|\n");
      }
    }
    if (len % 16) putchar('\n');
    return 0;
  }

  if (strcmp(argv[2], "--raw") == 0) {
    if (argc < 5) {
      fprintf(stderr, "usage: --raw <name> <out>\n");
      return 2;
    }
    FILE* out = fopen(argv[4], "wb");
    if (!out) die("fopen out");
    const size_t CHUNK = 1 << 20;
    std::vector<uint8_t> buf(CHUNK);
    uint64_t done = 0;
    while (done < tr->data_size) {
      size_t n = (size_t)std::min<uint64_t>(CHUNK, tr->data_size - done);
      pread_exact(fd, buf.data(), n, tr->data_offset + done);
      if (fwrite(buf.data(), 1, n, out) != n) die("fwrite");
      done += n;
    }
    fclose(out);
    printf("wrote %lu bytes to %s\n", (unsigned long)tr->data_size, argv[4]);
    return 0;
  }

  fprintf(stderr, "unknown mode: %s\n", argv[2]);
  return 2;
}
