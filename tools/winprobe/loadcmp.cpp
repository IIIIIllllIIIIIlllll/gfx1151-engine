// 探针10：校验 pload 的 Windows 直读路径（NO_BUFFERING + overlapped ReadFile，
// 4096 对齐逻辑）与映射 memcpy 逐字节一致。随机选 N 个区间（含不对齐偏移）
// 两种读法各算 FNV-1a，必须全部相等。
// 用法: loadcmp <file> [n_regions]
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint64_t fnv(const uint8_t* p, size_t n, uint64_t h) {
  for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}

int main(int argc, char** argv) {
  const char* path = argv[1];
  int regions = argc > 2 ? atoi(argv[2]) : 64;
  HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                          OPEN_EXISTING,
                          FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, nullptr);
  if (fh == INVALID_HANDLE_VALUE) { printf("open fail %lu\n", GetLastError()); return 1; }
  LARGE_INTEGER sz;
  GetFileSizeEx(fh, &sz);
  size_t fsize = (size_t)sz.QuadPart;
  HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
  const uint8_t* map = (const uint8_t*)MapViewOfFile(mh, FILE_MAP_READ, 0, 0, fsize);
  if (!map) { printf("map fail\n"); return 1; }

  const size_t WCHUNK = 16ull << 20;
  std::vector<uint8_t> bufalloc(WCHUNK + 4096);
  uint8_t* buf = (uint8_t*)(((uintptr_t)bufalloc.data() + 4095) & ~(uintptr_t)4095);

  uint64_t rng = 0x12345678;
  int bad = 0;
  for (int i = 0; i < regions; i++) {
    // 随机区间：起点任意（含不对齐），长度 1B..32MiB
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t off = (rng >> 33) % (fsize - (32ull << 20) - 1);
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    size_t len = 1 + (size_t)((rng >> 40) % (32ull << 20));
    uint64_t A = off & ~4095ull, E = (off + len + 4095) & ~4095ull;
    uint64_t h_ref = 1469598103934665603ULL, h_new = h_ref;
    // 参考：直接读映射
    h_ref = fnv(map + off, len, h_ref);
    // 新路径：对齐 overlapped 读 + 交集截取（镜像 pload wworker）
    for (uint64_t c = A; c < E; c += WCHUNK) {
      OVERLAPPED ov{};
      ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
      DWORD want = (DWORD)((E - c) < WCHUNK ? (E - c) : WCHUNK);
      ov.Offset = (DWORD)(c & 0xffffffffu);
      ov.OffsetHigh = (DWORD)(c >> 32);
      BOOL r = ReadFile(fh, buf, want, nullptr, &ov);
      if (!r && GetLastError() == ERROR_HANDLE_EOF) { CloseHandle(ov.hEvent); continue; }
      if (!r && GetLastError() != ERROR_IO_PENDING) {
        printf("ReadFile fail %lu\n", GetLastError()); return 1;
      }
      DWORD got = 0;
      if (!GetOverlappedResult(fh, &ov, &got, TRUE)) { printf("GQO fail\n"); return 1; }
      CloseHandle(ov.hEvent);
      uint64_t lo = c < off ? off : c;
      uint64_t hi = (c + got) < (off + len) ? (c + got) : (off + len);
      if (hi > lo) h_new = fnv(buf + (lo - c), (size_t)(hi - lo), h_new);
    }
    if (h_ref != h_new) {
      printf("region %d MISMATCH off=%llu len=%zu\n", i, (unsigned long long)off, len);
      bad++;
    }
  }
  printf("loadcmp: %d regions, %s\n", regions, bad ? "FAIL" : "ALL MATCH");
  return bad != 0;
}
