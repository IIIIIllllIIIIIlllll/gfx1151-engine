// gdec.cpp — Phase 2: gfx1151 GPU decode path (single-token forward).
// Phase 3a: prefill_batch — batched (GEMM) prefill; GDEC_NOPREFILLBATCH=1
// falls back to the old per-token forward() loop.
// Phase 3c: k_qsa_flash — batched causal flash attention for prefill;
// GDEC_QSA_LOOP=1 falls back to the per-token k_qsa_step loop.
// Phase 3b: k_gdn_chunk — chunked (64) GDN prefill, fp32;
// GDEC_GDN_LOOP=1 falls back to the per-token recurrence loop.
// Phase 3b-split: k_gdn_intra (per chunk-head, state-independent) +
// k_gdn_inter (per-head serial d-tile pass) replace k_gdn_chunk by default;
// GDEC_GDN_NOSPLIT=1 restores the single-kernel path.
// Phase 3b-strip: k_gdn_inter_strip splits the inter pass into 4 independent
// 32-column S strips per head (grid (4,48), ~25 KiB LDS/block, 2 blocks/CU).
// 32-wide strips quarter the redundant kcd/q/attn2/k DRAM re-reads of the
// original 16-wide version (~1.7x at 32K, bit-identical);
// GDEC_GDN_NOSTRIP=1 restores k_gdn_inter.
// Phase 3d-1: PLE batched prefill (ple_gpu_b); GDEC_PLE_LOOP=1 falls back
// to the per-token ple_gpu_t loop. MoE fused W4 GEMM (k_moe_w4_up/down read
// Q4C-P codes directly, bf16 v_dot2 with fp32 accumulate); GDEC_MOE_NAIVE=1
// falls back to grouped GEMV.
// Phase 3f: stable GPU MoE routing and 64-token expert tasks.
// GDEC_MOE_HOST_ROUTE=1 / GDEC_MOE_UNTILED=1 restore the respective old paths.
// Sparse prefill stages V in LDS; GDEC_QSA_GLOBAL_V=1 restores global V reads.
// Prefill GR emits cached BF16 inputs; GDEC_PREFILL_UNFUSED=1 restores separate kernels.
// MoE prefill passes BF16 activations directly; GDEC_MOE_FP32_IO=1 restores FP32 staging.
// QSA KV cache stores bf16 (f2bf RNE on write, fp32 reads/compute);
// GDEC_QSA_KV_BF16=1 enables the BF16 KV cache (experiment: measured parity).
// BF16 KV mode also runs sparse prefill through k_qsa_flash_bf16 (bf16 LDS
// tiles + fdot2 scores); the FP32 sparse path is bit-identical to before.
// GDEC_MOE_LT=1: prefill MoE via dequant-to-bf16 + per-expert hipBLASLt GEMM
// (batched-GEMM pipeline; different rounding than the fused W4 kernels, so
// it is opt-in). GDEC_MOE_LT_BF16=1 makes the expert
// GEMM outputs bf16 as well. Both need GPU routing + deterministic mode.
// GDEC_MOE_LT_OVL=1 (needs GDEC_MOE_LT) double-buffers the dequant scratch
// and runs the next 4-expert group's dequant on a side stream overlapping
// the current group's GEMMs; event-chained, bit-identical.
// Phase 3g: self-written bf16 WMMA dense GEMM (k_gemm_wmma) for the four
// prefill projection shapes at P >= 1024, opt-in via GDEC_GEMM_WMMA=1
// (~37-38 vs ~32 TFLOPS hipBLASLt on N=6144..12288 K=2560; N=2560 K=6144
// ~parity-to-+11%). Different K accumulation order than Tensile (not
// bit-identical).
// Phase 3h: tail-merge — a final prefill remainder of <= GDEC_PREFILL_TAIL_SLACK
// (default 1024 on Linux, 0 elsewhere) tokens is absorbed into the previous
// chunk instead of running as its own fixed-cost chunk; batch workspace is
// allocated at maxbatch_cap = maxbatch + slack. GDEC_GDN_WINDOW_CHUNKS now
// also sizes d_gdn_split_ws (was hardcoded 4; window>4 used to be an
// illegal-memory-access bug).
// M-RoPE (vision, stage 1b/1c): requests may carry image grid_thw triples
// (CLI --mrope-grid t,h,w / serve GEN suffix MROPE k t h w ...); the host
// expands them to per-token 3-row positions (reference get_rope_index
// semantics, tools/mrope_ref.py) and fills the k_rope_cs cos/sin table with
// the slot-interleaved frequencies; decode adds the request's rope_delta to
// the rope angle position (*d_pos + *d_rdelta). Text-only requests never
// touch the mrope path (delta 0, device-table fill) and are bitwise
// identical to before. GDEC_MROPE_DUMP=<file> dumps the filled table and
// per-decode-step rope cos/sin (debug; syncs the stream).
// Vision (stage 4): --vision-tower loads the ViT sidecar; GEN requests may
// then carry VIMG k binary patch frames (one per MROPE grid) whose embeds
// are masked-scattered over the image placeholder rows after the embedding
// gather in prefill_chunk (k_img_inject_*, bf16 source). Without the flag
// the engine is text-only and image requests whose placeholders are not
// covered by grid data are rejected, naming the flag).
// All weights (except the 47.7 GiB PLE table) are memcpy'd into one 68 GiB
// hipMalloc arena at startup (4 threads + pinned staging). PLE table rows
// (16 x 160 B per token) are read host-side from the mmap and uploaded
// through a small pinned staging buffer.
//
// Semantics follow reference/modeling_qwen4_exp.py exactly (same as ref.cpp).
// QSA selects 512 complete 4-token blocks plus all incomplete tail tokens.
// GDEC_QSA_DENSE=1 restores the old dense path for performance comparisons.
//
// Build: hipcc -O3 -o gdec gdec.cpp -lrocblas -lhipblaslt
// Usage: gdec <base.hgn> [overlay.hgn ...] --tokens 1,2,3 [--gen N] [--dump]
//        [--maxctx N]   (multiple overlays apply in order; later wins)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <rocprim/block/block_radix_sort.hpp>
#include <rocprim/block/block_scan.hpp>
#include <rocprim/device/device_radix_sort.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include "os_win32.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "../hgn.h"

#ifndef _WIN32
// POSIX side of the socket shim (Windows side lives in os_win32.h): lets the
// serve loop below use one spelling on both platforms.
typedef int sockfd_t;
static inline bool wsa_init() { return true; }
static inline sockfd_t sock_tcp() { return socket(AF_INET, SOCK_STREAM, 0); }
static inline int sock_close(sockfd_t fd) { return close(fd); }
static inline ssize_t sock_recv(sockfd_t fd, void* buf, size_t n, bool dontwait) {
  return recv(fd, buf, n, dontwait ? MSG_DONTWAIT : 0);
}
static inline ssize_t sock_send(sockfd_t fd, const void* buf, size_t n) {
  return send(fd, buf, n, MSG_NOSIGNAL);
}
static inline int sock_errno() { return errno; }
static inline const char* sock_strerror(int e) { return strerror(e); }

// POSIX side of the os_* file shims (Windows side lives in os_win32.h).
static inline int os_open_rd(const char* path) { return open(path, O_RDONLY); }
static inline int os_open_wr(const char* path) {
  return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
}
static inline int os_close(int fd) { return close(fd); }
static inline ssize_t os_write(int fd, const void* p, size_t n) {
  return write(fd, p, n);
}
static inline int64_t os_fsize(int fd) {
  struct stat sb;
  return fstat(fd, &sb) ? -1 : (int64_t)sb.st_size;
}
static inline void* os_map_ro(const char* path, size_t len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return nullptr;
  void* p = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  return p == MAP_FAILED ? nullptr : p;
}
static inline void os_unmap_ro(void* p, size_t len) { munmap(p, len); }
#endif

static inline void sock_rcvtimeo(sockfd_t fd, int sec) {
#ifdef _WIN32
  DWORD ms = (DWORD)(sec * 1000);
  setsockopt((SOCKET)fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
#else
  timeval tv{sec, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
}

using hgn::Checkpoint;
using hgn::Tensor;

#define CK(x)                                                          \
  do {                                                                 \
    hipError_t e_ = (x);                                               \
    if (e_ != hipSuccess) {                                            \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
              __FILE__, __LINE__);                                     \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

// "[2026-09-18 11:25:03.123]" — wall-clock prefix for human-facing logs.
// thread_local so concurrent loader/decode threads never share the buffer.
// One call per fprintf: two calls in one expression would alias.
static const char* log_ts() {
  static thread_local char b[32];
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  struct tm tmv;
  localtime_r(&ts.tv_sec, &tmv);
  snprintf(b, sizeof b, "[%04d-%02d-%02d %02d:%02d:%02d.%03d]",
           tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour,
           tmv.tm_min, tmv.tm_sec, (int)(ts.tv_nsec / 1000000));
  return b;
}

struct NgramMod {
  using token_t=int32_t; static constexpr token_t EMPTY=-1;
  int n=24,n_min=48,n_max=64; size_t learned=0; std::vector<token_t> table;
  NgramMod(){ if(const char* e=getenv("GDEC_NGRAM_MATCH")) n=std::max(4,std::min(64,atoi(e))); if(const char* e=getenv("GDEC_NGRAM_MIN")) n_min=std::max(1,std::min(64,atoi(e))); if(const char* e=getenv("GDEC_NGRAM_MAX")) n_max=std::max(1,std::min(64,atoi(e))); table.resize((size_t)4*1024*1024,EMPTY); }
  size_t slot(const std::vector<int64_t>&h,size_t off)const{uint64_t z=0;for(int i=0;i<n;i++) z=z*6364136223846793005ULL+(uint32_t)h[off+i];return(size_t)(z%table.size());}
  void reset(){std::fill(table.begin(),table.end(),EMPTY);learned=0;}
  void observe(const std::vector<int64_t>&h){if(h.size()<=(size_t)n)return;size_t lim=h.size()-n;if(learned>lim)learned=lim;for(size_t i=learned;i<lim;i++)table[slot(h,i)]=(token_t)h[i+n];learned=lim;}
  std::vector<int> draft(const std::vector<int64_t>&h,int pending,int want)const{std::vector<int>o;if(h.size()+1<(size_t)n)return o;std::vector<int64_t>c((size_t)n);for(int i=0;i<n-1;i++)c[i]=h[h.size()-(size_t)(n-1)+i];c[n-1]=pending;for(int i=0;i<std::min(want,n_max);i++){token_t t=table[slot(c,0)];if(t==EMPTY){if(i<n_min)o.clear();break;}o.push_back((int)t);for(int j=0;j<n-1;j++)c[j]=c[j+1];c[n-1]=(int)t;}return o;}
};

// ---------------- config (matches ref.cpp / config.json) --------------------
struct Cfg {
  int d = 2560;
  int layers = 48;
  int branches = 4;
  int gdn_hk = 16, gdn_dk = 128;
  int gdn_hv = 48, gdn_dv = 128;
  int gdn_conv = 4;
  int qsa_hq = 24, qsa_dh = 256;
  int qsa_hkv = 2;
  int rotary_dim = 64;
  int gr_rank = 320;
  int experts = 512, topk = 10, moe_mid = 640;
  float norm_eps = 1e-6f;
  double rope_theta = 1e7;
  int vocab = 248320;
};
static Cfg g_cfg;
static hipStream_t g_str = nullptr;  // capture stream during graph build, else default

static bool is_qsa(int layer) { return layer % 4 == 3; }

// ---------------- PLE 批量聚集（io_uring / Windows IOCP） ---------------------
// Linux: Raw-syscall io_uring batch pread of the 47.7 GiB PLE table, replacing
// the madvise(WILLNEED) + page-fault + memcpy two-phase gather in
// prefill_chunk. Reads land directly in the pinned staging buffer (no page
// faults, no memcpy); staging bytes are identical to the madvise path. Any
// setup failure (fd open/fadvise, table not in the base mapping, ring setup)
// falls back to the old path at runtime.
// Windows: IOCP（I/O completion port）批量 overlapped ReadFile，语义与
// io_uring 路径逐字节一致：同样的 (addrs, staging, table_va) 接口、同样的
// "失败即 fatal"约定。缓冲句柄（不加 NO_BUFFERING——160B 行不对齐扇区），
// 热行走缓存、冷行高队列深度下 NVMe 随机读（QD=512）。
static uint64_t g_ple_foff = 0;  // file offset of the PLE table in that file
static bool g_ple_off_ok = false;

#ifndef _WIN32
static int g_ple_fd = -1;        // extra O_RDONLY handle on the base .hgn

static void ple_uring_open(const char* path) {
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return;
  if (posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM) != 0) {  // kill readahead
    close(fd);
    return;
  }
  g_ple_fd = fd;
}

struct PleUring {
  int fd = -1;
  int err = 0;  // errno from the failed setup step, for the fallback message
  unsigned entries = 0;
  unsigned *sq_head = nullptr, *sq_tail = nullptr, *sq_mask = nullptr,
           *sq_array = nullptr;
  io_uring_sqe* sqes = nullptr;
  unsigned *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
  io_uring_cqe* cqes = nullptr;
  void* sq_ring = nullptr;
  size_t sq_ring_sz = 0;
  void* cq_ring = nullptr;
  size_t cq_ring_sz = 0;
  void* sqes_map = nullptr;
  size_t sqes_sz = 0;

  bool setup(unsigned n) {
    io_uring_params p{};
    fd = (int)syscall(__NR_io_uring_setup, n, &p);
    if (fd < 0) {
      err = errno;
      return false;
    }
    sq_ring_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    cq_ring_sz = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    sqes_sz = p.sq_entries * sizeof(io_uring_sqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
      if (cq_ring_sz < sq_ring_sz) cq_ring_sz = sq_ring_sz;
      sq_ring_sz = cq_ring_sz;
    }
    sq_ring = mmap(nullptr, sq_ring_sz, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (sq_ring == MAP_FAILED) {
      err = errno;
      goto fail;
    }
    if (p.features & IORING_FEAT_SINGLE_MMAP)
      cq_ring = sq_ring;
    else {
      cq_ring = mmap(nullptr, cq_ring_sz, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
      if (cq_ring == MAP_FAILED) {
        err = errno;
        goto fail;
      }
    }
    sqes_map = mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (sqes_map == MAP_FAILED) {
      err = errno;
      goto fail;
    }
    sq_head = (unsigned*)((char*)sq_ring + p.sq_off.head);
    sq_tail = (unsigned*)((char*)sq_ring + p.sq_off.tail);
    sq_mask = (unsigned*)((char*)sq_ring + p.sq_off.ring_mask);
    sq_array = (unsigned*)((char*)sq_ring + p.sq_off.array);
    cq_head = (unsigned*)((char*)cq_ring + p.cq_off.head);
    cq_tail = (unsigned*)((char*)cq_ring + p.cq_off.tail);
    cq_mask = (unsigned*)((char*)cq_ring + p.cq_off.ring_mask);
    cqes = (io_uring_cqe*)((char*)cq_ring + p.cq_off.cqes);
    sqes = (io_uring_sqe*)sqes_map;
    entries = p.sq_entries;
    return true;
  fail:
    if (sq_ring && sq_ring != MAP_FAILED) munmap(sq_ring, sq_ring_sz);
    if (cq_ring && cq_ring != MAP_FAILED && cq_ring != sq_ring)
      munmap(cq_ring, cq_ring_sz);
    close(fd);
    fd = -1;
    sq_ring = cq_ring = nullptr;
    return false;
  }

  // Pump n preads of 160 B each: row i -> staging + i*160, file offset
  // g_ple_foff + (addrs[i] - table_va). Keeps the ring full; CQE order is
  // irrelevant (distinct destinations). Any failed or short read is fatal —
  // no silent mid-batch fallback onto half-written staging.
  void run(const uint64_t* addrs, uint8_t* staging, uint64_t table_va,
           size_t n) {
    size_t sub = 0, done = 0;
    unsigned ltail = __atomic_load_n(sq_tail, __ATOMIC_RELAXED);
    while (done < n) {
      unsigned shead = __atomic_load_n(sq_head, __ATOMIC_ACQUIRE);
      unsigned fill = 0, space = entries - (ltail - shead);
      while (sub < n && fill < space) {
        unsigned idx = ltail & *sq_mask;
        io_uring_sqe* s = &sqes[idx];
        memset(s, 0, sizeof(*s));
        s->opcode = IORING_OP_READ;
        s->fd = g_ple_fd;
        s->addr = (uint64_t)(uintptr_t)(staging + sub * 160);
        s->len = 160;
        s->off = g_ple_foff + (addrs[sub] - table_va);
        s->user_data = sub;
        sq_array[idx] = idx;
        ltail++;
        sub++;
        fill++;
      }
      if (fill) __atomic_store_n(sq_tail, ltail, __ATOMIC_RELEASE);
      if ((int)syscall(__NR_io_uring_enter, fd, fill, 1,
                       IORING_ENTER_GETEVENTS, nullptr, 0) < 0) {
        fprintf(stderr, "ple_uring: io_uring_enter failed: %s\n",
                strerror(errno));
        abort();
      }
      unsigned ch = __atomic_load_n(cq_head, __ATOMIC_ACQUIRE);
      unsigned ct = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
      for (; ch != ct; ch++) {
        const io_uring_cqe* c = &cqes[ch & *cq_mask];
        if (c->res != 160) {
          fprintf(stderr, "ple_uring: read for row %llu failed: res=%d (%s)\n",
                  (unsigned long long)c->user_data, c->res,
                  c->res < 0 ? strerror(-c->res) : "short read");
          abort();
        }
        done++;
      }
      __atomic_store_n(cq_head, ch, __ATOMIC_RELEASE);
    }
  }
};

static PleUring g_ple_ring;
static bool g_ple_ring_tried = false;

static bool ple_uring_ready() {
  if (!getenv("GDEC_PLE_URING")) return false;
  if (g_ple_fd < 0 || !g_ple_off_ok) return false;
  if (!g_ple_ring_tried) {
    g_ple_ring_tried = true;
    if (!g_ple_ring.setup(256))  // fixed depth: one structural decision
      fprintf(stderr,
              "ple_uring: io_uring_setup failed (%s); using madvise path\n",
              strerror(g_ple_ring.err ? g_ple_ring.err : EINVAL));
  }
  return g_ple_ring.fd >= 0;
}
#define PLE_PUMP g_ple_ring

#else  // _WIN32 ------------------------------------------------------------

// IOCP 批量读：与 PleUring::run 相同语义——n 个 160B 行从文件偏移
// g_ple_foff + (addrs[i] - table_va) 读入 staging + i*160；任何失败/短读
// 立即 fatal，不做半途回退。
struct PleWin {
  HANDLE fh = INVALID_HANDLE_VALUE;
  HANDLE iocp = nullptr;
  static constexpr size_t DEPTH = 512;

  bool ok() const { return iocp != nullptr && g_ple_off_ok; }

  bool setup(const char* path) {
    fh = CreateFileA(path, GENERIC_READ,
                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                     nullptr, OPEN_EXISTING,
                     FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return false;
    iocp = CreateIoCompletionPort(fh, nullptr, 0, 0);
    if (!iocp) {
      CloseHandle(fh);
      fh = INVALID_HANDLE_VALUE;
      return false;
    }
    return true;
  }

  void run(const uint64_t* addrs, uint8_t* staging, uint64_t table_va,
           size_t n) {
    // decode 每 token 只读 16 行：槽位数按需收缩，避免每 token 分配 512 槽
    const size_t depth = std::min(DEPTH, n);
    std::vector<OVERLAPPED> ovs(depth);
    std::vector<size_t> rows(depth);  // slot -> row（报错定位）
    // 自由槽位队列：槽位只有在自己的 completion 到达后才能复用
    // （完成顺序任意，不能用 sub%DEPTH 环形推断）。
    std::vector<size_t> free_(depth);
    for (size_t i = 0; i < depth; i++) free_[i] = depth - 1 - i;
    size_t nfree = depth;
    size_t sub = 0, done = 0;
    while (done < n) {
      while (sub < n && nfree > 0) {
        size_t slot = free_[--nfree];
        rows[slot] = sub;
        OVERLAPPED* ov = &ovs[slot];
        memset(ov, 0, sizeof(*ov));
        uint64_t off = g_ple_foff + (addrs[sub] - table_va);
        ov->Offset = (DWORD)(off & 0xffffffffu);
        ov->OffsetHigh = (DWORD)(off >> 32);
        BOOL r = ReadFile(fh, staging + sub * 160, 160, nullptr, ov);
        if (!r && GetLastError() != ERROR_IO_PENDING) {
          fprintf(stderr, "ple_win: ReadFile row %zu failed: %lu\n", sub,
                  GetLastError());
          abort();
        }
        sub++;
      }
      DWORD got = 0;
      ULONG_PTR key = 0;
      OVERLAPPED* pov = nullptr;
      BOOL okr = GetQueuedCompletionStatus(iocp, &got, &key, &pov, INFINITE);
      size_t slot = (size_t)(pov - ovs.data());
      if (!okr || got != 160) {
        fprintf(stderr,
                "ple_win: read completion failed (row=%zu, got=%lu, err=%lu)\n",
                pov ? rows[slot] : (size_t)-1, (unsigned long)got,
                (unsigned long)GetLastError());
        abort();
      }
      free_[nfree++] = slot;
      done++;
    }
  }
};

static PleWin g_ple_win;

static void ple_uring_open(const char* path) {
  // 名字沿用 Linux 版本；Windows 下建立 IOCP 句柄
  g_ple_win.setup(path);
}

static bool ple_uring_ready() { return g_ple_win.ok(); }
#define PLE_PUMP g_ple_win

#endif  // _WIN32

constexpr int INDEX_BUDGET = 512;
constexpr int INDEX_BATCH = 1024;

__device__ __forceinline__ int qsa_source_pos(int slot, int visible,
                                              const int* blocks) {
  if (!blocks || visible < 2052) return slot;
  return slot < 2048 ? 4 * blocks[slot / 4] + slot % 4
                     : (visible / 4) * 4 + slot - 2048;
}

// KV cache element access: FP32 caches load directly; BF16 caches expand to
// fp32 in registers (LDS staging and arithmetic stay fp32 either way).
__device__ __forceinline__ float kv_ld1(const float* __restrict__ p) { return *p; }
__device__ __forceinline__ float kv_ld1(const uint16_t* __restrict__ p) {
  uint32_t u = (uint32_t)*p << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
__device__ __forceinline__ float4 kv_ld4(const float* __restrict__ p) {
  return *(const float4*)p;
}
__device__ __forceinline__ float4 kv_ld4(const uint16_t* __restrict__ p) {
  uint2 pk = *(const uint2*)p;  // 4 bf16, little-endian pairs
  uint32_t x0 = pk.x << 16, x1 = pk.x & 0xffff0000u;
  uint32_t x2 = pk.y << 16, x3 = pk.y & 0xffff0000u;
  float4 v;
  memcpy(&v.x, &x0, 4);
  memcpy(&v.y, &x1, 4);
  memcpy(&v.z, &x2, 4);
  memcpy(&v.w, &x3, 4);
  return v;
}

// Indexer Q and pooled K use the same 64 rotary dimensions as main attention.
// cs_tab (the k_rope_cs per-token cos/sin table, indexed by position-tab_base)
// replaces the inline fp64 cos/sin when the position is covered; values are
// bit-identical (same libm, same argument expression via g_rope_tab). The
// inline path remains for decode/MTP and for a pool block straddling the
// table's left edge (position < tab_base).
__device__ void index_norm_rope(float value, const float* weight, float* out,
                                int position, float eps, double theta,
                                const float2* cs_tab = nullptr, int tab_base = 0) {
  __shared__ float values[128], sums[128];
  int d = threadIdx.x;
  sums[d] = value * value;
  __syncthreads();
  for (int s = 64; s; s >>= 1) {
    if (d < s) sums[d] += sums[d + s];
    __syncthreads();
  }
  values[d] = value * rsqrtf(sums[0] / 128.f + eps) * (1.f + weight[d]);
  __syncthreads();
  float v = values[d];
  if (d < 64) {
    int i = d % 32;
    int ti = position - tab_base;
    float cs, sn;
    if (cs_tab && ti >= 0) {
      float2 c = cs_tab[ti * 32 + i];
      cs = c.x, sn = c.y;
    } else {
      double angle = position * pow(theta, -2.0 * i / 64.0);
      cs = (float)cos(angle), sn = (float)sin(angle);
    }
    v = d < 32 ? values[d] * cs - values[d + 32] * sn
                : values[d] * cs + values[d - 32] * sn;
  }
  out[d] = v;
}

__global__ void k_index_q(const float* proj, const float* norm, float* queries,
                           int P, float eps, double theta, const int* pos = nullptr,
                           int base = 0, const float2* cs_tab = nullptr,
                           const int* __restrict__ d_rdelta = nullptr) {
  int t = blockIdx.x, h = blockIdx.y;
  // d_rdelta (M-RoPE decode): rope angle position += request rope_delta.
  // Only passed by decode callers (cs_tab == nullptr there); the prefill
  // cs_tab path must keep position-base == batch-local row.
  int position = pos ? *pos : base + t;
  if (d_rdelta) position += *d_rdelta;
  index_norm_rope(proj[(size_t)t * 640 + h * 128 + threadIdx.x], norm,
                  queries + (size_t)t * 512 + h * 128, position, eps,
                  theta, cs_tab, base);
}

// Pooled indexer keys: absolute 4-token blocks. base = absolute position of
// batch row 0; a block straddling the left edge (base % 4 != 0) reads its
// history rows from the raw ring (same values the decode path's
// k_index_append would see, summed in the same order -> bitwise identical).
__global__ void k_index_pool(const float* proj, const float* norm, float* keys,
                              float eps, double theta, int base = 0,
                              const float* ring = nullptr,
                              const float2* cs_tab = nullptr) {
  int b = blockIdx.x + (base / 4), d = threadIdx.x;  // absolute block id
  float v = 0.f;
  for (int j = 0; j < 4; j++) {
    int lt = b * 4 + j - base;  // batch-local row; < 0 -> history
    v += lt >= 0 ? proj[(size_t)lt * 640 + 512 + d]
                 : ring[(size_t)((b * 4 + j) % 4) * 128 + d];
  }
  index_norm_rope(v * 0.25f, norm, keys + (size_t)b * 128, b * 4, eps, theta,
                  cs_tab, base);
}

__global__ void k_index_ring(const float* proj, float* ring, int P, int base = 0) {
  int t = max(0, P - 4) + (int)blockIdx.x;
  if (t < P)
    ring[((base + t) % 4) * 128 + threadIdx.x] = proj[(size_t)t * 640 + 512 + threadIdx.x];
}

__global__ void k_index_append(const float* proj, const float* norm, float* ring,
                                float* keys, const int* pos, float eps, double theta,
                                const int* __restrict__ d_rdelta = nullptr) {
  int t = *pos, d = threadIdx.x;
  ring[(t % 4) * 128 + d] = proj[512 + d];
  __syncthreads();
  if (t % 4 != 3) return;
  float v = 0.f;
  for (int j = 0; j < 4; j++) v += ring[j * 128 + d];
  // d_rdelta (M-RoPE decode): pooled key's rope position is its first token's
  // cache position + request rope_delta (zero for text-only).
  int position = t - 3 + (d_rdelta ? *d_rdelta : 0);
  index_norm_rope(v * 0.25f, norm, keys + (size_t)(t / 4) * 128,
                  position, eps, theta);
}

__global__ void k_index_scores(const float* queries, const float* keys, float* scores,
                                int stride, const int* pos) {
  int b = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  if (*pos < 2051 || b >= (*pos + 1) / 4) return;
  for (int h = 0; h < 4; h++) {
    float sum = 0.f;
    for (int d = lane; d < 128; d += warpSize)
      sum += queries[h * 128 + d] * keys[(size_t)b * 128 + d];
    for (int s = warpSize / 2; s; s >>= 1) sum += __shfl_down_sync(~0ull, sum, s);
    if (!lane) scores[h * stride + b] = sum;
  }
}

// Exact top-512 via bounded merges, using rocPRIM radix sort. Float scores are
// nonnegative; low bits break equal-score ties by smaller block id.
template <int Items = 4, bool Combined = false>
__global__ void k_index_select(const float* scores, int stride, int* selected,
                                int first, const int* pos = nullptr) {
  using TopSort = rocprim::block_radix_sort<uint64_t, 256, Items>;
  using IdSort = rocprim::block_radix_sort<int, 256, 2>;
  __shared__ union { typename TopSort::storage_type top; IdSort::storage_type ids; } storage;
  __shared__ uint64_t best[512];
  int q = blockIdx.x, t = pos ? *pos : first + q, n = (t + 1) / 4;
  int tid = threadIdx.x;
  if (n <= 512) {
    for (int i = tid; i < 512; i += 256) selected[(size_t)q * 512 + i] = i < n ? i : -1;
    return;
  }
  best[tid] = best[tid + 256] = 0;
  __syncthreads();
  for (int base = 0; base < n; base += 256 * Items - 512) {
    uint64_t items[Items];
    for (int i = 0; i < Items; i++) {
      int rank = tid * Items + i;
      if (rank < 512) items[i] = best[rank];
      else {
        int b = base + rank - 512;
        float s = 0.f;
        if constexpr (Combined) {
          if (b < n) s = scores[(size_t)q * stride + b];
        } else {
          if (b < n)
            for (int h = 0; h < 4; h++) s += fmaxf(0.f, scores[((size_t)q * 4 + h) * stride + b]);
          s *= 0.08838834764831845f;
        }
        items[i] = b < n ? ((uint64_t)__float_as_uint(s) << 32) | (0xffffffffu - (uint32_t)b) : 0;
      }
    }
    TopSort().sort_desc(items, storage.top);
    __syncthreads();
    for (int i = 0; i < Items; i++)
      if (tid * Items + i < 512) best[tid * Items + i] = items[i];
    __syncthreads();
  }
  int ids[2];
  for (int i = 0; i < 2; i++) ids[i] = (int)(0xffffffffu - (uint32_t)best[tid * 2 + i]);
  __syncthreads();
  IdSort().sort(ids, storage.ids);
  for (int i = 0; i < 2; i++) selected[(size_t)q * 512 + tid * 2 + i] = ids[i];
}

// Exact top-512 selection, bit-identical output to k_index_select (same
// combined-score formula and the same total order: score desc, then smaller
// block id), but sort-free. Four MSD radix passes over the fp32 bit pattern
// (scores are nonnegative, so bits are monotone) resolve the exact 512th
// score threshold s*; the selected set is {s > s*} plus the smallest-id
// elements of {s == s*}, which is exactly the old kernel's uint64-key order.
// Emission scans b ascending, so ids come out in the same ascending order the
// old kernel's final IdSort produced. Requires n <= 8192 (LDS combined-score
// cache); larger n keeps the old kernel. n <= 512 fills directly (same as old).
template <bool Combined = false>
__global__ void k_index_select_rs(const float* __restrict__ scores, int stride,
                                  int* __restrict__ selected, int first,
                                  const int* __restrict__ pos = nullptr) {
  __shared__ float sbuf[8192];
  __shared__ unsigned hist[256];
  __shared__ unsigned shr_cnt[8];
  __shared__ unsigned shr_eq[8];
  __shared__ int s_passcnt, s_passbyte, s_cntgt;
  const int q = blockIdx.x, t = pos ? *pos : first + q, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  if (n <= 512) {
    for (int i = tid; i < 512; i += 256)
      selected[(size_t)q * 512 + i] = i < n ? i : -1;
    return;
  }
  // combine (ReLU per head, ordered sum, scale) into LDS + top-byte histogram
  hist[tid] = 0;
  __syncthreads();
  const float* qb = scores + (size_t)q * 4 * stride;
  for (int b = tid; b < n; b += 256) {
    float s = 0.f;
    if constexpr (Combined) {
      s = scores[(size_t)q * stride + b];
    } else {
#pragma unroll
      for (int h = 0; h < 4; h++) s += fmaxf(0.f, qb[(size_t)h * stride + b]);
      s *= 0.08838834764831845f;
    }
    sbuf[b] = s;
    atomicAdd(&hist[__float_as_uint(s) >> 24], 1u);
  }
  __syncthreads();
  // 4 MSD radix passes -> exact 512th-largest score bits in prefix
  unsigned prefix = 0;
  int K = 512;
  if (tid == 0) s_cntgt = 0;
  for (int p = 0; p < 4; p++) {
    const int shift = (3 - p) * 8;
    if (p) {
      __syncthreads();  // hist reuse
      hist[tid] = 0;
      __syncthreads();
      for (int b = tid; b < n; b += 256) {
        unsigned bits = __float_as_uint(sbuf[b]);
        if ((bits >> (shift + 8)) == prefix)
          atomicAdd(&hist[(bits >> shift) & 0xffu], 1u);
      }
      __syncthreads();
    }
    if (tid == 0) {
      int acc = 0, chosen = 255;
      for (int i = 255; i >= 0; i--) {
        int c = (int)hist[i];
        if (acc + c >= K) {
          chosen = i;
          break;
        }
        acc += c;
      }
      s_passcnt = acc;
      s_passbyte = chosen;
    }
    __syncthreads();
    K -= s_passcnt;
    if (tid == 0) s_cntgt += s_passcnt;
    prefix = (prefix << 8) | (unsigned)s_passbyte;
    __syncthreads();
  }
  const unsigned sstar = prefix;
  const int cnt_gt = s_cntgt;  // < 512 by construction
  const int need = 512 - cnt_gt;  // tie slots taken by smallest ids
  // Single ascending-b emit: sel = (s > s*) || (s == s* && tie-rank < need).
  // The old kernel's final IdSort yields ascending ids, so ties and strict
  // greaters must interleave in one id-ordered scan (not two phases).
  int outbase = 0, eqbase = 0;
  const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
  for (int c0 = 0; c0 < n; c0 += 256) {
    const int b = c0 + tid;
    unsigned bits = b < n ? __float_as_uint(sbuf[b]) : 0u;
    bool gt = b < n && bits > sstar;
    bool eq = b < n && bits == sstar;
    unsigned long long bal_eq = __ballot_sync(~0ull, eq);
    int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
    if (lane == 0) shr_eq[w] = __popcll(bal_eq);
    __syncthreads();
    int eq_woff = 0;
    for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
    bool sel = gt || (eq && eqbase + eq_woff + eq_pre < need);
    unsigned long long bal_sel = __ballot_sync(~0ull, sel);
    if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
    __syncthreads();
    int woff = 0, wtot = 0, eqtot = 0;
    for (int i = 0; i < nw; i++) {
      int c = (int)shr_cnt[i];
      woff += i < w ? c : 0;
      wtot += c;
      eqtot += (int)shr_eq[i];
    }
    if (sel)
      selected[(size_t)q * 512 + outbase + woff +
               __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
    outbase += wtot;
    eqbase = min(eqbase + eqtot, need);
    __syncthreads();  // shr reuse
  }
}


// Stream the existing combined scores instead of storing an unbounded row in LDS.
// Four exact radix passes find the threshold, then an ascending scan emits IDs.
// The total ordering and tie handling are identical to the existing selectors.
template <bool Combined>
__device__ __forceinline__ float index_stream_score(const float* scores, int stride, int q, int b) {
  if constexpr (Combined) return scores[(size_t)q * stride + b];
  float s = 0.f;
  #pragma unroll
  for (int h = 0; h < 4; h++) s += fmaxf(0.f, scores[((size_t)q * 4 + h) * stride + b]);
  return s * 0.08838834764831845f;
}
template <bool Combined = false>
__global__ void k_index_select_stream(const float* __restrict__ scores, int stride,
                                  int* __restrict__ selected, int first,
                                  const int* __restrict__ pos = nullptr) {
  __shared__ unsigned hist[256];
  __shared__ unsigned shr_cnt[8];
  __shared__ unsigned shr_eq[8];
  __shared__ int s_passcnt, s_passbyte, s_cntgt;
  const int q = blockIdx.x, t = pos ? *pos : first + q, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  if (n <= 512) {
    for (int i = tid; i < 512; i += 256)
      selected[(size_t)q * 512 + i] = i < n ? i : -1;
    return;
  }
  // First pass: combine scores and build the top-byte histogram.
  hist[tid] = 0;
  __syncthreads();
  const float* qb = scores + (size_t)q * 4 * stride;
  for (int b = tid; b < n; b += 256) {
    float s = 0.f;
    if constexpr (Combined) {
      s = scores[(size_t)q * stride + b];
    } else {
#pragma unroll
      for (int h = 0; h < 4; h++) s += fmaxf(0.f, qb[(size_t)h * stride + b]);
      s *= 0.08838834764831845f;
    }
    atomicAdd(&hist[__float_as_uint(s) >> 24], 1u);
  }
  __syncthreads();
  // 4 MSD radix passes -> exact 512th-largest score bits in prefix
  unsigned prefix = 0;
  int K = 512;
  if (tid == 0) s_cntgt = 0;
  for (int p = 0; p < 4; p++) {
    const int shift = (3 - p) * 8;
    if (p) {
      __syncthreads();  // hist reuse
      hist[tid] = 0;
      __syncthreads();
      for (int b = tid; b < n; b += 256) {
        unsigned bits = __float_as_uint(index_stream_score<Combined>(scores, stride, q, b));
        if ((bits >> (shift + 8)) == prefix)
          atomicAdd(&hist[(bits >> shift) & 0xffu], 1u);
      }
      __syncthreads();
    }
    if (tid == 0) {
      int acc = 0, chosen = 255;
      for (int i = 255; i >= 0; i--) {
        int c = (int)hist[i];
        if (acc + c >= K) {
          chosen = i;
          break;
        }
        acc += c;
      }
      s_passcnt = acc;
      s_passbyte = chosen;
    }
    __syncthreads();
    K -= s_passcnt;
    if (tid == 0) s_cntgt += s_passcnt;
    prefix = (prefix << 8) | (unsigned)s_passbyte;
    __syncthreads();
  }
  const unsigned sstar = prefix;
  const int cnt_gt = s_cntgt;  // < 512 by construction
  const int need = 512 - cnt_gt;  // tie slots taken by smallest ids
  // Single ascending-b emit: sel = (s > s*) || (s == s* && tie-rank < need).
  // The old kernel's final IdSort yields ascending ids, so ties and strict
  // greaters must interleave in one id-ordered scan (not two phases).
  int outbase = 0, eqbase = 0;
  const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
  for (int c0 = 0; c0 < n; c0 += 256) {
    const int b = c0 + tid;
    unsigned bits = b < n ? __float_as_uint(index_stream_score<Combined>(scores, stride, q, b)) : 0u;
    bool gt = b < n && bits > sstar;
    bool eq = b < n && bits == sstar;
    unsigned long long bal_eq = __ballot_sync(~0ull, eq);
    int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
    if (lane == 0) shr_eq[w] = __popcll(bal_eq);
    __syncthreads();
    int eq_woff = 0;
    for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
    bool sel = gt || (eq && eqbase + eq_woff + eq_pre < need);
    unsigned long long bal_sel = __ballot_sync(~0ull, sel);
    if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
    __syncthreads();
    int woff = 0, wtot = 0, eqtot = 0;
    for (int i = 0; i < nw; i++) {
      int c = (int)shr_cnt[i];
      woff += i < w ? c : 0;
      wtot += c;
      eqtot += (int)shr_eq[i];
    }
    if (sel)
      selected[(size_t)q * 512 + outbase + woff +
               __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
    outbase += wtot;
    eqbase = min(eqbase + eqtot, need);
    __syncthreads();  // shr reuse
  }
}



// 16 queries x 32 keys: fuse FP32 heads, ReLU and combine, skipping
// causally unreachable tiles. rocBLAS gfx1151 MT32x32x8 SU32 SUS256
// starts the K loop at ((key / 32) & 1) * 64 and wraps at 128.
// Keep one FMA chain per head and the same ordered head reduction.
__global__ void __launch_bounds__(256)
k_index_scores_tiled(const float* __restrict__ q,
                     const float* __restrict__ keys,
                     float* __restrict__ out, int nb, int count, int first) {
  const int q0 = (int)blockIdx.y * 16, k0 = (int)blockIdx.x * 32;
  if (q0 >= count || k0 >= min(nb, (first + min(q0 + 16, count)) / 4)) return;
  __shared__ float qs[32][65];
  __shared__ float ks[32][33];
  const int tid = threadIdx.x, qr = tid / 16, kr = (tid % 16) * 2;
  float a[4][2] = {};
  for (int stage = 0; stage < 4; stage++) {
    const int d0 = (stage * 32 + ((int)blockIdx.x & 1) * 64) & 127;
    for (int j = tid; j < 512; j += 256) {
      int row = j / 8, d = (j % 8) * 4;
      float4 qv = make_float4(0, 0, 0, 0), kv = qv;
      if (q0 + row / 4 < count)
        qv = *reinterpret_cast<const float4*>(q + (size_t)(q0 * 4 + row) * 128 + d0 + d);
      if (row < 32 && k0 + row < nb)
        kv = *reinterpret_cast<const float4*>(keys + (size_t)(k0 + row) * 128 + d0 + d);
      qs[d][row] = qv.x; qs[d + 1][row] = qv.y;
      qs[d + 2][row] = qv.z; qs[d + 3][row] = qv.w;
      if (row < 32) {
        ks[d][row] = kv.x; ks[d + 1][row] = kv.y;
        ks[d + 2][row] = kv.z; ks[d + 3][row] = kv.w;
      }
    }
    __syncthreads();
#pragma unroll
    for (int d = 0; d < 32; d++) {
      float qv[4], kv[2];
#pragma unroll
      for (int h = 0; h < 4; h++) qv[h] = qs[d][qr * 4 + h];
#pragma unroll
      for (int c = 0; c < 2; c++) kv[c] = ks[d][kr + c];
#pragma unroll
      for (int h = 0; h < 4; h++)
#pragma unroll
        for (int c = 0; c < 2; c++) a[h][c] = fmaf(qv[h], kv[c], a[h][c]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int c = 0; c < 2; c++) {
    const int t = q0 + qr, b = k0 + kr + c;
    if (t < count && b < nb && b < (first + t + 1) / 4) {
      float s = 0.f;
#pragma unroll
      for (int h = 0; h < 4; h++) s += fmaxf(0.f, a[h][c]);
      out[(size_t)t * nb + b] = s * 0.08838834764831845f;
    }
  }
}

static void index_select(const float* scores, int stride, int* selected,
                         int first, int count, const int* pos = nullptr) {
  int n = pos ? stride : min(stride, (first + count) / 4);
  static bool old_sel = getenv("GDEC_INDEX_OLDSEL") != nullptr;
  if (!old_sel && n > 512 && n <= 8192) {
    k_index_select_rs<<<count, 256, 0, g_str>>>(scores, stride, selected, first,
                                                pos);
    return;
  }
  if (n <= 1024)
    k_index_select<4><<<count, 256, 0, g_str>>>(scores, stride, selected, first, pos);
  else if (n <= 2048)
    k_index_select<8><<<count, 256, 0, g_str>>>(scores, stride, selected, first, pos);
  else
    k_index_select<16><<<count, 256, 0, g_str>>>(scores, stride, selected, first, pos);
}

__global__ void k_moe_route_pack(const int* ids, int* keys, int* values, int pairs, int k) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pairs) return;
  keys[p] = ids[(p / k) * 16 + p % k];
  values[p] = p;
}

__global__ void k_moe_route_unpack(const int* keys, const int* values,
                                   const int* ids, const float* weights,
                                   int* tokidx, float* pw, int* eoff, int* pairids,
                                   int pairs, int k, int experts) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pairs) return;
  int e = keys[p], original = values[p], t = original / k;
  tokidx[p] = t;
  pw[p] = weights[t * 16 + original % k];
  if (pairids) {
    int rank = 0;
    for (int s = 0; s < k; s++) rank += ids[t * 16 + s] < e;
    pairids[t * k + rank] = p;
  }
  // Fill offsets for empty experts as well; each boundary owns a disjoint range.
  int previous = p ? keys[p - 1] : -1;
  if (previous != e)
    for (int j = previous + 1; j <= e; j++) eoff[j] = p;
  if (p == pairs - 1)
    for (int j = e + 1; j <= experts; j++) eoff[j] = pairs;
}

struct MoeTile { int expert, first, count; };

__global__ void k_moe_tiles(const int* eoff, MoeTile* tiles, int* total, int experts) {
  using Scan = rocprim::block_scan<int, 512>;
  __shared__ Scan::storage_type storage;
  int e = threadIdx.x;
  int first = e < experts ? eoff[e] : 0;
  int ne = e < experts ? eoff[e + 1] - first : 0;
  int count = (ne + 63) / 64, begin;
  Scan().exclusive_scan(count, begin, 0, storage);
  if (e == 511) *total = begin + count;
  for (int j = 0; j < count; j++) tiles[begin + j] = {e, first + j * 64, min(64, ne - j * 64)};
}

struct GpuMoeRouting {
  int *keys_in = nullptr, *keys_out = nullptr, *values_in = nullptr, *values_out = nullptr;
  void* scratch = nullptr;
  size_t scratch_bytes = 0;
  unsigned bits = 0;

  void init(int capacity, int experts) {
    for (int v = experts - 1; v; v >>= 1) bits++;
    CK(hipMalloc(&keys_in, (size_t)capacity * sizeof(int)));
    CK(hipMalloc(&keys_out, (size_t)capacity * sizeof(int)));
    CK(hipMalloc(&values_in, (size_t)capacity * sizeof(int)));
    CK(hipMalloc(&values_out, (size_t)capacity * sizeof(int)));
    CK(rocprim::radix_sort_pairs(nullptr, scratch_bytes, keys_in, keys_out,
                                 values_in, values_out, capacity, 0, bits));
    CK(hipMalloc(&scratch, scratch_bytes));
  }

  void run(const int* ids, const float* weights, int* tokidx, float* pw,
           int* eoff, int* pairids, int P, int k, int experts) {
    int pairs = P * k;
    k_moe_route_pack<<<(pairs + 255) / 256, 256, 0, g_str>>>(ids, keys_in, values_in, pairs, k);
    // rocPRIM's stable sort preserves the original token/slot order per expert.
    CK(rocprim::radix_sort_pairs(scratch, scratch_bytes, keys_in, keys_out,
                                 values_in, values_out, pairs, 0, bits, g_str));
    k_moe_route_unpack<<<(pairs + 255) / 256, 256, 0, g_str>>>(
        keys_out, values_out, ids, weights, tokidx, pw, eoff, pairids, pairs, k, experts);
  }

  ~GpuMoeRouting() {
    if (scratch) {
      CK(hipFree(keys_in)); CK(hipFree(keys_out));
      CK(hipFree(values_in)); CK(hipFree(values_out)); CK(hipFree(scratch));
    }
  }
};

// ============================ kernels =======================================

// --- Q4C-P GEMV: y[r] = sum_c cb[nib]*scale[r][c/32]*x[c]; one warp per row.
// v3 (decode GEMV governance): cb staged in LDS, fp32 x, 2 rows per warp.
// Bit-exact vs the original 1-row/warp version: same per-lane fma order,
// same shuffle reduction tree (verified d=0 in tools/decode_gemv_bench.cu).
__global__ void k_q4cp_gemv(const uint8_t* __restrict__ codes,
                            const uint8_t* __restrict__ scales,
                            const float* __restrict__ cb,
                            const float* __restrict__ x, float* __restrict__ y,
                            uint64_t rows, uint64_t cols, uint64_t scale_stride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t groups_per_row = cols / 32;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = row0 + 1 < rows;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < groups_per_row; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    const float* xg = x + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    y[row0] = acc0;
    if (has1) y[row0 + 1] = acc1;
  }
}

// --- q4cp multi-row GEMV: y[P,N] = x[P,K] . W[N,K]^T, weight streamed once ---
// Same dequant math as k_q4cp_gemv (16-entry codebook + fp16 group scale),
// but each warp applies its 2 weight rows to all P input rows. x is small and
// stays L2-resident across warps, so DRAM traffic ~ N*K/2 bytes regardless of
// P — this replaces the dequant->bf16->hipBLASLt path for small P, which
// costs ~4.5x the weight bytes in extra traffic.
template <int P, bool XBF16 = false>
__global__ void k_q4cp_gemv_mr(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const void* __restrict__ x,
                               float* __restrict__ y, uint64_t rows,
                               uint64_t cols, uint64_t scale_stride,
                               uint64_t xstride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t groups_per_row = cols / 32;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = row0 + 1 < rows;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t g = lane; g < groups_per_row; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
#pragma unroll
    for (int p = 0; p < P; p++) {
      float xv[32];
      if (XBF16) {
        const uint16_t* xg =
            (const uint16_t*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          uint4 t = *(const uint4*)(xg + i * 8);
          const uint16_t* th = (const uint16_t*)&t;
#pragma unroll
          for (int j = 0; j < 8; j++)
            xv[i * 8 + j] = __bfloat162float(*(__hip_bfloat16*)(&th[j]));
        }
      } else {
        const float* xg = (const float*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          float4 t = *(const float4*)(xg + i * 4);
          xv[i * 4] = t.x;
          xv[i * 4 + 1] = t.y;
          xv[i * 4 + 2] = t.z;
          xv[i * 4 + 3] = t.w;
        }
      }
      uint32_t u0[4] = {w0[0], w0[1], w0[2], w0[3]};
      uint32_t u1[4] = {w1[0], w1[1], w1[2], w1[3]};
      float b0 = 0.f, b1 = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          b0 += cbf[u0[i] & 0xf] * xv[i * 8 + j];
          b1 += cbf[u1[i] & 0xf] * xv[i * 8 + j];
          u0[i] >>= 4;
          u1[i] >>= 4;
        }
      }
      acc0[p] += b0 * s0;
      acc1[p] += b1 * s1;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- bf16-weight multi-row GEMV: y[P,N] = x[P,K] . W[N,K]^T (bf16 weights) ---
template <int P>
__global__ void k_bf16_gemv_mr(const uint16_t* __restrict__ w,
                               const float* __restrict__ x,
                               float* __restrict__ y, uint64_t rows,
                               uint64_t cols, uint64_t xstride) {
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  bool has1 = row0 + 1 < rows;
  const uint16_t* r0 = w + row0 * cols;
  const uint16_t* r1 = w + (row0 + 1) * cols;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t c = lane * 8; c < cols; c += warpSize * 8) {
    uint4 pk0 = *(const uint4*)(r0 + c);
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    if (has1) pk1 = *(const uint4*)(r1 + c);
    const uint16_t* h0 = (const uint16_t*)&pk0;
    const uint16_t* h1 = (const uint16_t*)&pk1;
    float w0[8], w1[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      w0[j] = __bfloat162float(*(__hip_bfloat16*)(&h0[j]));
      w1[j] = __bfloat162float(*(__hip_bfloat16*)(&h1[j]));
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      const float* xg = x + (size_t)p * xstride + c;
      float4 xa = *(const float4*)xg;
      float4 xb = *(const float4*)(xg + 4);
      float b0 = w0[0] * xa.x + w0[1] * xa.y + w0[2] * xa.z + w0[3] * xa.w +
                 w0[4] * xb.x + w0[5] * xb.y + w0[6] * xb.z + w0[7] * xb.w;
      float b1 = w1[0] * xa.x + w1[1] * xa.y + w1[2] * xa.z + w1[3] * xa.w +
                 w1[4] * xb.x + w1[5] * xb.y + w1[6] * xb.z + w1[7] * xb.w;
      acc0[p] += b0;
      acc1[p] += b1;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- q8g64 multi-row GEMV (overlay o_proj): [codes uint8 x cols][(s,m) fp16 per 64] ---
template <int P>
__global__ void k_q8g64_gemv_mr(const uint8_t* __restrict__ w,
                                const float* __restrict__ x,
                                float* __restrict__ y, uint64_t rows,
                                uint64_t cols, uint64_t xstride) {
  uint64_t stride = cols + cols / 64 * 4;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  bool has1 = row0 + 1 < rows;
  const uint8_t* r0 = w + row0 * stride;
  const uint8_t* r1 = w + (row0 + 1) * stride;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t g = lane; g < cols / 64; g += warpSize) {
    const uint8_t* c0 = r0 + g * 64;
    const __half* sm0 = (const __half*)(r0 + cols + g * 4);
    float s0 = __half2float(sm0[0]), m0 = __half2float(sm0[1]);
    float s1 = 0.f, m1 = 0.f;
    const uint8_t* c1 = (has1 ? r1 : r0) + g * 64;  // !has1: dead, s1=m1=0
    const __half* sm1 = (const __half*)(r1 + cols + g * 4);
    if (has1) {
      s1 = __half2float(sm1[0]);
      m1 = __half2float(sm1[1]);
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      const float* xg = x + (size_t)p * xstride + g * 64;
      float b0 = 0.f, b1 = 0.f, xs = 0.f;
#pragma unroll
      for (int j = 0; j < 64; j += 4) {
        float4 t = *(const float4*)(xg + j);
        b0 += (float)c0[j] * t.x + (float)c0[j + 1] * t.y +
              (float)c0[j + 2] * t.z + (float)c0[j + 3] * t.w;
        b1 += (float)c1[j] * t.x + (float)c1[j + 1] * t.y +
              (float)c1[j + 2] * t.z + (float)c1[j + 3] * t.w;
        xs += t.x + t.y + t.z + t.w;
      }
      // q8g64 dequant: w = c*s + m  ->  w.x = s*(c.x) + m*sum(x)
      acc0[p] += b0 * s0 + m0 * xs;
      acc1[p] += b1 * s1 + m1 * xs;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- q8g64 GEMV (overlay o_proj): per row [cols uint8][cols/64 x (fp16 s, fp16 m)]
__global__ void k_q8g64_gemv(const uint8_t* __restrict__ w, const float* __restrict__ x,
                             float* __restrict__ y, uint64_t rows, uint64_t cols) {
  uint64_t stride = cols + cols / 64 * 4;
  uint64_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rp = w + row * stride;
  float acc = 0.f;
  for (uint64_t g = lane; g < cols / 64; g += warpSize) {
    const uint8_t* codes = rp + g * 64;
    const __half* sm = (const __half*)(rp + cols + g * 4);
    float s = __half2float(sm[0]), mn = __half2float(sm[1]);
    float a = 0.f, cs = 0.f;
    const float* xg = x + g * 64;
#pragma unroll
    for (int j = 0; j < 64; j += 4) {
      float4 xv = *(const float4*)(xg + j);
      a += codes[j] * xv.x + codes[j + 1] * xv.y + codes[j + 2] * xv.z +
           codes[j + 3] * xv.w;
      cs += xv.x + xv.y + xv.z + xv.w;
    }
    acc += a * s + mn * cs;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down_sync(~0ull, acc, off);
  if (lane == 0) y[row] = acc;
}

// --- BF16 GEMV: one warp per row, float4 (8 bf16) loads
__global__ void k_bf16_gemv(const uint16_t* __restrict__ w, const float* __restrict__ x,
                            float* __restrict__ y, uint64_t rows, uint64_t cols) {
  uint64_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint16_t* rw = w + row * cols;
  float acc = 0.f;
  for (uint64_t c = lane * 8; c < cols; c += warpSize * 8) {
    uint4 pk = *(const uint4*)(rw + c);  // 8 bf16
    float4 xa = *(const float4*)(x + c);
    float4 xb = *(const float4*)(x + c + 4);
    const uint16_t* h = (const uint16_t*)&pk;
    float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int j = 0; j < 8; j++) {
      uint32_t u = (uint32_t)h[j] << 16;
      float wf;
      memcpy(&wf, &u, 4);
      acc += wf * xv[j];
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down_sync(~0ull, acc, off);
  if (lane == 0) y[row] = acc;
}

// --- Q4C-P single-row dequant (embedding gather): out[cols]
__global__ void k_q4cp_row(const uint8_t* __restrict__ codes,
                           const uint8_t* __restrict__ scales,
                           const float* __restrict__ cb, const int* __restrict__ d_token,
                           uint64_t cols, uint64_t scale_stride, float* __restrict__ out) {
  uint64_t row = (uint64_t)*d_token;
  uint64_t g = threadIdx.x;  // one block, cols/32 threads
  if (g * 32 >= cols) return;
  const uint8_t* rc = codes + row * (cols / 2) + g * 16;
  const uint8_t* rs = scales + row * scale_stride;
  float s = __half2float(*(const __half*)(rs + g * 2));
  uint4 pk = *(const uint4*)rc;
  uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t v = w[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      out[g * 32 + i * 8 + j] = __ldg(cb + (v & 0xf)) * s;
      v >>= 4;
    }
  }
}

// --- zero-centered RMSNorm, optionally grouped. grid = one block per group.
__global__ void k_rmsnorm_zc(const float* __restrict__ x, const float* __restrict__ w,
                             float* __restrict__ out, int n, float eps) {
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < n; i += blockDim.x) ss += x[i] * x[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / n + eps);
  for (int i = t; i < n; i += blockDim.x) out[i] = x[i] * inv * (1.f + w[i]);
}

__device__ inline uint16_t f2bf(float f);

// grouped zc rmsnorm over flat (ngroups x gsize): one block per group.
// Batched: grid may exceed ngroups; weight index wraps with % ngroups.
template <bool Bf16 = false>
__global__ void k_rmsnorm_zc_grouped(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ out, int gsize, float eps,
                                     int ngroups, uint16_t* __restrict__ bf16 = nullptr) {
  const float* xg = x + (size_t)blockIdx.x * gsize;
  const float* wg = w + (size_t)(blockIdx.x % ngroups) * gsize;
  float* og = out + (size_t)blockIdx.x * gsize;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < gsize; i += blockDim.x) ss += xg[i] * xg[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / gsize + eps);
  for (int i = t; i < gsize; i += blockDim.x) {
    float value = xg[i] * inv * (1.f + wg[i]);
    og[i] = value;
    if (Bf16) bf16[(size_t)blockIdx.x * gsize + i] = f2bf(value);
  }
}

// --- GDN conv update: state (3,qkv) shifts; out = silu(sum_j w[j]*hist[j])
// hist[0..2] = previous three inputs (oldest first), cur = current input.
__global__ void k_gdn_conv(const float* __restrict__ cur, const float* __restrict__ cw,
                           float* __restrict__ convstate, float* __restrict__ out,
                           int qkv) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= qkv) return;
  float* st = convstate;  // 3 x qkv
  float a = cw[ch * 4 + 0] * st[ch] + cw[ch * 4 + 1] * st[qkv + ch] +
            cw[ch * 4 + 2] * st[2 * qkv + ch] + cw[ch * 4 + 3] * cur[ch];
  st[ch] = st[qkv + ch];
  st[qkv + ch] = st[2 * qkv + ch];
  st[2 * qkv + ch] = cur[ch];
  float sg = 1.f / (1.f + expf(-a));
  out[ch] = a * sg;
}

// --- GDN recurrence, one block per v-head (128 threads).
// S layout: [h][j(128 k)][i(128 v)] row-major. q,k already l2normed; q prescaled.
__global__ void k_gdn_step(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, const float* __restrict__ g,
                           const float* __restrict__ beta, float* __restrict__ S,
                           float* __restrict__ out, int dk, int dv, int hv, int hk) {
  int h = blockIdx.x;
  int group = hv / hk;
  const float* kh = k + (h / group) * dk;
  const float* qh = q + (h / group) * dk;
  const float* vh = v + h * dv;
  float* Sh = S + (size_t)h * dk * dv;
  float alpha = expf(g[h]);
  float be = beta[h];
  // 4 j-chunks x dv threads: thread t owns column i=t%dv, j-chunk jc=t/dv
  int t = threadIdx.x;
  int i = t % dv, jc = t / dv;
  int j0 = jc * (dk / 4), j1 = j0 + dk / 4;
  __shared__ float part[4][128];  // partial kv / o per j-chunk
  __shared__ float sdelta[128];
  // pass 1: decay + kv partial
  float kv = 0.f;
  for (int j = j0; j < j1; j++) {
    float s = Sh[j * dv + i] * alpha;
    Sh[j * dv + i] = s;
    kv += s * kh[j];
  }
  part[jc][i] = kv;
  __syncthreads();
  float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
  float delta = (vh[i] - kvf) * be;
  if (jc == 0) sdelta[i] = delta;
  __syncthreads();
  delta = sdelta[i];
  // pass 2: rank-1 update + pass 3: output partial
  float o = 0.f;
  for (int j = j0; j < j1; j++) {
    float s = Sh[j * dv + i] + kh[j] * delta;
    Sh[j * dv + i] = s;
    o += s * qh[j];
  }
  __syncthreads();
  part[jc][i] = o;
  __syncthreads();
  if (jc == 0) out[h * dv + i] = part[0][i] + part[1][i] + part[2][i] + part[3][i];
}

// Short speculative batches: keep the entire recurrent state in registers.
// The four partial reductions and accumulation order follow k_gdn_step.
// qkv and out may alias; all input values are read before output is stored.
__global__ void k_gdn_verify(const float* qkv, const float* g48,
                             const float* b48, float* S, float* out,
                             int P, int stride) {
  const int h = blockIdx.x, i = threadIdx.x % 128;
  const int jc = threadIdx.x / 128, j0 = jc * 32;
  float* Sh = S + (size_t)h * 128 * 128;
  float state[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) state[j] = Sh[(j0 + j) * 128 + i];
  __shared__ float part[4][128];
  __shared__ float delta_shared[128];
  for (int row = 0; row < P; ++row) {
    const float* q = qkv + (size_t)row * stride + (h / 3) * 128;
    const float* k = q + 2048;
    const float value = qkv[(size_t)row * stride + 4096 + h * 128 + i];
    const float alpha = expf(g48[row * 48 + h]);
    const float beta = b48[row * 48 + h];
    float kv = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      state[j] *= alpha;
      kv += state[j] * k[j0 + j];
    }
    part[jc][i] = kv;
    __syncthreads();
    float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
    float delta = (value - kvf) * beta;
    if (jc == 0) delta_shared[i] = delta;
    __syncthreads();
    delta = delta_shared[i];
    float result = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      state[j] += k[j0 + j] * delta;
      result += state[j] * q[j0 + j];
    }
    __syncthreads();
    part[jc][i] = result;
    __syncthreads();
    if (jc == 0)
      out[(size_t)row * stride + 4096 + h * 128 + i] =
          part[0][i] + part[1][i] + part[2][i] + part[3][i];
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 32; ++j) Sh[(j0 + j) * 128 + i] = state[j];
}

// --- indexer proj snapshot: keep every verify row's raw ring value (the
// 128-float tail of the 640-wide proj) so a rollback can rebuild the 4-slot
// raw ring at any accepted depth.
__global__ void k_iproj_snap(const float* __restrict__ proj,
                             float* __restrict__ snap, int P) {
  int t = blockIdx.x, d = threadIdx.x;
  if (t < P) snap[(size_t)t * 128 + d] = proj[(size_t)t * 640 + 512 + d];
}

// --- GDN snapshot: serially advance S over the chunk's P rows (same math as
// k_gdn_step, no output), writing S after each row into snap[t]. S lives in
// registers (32/thread) and d_S is left untouched — the chunked scan after
// this kernel still sees and advances the pre-chunk state. Lets speculative
// rollback install the state at any accepted depth without a re-prefill.
// snap t-stride = ngdn*48*128*128.
__global__ void k_gdn_snap(const float* __restrict__ qkv,
                           const float* __restrict__ g48,
                           const float* __restrict__ b48,
                           const float* __restrict__ S, float* __restrict__ snap,
                           int P, int qkvstride, size_t snaptstride) {
  int h = blockIdx.x;  // 48 v-heads, group of 3 over 16 k-heads
  const int dk = 128, dv = 128;
  const float* Sh = S + (size_t)h * dk * dv;
  float* Sn = snap + (size_t)h * dk * dv;
  int t = threadIdx.x;
  int i = t % dv, jc = t / dv;
  int j0 = jc * (dk / 4);
  float sreg[dk / 4];
#pragma unroll
  for (int j = 0; j < dk / 4; j++) sreg[j] = Sh[(j0 + j) * dv + i];
  __shared__ float part[4][128];
  __shared__ float sdelta[128];
  for (int tt = 0; tt < P; tt++) {
    const float* row = qkv + (size_t)tt * qkvstride;
    const float* kh = row + 2048 + (h / 3) * dk;
    const float* vh = row + 4096 + h * dv;
    float alpha = expf(g48[tt * 48 + h]);
    float be = b48[tt * 48 + h];
    float kv = 0.f;
#pragma unroll
    for (int j = 0; j < dk / 4; j++) {
      sreg[j] *= alpha;
      kv += sreg[j] * kh[j0 + j];
    }
    part[jc][i] = kv;
    __syncthreads();
    float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
    float delta = (vh[i] - kvf) * be;
    if (jc == 0) sdelta[i] = delta;
    __syncthreads();
    delta = sdelta[i];
#pragma unroll
    for (int j = 0; j < dk / 4; j++) {
      sreg[j] += kh[j0 + j] * delta;
      Sn[(size_t)tt * snaptstride + (j0 + j) * dv + i] = sreg[j];
    }
    __syncthreads();
  }
}

// --- conv state snapshot: convst after row t = rows (t-2,t-1,t) of the raw
// qkv stream, with the live (pre-chunk) convst filling rows < 0. snap layout
// per t: [3][qkv], t-stride = ngdn*3*qkv.
__global__ void k_convst_snap(const float* __restrict__ qkvb,
                              const float* __restrict__ convst,
                              float* __restrict__ snap, int P, int qkv,
                              size_t snaptstride) {
  int t = blockIdx.x;
  float* out = snap + (size_t)t * snaptstride;
  for (int c = threadIdx.x; c < qkv; c += blockDim.x) {
    out[c] = t >= 2   ? qkvb[(size_t)(t - 2) * qkv + c]
             : t == 1 ? convst[2 * qkv + c]
                      : convst[qkv + c];
    out[qkv + c] = t >= 1 ? qkvb[(size_t)(t - 1) * qkv + c] : convst[2 * qkv + c];
    out[2 * qkv + c] = qkvb[(size_t)t * qkv + c];
  }
}

// --- indexer raw-ring rebuild after rollback: slot s must hold the proj row
// of the largest position p <= qpos with p%4==s (qpos = q+a+1). Rows from the
// verify chunk come from iproj_snap[qi][p-qbase], older rows from the
// checkpoint ring. One block per QSA layer.
__global__ void k_iraw_restore(const float* __restrict__ iproj_snap,
                               const float* __restrict__ ckpt_iraw,
                               float* __restrict__ iraw, int qbase, int qpos,
                               int snapstride) {
  int qi = blockIdx.x, d = threadIdx.x;
  const float* snapl = iproj_snap + (size_t)qi * snapstride * 128;
  for (int s = 0; s < 4; s++) {
    int p = qpos - ((qpos % 4 - s + 4) % 4);
    const float* src = p >= qbase
                           ? snapl + (size_t)(p - qbase) * 128
                           : ckpt_iraw + (size_t)qi * 512 + (size_t)(p % 4) * 128;
    iraw[(size_t)qi * 512 + s * 128 + d] = src[d];
  }
}

// --- GDN q/k L2 norm + q scale: one block per qk head
__global__ void k_l2norm_qk(float* __restrict__ q, float* __restrict__ k, int dk,
                            float qscale) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  float* qh = q + h * dk;
  float* kh = k + h * dk;
  float v = (t < dk) ? qh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float iq = rsqrtf(sm[0] + 1e-6f);
  __syncthreads();
  float kv = (t < dk) ? kh[t] : 0.f;
  sm[t] = kv * kv;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float ik = rsqrtf(sm[0] + 1e-6f);
  if (t < dk) {
    qh[t] = v * iq * qscale;
    kh[t] = kv * ik;
  }
}

// --- GDN gates: alpha_decay[g] = -exp(A_log)*softplus(a+dt_bias) (= log decay),
//     beta = sigmoid(b).  hv threads.  Batched: blockIdx.x = token row (stride).
__global__ void k_gdn_gates(const float* __restrict__ a, const float* __restrict__ b,
                            const float* __restrict__ A_log,
                            const float* __restrict__ dt_bias,
                            float* __restrict__ g_out, float* __restrict__ beta_out,
                            int hv, int stride) {
  int h = threadIdx.x;
  if (h >= hv) return;
  size_t off = (size_t)blockIdx.x * stride;
  a += off;
  b += off;
  g_out += off;
  beta_out += off;
  float sp = a[h] + dt_bias[h];
  sp = sp > 20.f ? sp : log1pf(expf(sp));
  g_out[h] = -expf(A_log[h]) * sp;
  beta_out[h] = 1.f / (1.f + expf(-b[h]));
}

// --- GDN gated norm: per v-head plain RMSNorm(128) * sigmoid(z)
// Batched: grid.y = token row; y/out row stride ys, z row stride zs.
// Bf16: additionally emit compact [grid.y, gridDim.x*dv] bf16 (same f2bf RNE
// as k_f32_to_bf16 on out, so the consumer GEMM can skip its conversion).
template <bool Bf16 = false>
__global__ void k_gdn_gatednorm(const float* __restrict__ y, const float* __restrict__ z,
                                const float* __restrict__ w, float* __restrict__ out,
                                int dv, float eps, int ys, int zs,
                                uint16_t* __restrict__ bf16 = nullptr) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  const float* yh = y + (size_t)blockIdx.y * ys + h * dv;
  const float* zh = z + (size_t)blockIdx.y * zs + h * dv;
  float* oh = out + (size_t)blockIdx.y * ys + h * dv;
  float v = (t < dv) ? yh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / dv + eps);
  if (t < dv) {
    float zz = zh[t];
    float value = v * inv * w[t] / (1.f + expf(-zz));
    oh[t] = value;
    if (Bf16) bf16[((size_t)blockIdx.y * gridDim.x + h) * dv + t] = f2bf(value);
  }
}

// --- GDN gated norm, warp-per-token (dv = 128 fixed): one wave32 per
// (token, head); lane owns dims lane, lane+32, lane+64, lane+96. Reduction
// reproduces the smem tree of k_gdn_gatednorm exactly: the off=64 layer
// pairs (d, d+64), off=32 adds the (d+32, d+96) pair, shfl_down 16..1 does
// the rest -> bit-exact, but one block covers 32 tokens instead of 1.
// grid (heads, ceil(P/32)), block 128 (4 warps); warp w takes tokens
// blockIdx.y*32 + k*4 + w, k = 0..7.
template <bool Bf16 = false>
__global__ void k_gdn_gatednorm_b(const float* __restrict__ y,
                                  const float* __restrict__ z,
                                  const float* __restrict__ w,
                                  float* __restrict__ out, float eps, int ys, int zs,
                                  uint16_t* __restrict__ bf16, int P) {
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float w0 = w[lane], w1 = w[lane + 32], w2 = w[lane + 64],
              w3 = w[lane + 96];
#pragma unroll 2
  for (int k = 0; k < 8; k++) {
    int t = blockIdx.y * 32 + k * 4 + warp;  // warp-uniform
    if (t >= P) return;
    const float* yh = y + (size_t)t * ys + h * 128;
    const float* zh = z + (size_t)t * zs + h * 128;
    float* oh = out ? out + (size_t)t * ys + h * 128 : nullptr;
    float v0 = yh[lane], v1 = yh[lane + 32], v2 = yh[lane + 64],
          v3 = yh[lane + 96];
    // Bit-exactness vs the smem tree: products must round individually before
    // the adds, but ffp-contract=fast would fuse mul+add into FMA (the old
    // kernel is saved by its smem round-trip). The empty asm barrier
    // materializes each product as a rounded VGPR value, no memory traffic.
    float q0 = v0 * v0, q1 = v1 * v1, q2 = v2 * v2, q3 = v3 * v3;
    asm volatile("" : "+v"(q0), "+v"(q1), "+v"(q2), "+v"(q3));
    float s64 = q0 + q2;
    float s96 = q1 + q3;
    float s = s64 + s96;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(~0ull, s, off);
    s = __shfl_sync(~0ull, s, 0);
    float inv = rsqrtf(s / 128 + eps);
    float r0 = v0 * inv * w0 / (1.f + expf(-zh[lane]));
    float r1 = v1 * inv * w1 / (1.f + expf(-zh[lane + 32]));
    float r2 = v2 * inv * w2 / (1.f + expf(-zh[lane + 64]));
    float r3 = v3 * inv * w3 / (1.f + expf(-zh[lane + 96]));
    if (out) {  // fp32 out is dead when the consumer GEMM reads only the bf16
      oh[lane] = r0;
      oh[lane + 32] = r1;
      oh[lane + 64] = r2;
      oh[lane + 96] = r3;
    }
    if (Bf16) {
      uint16_t* bp = bf16 + ((size_t)t * gridDim.x + h) * 128;
      bp[lane] = f2bf(r0);
      bp[lane + 32] = f2bf(r1);
      bp[lane + 64] = f2bf(r2);
      bp[lane + 96] = f2bf(r3);
    }
  }
}

// --- RoPE partial: rotate first rotary_dim dims of each head (half-split).
// vec: (nheads, dh). One warp per head... simple: one thread per pair.
// d_rdelta (M-RoPE decode): rope angle position is *d_pos + *d_rdelta;
// absent/zero for text-only requests (bitwise identical to before).
__global__ void k_rope(float* __restrict__ vec, int nheads, int dh, int rotary,
                       const int* __restrict__ d_pos, double theta,
                       const int* __restrict__ d_rdelta = nullptr) {
  int pos = *d_pos;
  if (d_rdelta) pos += *d_rdelta;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // over nheads * rotary/2
  int half = rotary / 2;
  if (idx >= nheads * half) return;
  int h = idx / half, i = idx % half;
  double ang = pos * pow(theta, -2.0 * i / rotary);
  float cs = (float)cos(ang), sn = (float)sin(ang);
  float* p = vec + h * dh;
  float x0 = p[i], x1 = p[i + half];
  p[i] = x0 * cs - x1 * sn;
  p[i + half] = x0 * sn + x1 * cs;
}

// --- QSA attention step: one block per q head, threads over positions.
// kcache/vcache: (pos, hkv*dh) accumulated; current token's k/v already appended.
template <typename KVT = float>
__global__ void k_qsa_step(const float* __restrict__ q, const KVT* __restrict__ kc,
                           const KVT* __restrict__ vc, float* __restrict__ out,
                           const int* __restrict__ d_pos, int dh, int hkv, int hq,
                           const int* selected = nullptr, bool skip_sparse = false) {
  int visible = *d_pos + 1;
  bool sparse = selected && visible >= 2052;
  if (sparse && skip_sparse) return;
  int ntok = sparse ? 2048 + visible % 4 : visible;
  int h = blockIdx.x;
  int kvh = h / (hq / hkv);
  const float* qh = q + h * dh;
  float scale = 1.f / sqrtf((float)dh);
  // pass 1: scores + running max (threads strided over positions)
  int t = threadIdx.x;
  float mx = -1e30f;
  for (int p = t; p < ntok; p += blockDim.x) {
    const KVT* kp = kc + ((size_t)qsa_source_pos(p, visible, selected) * hkv + kvh) * dh;
    float acc = 0.f;
    for (int i = 0; i < dh; i++) acc += qh[i] * kv_ld1(kp + i);
    mx = fmaxf(mx, acc * scale);
  }
  // block max reduce
  __shared__ float red[1024];
  red[t] = mx;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) red[t] = fmaxf(red[t], red[t + off]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  // pass 2: exp sum
  float se = 0.f;
  for (int p = t; p < ntok; p += blockDim.x) {
    const KVT* kp = kc + ((size_t)qsa_source_pos(p, visible, selected) * hkv + kvh) * dh;
    float acc = 0.f;
    for (int i = 0; i < dh; i++) acc += qh[i] * kv_ld1(kp + i);
    se += expf(acc * scale - mx);
  }
  red[t] = se;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    __syncthreads();
  }
  float invsum = 1.f / red[0];
  // pass 3: weighted v — threads over dh? need per-dim sums. dh=256, block 256.
  // recompute weights per position serially but parallel over dh.
  if (t < dh) {
    float acc = 0.f;
    for (int p = 0; p < ntok; p++) {
      int source = qsa_source_pos(p, visible, selected);
      const KVT* kp = kc + ((size_t)source * hkv + kvh) * dh;
      float s = 0.f;
      for (int i = 0; i < dh; i++) s += qh[i] * kv_ld1(kp + i);
      float w = expf(s * scale - mx) * invsum;
      acc += w * kv_ld1(vc + ((size_t)source * hkv + kvh) * dh + t);
    }
    out[h * dh + t] = acc;
  }
}

// --- per-head zc rmsnorm for QSA q/k (dh=256, one block per head)
// (reuse k_rmsnorm_zc with n=256 per block over head pointer)

// --- sigmoid gate multiply: out[i] = a[i] * sigmoid(g[i])
__global__ void k_sigmoid_gate(const float* __restrict__ a, const float* __restrict__ g,
                               float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] / (1.f + expf(-g[i]));
}

// --- GR read: needs gemv results; combine on device.
// t = silu(Wd Rhat / 4) computed via gemv into t320 then k_silu_scale.
__global__ void k_silu_scale(float* __restrict__ v, float inv_div, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = v[i] * inv_div;
    v[i] = x / (1.f + expf(-x));
  }
}
__global__ void k_sigmoid_inplace(float* __restrict__ v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] = 1.f / (1.f + expf(-v[i]));
}
// x_out[i] = mean_b G[b*d+i] * Rhat[b*d+i]
__global__ void k_gr_combine(const float* __restrict__ G, const float* __restrict__ Rhat,
                             float* __restrict__ x, int d, int branches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  float acc = 0.f;
  for (int b = 0; b < branches; b++) acc += G[b * d + i] * Rhat[b * d + i];
  x[i] = acc / branches;
}
// GR write: s = 2*sigmoid(w/4); R[b*d+i] += s[b] * y[i]
__global__ void k_gr_write(const float* __restrict__ w4, float* __restrict__ R,
                           const float* __restrict__ y, int d, int branches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  for (int b = 0; b < branches; b++) {
    float s = 2.f / (1.f + expf(-w4[b] * 0.25f));
    R[b * d + i] += s * y[i];
  }
}

// --- MoE router: softmax + top-k done on host (512 floats download is cheap),
//     experts gathered per selected expert via row-offset GEMV.
// silu(gate)*up for mid dims
__global__ void k_silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                           float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = g[i];
    out[i] = a / (1.f + expf(-a)) * u[i];
  }
}
// acc[r] += w * y[r]
__global__ void k_axpy(float* __restrict__ acc, const float* __restrict__ y, float w,
                       int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] += w * y[i];
}

// --- GPU router: softmax + top-k over n logits, one block. ------------------
// out: ids[k], w[k] with w = p_e / sum(selected p) (official norm_topk_prob).
// Batched: one block per token, ids/ws rows of 16.
__global__ void k_router_topk(const float* __restrict__ logits, int* __restrict__ ids,
                              float* __restrict__ ws, int n, int k) {
  logits += (size_t)blockIdx.x * n;
  ids += (size_t)blockIdx.x * 16;
  ws += (size_t)blockIdx.x * 16;
  __shared__ float p[512];
  __shared__ float sval[512];
  __shared__ int sidx[512];
  __shared__ float raw[16];
  int t = threadIdx.x;
  float v = logits[t];  // n == 512 == blockDim
  // block max
  sval[t] = v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sval[t] = fmaxf(sval[t], sval[t + off]);
    __syncthreads();
  }
  float mx = sval[0];
  p[t] = expf(v - mx);
  __syncthreads();
  sval[t] = p[t];
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sval[t] += sval[t + off];
    __syncthreads();
  }
  float inv_se = 1.f / sval[0];
  p[t] *= inv_se;
  __syncthreads();
  for (int j = 0; j < k; j++) {
    sval[t] = p[t];
    sidx[t] = t;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (t < off && sval[t + off] > sval[t]) {
        sval[t] = sval[t + off];
        sidx[t] = sidx[t + off];
      }
      __syncthreads();
    }
    if (t == 0) {
      ids[j] = sidx[0];
      raw[j] = sval[0];
      p[sidx[0]] = -1e30f;
    }
    __syncthreads();
  }
  if (t == 0) {
    float sw = 0.f;
    for (int j = 0; j < k; j++) sw += raw[j];
    for (int j = 0; j < k; j++) ws[j] = raw[j] / sw;
  }
}

// --- grouped expert GEMV (shared x): slot-major y[(slot, r)] ---------------
// Batched: nslots = P*kpt; slot -> token tok = slot/kpt, expert s = slot%kpt;
// x = xs + tok*x_stride, expert id = ids[tok*id_stride + s].
// v3 (decode GEMV governance): cb in LDS + 2 rows per warp; bit-exact vs the
// original (same per-lane fma order and reduction). Grid must now cover
// nslots * ceil(rows_per/2) warps, not nslots * rows_per.
__global__ void k_q4cp_gemv_gg(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const float* __restrict__ x, float* __restrict__ y,
                               const int* __restrict__ ids, uint64_t rows_per,
                               uint64_t cols, uint64_t scale_stride, int nslots,
                               uint64_t x_stride, int id_stride, int kpt) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t pairs_per_slot = (rows_per + 1) / 2;
  uint64_t gw = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)nslots * pairs_per_slot) return;
  int slot = gw / pairs_per_slot;
  int tok = slot / kpt, s = slot - tok * kpt;
  uint64_t row0 =
      (uint64_t)ids[tok * id_stride + s] * rows_per + (gw % pairs_per_slot) * 2;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = (gw % pairs_per_slot) * 2 + 1 < rows_per;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  const float* xt = x + (size_t)tok * x_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < cols / 32; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    const float* xg = xt + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    uint64_t out = (uint64_t)slot * rows_per + (gw % pairs_per_slot) * 2;
    y[out] = acc0;
    if (has1) y[out + 1] = acc1;
  }
}

// --- grouped silu(gate)*up: guv (nslots, 2*mid) -> hid (nslots, mid) --------
__global__ void k_silu_mul_g(const float* __restrict__ guv, float* __restrict__ hid,
                             int mid, int nslots) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nslots * mid) return;
  int slot = i / mid, r = i % mid;
  float a = guv[slot * 2 * mid + r];
  hid[i] = a / (1.f + expf(-a)) * guv[slot * 2 * mid + mid + r];
}

// --- grouped expert GEMV (per-slot x) with fused weighted accumulate --------
// acc[tok*acc_stride + r] += ws[tok*id_stride+s] * dot(W[ids[...]*rows_per + r],
// xs + slot*x_stride). v3: full warp per row, 2 rows per warp, cb in LDS;
// per-row math bit-exact vs the LPW=32 original on wave32 (the atomicAdd set
// per output element is unchanged). Grid must cover nslots*ceil(rows_per/2)
// warps.
__global__ void k_q4cp_gemv_gd(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const float* __restrict__ xs, float* __restrict__ acc,
                               const int* __restrict__ ids, const float* __restrict__ ws,
                               uint64_t rows_per, uint64_t cols, uint64_t scale_stride,
                               uint64_t x_stride, int nslots, int id_stride,
                               uint64_t acc_stride, int kpt) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t pairs_per_slot = (rows_per + 1) / 2;
  int lane = threadIdx.x % warpSize;
  uint64_t gw = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)nslots * pairs_per_slot) return;
  int slot = gw / pairs_per_slot;
  uint64_t rp = gw % pairs_per_slot;
  int tok = slot / kpt, s = slot - tok * kpt;
  uint64_t row0 = (uint64_t)ids[tok * id_stride + s] * rows_per + rp * 2;
  const float* x = xs + (uint64_t)slot * x_stride;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = rp * 2 + 1 < rows_per;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < cols / 32; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    const float* xg = x + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    float w = ws[tok * id_stride + s];
    atomicAdd(acc + (size_t)tok * acc_stride + rp * 2, w * acc0);
    if (has1) atomicAdd(acc + (size_t)tok * acc_stride + rp * 2 + 1, w * acc1);
  }
}

// Small-P verify down projection: one wave covers two output rows and keeps
// all top-10 expert partials in registers. Two 16-lane subgroups avoid the
// 12 idle lanes of the K=640 full-wave mapping and eliminate atomic output
// accumulation. This is intentionally limited to topk=10 by the host route.
template <int KPT>
__global__ void k_q4cp_gemv_gd_topk_h16(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const float* __restrict__ cb, const float* __restrict__ xs,
    float* __restrict__ out, const int* __restrict__ ids,
    const float* __restrict__ ws, uint64_t rows_per, uint64_t cols,
    uint64_t scale_stride, uint64_t x_stride, int P, int id_stride,
    uint64_t out_stride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t row_pairs = (rows_per + 1) / 2;
  uint64_t gw =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)P * row_pairs) return;
  int tok = gw / row_pairs;
  uint64_t rp = gw % row_pairs;
  int half = (threadIdx.x % warpSize) / 16;
  int lane = threadIdx.x & 15;
  uint64_t r = rp * 2 + half;
  if (r >= rows_per) return;
  float acc[KPT] = {};
#pragma unroll
  for (int s = 0; s < KPT; s++) {
    uint64_t row = (uint64_t)ids[tok * id_stride + s] * rows_per + r;
    const uint8_t* rc = codes + row * (cols / 2);
    const uint8_t* rs = scales + row * scale_stride;
    const float* x = xs + ((size_t)tok * KPT + s) * x_stride;
    for (uint64_t g = lane; g < cols / 32; g += 16) {
      uint4 pk = *(const uint4*)(rc + g * 16);
      float sc = __half2float(*(const __half*)(rs + g * 2));
      const float* xg = x + g * 32;
      float xv[32];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float4 t = *(const float4*)(xg + i * 4);
        xv[i * 4] = t.x;
        xv[i * 4 + 1] = t.y;
        xv[i * 4 + 2] = t.z;
        xv[i * 4 + 3] = t.w;
      }
      uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t v = w[i];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          a += cbf[v & 0xf] * xv[i * 8 + j];
          v >>= 4;
        }
      }
      acc[s] += a * sc;
    }
  }
#pragma unroll
  for (int s = 0; s < KPT; s++)
    for (int off = 8; off > 0; off >>= 1)
      acc[s] += __shfl_down_sync(~0ull, acc[s], off, 16);
  if (lane == 0) {
    float y = 0.f;
#pragma unroll
    for (int s = 0; s < KPT; s++) y += ws[tok * id_stride + s] * acc[s];
    out[(size_t)tok * out_stride + r] = y;
  }
}

// --- PLE conv (dilation 3, k=4): ring 9 x 10240; tap j pairs t-(3-j)*3
__global__ void k_ple_conv(const float* __restrict__ ring,
                           const int* __restrict__ d_ringpos,
                           const float* __restrict__ cw, float* __restrict__ out,
                           int n) {
  int ringpos = *d_ringpos;
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= n) return;
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int back = (3 - j) * 3;
    int rp = ((ringpos - back) % 9 + 9) % 9;
    a += cw[ch * 4 + j] * ring[(size_t)rp * n + ch];
  }
  out[ch] = a / (1.f + expf(-a));  // silu
}

// grouped zc rmsnorm into PLE ring slot *d_ringpos (ring = 9 x 10240)
__global__ void k_rmsnorm_zc_ring(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ ring,
                                  const int* __restrict__ d_ringpos, int gsize,
                                  float eps) {
  float* out = ring + (size_t)(*d_ringpos) * 10240;
  const float* xg = x + (size_t)blockIdx.x * gsize;
  const float* wg = w + (size_t)blockIdx.x * gsize;
  float* og = out + (size_t)blockIdx.x * gsize;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < gsize; i += blockDim.x) ss += xg[i] * xg[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / gsize + eps);
  for (int i = t; i < gsize; i += blockDim.x) og[i] = xg[i] * inv * (1.f + wg[i]);
}

// --- QSA kv cache append at *d_pos (replaces pos-dependent memcpy) ----------
__global__ void k_store_kv(const float* __restrict__ kb, const float* __restrict__ vb,
                           float* __restrict__ kc, float* __restrict__ vc,
                           const int* __restrict__ d_pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 512) return;
  size_t off = (size_t)(*d_pos) * 512 + i;
  kc[off] = kb[i];
  vc[off] = vb[i];
}

// BF16 variant of the above: same layout, values rounded RNE via f2bf.
__global__ void k_store_kv_bf16(const float* __restrict__ kb,
                                const float* __restrict__ vb,
                                uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                const int* __restrict__ d_pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 512) return;
  size_t off = (size_t)(*d_pos) * 512 + i;
  kc[off] = f2bf(kb[i]);
  vc[off] = f2bf(vb[i]);
}

// --- end-of-step: pos++, ringpos = (ringpos+1)%9 ----------------------------
__global__ void k_step_incr(int* __restrict__ d_pos, int* __restrict__ d_ringpos) {
  *d_pos += 1;
  *d_ringpos = (*d_ringpos + 1) % 9;
}
// R[ch] += U[ch] + convout[ch]
__global__ void k_add2(float* __restrict__ R, const float* __restrict__ a,
                       const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) R[i] += a[i] + b[i];
}
__global__ void k_acc(float* __restrict__ acc, const float* __restrict__ a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] += a[i];
}
// Copy P contiguous d-rows into the heads of 4d-stride rows (k_repeat4_b then
// expands them in place).
__global__ void k_row_head_copy(const float* __restrict__ x,
                                float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  R[(size_t)t * 4 * d + c] = x[(size_t)t * d + c];
}
// Copy P contiguous d-rows into branch slot b of 4d-stride rows.
__global__ void k_row_branch_copy(const float* __restrict__ x,
                                  float* __restrict__ R, int d, int b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  R[(size_t)t * 4 * d + (size_t)b * d + c] = x[(size_t)t * d + c];
}
// Broadcast-add P contiguous d-rows onto all four branches of 4d-stride rows.
__global__ void k_acc_broadcast4(const float* __restrict__ x,
                                 float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  float v = x[(size_t)t * d + c];
  size_t base = (size_t)t * 4 * d + c;
  R[base] += v;
  R[base + d] += v;
  R[base + 2 * d] += v;
  R[base + 3 * d] += v;
}
// PLE per-branch gate: g_m = sum(norm_k_m * norm_q_m)/sqrt(d) -> signed sqrt ->
// sigmoid; U[m*d+i] = alpha_m * v[i]
__global__ void k_ple_gate(const float* __restrict__ kn, const float* __restrict__ qn,
                           const float* __restrict__ v, float* __restrict__ U, int d,
                           int branches) {
  int m = blockIdx.x;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float acc = 0.f;
  for (int i = t; i < d; i += blockDim.x) acc += kn[m * d + i] * qn[m * d + i];
  sm[t] = acc;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float g = sm[0] / sqrtf((float)d);
  g = (g < 0 ? -1.f : 1.f) * sqrtf(fmaxf(fabsf(g), 1e-6f));
  float alpha = 1.f / (1.f + expf(-g));
  for (int i = t; i < d; i += blockDim.x) U[m * d + i] = alpha * v[i];
}

// ================== PLE batched prefill (Phase 3d-1) kernels =================

// --- k_ple_gate over P tokens: block (m, t); kn/qn/U rows are [P, branches*d]
__global__ void k_ple_gate_b(const float* __restrict__ kn,
                             const float* __restrict__ qn,
                             const float* __restrict__ v, float* __restrict__ U, int d,
                             int branches) {
  int m = blockIdx.x, t = blockIdx.y;
  size_t row = ((size_t)t * branches + m) * d;
  const float* knm = kn + row;
  const float* qnm = qn + row;
  const float* vt = v + (size_t)t * d;
  float* Um = U + row;
  __shared__ float sm[1024];
  int i = threadIdx.x;
  float acc = 0.f;
  for (int j = i; j < d; j += blockDim.x) acc += knm[j] * qnm[j];
  sm[i] = acc;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (i < off) sm[i] += sm[i + off];
    __syncthreads();
  }
  float g = sm[0] / sqrtf((float)d);
  g = (g < 0 ? -1.f : 1.f) * sqrtf(fmaxf(fabsf(g), 1e-6f));
  float alpha = 1.f / (1.f + expf(-g));
  for (int j = i; j < d; j += blockDim.x) Um[j] = alpha * vt[j];
}

// --- batched PLE conv (dilation 3, k=4): the 9-slot ring aliases tap back-9
// to the current position, so out[t] = silu(cw0*Un[t] + cw1*Un[t-6] +
// cw2*Un[t-3] + cw3*Un[t]); missing (negative) positions read as zero.
__global__ void k_ple_conv_b(const float* __restrict__ Un,
                             const float* __restrict__ cw, float* __restrict__ out,
                             int P, int n, int base = 0,
                             const float* __restrict__ ring = nullptr) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= (int64_t)P * n) return;
  int t = (int)(i / n), ch = (int)(i % n);
  int a = base + t;  // absolute position
  const float* c4 = cw + (size_t)ch * 4;
  float v = c4[0] * Un[i];
  // taps reach back 6/3 positions; before the batch start they read the ring
  // (holds the last 9 absolute positions), at sequence start they vanish
  if (a >= 6) v += c4[1] * (t >= 6 ? Un[i - (size_t)6 * n]
                                    : ring[(size_t)((a - 6) % 9) * n + ch]);
  if (a >= 3) v += c4[2] * (t >= 3 ? Un[i - (size_t)3 * n]
                                    : ring[(size_t)((a - 3) % 9) * n + ch]);
  v += c4[3] * Un[i];
  out[i] = v / (1.f + expf(-v));
}

// --- PLE ring handoff: ring slot (base+P-9+j)%9 = Un[P-9+j] (negative pos skipped)
__global__ void k_ple_ring_wr(const float* __restrict__ Un, float* __restrict__ ring,
                              int P, int n, int base = 0) {
  int pos = P - 9 + blockIdx.x;
  if (pos < 0) return;
  const float* src = Un + (size_t)pos * n;
  float* dst = ring + (size_t)((base + pos) % 9) * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// --- PLE fp8 staging dequant: raw e4m3 bytes (16 rows x 160 per token, packed
// 2560/token) -> f32 embeddings. Bit-exact with hgn::fp8e4m3_to_f32 * scale.
__device__ inline float fp8e4m3_f32_dev(uint8_t v) {
  uint32_t s = v >> 7, e = (v >> 3) & 0xf, m = v & 7;
  float r;
  if (e == 0) r = ldexpf((float)m, -9);
  else if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
  else r = ldexpf((float)(8 + m), (int)e - 10);
  return s ? -r : r;
}

__global__ void k_ple_fp8_dequant(const uint8_t* __restrict__ src,
                                  float* __restrict__ dst, float scale,
                                  size_t n16) {  // n16 = elements/16
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  uint4 raw = ((const uint4*)src)[i];
  const uint8_t* b = (const uint8_t*)&raw;
  float t[16];
#pragma unroll
  for (int j = 0; j < 16; j++) t[j] = fp8e4m3_f32_dev(b[j]) * scale;
  float4* d4 = (float4*)dst + i * 4;
  d4[0] = make_float4(t[0], t[1], t[2], t[3]);
  d4[1] = make_float4(t[4], t[5], t[6], t[7]);
  d4[2] = make_float4(t[8], t[9], t[10], t[11]);
  d4[3] = make_float4(t[12], t[13], t[14], t[15]);
}

// --- argmax over n floats -> writes int id
__global__ void k_argmax(const float* __restrict__ x, int n, int* __restrict__ out) {
  __shared__ float bv[1024];
  __shared__ int bi[1024];
  int t = threadIdx.x;
  float best = -1e30f;
  int bidx = 0;
  for (int i = t; i < n; i += blockDim.x) {
    if (x[i] > best) {
      best = x[i];
      bidx = i;
    }
  }
  bv[t] = best;
  bi[t] = bidx;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off && bv[t + off] > bv[t]) {
      bv[t] = bv[t + off];
      bi[t] = bi[t + off];
    }
    __syncthreads();
  }
  if (t == 0) out[0] = bi[0];
}

// vector copy / fill helpers
__global__ void k_repeat4(const float* __restrict__ x, float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) {
    R[i] = x[i];
    R[d + i] = x[i];
    R[2 * d + i] = x[i];
    R[3 * d + i] = x[i];
  }
}

// batched embedding gather for prefill: one block per token, out rows are
// outstride apart (the 4*d Rb row). Same math as k_q4cp_row.
__global__ void k_q4cp_row_b(const uint8_t* __restrict__ codes,
                             const uint8_t* __restrict__ scales,
                             const float* __restrict__ cb,
                             const int* __restrict__ d_tokens, uint64_t cols,
                             uint64_t scale_stride, float* __restrict__ out,
                             uint64_t outstride) {
  uint64_t row = (uint64_t)d_tokens[blockIdx.x];
  out += blockIdx.x * outstride;
  uint64_t g = threadIdx.x;  // one block per token, cols/32 threads
  if (g * 32 >= cols) return;
  const uint8_t* rc = codes + row * (cols / 2) + g * 16;
  const uint8_t* rs = scales + row * scale_stride;
  float s = __half2float(*(const __half*)(rs + g * 2));
  uint4 pk = *(const uint4*)rc;
  uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t v = w[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      out[g * 32 + i * 8 + j] = __ldg(cb + (v & 0xf)) * s;
      v >>= 4;
    }
  }
}

// batched repeat4 over P rows of 4*d contiguous floats (x == R in-place safe)
__global__ void k_repeat4_b(const float* __restrict__ x, float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  size_t base = (size_t)t * 4 * d + c;
  float v = x[base];
  R[base] = v;
  R[base + d] = v;
  R[base + 2 * d] = v;
  R[base + 3 * d] = v;
}

// ================== prefill batch (Phase 3a) kernels =========================

// fp32 -> bf16 (round to nearest even)
__device__ inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

__device__ inline float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

__global__ void k_bf16_to_f32(const uint16_t* __restrict__ x,
                              float* __restrict__ out, uint64_t n) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = bf2f(x[i]);
}

// Direct embedding gather into the four-branch BF16 hyper-connection state.
__global__ void k_q4cp_row_b_hc_bf16(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const float* __restrict__ cb, const int* __restrict__ d_tokens,
    uint64_t cols, uint64_t scale_stride, uint16_t* __restrict__ out) {
  uint64_t row = (uint64_t)d_tokens[blockIdx.x];
  out += (size_t)blockIdx.x * 4 * cols;
  uint64_t g = threadIdx.x;
  if (g * 32 >= cols) return;
  const uint8_t* rc = codes + row * (cols / 2) + g * 16;
  const uint8_t* rs = scales + row * scale_stride;
  float s = __half2float(*(const __half*)(rs + g * 2));
  uint4 pk = *(const uint4*)rc;
  uint32_t packed[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t v = packed[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int c = (int)(g * 32 + i * 8 + j);
      uint16_t value = f2bf(__ldg(cb + (v & 0xf)) * s);
      out[c] = value;
      out[cols + c] = value;
      out[2 * cols + c] = value;
      out[3 * cols + c] = value;
      v >>= 4;
    }
  }
}

__global__ void k_rmsnorm_zc_grouped_bf16(
    const uint16_t* __restrict__ x, const float* __restrict__ w,
    uint16_t* __restrict__ out, int gsize, float eps, int ngroups) {
  const uint16_t* xg = x + (size_t)blockIdx.x * gsize;
  const float* wg = w + (size_t)(blockIdx.x % ngroups) * gsize;
  uint16_t* og = out + (size_t)blockIdx.x * gsize;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < gsize; i += blockDim.x) {
    float value = bf2f(xg[i]);
    ss += value * value;
  }
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / gsize + eps);
  for (int i = t; i < gsize; i += blockDim.x)
    og[i] = f2bf(bf2f(xg[i]) * inv * (1.f + wg[i]));
}

// --- Q4C-P full dequant to bf16: one thread per 32-col group
__global__ void k_dequant_q4cp_bf16(const uint8_t* __restrict__ codes,
                                    const uint8_t* __restrict__ scales,
                                    const float* __restrict__ cb,
                                    uint16_t* __restrict__ out, uint64_t rows,
                                    uint64_t cols, uint64_t scale_stride) {
  uint64_t gpr = cols / 32;
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= rows * gpr) return;
  uint64_t r = gid / gpr, g = gid % gpr;
  const uint8_t* rc = codes + r * (cols / 2) + g * 16;
  float s = __half2float(*(const __half*)(scales + r * scale_stride + g * 2));
  uint16_t* o = out + r * cols + g * 32;
#pragma unroll
  for (int b = 0; b < 16; b++) {
    uint32_t v = rc[b];
    o[2 * b] = f2bf(__ldg(cb + (v & 0xf)) * s);
    o[2 * b + 1] = f2bf(__ldg(cb + (v >> 4)) * s);
  }
}

// --- q8g64 full dequant to bf16: w'[r][c] = code*s + m (one thread per 64-col group)
__global__ void k_dequant_q8g64_bf16(const uint8_t* __restrict__ w,
                                     uint16_t* __restrict__ out, uint64_t rows,
                                     uint64_t cols) {
  uint64_t gpr = cols / 64;
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= rows * gpr) return;
  uint64_t r = gid / gpr, g = gid % gpr;
  const uint8_t* rp = w + r * (cols + gpr * 4);
  const uint8_t* codes = rp + g * 64;
  const __half* sm = (const __half*)(rp + cols + g * 4);
  float s = __half2float(sm[0]), mn = __half2float(sm[1]);
  uint16_t* o = out + r * cols + g * 64;
#pragma unroll
  for (int j = 0; j < 64; j++) o[j] = f2bf(codes[j] * s + mn);
}

// --- fp32 [P,n] (row stride xs) -> compact bf16 [P,n]
__global__ void k_f32_to_bf16(const float* __restrict__ x, uint16_t* __restrict__ out,
                              uint64_t xs, int P, int n) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= (uint64_t)P * n) return;
  uint64_t t = i / n, c = i % n;
  out[i] = f2bf(x[t * xs + c]);
}

// 4 floats/thread variant; requires n % 4 == 0, xs % 4 == 0 and 16B-aligned
// base pointers (host-checked, scalar kernel is the fallback)
__global__ void k_f32_to_bf16_v4(const float* __restrict__ x,
                                 uint16_t* __restrict__ out, uint64_t xs, int P,
                                 int n) {
  uint64_t n4 = (uint64_t)n / 4;
  uint64_t i4 = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i4 >= (uint64_t)P * n4) return;
  uint64_t t = i4 / n4;
  uint64_t c = (i4 - t * n4) * 4;
  float4 v = *(const float4*)(x + t * xs + c);
  uint16_t o[4] = {f2bf(v.x), f2bf(v.y), f2bf(v.z), f2bf(v.w)};
  uint2 pk;
  pk.x = (uint32_t)o[0] | ((uint32_t)o[1] << 16);
  pk.y = (uint32_t)o[2] | ((uint32_t)o[3] << 16);
  *(uint2*)(out + i4 * 4) = pk;
}

// Compact BF16 [P,512] V rows into block-transposed
// [block][kvh][dim][slot] storage. Each thread owns one dimension of one KV
// head and packs the four token slots, so the WMMA V gather can issue one
// contiguous 64-bit load per four-key group. Partial edge blocks only update
// slots covered by this chunk; masked WMMA keys never consume the others.
__global__ void k_f32_to_bf16_v4_bt(const float* __restrict__ x,
                                    uint16_t* __restrict__ out, uint64_t xs,
                                    int P, int n, int base) {
  const int dims = n / 2;
  const int b0 = base / 4;
  const int b1 = (base + P + 3) / 4;
  uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  uint64_t work = (uint64_t)(b1 - b0) * n;
  if (gid >= work) return;
    int b = b0 + (int)(gid / n);
    int hd = (int)(gid % n);
  int kvh = hd / dims;
  int dim = hd - kvh * dims;
  uint16_t v[4] = {};
  int valid = 0;
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    int abs = b * 4 + slot;
    if (abs >= base && abs < base + P) {
      v[slot] = f2bf(x[(size_t)(abs - base) * xs + kvh * (n / 2) + dim]);
      valid |= 1 << slot;
    }
  }
  uint16_t* dst = out + (((size_t)b * 2 + kvh) * (n / 2) + dim) * 4;
  if (valid == 0xf) {
    uint2 pk;
    pk.x = (uint32_t)v[0] | ((uint32_t)v[1] << 16);
    pk.y = (uint32_t)v[2] | ((uint32_t)v[3] << 16);
    *(uint2*)dst = pk;
  } else {
#pragma unroll
    for (int slot = 0; slot < 4; ++slot)
      if (valid & (1 << slot)) dst[slot] = v[slot];
  }
}

// --- batched GR combine: x[t*d+i] = mean_b G[t][b*d+i] * Rhat[t][b*d+i]
template <bool Fused = false>
__global__ void k_gr_combine_b(const float* __restrict__ G, const float* __restrict__ Rhat,
                               float* __restrict__ x, int d, int branches, int P,
                               uint16_t* __restrict__ bf16 = nullptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * d) return;
  int t = i / d, c = i - t * d;
  const float* Gt = G + (size_t)t * branches * d;
  const float* Rt = Rhat + (size_t)t * branches * d;
  float acc = 0.f;
  for (int b = 0; b < branches; b++) {
    float gate = Gt[b * d + c];
    if (Fused) gate = 1.f / (1.f + expf(-gate));
    acc += gate * Rt[b * d + c];
  }
  float value = acc / branches;
  x[i] = value;
  if (Fused) bf16[i] = f2bf(value);
}

// BF16 HC state; one block per token.
__global__ void k_gr_combine_b_hc_bf16(
    const float* __restrict__ G, const uint16_t* __restrict__ Rhat,
    float* __restrict__ x, int d, int branches, int P,
    uint16_t* __restrict__ xbf16) {
  int t = blockIdx.x;
  if (t >= P) return;
  const float* Gt = G + (size_t)t * branches * d;
  const uint16_t* Rt = Rhat + (size_t)t * branches * d;
  float* xt = x + (size_t)t * d;
  uint16_t* x16t = xbf16 + (size_t)t * d;
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    float acc = 0.f;
    for (int b = 0; b < branches; b++) {
      float gate = 1.f / (1.f + expf(-Gt[b * d + c]));
      acc += gate * bf2f(Rt[b * d + c]);
    }
    float value = acc / branches;
    xt[c] = value;
    x16t[c] = f2bf(value);
  }
}

// --- batched GR write: R[t][b*d+i] += 2*sigmoid(w4[t][b]/4) * y[t*d+i]
__global__ void k_gr_write_b(const float* __restrict__ w4, float* __restrict__ R,
                             const float* __restrict__ y, int d, int branches, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * d) return;
  int t = i / d, c = i - t * d;
  float* Rt = R + (size_t)t * branches * d;
  const float* wt = w4 + t * branches;
  for (int b = 0; b < branches; b++) {
    float s = 2.f / (1.f + expf(-wt[b] * 0.25f));
    Rt[b * d + c] += s * y[i];
  }
}

__global__ void k_gr_write_b_hc_bf16(const float* __restrict__ w4,
                                     uint16_t* __restrict__ R,
                                     const float* __restrict__ y, int d,
                                     int branches, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * d) return;
  int t = i / d, c = i - t * d;
  uint16_t* Rt = R + (size_t)t * branches * d;
  const float* wt = w4 + t * branches;
  for (int b = 0; b < branches; b++) {
    float s = 2.f / (1.f + expf(-wt[b] * 0.25f));
    Rt[b * d + c] = f2bf(bf2f(Rt[b * d + c]) + s * y[i]);
  }
}

// BF16 residual state and BF16 normalized output. The rounded residual is
// accumulated into the norm (scatter/store-before-norm order), halving HC
// state traffic.
__global__ void k_gr_scatter_norm_b_hc_bf16(
    const float* __restrict__ w4, uint16_t* __restrict__ R,
    const float* __restrict__ y, const float* __restrict__ w,
    uint16_t* __restrict__ Rhat, int d, int branches, int P, float eps) {
  int group = blockIdx.x;
  if (group >= P * branches) return;
  int t = group / branches;
  int b = group - t * branches;
  int lane = threadIdx.x;
  uint16_t* Rg = R + (size_t)group * d;
  const float* yg = y + (size_t)t * d;
  const float* wg = w + (size_t)b * d;
  uint16_t* og = Rhat + (size_t)group * d;
  __shared__ float sm[256];
  __shared__ float inject;
  if (lane == 0)
    inject = 2.f / (1.f + expf(-w4[(size_t)t * branches + b] * 0.25f));
  __syncthreads();

  float values[10];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 10; j++) {
    int c = lane + j * 256;
    float rounded = 0.f;
    if (c < d) {
      uint16_t r16 = f2bf(bf2f(Rg[c]) + inject * yg[c]);
      Rg[c] = r16;
      rounded = bf2f(r16);
      ss += rounded * rounded;
    }
    values[j] = rounded;
  }
  sm[lane] = ss;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (lane < off) sm[lane] += sm[lane + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / d + eps);
#pragma unroll
  for (int j = 0; j < 10; j++) {
    int c = lane + j * 256;
    if (c < d) og[c] = f2bf(values[j] * inv * (1.f + wg[c]));
  }
}

// One-block-per-token variant: y is staged in LDS once instead of being
// re-read by each of the 4 branch blocks. Per-branch arithmetic (inject,
// bf16 round-trip, 256-lane reduction tree, norm) is kept identical to
// k_gr_scatter_norm_b_hc_bf16, so results are bit-exact.
__global__ void k_gr_scatter_norm_b_hc_bf16_f4(
    const float* __restrict__ w4, uint16_t* __restrict__ R,
    const float* __restrict__ y, const float* __restrict__ w,
    uint16_t* __restrict__ Rhat, int d, int branches, int P, float eps) {
  int t = blockIdx.x;
  if (t >= P) return;
  int lane = threadIdx.x;
  const float* yg = y + (size_t)t * d;
  __shared__ float sm[256];
  __shared__ float s_y[2560];
  __shared__ float inject;
  for (int c = lane; c < d; c += 256) s_y[c] = yg[c];
  __syncthreads();
  for (int b = 0; b < branches; b++) {
    uint16_t* Rg = R + ((size_t)t * branches + b) * d;
    const float* wg = w + (size_t)b * d;
    uint16_t* og = Rhat + ((size_t)t * branches + b) * d;
    if (lane == 0)
      inject = 2.f / (1.f + expf(-w4[(size_t)t * branches + b] * 0.25f));
    __syncthreads();
    float values[10];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 10; j++) {
      int c = lane + j * 256;
      float rounded = 0.f;
      if (c < d) {
        uint16_t r16 = f2bf(bf2f(Rg[c]) + inject * s_y[c]);
        Rg[c] = r16;
        rounded = bf2f(r16);
        ss += rounded * rounded;
      }
      values[j] = rounded;
    }
    sm[lane] = ss;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
      if (lane < off) sm[lane] += sm[lane + off];
      __syncthreads();
    }
    float inv = rsqrtf(sm[0] / d + eps);
#pragma unroll
    for (int j = 0; j < 10; j++) {
      int c = lane + j * 256;
      if (c < d) og[c] = f2bf(values[j] * inv * (1.f + wg[c]));
    }
    __syncthreads();
  }
}
// Fuse a hyper-connection residual update with the following grouped norm.
// The model uses d=2560 and 1024-thread blocks, so each lane retains at most
// three updated values across the reduction instead of reloading R.
template <bool Bf16 = false>
__global__ void k_gr_scatter_norm_b(const float* __restrict__ w4,
                                    float* __restrict__ R,
                                    const float* __restrict__ y,
                                    const float* __restrict__ w,
                                    float* __restrict__ Rhat, int d,
                                    int branches, int P, float eps,
                                    uint16_t* __restrict__ bf16 = nullptr) {
  int group = blockIdx.x;
  if (group >= P * branches) return;
  int t = group / branches;
  int b = group - t * branches;
  int lane = threadIdx.x;
  float* Rg = R + (size_t)group * d;
  const float* yg = y + (size_t)t * d;
  const float* wg = w + (size_t)b * d;
  float* og = Rhat + (size_t)group * d;
  __shared__ float sm[1024];
  __shared__ float inject;
  if (lane == 0)
    inject = 2.f / (1.f + expf(-w4[(size_t)t * branches + b] * 0.25f));
  __syncthreads();

  float values[3] = {0.f, 0.f, 0.f};
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 3; j++) {
    int c = lane + j * blockDim.x;
    if (c < d) {
      float value = Rg[c] + inject * yg[c];
      Rg[c] = value;
      values[j] = value;
      ss += value * value;
    }
  }
  sm[lane] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (lane < off) sm[lane] += sm[lane + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / d + eps);
#pragma unroll
  for (int j = 0; j < 3; j++) {
    int c = lane + j * blockDim.x;
    if (c < d) {
      float value = values[j] * inv * (1.f + wg[c]);
      og[c] = value;
      if (Bf16) bf16[(size_t)group * d + c] = f2bf(value);
    }
  }
}

// --- batched RoPE: grid.y = token row t (pos = t); rotation identical to k_rope
// --- RoPE pow table: g_rope_tab[i] = pow(theta, -2.0*i/rotary), filled on
// device with the same libm pow and the same argument expression k_rope_b
// used to compute inline -> per-element angles are bit-identical, the table
// just removes 33M redundant pow calls per prefill layer.
__device__ double g_rope_tab[64];
__global__ void k_rope_tab_init(double theta, int rotary) {
  int i = threadIdx.x;
  if (i < rotary / 2) g_rope_tab[i] = pow(theta, -2.0 * i / rotary);
}
static void ensure_rope_tab(double theta, int rotary) {
  static double cur_theta = -1;
  static int cur_rotary = -1;
  if (cur_theta != theta || cur_rotary != rotary) {
    k_rope_tab_init<<<1, 64, 0, g_str>>>(theta, rotary);
    cur_theta = theta;
    cur_rotary = rotary;
  }
}

__global__ void k_rope_b(float* __restrict__ vec, int nheads, int dh, int rotary,
                         double theta, int base = 0) {
  int pos = blockIdx.y;      // batch-local row
  int apos = base + pos;     // absolute position (rope angle)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // over nheads * rotary/2
  int half = rotary / 2;
  if (idx >= nheads * half) return;
  int h = idx / half, i = idx % half;
  double ang = apos * g_rope_tab[i];
  float cs = (float)cos(ang), sn = (float)sin(ang);
  float* p = vec + ((size_t)pos * nheads + h) * dh;
  float x0 = p[i], x1 = p[i + half];
  p[i] = x0 * cs - x1 * sn;
  p[i + half] = x0 * sn + x1 * cs;
  (void)theta;
}

// Per-(token, i) rope cos/sin table: one fp64 transcendental pair per (pos, i)
// instead of one per (head, pos, i) — 24x less fp64 work on gfx1151, and the
// consumers keep full thread parallelism. Values are bit-identical to the
// inline expressions in k_rope_b / the fused QSA prep kernels (same libm,
// same argument expression).
__global__ void k_rope_cs(float2* __restrict__ tab, int base, int P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= P * 32) return;
  double ang = (base + (idx >> 5)) * g_rope_tab[idx & 31];
  tab[idx] = make_float2((float)cos(ang), (float)sin(ang));
}

// M-RoPE debug dump: the 32 cos/sin pairs k_rope would use this decode step
// (same inline fp64 expression), for rope position *d_pos + *d_rdelta.
__global__ void k_rope_dump(float2* __restrict__ out, int* __restrict__ rpos,
                            const int* __restrict__ d_pos,
                            const int* __restrict__ d_rdelta, double theta) {
  int i = threadIdx.x;
  int pos = *d_pos + (d_rdelta ? *d_rdelta : 0);
  if (i == 0) *rpos = pos;
  if (i < 32) {
    double ang = pos * pow(theta, -2.0 * i / 64.0);
    out[i] = make_float2((float)cos(ang), (float)sin(ang));
  }
}

// M-RoPE slot interleave (mrope_section = [11, 11, 10], see
// reference/flash_next_config.json and tools/mrope_ref.py): freq index i
// takes its position from row H when i % 3 == 1 and i < 3*11 (1,4,...,31),
// from row W when i % 3 == 2 and i < 3*10 (2,5,...,29), else from row T.
static int mrope_slot_row(int i) {
  if (i % 3 == 1 && i < 33) return 1;
  if (i % 3 == 2 && i < 30) return 2;
  return 0;
}

// --- QSA flash prefill: full causal attention over P positions, fp32.
// Tiled: block = 128 threads (4 warps) x 16 q rows x one head; positions
// stream in 8-wide chunks. q and K chunks live in LDS (pitch 260 floats:
// float4-aligned, bank-spread; an LDS b128 wavefront costs ~4 cycles
// regardless of broadcast, measured on gfx1151); V is read straight from
// the global cache (L2) with contiguous 4-row-broadcast wavefronts.
// 24.4 KiB LDS per block -> two blocks per WGP so one block's staging and
// syncthreads overlap the other's compute.
// Online softmax per 8-score chunk; a row's scores live in the 8 lanes
// (r%4)*8..+7 of its warp; e crosses to the O accumulation via shfl.
// fp32 throughout.
// SharedV specializes sparse prefill for 12 query heads per KV head: 12 Q
// rows plus 8 K/V rows use 29120 bytes LDS. The fourth warp stages data but
// skips head arithmetic. Score, softmax, and V accumulation order are unchanged.
#define QSA_TR 16   // q rows per block
#define QSA_KT 8    // positions per chunk
#define QSA_LP 260  // LDS row pitch (floats, float4-aligned, bank-spread)
#define QSA_DEC_NSPLIT 32  // decode flash-decoding splits (grid.x = splits, y = kvh)
template <bool Sparse = false, bool SharedV = false, typename KVT = float,
          bool Split = false>
__global__ void k_qsa_flash(const float* __restrict__ q, const KVT* __restrict__ kc,
                            const KVT* __restrict__ vc, float* __restrict__ out, int P,
                            int dh, int hkv, int hq, const int* selected = nullptr,
                            int first = 0, const int* pos = nullptr,
                            float* pacc = nullptr, float* pml = nullptr,
                            int kbase = 0) {
  static_assert(!SharedV || Sparse, "shared V is a sparse-attention specialization");
  const int h = blockIdx.y;
  const int kvh = Sparse ? h : h / (hq / hkv);
  const int token = pos ? *pos : first + (int)blockIdx.x;
  // Dense decode tokens (< 2051) are handled here too: ntok = token+1 and
  // qsa_source_pos degrades to identity, so the split partials cover the
  // whole visible range (empty splits write m=-1e30, l=0 and merge as 0).
  const int* blocks =
      Sparse ? selected + (size_t)(Split ? 0 : blockIdx.x) * 512 : nullptr;
  const int ts = blockIdx.x * QSA_TR;   // first q row of this tile
  // Sparse: q/out rows are batch-local (token - kbase); token stays absolute
  // for ntok/source bookkeeping. Dense: rows local, keys/mask absolute (kbase).
  const int qrow = token - kbase;
  const int ntok = Sparse ? (token < 2051 ? token + 1 : 2048 + (token + 1) % 4)
                          : min(kbase + P, kbase + ts + QSA_TR);
  constexpr int QR = SharedV ? 12 : QSA_TR;
  __shared__ float qs[QR][QSA_LP];
  __shared__ float kt[QSA_KT][QSA_LP];
  __shared__ float vt[SharedV ? QSA_KT : 1][SharedV ? QSA_LP : 1];
  const int tid = threadIdx.x;
  // stage q tile (rows >= P read as zero; those rows never write out)
  for (int i4 = tid; i4 < QR * (dh / 4); i4 += 128) {
    int rl = i4 / (dh / 4), d4 = i4 % (dh / 4);
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (Sparse ? rl < hq / hkv : ts + rl < P)
      v = *(const float4*)(q + (Sparse ? (size_t)(pos ? 0 : qrow) * hq + kvh * (hq / hkv) + rl
                                      : (size_t)(ts + rl) * hq + h) * dh + d4 * 4);
    *(float4*)&qs[rl][d4 * 4] = v;
  }
  const int w = tid / warpSize, lane = tid % warpSize;
  const int rl = lane / QSA_KT;      // row within warp: rows 4w+rl
  const int pl = lane % QSA_KT;      // position within chunk (score phase)
  const int row = 4 * w + rl;
  const int rg = Sparse ? token : ts + row;
  float m_run = -1e30f, l_run = 0.f; // online softmax state for row rg
  float4 acc[8];                     // dims dg*4 + 32*f, dg = lane % 8
  for (int f = 0; f < 8; f++) acc[f] = make_float4(0.f, 0.f, 0.f, 0.f);
  const float scale = 1.f / sqrtf((float)dh);
  // K chunk staging slice: thread tid stages row tid/16, floats (tid%16)*16..
  const int spi = tid / 16, spf = (tid % 16) * 16;
  const KVT* kg = kc + (size_t)kvh * dh + spf;
  float4 st0, st1, st2, st3;
  if (!Sparse) {
    int source = min(spi, ntok - 1);
    if (Sparse) source = qsa_source_pos(source, token + 1, blocks);
    size_t off = (size_t)source * hkv * dh;
    st0 = kv_ld4(kg + off);
    st1 = kv_ld4(kg + off + 4);
    st2 = kv_ld4(kg + off + 8);
    st3 = kv_ld4(kg + off + 12);
    if (spi >= ntok) st0 = st1 = st2 = st3 = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  // Split (decode flash-decoding): this block covers a contiguous slice of
  // the logical position space; unnormalized partials go to pacc/pml and are
  // merged by k_qsa_flash_combine.
  int p_lo = 0, p_hi = ntok;
  if (Split) {
    int nchunks = (ntok + QSA_KT - 1) / QSA_KT;
    int cpb = (nchunks + QSA_DEC_NSPLIT - 1) / QSA_DEC_NSPLIT;
    p_lo = blockIdx.x * cpb * QSA_KT;
    p_hi = min(ntok, p_lo + cpb * QSA_KT);
  }
  for (int p0 = p_lo; p0 < p_hi; p0 += QSA_KT) {
    // make the prefetched chunk visible, then prefetch the next one
    if (Sparse) {
      // Sparse address bookkeeping plus the 16-float prefetch spills VGPRs.
      int source = qsa_source_pos(min(p0 + spi, ntok - 1), token + 1, blocks);
      const KVT* src = kg + (size_t)source * hkv * dh;
      *(float4*)&kt[spi][spf] = kv_ld4(src);
      *(float4*)&kt[spi][spf + 4] = kv_ld4(src + 4);
      *(float4*)&kt[spi][spf + 8] = kv_ld4(src + 8);
      *(float4*)&kt[spi][spf + 12] = kv_ld4(src + 12);
      if (SharedV) {
        const KVT* vs = vc + ((size_t)source * hkv + kvh) * dh + spf;
        *(float4*)&vt[spi][spf] = kv_ld4(vs);
        *(float4*)&vt[spi][spf + 4] = kv_ld4(vs + 4);
        *(float4*)&vt[spi][spf + 8] = kv_ld4(vs + 8);
        *(float4*)&vt[spi][spf + 12] = kv_ld4(vs + 12);
      }
    } else {
      *(float4*)&kt[spi][spf] = st0;
      *(float4*)&kt[spi][spf + 4] = st1;
      *(float4*)&kt[spi][spf + 8] = st2;
      *(float4*)&kt[spi][spf + 12] = st3;
    }
    __syncthreads();
    if (!Sparse && p0 + QSA_KT < ntok) {
      int pos = p0 + QSA_KT + spi;
      int source = min(pos, ntok - 1);
      if (Sparse) source = qsa_source_pos(source, token + 1, blocks);
      size_t off = (size_t)source * hkv * dh;
      st0 = kv_ld4(kg + off);
      st1 = kv_ld4(kg + off + 4);
      st2 = kv_ld4(kg + off + 8);
      st3 = kv_ld4(kg + off + 12);
      if (pos >= ntok) st0 = st1 = st2 = st3 = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    // scores: lane (rl, pl) owns (row 4w+rl, pos p0+pl); float4 reads are
    // bank-conflict-free (4 rl / 8 pl groups tile all 32 banks)
    if (!SharedV || row < QR) {
      const float4* q4 = (const float4*)&qs[4 * w + rl][0];
      const float4* k4 = (const float4*)&kt[pl][0];
      float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);  // 4 partials: break the
      for (int d4 = 0; d4 < dh / 4; d4++) {         // serial FMA chain (ILP)
        float4 qv = q4[d4];
        float4 kv = k4[d4];
        s4.x += qv.x * kv.x;
        s4.y += qv.y * kv.y;
        s4.z += qv.z * kv.z;
        s4.w += qv.w * kv.w;
      }
      float s = ((s4.x + s4.y) + (s4.z + s4.w)) * scale;
      bool masked = Sparse ? p0 + pl >= ntok : p0 + pl > kbase + rg;
      if (masked) s = -1e30f;
      // chunk max/sum over the 8 lanes of this row (width-8 shfl groups)
      float cm = s;
      for (int off = QSA_KT / 2; off > 0; off >>= 1)
        cm = fmaxf(cm, __shfl_down_sync(~0ull, cm, off, QSA_KT));
      cm = __shfl_sync(~0ull, cm, 0, QSA_KT);
      float mn = fmaxf(m_run, cm);
      float c = expf(m_run - mn);    // 1.0 when both are -1e30 (masked chunk)
      float e = expf(s - mn);
      if (masked) e = 0.f;
      float cl = e;
      for (int off = QSA_KT / 2; off > 0; off >>= 1)
        cl += __shfl_down_sync(~0ull, cl, off, QSA_KT);
      cl = __shfl_sync(~0ull, cl, 0, QSA_KT);
      l_run = l_run * c + cl;
      m_run = mn;
      // O accumulation: lane (rl, dg) owns dims dg*4 + 32*f. Rescale by c once
      // per chunk, then accumulate e_p * V_p; e_p comes from lane rl*8+p.
      const int dg = lane % QSA_KT;
      for (int f = 0; f < 8; f++) {
        acc[f].x *= c;
        acc[f].y *= c;
        acc[f].z *= c;
        acc[f].w *= c;
      }
      auto accumulate_v = [&](int p) {
        float ev = __shfl_sync(~0ull, e, rl * QSA_KT + p);
        // clamp the row: masked positions multiply by ev == 0 (their cache
        // rows may be stale/NaN or, in unit tests, out of bounds)
        if (SharedV) {
          const float4* v4 = (const float4*)&vt[p][dg * 4];
          for (int f = 0; f < 8; f++) {
            float4 vv = v4[f * 8];
            acc[f].x += ev * vv.x;
            acc[f].y += ev * vv.y;
            acc[f].z += ev * vv.z;
            acc[f].w += ev * vv.w;
          }
        } else {
          int pr = min(p0 + p, ntok - 1);
          if (Sparse) pr = qsa_source_pos(pr, token + 1, blocks);
          const KVT* vg = vc + ((size_t)pr * hkv + kvh) * dh + dg * 4;
          for (int f = 0; f < 8; f++) {
            float4 vv = kv_ld4(vg + f * 32);
            acc[f].x += ev * vv.x;
            acc[f].y += ev * vv.y;
            acc[f].z += ev * vv.z;
            acc[f].w += ev * vv.w;
          }
        }
      };
      if (Sparse) {
#pragma unroll 1
        for (int p = 0; p < QSA_KT; p++) accumulate_v(p);
      } else {
#pragma unroll
        for (int p = 0; p < QSA_KT; p++) accumulate_v(p);
      }
    }
    __syncthreads();
  }
  if (Split) {
    // unnormalized partial: acc is relative to this slice's m_run/l_run
    if (row < hq / hkv) {
      size_t base =
          ((size_t)h * QSA_DEC_NSPLIT + blockIdx.x) * (hq / hkv) + row;
      float* pa = pacc + base * dh + (lane % QSA_KT) * 4;
      for (int f = 0; f < 8; f++) *(float4*)(pa + 32 * f) = acc[f];
      if (lane % QSA_KT == 0) {
        pml[base * 2] = m_run;
        pml[base * 2 + 1] = l_run;
      }
    }
    return;
  }
  if (Sparse ? row < hq / hkv : rg < P) {
    float inv = 1.f / l_run;
    float* o = out + (Sparse ? (size_t)(pos ? 0 : qrow) * hq + kvh * (hq / hkv) + row
                             : (size_t)rg * hq + h) * dh + (lane % QSA_KT) * 4;
    for (int f = 0; f < 8; f++) {
      float4 v = acc[f];
      v.x *= inv;
      v.y *= inv;
      v.z *= inv;
      v.w *= inv;
      *(float4*)(o + 32 * f) = v;
    }
  }
}

// --- k_qsa_flash_combine: merge QSA_DEC_NSPLIT split partials (decode) ------
// Standard flash-decoding merge: out = sum_s exp(m_s - M) * acc_s / L with
// L = sum_s exp(m_s - M) * l_s. One block per (q row, kv head).
__global__ void k_qsa_flash_combine(const float* __restrict__ pml,
                                    const float* __restrict__ pacc,
                                    float* __restrict__ out, int qpg, int dh,
                                    const int* __restrict__ d_pos) {
  (void)d_pos;  // dense tokens now take this path too (no <2051 early-out)
  const int h = blockIdx.y, row = blockIdx.x;
  const float* pm = pml + ((size_t)h * QSA_DEC_NSPLIT * qpg + row) * 2;
  float M = -1e30f, L = 0.f;
  for (int s = 0; s < QSA_DEC_NSPLIT; s++) {
    float m = pm[(size_t)s * qpg * 2], l = pm[(size_t)s * qpg * 2 + 1];
    float mn = fmaxf(M, m);
    L = L * expf(M - mn) + l * expf(m - mn);
    M = mn;
  }
  float inv = 1.f / L;
  for (int d4 = threadIdx.x; d4 < dh / 4; d4 += blockDim.x) {
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < QSA_DEC_NSPLIT; s++) {
      float w = expf(pm[(size_t)s * qpg * 2] - M);
      float4 a = *(const float4*)(pacc +
                                  (((size_t)h * QSA_DEC_NSPLIT + s) * qpg + row) * dh +
                                  d4 * 4);
      o.x += w * a.x;
      o.y += w * a.y;
      o.z += w * a.z;
      o.w += w * a.w;
    }
    o.x *= inv;
    o.y *= inv;
    o.z *= inv;
    o.w *= inv;
    *(float4*)(out + ((size_t)h * qpg + row) * dh + d4 * 4) = o;  // token 0
  }
}

// --- k_qsa_flash_bf16: BF16 LDS variant of k_qsa_flash<true, true> ----------
// Used only by the BF16 KV mode (!qsa_kv_fp32) sparse prefill SharedV launch;
// the FP32 sparse path above is untouched. qs/kt/vt tiles are stored bf16
// (fp32 global -> f2bf RNE on stage) for ~15KB LDS and 4 blocks/CU (vs 2 at
// 29120B); QK uses fdot2_f32_bf16 with fp32 accumulators; AV stays fp32 FMA
// on bf16->fp32 expanded V. Row pitch QSA_LPB bf16: 264 (=528B) keeps the
// fp32 pitch's 4-bank row rotation (528 mod 128 = 16), so the 8-row kt access
// pattern stays conflict-free. Measured (tools/qsa_bench.cu): 32K sparse
// kernel 1179->459ms, max abs diff 2.8e-3 vs the fp32 staged kernel.
// Override with -DQSA_LPB=... for pitch experiments.
#ifndef QSA_LPB
#define QSA_LPB 264
#endif

__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  return (uint32_t)f2bf(a) | ((uint32_t)f2bf(b) << 16);
}

// stage 16 K/V elements from global into a bf16 LDS row slice. FP32 global
// is converted with f2bf RNE; BF16 global (the GDEC_QSA_KV_BF16 production
// layout) is a straight copy with no conversion work.
template <typename KVT>
__device__ __forceinline__ void stage16_bf16(uint16_t* dst, const KVT* src) {
  if constexpr (sizeof(KVT) == 4) {
    float4 a = *(const float4*)src, b = *(const float4*)(src + 4);
    float4 c = *(const float4*)(src + 8), d = *(const float4*)(src + 12);
    *(uint4*)dst = {pack_bf2(a.x, a.y), pack_bf2(a.z, a.w),
                    pack_bf2(b.x, b.y), pack_bf2(b.z, b.w)};
    *(uint4*)(dst + 8) = {pack_bf2(c.x, c.y), pack_bf2(c.z, c.w),
                          pack_bf2(d.x, d.y), pack_bf2(d.z, d.w)};
  } else {
    const uint4* s = (const uint4*)src;  // 16 bf16 = 32B
    *(uint4*)dst = s[0];
    *(uint4*)(dst + 8) = s[1];
  }
}

template <typename KVT = float>
__global__ void k_qsa_flash_bf16(const float* __restrict__ q,
                                 const KVT* __restrict__ kc,
                                 const KVT* __restrict__ vc,
                                 float* __restrict__ out, int P, int dh, int hkv,
                                 int hq, const int* __restrict__ selected,
                                 int first, int kbase = 0) {
  const int kvh = blockIdx.y;
  const int token = first + (int)blockIdx.x;
  const int qrow = token - kbase;  // batch-local q/out row; token stays absolute
  const int* blocks = selected + (size_t)blockIdx.x * 512;
  const int ntok = token < 2051 ? token + 1 : 2048 + (token + 1) % 4;
  constexpr int QR = 12;
  __shared__ __align__(16) uint16_t qs[QR][QSA_LPB];
  __shared__ __align__(16) uint16_t kt[QSA_KT][QSA_LPB];
  __shared__ __align__(16) uint16_t vt[QSA_KT][QSA_LPB];
  const int tid = threadIdx.x;
  // stage q tile (12 rows; fp32 global -> bf16 LDS)
  for (int i4 = tid; i4 < QR * (dh / 4); i4 += 128) {
    int r = i4 / (dh / 4), d4 = i4 % (dh / 4);
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < hq / hkv)
      v = *(const float4*)(q + ((size_t)qrow * hq + kvh * (hq / hkv) + r) * dh + d4 * 4);
    uint2 pk = {pack_bf2(v.x, v.y), pack_bf2(v.z, v.w)};
    *(uint2*)&qs[r][d4 * 4] = pk;
  }
  const int w = tid / warpSize, lane = tid % warpSize;
  const int rl = lane / QSA_KT;      // row within warp: rows 4w+rl
  const int pl = lane % QSA_KT;      // position within chunk (score phase)
  const int row = 4 * w + rl;
  float m_run = -1e30f, l_run = 0.f; // online softmax state
  float4 acc[8];                     // dims dg*4 + 32*f, dg = lane % 8
  for (int f = 0; f < 8; f++) acc[f] = make_float4(0.f, 0.f, 0.f, 0.f);
  const float scale = 1.f / sqrtf((float)dh);
  // K/V chunk staging slice: thread tid stages row tid/16, elems (tid%16)*16..
  const int spi = tid / 16, spf = (tid % 16) * 16;
  const KVT* kg = kc + (size_t)kvh * dh + spf;
  for (int p0 = 0; p0 < ntok; p0 += QSA_KT) {
    // stage K and V chunks into the bf16 LDS tiles
    int source = qsa_source_pos(min(p0 + spi, ntok - 1), token + 1, blocks);
    stage16_bf16(&kt[spi][spf], kg + (size_t)source * hkv * dh);
    stage16_bf16(&vt[spi][spf], vc + ((size_t)source * hkv + kvh) * dh + spf);
    __syncthreads();
    if (row < QR) {
      // scores: lane (rl, pl) owns (row 4w+rl, pos p0+pl); uint4 (8 bf16)
      // reads keep the fp32 float4 pattern's conflict-free bank layout
      const uint4* q8 = (const uint4*)&qs[row][0];
      const uint4* k8 = (const uint4*)&kt[pl][0];
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;  // 4 partials (ILP)
      for (int d8 = 0; d8 < dh / 8; d8++) {
        uint4 qv = q8[d8];
        uint4 kv = k8[d8];
        const __hip_bfloat162* qp = (const __hip_bfloat162*)&qv;
        const __hip_bfloat162* kp = (const __hip_bfloat162*)&kv;
        s0 = __builtin_amdgcn_fdot2_f32_bf16(qp[0], kp[0], s0, false);
        s1 = __builtin_amdgcn_fdot2_f32_bf16(qp[1], kp[1], s1, false);
        s2 = __builtin_amdgcn_fdot2_f32_bf16(qp[2], kp[2], s2, false);
        s3 = __builtin_amdgcn_fdot2_f32_bf16(qp[3], kp[3], s3, false);
      }
      float s = ((s0 + s1) + (s2 + s3)) * scale;
      bool masked = p0 + pl >= ntok;
      if (masked) s = -1e30f;
      // chunk max/sum over the 8 lanes of this row (width-8 shfl groups)
      float cm = s;
      for (int off = QSA_KT / 2; off > 0; off >>= 1)
        cm = fmaxf(cm, __shfl_down_sync(~0ull, cm, off, QSA_KT));
      cm = __shfl_sync(~0ull, cm, 0, QSA_KT);
      float mn = fmaxf(m_run, cm);
      float c2 = expf(m_run - mn);   // 1.0 when both are -1e30 (masked chunk)
      float e = expf(s - mn);
      if (masked) e = 0.f;
      float cl = e;
      for (int off = QSA_KT / 2; off > 0; off >>= 1)
        cl += __shfl_down_sync(~0ull, cl, off, QSA_KT);
      cl = __shfl_sync(~0ull, cl, 0, QSA_KT);
      l_run = l_run * c2 + cl;
      m_run = mn;
      // O accumulation: lane (rl, dg) owns dims dg*4 + 32*f, fp32 FMA on
      // bf16->fp32 expanded V (ev == 0 kills stale masked rows, as before)
      const int dg = lane % QSA_KT;
      for (int f = 0; f < 8; f++) {
        acc[f].x *= c2;
        acc[f].y *= c2;
        acc[f].z *= c2;
        acc[f].w *= c2;
      }
      auto accumulate_v = [&](int p) {
        float ev = __shfl_sync(~0ull, e, rl * QSA_KT + p);
        const uint16_t* vrow = &vt[p][dg * 4];
        for (int f = 0; f < 8; f++) {
          float4 vv = kv_ld4(vrow + f * 32);
          acc[f].x += ev * vv.x;
          acc[f].y += ev * vv.y;
          acc[f].z += ev * vv.z;
          acc[f].w += ev * vv.w;
        }
      };
#pragma unroll 1
      for (int p = 0; p < QSA_KT; p++) accumulate_v(p);
    }
    __syncthreads();
  }
  if (row < hq / hkv) {
    float inv = 1.f / l_run;
    float* o = out + ((size_t)qrow * hq + kvh * (hq / hkv) + row) * dh +
               (lane % QSA_KT) * 4;
    for (int f = 0; f < 8; f++) {
      float4 v = acc[f];
      v.x *= inv;
      v.y *= inv;
      v.z *= inv;
      v.w *= inv;
      *(float4*)(o + 32 * f) = v;
    }
  }
}

// --- k_qsa_wmma: WMMA bf16 sparse flash attention (256 thr = 8 warps) ---------
// Opt-in (GDEC_QSA_WMMA=1, BF16 KV mode only): replaces k_qsa_flash_bf16 for
// the sparse tail. One block per (token, kvh); keys stream in 128-key block
// iterations and warp w scores its own 16-key stripe (S = Q·Kᵀ WMMA, dual
// accumulators for ILP). Softmax uses a BLOCK-wide online maximum (two-stage
// LDS reduction), so every warp owns a fixed 32-dim slice of O and the
// accumulators never need a cross-warp merge; l is reduced once at the end.
// P is quantized bf16 (RNE) into a double-buffered LDS tile and read back as
// the PV A fragment — the WMMA C layout cannot feed back as A, and an LDS
// round-trip is the standard fix. V B fragments come from either the
// transposed cache (TransposedV, vct) or a row-major gather + in-register
// 16x16 transpose (tr16x16_bf16, no LDS).
// gfx1151 WMMA fragment layout (measured with local probe kernels on this
// GPU — hardware behavior, not implementation choice):
//   A(r,k): r<8 -> lane 2r elem k; r>=8 -> lane 17+2(r-8) elem k (rest ignored)
//   B(k,c): lane c elem k AND lane 16+c elem k (duplicate)
//   C(r,c): lane c + 16*(r>=8), elem r%8
// Numerics: bf16 P + WMMA accumulation (not bitwise fp32); measured maxabs
// 1.6e-4 vs the bf16-P fp32 reference path.
using qw_shortx16 = __attribute__((ext_vector_type(16))) short;
using qw_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ qw_shortx16 qw_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  qw_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
__device__ __forceinline__ qw_shortx16 qw_zero16() {
  short tmp[16] = {};
  qw_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
// In-register 16x16 bf16 transpose within each 16-lane half (gfx1151,
// wave32). Input: lane l (of the half) holds u32 a[8], a[i] =
// pack(M[l][2i], M[l][2i+1]) — one matrix row, 32B. Output: lane c holds
// u32 a[j] = pack(M[2j][c], M[2j+1][c]) — one matrix column, i.e. a WMMA
// B fragment tile. 20 shfls, no LDS.
// Derivation (CPU-sim verified in tools/qsa_wmma_rm.cu): lane l = 2q+g.
// Even lanes hold rows of the 8x8 u32 matrix E[s][i], odd lanes O[s][i].
// Stage A: 8x8 u32 transpose per sub-group (masks 2,4,8); butterfly swaps
// slot j on lane q with slot j^p on lane q^p, so the sent value is a[j^p]
// and each step snapshots into t[] (partner may overwrite a slot before
// our loop iteration reads it). Stage B (mask 1): adjacent lanes (2i,
// 2i+1) swap u32s and pick halves by g. Every lane executes every shfl
// (convergence), selecting afterwards.
__device__ __forceinline__ void tr16x16_bf16(unsigned a[8]) {
  const unsigned l = threadIdx.x & 15;
  const unsigned q = l >> 1;
#pragma unroll
  for (int m = 2; m <= 8; m <<= 1) {
    const unsigned p = m >> 1;
    unsigned t[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) t[j] = a[j];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned peer = __shfl_xor_sync(~0ull, t[j ^ p], m);
      if (((unsigned)j & p) != (q & p)) a[j] = peer;
    }
  }
  const unsigned g = l & 1;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const unsigned peer = __shfl_xor_sync(~0ull, a[j], 1);
    a[j] = g ? ((peer >> 16) | (a[j] & 0xFFFF0000u))
             : ((a[j] & 0xFFFFu) | (peer << 16));
  }
}
#define QW_DH 256
#define QW_QR 12
#define QW_KTILE 16
#define QW_KITER 128
#define QW_NITER 17   // ceil(2051 / 128)
#define QW_NKSTEP (QW_KITER / QW_KTILE)
template <bool TransposedV = false>
__global__ void __launch_bounds__(256)
    k_qsa_wmma(const float* __restrict__ q, const uint16_t* __restrict__ kc,
               const uint16_t* __restrict__ vc, const uint16_t* __restrict__ vct,
               float* __restrict__ out, int P,
               int dh, int hkv, int hq, const int* __restrict__ selected,
               int first, int kbase = 0) {
  (void)P;
  const int token = first + (int)blockIdx.x, kvh = blockIdx.y;
  const int qrow = token - kbase;  // batch-local q/out row; token stays absolute
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __shared__ uint16_t Qs[16][QW_DH + 8];   // 8704B
  __shared__ uint16_t Pbuf[2][16][136];    // [phase][qrow][key], double-buffered
  __shared__ float Mx[8][16];
  __shared__ float Lf[8][16];
  // Neither V path uses an LDS tile: TransposedV gathers from the
  // transposed cache, the row-major path transposes in registers.
  // stage q tile (fp32 global -> bf16 RNE LDS; rows >= 12 stay zero)
  {
    const float* qb = q + ((size_t)qrow * hq + kvh * (hq / hkv)) * dh;
    for (int i = threadIdx.x; i < 16 * (QW_DH / 8); i += 256) {
      int r = i / (QW_DH / 8), c8 = i % (QW_DH / 8);
      uint4 v = {0, 0, 0, 0};
      if (r < hq / hkv) {
        const float4* q4 = (const float4*)(qb + r * dh + c8 * 8);
        float4 a = q4[0], b = q4[1];
        v.x = pack_bf2(a.x, a.y); v.y = pack_bf2(a.z, a.w);
        v.z = pack_bf2(b.x, b.y); v.w = pack_bf2(b.z, b.w);
      }
      *(uint4*)&Qs[r][c8 * 8] = v;
    }
  }
  __syncthreads();
  const int* blocks = selected + (size_t)blockIdx.x * 512;
  const int ntok = 2048 + (token + 1) % 4;
  const long tail0 = ((token + 1) / 4) * 4L;   // dense tail (visible = token+1)
  float m_run[8], l_run[8];   // per-elem qrow state (row = lane<16 ? e : 8+e)
#pragma unroll
  for (int e = 0; e < 8; ++e) { m_run[e] = -1e30f; l_run[e] = 0.f; }
  float acc[2][8];            // 2 d-tiles; elem e -> qrow (lane<16 ? e : 8+e),
                              // lane c16 -> dim w*32 + dt*16 + c16
#pragma unroll
  for (int dt = 0; dt < 2; ++dt)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[dt][e] = 0.f;
  const float scale = 1.f / sqrtf((float)dh);
  int qrowA = -1;             // A=Q role lane -> qrow
  if (lane < 16 && !(lane & 1)) qrowA = lane / 2;
  else if (lane >= 17 && (lane & 1)) qrowA = 8 + (lane - 17) / 2;
  const int keylane = lane & 15;
#pragma unroll 1
  for (int it = 0; it < QW_NITER; ++it) {
    const int kbase = it * QW_KITER + w * QW_KTILE;
    // ---------------- QK: S = Q·Kᵀ ----------------
    qw_floatx8 s0 = {0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0;
    int gkey = kbase + keylane;
    long src = gkey < ntok ? (gkey < 2048 ? 4L * blocks[gkey / 4] + gkey % 4
                                          : tail0 + gkey - 2048)
                           : 0;
    const uint16_t* kp = kc + ((size_t)src * hkv + kvh) * dh;
#pragma unroll
    for (int k4 = 0; k4 < 4; ++k4) {
      qw_shortx16 a[4], b[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        a[j] = qrowA >= 0 ? qw_ld16(&Qs[qrowA][(k4 * 4 + j) * 16]) : qw_zero16();
        b[j] = qw_ld16(kp + (k4 * 4 + j) * 16);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if ((k4 * 4 + j) & 1)
          s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[j], b[j], s1);
        else
          s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[j], b[j], s0);
      }
    }
    qw_floatx8 s;
#pragma unroll
    for (int e = 0; e < 8; ++e) s[e] = (s0[e] + s1[e]) * scale;
    if (gkey >= ntok)
#pragma unroll
      for (int e = 0; e < 8; ++e) s[e] = -1e30f;
    // warp-level row max via butterflies (halves stay separate: xor < 16)
    float pm[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float m = s[e];
      m = fmaxf(m, __shfl_xor_sync(~0ull, m, 1));
      m = fmaxf(m, __shfl_xor_sync(~0ull, m, 2));
      m = fmaxf(m, __shfl_xor_sync(~0ull, m, 4));
      m = fmaxf(m, __shfl_xor_sync(~0ull, m, 8));
      pm[e] = m;
    }
    // row j (j<8): lane j elem j; row 8+j: lane 16+j elem j
    if (lane < 8) Mx[w][lane] = pm[lane];
    else if (lane >= 16 && lane < 24) Mx[w][8 + lane - 16] = pm[lane - 16];
    __syncthreads();
    // single-barrier block max: lane l<16 reduces row l, broadcast via shfl
    float mrow = -1e30f;
    if (lane < 16) {
      float m = Mx[0][lane];
#pragma unroll
      for (int ww = 1; ww < 8; ++ww) m = fmaxf(m, Mx[ww][lane]);
      mrow = m;
    }
    // per-elem: p = exp(s - mn), l partial, cf rescale
    uint16_t pb[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      float mn = fmaxf(m_run[e], __shfl_sync(~0ull, mrow, r));
      float p = (s[e] <= -1e29f) ? 0.f : __expf(s[e] - mn);
      pb[e] = f2bf(p);
      float cf = __expf(m_run[e] - mn);
      float ls = p;
      ls += __shfl_xor_sync(~0ull, ls, 1);
      ls += __shfl_xor_sync(~0ull, ls, 2);
      ls += __shfl_xor_sync(~0ull, ls, 4);
      ls += __shfl_xor_sync(~0ull, ls, 8);
      l_run[e] = l_run[e] * cf + ls;
      m_run[e] = mn;
      acc[0][e] *= cf;
      acc[1][e] *= cf;
    }
    // ---------------- P to LDS transposed ----------------
    {
      int wkey = w * QW_KTILE + keylane;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int r = (lane < 16) ? e : 8 + e;
        Pbuf[it & 1][r][wkey] = pb[e];
      }
    }
    __syncthreads();
    // ---------------- PV: warp w owns dims [w*32, +32) ----------------
    int pr = (lane < 16) ? lane / 2 : 8 + (lane - 17) / 2;
    bool pact = (lane < 16 && !(lane & 1)) || (lane >= 17 && (lane & 1));
#pragma unroll 1
    for (int ks = 0; ks < QW_NKSTEP; ++ks) {
      qw_shortx16 ap = pact ? qw_ld16(&Pbuf[it & 1][pr][ks * 16]) : qw_zero16();
      // B = V frag: lane c elem k = V[key k][dim dt*16 + c16]. The
      // transposed cache path loads four slots per block in one 64-bit read
      // (base pointers per 4-key block, all 8 loads issued before the 2
      // wmmas to hide gather latency); the row-major path gathers one
      // 16-key x 16-dim tile per half-warp (same wide global pattern as
      // the old LDS path) and transposes it in registers (tr16x16_bf16),
      // then swaps tiles between halves — no LDS, no vct.
      if constexpr (TransposedV) {
        const uint16_t* vb[4];
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
          int vk = it * QW_KITER + ks * QW_KTILE + blk * 4;
          long vs = vk < ntok ? (vk < 2048 ? 4L * blocks[vk / 4] + vk % 4
                                           : tail0 + vk - 2048)
                              : 0;
          vb[blk] = vct + ((((size_t)vs / 4) * hkv + kvh) * dh + w * 32) * 4;
        }
        unsigned long long pk[2][4];
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          int dim = dt * 16 + (lane & 15);
#pragma unroll
          for (int blk = 0; blk < 4; ++blk)
            pk[dt][blk] = *(const unsigned long long*)(vb[blk] + dim * 4);
        }
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          short btmp[16];
#pragma unroll
          for (int blk = 0; blk < 4; ++blk)
#pragma unroll
            for (int slot = 0; slot < 4; ++slot)
              btmp[blk * 4 + slot] = (short)(pk[dt][blk] >> (slot * 16));
          qw_shortx16 bv;
          memcpy(&bv, btmp, 32);
          qw_floatx8 o;
#pragma unroll
          for (int e = 0; e < 8; ++e) o[e] = acc[dt][e];
          o = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ap, bv, o);
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[dt][e] = o[e];
        }
      } else {
        // lane&15 = key, lane>>4 = d-tile: 2 x uint4 = 16 dims per lane.
        int vk = it * QW_KITER + ks * QW_KTILE + (lane & 15);
        long vs = vk < ntok ? (vk < 2048 ? 4L * blocks[vk / 4] + vk % 4
                                         : tail0 + vk - 2048)
                            : 0;
        const uint16_t* vp =
            vc + ((size_t)vs * hkv + kvh) * dh + w * 32 + (lane >> 4) * 16;
        unsigned va[8];
        *(uint4*)&va[0] = *(const uint4*)vp;
        *(uint4*)&va[4] = *(const uint4*)(vp + 8);
        tr16x16_bf16(va);  // half (lane>>4) now holds B cols of d-tile (lane>>4)
        unsigned vo[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) vo[j] = __shfl_xor_sync(~0ull, va[j], 16);
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
          unsigned bs[8];
#pragma unroll
          for (int j = 0; j < 8; ++j)
            bs[j] = ((unsigned)(lane >> 4) == (unsigned)dt) ? va[j] : vo[j];
          qw_shortx16 bv;
          memcpy(&bv, bs, 32);
          qw_floatx8 o;
#pragma unroll
          for (int e = 0; e < 8; ++e) o[e] = acc[dt][e];
          o = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ap, bv, o);
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[dt][e] = o[e];
        }
      }
    }
    // no reuse barrier: double-buffered Pbuf + next iter's max sync is enough
  }
  // ---------------- final: block-wide l sum + write ----------------
  if (lane < 8) Lf[w][lane] = l_run[lane];
  else if (lane >= 16 && lane < 24) Lf[w][8 + lane - 16] = l_run[lane - 16];
  __syncthreads();
  float lfin[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int r = (lane < 16) ? e : 8 + e;
    float t = 0;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) t += Lf[ww][r];
    lfin[e] = 1.f / t;
  }
  int c16 = lane & 15;
  float* ob = out + ((size_t)qrow * hq + kvh * (hq / hkv)) * dh;
#pragma unroll
  for (int dt = 0; dt < 2; ++dt)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      if (r < hq / hkv) ob[(size_t)r * dh + w * 32 + dt * 16 + c16] = acc[dt][e] * lfin[e];
    }
}

// --- k_qsa_wmma6: dim-split WMMA sparse flash attention (128 thr = 4 warps) --
// Opt-in (GDEC_QSA_WMMA6=1, on top of the transposed-V BF16 path). Same data
// layout and masking as k_qsa_wmma, but warp w owns dims [w*64, +64) for BOTH
// QK and PV (no key/dim role switch): Q lives in registers as four resident
// bf16 A fragments (fp32 global -> RNE on load, rows >= 12 zero). Per 16-key
// group the four warps' QK partial S fragments are summed through a
// double-buffered LDS area in fixed w0->w3 order; every warp then runs the
// full online softmax redundantly (warp-local butterflies only — the warp
// already sees all 16 keys — so no block-wide max/l reductions). P is
// quantized bf16 RNE into a warp-private LDS tile and read back as the PV A
// fragment; PV B fragments come straight from the transposed V cache (four
// 16-dim C tiles per warp, four 4-key slots per 64-bit load). Fragment
// layouts as documented on k_qsa_wmma.
__global__ void __launch_bounds__(128)
    k_qsa_wmma6(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                const uint16_t* __restrict__ vc, const uint16_t* __restrict__ vct,
                float* __restrict__ out, int P,
                int dh, int hkv, int hq, const int* __restrict__ selected,
                int first, int kbase = 0) {
  (void)P;
  (void)vc;
  const int token = first + (int)blockIdx.x, kvh = blockIdx.y;
  const int qrow = token - kbase;  // batch-local q/out row; token stays absolute
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  __shared__ int Sel[512];              // selected block table, 2KB
  __shared__ float Sbuf[2][4][16][17];  // [phase][warp][qrow][key], 8.7KB
  __shared__ uint16_t Pbuf[4][16][24];  // [warp][qrow][key], 3KB
  const int* blocks = selected + (size_t)blockIdx.x * 512;
  ((int4*)Sel)[threadIdx.x] = ((const int4*)blocks)[threadIdx.x];
  // Resident Q: A-fragment lanes (even 0-14 / odd 17-31) hold q row
  // lane/2 / 8+(lane-17)/2, dims [w*64, +64) as four 16-dim fragments.
  int qrowA = -1;
  if (lane < 16 && !(lane & 1)) qrowA = lane / 2;
  else if (lane >= 17 && (lane & 1)) qrowA = 8 + (lane - 17) / 2;
  qw_shortx16 qa[4];
  {
    const float* qb = q + ((size_t)qrow * hq + kvh * (hq / hkv)) * dh;
#pragma unroll
    for (int ss = 0; ss < 4; ++ss) {
      if (qrowA >= 0 && qrowA < hq / hkv) {
        const float4* q4 =
            (const float4*)(qb + (size_t)qrowA * dh + w * 64 + ss * 16);
        float4 a = q4[0], b = q4[1], c = q4[2], d = q4[3];
        uint32_t pk[8] = {pack_bf2(a.x, a.y), pack_bf2(a.z, a.w),
                          pack_bf2(b.x, b.y), pack_bf2(b.z, b.w),
                          pack_bf2(c.x, c.y), pack_bf2(c.z, c.w),
                          pack_bf2(d.x, d.y), pack_bf2(d.z, d.w)};
        memcpy(&qa[ss], pk, 32);
      } else {
        qa[ss] = qw_zero16();
      }
    }
  }
  __syncthreads();  // Sel visible block-wide
  const int ntok = 2048 + (token + 1) % 4;
  const long tail0 = ((token + 1) / 4) * 4L;  // dense tail (visible = token+1)
  const float scale = 1.f / sqrtf((float)dh);
  const int keylane = lane & 15;
  float m_run[8], l_run[8];  // per-elem qrow state (row = lane<16 ? e : 8+e)
#pragma unroll
  for (int e = 0; e < 8; ++e) { m_run[e] = -1e30f; l_run[e] = 0.f; }
  float acc[4][8];           // 4 d-tiles; lane c16 -> dim w*64 + dt*16 + c16
#pragma unroll
  for (int dt = 0; dt < 4; ++dt)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[dt][e] = 0.f;
  const int ngroups = (ntok + 15) >> 4;
#pragma unroll 1
  for (int g = 0; g < ngroups; ++g) {
    // ---------------- QK: partial S over this warp's 64 dims ----------------
    const int gkey = g * 16 + keylane;
    long src = gkey < ntok ? (gkey < 2048 ? 4L * Sel[gkey >> 2] + (gkey & 3)
                                          : tail0 + gkey - 2048)
                           : 0;
    const uint16_t* kp = kc + ((size_t)src * hkv + kvh) * dh + w * 64;
    qw_floatx8 s = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int ss = 0; ss < 4; ++ss) {
      qw_shortx16 b = qw_ld16(kp + ss * 16);
      s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qa[ss], b, s);
    }
    // ---------------- cross-warp S reduction, fixed w0 -> w3 ----------------
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      Sbuf[g & 1][w][r][keylane] = s[e];
    }
    // Double-buffered: the single barrier per group also orders next group's
    // writes against this group's reads (they land in the other phase).
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      float t = Sbuf[g & 1][0][r][keylane] + Sbuf[g & 1][1][r][keylane];
      t += Sbuf[g & 1][2][r][keylane];
      t += Sbuf[g & 1][3][r][keylane];
      s[e] = t * scale;
    }
    if (gkey >= ntok)
#pragma unroll
      for (int e = 0; e < 8; ++e) s[e] = -1e30f;
    // ---------------- online softmax (warp-redundant, identical) -----------
    uint16_t pb[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float pm = s[e];
      pm = fmaxf(pm, __shfl_xor_sync(~0ull, pm, 1));
      pm = fmaxf(pm, __shfl_xor_sync(~0ull, pm, 2));
      pm = fmaxf(pm, __shfl_xor_sync(~0ull, pm, 4));
      pm = fmaxf(pm, __shfl_xor_sync(~0ull, pm, 8));
      float mn = fmaxf(m_run[e], pm);
      float p = (s[e] <= -1e29f) ? 0.f : __expf(s[e] - mn);
      pb[e] = f2bf(p);
      float cf = __expf(m_run[e] - mn);
      float ls = p;
      ls += __shfl_xor_sync(~0ull, ls, 1);
      ls += __shfl_xor_sync(~0ull, ls, 2);
      ls += __shfl_xor_sync(~0ull, ls, 4);
      ls += __shfl_xor_sync(~0ull, ls, 8);
      float lc = l_run[e] * cf;
      asm volatile("" : "+v"(lc));  // keep the rescale out of fma contraction
      l_run[e] = lc + ls;
      m_run[e] = mn;
#pragma unroll
      for (int dt = 0; dt < 4; ++dt) acc[dt][e] *= cf;
    }
    // ---------------- P to warp-private LDS, back as PV A fragment ---------
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      Pbuf[w][r][keylane] = pb[e];
    }
    __syncwarp();
    int pr = (lane < 16) ? lane / 2 : 8 + (lane - 17) / 2;
    bool pact = (lane < 16 && !(lane & 1)) || (lane >= 17 && (lane & 1));
    qw_shortx16 ap = pact ? qw_ld16(&Pbuf[w][pr][0]) : qw_zero16();
    // ---------------- PV: B from the transposed V cache, 4 C tiles ---------
    const uint16_t* vb[4];
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
      int vk = g * 16 + blk * 4;  // 4-aligned: slots 0-3 of one vct block
      long vs = vk < ntok ? (vk < 2048 ? 4L * Sel[vk >> 2] + (vk & 3)
                                       : tail0 + vk - 2048)
                          : 0;
      vb[blk] = vct + ((((size_t)vs >> 2) * hkv + kvh) * dh + w * 64) * 4;
    }
    unsigned long long pk[4][4];
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      int dim = dt * 16 + keylane;
#pragma unroll
      for (int blk = 0; blk < 4; ++blk)
        pk[dt][blk] = *(const unsigned long long*)(vb[blk] + dim * 4);
    }
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      short btmp[16];
#pragma unroll
      for (int blk = 0; blk < 4; ++blk)
#pragma unroll
        for (int slot = 0; slot < 4; ++slot)
          btmp[blk * 4 + slot] = (short)(pk[dt][blk] >> (slot * 16));
      qw_shortx16 bv;
      memcpy(&bv, btmp, 32);
      qw_floatx8 o;
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = acc[dt][e];
      o = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ap, bv, o);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[dt][e] = o[e];
    }
  }
  // ---------------- final: warp-local l, each warp writes its 64 dims ------
  float lfin[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) lfin[e] = 1.f / l_run[e];
  float* ob = out + ((size_t)qrow * hq + kvh * (hq / hkv)) * dh;
#pragma unroll
  for (int dt = 0; dt < 4; ++dt)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      int r = (lane < 16) ? e : 8 + e;
      if (r < hq / hkv) ob[(size_t)r * dh + w * 64 + dt * 16 + keylane] = acc[dt][e] * lfin[e];
    }
}

// forward decls (defined below main model code)
template <bool Rope = false>
__global__ void k_qsa_qsplit(const float* __restrict__ qg, const float* __restrict__ qnw,
                             float* __restrict__ qs, float* __restrict__ gs, int dh,
                             float eps, int rope_base = 0,
                             const float2* __restrict__ ropecs = nullptr);
template <bool Bf16 = false>
__global__ void k_qsa_kprep(const float* __restrict__ kb, const float* __restrict__ knw,
                            float* __restrict__ kout, uint16_t* __restrict__ kc,
                            float eps, int rope_base,
                            const float2* __restrict__ ropecs);
__global__ void k_axpy_sg(float* __restrict__ acc, const float* __restrict__ y,
                          const float* __restrict__ sg, int n, int total);

// [gemm-wmma-begin]
// --- k_gemm_wmma: self-written bf16 WMMA dense GEMM (Phase 3g) --------------
// Y[P,N] f32 = X[P,K]bf16 * W[N,K]bf16^T, all row-major (ld K, K, N), fp32
// accum. WMMA "A" operand = X rows (K-contig), "B" operand = W rows (K-contig)
// — a B fragment for output column n is exactly W[n][k0..k0+15], so no
// transpose exists anywhere; the epilogue walks lane-contiguous n (coalesced).
// Fragment layouts are the hardware ones documented above k_qsa_wmma.
// Block tile BM(tokens) x BP(weights), NT threads = (NT/32) warps as MW x PW,
// warp owns WM x WP 16x16 frags, K stepped by KST, LDS double buffer:
//   - X tile row-major, row stride KST+8; A-fragment lanes use a dummy-row
//     assignment so every 8-lane LDS phase touches 8 distinct banks
//     (probe-calibrated conflict-free; naive clamp-to-row-0 costs 1 extra
//     phase per ds_load_b128).
//   - W tile grouped in 8-row regions, region stride 8*(KST+8)+8: B-fragment
//     loads are one ds_load_b128 each, broadcast-friendly, conflict-free.
//   - Unconditional global->reg->LDS staging (tail K-step index clamped to
//     ksteps-1, recomputed into a dead buffer): conditional staging makes the
//     allocator spill the staging registers to scratch, exposing full DRAM
//     latency in the main loop (the single biggest perf bug found in the
//     prototype, tools/wmma_gemm_proto.cu).
//   - gwmma_sync(): __syncthreads() also emits buffer_gl0_inv on gfx11,
//     draining the memory pipeline every K-step; inputs are read-only, so
//     only the lgkmcnt/vscnt drains are needed.
//   - Grouped-M CTA swizzle (gm) for L2 reuse.
// P tail: staging rows clamp to row P-1 (duplicate compute, discarded) and
// the epilogue is predicated, so any P works. Hard requirements (host-
// checked, like the MoE kernels below): N % BP == 0, K % KST == 0 — no
// runtime guards in the unrolled loops (they defeat unrolling and spill the
// accumulators to scratch).
// Measured on gfx1151 (proto, median of 20, vs hipBLASLt ~23-32 TFLOPS):
// N=2560 K=6144 ~24.5 TFLOPS (d3 cfg 512t w2x8 f4x2, gm=16);
// N=6144/10240/12288 K=2560 ~37.4-37.9 TFLOPS (d9 cfg 256t w2x4 f4x4, gm=4).
__device__ __forceinline__ void gwmma_sync() {
  __asm__ volatile(
      "s_waitcnt lgkmcnt(0)\n\t"
      "s_waitcnt_vscnt null, 0x0\n\t"
      "s_barrier" ::
          : "memory");
}

template <int BM, int BP, int MW, int PW, int WM, int WP, int KST, int NT>
__global__ void __launch_bounds__(NT)
    k_gemm_wmma(const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
                float* __restrict__ Y, int P, int K, int N, int gm) {
  constexpr int LSA = KST + 8;        // LDS row stride (elems)
  constexpr int BREG = 8 * LSA + 8;   // W 8-row region stride (elems)
  constexpr int NW = NT / 32;         // warps per block
  constexpr int AN = BM * (KST / 8) / NT;  // uint4 X loads per thread
  constexpr int BN = BP * (KST / 8) / NT;  // uint4 W loads per thread
  static_assert(MW * PW == NW, "warp grid mismatch");
  static_assert(BM == MW * WM * 16 && BP == PW * WP * 16, "tile mismatch");
  static_assert(BM * (KST / 8) % NT == 0 && BP * (KST / 8) % NT == 0,
                "staging mismatch");
  __shared__ uint16_t As[2][BM * LSA];
  __shared__ uint16_t Bs[2][(BP / 8) * BREG];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  // grouped swizzle for L2/MALL reuse (group along token blocks)
  const int num_m = (P + BM - 1) / BM, num_n = N / BP;
  const int bid = blockIdx.x;
  const int per_group = gm * num_n;
  const int gid = bid / per_group, rem = bid % per_group;
  const int first_m = gid * gm;
  const int gs = min(gm, num_m - first_m);
  const int bm = first_m + rem % gs;
  const int bn = rem / gs;
  const int m0 = bm * BM, n0 = bn * BP;

  const int wm0 = (w / PW) * (WM * 16);
  const int wp0 = (w % PW) * (WP * 16);
  // WMMA A layout: lane 2r -> row r, lane 17+2(r-8) -> row 8+r; other lanes
  // are ignored by the hardware. The ignored lanes are assigned complementary
  // dummy rows so every 8-lane LDS phase touches 8 distinct banks.
  const int base_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? base_ : ((base_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((base_ + 4) & 7) : base_);
  const int ksteps = K / KST;

  qw_floatx8 acc[WM][WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  // loop-invariant LDS byte offsets for this lane's fragment reads
  uint32_t aoff[WM], boff[WP];
#pragma unroll
  for (int i = 0; i < WM; ++i)
    aoff[i] = ((wm0 + i * 16 + qrowA) * LSA) * 2;
#pragma unroll
  for (int j = 0; j < WP; ++j) {
    const int pl = wp0 + j * 16 + (lane & 15);
    boff[j] = ((pl >> 3) * BREG + (pl & 7) * LSA) * 2;
  }
  // staging: bank-phase-paired row mapping. A warp stages RPW=32/(KST/8)
  // rows x (KST/8) 16B units; a ds_store_b128 retires in 8-lane phases, and
  // with the LSA=80B row stride any two CONSECUTIVE rows in a phase always
  // collide on one 4-bank group (the second row's 4 blocks always wrap onto
  // byte offset 128 == bank 0). Pairing rows (b, b+RPW/2) per phase makes
  // each phase cover all 32 banks exactly once (SQC_LDS_BANK_CONFLICT:
  // 39.8M -> 0 per dispatch). Same warp-global address set, so global
  // coalescing is unchanged; LDS layout and all load paths untouched.
  constexpr int Q8 = KST / 8;
  constexpr int RPW = 32 / Q8;
  static_assert(RPW % 2 == 0 && KST % 8 == 0, "store mapping mismatch");
  const uint16_t* ga[AN];
  const uint16_t* gb[BN];
  uint32_t saoff[AN], sboff[BN];
#pragma unroll
  for (int j = 0; j < AN; ++j) {
    const int idx = tid + j * NT;
    const int q = idx % Q8, b = (idx / Q8) % RPW;
    const int row = (idx / (Q8 * RPW)) * RPW + (b & 1) * (RPW / 2) + (b / 2);
    ga[j] = X + (size_t)min(m0 + row, P - 1) * K + q * 8;
    saoff[j] = (row * LSA + q * 8) * 2;
  }
#pragma unroll
  for (int j = 0; j < BN; ++j) {
    const int idx = tid + j * NT;
    const int q = idx % Q8, b = (idx / Q8) % RPW;
    const int row = (idx / (Q8 * RPW)) * RPW + (b & 1) * (RPW / 2) + (b / 2);
    gb[j] = W + (size_t)min(n0 + row, N - 1) * K + q * 8;
    sboff[j] = ((row >> 3) * BREG + (row & 7) * LSA + q * 8) * 2;
  }

#define GWMMA_STAGE_LOAD(RA, RB, KS)                    \
  do {                                                  \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)      \
        RA[j] = *(const uint4*)(ga[j] + (KS)*KST);      \
    _Pragma("unroll") for (int j = 0; j < BN; ++j)      \
        RB[j] = *(const uint4*)(gb[j] + (KS)*KST);      \
  } while (0)
#define GWMMA_STAGE_STORE(BUF, RA, RB)                          \
  do {                                                          \
    _Pragma("unroll") for (int j = 0; j < AN; ++j)              \
        *(uint4*)((char*)&As[BUF][0] + saoff[j]) = RA[j];       \
    _Pragma("unroll") for (int j = 0; j < BN; ++j)              \
        *(uint4*)((char*)&Bs[BUF][0] + sboff[j]) = RB[j];       \
  } while (0)
#define GWMMA_COMPUTE(CUR)                                              \
  do {                                                                  \
    const char* Ab = (const char*)&As[CUR][0];                          \
    const char* Bb = (const char*)&Bs[CUR][0];                          \
    _Pragma("unroll") for (int kh = 0; kh < KST / 16; ++kh) {          \
      qw_shortx16 af[WM];                                               \
      _Pragma("unroll") for (int i = 0; i < WM; ++i)                    \
          af[i] = qw_ld16((const uint16_t*)(Ab + aoff[i] + kh * 32));   \
      _Pragma("unroll") for (int j = 0; j < WP; ++j) {                  \
        const qw_shortx16 bfj =                                         \
            qw_ld16((const uint16_t*)(Bb + boff[j] + kh * 32));         \
        _Pragma("unroll") for (int i = 0; i < WM; ++i)                  \
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(    \
                af[i], bfj, acc[i][j]);                                 \
      }                                                                 \
    }                                                                   \
  } while (0)

  uint4 ra0[AN], rb0[BN];
  GWMMA_STAGE_LOAD(ra0, rb0, 0);
  GWMMA_STAGE_STORE(0, ra0, rb0);
  gwmma_sync();

#pragma unroll 1
  for (int ks = 0; ks < ksteps; ++ks) {
    GWMMA_STAGE_LOAD(ra0, rb0, min(ks + 1, ksteps - 1));
    GWMMA_COMPUTE(ks & 1);
    GWMMA_STAGE_STORE((ks & 1) ^ 1, ra0, rb0);
    gwmma_sync();
  }
#undef GWMMA_STAGE_LOAD
#undef GWMMA_STAGE_STORE
#undef GWMMA_COMPUTE

  const int rh = (lane >> 4) * 8, cl = lane & 15;
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WP; ++j) {
      const int p0 = m0 + wm0 + i * 16 + rh;
      const int n = n0 + wp0 + j * 16 + cl;
      if (n < N)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          if (p0 + e < P) Y[(size_t)(p0 + e) * N + n] = acc[i][j][e];
    }
}
// [gemm-wmma-end]

// ================== fused MoE W4 GEMM (Phase 3d-1) kernels ===================
// Pairs bucketed by expert: expert e owns pairs eoff[e]..eoff[e+1), token of
// pair p = tokidx[p], router weight = pw[p]. Both kernels read Q4C-P codes
// directly (fp32 accum, no bf16 intermediates): block = (expert, MOE_CT-col
// tile), 256 threads with an 8-row x FD_CPT-col accumulator tile each, 64-pair
// row tiles, x staged in LDS in 32-wide K chunks. Weight codes are streamed
// exactly once per expert per launch. Per chunk the block dequantizes the
// code tile ONCE into an fp32 LDS tile ([k][col], padded pitch), so the
// compute loop is pure FMA — per-warp re-decoding of packed codes was the
// bottleneck of the earlier version (measured ~40% of issue slots), and the
// 8-row tile halves LDS bytes per FMA vs 4 rows (LDS bandwidth is the other
// wall: gfx1151 delivers ~1B per fp32 FMA). Numerics: one 32-wide scale
// group per chunk; the chunk partial sums are scaled and folded into the
// accumulators at chunk end, so per-element accumulation order matches the
// grouped GEMV path bit for bit.
// Hard requirements (host-checked): out cols % MOE_CT == 0, K cols % MOE_KC
// == 0 — no runtime guards in the unrolled loops (they defeat unrolling and
// spill the accumulators to scratch).
#define MOE_RT 64
// Output-column tile. 128 is the measured optimum on gfx1151: halving to 64
// halves per-block accumulators/LDS and doubles grid.x, but each 64-row block
// still runs the full K-chunk loop, so x staging and barrier count per unit
// output double — measured 2026-09-09 (tools/MOE_OCC.md): 32K up +12%, down
// +30% slower; occupancy gain does not compensate. Rebuild with
// -DGDEC_MOE_CT64 to rerun that experiment (compile-time switch; the GDEC_*
// runtime fallbacks are unaffected).
#ifdef GDEC_MOE_CT64
#define MOE_CT 64
#else
#define MOE_CT 128
#endif
#define MOE_KC 32             // one 32-wide scale group per chunk
#define MOE_XP 36             // xs row pitch (32 + pad against store conflicts)
#define MOE_WP (MOE_CT + 4)   // wf row pitch (pad: conflict-free stage+compute)
// fdot2_bf16 variant tiling (xs/wf staged as bf162 pairs along k)
#define FD_RT 64
#define FD_CT MOE_CT
// K chunk. Default: 32-wide single-buffered dual-barrier loop. The 16-wide
// double-buffered single-barrier variant (-DGDEC_MOE_KC16DB, stage of chunk
// i+1 overlaps compute of chunk i, bit-identical 32-wide scale fold via
// chunk partials living across two 16-chunks) was MEASURED A REGRESSION
// (2026-09-09: packed up +13%, packed down +8% @32K — halving per-chunk
// compute makes the per-chunk stage/loop overhead dominate, barrier parity
// notwithstanding) and is kept opt-in, like GDEC_MOE_CT64.
#ifndef GDEC_MOE_KC16DB
#define FD_KC 32
#define FD_XP 20              // xs2 row pitch in bf162 (80B, 16B-aligned)
#else
#define FD_KC 16
#define FD_XP 12              // xs2 row pitch in bf162 (48B, 16B-aligned)
#endif
#define FD_WP (FD_CT + 4)     // wf2 row pitch in bf162 (bank-rotating)
#define FD_CPT (FD_CT / 32)   // output columns per thread
// Down-only wide column tile experiment: -DGDEC_MOE_DOWN_CT256 (d=2560 is
// divisible by 256; moe_mid=640 is not, so up always keeps FD_CT).
#ifdef GDEC_MOE_DOWN_CT256
#define FD_CT_DN 256
#else
#define FD_CT_DN FD_CT
#endif
#define FD_WP_DN (FD_CT_DN + 4)
#define FD_CPT_DN (FD_CT_DN / 32)
#if defined(GDEC_MOE_DOWN_CT256) && !defined(GDEC_MOE_KC16DB)
#error "GDEC_MOE_DOWN_CT256 requires the KC16 double-buffered loop (256 columns exceed the 256-thread KC32 staging layout)"
#endif
// Min blocks/CU hint: CT=64 targets 4 resident blocks, CT=128 fits 2.
#ifdef GDEC_MOE_LB2
#define FD_MINB 2
#else
#define FD_MINB (FD_CPT == 2 ? 4 : 2)
#endif

// up: hid[p*mid + c] = silu(g)*u, g = W_e[c].x[tokidx[p]], u = W_e[mid+c].x[.]
// (W_e = [2*mid, cols] Q4C-P). x rows stride cols. Grid is (col-tile, expert)
// so same-expert blocks launch adjacently and share the gathered x rows
// through L2. The next chunk's global loads (codes/scales) are issued right
// after the LDS stores so their latency hides behind the dot-product loop.
// Compute: v_dot2_f32_bf16 (fp32 accum) over bf162 pairs staged in LDS;
// activations and dequantized cb weights are rounded to bf16 on staging
// (bf16-level precision, as elsewhere in the server). The 32-group scale is
// still applied to fp32 chunk partials, keeping the accumulation structure of
// the grouped GEMV path.
// Packed reads cached BF16 x and writes BF16 hidden values; the old down kernel
// performed exactly this rounding when staging the FP32 hidden values.
template <bool Tiled = false, bool Packed = false,
          bool RowDot = Packed, bool PairTable = Packed>
__global__ void __launch_bounds__(256, FD_MINB)
    k_moe_w4_up(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                const float* __restrict__ cb, const float* __restrict__ x,
                const int* __restrict__ tokidx, const int* __restrict__ eoff,
                float* __restrict__ hid, int rows_per, int cols, int mid,
                uint64_t scale_stride, const MoeTile* tiles = nullptr, const int* ntiles = nullptr,
                const uint16_t* packed_x = nullptr, uint16_t* packed_hid = nullptr) {
  if (Tiled && blockIdx.y >= *ntiles) return;
  const int e = Tiled ? tiles[blockIdx.y].expert : blockIdx.y, c0 = blockIdx.x * FD_CT;
  const int n0 = Tiled ? tiles[blockIdx.y].first : eoff[e];
  const int ne = Tiled ? tiles[blockIdx.y].count : eoff[e + 1] - n0;
  if (!ne) return;
#ifndef GDEC_MOE_KC16DB
  // One byte selects two already-rounded codebook values. The first token-map
  // barrier below publishes the table before any weight tile consumes it.
  __shared__ __hip_bfloat162 cb_pairs[PairTable ? 256 : 1];
  if (PairTable)
    cb_pairs[threadIdx.x] = __halves2bfloat162(
        __float2bfloat16(cb[threadIdx.x & 15]),
        __float2bfloat16(cb[threadIdx.x >> 4]));
  auto decode_pair = [&](uint32_t codes2) {
    if constexpr (PairTable) return cb_pairs[codes2 & 255];
    else return __halves2bfloat162(__float2bfloat16(__ldg(cb + (codes2 & 15))),
                                  __float2bfloat16(__ldg(cb + ((codes2 >> 4) & 15))));
  };
  __shared__ __hip_bfloat162 xs2[FD_RT][FD_XP];
  __shared__ __hip_bfloat162 wfg2[FD_KC / 2][FD_WP];  // gate weights, dequantized
  __shared__ __hip_bfloat162 wfu2[FD_KC / 2][FD_WP];  // up weights, dequantized
  __shared__ float wsg[FD_CT], wsu[FD_CT];            // per-column group scales
#else
  // double-buffered: stage of chunk i+1 overlaps compute of chunk i
  __shared__ __hip_bfloat162 xs2[2][FD_RT][FD_XP];
  __shared__ __hip_bfloat162 wfg2[2][FD_KC / 2][FD_WP];  // gate weights
  __shared__ __hip_bfloat162 wfu2[2][FD_KC / 2][FD_WP];  // up weights
  __shared__ float wsg[2][FD_CT], wsu[2][FD_CT];         // per-column group scales
#endif
  const int cg = threadIdx.x & 31, rg = threadIdx.x >> 5;  // FD_CPT cols x 8 rows
  const uint8_t* ec = codes + (size_t)e * rows_per * (cols / 2);
  const uint8_t* es = scales + (size_t)e * rows_per * scale_stride;
  const size_t rowb = (size_t)cols / 2;
  const size_t uoff = (size_t)mid * rowb;
  const size_t usoff = (size_t)mid * scale_stride;
  for (int r0 = 0; r0 < ne; r0 += FD_RT) {
    const int nr = min(FD_RT, ne - r0);
    float accg[FD_CPT][8] = {}, accu[FD_CPT][8] = {};  // [cc][r]
    __shared__ int tk[FD_RT];  // row -> token, resolved once per row tile
    if (threadIdx.x < FD_RT)
      tk[threadIdx.x] = threadIdx.x < nr ? tokidx[n0 + r0 + threadIdx.x] : 0;
    __syncthreads();
#ifndef GDEC_MOE_KC16DB
    // staging slots: codes row tid/2, half tid%2 (8B = 16 nibbles);
    // with FD_CT=64 only the first 128 threads stage weights (sact).
    const int srow = threadIdx.x >> 1, sseg = threadIdx.x & 1;
    const bool sact = srow < FD_CT;
    const uint8_t* rb = ec + (size_t)(c0 + srow) * rowb + sseg * 8;
    const uint8_t* sb = es + (size_t)(c0 + srow) * scale_stride;
    uint2 pgc, puc;
    float sg, su;
    auto stage_load = [&](int k0) {
      pgc = *(const uint2*)(rb + k0 / 2);
      puc = *(const uint2*)(rb + uoff + k0 / 2);
      sg = __half2float(*(const __half*)(sb + (k0 / 32) * 2));
      su = __half2float(*(const __half*)(sb + usoff + (k0 / 32) * 2));
    };
    if (sact) stage_load(0);
    // x gather prefetch: this thread's staging slot is row xr, pairs
    // xp4..xp4+3 (16B per chunk). When enabled, the next chunk's global loads
    // are issued right after the LDS stores, beside stage_load, so the gather
    // latency hides behind the dot-product loop. Bit-identical either way:
    // same bytes gathered, same conversions, only the issue point moves.
    // Measured 2026-09-09 (KC32): pays off only on the fp32 path (2x16B
    // loads, up -3.5%); on the packed bf16 path the 16B gather is too cheap
    // to amortize the register live range (packed up +3%), so it is gated
    // off there. -DGDEC_MOE_NO_XPREF disables it everywhere.
#ifdef GDEC_MOE_NO_XPREF
    constexpr bool XPREF = false;
#else
    constexpr bool XPREF = !Packed;
#endif
    const int xr = threadIdx.x >> 2, xp4 = (threadIdx.x & 3) * 4;
    const bool xact = xr < nr;
    uint4 xp0, xp1;
    auto x_load = [&](int k0) {
      if (!xact) return;
      const size_t xoff = (size_t)tk[xr] * cols + xp4 * 2 + k0;
      if (Packed) {
        xp0 = *(const uint4*)(packed_x + xoff);
      } else {
        *(float4*)&xp0 = *(const float4*)(x + xoff);
        *(float4*)&xp1 = *(const float4*)(x + xoff + 4);
      }
    };
    if (XPREF) x_load(0);
    for (int k0 = 0; k0 < cols; k0 += FD_KC) {
      __syncthreads();
      {
        // gather x rows -> bf162 pairs in LDS: 64 rows x 16 pairs = 1024
        // bf162, 256 threads x 4 bf162 (16B) each
        {
          __hip_bfloat162 v[4];
          if (xact) {  // 4 pairs = 8 k values
            if (Packed) {
              if (XPREF) *(uint4*)v = xp0;
              else *(uint4*)v = *(const uint4*)(packed_x + (size_t)tk[xr] * cols + k0 + xp4 * 2);
            } else {
              float4 x0, x1;
              if (XPREF) {
                x0 = *(const float4*)&xp0;
                x1 = *(const float4*)&xp1;
              } else {
                const float* xp = x + (size_t)tk[xr] * cols + k0 + xp4 * 2;
                x0 = *(const float4*)xp;
                x1 = *(const float4*)(xp + 4);
              }
              v[0] = __halves2bfloat162(__float2bfloat16(x0.x), __float2bfloat16(x0.y));
              v[1] = __halves2bfloat162(__float2bfloat16(x0.z), __float2bfloat16(x0.w));
              v[2] = __halves2bfloat162(__float2bfloat16(x1.x), __float2bfloat16(x1.y));
              v[3] = __halves2bfloat162(__float2bfloat16(x1.z), __float2bfloat16(x1.w));
            }
          } else {
            __hip_bfloat162 z = __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
            v[0] = z; v[1] = z; v[2] = z; v[3] = z;
          }
          *(float4*)&xs2[xr][xp4] = *(float4*)v;
        }
        // dequant staged codes into bf162 wf tiles (pairs along k, raw cb
        // values; the 32-group scale is applied to the chunk partial sums)
        if (sact) {
          uint32_t vg0 = pgc.x, vg1 = pgc.y, vu0 = puc.x, vu1 = puc.y;
          if (sseg == 0) {
            wsg[srow] = sg;
            wsu[srow] = su;
          }
#pragma unroll
          for (int j = 0; j < 4; j++) {
            wfg2[sseg * 8 + j][srow] = decode_pair(vg0);
            wfu2[sseg * 8 + j][srow] = decode_pair(vu0);
            vg0 >>= 8;
            vu0 >>= 8;
          }
#pragma unroll
          for (int j = 0; j < 4; j++) {
            wfg2[sseg * 8 + 4 + j][srow] = decode_pair(vg1);
            wfu2[sseg * 8 + 4 + j][srow] = decode_pair(vu1);
            vg1 >>= 8;
            vu1 >>= 8;
          }
        }
      }
      if (k0 + FD_KC < cols) {
        if (sact) stage_load(k0 + FD_KC);
        if (XPREF) x_load(k0 + FD_KC);
      }
      __syncthreads();
      // compute: v_dot2_f32_bf16 into chunk-partial accumulators (one
      // 32-group per chunk), group scale applied at chunk end. Scales are
      // per output column, from LDS.
      float sgv[FD_CPT], suv[FD_CPT];
#pragma unroll
      for (int cc = 0; cc < FD_CPT; cc++) {
        sgv[cc] = wsg[cg * FD_CPT + cc];
        suv[cc] = wsu[cg * FD_CPT + cc];
      }
      float acg[FD_CPT][8] = {}, acu[FD_CPT][8] = {};
#pragma unroll 1
      for (int i = 0; i < FD_KC / 8; i++) {
        __hip_bfloat162 xv[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++)
          *(float4*)xv[r] = *(const float4*)&xs2[rg * 8 + r][i * 4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          __hip_bfloat162 wgv[FD_CPT], wuv[FD_CPT];
          if (FD_CPT == 4) {
            *(float4*)wgv = *(const float4*)&wfg2[i * 4 + kk][cg * FD_CPT];
            *(float4*)wuv = *(const float4*)&wfu2[i * 4 + kk][cg * FD_CPT];
          } else {
            *(uint2*)wgv = *(const uint2*)&wfg2[i * 4 + kk][cg * FD_CPT];
            *(uint2*)wuv = *(const uint2*)&wfu2[i * 4 + kk][cg * FD_CPT];
          }
          // Interleave independent output columns within each token row.
          // Column-major expansion regressed packed up by about 12% on gfx1151.
          if constexpr (RowDot && FD_CPT == 4) {
#pragma unroll
            for (int r = 0; r < 8; r++) {
              acg[0][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[0], xv[r][kk], acg[0][r], false);
              acg[1][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[1], xv[r][kk], acg[1][r], false);
              acg[2][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[2], xv[r][kk], acg[2][r], false);
              acg[3][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[3], xv[r][kk], acg[3][r], false);
              acu[0][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[0], xv[r][kk], acu[0][r], false);
              acu[1][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[1], xv[r][kk], acu[1][r], false);
              acu[2][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[2], xv[r][kk], acu[2][r], false);
              acu[3][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[3], xv[r][kk], acu[3][r], false);
            }
          } else {
#pragma unroll
            for (int cc = 0; cc < FD_CPT; cc++)
#pragma unroll
              for (int r = 0; r < 8; r++) {
                acg[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[cc], xv[r][kk], acg[cc][r], false);
                acu[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[cc], xv[r][kk], acu[cc][r], false);
              }
          }
        }
      }
#pragma unroll
      for (int cc = 0; cc < FD_CPT; cc++)
#pragma unroll
        for (int r = 0; r < 8; r++) {
          accg[cc][r] += sgv[cc] * acg[cc][r];
          accu[cc][r] += suv[cc] * acu[cc][r];
        }
    }
#else  // FD_KC=16 double-buffered, one barrier per chunk
    // staging slots: thread t stages weight column t%FD_CT, gate (t<FD_CT)
    // or up (t<2*FD_CT) — 8B = 16 nibbles per 16-chunk; x slot is row
    // xr=tid/4, pairs (tid%4)*2..+1 (8B per chunk).
    const int srow = threadIdx.x & (FD_CT - 1), sseg = threadIdx.x / FD_CT;
    const bool sact = sseg < 2;
    const uint8_t* rb = ec + (size_t)(c0 + srow) * rowb + sseg * uoff;
    const uint8_t* sb = es + (size_t)(c0 + srow) * scale_stride + sseg * usoff;
    uint2 pk2;
    float sc;
    auto stage_load = [&](int k0) {  // k0 = 16-wide chunk offset
      pk2 = *(const uint2*)(rb + k0 / 2);
      sc = __half2float(*(const __half*)(sb + (k0 / 32) * 2));
    };
    const int xr = threadIdx.x >> 2, xp2 = (threadIdx.x & 3) * 2;
    const bool xact = xr < nr;
    const size_t xoff = (size_t)tk[xr] * cols + xp2 * 2;
    uint4 xp;  // next chunk's x: 8B packed / 16B fp32 (see k_moe_w4_up notes)
    auto x_load = [&](int k0) {
      if (!xact) return;
      if (Packed) *(uint2*)&xp = *(const uint2*)(packed_x + xoff + k0);
      else *(float4*)&xp = *(const float4*)(x + xoff + k0);
    };
    auto x_store = [&](int buf, int k0) {
      __hip_bfloat162 v[2];
      bool got = xact;
#ifdef GDEC_MOE_NO_XPREF
      if (got) {
        if (Packed) {
          *(uint2*)v = *(const uint2*)(packed_x + xoff + k0);
        } else {
          float4 f = *(const float4*)(x + xoff + k0);
          v[0] = __halves2bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
          v[1] = __halves2bfloat162(__float2bfloat16(f.z), __float2bfloat16(f.w));
        }
      }
#else
      (void)k0;
      if (got) {
        if (Packed) {
          *(uint2*)v = *(const uint2*)&xp;
        } else {
          float4 f = *(const float4*)&xp;
          v[0] = __halves2bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
          v[1] = __halves2bfloat162(__float2bfloat16(f.z), __float2bfloat16(f.w));
        }
      }
#endif
      if (!got) {
        __hip_bfloat162 z = __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
        v[0] = z; v[1] = z;
      }
      *(uint2*)&xs2[buf][xr][xp2] = *(uint2*)v;
    };
    auto stage_w = [&](int buf) {  // dequant staged codes into buffer buf
      if (!sact) return;
      __hip_bfloat162(*wfb)[FD_WP] = (sseg ? wfu2 : wfg2)[buf];
      if (sseg) wsu[buf][srow] = sc; else wsg[buf][srow] = sc;
      uint32_t v0 = pk2.x, v1 = pk2.y;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wfb[j][srow] = __halves2bfloat162(__float2bfloat16(__ldg(cb + (v0 & 0xf))), __float2bfloat16(__ldg(cb + ((v0 >> 4) & 0xf))));
        v0 >>= 8;
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wfb[4 + j][srow] = __halves2bfloat162(__float2bfloat16(__ldg(cb + (v1 & 0xf))), __float2bfloat16(__ldg(cb + ((v1 >> 4) & 0xf))));
        v1 >>= 8;
      }
    };
    // prologue: stage chunk 0 into buffer 0 (the tk barrier above is also
    // the WAR barrier against the previous row tile's compute reads)
    if (sact) stage_load(0);
#ifndef GDEC_MOE_NO_XPREF
    x_load(0);
#endif
    stage_w(0);
    x_store(0, 0);
    __syncthreads();
    // The chunk-partial accumulators live across the two 16-chunks of each
    // 32-wide scale group (k ascending), so the scale fold lands on exactly
    // the same sums as the 32-wide chunk loop: bit-identical.
    float acg[FD_CPT][8] = {}, acu[FD_CPT][8] = {};
    const int nchunk = cols / FD_KC;  // even: cols % 32 == 0 (host-checked)
    for (int ci = 0; ci < nchunk; ci++) {
      const int cur = ci & 1, nxt = cur ^ 1, k0 = ci * FD_KC;
      // issue chunk ci+1's global loads; latency hides behind the compute
      if (ci + 1 < nchunk) {
        if (sact) stage_load(k0 + FD_KC);
#ifndef GDEC_MOE_NO_XPREF
        x_load(k0 + FD_KC);
#endif
      }
      float sgv[FD_CPT], suv[FD_CPT];
#pragma unroll
      for (int cc = 0; cc < FD_CPT; cc++) {
        sgv[cc] = wsg[cur][cg * FD_CPT + cc];
        suv[cc] = wsu[cur][cg * FD_CPT + cc];
      }
#pragma unroll 1
      for (int i = 0; i < FD_KC / 8; i++) {
        __hip_bfloat162 xv[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++)
          *(float4*)xv[r] = *(const float4*)&xs2[cur][rg * 8 + r][i * 4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          __hip_bfloat162 wgv[FD_CPT], wuv[FD_CPT];
          if (FD_CPT == 4) {
            *(float4*)wgv = *(const float4*)&wfg2[cur][i * 4 + kk][cg * FD_CPT];
            *(float4*)wuv = *(const float4*)&wfu2[cur][i * 4 + kk][cg * FD_CPT];
          } else {
            *(uint2*)wgv = *(const uint2*)&wfg2[cur][i * 4 + kk][cg * FD_CPT];
            *(uint2*)wuv = *(const uint2*)&wfu2[cur][i * 4 + kk][cg * FD_CPT];
          }
#pragma unroll
          for (int cc = 0; cc < FD_CPT; cc++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
              acg[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wgv[cc], xv[r][kk], acg[cc][r], false);
              acu[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wuv[cc], xv[r][kk], acu[cc][r], false);
            }
        }
      }
      if (ci & 1) {  // end of a 32-wide scale group: fold chunk partials
#pragma unroll
        for (int cc = 0; cc < FD_CPT; cc++)
#pragma unroll
          for (int r = 0; r < 8; r++) {
            accg[cc][r] += sgv[cc] * acg[cc][r];
            accu[cc][r] += suv[cc] * acu[cc][r];
            acg[cc][r] = 0.f;
            acu[cc][r] = 0.f;
          }
      }
      // stage chunk ci+1 into the other buffer (safe: everyone finished
      // reading it at chunk ci-1, separated by the barrier below)
      if (ci + 1 < nchunk) {
        stage_w(nxt);
        x_store(nxt, k0 + FD_KC);
      }
      __syncthreads();
    }
#endif
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int row = rg * 8 + r;
      if (row >= nr) continue;
      size_t hi = (size_t)(n0 + r0 + row) * mid + c0 + cg * FD_CPT;
#pragma unroll
      for (int cc = 0; cc < FD_CPT; cc++) {
        float g = accg[cc][r], u = accu[cc][r];
        float value = g / (1.f + expf(-g)) * u;
        if (Packed) packed_hid[hi + cc] = f2bf(value);
        else hid[hi + cc] = value;
      }
    }
  }
}

// down: acc[tokidx[p]*rows_per + r] += pw[p] * (W_e[r] . hid[p])
// (W_e = [rows_per, cols] Q4C-P). hid rows have stride cols. Same staging
// structure as k_moe_w4_up; single projection, so wf2 + xs2 stay small enough
// for several resident blocks per CU. Weighted outputs either scatter to tokens or
// occupy separate pair rows for deterministic reduction. bf162/fdot2 compute,
// same numerics note as k_moe_w4_up.
template <bool Scatter = true, bool Tiled = false, bool Packed = false,
          bool PairTable = Packed, bool PairsBf16 = false>
__global__ void __launch_bounds__(256, FD_MINB)
    k_moe_w4_down(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                  const float* __restrict__ cb, const float* __restrict__ hid,
                  const int* __restrict__ tokidx, const float* __restrict__ pw,
                  const int* __restrict__ eoff, float* __restrict__ acc, int rows_per,
                  int cols, uint64_t scale_stride,
                  const MoeTile* tiles = nullptr, const int* ntiles = nullptr,
                  const uint16_t* packed_hid = nullptr) {
  if (Tiled && blockIdx.y >= *ntiles) return;
  const int e = Tiled ? tiles[blockIdx.y].expert : blockIdx.y, c0 = blockIdx.x * FD_CT_DN;
  const int n0 = Tiled ? tiles[blockIdx.y].first : eoff[e];
  const int ne = Tiled ? tiles[blockIdx.y].count : eoff[e + 1] - n0;
  if (!ne) return;
#ifndef GDEC_MOE_KC16DB
  // The first K-loop barrier publishes the table before weight staging.
  __shared__ __hip_bfloat162 cb_pairs[PairTable ? 256 : 1];
  if (PairTable)
    cb_pairs[threadIdx.x] = __halves2bfloat162(
        __float2bfloat16(cb[threadIdx.x & 15]),
        __float2bfloat16(cb[threadIdx.x >> 4]));
  __shared__ __hip_bfloat162 xs2[FD_RT][FD_XP];
  __shared__ __hip_bfloat162 wf2[FD_KC / 2][FD_WP_DN];  // weights, dequantized
  __shared__ float wsc[FD_CT_DN];                       // per-column group scales
#else
  __shared__ __hip_bfloat162 xs2[2][FD_RT][FD_XP];
  __shared__ __hip_bfloat162 wf2[2][FD_KC / 2][FD_WP_DN];  // weights, dequantized
  __shared__ float wsc[2][FD_CT_DN];                       // per-column group scales
#endif
  const int cg = threadIdx.x & 31, rg = threadIdx.x >> 5;
  const uint8_t* ec = codes + (size_t)e * rows_per * (cols / 2);
  const uint8_t* es = scales + (size_t)e * rows_per * scale_stride;
  const size_t rowb = (size_t)cols / 2;
  for (int r0 = 0; r0 < ne; r0 += FD_RT) {
    const int nr = min(FD_RT, ne - r0);
    float accv[FD_CPT_DN][8] = {};  // [cc][r]
#ifndef GDEC_MOE_KC16DB
    const int srow = threadIdx.x >> 1, sseg = threadIdx.x & 1;
    const bool sact = srow < FD_CT_DN;  // CT=64: only half the threads stage
    const uint8_t* rb = ec + (size_t)(c0 + srow) * rowb + sseg * 8;
    const uint8_t* sb = es + (size_t)(c0 + srow) * scale_stride;
    uint2 pk2;
    float sc;
    auto stage_load = [&](int k0) {
      pk2 = *(const uint2*)(rb + k0 / 2);
      sc = __half2float(*(const __half*)(sb + (k0 / 32) * 2));
    };
    if (sact) stage_load(0);
    // x gather prefetch (see k_moe_w4_up): row xr, pairs xp4..xp4+3; loads
    // issued one chunk ahead beside stage_load. Bit-identical either way.
    // Gated to the fp32 path (packed gather is too cheap to amortize the
    // register live range); -DGDEC_MOE_NO_XPREF disables it everywhere.
#ifdef GDEC_MOE_NO_XPREF
    constexpr bool XPREF = false;
#else
    constexpr bool XPREF = !Packed;
#endif
    const int xr = threadIdx.x >> 2, xp4 = (threadIdx.x & 3) * 4;
    const bool xact = xr < nr;
    uint4 xp0, xp1;
    auto x_load = [&](int k0) {
      if (!xact) return;
      const size_t xoff = (size_t)(n0 + r0 + xr) * cols + xp4 * 2 + k0;
      if (Packed) {
        xp0 = *(const uint4*)(packed_hid + xoff);
      } else {
        *(float4*)&xp0 = *(const float4*)(hid + xoff);
        *(float4*)&xp1 = *(const float4*)(hid + xoff + 4);
      }
    };
    if (XPREF) x_load(0);
    for (int k0 = 0; k0 < cols; k0 += FD_KC) {
      __syncthreads();
      {
        {
          __hip_bfloat162 v[4];  // 256 threads x 16B = 64 rows x 16 pairs
          if (xact) {
            if (Packed) {
              if (XPREF) *(uint4*)v = xp0;
              else *(uint4*)v = *(const uint4*)(packed_hid + (size_t)(n0 + r0 + xr) * cols + k0 + xp4 * 2);
            } else {
              float4 x0, x1;
              if (XPREF) {
                x0 = *(const float4*)&xp0;
                x1 = *(const float4*)&xp1;
              } else {
                const float* xp = hid + (size_t)(n0 + r0 + xr) * cols + k0 + xp4 * 2;
                x0 = *(const float4*)xp;
                x1 = *(const float4*)(xp + 4);
              }
              v[0] = __halves2bfloat162(__float2bfloat16(x0.x), __float2bfloat16(x0.y));
              v[1] = __halves2bfloat162(__float2bfloat16(x0.z), __float2bfloat16(x0.w));
              v[2] = __halves2bfloat162(__float2bfloat16(x1.x), __float2bfloat16(x1.y));
              v[3] = __halves2bfloat162(__float2bfloat16(x1.z), __float2bfloat16(x1.w));
            }
          } else {
            __hip_bfloat162 z = __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
            v[0] = z; v[1] = z; v[2] = z; v[3] = z;
          }
          *(float4*)&xs2[xr][xp4] = *(float4*)v;
        }
        if (sact) {
          uint32_t vw0 = pk2.x, vw1 = pk2.y;
          if (sseg == 0) wsc[srow] = sc;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            wf2[sseg * 8 + j][srow] = PairTable ? cb_pairs[vw0 & 255] :
              __halves2bfloat162(__float2bfloat16(__ldg(cb + (vw0 & 0xf))), __float2bfloat16(__ldg(cb + ((vw0 >> 4) & 0xf))));
            vw0 >>= 8;
          }
#pragma unroll
          for (int j = 0; j < 4; j++) {
            wf2[sseg * 8 + 4 + j][srow] = PairTable ? cb_pairs[vw1 & 255] :
              __halves2bfloat162(__float2bfloat16(__ldg(cb + (vw1 & 0xf))), __float2bfloat16(__ldg(cb + ((vw1 >> 4) & 0xf))));
            vw1 >>= 8;
          }
        }
      }
      if (k0 + FD_KC < cols) {
        if (sact) stage_load(k0 + FD_KC);
        if (XPREF) x_load(k0 + FD_KC);
      }
      __syncthreads();
      float sv[FD_CPT_DN];
#pragma unroll
      for (int cc = 0; cc < FD_CPT_DN; cc++) sv[cc] = wsc[cg * FD_CPT_DN + cc];
      float acv[FD_CPT_DN][8] = {};
#pragma unroll 1
      for (int i = 0; i < FD_KC / 8; i++) {
        __hip_bfloat162 xv[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++)
          *(float4*)xv[r] = *(const float4*)&xs2[rg * 8 + r][i * 4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          __hip_bfloat162 wv[FD_CPT_DN];
          if (FD_CPT_DN == 4)
            *(float4*)wv = *(const float4*)&wf2[i * 4 + kk][cg * FD_CPT_DN];
          else
            *(uint2*)wv = *(const uint2*)&wf2[i * 4 + kk][cg * FD_CPT_DN];
#pragma unroll
          for (int cc = 0; cc < FD_CPT_DN; cc++)
#pragma unroll
            for (int r = 0; r < 8; r++)
              acv[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wv[cc], xv[r][kk], acv[cc][r], false);
        }
      }
#pragma unroll
      for (int cc = 0; cc < FD_CPT_DN; cc++)
#pragma unroll
        for (int r = 0; r < 8; r++) accv[cc][r] += sv[cc] * acv[cc][r];
    }
#else  // FD_KC=16 double-buffered, one barrier per chunk
    // staging slots: thread t stages weight column t (t < FD_CT_DN), 8B =
    // 16 nibbles per 16-chunk; x slot is row xr=tid/4, pairs (tid%4)*2..+1.
    const int srow = threadIdx.x & (FD_CT_DN - 1);
    const bool sact = threadIdx.x < FD_CT_DN;
    const uint8_t* rb = ec + (size_t)(c0 + srow) * rowb;
    const uint8_t* sb = es + (size_t)(c0 + srow) * scale_stride;
    uint2 pk2;
    float sc;
    auto stage_load = [&](int k0) {  // k0 = 16-wide chunk offset
      pk2 = *(const uint2*)(rb + k0 / 2);
      sc = __half2float(*(const __half*)(sb + (k0 / 32) * 2));
    };
    const int xr = threadIdx.x >> 2, xp2 = (threadIdx.x & 3) * 2;
    const bool xact = xr < nr;
    const size_t xoff = (size_t)(n0 + r0 + xr) * cols + xp2 * 2;
    uint4 xp;  // next chunk's x: 8B packed / 16B fp32
    auto x_load = [&](int k0) {
      if (!xact) return;
      if (Packed) *(uint2*)&xp = *(const uint2*)(packed_hid + xoff + k0);
      else *(float4*)&xp = *(const float4*)(hid + xoff + k0);
    };
    auto x_store = [&](int buf, int k0) {
      __hip_bfloat162 v[2];
      bool got = xact;
#ifdef GDEC_MOE_NO_XPREF
      if (got) {
        if (Packed) {
          *(uint2*)v = *(const uint2*)(packed_hid + xoff + k0);
        } else {
          float4 f = *(const float4*)(hid + xoff + k0);
          v[0] = __halves2bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
          v[1] = __halves2bfloat162(__float2bfloat16(f.z), __float2bfloat16(f.w));
        }
      }
#else
      (void)k0;
      if (got) {
        if (Packed) {
          *(uint2*)v = *(const uint2*)&xp;
        } else {
          float4 f = *(const float4*)&xp;
          v[0] = __halves2bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
          v[1] = __halves2bfloat162(__float2bfloat16(f.z), __float2bfloat16(f.w));
        }
      }
#endif
      if (!got) {
        __hip_bfloat162 z = __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
        v[0] = z; v[1] = z;
      }
      *(uint2*)&xs2[buf][xr][xp2] = *(uint2*)v;
    };
    auto stage_w = [&](int buf) {
      if (!sact) return;
      wsc[buf][srow] = sc;
      uint32_t v0 = pk2.x, v1 = pk2.y;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wf2[buf][j][srow] = __halves2bfloat162(__float2bfloat16(__ldg(cb + (v0 & 0xf))), __float2bfloat16(__ldg(cb + ((v0 >> 4) & 0xf))));
        v0 >>= 8;
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wf2[buf][4 + j][srow] = __halves2bfloat162(__float2bfloat16(__ldg(cb + (v1 & 0xf))), __float2bfloat16(__ldg(cb + ((v1 >> 4) & 0xf))));
        v1 >>= 8;
      }
    };
    // prologue: stage chunk 0 into buffer 0 (WAR-safe: no barrier since the
    // previous row tile's last compute, but chunk 0 reads nothing and the
    // previous tile ended with a barrier)
    if (sact) stage_load(0);
#ifndef GDEC_MOE_NO_XPREF
    x_load(0);
#endif
    stage_w(0);
    x_store(0, 0);
    __syncthreads();
    // chunk partials live across both 16-chunks of each 32-wide scale group
    // (k ascending) — same fold points as the 32-wide loop, bit-identical.
    float acv[FD_CPT_DN][8] = {};
    const int nchunk = cols / FD_KC;  // even: cols % 32 == 0 (host-checked)
    for (int ci = 0; ci < nchunk; ci++) {
      const int cur = ci & 1, nxt = cur ^ 1, k0 = ci * FD_KC;
      if (ci + 1 < nchunk) {
        if (sact) stage_load(k0 + FD_KC);
#ifndef GDEC_MOE_NO_XPREF
        x_load(k0 + FD_KC);
#endif
      }
      float sv[FD_CPT_DN];
#pragma unroll
      for (int cc = 0; cc < FD_CPT_DN; cc++) sv[cc] = wsc[cur][cg * FD_CPT_DN + cc];
#pragma unroll 1
      for (int i = 0; i < FD_KC / 8; i++) {
        __hip_bfloat162 xv[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++)
          *(float4*)xv[r] = *(const float4*)&xs2[cur][rg * 8 + r][i * 4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          __hip_bfloat162 wv[FD_CPT_DN];
          if (FD_CPT_DN == 8) {
            *(float4*)wv = *(const float4*)&wf2[cur][i * 4 + kk][cg * FD_CPT_DN];
            *(float4*)(wv + 4) = *(const float4*)&wf2[cur][i * 4 + kk][cg * FD_CPT_DN + 4];
          } else if (FD_CPT_DN == 4)
            *(float4*)wv = *(const float4*)&wf2[cur][i * 4 + kk][cg * FD_CPT_DN];
          else
            *(uint2*)wv = *(const uint2*)&wf2[cur][i * 4 + kk][cg * FD_CPT_DN];
#pragma unroll
          for (int cc = 0; cc < FD_CPT_DN; cc++)
#pragma unroll
            for (int r = 0; r < 8; r++)
              acv[cc][r] = __builtin_amdgcn_fdot2_f32_bf16(wv[cc], xv[r][kk], acv[cc][r], false);
        }
      }
      if (ci & 1) {  // end of a 32-wide scale group: fold chunk partials
#pragma unroll
        for (int cc = 0; cc < FD_CPT_DN; cc++)
#pragma unroll
          for (int r = 0; r < 8; r++) {
            accv[cc][r] += sv[cc] * acv[cc][r];
            acv[cc][r] = 0.f;
          }
      }
      if (ci + 1 < nchunk) {
        stage_w(nxt);
        x_store(nxt, k0 + FD_KC);
      }
      __syncthreads();
    }
#endif
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int row = rg * 8 + r;
      if (row >= nr) continue;
      int p = n0 + r0 + row;
      size_t base = (size_t)(Scatter ? tokidx[p] : p) * rows_per + c0 + cg * FD_CPT_DN;
      if constexpr (!Scatter && PairsBf16) {
        // Unweighted bf16 pair rows (f2bf RNE); pw is applied in fp32 by
        // k_moe_reduce_pw_bf16, which rounds later than weight-then-round.
#pragma unroll
        for (int cc = 0; cc < FD_CPT_DN; cc++)
          ((uint16_t*)acc)[base + cc] = f2bf(accv[cc][r]);
      } else {
        float w = pw[p];
#pragma unroll
        for (int cc = 0; cc < FD_CPT_DN; cc++) {
          float value = w * accv[cc][r];
          if (Scatter) atomicAdd(acc + base + cc, value);
          else acc[base + cc] = value;
        }
      }
    }
  }
}

__global__ void k_moe_reduce(const float* pairs, const int* pairids, float* out,
                              int P, int k, int D) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)P * D) return;
  int t = i / D, d = i % D;
  float sum = 0.f;
  for (int s = 0; s < k; s++) sum += pairs[(size_t)pairids[t * k + s] * D + d];
  out[i] = sum;
}

// Fixed top-k reduction candidate. Pair ids are shared by all output columns
// in a block, while pair values remain coalesced across the 256 output lanes.
// The accumulation order is exactly slot 0..k-1, matching k_moe_reduce.
template <int K = 10>
__global__ void k_moe_reduce_fast(const float* __restrict__ pairs,
                                  const int* __restrict__ pairids,
                                  float* __restrict__ out, int P, int D) {
  __shared__ int ids[K];
  int tid = threadIdx.x;
  int t = blockIdx.y, d = tid + (int)blockIdx.x * blockDim.x;
  if (t >= P) return;
  if (tid < K) ids[tid] = pairids[t * K + tid];
  __syncthreads();
  if (d >= D) return;
  float sum = 0.f;
#pragma unroll
  for (int s = 0; s < K; s++) sum += pairs[(size_t)ids[s] * D + d];
  out[(size_t)t * D + d] = sum;
}

// ============ MoE prefill via dequant + hipBLASLt (GDEC_MOE_LT, opt-in) ======
// Batched-GEMM prefill pipeline: per layer, dequant Q4C-P expert weights to
// bf16 in small expert groups (scratch reused, MALL-hot), gather pair rows,
// per-expert bf16 GEMM via hipBLASLt, silu, per-expert down GEMM,
// deterministic weighted reduce.
// Numerics differ from the fused W4 kernels: w = bf16(fp16_scale * cb[q])
// (scale folded into the weight instead of applied to fp32 group partials)
// and Tensile WMMA accumulation order. Prototype: tools/moe_lt_bench.cu.
// Requires GPU routing + deterministic mode (uses d_tokidx/d_eoff/d_pairids).

// Dequant Q4C-P rows to bf16 [rows][cols] (k contiguous): one thread per
// 32-wide scale group (16B codes -> 64B bf16).
__global__ void k_moe_deq_bf16(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               __hip_bfloat16* __restrict__ out, int rows,
                               int cols, uint64_t scale_stride) {
  int groups_per_row = cols / 32;
  int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (int64_t)rows * groups_per_row) return;
  int r = (int)(gid / groups_per_row), g = (int)(gid % groups_per_row);
  const uint8_t* cb16 = codes + (size_t)r * (cols / 2) + g * 16;
  float sc = __half2float(*(const __half*)(scales + (size_t)r * scale_stride + g * 2));
  __hip_bfloat16* o = out + (size_t)r * cols + g * 32;
  __hip_bfloat16 w[32];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    uint8_t b = cb16[j];
    w[2 * j] = __float2bfloat16(sc * cb[b & 15]);
    w[2 * j + 1] = __float2bfloat16(sc * cb[b >> 4]);
  }
  *(uint4*)(o) = *(uint4*)&w[0];
  *(uint4*)(o + 8) = *(uint4*)&w[8];
  *(uint4*)(o + 16) = *(uint4*)&w[16];
  *(uint4*)(o + 24) = *(uint4*)&w[24];
}

// Gather activation rows by pair: xg[p][:] = x[tokidx[p]][:] (bf16, K elems).
__global__ void k_moe_gather_bf16(__hip_bfloat16* __restrict__ xg,
                                  const __hip_bfloat16* __restrict__ x,
                                  const int* __restrict__ tokidx, int K) {
  int p = blockIdx.x;
  const __hip_bfloat16* s = x + (size_t)tokidx[p] * K;
  __hip_bfloat16* dd = xg + (size_t)p * K;
  for (int i = threadIdx.x; i < K / 8; i += blockDim.x)
    *(uint4*)(dd + i * 8) = *(const uint4*)(s + i * 8);
}

// silu(g)*u on fp32 guv [slots][2*mid] -> bf16 hid [slots][mid].
__global__ void k_moe_silu_f32_bf16(const float* __restrict__ guv,
                                    __hip_bfloat16* __restrict__ hid, int mid,
                                    int64_t nslots) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nslots * mid) return;
  int64_t slot = i / mid;
  int r = (int)(i % mid);
  float a = guv[slot * 2 * mid + r];
  hid[i] = __float2bfloat16(a / (1.f + expf(-a)) * guv[slot * 2 * mid + mid + r]);
}

// silu(g)*u on bf16 guv [slots][2*mid] -> bf16 hid [slots][mid].
__global__ void k_moe_silu_bf16(const __hip_bfloat16* __restrict__ guv,
                                __hip_bfloat16* __restrict__ hid, int mid,
                                int64_t nslots) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nslots * mid) return;
  int64_t slot = i / mid;
  int r = (int)(i % mid);
  float a = __bfloat162float(guv[slot * 2 * mid + r]);
  hid[i] = __float2bfloat16(a / (1.f + expf(-a)) *
                            __bfloat162float(guv[slot * 2 * mid + mid + r]));
}

// Weighted deterministic reduce: out[t][d] = sum_s pw[ids[s]] * pairs[ids[s]][d]
// (same slot order as k_moe_reduce_fast; the router weight is applied here
// instead of in the down kernel).
template <int K>
__global__ void k_moe_reduce_pw(const float* __restrict__ pairs,
                                const int* __restrict__ pairids,
                                const float* __restrict__ pw,
                                float* __restrict__ out, int P, int Dd) {
  __shared__ int ids[K];
  int tid = threadIdx.x;
  int t = blockIdx.y, d = tid + (int)blockIdx.x * blockDim.x;
  if (t >= P) return;
  if (tid < K) ids[tid] = pairids[t * K + tid];
  __syncthreads();
  if (d >= Dd) return;
  float sum = 0.f;
#pragma unroll
  for (int s = 0; s < K; s++) sum += pw[ids[s]] * pairs[(size_t)ids[s] * Dd + d];
  out[(size_t)t * Dd + d] = sum;
}

// bf16 pair variant (fp32 accumulate).
template <int K>
__global__ void k_moe_reduce_pw_bf16(const __hip_bfloat16* __restrict__ pairs,
                                     const int* __restrict__ pairids,
                                     const float* __restrict__ pw,
                                     float* __restrict__ out, int P, int Dd) {
  __shared__ int ids[K];
  int tid = threadIdx.x;
  int t = blockIdx.y, d = tid + (int)blockIdx.x * blockDim.x;
  if (t >= P) return;
  if (tid < K) ids[tid] = pairids[t * K + tid];
  __syncthreads();
  if (d >= Dd) return;
  float sum = 0.f;
#pragma unroll
  for (int s = 0; s < K; s++)
    sum += pw[ids[s]] * __bfloat162float(pairs[(size_t)ids[s] * Dd + d]);
  out[(size_t)t * Dd + d] = sum;
}

// Debug instrumentation (GDEC_MOE_LT_CMP): block-reduced max abs diff of the
// LT path's intermediates against the fused W4 reference run in the same
// process on identical inputs. out = [max|diff|, max|ref|] as ordered ints
// (non-negative floats order like their int bit patterns).
__global__ void k_cmp_absmax_bf16(const __hip_bfloat16* __restrict__ a,
                                  const __hip_bfloat16* __restrict__ b,
                                  int64_t n, int* __restrict__ out) {
  float md = 0.f, mr = 0.f;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    float r = __bfloat162float(b[i]);
    md = fmaxf(md, fabsf(__bfloat162float(a[i]) - r));
    mr = fmaxf(mr, fabsf(r));
  }
  for (int o = 16; o; o >>= 1) {
    md = fmaxf(md, __shfl_xor_sync(~0ull, md, o));
    mr = fmaxf(mr, __shfl_xor_sync(~0ull, mr, o));
  }
  __shared__ float sd[8], sr[8];
  int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
  if (!ln) { sd[w] = md; sr[w] = mr; }
  __syncthreads();
  if (!w) {
    md = ln < 8 ? sd[ln] : 0.f;
    mr = ln < 8 ? sr[ln] : 0.f;
    for (int o = 4; o; o >>= 1) {
      md = fmaxf(md, __shfl_xor_sync(~0ull, md, o));
      mr = fmaxf(mr, __shfl_xor_sync(~0ull, mr, o));
    }
    if (!ln) {
      atomicMax(out, __float_as_int(md));
      atomicMax(out + 1, __float_as_int(mr));
    }
  }
}

// Same, but ref holds router-weighted fp32 pairs while lt holds unweighted
// bf16 pairs: diff = ref[i] - pw[i/dd] * lt[i].
__global__ void k_cmp_absmax_pw(const float* __restrict__ ref,
                                const __hip_bfloat16* __restrict__ lt,
                                const float* __restrict__ pw, int npairs, int dd,
                                int* __restrict__ out) {
  float md = 0.f, mr = 0.f;
  int64_t n = (int64_t)npairs * dd;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    float r = ref[i];
    md = fmaxf(md, fabsf(r - pw[i / dd] * __bfloat162float(lt[i])));
    mr = fmaxf(mr, fabsf(r));
  }
  for (int o = 16; o; o >>= 1) {
    md = fmaxf(md, __shfl_xor_sync(~0ull, md, o));
    mr = fmaxf(mr, __shfl_xor_sync(~0ull, mr, o));
  }
  __shared__ float sd[8], sr[8];
  int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
  if (!ln) { sd[w] = md; sr[w] = mr; }
  __syncthreads();
  if (!w) {
    md = ln < 8 ? sd[ln] : 0.f;
    mr = ln < 8 ? sr[ln] : 0.f;
    for (int o = 4; o; o >>= 1) {
      md = fmaxf(md, __shfl_xor_sync(~0ull, md, o));
      mr = fmaxf(mr, __shfl_xor_sync(~0ull, mr, o));
    }
    if (!ln) {
      atomicMax(out, __float_as_int(md));
      atomicMax(out + 1, __float_as_int(mr));
    }
  }
}

// ================== GDN chunked prefill (Phase 3b) kernels ===================

// --- batched conv+silu: out[t][ch] = silu(sum_j cw[ch][j]*hist_j + cw[ch][3]*x[t])
// hist_j = x[t-3+j], t-3+j < 0 reads convst (recent 3 pre-prefill inputs, oldest
// first — matches k_gdn_conv). Not in-place: out must differ from x.
// float4 over channels (qkv % 4 == 0): per-lane op order identical to the
// scalar version -> bit-exact, 4x fewer threads.
__global__ void k_gdn_conv_b(const float* __restrict__ x, const float* __restrict__ cw,
                             const float* __restrict__ convst, float* __restrict__ out,
                             int P, int qkv) {
  int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) * 4;
  if (i >= (int64_t)P * qkv) return;
  int t = (int)(i / qkv), ch = (int)(i - (int64_t)t * qkv);
  float4 xv = *(const float4*)(x + i);
  float4 a;
  a.x = cw[(ch + 0) * 4 + 3] * xv.x;
  a.y = cw[(ch + 1) * 4 + 3] * xv.y;
  a.z = cw[(ch + 2) * 4 + 3] * xv.z;
  a.w = cw[(ch + 3) * 4 + 3] * xv.w;
#pragma unroll
  for (int j = 0; j < 3; j++) {
    int tt = t - 3 + j;
    const float* src =
        tt >= 0 ? x + (size_t)tt * qkv + ch : convst + (tt + 3) * qkv + ch;
    float4 v = *(const float4*)src;
    a.x += cw[(ch + 0) * 4 + j] * v.x;
    a.y += cw[(ch + 1) * 4 + j] * v.y;
    a.z += cw[(ch + 2) * 4 + j] * v.z;
    a.w += cw[(ch + 3) * 4 + j] * v.w;
  }
  float4 o;
  o.x = a.x / (1.f + expf(-a.x));
  o.y = a.y / (1.f + expf(-a.y));
  o.z = a.z / (1.f + expf(-a.z));
  o.w = a.w / (1.f + expf(-a.w));
  *(float4*)(out + i) = o;
}

// --- write back conv state: convst[j] = last-3 inputs after P tokens.
// Per-channel independent; old values are read into registers before writing.
__global__ void k_convst_update(float* __restrict__ convst, const float* __restrict__ x,
                                int P, int qkv) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= qkv) return;
  float o1 = convst[qkv + ch], o2 = convst[2 * qkv + ch];
  float n2 = x[(size_t)(P - 1) * qkv + ch];
  float n1 = P >= 2 ? x[(size_t)(P - 2) * qkv + ch] : o2;
  // P=1 keeps old rows 1,2; P=2 keeps only old row 2.
  float n0 = P >= 3 ? x[(size_t)(P - 3) * qkv + ch] : (P == 2 ? o2 : o1);
  convst[ch] = n0;
  convst[qkv + ch] = n1;
  convst[2 * qkv + ch] = n2;
}

// --- batched GDN q/k L2 norm + q scale: grid (hk, P) over [P, qkvstride] rows.
// q heads at t*qkvstride + h*dk, k heads at t*qkvstride + hk*dk + h*dk.
__global__ void k_l2norm_qk_b(float* __restrict__ qkv, int dk, float qscale,
                              int qkvstride, int hk) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  float* qh = qkv + (size_t)blockIdx.y * qkvstride + h * dk;
  float* kh = qkv + (size_t)blockIdx.y * qkvstride + (size_t)hk * dk + h * dk;
  float v = (t < dk) ? qh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float iq = rsqrtf(sm[0] + 1e-6f);
  __syncthreads();
  float kv = (t < dk) ? kh[t] : 0.f;
  sm[t] = kv * kv;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float ik = rsqrtf(sm[0] + 1e-6f);
  if (t < dk) {
    qh[t] = v * iq * qscale;
    kh[t] = kv * ik;
  }
}

// --- GDN chunked prefill (torch_chunk_gated_delta_rule equivalent), fp32.
// One block (256 threads) per v-head; chunks of 64 tokens serially.
// qkv: conv-ed rows [P, qkvstride]; v-head h reads q,k at kh=h/3 (GVA mapping
// of k_gdn_step: 48 v-heads share 16 k-heads), v at 4096+h*128. Output is
// written in place over v. q arrives already l2normed and scaled by
// 1/sqrt(128) (k_l2norm_qk), so no extra scale here. g/beta rows [P, 48];
// g is the per-token log-decay (chunk-local cumsum, padded zeros contribute
// nothing). S [128,128] per head is read, updated per chunk, written back.
// ws: per-head scratch [2][64][128] fp32 (value'/v_new, k_cumdecay).
// Optimized (Phase 3d-2): the naive version streamed S from global three
// times per chunk (~8MB/chunk/head) and re-read q/v rows per output element.
// Here S is processed once per chunk in 16-column LDS d-tiles that fuse
// v_new = value' - kcd@S, out = qeg@S + attn2@v_new and the S update into a
// single read+write; the v'/kcd pass maps each thread to one d-column (32
// rows) so v rows are read once with no staging; the 63-step forward
// substitution runs warp-synchronously (one warp, lanes over columns, no
// block barriers); expf(gcum) / expf(glast-gcum) are hoisted to per-chunk
// vectors (bit-identical values). Per-element accumulation orders (j/d/r
// ascending) are unchanged, so results match the naive version bit for bit.
#ifndef GDN_NT
#define GDN_NT 1024  // k_gdn_chunk block size (occupancy tuning knob)
#endif
__global__ void k_gdn_chunk(const float* __restrict__ qkv, const float* __restrict__ gb,
                            const float* __restrict__ bb, float* __restrict__ S,
                            float* __restrict__ out, float* __restrict__ ws, int P,
                            int qkvstride) {
  const int CH = 64, DK = 128;
  const int NT = GDN_NT;  // launch bound: blockDim.x must match
  int h = blockIdx.x;
  int kh = h / 3;
  int tid = threadIdx.x;
  __shared__ float s_k[CH][DK + 4];  // row pad: 32-way bank conflicts -> 4-way
  __shared__ float s_attn[CH][CH];
  __shared__ float s_extra[3072];  // q quarter tile [16][128] | S tile [128][16] + vnew [64][16]
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH], s_glast;
  // s_beta is reused for expf(glast-gcum) after the v'/kcd pass
  S += (size_t)h * DK * DK;
  float* wvp = ws + (size_t)h * 2 * CH * DK;  // value' then v_new
  float* wkcd = wvp + CH * DK;                // k_cumdecay
  int nchunks = (P + CH - 1) / CH;
  for (int c = 0; c < nchunks; c++) {
    int t0 = c * CH;
    // load k tile, g, beta; per-chunk cumsum of g (pad rows: k=0, g=0, beta=0)
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      s_k[t][d] =
          (t0 + t < P) ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d] : 0.f;
    }
    if (tid < CH) {
      s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
      s_beta[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
    }
    __syncthreads();
    if (tid == 0) {
      float acc = 0.f;
      for (int i = 0; i < CH; i++) {
        acc += s_gcum[i];
        s_gcum[i] = acc;
      }
      s_glast = s_gcum[CH - 1];
    }
    __syncthreads();
    if (tid < CH) s_eg[tid] = expf(s_gcum[tid]);
    // attn[r][j] = -beta[r] * (k[r].k[j]) * exp(gcum[r]-gcum[j]) for j < r, else 0
    for (int i = tid; i < CH * CH; i += NT) {
      int r = i / CH, j = i % CH;
      float a = 0.f;
      if (j < r) {
        float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        for (int d = 0; d < DK; d += 4) {
          e0 += s_k[r][d] * s_k[j][d];
          e1 += s_k[r][d + 1] * s_k[j][d + 1];
          e2 += s_k[r][d + 2] * s_k[j][d + 2];
          e3 += s_k[r][d + 3] * s_k[j][d + 3];
        }
        float dot = (e0 + e1) + (e2 + e3);
        a = -s_beta[r] * dot * expf(s_gcum[r] - s_gcum[j]);
      }
      s_attn[r][j] = a;
    }
    __syncthreads();
    // forward substitution: attn[i,:i] += sum_m attn[i,m](old) * attn[m,:i](updated).
    // Rows are serially dependent; one warp processes them with lanes over
    // columns (m ascending per element — same order as the naive loop).
    {
      // NT threads = NT/64 m-groups (tid>>6) x 64 columns (tid&63); per row each
      // group sums its strided m range into s_extra, then group 0 merges.
      const int j = tid & (CH - 1), mg = tid >> 6;
      for (int i = 1; i < CH; i++) {
        float p = 0.f;
        for (int m = mg; m < i; m += NT / CH) p += s_attn[i][m] * s_attn[m][j];
        s_extra[mg * CH + j] = p;
        __syncthreads();
        if (mg == 0 && j < i) {
          float s = s_attn[i][j];
#pragma unroll
          for (int g = 0; g < NT / CH; g += 2)
            s += s_extra[g * CH + j] + s_extra[(g + 1) * CH + j];
          s_attn[i][j] = s;
        }
        __syncthreads();
      }
    }
    __syncthreads();
    if (tid < CH) s_attn[tid][tid] += 1.f;
    __syncthreads();
    // value' = attn @ (v*beta); k_cumdecay = attn @ (k*beta*exp(gcum)).
    // Thread owns one d-column (rows r = tid/128 + 2n): each v row is loaded
    // once per thread and s_attn[r][j] reads broadcast within thread pairs.
    {
      const int d = tid & (DK - 1);
      float av[CH * DK / NT], ak[CH * DK / NT];
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++) av[n] = 0.f, ak[n] = 0.f;
      // software-pipeline the global v row reads one step ahead
      float vj_nxt = (t0 < P) ? qkv[(size_t)t0 * qkvstride + 4096 + h * DK + d] : 0.f;
      for (int j = 0; j < CH; j++) {
        float vj = vj_nxt;
        if (j + 1 < CH) {
          int tn = t0 + j + 1;
          vj_nxt = (tn < P) ? qkv[(size_t)tn * qkvstride + 4096 + h * DK + d] : 0.f;
        }
        float vjb = vj * s_beta[j];
        float kjb = s_k[j][d] * s_beta[j] * s_eg[j];
        int r = tid >> 7;
#pragma unroll
        for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
          float a = s_attn[r][j];
          av[n] += a * vjb;
          ak[n] += a * kjb;
        }
      }
      int r = tid >> 7;
#pragma unroll
      for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
        wvp[r * DK + d] = av[n];
        wkcd[r * DK + d] = ak[n];
      }
    }
    __syncthreads();
    // attn2[r][j] = (q[r].k[j]) * exp(gcum[r]-gcum[j]) for j <= r (overwrite
    // s_attn); q staged in 16-row quarter tiles
    for (int q0 = 0; q0 < CH; q0 += 16) {
      for (int i = tid; i < 16 * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + q0 + t < P)
                         ? qkv[(size_t)(t0 + q0 + t) * qkvstride + kh * DK + dd]
                         : 0.f;
      }
      __syncthreads();
      for (int i = tid; i < 16 * CH; i += NT) {
        int r = q0 + i / CH, j = i % CH;
        float a = 0.f;
        if (j <= r && t0 + r < P) {
          const float* qr = &s_extra[(i / CH) * DK];
          float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
          for (int dd = 0; dd < DK; dd += 4) {
            e0 += qr[dd] * s_k[j][dd];
            e1 += qr[dd + 1] * s_k[j][dd + 1];
            e2 += qr[dd + 2] * s_k[j][dd + 2];
            e3 += qr[dd + 3] * s_k[j][dd + 3];
          }
          float dot = (e0 + e1) + (e2 + e3);
          a = dot * expf(s_gcum[r] - s_gcum[j]);
        }
        s_attn[r][j] = a;
      }
      __syncthreads();
    }
    // fused d-tile pass over S (16 columns per tile): v_new, out, S update.
    // expf(glast - gcum) reuses s_beta (dead after the v'/kcd pass)
    if (tid < CH) s_beta[tid] = expf(s_glast - s_gcum[tid]);
    float egl = expf(s_glast);
    __syncthreads();
    float* s_St = s_extra;          // [DK][16]
    float* s_vn = s_extra + 2048;   // [CH][16]
    const int dc = tid & 15;
    const int RPT = CH * 16 / NT;   // S-phase rows per thread
    const int JPT = DK * 16 / NT;   // S-phase j-rows per thread
    const int RG = NT / 16;         // row-group stride
    for (int d0 = 0; d0 < DK; d0 += 16) {
      for (int i = tid; i < DK * 16; i += NT) s_St[i] = S[(i / 16) * DK + d0 + (i % 16)];
      __syncthreads();
      // v_new[r][dc] = value'[r][d] - kcd[r] . S[:, d]  (j ascending)
      {
        // j-outer: s_St tile loads shared across the rows per thread
        int r = tid >> 4;
        const float* w0 = wkcd + r * DK;
        float e[RPT][4];
#pragma unroll
        for (int n = 0; n < RPT; n++)
#pragma unroll
          for (int u = 0; u < 4; u++) e[n][u] = 0.f;
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j * 16 + dc], s1 = s_St[(j + 1) * 16 + dc];
          float s2 = s_St[(j + 2) * 16 + dc], s3 = s_St[(j + 3) * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            float4 w = *(const float4*)(w0 + n * RG * DK + j);
            e[n][0] += w.x * s0;
            e[n][1] += w.y * s1;
            e[n][2] += w.z * s2;
            e[n][3] += w.w * s3;
          }
        }
#pragma unroll
        for (int n = 0; n < RPT; n++, r += RG)
          s_vn[r * 16 + dc] =
              wvp[r * DK + d0 + dc] - ((e[n][0] + e[n][1]) + (e[n][2] + e[n][3]));
      }
      __syncthreads();
      // out[r][d] = (q[r]*exp(gcum[r])) . S[:, d] + attn2[r] . v_new[:, d]
      {
        // j-outer with row accumulators: s_St / s_vn loads shared across rows
        int r = tid >> 4;
        const float* q0 = qkv + kh * DK;
        bool v[RPT];
        float eg[RPT], o[RPT];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          v[n] = (t0 + r + n * RG) < P;
          eg[n] = s_eg[r + n * RG];
          o[n] = 0.f;
        }
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j * 16 + dc], s1 = s_St[(j + 1) * 16 + dc];
          float s2 = s_St[(j + 2) * 16 + dc], s3 = s_St[(j + 3) * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            const float* qr = q0 + (size_t)(v[n] ? t0 + r + n * RG : t0) * qkvstride;
            float4 q4 = *(const float4*)(qr + j);
            o[n] += q4.x * s0 + q4.y * s1 + q4.z * s2 + q4.w * s3;
          }
        }
#pragma unroll
        for (int n = 0; n < RPT; n++) o[n] *= eg[n];
        for (int j = 0; j < CH; j++) {
          float vn = s_vn[j * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++)
            o[n] += ((const float*)s_attn)[(r + n * RG) * CH + j] * vn;
        }
#pragma unroll
        for (int n = 0; n < RPT; n++)
          if (v[n])
            out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] = o[n];
      }
      // S[j][d] = S[j][d]*exp(glast) + sum_r k[r][j]*exp(glast-gcum[r])*vnew[r][d]
      {
        // r-outer with j accumulators: s_vn / s_beta loads shared across j
        int j0 = tid >> 4;
        float a[JPT];
#pragma unroll
        for (int n = 0; n < JPT; n++) a[n] = 0.f;
        for (int r = 0; r < CH; r++) {
          float vn = s_vn[r * 16 + dc];
          float bt = s_beta[r];
          const float* kr = &s_k[r][0];
#pragma unroll
          for (int n = 0; n < JPT; n++) a[n] += kr[j0 + n * RG] * bt * vn;
        }
#pragma unroll
        for (int n = 0; n < JPT; n++) {
          int j = j0 + n * RG;
          S[j * DK + d0 + dc] = s_St[j * 16 + dc] * egl + a[n];
        }
      }
      __syncthreads();
    }
  }
}

// --- Phase 3b split: k_gdn_chunk as an intra/inter kernel pair --------------
// k_gdn_intra is state-independent: one block per (chunk, head), grid
// (nchunks, 48); it runs the load/cumsum/attn/forward-substitution/value'/kcd/
// attn2 phases verbatim from k_gdn_chunk and stages the chunk-local results
// fp32 in ws. k_gdn_inter keeps the per-head serial chunk scan but only runs
// the fused d-tile pass (v_new / out / S update), re-reading k and q from qkv
// (out still overwrites the v segment in place; v was fully consumed by
// k_gdn_intra). All per-element accumulation orders (j/d/r ascending) and the
// hoisted expf() values match k_gdn_chunk, so out and S are bit-identical to
// the single-kernel path. GDEC_GDN_NOSPLIT=1 selects the old kernel instead.
// Workspace layout per (chunk c, head h), base = ws + (c*48+h)*GDN_SPLIT_WS_FLOATS:
//   [0,64) gcum (inclusive; [63] = glast) | [64,4160) attn2 [64][64] |
//   [4160,12352) value' [64][128] | [12352,20544) kcd [64][128]
//   (20544 floats = 80.25 KiB per chunk-head; 16B-aligned sections)
#define GDN_WS_ATTN2 64
#define GDN_WS_VP (GDN_WS_ATTN2 + 64 * 64)
#define GDN_WS_KCD (GDN_WS_VP + 64 * 128)
#define GDN_SPLIT_WS_FLOATS (GDN_WS_KCD + 64 * 128)
// dev-only per-phase clock64 accounting (build with -DGDEC_PHASE_PROF;
// zeroed per prefill_chunk call, dumped on GDEC_PHASE_PROF env). Slots:
// 0 staging+cumsum, 1 attn dots, 2 solve, 3 vp/kcd, 4 attn2+writeout,
// 5 strip S staging, 6 strip v_new, 7 strip out, 8 strip S update,
// 9 intra total, 10 strip total (per-block cycles, tid0 atomicAdd).
#ifdef GDEC_PHASE_PROF
__device__ unsigned long long g_ph[16];
// gfx1151 has no s_memrealtime; clock64 is per-CU. Compiler barriers pin the
// reads in place, and the guard drops wrapped/preempted samples (2^64 junk
// seen with the unguarded version).
__device__ __forceinline__ long long ph_clk() {
  __asm__ volatile("" ::: "memory");
  long long v = clock64();
  __asm__ volatile("" ::: "memory");
  return v;
}
#define PH_DECL(t) long long t = ph_clk()
#define PH_ACC(slot, t)                          \
  do {                                           \
    __syncthreads();                             \
    long long e = ph_clk();                      \
    long long d = e - (t);                       \
    if (threadIdx.x == 0 && d >= 0 && d < (1LL << 40)) \
      atomicAdd(&g_ph[slot], (unsigned long long)d); \
  } while (0)
#else
#define PH_DECL(t)
#define PH_ACC(slot, t)
#endif
#ifndef GDN_NT_INTRA
// NOTE: values other than GDN_NT change the forward-substitution m-grouping
// and break bit-identity with k_gdn_chunk (still well within ktest tolerance).
#define GDN_NT_INTRA GDN_NT
#endif

// k_gdn_intra is templated on the block size; NT=GDN_NT keeps the
// forward-substitution m-grouping (and thus the results) bit-identical to
// k_gdn_chunk. (gfx1151 has only 64 KiB LDS per WGP, so the ~63 KiB block
// can never co-reside with a second one — smaller NT does not raise
// occupancy; measured 512 threads merely trades barrier cost for compute.)
template <int NT, bool WaveSolve = false, bool Ut5 = false>
__global__ void k_gdn_intra(const float* __restrict__ qkv, const float* __restrict__ gb,
                            const float* __restrict__ bb, float* __restrict__ ws, int P,
                            int qkvstride) {
  const int CH = 64, DK = 128;
  int c = blockIdx.x, h = blockIdx.y;
  int kh = h / 3;
  int tid = threadIdx.x;
  __shared__ float s_k[CH][DK + 4];  // row pad: 32-way bank conflicts -> 4-way
  __shared__ float s_attn[CH][CH];
  __shared__ float s_extra[3072];  // q quarter tile [16][128]
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH], s_glast;
  float* wsb = ws + ((size_t)c * 48 + h) * GDN_SPLIT_WS_FLOATS;
  float* wvp = wsb + GDN_WS_VP;    // value'
  float* wkcd = wsb + GDN_WS_KCD;  // k_cumdecay
  int t0 = c * CH;
  PH_DECL(tk);
  // load k tile, g, beta; per-chunk cumsum of g (pad rows: k=0, g=0, beta=0)
  for (int i = tid; i < CH * DK; i += NT) {
    int t = i / DK, d = i % DK;
    s_k[t][d] =
        (t0 + t < P) ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d] : 0.f;
  }
  if (tid < CH) {
    s_gcum[tid] = (t0 + tid < P) ? gb[(size_t)(t0 + tid) * 48 + h] : 0.f;
    s_beta[tid] = (t0 + tid < P) ? bb[(size_t)(t0 + tid) * 48 + h] : 0.f;
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0.f;
    for (int i = 0; i < CH; i++) {
      acc += s_gcum[i];
      s_gcum[i] = acc;
    }
    s_glast = s_gcum[CH - 1];
  }
  __syncthreads();
  if (tid < CH) {
    s_eg[tid] = expf(s_gcum[tid]);
    wsb[tid] = s_gcum[tid];  // stage gcum for k_gdn_inter
  }
  PH_ACC(0, tk);
  PH_DECL(t1);
  // attn[r][j] = -beta[r] * (k[r].k[j]) * exp(gcum[r]-gcum[j]) for j < r, else 0
  for (int i = tid; i < CH * CH; i += NT) {
    int r = i / CH, j = i % CH;
    float a = 0.f;
    if (j < r) {
      float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
      for (int d = 0; d < DK; d += 4) {
        e0 += s_k[r][d] * s_k[j][d];
        e1 += s_k[r][d + 1] * s_k[j][d + 1];
        e2 += s_k[r][d + 2] * s_k[j][d + 2];
        e3 += s_k[r][d + 3] * s_k[j][d + 3];
      }
      float dot = (e0 + e1) + (e2 + e3);
      a = -s_beta[r] * dot * expf(s_gcum[r] - s_gcum[j]);
    }
    s_attn[r][j] = a;
  }
  __syncthreads();
  PH_ACC(1, t1);
  PH_DECL(t2);
  if constexpr (Ut5) {
  // ut5-style blocked inverse of (I - A), A strictly lower in s_attn.
  // Diagonal 16x16 blocks: the same serial row recurrence as WaveSolve on
  // four independent 256-thread groups; off-block T entries are exact zeros
  // in the full recurrence, so diag blocks stay bit-identical. Off-diagonal
  // blocks: T_jb = D_j^-1 . (A_jb + sum_{k=b}^{j-1} A_jk T_kb) in three
  // dependency stages of 16x16x16 fp32 tile GEMMs (reassociation only).
  // s_extra layout: U tiles [0,768), T tiles [1024,2560); slot of (j,b) is
  // tl = j*(j-1)/2 + b.
  static_assert(NT == 1024, "ut5 solve uses 4x256 threads");
  {
    const int g = tid >> 8, el = tid & 255;
    const int er = el >> 4, ec = el & 15;
    {
      const int col = er, mg = ec, base = g * 16;
      float solved = 0.f;
      for (int i = 1; i < 16; ++i) {
        float p = 0.f;
        if (mg < i) p = s_attn[base + i][base + mg] * solved;
        float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
        float value = s_attn[base + i][base + col];
#pragma unroll
        for (int gg = 0; gg < 16; gg += 2)
          value += __shfl_sync(~0ull, pair, gg, 16);
        if (i == mg && col < i) solved = value;
      }
      __syncthreads();  // all warps done reading A before it is overwritten
      if (col < mg) s_attn[base + mg][base + col] = solved;
    }
    __syncthreads();
    // stage A: T_10, T_21, T_32 (U = A_jb + A_jb . S_bb)
    if (g < 3) {
      const int j = g + 1, b = g;
      float a[16];
#pragma unroll
      for (int n = 0; n < 16; n++) a[n] = s_attn[j * 16 + er][b * 16 + n];
      float u = a[ec];
#pragma unroll
      for (int n = 0; n < 16; n++)
        u += a[n] * s_attn[b * 16 + n][b * 16 + ec];
      s_extra[g * 256 + el] = u;
    }
    __syncthreads();
    if (g < 3) {
      const int j = g + 1, b = g;
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[j * 16 + er][j * 16 + m];
      float t = s_extra[g * 256 + el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[g * 256 + m * 16 + ec];
      s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
    }
    __syncthreads();
    // stage B: T_20, T_31 (U = A_jb + A_jb . S_bb + A_j,b+1 . T_b+1,b)
    if (g < 2) {
      const int j = g + 2, b = g;
      float a0[16], a1[16];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        a0[n] = s_attn[j * 16 + er][b * 16 + n];
        a1[n] = s_attn[j * 16 + er][(b + 1) * 16 + n];
      }
      float u = a0[ec];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        u += a0[n] * s_attn[b * 16 + n][b * 16 + ec];
        u += a1[n] * s_extra[1024 + ((b + 1) * b / 2 + b) * 256 + n * 16 + ec];
      }
      s_extra[g * 256 + el] = u;
    }
    __syncthreads();
    if (g < 2) {
      const int j = g + 2, b = g;
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[j * 16 + er][j * 16 + m];
      float t = s_extra[g * 256 + el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[g * 256 + m * 16 + ec];
      s_extra[1024 + (j * (j - 1) / 2 + b) * 256 + el] = t;
    }
    __syncthreads();
    // stage C: T_30 (U = A_30 + A_30.S_00 + A_31.T_10 + A_32.T_20)
    if (g == 0) {
      float a0[16], a1[16], a2[16];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        a0[n] = s_attn[48 + er][n];
        a1[n] = s_attn[48 + er][16 + n];
        a2[n] = s_attn[48 + er][32 + n];
      }
      float u = a0[ec];
#pragma unroll
      for (int n = 0; n < 16; n++) {
        u += a0[n] * s_attn[n][ec];
        u += a1[n] * s_extra[1024 + n * 16 + ec];        // T_10
        u += a2[n] * s_extra[1024 + 256 + n * 16 + ec];  // T_20
      }
      s_extra[el] = u;
    }
    __syncthreads();
    if (g == 0) {
      float sj[16];
#pragma unroll
      for (int m = 0; m < 16; m++) sj[m] = s_attn[48 + er][48 + m];
      float t = s_extra[el];
#pragma unroll
      for (int m = 0; m < 16; m++)
        t += sj[m] * s_extra[m * 16 + ec];
      s_extra[1024 + 3 * 256 + el] = t;
    }
    __syncthreads();
    // publish the off-diagonal blocks into s_attn
    for (int j = 1; j < 4; j++)
      for (int b = 0; b < j; b++) {
        const int tl = j * (j - 1) / 2 + b;
        for (int e = tid; e < 256; e += NT)
          s_attn[j * 16 + (e >> 4)][b * 16 + (e & 15)] =
              s_extra[1024 + tl * 256 + e];
      }
  }

  } else if constexpr (WaveSolve) {
  // Each 16-lane group owns a column of the triangular inverse. Lane g
  // keeps rows g,g+16,g+32,g+48 in registers. Preserve the old 16 partial
  // sums and its sequential pair merges, without publishing every row in LDS.
  static_assert(NT == 1024, "the bit-exact solve uses 16 m-groups");
  {
    const int col = tid / 16, mg = tid % 16;
    float solved[4] = {};
    for (int i = 1; i < CH; ++i) {
      float p = 0.f;
#pragma unroll
      for (int slot = 0; slot < 4; ++slot) {
        int m = mg + 16 * slot;
        if (m < i) p += s_attn[i][m] * solved[slot];
      }
      float pair = p + __shfl_xor_sync(~0ull, p, 1, 16);
      float value = s_attn[i][col];
#pragma unroll
      for (int g = 0; g < 16; g += 2)
        value += __shfl_sync(~0ull, pair, g, 16);
#pragma unroll
      for (int slot = 0; slot < 4; ++slot)
        if (i == mg + 16 * slot && col < i) solved[slot] = value;
    }
    // The attn2 workspace is not live yet. Use it as a transpose buffer;
    // all waves must finish reading the original matrix before LDS changes.
#pragma unroll
    for (int slot = 0; slot < 4; ++slot)
      wsb[GDN_WS_ATTN2 + col * CH + mg + 16 * slot] = solved[slot];
    __syncthreads();
    for (int i = tid; i < CH * CH; i += NT)
      s_attn[i / CH][i % CH] = wsb[GDN_WS_ATTN2 + (i % CH) * CH + i / CH];
  }

  } else {
  // forward substitution: attn[i,:i] += sum_m attn[i,m](old) * attn[m,:i](updated).
  // Rows are serially dependent; one warp processes them with lanes over
  // columns (m ascending per element — same order as the naive loop).
  {
    // NT threads = NT/64 m-groups (tid>>6) x 64 columns (tid&63); per row each
    // group sums its strided m range into s_extra, then group 0 merges.
    const int j = tid & (CH - 1), mg = tid >> 6;
    for (int i = 1; i < CH; i++) {
      float p = 0.f;
      for (int m = mg; m < i; m += NT / CH) p += s_attn[i][m] * s_attn[m][j];
      s_extra[mg * CH + j] = p;
      __syncthreads();
      if (mg == 0 && j < i) {
        float s = s_attn[i][j];
#pragma unroll
        for (int g = 0; g < NT / CH; g += 2)
          s += s_extra[g * CH + j] + s_extra[(g + 1) * CH + j];
        s_attn[i][j] = s;
      }
      __syncthreads();
    }
  }

  }
  __syncthreads();
  PH_ACC(2, t2);
  PH_DECL(t3);
  if (tid < CH) s_attn[tid][tid] += 1.f;
  __syncthreads();
  // value' = attn @ (v*beta); k_cumdecay = attn @ (k*beta*exp(gcum)).
  // Thread owns one d-column (rows r = tid/128 + 2n). v rows are staged
  // through s_extra in 16-row bands (s_extra is dead between the substitution
  // and the attn2 q staging) so the 64 serial global-read latencies collapse
  // into four coalesced band loads; s_attn[r][j] reads broadcast within
  // thread pairs. Values and accumulation order are unchanged.
  {
    const int d = tid & (DK - 1);
    float av[CH * DK / NT], ak[CH * DK / NT];
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++) av[n] = 0.f, ak[n] = 0.f;
    const int BAND = 16;
    for (int j0 = 0; j0 < CH; j0 += BAND) {
      for (int i = tid; i < BAND * DK; i += NT) {
        int t = i / DK, dd = i % DK;
        s_extra[i] = (t0 + j0 + t < P)
                         ? qkv[(size_t)(t0 + j0 + t) * qkvstride + 4096 + h * DK + dd]
                         : 0.f;
      }
      __syncthreads();
      for (int jj = 0; jj < BAND; jj++) {
        int j = j0 + jj;
        float vjb = s_extra[jj * DK + d] * s_beta[j];
        float kjb = s_k[j][d] * s_beta[j] * s_eg[j];
        int r = tid >> 7;
#pragma unroll
        for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
          float a = s_attn[r][j];
          av[n] += a * vjb;
          ak[n] += a * kjb;
        }
      }
      __syncthreads();
    }
    int r = tid >> 7;
#pragma unroll
    for (int n = 0; n < CH * DK / NT; n++, r += NT / DK) {
      wvp[r * DK + d] = av[n];
      wkcd[r * DK + d] = ak[n];
    }
  }
  __syncthreads();
  PH_ACC(3, t3);
  PH_DECL(t4);
  // attn2[r][j] = (q[r].k[j]) * exp(gcum[r]-gcum[j]) for j <= r (overwrite
  // s_attn); q staged in 16-row quarter tiles
  for (int q0 = 0; q0 < CH; q0 += 16) {
    for (int i = tid; i < 16 * DK; i += NT) {
      int t = i / DK, dd = i % DK;
      s_extra[i] = (t0 + q0 + t < P)
                       ? qkv[(size_t)(t0 + q0 + t) * qkvstride + kh * DK + dd]
                       : 0.f;
    }
    __syncthreads();
    for (int i = tid; i < 16 * CH; i += NT) {
      int r = q0 + i / CH, j = i % CH;
      float a = 0.f;
      if (j <= r && t0 + r < P) {
        const float* qr = &s_extra[(i / CH) * DK];
        float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        for (int dd = 0; dd < DK; dd += 4) {
          e0 += qr[dd] * s_k[j][dd];
          e1 += qr[dd + 1] * s_k[j][dd + 1];
          e2 += qr[dd + 2] * s_k[j][dd + 2];
          e3 += qr[dd + 3] * s_k[j][dd + 3];
        }
        float dot = (e0 + e1) + (e2 + e3);
        a = dot * expf(s_gcum[r] - s_gcum[j]);
      }
      s_attn[r][j] = a;
    }
    __syncthreads();
  }
  // stage attn2 for k_gdn_inter
  for (int i = tid; i < CH * CH; i += NT) wsb[GDN_WS_ATTN2 + i] = ((const float*)s_attn)[i];
  PH_ACC(4, t4);
  PH_ACC(9, tk);
}

#ifndef GDN_NT_INTER
#define GDN_NT_INTER GDN_NT  // k_gdn_inter block size (bit-exact at any NT)
#endif

template <int NT>
__global__ void k_gdn_inter(const float* __restrict__ qkv, float* __restrict__ S,
                            float* __restrict__ out, const float* __restrict__ ws,
                            int P, int qkvstride) {
  const int CH = 64, DK = 128;
  int h = blockIdx.x;
  int kh = h / 3;
  int tid = threadIdx.x;
  __shared__ float s_k[CH][DK + 4];  // row pad: 32-way bank conflicts -> 4-way
  __shared__ float s_attn[CH][CH];   // attn2 staged from ws
  __shared__ float s_extra[3072];    // S tile [128][16] + vnew [64][16]
  __shared__ float s_gcum[CH], s_beta[CH], s_eg[CH];
  // s_beta holds expf(glast-gcum); glast = gcum[CH-1]
  S += (size_t)h * DK * DK;
  int nchunks = (P + CH - 1) / CH;
  for (int c = 0; c < nchunks; c++) {
    int t0 = c * CH;
    const float* wsb = ws + ((size_t)c * 48 + h) * GDN_SPLIT_WS_FLOATS;
    const float* wvp = wsb + GDN_WS_VP;    // value'
    const float* wkcd = wsb + GDN_WS_KCD;  // k_cumdecay
    // stage k rows (S update; pad rows: k=0), gcum, attn2 and the first S tile
    for (int i = tid; i < CH * DK; i += NT) {
      int t = i / DK, d = i % DK;
      s_k[t][d] =
          (t0 + t < P) ? qkv[(size_t)(t0 + t) * qkvstride + 2048 + kh * DK + d] : 0.f;
    }
    if (tid < CH) s_gcum[tid] = wsb[tid];
    for (int i = tid; i < CH * CH; i += NT)
      ((float*)s_attn)[i] = wsb[GDN_WS_ATTN2 + i];
    for (int i = tid; i < DK * 16; i += NT)
      s_extra[i] = S[(i / 16) * DK + (i % 16)];  // S tile d0=0
    __syncthreads();
    // fused d-tile pass over S (16 columns per tile): v_new, out, S update.
    if (tid < CH) {
      s_eg[tid] = expf(s_gcum[tid]);
      s_beta[tid] = expf(s_gcum[CH - 1] - s_gcum[tid]);  // expf(glast - gcum)
    }
    float egl = expf(s_gcum[CH - 1]);  // expf(glast)
    __syncthreads();
    float* s_St = s_extra;          // [DK][16]
    float* s_vn = s_extra + 2048;   // [CH][16]
    const int dc = tid & 15;
    const int RPT = CH * 16 / NT;   // S-phase rows per thread
    const int JPT = DK * 16 / NT;   // S-phase j-rows per thread
    const int RG = NT / 16;         // row-group stride
    for (int d0 = 0; d0 < DK; d0 += 16) {
      // prefetch the next S tile into registers; the loads overlap this
      // tile's compute (disjoint columns, no hazard with the S update below)
      float pf[JPT];
      if (d0 + 16 < DK) {
        int i = tid;
#pragma unroll
        for (int n = 0; n < JPT; n++, i += NT)
          pf[n] = S[(i / 16) * DK + d0 + 16 + (i % 16)];
      }
      // v_new[r][dc] = value'[r][d] - kcd[r] . S[:, d]  (j ascending)
      {
        // j-outer: s_St tile loads shared across the rows per thread
        int r = tid >> 4;
        const float* w0 = wkcd + r * DK;
        float e[RPT][4];
#pragma unroll
        for (int n = 0; n < RPT; n++)
#pragma unroll
          for (int u = 0; u < 4; u++) e[n][u] = 0.f;
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j * 16 + dc], s1 = s_St[(j + 1) * 16 + dc];
          float s2 = s_St[(j + 2) * 16 + dc], s3 = s_St[(j + 3) * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            float4 w = *(const float4*)(w0 + n * RG * DK + j);
            e[n][0] += w.x * s0;
            e[n][1] += w.y * s1;
            e[n][2] += w.z * s2;
            e[n][3] += w.w * s3;
          }
        }
#pragma unroll
        for (int n = 0; n < RPT; n++, r += RG)
          s_vn[r * 16 + dc] =
              wvp[r * DK + d0 + dc] - ((e[n][0] + e[n][1]) + (e[n][2] + e[n][3]));
      }
      __syncthreads();
      // out[r][d] = (q[r]*exp(gcum[r])) . S[:, d] + attn2[r] . v_new[:, d]
      {
        // j-outer with row accumulators: s_St / s_vn loads shared across rows
        int r = tid >> 4;
        const float* q0 = qkv + kh * DK;
        bool v[RPT];
        float eg[RPT], o[RPT];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          v[n] = (t0 + r + n * RG) < P;
          eg[n] = s_eg[r + n * RG];
          o[n] = 0.f;
        }
        for (int j = 0; j < DK; j += 4) {
          float s0 = s_St[j * 16 + dc], s1 = s_St[(j + 1) * 16 + dc];
          float s2 = s_St[(j + 2) * 16 + dc], s3 = s_St[(j + 3) * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++) {
            const float* qr = q0 + (size_t)(v[n] ? t0 + r + n * RG : t0) * qkvstride;
            float4 q4 = *(const float4*)(qr + j);
            o[n] += q4.x * s0 + q4.y * s1 + q4.z * s2 + q4.w * s3;
          }
        }
#pragma unroll
        for (int n = 0; n < RPT; n++) o[n] *= eg[n];
        for (int j = 0; j < CH; j++) {
          float vn = s_vn[j * 16 + dc];
#pragma unroll
          for (int n = 0; n < RPT; n++)
            o[n] += ((const float*)s_attn)[(r + n * RG) * CH + j] * vn;
        }
#pragma unroll
        for (int n = 0; n < RPT; n++)
          if (v[n])
            out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] = o[n];
      }
      // S[j][d] = S[j][d]*exp(glast) + sum_r k[r][j]*exp(glast-gcum[r])*vnew[r][d]
      {
        // r-outer with j accumulators: s_vn / s_beta loads shared across j
        int j0 = tid >> 4;
        float a[JPT];
#pragma unroll
        for (int n = 0; n < JPT; n++) a[n] = 0.f;
        for (int r = 0; r < CH; r++) {
          float vn = s_vn[r * 16 + dc];
          float bt = s_beta[r];
          const float* kr = &s_k[r][0];
#pragma unroll
          for (int n = 0; n < JPT; n++) a[n] += kr[j0 + n * RG] * bt * vn;
        }
#pragma unroll
        for (int n = 0; n < JPT; n++) {
          int j = j0 + n * RG;
          S[j * DK + d0 + dc] = s_St[j * 16 + dc] * egl + a[n];
        }
      }
      __syncthreads();  // all reads of the current S tile are done
      // commit the prefetched tile over the dead one, then re-sync
      if (d0 + 16 < DK) {
        int i = tid;
#pragma unroll
        for (int n = 0; n < JPT; n++, i += NT) s_St[i] = pf[n];
      }
      __syncthreads();
    }
  }
}

// --- Phase 3b-strip: k_gdn_inter with the S v-dimension split into 8 strips.
// grid (8 strips, 48 heads); block (strip s, head h) owns S columns
// [16s, 16s+16) for the whole chunk scan. The strip's S tile [128][16] stays
// resident in LDS (loaded once, written back once; nonzero initial state
// supported), k/attn2 are re-read from global instead of staged, so a block
// needs ~13 KiB LDS and 4 blocks co-reside per CU (vs 1 for k_gdn_inter).
// Every output element keeps k_gdn_inter's accumulation order exactly
// (v_new/out: j ascending, 4-grouped (e0+e1)+(e2+e3); S update: r ascending,
// single thread per element; eg/beta/egl from the same ws gcum values), so
// out and S are bit-identical to k_gdn_inter and k_gdn_chunk. 3 barriers per
// chunk. Selected by default; GDEC_GDN_NOSTRIP=1 restores k_gdn_inter.
#define GDN_NT_STRIP 256

__global__ void __launch_bounds__(GDN_NT_STRIP)
    k_gdn_inter_strip(const float* __restrict__ qkv, float* __restrict__ S,
                      float* __restrict__ out, const float* __restrict__ ws, int P,
                      int qkvstride) {
  const int CH = 64, DK = 128, NT = GDN_NT_STRIP;
  const int h = blockIdx.y;
  const int kh = h / 3;
  const int tid = threadIdx.x;
  const int d0 = blockIdx.x * 32;  // strip base column (32-wide: 4 blocks/head)
  __shared__ float s_St[DK][32];   // resident S strip (16 KiB)
  __shared__ float s_vn[CH][32];   // v_new (8 KiB)
  __shared__ float s_eg[CH], s_beta[CH];
  S += (size_t)h * DK * DK;
  PH_DECL(tk);
  // load the strip's initial S (may be a nonzero state)
  for (int i = tid; i < DK * 32; i += NT)
    s_St[i / 32][i % 32] = S[(i / 32) * DK + d0 + (i % 32)];
  __syncthreads();
  PH_ACC(5, tk);
  const int nchunks = (P + CH - 1) / CH;
  const int dc = tid & 31;
  const int RG = NT / 32;        // row-group stride (8)
  const int RPT = CH * 32 / NT;  // rows per thread in the v_new/out phases (8)
  for (int c = 0; c < nchunks; c++) {
    const int t0 = c * CH;
    const float* wsb = ws + ((size_t)c * 48 + h) * GDN_SPLIT_WS_FLOATS;
    const float* wvp = wsb + GDN_WS_VP;    // value'
    const float* wkcd = wsb + GDN_WS_KCD;  // k_cumdecay
    const float* wattn2 = wsb + GDN_WS_ATTN2;
    // gates straight from ws (no staging); consumed after the v_new barrier
    if (tid < CH) {
      float gci = wsb[tid];
      s_eg[tid] = expf(gci);
      s_beta[tid] = expf(wsb[CH - 1] - gci);  // expf(glast - gcum)
    }
    const float egl = expf(wsb[CH - 1]);  // expf(glast)
    PH_DECL(t6);
    // v_new[r][dc] = value'[r][d0+dc] - kcd[r] . S[:, d0+dc]  (j ascending)
    {
      int r = tid >> 5;
      const float* w0 = wkcd + r * DK;
      float e[RPT][4];
#pragma unroll
      for (int n = 0; n < RPT; n++)
#pragma unroll
        for (int u = 0; u < 4; u++) e[n][u] = 0.f;
      for (int j = 0; j < DK; j += 4) {
        float s0 = s_St[j][dc], s1 = s_St[j + 1][dc];
        float s2 = s_St[j + 2][dc], s3 = s_St[j + 3][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          float4 w = *(const float4*)(w0 + n * RG * DK + j);
          e[n][0] += w.x * s0;
          e[n][1] += w.y * s1;
          e[n][2] += w.z * s2;
          e[n][3] += w.w * s3;
        }
      }
#pragma unroll
      for (int n = 0; n < RPT; n++, r += RG)
        s_vn[r][dc] =
            wvp[r * DK + d0 + dc] - ((e[n][0] + e[n][1]) + (e[n][2] + e[n][3]));
    }
    __syncthreads();  // s_vn / s_eg / s_beta ready
    PH_ACC(6, t6);
    PH_DECL(t7);
    // out[r][d0+dc] = (q[r]*exp(gcum[r])) . S[:, d0+dc] + attn2[r] . v_new[:, dc]
    {
      int r = tid >> 5;
      const float* q0 = qkv + kh * DK;
      bool v[RPT];
      float eg[RPT], o[RPT];
#pragma unroll
      for (int n = 0; n < RPT; n++) {
        v[n] = (t0 + r + n * RG) < P;
        eg[n] = s_eg[r + n * RG];
        o[n] = 0.f;
      }
      for (int j = 0; j < DK; j += 4) {
        float s0 = s_St[j][dc], s1 = s_St[j + 1][dc];
        float s2 = s_St[j + 2][dc], s3 = s_St[j + 3][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++) {
          const float* qr = q0 + (size_t)(v[n] ? t0 + r + n * RG : t0) * qkvstride;
          float4 q4 = *(const float4*)(qr + j);
          o[n] += q4.x * s0 + q4.y * s1 + q4.z * s2 + q4.w * s3;
        }
      }
#pragma unroll
      for (int n = 0; n < RPT; n++) o[n] *= eg[n];
      for (int j = 0; j < CH; j++) {
        float vn = s_vn[j][dc];
#pragma unroll
        for (int n = 0; n < RPT; n++)
          o[n] += wattn2[(r + n * RG) * CH + j] * vn;
      }
#pragma unroll
      for (int n = 0; n < RPT; n++)
        if (v[n])
          out[(size_t)(t0 + r + n * RG) * qkvstride + 4096 + h * DK + d0 + dc] = o[n];
    }
    __syncthreads();  // all reads of the old S strip are done
    PH_ACC(7, t7);
    PH_DECL(t8);
    // S[j][d0+dc] = S*egl + sum_r k[r][j]*exp(glast-gcum[r])*vnew[r][dc]
    // (r ascending, single thread per element; lane holds 4 consecutive j so
    // the k reads coalesce into float4)
    {
      const int jg = tid >> 3;        // j = 4*jg .. 4*jg+3
      const int dc2 = (tid & 7) * 4;  // dc = dc2 .. dc2+3
      float a[4][4];
#pragma unroll
      for (int jj = 0; jj < 4; jj++)
#pragma unroll
        for (int u = 0; u < 4; u++) a[jj][u] = 0.f;
      for (int r = 0; r < CH; r++) {
        float4 k4 = {0.f, 0.f, 0.f, 0.f};  // tail chunks: k zero-padded
        if (t0 + r < P)
          k4 = *(const float4*)(qkv + (size_t)(t0 + r) * qkvstride + 2048 +
                                kh * DK + jg * 4);
        const float bt = s_beta[r];
        const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const float vn = s_vn[r][dc2 + u];
#pragma unroll
          for (int jj = 0; jj < 4; jj++) a[jj][u] += kv[jj] * bt * vn;
        }
      }
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const int j = jg * 4 + jj;
#pragma unroll
        for (int u = 0; u < 4; u++)
          s_St[j][dc2 + u] = s_St[j][dc2 + u] * egl + a[jj][u];
      }
    }
    __syncthreads();  // S strip updated; s_vn reusable next chunk
    PH_ACC(8, t8);
  }
  // write back the strip's final state
  for (int i = tid; i < DK * 32; i += NT)
    S[(i / 32) * DK + d0 + (i % 32)] = s_St[i / 32][i % 32];
  PH_ACC(10, tk);
}

// ============================ host side =====================================

// --- Windows 单 arena 分配器 ------------------------------------------------
// Windows/ROCm 路由规则（tools/winprobe 实测）：进程累计中小额分配 >32 GiB 后
// 新分配会被路由到共享内存（主机 RAM，越界即灾难）；而单次大额 hipMalloc
// （实测 40-95 GiB）整体落 VRAM。且 postarena 探针证明：大额 arena 之后的
// 小额 hipMalloc 仍然安全。所以 Windows 下全部持久设备内存来自一次大额
// hipMalloc（g_devarena），bump 分配；溢出时回退 hipMalloc（小额、安全）。
// Linux 行为不变：DALLOC 就是 hipMalloc。
#ifdef _WIN32
static void* g_devarena = nullptr;
static size_t g_devarena_sz = 0, g_devarena_off = 0;
static hipError_t dalloc_arena(void** p, size_t bytes) {
  if (!g_devarena) return hipMalloc(p, bytes);
  size_t off = (g_devarena_off + 255) & ~(size_t)255;
  if (off + bytes > g_devarena_sz) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      fprintf(stderr,
              "arena: capacity %.2f GiB exhausted at +%zu B; further allocations "
              "fall back to hipMalloc\n",
              g_devarena_sz / 1073741824.0, bytes);
    }
    return hipMalloc(p, bytes);
  }
  *p = (char*)g_devarena + off;
  g_devarena_off = off + bytes;
  return hipSuccess;
}
#define DALLOC(p, s) CK(dalloc_arena((void**)(p), (size_t)(s)))
#else
#define DALLOC(p, s) CK(hipMalloc(p, s))
#endif

// 与 devarena_estimate 共享的 maxbatch 决策。32K 是 Linux 上实测能与 68 GiB
// 模型及长上下文 KV 共存的最大批（GDEC_PREFILL_CHUNK 可压低）；Windows 受
// 96 GiB VRAM 硬门槛限制默认 8192（见 PORTING-WINDOWS.md）。
static int eff_maxbatch(int maxctx) {
  if (const char* e = getenv("GDEC_PREFILL_CHUNK")) {
    int v = atoi(e);
    if (v > 0) return std::min(maxctx, v);
    return maxctx;
  }
#ifdef _WIN32
  return std::min(maxctx, 8192);
#else
  return std::min(maxctx, 32768);
#endif
}

// 尾包合并 slack：prompt 切成 maxbatch chunk 后若最后只剩 ≤ slack 个
// token，则并入前一个 chunk（该 chunk 最大 maxbatch+slack）。workspace
// 按 maxbatch+slack 分配。0 禁用。Windows arena 有 95 GiB 硬顶，默认关。
static int eff_tail_slack() {
  if (const char* e = getenv("GDEC_PREFILL_TAIL_SLACK"))
    return std::max(0, atoi(e));
#ifdef _WIN32
  return 0;
#else
  return 1024;
#endif
}

// 镜像 GpuModel 构造函数的分配公式（含相同 env 分支），用于 arena 精确 sizing。
// 每项 256B 对齐后求和；估少了一律有 dalloc_arena 的 hipMalloc 回退兜底。
static size_t devarena_estimate(const Checkpoint& ck, int maxctx) {
  size_t total = 0;
  auto add = [&](size_t b) { total += (b + 255) & ~(size_t)255; };
  // 权重：Windows 下专家也进 arena（无 hipHostRegister 直读；PLE 表/配置留盘）
  for (auto& kv : ck.tensors()) {
    const Tensor& t = kv.second;
    if (t.dtype == 10 || t.dtype == 4) continue;
    add(t.data_size);
  }
  const bool qsa_dense = getenv("GDEC_QSA_DENSE") != nullptr;
  const bool kv_fp32 = getenv("GDEC_QSA_KV_BF16") == nullptr;
  const bool index_f32 = getenv("GDEC_INDEX_BF16") == nullptr;
  const bool btv = getenv("GDEC_QSA_WMMA") && !kv_fp32 &&
                   getenv("GDEC_QSA_WMMA_BTV");
  const bool fused = getenv("GDEC_PREFILL_UNFUSED") == nullptr;
  const bool moe_det = !qsa_dense && getenv("GDEC_MOE_ATOMIC") == nullptr;
  const bool moe_lt_ok = getenv("GDEC_MOE_LT") && g_cfg.topk == 10 && moe_det &&
                         !getenv("GDEC_MOE_HOST_ROUTE") &&
                         !getenv("GDEC_MOE_ORDERED");
  const bool lt_cmp = getenv("GDEC_MOE_LT_CMP") != nullptr;
  const bool lt_ovl = getenv("GDEC_MOE_LT_OVL") != nullptr;
  const bool gdn_split =
      !getenv("GDEC_GDN_NOSPLIT") && !getenv("GDEC_GDN_LOOP");
  const bool gdn_str = getenv("GDEC_GDN_STREAM") != nullptr;
  const int gdn_window_chunks = [] {
    const char* e = getenv("GDEC_GDN_WINDOW_CHUNKS");
    return e ? std::max(1, atoi(e)) : 4;
  }();
  const size_t mc = (size_t)maxctx,
               BP = (size_t)(eff_maxbatch(maxctx) + eff_tail_slack());
  const size_t ngdn = g_cfg.layers - g_cfg.layers / 4, nqsa = g_cfg.layers / 4;
  const size_t ib = (mc + 3) / 4, rb = (size_t)g_cfg.branches * g_cfg.d;
  add((size_t)64 << 20);  // decode 单行小缓冲合计（d_R..d_logits 等）
  add(ngdn * 48 * 128 * 128 * 4);           // d_S
  add(ngdn * 3 * 10240 * 4);                // d_convst
  add(ngdn * 10240 * 4 * 4);                // d_convw
  add((size_t)g_cfg.vocab * 4 * 65);        // d_logits_b
  add(ngdn * 48 * 128 * 128 * 4 + ngdn * 3 * 10240 * 4 + 9 * 10240 * 4 +
      (qsa_dense ? 0 : nqsa * 4 * 128 * 4));  // d_ckpt
  add((size_t)48 * 2 * 64 * 128 * 4);       // d_gdn_ws
  if (gdn_split)
    add((gdn_str ? std::min<size_t>(gdn_window_chunks, (BP + 63) / 64)
                 : (BP + 63) / 64) * 48 *
        (size_t)GDN_SPLIT_WS_FLOATS * 4);
  add((size_t)64 << 20);                    // d_wbf16
  add((size_t)64 << 20);                    // d_ltws
  if (!qsa_dense) {
    add(mc * 640 * 4);                      // d_iproj
    add(mc * 512 * 4);                      // d_iq
    add(nqsa * ib * 128 * 4);               // d_ik
    add(nqsa * 4 * 128 * 4);                // d_iraw
    add(nqsa * 256 * 4);                    // d_inorm
    if (index_f32) add(nqsa * 640 * 2560 * 4);  // d_iw
    add((size_t)std::min(mc, (size_t)INDEX_BATCH) * 4 * ib * 4);  // d_iscores
    add(mc * INDEX_BUDGET * 4);             // d_iselected
  }
  if (kv_fp32) {
    add(2 * nqsa * mc * 512 * 4);           // d_kc + d_vc
  } else {
    add(2 * nqsa * mc * 512 * 2);           // d_kcb + d_vcb
    if (btv) add(nqsa * ib * 2 * 256 * 4 * 2);  // d_vcbt
  }
  add(2 * mc * 4);                          // d_posarr + d_ringarr
  // prefill 批缓冲（BP 维）
  add(10 * BP * 10240 * 4);   // Rb Rhatb Gb qkvb convb keysb qnb Ub Unb convoutb
  add(6 * BP * 2560 * 4);     // xb yb moeaccb eyb eb plevb
  add(BP * 320 * 4);          // t320b
  add(BP * 6144 * 4 * 2);     // zb attnb
  add(BP * 48 * 4 * 2);       // g48b beta48b
  add(BP * 12288 * 4);        // qgb
  add(BP * 6144 * 4 * 2);     // qsb gsb
  add(BP * 512 * 4 * 3);      // kbb vbb routerb
  add(BP * 32 * sizeof(float2));  // ropecs
  add(BP * 4 * 4 + BP * 16 * 4 * 2 + BP * 4);  // w4b topidsb topwb sgb
  add(BP * g_cfg.topk * (moe_det ? 2560 : 1280) * 4);  // guvb
  add(BP * g_cfg.topk * 640 * 4);                    // hidb
  add(BP * 10240 * 2 + BP * 6144 * 2);               // xbf16 xbf16b
  if (fused) add(BP * 10240 * 2);                    // Rhatbf16
  add(BP * g_cfg.topk * 4 * 2);                      // tokidx pweight
  add((g_cfg.experts + 1) * 4);                      // eoff
  if (moe_det) add(BP * g_cfg.topk * 4);             // pairids
  add(4 * BP * g_cfg.topk * 4 + ((size_t)8 << 20));  // moe_routing keys + scratch
  add(((BP * g_cfg.topk + 63) / 64 + g_cfg.experts) * sizeof(MoeTile) + 4);  // moetiles
  if (moe_lt_ok) {
    add(BP * g_cfg.topk * g_cfg.d * 2);         // moexg
    add((size_t)4 * 2 * g_cfg.moe_mid * g_cfg.d * 2 * (lt_ovl ? 2 : 1));  // moewup
    add((size_t)4 * g_cfg.d * g_cfg.moe_mid * 2 * (lt_ovl ? 2 : 1));      // moewdn
    if (lt_cmp) {
      add(BP * g_cfg.topk * g_cfg.moe_mid * 2);  // hidb2
      add(BP * g_cfg.topk * g_cfg.d * 4);        // pairs2
      add(8);                                    // cmpi
    }
  }
  add(BP * 4 + BP * sizeof(int2));            // tokarr img_inj
  // MTP draft head（构造函数同款开启条件）
  const bool moe_default = moe_det && !getenv("GDEC_MOE_ORDERED") &&
                           !getenv("GDEC_MOE_HOST_ROUTE") &&
                           !getenv("GDEC_MOE_UNTILED") &&
                           !getenv("GDEC_MOE_FP32_IO");
  if (ck.find("mtp.fc_hidden.weight") && moe_default) {
    add(rb * 4 * 3 + (size_t)g_cfg.d * 4 * 5 + 8);  // mR mRhat mh me meo mt mnorm_*
    add(2 * mc * 512 * (kv_fp32 ? 4 : 2));          // MTP KV
    add(3 * BP * 10240 * 4);                        // mRb mRhatb mtap
    add(512 * 4);                                   // mring_ckpt
    add(9 * ngdn * 48 * 128 * 128 * 4);             // Ssnap
    add(9 * ngdn * 3 * 10240 * 4);                  // csnap
    add(nqsa * 9 * 128 * 4);                        // iproj_snap
    if (!qsa_dense) {
      add(ib * 128 * 4 + 4 * 128 * 4 + 256 * 4);    // mik miraw minorm
      if (index_f32) add((size_t)640 * 2560 * 4);   // miw
    }
  }
  add((size_t)1536 << 20);  // vision tower 权重（dup_dev 进 arena）约 1.5 GiB
  return total;
}

// Windows：在 load_arena 之前一次性 hipMalloc 整个 arena（大额单次才进 VRAM）。
static void devarena_init(const Checkpoint& ck, int maxctx) {
#ifdef _WIN32
  size_t need = devarena_estimate(ck, maxctx);
  double slack_gb = 4.0, cap_gb = 95.0;  // bigalloc 实测 95 GiB 成功 / 96 失败
  if (const char* e = getenv("GDEC_ARENA_SLACK_GB")) slack_gb = atof(e);
  if (const char* e = getenv("GDEC_ARENA_CAP_GB")) cap_gb = atof(e);
  size_t target = need + (size_t)(slack_gb * 1073741824.0);
  size_t cap = (size_t)(cap_gb * 1073741824.0);
  if (target > cap) {
    fprintf(stderr,
            "arena: estimate %.2f + slack %.2f GiB exceeds cap %.2f GiB; "
            "clamping (overflow falls back to hipMalloc)\n",
            need / 1073741824.0, slack_gb, cap_gb);
    target = cap;
  }
  CK(hipMalloc(&g_devarena, target));
  g_devarena_sz = target;
  fprintf(stderr, "devarena: %.2f GiB reserved (estimate %.2f GiB)\n",
          target / 1073741824.0, need / 1073741824.0);
#else
  (void)ck;
  (void)maxctx;
#endif
}

// --- weight arena: copy trunk tensors (except the 47.7 GiB PLE table and the
// tiny u64 config arrays) into one big device allocation at startup ----------
static std::unordered_map<std::string, const uint8_t*> g_devtensor;

// parallel loader: N worker threads memcpy page-cache -> pinned staging -> H2D
static void load_arena(const Checkpoint& ck) {
  // Expert weights (~65 GiB) never enter the arena: they are hipHostRegister'd
  // per tensor and kernels read the host pointers directly (gfx1151 unified
  // memory, 220 GB/s measured, same DRAM as GTT, see tools/tgv2_probe*.cu).
  // Registration is mandatory: direct reads of non-resident pages fault.
  // Per-tensor registration of 65 GiB takes ~2.6 s with no GTT shadow.
  // MoE sparsity (10/512 experts per token) makes the sparse reads cheap, and
  // TG/PP measured at parity with the old all-in-arena form, which is gone.
  // Windows: hipHostRegister 这条直读路不存在（65 GiB pin 进 32 GiB RAM 物理
  // 不可能）；专家与普通张量一样拷贝进 devarena，VRAM 直读（同一片 DRAM）。
  auto is_experts = [](const std::string& n) {
    return n.find(".mlp.experts.") != std::string::npos;
  };
  size_t total = 0, xbytes = 0;
  for (auto& kv : ck.tensors()) {
    const Tensor& t = kv.second;
    if (t.dtype == 10 || t.dtype == 4) continue;  // PLE table / u64 config
#ifndef _WIN32
    if (is_experts(t.name)) {
      hipError_t re =
          hipHostRegister((void*)t.data, t.data_size, hipHostRegisterMapped);
      if (re != hipSuccess)
        throw std::runtime_error(std::string("hipHostRegister ") + t.name +
                                 ": " + hipGetErrorString(re));
      g_devtensor[t.name] = t.data;  // 注册后 host 指针即设备指针,kernel 直读
      xbytes += t.data_size;
      continue;
    }
#endif
    total += (t.data_size + 255) & ~255ull;
  }
  fprintf(stderr, "weight arena: %.1f GiB (experts direct-read: %.1f GiB)\n",
          total / 1073741824.0, xbytes / 1073741824.0);
  uint8_t* arena = nullptr;
  DALLOC(&arena, total);

  struct Job {
    std::string name;
    const uint8_t* src;
    uint8_t* dst;
    size_t size;
  };
  std::vector<Job> jobs;
  size_t off = 0;
  for (auto& kv : ck.tensors()) {
    const Tensor& t = kv.second;
    if (t.dtype == 10 || t.dtype == 4) continue;
#ifndef _WIN32
    if (is_experts(t.name)) continue;  // 已在上面登记 host 指针
#endif
    jobs.push_back({t.name, t.data, arena + off, t.data_size});
    g_devtensor[t.name] = arena + off;
    off += (t.data_size + 255) & ~255ull;
  }
  // big tensors first so stragglers don't serialize at the end
  std::sort(jobs.begin(), jobs.end(),
            [](const Job& a, const Job& b) { return a.size > b.size; });

  const int NW = 4;
  std::atomic<size_t> next{0}, done_bytes{0};
  std::atomic<long long> pload_last{0};
  // 0.5 s cadence: \r on a terminal, one timestamped line per tick in log
  // files (unthrottled \r fragments turn logs into a single blob).
  auto prog_tick = [&] {
    auto nowms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count();
    long long prev = pload_last.load();
    if (nowms - prev >= 500 && pload_last.compare_exchange_strong(prev, nowms)) {
      if (isatty(2))
        fprintf(stderr, "\rpload %.1f/%.1f GiB", done_bytes / 1073741824.0,
                total / 1073741824.0);
      else
        fprintf(stderr, "%s pload %.1f/%.1f GiB\n", log_ts(),
                done_bytes / 1073741824.0, total / 1073741824.0);
      fflush(stderr);
    }
  };
#ifdef _WIN32
  // Windows: memcpy 缺页读在主机 RAM 紧张时因 page cache 逐出抖动退化到
  // ~30 MB/s。改用 hgn.h 保持打开的 OVERLAPPED|NO_BUFFERING 句柄直接
  // ReadFile 进 pinned slot（4 线程 x 4 槽 x 16 MiB），再 H2D —— 完全不进
  // page cache，无逐出抖动。扇区对齐：offset/length/buffer 均按 4096。
  constexpr size_t WCHUNK = 16ull << 20;
  constexpr int WSLOTS = 4;
  struct WSlot {
    OVERLAPPED ov{};
    uint8_t* alloc = nullptr;
    uint8_t* buf = nullptr;  // 4096-aligned inside alloc
    bool busy = false;
    uint64_t c0 = 0;         // 该槽本次读的文件起始偏移
  };
  WSlot wslots[NW][WSLOTS];
  hipStream_t st[NW];
  for (int i = 0; i < NW; i++) {
    CK(hipStreamCreate(&st[i]));
    for (auto& s : wslots[i]) {
      s.ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
      CK(hipHostMalloc((void**)&s.alloc, WCHUNK + 4096, hipHostMallocDefault));
      s.buf = (uint8_t*)(((uintptr_t)s.alloc + 4095) & ~(uintptr_t)4095);
    }
  }
  auto wworker = [&](int wid) {
    WSlot* slots = wslots[wid];
    for (;;) {
      size_t ji = next.fetch_add(1);
      if (ji >= jobs.size()) break;
      const Job& j = jobs[ji];
      // 所属映射 -> （句柄， 文件偏移）
      HANDLE h = nullptr;
      uint64_t foff = 0;
      for (auto& mp : ck.mappings()) {
        if (j.src >= mp.base && j.src < mp.base + mp.len) {
          foff = (uint64_t)(j.src - mp.base);
          h = (HANDLE)mp.os_handle;
          break;
        }
      }
      if (!h) {
        fprintf(stderr, "pload: no os handle for %s\n", j.name.c_str());
        abort();
      }
      const uint64_t A = foff & ~4095ull, E = (foff + j.size + 4095) & ~4095ull;
      int si = 0;
      auto drain = [&](WSlot& s) {  // 等完成 + H2D 落在张量区间内的部分
        DWORD got = 0;
        if (!GetOverlappedResult(h, &s.ov, &got, TRUE)) {
          fprintf(stderr, "pload: overlapped read failed on %s: %lu\n",
                  j.name.c_str(), GetLastError());
          abort();
        }
        s.busy = false;
        uint64_t lo = std::max(s.c0, foff),
                 hi = std::min(s.c0 + got, foff + j.size);
        if (hi > lo) {
          CK(hipMemcpyAsync(j.dst + (lo - foff), s.buf + (lo - s.c0), hi - lo,
                            hipMemcpyHostToDevice, st[wid]));
          CK(hipStreamSynchronize(st[wid]));
          done_bytes += hi - lo;
          prog_tick();
        }
      };
      for (uint64_t c = A; c < E; c += WCHUNK) {
        WSlot& s = slots[si];
        si = (si + 1) % WSLOTS;
        if (s.busy) drain(s);
        DWORD len = (DWORD)std::min<uint64_t>(WCHUNK, E - c);
        ResetEvent(s.ov.hEvent);
        s.ov.Offset = (DWORD)(c & 0xffffffffu);
        s.ov.OffsetHigh = (DWORD)(c >> 32);
        s.c0 = c;
        BOOL r = ReadFile(h, s.buf, len, nullptr, &s.ov);
        if (!r) {
          DWORD e = GetLastError();
          if (e == ERROR_HANDLE_EOF) continue;  // 整块在 EOF 之外（未排队，勿 drain）
          if (e != ERROR_IO_PENDING) {
            fprintf(stderr, "pload: ReadFile %s failed: %lu\n", j.name.c_str(), e);
            abort();
          }
        }
        s.busy = true;
      }
      for (int k = 0; k < WSLOTS; k++)
        if (slots[k].busy) drain(slots[k]);
    }
  };
  std::vector<std::thread> ths;
  for (int i = 0; i < NW; i++) ths.emplace_back(wworker, i);
  for (auto& t : ths) t.join();
  CK(hipDeviceSynchronize());
  for (int i = 0; i < NW; i++) {
    for (auto& s : wslots[i]) {
      CK(hipHostFree(s.alloc));
      CloseHandle(s.ov.hEvent);
    }
    CK(hipStreamDestroy(st[i]));
  }
#else
  const size_t STAGE = 64ull << 20;
  uint8_t* stage[NW];
  hipStream_t st[NW];
  for (int i = 0; i < NW; i++) {
    CK(hipHostMalloc(&stage[i], STAGE, hipHostMallocDefault));
    CK(hipStreamCreate(&st[i]));
  }
  auto worker = [&](int wid) {
    for (;;) {
      size_t ji = next.fetch_add(1);
      if (ji >= jobs.size()) break;
      const Job& j = jobs[ji];
      size_t copied = 0;
      while (copied < j.size) {
        size_t n = std::min(STAGE, j.size - copied);
        memcpy(stage[wid], j.src + copied, n);  // faults page cache in
        CK(hipMemcpyAsync(j.dst + copied, stage[wid], n, hipMemcpyHostToDevice,
                          st[wid]));
        CK(hipStreamSynchronize(st[wid]));
        copied += n;
        done_bytes += n;
        prog_tick();
      }
    }
  };
  std::vector<std::thread> ths;
  for (int i = 0; i < NW; i++) ths.emplace_back(worker, i);
  for (auto& t : ths) t.join();
  CK(hipDeviceSynchronize());
  for (int i = 0; i < NW; i++) {
    CK(hipHostFree(stage[i]));
    CK(hipStreamDestroy(st[i]));
  }
#endif
  if (isatty(2)) fprintf(stderr, "\n");
  fprintf(stderr, "%s pload done: %.1f GiB\n", log_ts(), total / 1073741824.0);
}

// device pointer for a tensor's data
static const uint8_t* dev_ptr(const Tensor& t) {
  auto it = g_devtensor.find(t.name);
  if (it == g_devtensor.end())
    throw std::runtime_error("tensor not in arena: " + t.name);
  return it->second;
}

// --- device weight view for GEMV dispatch ------------------------------------
struct WView {
  int dtype;
  const uint8_t* data;  // device pointer
  uint64_t rows, cols, scale_stride;
};

static WView wview(const Tensor& t) {
  WView w;
  w.dtype = t.dtype;
  w.data = dev_ptr(t);
  w.cols = t.dims[t.ndims - 1];
  w.rows = t.numel() / w.cols;
  if (t.dtype == 5) {
    uint64_t codes_bytes = w.rows * w.cols / 2;
    w.scale_stride = ((w.cols / 32 * 2) + 15) & ~15ull;
    (void)codes_bytes;
  } else {
    w.scale_stride = 0;
  }
  return w;
}

static void gemv(const WView& w, const float* x, float* y, uint64_t row_off = 0,
                 uint64_t nrows = 0) {
  uint64_t rows = nrows ? nrows : w.rows;
  int warps_per_block = 8;
  int threads = warps_per_block * 64;
  uint64_t blocks = (rows + warps_per_block - 1) / warps_per_block;
  if (w.dtype == 5) {
    const uint8_t* codes = w.data + 64 + row_off * (w.cols / 2);
    const uint8_t* scales =
        w.data + 64 + w.rows * w.cols / 2 + row_off * w.scale_stride;
    // v3 kernel: 2 rows per warp, blockDim/warpSize warps per block
    uint64_t wpb = threads / 32;
    uint64_t blocks_v3 = (rows + wpb * 2 - 1) / (wpb * 2);
    k_q4cp_gemv<<<(unsigned)blocks_v3, threads, 0, g_str>>>(codes, scales, (const float*)w.data, x,
                                               y, rows, w.cols, w.scale_stride);
  } else if (w.dtype == 7) {
    uint64_t stride = w.cols + w.cols / 64 * 4;
    k_q8g64_gemv<<<(unsigned)blocks, threads, 0, g_str>>>(w.data + row_off * stride, x, y, rows,
                                                w.cols);
  } else if (w.dtype == 0) {
    k_bf16_gemv<<<(unsigned)blocks, threads, 0, g_str>>>(
        (const uint16_t*)w.data + row_off * w.cols, x, y, rows, w.cols);
  } else {
    throw std::runtime_error("gemv: unsupported dtype");
  }
}

// Multi-row GEMV dispatch (P in 1..8) for q4cp (dtype 5, fp32/bf16 x), bf16
// (dtype 0) and q8g64 (dtype 7) weights: streams the weight once for all P
// rows. Returns false when the shape/layout is unsupported (caller falls
// back to the dequant->bf16->hipBLASLt path).
static bool gemv_multi_q4cp(const WView& w, const void* x, float* y, int P,
                            uint64_t xstride, bool xbf16 = false) {
  if (getenv("GDEC_NO_GEMVMR")) return false;
  if (P < 1 || P > 8) return false;
  if (!xstride) xstride = w.cols;
  if (xbf16) {
    if (xstride % 8 || (((uintptr_t)x | (uintptr_t)y) & 15)) return false;
  } else if (xstride % 4 || (((uintptr_t)x | (uintptr_t)y) & 15))
    return false;
  int threads = 512;
  uint64_t wpb = threads / 32;
  uint64_t blocks = (w.rows + wpb * 2 - 1) / (wpb * 2);
  if (w.dtype == 0 && !xbf16 && w.cols % 8 == 0) {
    switch (P) {
#define GDEC_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_bf16_gemv_mr<PP><<<(unsigned)blocks, threads, 0, g_str>>>(           \
        (const uint16_t*)w.data, (const float*)x, y, w.rows, w.cols,       \
        xstride);                                                          \
    break;
      GDEC_GEMV_MR_CASE(1)
      GDEC_GEMV_MR_CASE(2)
      GDEC_GEMV_MR_CASE(3)
      GDEC_GEMV_MR_CASE(4)
      GDEC_GEMV_MR_CASE(5)
      GDEC_GEMV_MR_CASE(6)
      GDEC_GEMV_MR_CASE(7)
      GDEC_GEMV_MR_CASE(8)
#undef GDEC_GEMV_MR_CASE
      default:
        return false;
    }
    return true;
  }
  if (w.dtype == 7 && !xbf16 && w.cols % 64 == 0) {
    switch (P) {
#define GDEC_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_q8g64_gemv_mr<PP><<<(unsigned)blocks, threads, 0, g_str>>>(          \
        w.data, (const float*)x, y, w.rows, w.cols, xstride);              \
    break;
      GDEC_GEMV_MR_CASE(1)
      GDEC_GEMV_MR_CASE(2)
      GDEC_GEMV_MR_CASE(3)
      GDEC_GEMV_MR_CASE(4)
      GDEC_GEMV_MR_CASE(5)
      GDEC_GEMV_MR_CASE(6)
      GDEC_GEMV_MR_CASE(7)
      GDEC_GEMV_MR_CASE(8)
#undef GDEC_GEMV_MR_CASE
      default:
        return false;
    }
    return true;
  }
  if (w.dtype != 5) return false;
  const uint8_t* codes = w.data + 64;
  const uint8_t* scales = w.data + 64 + w.rows * w.cols / 2;
  if (xbf16) switch (P) {
#define GDEC_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_q4cp_gemv_mr<PP, true><<<(unsigned)blocks, threads, 0, g_str>>>(     \
        codes, scales, (const float*)w.data, x, y, w.rows, w.cols,         \
        w.scale_stride, xstride);                                          \
    break;
    GDEC_GEMV_MR_CASE(1)
    GDEC_GEMV_MR_CASE(2)
    GDEC_GEMV_MR_CASE(3)
    GDEC_GEMV_MR_CASE(4)
    GDEC_GEMV_MR_CASE(5)
    GDEC_GEMV_MR_CASE(6)
    GDEC_GEMV_MR_CASE(7)
    GDEC_GEMV_MR_CASE(8)
#undef GDEC_GEMV_MR_CASE
    default:
      return false;
  }
  else switch (P) {
#define GDEC_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_q4cp_gemv_mr<PP, false><<<(unsigned)blocks, threads, 0, g_str>>>(    \
        codes, scales, (const float*)w.data, x, y, w.rows, w.cols,         \
        w.scale_stride, xstride);                                          \
    break;
    GDEC_GEMV_MR_CASE(1)
    GDEC_GEMV_MR_CASE(2)
    GDEC_GEMV_MR_CASE(3)
    GDEC_GEMV_MR_CASE(4)
    GDEC_GEMV_MR_CASE(5)
    GDEC_GEMV_MR_CASE(6)
    GDEC_GEMV_MR_CASE(7)
    GDEC_GEMV_MR_CASE(8)
#undef GDEC_GEMV_MR_CASE
    default:
      return false;
  }
  return true;
}

// ==================== Vision tower (ViT) — vision stage 4 ====================
// Ported from src/gpu/vit.cpp (stage 3, 27-layer parity vs the PyTorch
// reference, see HANDOVER_2026-09-13-VIT.md). Numeric path: fp16 GEMM I/O
// (weights bf16->fp16 host-side, exact at normal magnitudes) + split-fp16
// dual GEMM for patch_embed/fc1/fc2 inputs; residuals/LN/softmax fp32,
// eps=1e-6; MLP tanh-GELU, merger erf-GELU; non-causal WMMA flash attention.
// Loaded only via --vision-tower <sidecar.hgn>; without it the engine is
// bitwise identical to the text-only build (image requests are rejected).

__device__ inline uint16_t vit_f2h(float f) {  // fp32->fp16 RNE
  return __half_as_ushort(__float2half_rn(f));
}
__device__ inline float vit_h2f(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}

__global__ void k_vit_f32_to_fp16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = vit_f2h(x[i]);
}

// split-fp16: x = hi + lo (lo keeps the residual hi rounded away; two-GEMM
// accumulation ~= fp32 input precision). Required for patch/fc1/fc2 inputs
// (stage-3 attribution: input rounding amplified over 27 layers).
__global__ void k_vit_f32_split_fp16(const float* x, uint16_t* hi, uint16_t* lo,
                                     uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float h = vit_h2f(vit_f2h(x[i]));
  hi[i] = vit_f2h(x[i]);
  lo[i] = vit_f2h(x[i] - h);
}

__global__ void k_vit_bias_add(float* y, const float* b, int P, int N) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (uint64_t)P * N) y[i] += b[i % N];
}

__global__ void k_vit_bias_resadd(float* h, const float* y, const float* b,
                                  int P, int N) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (uint64_t)P * N) h[i] += y[i] + b[i % N];
}

__global__ void k_vit_layernorm(const float* x, float* y, const float* w,
                                const float* b, int N, float eps) {
  int row = blockIdx.x;
  const float* xr = x + (uint64_t)row * N;
  float* yr = y + (uint64_t)row * N;
  __shared__ float red[256];
  float s = 0, ss = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v = xr[i];
    s += v;
    ss += v * v;
  }
  red[threadIdx.x] = s;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float mean = red[0] / N;
  __syncthreads();
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float var = red[0] / N - mean * mean;
  float rstd = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

// pos_embed gather + 4-tap bilinear weights, added into residual h (P x 1152)
__global__ void k_vit_pos_embed(float* h, const uint16_t* table,
                                const int* idx, const float* wts, int P) {
  int p = blockIdx.x;
  for (int d = threadIdx.x; d < 1152; d += blockDim.x) {
    float acc = 0;
    for (int t = 0; t < 4; t++)
      acc += wts[p * 4 + t] *
             vit_h2f(table[(uint64_t)idx[p * 4 + t] * 1152 + d]);
    h[(uint64_t)p * 1152 + d] += acc;
  }
}

// rope on q,k (fp32 qkv (P,3456)) -> fp16 q/k/v (P,1152). rotate_half with
// per-position (P,72) cos/sin broadcast across the 16 heads.
__global__ void k_vit_rope_qkv(const float* qkv, const float* cos_t,
                               const float* sin_t, uint16_t* qb, uint16_t* kb,
                               uint16_t* vb, int P) {
  int p = blockIdx.x;
  int h = blockIdx.y;
  int d = threadIdx.x;  // 36 threads
  if (d >= 36) return;
  const float* qr = qkv + (uint64_t)p * 3456 + h * 72;
  const float* kr = qr + 1152;
  const float* vr = qr + 2304;
  const float* cr = cos_t + (uint64_t)p * 72;
  const float* sr = sin_t + (uint64_t)p * 72;
  float c0 = cr[d], s0 = sr[d], c1 = cr[d + 36], s1 = sr[d + 36];
  float q0 = qr[d], q1 = qr[d + 36];
  float k0 = kr[d], k1 = kr[d + 36];
  uint64_t o = (uint64_t)p * 1152 + h * 72;
  qb[o + d] = vit_f2h(q0 * c0 - q1 * s0);
  qb[o + d + 36] = vit_f2h(q1 * c1 + q0 * s1);
  kb[o + d] = vit_f2h(k0 * c0 - k1 * s0);
  kb[o + d + 36] = vit_f2h(k1 * c1 + k0 * s1);
  vb[o + d] = vit_f2h(vr[d]);
  vb[o + d + 36] = vit_f2h(vr[d + 36]);
}

using vit_shortx16 = __attribute__((ext_vector_type(16))) short;
using vit_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ vit_shortx16 vit_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  vit_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}

// WMMA fp16 flash attention (non-causal, single segment); gfx1151 fragment
// layout identical to k_qsa_wmma (see its comment). grid (ceil(P/128), 16
// heads), block 256 = 8 warps, warp w owns q rows [w*16,+16). Online softmax
// fp32; P quantized fp16 (stage-3: probs rounding negligible).
__global__ void __launch_bounds__(256)
k_vit_flash_wmma(const uint16_t* __restrict__ qg, const uint16_t* __restrict__ kg,
                 const uint16_t* __restrict__ vg, float* __restrict__ out,
                 int P, float scale) {
  __shared__ uint16_t Qs[128][88], Ks[64][88], Vs[64][88], Pbuf[128][72];
  const int r0 = blockIdx.x * 128;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, w = tid >> 5;
  const int colbase = lane & 15;
  for (int i = tid; i < 128 * 44; i += 256) {
    int rr = i / 44, jj = i % 44;
    uint32_t val = 0;
    if (r0 + rr < P && jj < 36)
      val = ((const uint32_t*)(qg + (size_t)(r0 + rr) * 1152 + h * 72))[jj];
    ((uint32_t*)Qs[rr])[jj] = val;
  }
  __syncthreads();
  int qrowA = -1;
  if (lane < 16 && !(lane & 1)) qrowA = lane / 2;
  else if (lane >= 17 && (lane & 1)) qrowA = 8 + (lane - 17) / 2;
  float m_run[8], l_run[8];
  vit_floatx8 acc[5];
#pragma unroll
  for (int dc = 0; dc < 5; dc++) acc[dc] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int e = 0; e < 8; e++) {
    m_run[e] = -1e30f;
    l_run[e] = 0.f;
  }
  const int nkt = (P + 63) / 64;
  for (int kt = 0; kt < nkt; kt++) {
    const int c0 = kt * 64;
    __syncthreads();
    for (int i = tid; i < 64 * 44; i += 256) {
      int rr = i / 44, jj = i % 44;
      uint32_t kv = 0, vv = 0;
      if (c0 + rr < P && jj < 36) {
        kv = ((const uint32_t*)(kg + (size_t)(c0 + rr) * 1152 + h * 72))[jj];
        vv = ((const uint32_t*)(vg + (size_t)(c0 + rr) * 1152 + h * 72))[jj];
      }
      ((uint32_t*)Ks[rr])[jj] = kv;
      ((uint32_t*)Vs[rr])[jj] = vv;
    }
    __syncthreads();
    vit_floatx8 s[4];
#pragma unroll
    for (int f = 0; f < 4; f++) s[f] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 5; kk++) {
      vit_shortx16 a = {0};
      if (qrowA >= 0) a = vit_ld16(&Qs[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int f = 0; f < 4; f++) {
        short bt[16];
        int krow = f * 16 + colbase;
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Ks[krow][kk * 16 + k];
        vit_shortx16 b;
        memcpy(&b, bt, 32);
        s[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s[f]);
      }
    }
    float mx[8];
#pragma unroll
    for (int e = 0; e < 8; e++) mx[e] = -1e30f;
#pragma unroll
    for (int f = 0; f < 4; f++) {
      int col = c0 + f * 16 + colbase;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        float v = s[f][e] * scale;
        if (col >= P) v = -1e30f;
        s[f][e] = v;
        mx[e] = fmaxf(mx[e], v);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1)
        mx[e] = fmaxf(mx[e], __shfl_xor(mx[e], o));
      float mnew = fmaxf(m_run[e], mx[e]);
      float corr = expf(m_run[e] - mnew);
      float es = 0.f;
#pragma unroll
      for (int f = 0; f < 4; f++) {
        float p = expf(s[f][e] - mnew);
        s[f][e] = p;
        es += p;
      }
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1) es += __shfl_xor(es, o);
      l_run[e] = l_run[e] * corr + es;
      m_run[e] = mnew;
#pragma unroll
      for (int dc = 0; dc < 5; dc++) acc[dc][e] *= corr;
    }
#pragma unroll
    for (int f = 0; f < 4; f++)
#pragma unroll
      for (int e = 0; e < 8; e++)
        Pbuf[w * 16 + (lane < 16 ? e : 8 + e)][f * 16 + colbase] =
            vit_f2h(s[f][e]);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      vit_shortx16 a = {0};
      if (qrowA >= 0) a = vit_ld16(&Pbuf[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int dc = 0; dc < 5; dc++) {
        short bt[16];
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Vs[kk * 16 + k][dc * 16 + colbase];
        vit_shortx16 b;
        memcpy(&b, bt, 32);
        acc[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[dc]);
      }
    }
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    int row = w * 16 + (lane < 16 ? e : 8 + e);
    if (r0 + row >= P) continue;
    float inv = 1.f / l_run[e];
#pragma unroll
    for (int dc = 0; dc < 5; dc++) {
      int col = dc * 16 + colbase;
      if (col < 72)
        out[(size_t)(r0 + row) * 1152 + h * 72 + col] = acc[dc][e] * inv;
    }
  }
}

// k-tile double-buffered k_vit_flash_wmma (2026-09-13, mirrors vit.cpp
// k_flash_wmma_db): next K/V tile staged in 24 uint4 registers issued at
// tile top (overlaps all compute); Ks written after S1, VsT after S2 (V
// stored transposed VsT[80][68] so the PV B-gather becomes wide
// ds_load_2addr_b64 instead of 320 scalar u16 loads). softmax expf ->
// __expf (v_exp_f32; probs quantize to fp16 downstream, three-image cmp
// verified). fp16 in / fp32 accumulate and online-softmax order unchanged.
__global__ void __launch_bounds__(256)
k_vit_flash_wmma_db(const uint16_t* __restrict__ qg,
                    const uint16_t* __restrict__ kg,
                    const uint16_t* __restrict__ vg, float* __restrict__ out,
                    int P, float scale) {
  __shared__ uint16_t Qs[128][88], Ks[64][88], VsT[80][68], Pbuf[128][72];
  const int r0 = blockIdx.x * 128;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, w = tid >> 5;
  const int colbase = lane & 15;
  for (int i = tid; i < 128 * 44; i += 256) {
    int rr = i / 44, jj = i % 44;
    uint32_t val = 0;
    if (r0 + rr < P && jj < 36)
      val = ((const uint32_t*)(qg + (size_t)(r0 + rr) * 1152 + h * 72))[jj];
    ((uint32_t*)Qs[rr])[jj] = val;
  }
  for (int i = tid; i < 64 * 2; i += 256) {  // Ks padding cols [72,88)
    int rr = i >> 1, c9 = 9 + (i & 1);
    ((uint4*)Ks[rr])[c9] = uint4{0, 0, 0, 0};
  }
  for (int i = tid; i < 80 * 68; i += 256)  // VsT zero once (pad rows/cols)
    ((uint16_t*)VsT)[i] = 0;
  auto stage = [&](int c0, uint4* kr, uint4* vr) {
#pragma unroll
    for (int u = 0; u < 3; u++) {
      int i = tid + u * 256;
      uint4 k = {0, 0, 0, 0}, v = {0, 0, 0, 0};
      if (i < 576) {
        int rr = i / 9, c9 = i - rr * 9;
        if (c0 + rr < P) {
          k = *(const uint4*)(kg + (size_t)(c0 + rr) * 1152 + h * 72 + c9 * 8);
          v = *(const uint4*)(vg + (size_t)(c0 + rr) * 1152 + h * 72 + c9 * 8);
        }
      }
      kr[u] = k;
      vr[u] = v;
    }
  };
  auto storeK = [&](const uint4* r) {
#pragma unroll
    for (int u = 0; u < 3; u++) {
      int i = tid + u * 256;
      if (i < 576) {
        int rr = i / 9, c9 = i - rr * 9;
        ((uint4*)Ks[rr])[c9] = r[u];
      }
    }
  };
  auto storeV = [&](const uint4* r) {  // transposed: VsT[dim][key]
#pragma unroll
    for (int u = 0; u < 3; u++) {
      int i = tid + u * 256;
      if (i < 576) {
        int rr = i / 9, c9 = i - rr * 9;
        uint16_t tmp[8];
        *(uint4*)tmp = r[u];
#pragma unroll
        for (int j = 0; j < 8; j++) VsT[c9 * 8 + j][rr] = tmp[j];
      }
    }
  };
  const int nkt = (P + 63) / 64;
  uint4 kn[3], vn[3];
  stage(0, kn, vn);
  storeK(kn);
  storeV(vn);
  __syncthreads();
  int qrowA = -1;
  if (lane < 16 && !(lane & 1)) qrowA = lane / 2;
  else if (lane >= 17 && (lane & 1)) qrowA = 8 + (lane - 17) / 2;
  float m_run[8], l_run[8];
  vit_floatx8 acc[5];
#pragma unroll
  for (int dc = 0; dc < 5; dc++) acc[dc] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int e = 0; e < 8; e++) {
    m_run[e] = -1e30f;
    l_run[e] = 0.f;
  }
  for (int kt = 0; kt < nkt; kt++) {
    const int c0 = kt * 64;
    if (kt + 1 < nkt) stage(c0 + 64, kn, vn);  // prefetch next tile
    vit_floatx8 s[4];
#pragma unroll
    for (int f = 0; f < 4; f++) s[f] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 5; kk++) {
      vit_shortx16 a = {0};
      if (qrowA >= 0) a = vit_ld16(&Qs[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int f = 0; f < 4; f++) {
        short bt[16];
        int krow = f * 16 + colbase;
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)Ks[krow][kk * 16 + k];
        vit_shortx16 b;
        memcpy(&b, bt, 32);
        s[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, s[f]);
      }
    }
    float mx[8];
#pragma unroll
    for (int e = 0; e < 8; e++) mx[e] = -1e30f;
#pragma unroll
    for (int f = 0; f < 4; f++) {
      int col = c0 + f * 16 + colbase;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        float v = s[f][e] * scale;
        if (col >= P) v = -1e30f;
        s[f][e] = v;
        mx[e] = fmaxf(mx[e], v);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; e++) {
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1)
        mx[e] = fmaxf(mx[e], __shfl_xor(mx[e], o));
      float mnew = fmaxf(m_run[e], mx[e]);
      float corr = __expf(m_run[e] - mnew);
      float es = 0.f;
#pragma unroll
      for (int f = 0; f < 4; f++) {
        float p = __expf(s[f][e] - mnew);
        s[f][e] = p;
        es += p;
      }
#pragma unroll
      for (int o = 1; o <= 8; o <<= 1) es += __shfl_xor(es, o);
      l_run[e] = l_run[e] * corr + es;
      m_run[e] = mnew;
#pragma unroll
      for (int dc = 0; dc < 5; dc++) acc[dc][e] *= corr;
    }
#pragma unroll
    for (int f = 0; f < 4; f++)
#pragma unroll
      for (int e = 0; e < 8; e++)
        Pbuf[w * 16 + (lane < 16 ? e : 8 + e)][f * 16 + colbase] =
            vit_f2h(s[f][e]);
    __syncthreads();  // S1: scores done with Ks, Pbuf visible
    if (kt + 1 < nkt) storeK(kn);
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      vit_shortx16 a = {0};
      if (qrowA >= 0) a = vit_ld16(&Pbuf[w * 16 + qrowA][kk * 16]);
#pragma unroll
      for (int dc = 0; dc < 5; dc++) {
        short bt[16];
#pragma unroll
        for (int k = 0; k < 16; k++)
          bt[k] = (short)VsT[dc * 16 + colbase][kk * 16 + k];
        vit_shortx16 b;
        memcpy(&b, bt, 32);
        acc[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[dc]);
      }
    }
    __syncthreads();  // S2: PV done with VsT/Pbuf, Ks store visible
    if (kt + 1 < nkt) storeV(vn);  // next PV is past next S1: visible
  }
#pragma unroll
  for (int e = 0; e < 8; e++) {
    int row = w * 16 + (lane < 16 ? e : 8 + e);
    if (r0 + row >= P) continue;
    float inv = 1.f / l_run[e];
#pragma unroll
    for (int dc = 0; dc < 5; dc++) {
      int col = dc * 16 + colbase;
      if (col < 72)
        out[(size_t)(r0 + row) * 1152 + h * 72 + col] = acc[dc][e] * inv;
    }
  }
}

__global__ void k_vit_gelu_tanh_split(const float* x, uint16_t* hi,
                                      uint16_t* lo, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  float g = 0.5f * v * (1.f + tanhf(u));
  float h = vit_h2f(vit_f2h(g));
  hi[i] = vit_f2h(g);
  lo[i] = vit_f2h(g - h);
}

__global__ void k_vit_gelu_erf_fp16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  y[i] = vit_f2h(0.5f * v * (1.f + erff(v * 0.7071067811865476f)));
}

// fp32 -> bf16 RNE (f2bf, same rounding as torch .to(bf16)); merger output
// is quantized before injection, matching HF masked_scatter of bf16
// image_embeds into bf16 inputs_embeds.
__global__ void k_vit_f32_to_bf16(const float* x, uint16_t* y, uint64_t n) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = f2bf(x[i]);
}

// masked_scatter injection: overwrite prefill stream rows (chunk-local,
// [P,10240], 4 repeated 2560 branches) at image placeholder positions with
// the ViT embeds. One block per token; map[i] = (src embed row, dst row).
__global__ void k_img_inject_f32(const uint16_t* __restrict__ emb,
                                 const int2* __restrict__ map,
                                 float* __restrict__ Rb) {
  int src = map[blockIdx.x].x, dst = map[blockIdx.x].y;
  const uint16_t* s = emb + (size_t)src * 2560;
  float* r = Rb + (size_t)dst * 10240;
  for (int d = threadIdx.x; d < 2560; d += blockDim.x) {
    float v = bf2f(s[d]);
    r[d] = v;
    r[2560 + d] = v;
    r[5120 + d] = v;
    r[7680 + d] = v;
  }
}
__global__ void k_img_inject_bf16(const uint16_t* __restrict__ emb,
                                  const int2* __restrict__ map,
                                  uint16_t* __restrict__ Rb) {
  int src = map[blockIdx.x].x, dst = map[blockIdx.x].y;
  const uint16_t* s = emb + (size_t)src * 2560;
  uint16_t* r = Rb + (size_t)dst * 10240;
  for (int d = threadIdx.x; d < 2560; d += blockDim.x) {
    uint16_t v = s[d];
    r[d] = v;
    r[2560 + d] = v;
    r[5120 + d] = v;
    r[7680 + d] = v;
  }
}

// ---- npy I/O (--vision-test offline dumps only) ----
// FNV-1a over the raw patch bytes of every image in a request. Guards the
// M-RoPE cont path: placeholder token runs are pixel-agnostic, so prefix
// matching alone would pair a re-sent history with a DIFFERENT same-shape
// image's cached KV.
static uint64_t fnv1a_patches(const std::vector<std::vector<float>>& vp) {
  if (vp.empty()) return 0;
  uint64_t h = 14695981039346656037ull;
  for (const auto& v : vp) {
    const uint8_t* p = (const uint8_t*)v.data();
    size_t n = v.size() * sizeof(float);
    for (size_t i = 0; i < n; i++) {
      h ^= p[i];
      h *= 1099511628211ull;
    }
  }
  return h;
}
struct VitNpy {
  std::vector<uint64_t> shape;
  char descr[8] = {0};
  std::vector<uint8_t> data;
  uint64_t numel() const {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

static VitNpy vit_npy_read(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) throw std::runtime_error(std::string("open ") + path);
  uint8_t magic[6];
  if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6))
    throw std::runtime_error("bad npy magic");
  uint8_t ver[2];
  if (fread(ver, 1, 2, f) != 2) throw std::runtime_error("npy ver");
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t s;
    if (fread(&s, 2, 1, f) != 1) throw std::runtime_error("npy hlen");
    hlen = s;
  } else {
    if (fread(&hlen, 4, 1, f) != 1) throw std::runtime_error("npy hlen");
  }
  std::string hdr(hlen, '\0');
  if (fread(hdr.data(), 1, hlen, f) != hlen) throw std::runtime_error("npy hdr");
  VitNpy r;
  auto dp = hdr.find("'descr'");
  if (dp == std::string::npos) dp = hdr.find("\"descr\"");
  if (dp == std::string::npos) throw std::runtime_error("npy descr");
  auto q1 = hdr.find('\'', dp + 7);
  if (q1 == std::string::npos) q1 = hdr.find('"', dp + 7);
  sscanf(hdr.c_str() + q1 + 1, "%3[^'\"]", r.descr);
  if (hdr.find("False") == std::string::npos)
    throw std::runtime_error("fortran order unsupported");
  auto sp = hdr.find("'shape'");
  if (sp == std::string::npos) sp = hdr.find("\"shape\"");
  auto lp = hdr.find('(', sp), rp = hdr.find(')', sp);
  std::string dims = hdr.substr(lp + 1, rp - lp - 1);
  for (char& c : dims)
    if (c == ',') c = ' ';
  {
    std::istringstream iss(dims);
    uint64_t d;
    while (iss >> d) r.shape.push_back(d);
  }
  if (strcmp(r.descr, "<f4")) throw std::runtime_error("npy dtype must be <f4");
  r.data.resize(r.numel() * 4);
  if (fread(r.data.data(), 1, r.data.size(), f) != r.data.size())
    throw std::runtime_error("npy data short");
  fclose(f);
  return r;
}

static void vit_npy_write_f32(const char* path, const float* p, uint64_t rows,
                              uint64_t cols) {
  FILE* f = fopen(path, "wb");
  if (!f) throw std::runtime_error(std::string("write ") + path);
  char hdr[256];
  int n = snprintf(hdr, sizeof(hdr),
                   "{'descr': '<f4', 'fortran_order': False, 'shape': (%llu, %llu), }",
                   (unsigned long long)rows, (unsigned long long)cols);
  int total = 10 + n + 1;
  int pad = (64 - total % 64) % 64;
  fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  uint16_t hl = n + pad + 1;
  fwrite(&hl, 2, 1, f);
  fwrite(hdr, 1, n, f);
  for (int i = 0; i < pad; i++) fputc(' ', f);
  fputc('\n', f);
  fwrite(p, 4, rows * cols, f);
  fclose(f);
}

static void vit_npy_write_u16(const char* path, const uint16_t* p,
                              uint64_t rows, uint64_t cols) {
  FILE* f = fopen(path, "wb");
  if (!f) throw std::runtime_error(std::string("write ") + path);
  char hdr[256];
  int n = snprintf(hdr, sizeof(hdr),
                   "{'descr': '<u2', 'fortran_order': False, 'shape': (%llu, %llu), }",
                   (unsigned long long)rows, (unsigned long long)cols);
  int total = 10 + n + 1;
  int pad = (64 - total % 64) % 64;
  fwrite("\x93NUMPY\x01\x00", 1, 8, f);
  uint16_t hl = n + pad + 1;
  fwrite(&hl, 2, 1, f);
  fwrite(hdr, 1, n, f);
  for (int i = 0; i < pad; i++) fputc(' ', f);
  fputc('\n', f);
  fwrite(p, 2, rows * cols, f);
  fclose(f);
}

// ---- VisionTower: weights + forward, handles borrowed from the engine ----
struct VisionTower {
  hipblasLtHandle_t lth = nullptr;
  rocblas_handle rbh = nullptr;
  void* d_ltws = nullptr;
  size_t lt_ws = 0;

  struct BlockW {
    uint16_t *qkv_w, *proj_w, *fc1_w, *fc2_w;
    float *qkv_b, *proj_b, *fc1_b, *fc2_b, *ln1_w, *ln1_b, *ln2_w, *ln2_b;
  };
  uint16_t* patch_w = nullptr;   // 1152x1536
  float* patch_b = nullptr;
  uint16_t* pos_table = nullptr; // 2304x1152
  BlockW blk[27];
  float *mg_ln_w = nullptr, *mg_ln_b = nullptr, *mg_fc1_b = nullptr,
        *mg_fc2_b = nullptr;
  uint16_t *mg_fc1_w = nullptr, *mg_fc2_w = nullptr;  // 4608x4608, 2560x4608

  struct LtEntry {
    hipblasLtMatmulDesc_t opd;
    hipblasLtMatrixLayout_t Al, Bl, Cl, Dl;
    hipblasLtMatmulAlgo_t algo;
  };
  using LtKey = std::tuple<int, int, int, int, int, int, int, int, int>;
  std::map<LtKey, LtEntry> lt_cache;

  // grow-only activation buffers (sized by the largest image seen)
  int capP = 0;
  float *d_h = nullptr, *d_n = nullptr, *d_qkv = nullptr, *d_attn = nullptr,
        *d_mlp = nullptr, *d_out = nullptr;
  uint16_t *d_xbf = nullptr, *d_xlo = nullptr, *d_qb = nullptr, *d_kb = nullptr,
           *d_vb = nullptr, *d_emb_bf16 = nullptr;
  float *d_cos = nullptr, *d_sin = nullptr;
  int* d_idx = nullptr;
  float* d_wts = nullptr;

  void lt_gemm(hipblasOperation_t opA, hipblasOperation_t opB, int M, int N,
               int K, const uint16_t* A, int lda, const uint16_t* B, int ldb,
               void* C, int ldc, float alpha, float beta, bool c16 = false) {
    LtKey key{(int)opA, (int)opB, M, N, K, lda, ldb, ldc, (int)c16};
    auto it = lt_cache.find(key);
    if (it == lt_cache.end()) {
      LtEntry e;
      hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &opA, sizeof(opA));
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB,
                                      &opB, sizeof(opB));
      hipblasLtMatrixLayoutCreate(&e.Al, HIP_R_16F, opA == HIPBLAS_OP_N ? M : K,
                                  opA == HIPBLAS_OP_N ? K : M, lda);
      hipblasLtMatrixLayoutCreate(&e.Bl, HIP_R_16F, opB == HIPBLAS_OP_N ? K : N,
                                  opB == HIPBLAS_OP_N ? N : K, ldb);
      hipDataType cdt = c16 ? HIP_R_16F : HIP_R_32F;
      hipblasLtMatrixLayoutCreate(&e.Cl, cdt, M, N, ldc);
      hipblasLtMatrixLayoutCreate(&e.Dl, cdt, M, N, ldc);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws,
          sizeof(lt_ws));
      hipblasLtMatmulHeuristicResult_t hres[16];
      int nres = 0;
      hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, e.Bl, e.Cl, e.Dl, pref,
                                      16, hres, &nres);
      float one = 1.f, zero = 0.f;
      int best = -1;
      double best_ms = 1e30;
      for (int i = 0; i < nres; i++) {
        if (hres[i].state != HIPBLAS_STATUS_SUCCESS) continue;
        if (hres[i].workspaceSize > lt_ws) continue;
        hipblasStatus_t st =
            hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl, &zero, C, e.Cl,
                            C, e.Dl, &hres[i].algo, d_ltws, lt_ws, g_str);
        hipError_t sync_err = hipStreamSynchronize(g_str);
        if (st != HIPBLAS_STATUS_SUCCESS || sync_err != hipSuccess) {
          (void)hipGetLastError();
          continue;
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 3; r++)
          hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl, &zero, C, e.Cl,
                          C, e.Dl, &hres[i].algo, d_ltws, lt_ws, g_str);
        sync_err = hipStreamSynchronize(g_str);
        (void)sync_err;
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        if (ms < best_ms) {
          best_ms = ms;
          best = i;
        }
      }
      if (best < 0) {
        if (c16) {
          fprintf(stderr, "vit gemm c16 no LT algo M=%d N=%d K=%d\n", M, N, K);
          exit(1);
        }
        rocblas_status st = rocblas_gemm_ex(
            rbh, (rocblas_operation)opA, (rocblas_operation)opB, M, N, K, &one,
            A, rocblas_datatype_f16_r, lda, B, rocblas_datatype_f16_r, ldb,
            &zero, C, rocblas_datatype_f32_r, ldc, C, rocblas_datatype_f32_r,
            ldc, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        if (st != rocblas_status_success) {
          fprintf(stderr, "vit gemm failed M=%d N=%d K=%d (rb %d)\n", M, N, K,
                  (int)st);
          exit(1);
        }
        hipblasLtMatmulDescDestroy(e.opd);
        hipblasLtMatrixLayoutDestroy(e.Al);
        hipblasLtMatrixLayoutDestroy(e.Bl);
        hipblasLtMatrixLayoutDestroy(e.Cl);
        hipblasLtMatrixLayoutDestroy(e.Dl);
        return;
      }
      e.algo = hres[best].algo;
      it = lt_cache.emplace(key, e).first;
    }
    hipblasLtMatmul(lth, it->second.opd, &alpha, A, it->second.Al, B,
                    it->second.Bl, &beta, C, it->second.Cl, C, it->second.Dl,
                    &it->second.algo, d_ltws, lt_ws, g_str);
  }

  // row-major projection: Y(PxN) f32 = X(PxK) fp16 @ W(NxK) fp16^T
  void proj_gemm(const uint16_t* X, const uint16_t* W, float* Y, int P, int N,
                 int K) {
    lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, X, K, Y, N, 1.f, 0.f);
  }
  void proj_gemm_split(const uint16_t* hi, const uint16_t* lo,
                       const uint16_t* W, float* Y, int P, int N, int K) {
    lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, hi, K, Y, N, 1.f, 0.f);
    lt_gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, P, K, W, K, lo, K, Y, N, 1.f, 1.f);
  }

  static void* dup_dev(const void* p, size_t bytes) {
    void* d;
    DALLOC(&d, bytes);
    CK(hipMemcpy(d, p, bytes, hipMemcpyHostToDevice));
    return d;
  }
  static float* dup_dev_bf16_to_f32(const Tensor& t) {
    uint64_t n = t.numel();
    std::vector<float> h(n);
    const uint16_t* p = (const uint16_t*)t.data;
    for (uint64_t i = 0; i < n; i++) h[i] = hgn::bf16_to_f32(p[i]);
    return (float*)dup_dev(h.data(), n * 4);
  }

  void load(const char* path) {
    Checkpoint ck(path);
    // GEMM weights bf16 -> fp16 host-side (exact at normal magnitudes).
    auto bf = [&](const char* name) -> uint16_t* {
      const Tensor& t = ck.at(name);
      if (t.dtype != 0) throw std::runtime_error(std::string("vit dtype ") + name);
      uint64_t n = t.numel();
      std::vector<uint16_t> h(n);
      const uint16_t* p = (const uint16_t*)t.data;
      for (uint64_t i = 0; i < n; i++)
        h[i] = vit_f2h_host(hgn::bf16_to_f32(p[i]));
      return (uint16_t*)dup_dev(h.data(), n * 2);
    };
    auto f32 = [&](const char* name) { return dup_dev_bf16_to_f32(ck.at(name)); };
    patch_w = bf("visual.patch_embed.proj.weight");
    patch_b = f32("visual.patch_embed.proj.bias");
    pos_table = bf("visual.pos_embed.weight");
    char nm[128];
    for (int i = 0; i < 27; i++) {
      BlockW& b = blk[i];
      snprintf(nm, 128, "visual.blocks.%d.attn.qkv.weight", i);
      b.qkv_w = bf(nm);
      snprintf(nm, 128, "visual.blocks.%d.attn.qkv.bias", i);
      b.qkv_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.attn.proj.weight", i);
      b.proj_w = bf(nm);
      snprintf(nm, 128, "visual.blocks.%d.attn.proj.bias", i);
      b.proj_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.mlp.linear_fc1.weight", i);
      b.fc1_w = bf(nm);
      snprintf(nm, 128, "visual.blocks.%d.mlp.linear_fc1.bias", i);
      b.fc1_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.mlp.linear_fc2.weight", i);
      b.fc2_w = bf(nm);
      snprintf(nm, 128, "visual.blocks.%d.mlp.linear_fc2.bias", i);
      b.fc2_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.norm1.weight", i);
      b.ln1_w = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.norm1.bias", i);
      b.ln1_b = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.norm2.weight", i);
      b.ln2_w = f32(nm);
      snprintf(nm, 128, "visual.blocks.%d.norm2.bias", i);
      b.ln2_b = f32(nm);
    }
    mg_ln_w = f32("visual.merger.norm.weight");
    mg_ln_b = f32("visual.merger.norm.bias");
    mg_fc1_w = bf("visual.merger.linear_fc1.weight");
    mg_fc1_b = f32("visual.merger.linear_fc1.bias");
    mg_fc2_w = bf("visual.merger.linear_fc2.weight");
    mg_fc2_b = f32("visual.merger.linear_fc2.bias");
    fprintf(stderr, "vision: tower loaded from %s (%zu tensors)\n", path,
            ck.tensor_count());
  }

  static uint16_t vit_f2h_host(float f) {
    __half h = __float2half_rn(f);
    return __half_as_ushort(h);
  }

  void ensure(int P) {
    if (P <= capP) return;
    if (capP) {
      (void)hipFree(d_h); (void)hipFree(d_n); (void)hipFree(d_qkv);
      (void)hipFree(d_attn);
      (void)hipFree(d_mlp); (void)hipFree(d_out); (void)hipFree(d_xbf);
      (void)hipFree(d_xlo);
      (void)hipFree(d_qb); (void)hipFree(d_kb); (void)hipFree(d_vb);
      (void)hipFree(d_emb_bf16);
      (void)hipFree(d_cos); (void)hipFree(d_sin); (void)hipFree(d_idx);
      (void)hipFree(d_wts);
    }
    int M = P / 4;
    CK(hipMalloc(&d_h, (size_t)P * 1152 * 4));
    CK(hipMalloc(&d_n, (size_t)P * 1152 * 4));
    CK(hipMalloc(&d_qkv, (size_t)P * 3456 * 4));
    CK(hipMalloc(&d_attn, (size_t)P * 1152 * 4));
    CK(hipMalloc(&d_mlp, (size_t)P * 4608 * 4));  // merger fc1 also needs 4608
    CK(hipMalloc(&d_out, (size_t)M * 2560 * 4));
    CK(hipMalloc(&d_xbf, (size_t)P * 4608 * 2));
    CK(hipMalloc(&d_xlo, (size_t)P * 4608 * 2));
    CK(hipMalloc(&d_qb, (size_t)P * 1152 * 2));
    CK(hipMalloc(&d_kb, (size_t)P * 1152 * 2));
    CK(hipMalloc(&d_vb, (size_t)P * 1152 * 2));
    CK(hipMalloc(&d_emb_bf16, (size_t)M * 2560 * 2));
    CK(hipMalloc(&d_cos, (size_t)P * 72 * 4));
    CK(hipMalloc(&d_sin, (size_t)P * 72 * 4));
    CK(hipMalloc(&d_idx, (size_t)P * 4 * 4));
    CK(hipMalloc(&d_wts, (size_t)P * 4 * 4));
    capP = P;
  }

  // pos_embed 4-tap indices/weights + rope cos/sin (block-major (h,w) order,
  // replicating transformers vision_utils; host fp64 -> fp32).
  static void host_pos_interp(int gh, int gw, std::vector<int>& idx,
                              std::vector<float>& wts) {
    const int side = 48, merge = 2;
    int P = gh * gw;
    idx.resize(P * 4);
    wts.resize(P * 4);
    int blocks_w = gw / merge;
    for (int p = 0; p < P; p++) {
      int in_col = p % merge, in_row = (p / merge) % merge;
      int block_col = (p / (merge * merge)) % blocks_w;
      int block_row = p / (merge * merge * blocks_w);
      int row = block_row * merge + in_row;
      int col = block_col * merge + in_col;
      double src_r = (double)row * (side - 1) / std::max(gh - 1, 1);
      double src_c = (double)col * (side - 1) / std::max(gw - 1, 1);
      int fr = (int)floor(src_r), fc = (int)floor(src_c);
      double wr[2] = {1.0 - (src_r - fr), src_r - fr};
      double wc[2] = {1.0 - (src_c - fc), src_c - fc};
      int tr[2] = {std::min(std::max(fr, 0), side - 1),
                   std::min(std::max(fr + 1, 0), side - 1)};
      int tc[2] = {std::min(std::max(fc, 0), side - 1),
                   std::min(std::max(fc + 1, 0), side - 1)};
      int t = 0;
      for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++) {
          idx[p * 4 + t] = tr[a] * side + tc[b];
          wts[p * 4 + t] = (float)(wr[a] * wc[b]);
          t++;
        }
    }
  }

  static void host_rope(int gh, int gw, std::vector<float>& cos_t,
                        std::vector<float>& sin_t) {
    const int merge = 2;
    int P = gh * gw;
    cos_t.resize(P * 72);
    sin_t.resize(P * 72);
    double inv_freq[18];
    for (int i = 0; i < 18; i++)
      inv_freq[i] = 1.0 / pow(10000.0, (2.0 * i) / 36.0);
    int blocks_w = gw / merge;
    for (int p = 0; p < P; p++) {
      int in_col = p % merge, in_row = (p / merge) % merge;
      int block_col = (p / (merge * merge)) % blocks_w;
      int block_row = p / (merge * merge * blocks_w);
      int row = block_row * merge + in_row;
      int col = block_col * merge + in_col;
      float rot[36];
      for (int i = 0; i < 18; i++) rot[i] = (float)(row * inv_freq[i]);
      for (int i = 0; i < 18; i++) rot[18 + i] = (float)(col * inv_freq[i]);
      for (int d = 0; d < 36; d++) {
        cos_t[p * 72 + d] = cos_t[p * 72 + 36 + d] = cosf(rot[d]);
        sin_t[p * 72 + d] = sin_t[p * 72 + 36 + d] = sinf(rot[d]);
      }
    }
  }

  // Full forward: patches fp32 host (P*1536, block-major (C,T,ph,pw) order,
  // preprocessing is the client's job) -> d_emb_bf16 [P/4, 2560]. Returns
  // M = P/4. dump_prefix != nullptr writes vit.cpp-compatible layer dumps
  // (offline --vision-test only; syncs the stream).
  int forward(const float* patches, int P, int gh, int gw,
              const char* dump_prefix = nullptr) {
    if (gh * gw != P || gh % 2 || gw % 2) {
      fprintf(stderr, "vision: bad grid (%d,%d) for P=%d\n", gh, gw, P);
      return -1;
    }
    ensure(P);
    int M = P / 4;
    std::string ddir;
    if (dump_prefix) {
      ddir = std::string(dump_prefix) + ".layers";
      mkdir(ddir.c_str(), 0777);
    }
    auto dump_f32 = [&](const char* name, const float* dev, uint64_t rows,
                        uint64_t cols) {
      if (!dump_prefix) return;
      std::vector<float> h(rows * cols);
      CK(hipMemcpy(h.data(), dev, h.size() * 4, hipMemcpyDeviceToHost));
      vit_npy_write_f32((ddir + "/" + name).c_str(), h.data(), rows, cols);
    };

    // rope cos/sin
    std::vector<float> hcos, hsin;
    host_rope(gh, gw, hcos, hsin);
    CK(hipMemcpy(d_cos, hcos.data(), hcos.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_sin, hsin.data(), hsin.size() * 4, hipMemcpyHostToDevice));
    if (dump_prefix) {
      vit_npy_write_f32((ddir + "/rope_cos.npy").c_str(), hcos.data(), P, 72);
      vit_npy_write_f32((ddir + "/rope_sin.npy").c_str(), hsin.data(), P, 72);
    }

    const float scale = powf(72.f, -0.5f);
    auto run_block = [&](int bi) {
      BlockW& B = blk[bi];
      k_vit_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, B.ln1_w, B.ln1_b, 1152,
                                            1e-6f);
      k_vit_f32_to_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_n, d_xbf, (size_t)P * 1152);
      proj_gemm(d_xbf, B.qkv_w, d_qkv, P, 3456, 1152);
      k_vit_bias_add<<<((size_t)P * 3456 + 255) / 256, 256, 0, g_str>>>(
          d_qkv, B.qkv_b, P, 3456);
      {
        dim3 grid(P, 16);
        k_vit_rope_qkv<<<grid, 36, 0, g_str>>>(d_qkv, d_cos, d_sin, d_qb,
                                               d_kb, d_vb, P);
      }
      {
        dim3 grid((P + 127) / 128, 16);
        k_vit_flash_wmma_db<<<grid, 256, 0, g_str>>>(d_qb, d_kb, d_vb, d_attn,
                                                     P, scale);
      }
      k_vit_f32_to_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_attn, d_xbf, (size_t)P * 1152);
      proj_gemm(d_xbf, B.proj_w, d_qkv, P, 1152, 1152);
      k_vit_bias_resadd<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_h, d_qkv, B.proj_b, P, 1152);
      k_vit_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, B.ln2_w, B.ln2_b, 1152,
                                            1e-6f);
      k_vit_f32_split_fp16<<<(P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_n, d_xbf, d_xlo, (size_t)P * 1152);
      proj_gemm_split(d_xbf, d_xlo, B.fc1_w, d_mlp, P, 4304, 1152);
      k_vit_bias_add<<<((size_t)P * 4304 + 255) / 256, 256, 0, g_str>>>(
          d_mlp, B.fc1_b, P, 4304);
      k_vit_gelu_tanh_split<<<((size_t)P * 4304 + 255) / 256, 256, 0, g_str>>>(
          d_mlp, d_xbf, d_xlo, (size_t)P * 4304);
      proj_gemm_split(d_xbf, d_xlo, B.fc2_w, d_qkv, P, 1152, 4304);
      k_vit_bias_resadd<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(
          d_h, d_qkv, B.fc2_b, P, 1152);
      if (dump_prefix) {
        char nm[32];
        snprintf(nm, sizeof nm, "block%02d.npy", bi);
        dump_f32(nm, d_h, P, 1152);
      }
    };

    // patch_embed: fp32 -> split-fp16, dual GEMM + bias
    CK(hipMemcpy(d_mlp, patches, (size_t)P * 1536 * 4,
                 hipMemcpyHostToDevice));  // d_mlp doubles as staging
    k_vit_f32_split_fp16<<<(P * 1536 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, d_xbf, d_xlo, (size_t)P * 1536);
    proj_gemm_split(d_xbf, d_xlo, patch_w, d_h, P, 1152, 1536);
    k_vit_bias_add<<<((size_t)P * 1152 + 255) / 256, 256, 0, g_str>>>(
        d_h, patch_b, P, 1152);
    dump_f32("patchembed.npy", d_h, P, 1152);

    // pos_embed gather
    {
      std::vector<int> hidx;
      std::vector<float> hwts;
      host_pos_interp(gh, gw, hidx, hwts);
      CK(hipMemcpy(d_idx, hidx.data(), hidx.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(d_wts, hwts.data(), hwts.size() * 4, hipMemcpyHostToDevice));
      k_vit_pos_embed<<<P, 256, 0, g_str>>>(d_h, pos_table, d_idx, d_wts, P);
    }
    dump_f32("embed.npy", d_h, P, 1152);

    for (int bi = 0; bi < 27; bi++) run_block(bi);

    // merger: LN -> fc1 -> erf-GELU -> fc2 (+bias), then bf16 RNE quantize
    k_vit_layernorm<<<P, 256, 0, g_str>>>(d_h, d_n, mg_ln_w, mg_ln_b, 1152,
                                          1e-6f);
    dump_f32("merger_ln.npy", d_n, M, 4608);
    k_vit_f32_to_fp16<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_n, d_xbf, (size_t)M * 4608);
    proj_gemm(d_xbf, mg_fc1_w, d_mlp, M, 4608, 4608);
    k_vit_bias_add<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, mg_fc1_b, M, 4608);
    dump_f32("merger_fc1.npy", d_mlp, M, 4608);
    k_vit_gelu_erf_fp16<<<((size_t)M * 4608 + 255) / 256, 256, 0, g_str>>>(
        d_mlp, d_xbf, (size_t)M * 4608);
    proj_gemm(d_xbf, mg_fc2_w, d_out, M, 2560, 4608);
    k_vit_bias_add<<<((size_t)M * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_out, mg_fc2_b, M, 2560);
    k_vit_f32_to_bf16<<<((size_t)M * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_out, d_emb_bf16, (size_t)M * 2560);
    dump_f32("merged.npy", d_out, M, 2560);
    if (dump_prefix) {
      std::vector<uint16_t> h((size_t)M * 2560);
      CK(hipMemcpy(h.data(), d_emb_bf16, h.size() * 2, hipMemcpyDeviceToHost));
      vit_npy_write_u16((ddir + "/embeds.bf16.npy").c_str(), h.data(), M, 2560);
    }
    return M;
  }
};

// --- model -------------------------------------------------------------------
struct GpuModel {
  const Checkpoint& ckpt;
  bool ple_on = true;
  int maxctx;
  int maxbatch;  // prefill chunk size: batch buffers are allocated at
                 // maxbatch_cap (maxbatch + tail-merge slack)
  int maxbatch_cap;
  int pos = 0;
  NgramMod ngram;
  // Adaptive ngram verify chunk rows, shared by both chain loops and kept
  // across requests (the content mix is a workload property, so warmup should
  // not repeat per request). 0 = the GDEC_NGRAM_CHUNK env is still unparsed.
  int ng_cs = 0;
  bool ng_cs_fixed = false;
  bool ng_legacy = false;  // GDEC_NGRAM_CHUNK=0: legacy single-pass verify
  void ng_chunk_init() {
    if (ng_cs) return;
    const char* e = getenv("GDEC_NGRAM_CHUNK");
    const int v = e ? atoi(e) : -1;
    ng_legacy = v == 0;
    ng_cs_fixed = v > 0;
    ng_cs = ng_cs_fixed ? std::max(1, std::min(64, v)) : 16;
  }

  // device state
  float *d_R, *d_Rhat, *d_x, *d_y, *d_emb;
  float *d_t320, *d_G, *d_w4;
  float *d_qkv, *d_g48, *d_beta48, *d_z;  // GDN scratch (z also used by QSA gate)
  float* d_S = nullptr;                   // (gdn_layers, 48, 128, 128)
  float* d_convst = nullptr;              // (gdn_layers, 3, 10240)
  float* d_convw = nullptr;               // (gdn_layers, 10240, 4) fp32
  float *d_Alog, *d_dtbias;               // (gdn_layers, 48)
  float* d_gdnnorm = nullptr;             // (gdn_layers, 128)
  float *d_kc = nullptr, *d_vc = nullptr;  // (qsa_layers, maxctx, 512), FP32 mode
  uint16_t *d_kcb = nullptr, *d_vcb = nullptr;  // BF16 KV cache (default mode)
  uint16_t *d_vcbt = nullptr;  // optional BF16 V cache: [block][kvh][dim][slot]
  float *d_qg, *d_qs, *d_gs, *d_kb, *d_vb, *d_attn;
  float *d_pacc = nullptr, *d_pml = nullptr;  // QSA decode flash-decoding partials
  float *d_iproj = nullptr, *d_iq = nullptr, *d_ik = nullptr;
  float *d_iraw = nullptr, *d_inorm = nullptr, *d_iscores = nullptr;
  float* d_iw = nullptr;
  // Block ranking is sensitive to projection rounding; retain FP32 here.
  bool index_f32 = getenv("GDEC_INDEX_BF16") == nullptr;
  bool index_fused2 = getenv("GDEC_INDEX_FUSED2") != nullptr;
  bool index_stream_select = getenv("GDEC_INDEX_STREAM_SELECT") != nullptr;
  int* d_iselected = nullptr;
  int index_blocks = 0;
  bool qsa_dense = getenv("GDEC_QSA_DENSE") != nullptr;
  // BF16 KV cache: measured parity at 32K (QSA is latency-bound, not DRAM-bound)
  // and flips the 8192 oracle -> opt-in experiment, default stays FP32.
  bool qsa_kv_fp32 = getenv("GDEC_QSA_KV_BF16") == nullptr;  // default: FP32 cache
  // WMMA sparse prefill attention (k_qsa_wmma): opt-in on top of BF16 KV mode.
  // bf16 P + WMMA accumulation is not bitwise fp32 — FP32 kernels stay default.
  bool qsa_wmma = getenv("GDEC_QSA_WMMA") != nullptr;
  // Keep the established row-major cache for decode, MTP, snapshots, and
  // the default WMMA path; this second cache is a prefill-only opt-in.
  bool qsa_wmma_btv = qsa_wmma && !qsa_kv_fp32 &&
                      getenv("GDEC_QSA_WMMA_BTV") != nullptr;
  // GDEC_QSA_WMMA6=1: dim-split 128-thread rewrite (k_qsa_wmma6); layered on
  // the transposed-V path since it reads B fragments from that cache.
  bool qsa_wmma6 = qsa_wmma_btv && getenv("GDEC_QSA_WMMA6") != nullptr;
  size_t qsa_vt_blocks = 0;
  bool prefill_fused = getenv("GDEC_PREFILL_UNFUSED") == nullptr;
  bool gr_bf16 = getenv("GDEC_GR_BF16") != nullptr && prefill_fused;
  bool gr_scatter_norm = getenv("GDEC_GR_SCATTER_NORM") != nullptr || gr_bf16;
  bool moe_deterministic = !qsa_dense && getenv("GDEC_MOE_ATOMIC") == nullptr;
  float* d_router;                        // 512
  int* d_topids;                          // 16
  float* d_topw;                          // 16
  float *d_guv, *d_hid, *d_ey, *d_moeacc, *d_sg;
  float *d_e, *d_keys, *d_v, *d_kn, *d_qn, *d_U, *d_ring, *d_convout;
  float* d_ple_convw;                     // 10240*4
  float *d_ple_nk, *d_ple_nq, *d_ple_nc;  // 10240 each
  float* d_logits;
  float* d_logits_b = nullptr;  // [65][vocab] speculative verify lm_head scratch
  int* d_argmax;
  float* d_hnorm_mixer;  // 10240

  // ---- prefill batch state (Phase 3a), sized maxctx ----
  rocblas_handle rbh = nullptr;
  hipblasLtHandle_t lth = nullptr;   // prefill GEMMs (Phase 3e)
  void* d_ltws = nullptr;            // hipBLASLt workspace
  size_t lt_ws_bytes = (size_t)64 << 20;
  // (N,K,P[,flags]) -> full LT descriptor set. Descriptors are shape-implied
  // and self-contained, so cache them for process lifetime instead of
  // create/destroy per call (~55 ms per 8-token chunk of pure host time).
  struct LtEntry {
    hipblasLtMatmulAlgo_t algo;
    hipblasLtMatmulDesc_t opd = nullptr;
    hipblasLtMatrixLayout_t Al = nullptr, Bl = nullptr, Cl = nullptr,
                          Dl = nullptr;
  };
  std::unordered_map<uint64_t, LtEntry> lt_algos;
  int* d_posarr;   // [maxctx] d_posarr[i] = i
  int* d_ringarr;  // [maxctx] d_ringarr[i] = i % 9
  int* d_tokarr;   // [maxctx] prefill token ids
  float2* d_ropecs = nullptr;  // [maxbatch][32] per-token rope cos/sin table
  int ropecs_base_ = -1, ropecs_P_ = -1;  // cache key of the last k_rope_cs fill
  unsigned ropecs_gen_ = 0;  // mrope state generation, part of the cache key
  // ---- M-RoPE (vision) request state ----
  // Empty mrope_pos_ = plain text request: rope is untouched (device-table
  // fill, decode delta 0) and bitwise identical to the pre-mrope engine.
  bool mrope_active_ = false;
  int rope_delta_ = 0;                       // host mirror of d_rdelta
  unsigned mrope_gen_ = 0;                   // bumps on set/clear
  std::vector<std::array<int, 3>> mrope_pos_;  // per-prompt-token [T,H,W]
  std::vector<std::array<int, 3>> mrope_grids_;  // request grids (kvsnap meta)
  std::vector<float2> h_ropecs_;             // host fill staging
  int* d_rdelta = nullptr;                   // device scalar: decode rope delta
  float2* d_cs1 = nullptr;                   // dump scratch (32 cos/sin)
  int* d_rpos1 = nullptr;                    // dump scratch (rope position)
  FILE* mrope_dump_ = nullptr;               // GDEC_MROPE_DUMP sink
  bool mrope_dump_meta_ = false;
  // ---- Vision tower (stage 4) ----
  // vision_on only via load_vision (--vision-tower). Per-request state lives
  // and dies with the mrope request lifecycle (clear_mrope drops it): the
  // stitched bf16 embeds and the (src row, prompt pos) injection map.
  VisionTower vt;
  bool vision_on = false;
  uint16_t* d_img_emb = nullptr;   // [img_emb_rows, 2560] bf16
  size_t img_emb_rows = 0, img_emb_cap = 0;
  std::vector<int2> img_inj;       // per image token: (embed row, prompt pos)
  uint64_t img_hash_ = 0;          // FNV-1a of the request's patch bytes (cont guard)
  int2* d_img_inj = nullptr;       // [maxbatch] chunk-local upload scratch
  std::vector<int2> h_img_inj;     // host staging for the chunk slice
  float *d_Rb, *d_Rhatb, *d_xb, *d_yb, *d_Gb, *d_t320b, *d_w4b;
  float *d_qkvb, *d_zb, *d_g48b, *d_beta48b;
  float *d_qgb, *d_qsb, *d_gsb, *d_kbb, *d_vbb, *d_attnb;
  float* d_routerb;
  int* d_topidsb;
  float* d_topwb;
  float *d_guvb, *d_hidb, *d_moeaccb, *d_eyb, *d_sgb;
  float* d_eb;              // [maxctx, 2560] PLE embeddings
  float* ple_eb = nullptr;  // pinned staging [maxctx, 2560]
  uint8_t* ple_eb_dev = nullptr;  // device alias of ple_eb (zero-copy fp8 staging)
  uint16_t* d_wbf16;        // dequant workspace (64 MiB)
  uint16_t* d_xbf16;        // [maxctx, 10240] bf16 activations
  uint16_t* d_Rhatbf16 = nullptr;  // survives intermediate projection conversions
  // Non-cacheable gemm inputs (d_hidb, d_t320b, attn/conv outs) convert here so
  // they don't evict the d_xb/d_Rhatb cache entries in d_xbf16.
  uint16_t* d_xbf16b = nullptr;  // [maxctx, 6144] bf16
  // GDEC_MOE_LT buffers (allocated only when enabled): gathered bf16 pair rows
  // and per-group dequant scratch.
  __hip_bfloat16* d_moexg = nullptr;   // [BP*topk, d]
  __hip_bfloat16* d_moewup = nullptr;  // [MOE_LT_GRP(x2 if OVL), 2*moe_mid, d]
  __hip_bfloat16* d_moewdn = nullptr;  // [MOE_LT_GRP(x2 if OVL), d, moe_mid]
  uint16_t* d_hidb2 = nullptr;  // GDEC_MOE_LT_CMP: fused-reference packed hid
  float* d_pairs2 = nullptr;    // GDEC_MOE_LT_CMP: fused-reference weighted pairs
  int* d_cmpi = nullptr;        // GDEC_MOE_LT_CMP: [max|diff|, max|ref|] ordered int
  // bf16 conversion cache: d_xb/d_Rhatb are converted once per layer section
  // instead of once per gemm; invalidated where those buffers are written
  const float* xf16_src = nullptr;
  uint64_t xf16_xs = 0;
  int xf16_P = 0, xf16_K = 0;
  // Expert-bucketed MoE with stable GPU routing and independently scheduled tiles.
  int* d_tokidx;       // [maxctx*topk] expert-sorted token indices
  float* d_pweight;    // [maxctx*topk] expert-sorted router weights
  int* d_eoff;         // [experts+1] expert pair offsets
  int* d_pairids = nullptr;  // [maxctx*topk] pair rows in ascending expert order
  GpuMoeRouting moe_routing;
  MoeTile* d_moetiles = nullptr;
  int* d_moentiles = nullptr;
  // GDN chunked prefill (Phase 3b)
  float* d_convb;    // [maxctx, 10240] post-conv qkv (conv is not in-place)
  float* d_gdn_ws;   // [48, 2, 64, 128] per-head chunk scratch
  float* d_gdn_split_ws = nullptr;  // [nchunks, 48] intra->inter staging (80.25 KiB each)
  // PLE batched prefill (Phase 3d-1)
  float* d_keysb;     // [maxctx, 10240] key_proj out, normed in place
  float* d_plevb;     // [maxctx, 2560] value_proj out
  float* d_qnb;       // [maxctx, 10240] normed R rows
  float* d_Ub;        // [maxctx, 10240] gated values (pre-norm)
  float* d_Unb;       // [maxctx, 10240] normed U (conv input / ring source)
  float* d_convoutb;  // [maxctx, 10240] conv outputs

  // PLE host state
  std::vector<int64_t> chist;
  int64_t ple_mult[3], ple_vsz[16], ple_voff[16];
  const Tensor* ple_table = nullptr;
  float ple_tscale = 1.f;
  int ple_ringpos = 0;           // host mirror for direct mode
  int* d_token = nullptr;        // device scalars for graph-replayable kernels
  int* d_pos = nullptr;
  int* d_ringpos = nullptr;
  int* p_token = nullptr;        // pinned staging for token id
  int pad_id = 248044;

  // ---- MTP speculative draft head (Phase A) ----
  // Single QSA+MoE layer on top of the trunk's 48-layer stream; predicts
  // t_{p+2} from (trunk h_p, emb(t_{p+1})) (DeepSeek-style wiring). Owns its
  // stream/KV/indexer state; reuses trunk scratch (strictly serial use).
  bool mtp_avail = false;      // checkpoint has mtp.* tensors and env is default
  bool ngram_verify_active = false;
  bool ngram_legacy_gdn = getenv("GDEC_NGRAM_LEGACY_GDN") != nullptr;
  bool ngram_legacy_final = getenv("GDEC_NGRAM_LEGACY_FINAL") != nullptr;
  bool mtp_tap_capture = false;  // prefill_chunk copies d_Rb rows into d_mtap
  int mtp_norm_mode = 0;       // 0 = full-10240 RMSNorm (H1), 1 = 4x2560 (H2)
  int mtp_hagg_mode = 0;       // 0 = sum branches + repeat4, 1 = per-branch + bcast e
  int mtp_pos = 0;             // host mirror of the MTP stream position
  float* d_mR = nullptr;       // MTP hyper stream (10240)
  float* d_mRhat = nullptr;    // MTP normed stream (10240)
  float* d_mh = nullptr;       // input h-path accumulator (2560)
  float* d_me = nullptr;       // input e-path normed embed (2560)
  float* d_meo = nullptr;      // input e-path fc_embedding out (2560)
  float* d_mt = nullptr;       // input per-branch gemv tmp (2560)
  float* d_mnorm_e = nullptr;  // mtp.pre_fc_norm_embedding [2560] fp32
  float* d_mnorm_h = nullptr;  // mtp.pre_fc_norm_hidden [10240] fp32
  int* d_mtok = nullptr;       // device scalar: draft input token
  int* d_mpos = nullptr;       // device scalar: MTP position
  float* d_mkcf = nullptr;     // MTP KV, FP32 mode (maxctx, 512)
  float* d_mvcf = nullptr;
  uint16_t* d_mkcb = nullptr;  // MTP KV, BF16 mode
  uint16_t* d_mvcb = nullptr;
  float* d_mik = nullptr;      // MTP indexer pooled keys (index_blocks, 128)
  float* d_miraw = nullptr;    // MTP indexer raw ring (4, 128)
  float* d_minorm = nullptr;   // MTP indexer norms q[128] + k[128]
  float* d_miw = nullptr;      // MTP indexer fp32 proj (640, 2560)
  float* d_mRb = nullptr;      // batch stream (maxbatch, 10240)
  float* d_mRhatb = nullptr;   // batch normed stream
  float* d_mtap = nullptr;     // captured trunk taps (maxbatch, 10240)
  // Stats for the most recent speculative request, emitted on the terminal
  // serve D line so clients can distinguish MTP from serial decode.
  int last_spec_rounds = 0;
  int last_spec_commit = 0;
  int last_spec_rollbacks = 0;
  int last_spec_proposed = 0;  // sum of drafts offered (D line trailing field)
  // Live progress reporting (2026-09-13): 1s-cadence prefill/decode stats on
  // stderr. Prefill is armed by serve around the prompt ingest only (never
  // for spec-verify batches) and uses NON-BLOCKING event probes — no stream
  // sync, so the pipeline is undisturbed. Decode reads the live spec
  // counters the loops bump every round.
  int prog_total = 0;         // prompt tokens to prefill this request (0=off)
  int64_t prog_base = 0;      // prompt tokens completed before current chunk
  int64_t prog_last_est = 0;  // est tokens at the last report (interval rate)
  int64_t prog_req = -1;      // request id stamped on progress lines
  std::chrono::steady_clock::time_point prog_t0, prog_last;
  hipEvent_t prog_ev[64] = {};  // per-layer completion probes (DisableTiming)
  int live_spec_rounds = 0;
  int live_spec_commits = 0;
  int live_spec_proposed = 0;
  // Trunk rollback checkpoint for verify: S + convst + PLE ring + trunk iraw,
  // plus a separate tiny slot for the MTP indexer ring (saved post-draft-1).
  float* d_ckpt = nullptr;
  size_t ckpt_S = 0, ckpt_convst = 0, ckpt_ring = 0, ckpt_iraw = 0;
  float* d_mring_ckpt = nullptr;  // 512 floats
  int* d_argmax_arr = nullptr;    // per-row verify argmax (<= 65 rows)
  // Rollback snapshots (spec verify): S/convst per accepted depth, indexer
  // proj rows per QSA layer. Written only when spec_snap is set.
  float* d_Ssnap = nullptr;    // [maxbatch+1][ngdn][48*128*128]
  float* d_csnap = nullptr;    // [maxbatch+1][ngdn][3*10240]
  float* d_iproj_snap = nullptr;  // [nqsa][maxbatch+1][128]
  bool spec_snap = false;         // set around spec verify chunks only

  explicit GpuModel(const Checkpoint& c, int mc) : ckpt(c), maxctx(mc) {
    // Prefill workspace is sized by maxbatch, NOT maxctx: long prompts are
    // processed in maxbatch-token chunks via the same continuation path as
    // KV reuse, so a 256K context costs chunk-sized workspace (~4.5 GB at
    // 8192) instead of ~541 KB/token of context.
    maxbatch = eff_maxbatch(maxctx);
    maxbatch_cap = std::min(maxctx, maxbatch + eff_tail_slack());
    int rb = g_cfg.branches * g_cfg.d;
    DALLOC(&d_R, rb * 4);
    DALLOC(&d_Rhat, rb * 4);
    DALLOC(&d_x, rb * 4);
    DALLOC(&d_y, rb * 4);
    DALLOC(&d_emb, g_cfg.d * 4);
    DALLOC(&d_t320, g_cfg.gr_rank * 4);
    DALLOC(&d_G, rb * 4);
    DALLOC(&d_w4, 16);
    DALLOC(&d_qkv, 10240 * 4);
    DALLOC(&d_g48, 48 * 4);
    DALLOC(&d_beta48, 48 * 4);
    DALLOC(&d_z, 6144 * 4);
    int ngdn = g_cfg.layers - g_cfg.layers / 4;  // 36
    DALLOC(&d_S, (size_t)ngdn * 48 * 128 * 128 * 4);
    CK(hipMemset(d_S, 0, (size_t)ngdn * 48 * 128 * 128 * 4));
    DALLOC(&d_convst, (size_t)ngdn * 3 * 10240 * 4);
    CK(hipMemset(d_convst, 0, (size_t)ngdn * 3 * 10240 * 4));
    DALLOC(&d_convw, (size_t)ngdn * 10240 * 4 * 4);
    DALLOC(&d_Alog, (size_t)ngdn * 48 * 4);
    DALLOC(&d_dtbias, (size_t)ngdn * 48 * 4);
    DALLOC(&d_gdnnorm, (size_t)ngdn * 128 * 4);
    int nqsa = g_cfg.layers / 4;  // 12
    if (!qsa_dense) {
      index_blocks = (maxctx + 3) / 4;
      DALLOC(&d_iproj, (size_t)maxctx * 640 * 4);
      DALLOC(&d_iq, (size_t)maxctx * 512 * 4);
      DALLOC(&d_ik, (size_t)nqsa * index_blocks * 128 * 4);
      DALLOC(&d_iraw, (size_t)nqsa * 4 * 128 * 4);
      CK(hipMemset(d_iraw, 0, (size_t)nqsa * 4 * 128 * 4));
      DALLOC(&d_inorm, (size_t)nqsa * 256 * 4);
      if (index_f32) DALLOC(&d_iw, (size_t)nqsa * 640 * 2560 * 4);
      DALLOC(&d_iscores, (size_t)std::min(maxctx, INDEX_BATCH) * 4 * index_blocks * 4);
      DALLOC(&d_iselected, (size_t)maxctx * INDEX_BUDGET * 4);
    }
    if (qsa_kv_fp32) {
      DALLOC(&d_kc, (size_t)nqsa * maxctx * 512 * 4);
      DALLOC(&d_vc, (size_t)nqsa * maxctx * 512 * 4);
    } else {
      DALLOC(&d_kcb, (size_t)nqsa * maxctx * 512 * 2);
      DALLOC(&d_vcb, (size_t)nqsa * maxctx * 512 * 2);
      if (qsa_wmma_btv) {
        qsa_vt_blocks = (size_t)(maxctx + 3) / 4;
        DALLOC(&d_vcbt, (size_t)nqsa * qsa_vt_blocks * 2 * 256 * 4 * 2);
      }
    }
    DALLOC(&d_qg, 12288 * 4);
    DALLOC(&d_qs, 24 * 256 * 4);
    DALLOC(&d_gs, 24 * 256 * 4);
    DALLOC(&d_kb, 512 * 4);
    DALLOC(&d_vb, 512 * 4);
    DALLOC(&d_attn, 24 * 256 * 4);
    DALLOC(&d_pacc, (size_t)2 * QSA_DEC_NSPLIT * 12 * 256 * 4);
    DALLOC(&d_pml, (size_t)2 * QSA_DEC_NSPLIT * 12 * 2 * 4);
    DALLOC(&d_router, 512 * 4);
    DALLOC(&d_topids, 16 * 4);
    DALLOC(&d_topw, 16 * 4);
    DALLOC(&d_guv, 16 * 1280 * 4);
    DALLOC(&d_hid, 16 * 640 * 4);
    DALLOC(&d_ey, g_cfg.d * 4);
    DALLOC(&d_moeacc, g_cfg.d * 4);
    DALLOC(&d_sg, 4);
    DALLOC(&d_e, 2560 * 4);
    CK(hipHostMalloc((void**)&ple_e, 2560 * 4, hipHostMallocDefault));
    DALLOC(&d_token, 4);
    DALLOC(&d_pos, 4);
    DALLOC(&d_ringpos, 4);
    DALLOC(&d_rdelta, 4);
    CK(hipMemset(d_pos, 0, 4));
    CK(hipMemset(d_ringpos, 0, 4));
    CK(hipMemset(d_rdelta, 0, 4));
    if (getenv("GDEC_MROPE_DUMP")) {
      DALLOC(&d_cs1, 32 * sizeof(float2));
      DALLOC(&d_rpos1, 4);
    }
    CK(hipHostMalloc((void**)&p_token, 4, hipHostMallocDefault));
    DALLOC(&d_keys, 10240 * 4);
    DALLOC(&d_v, 2560 * 4);
    DALLOC(&d_kn, 10240 * 4);
    DALLOC(&d_qn, 10240 * 4);
    DALLOC(&d_U, 10240 * 4);
    DALLOC(&d_ring, (size_t)9 * 10240 * 4);
    CK(hipMemset(d_ring, 0, (size_t)9 * 10240 * 4));
    DALLOC(&d_convout, 10240 * 4);
    DALLOC(&d_ple_convw, 10240 * 4 * 4);
    DALLOC(&d_ple_nk, 10240 * 3 * 4);
    DALLOC(&d_logits, (size_t)g_cfg.vocab * 4);
    // Ngram verifies up to 64 drafts plus the current token; MTP needs nine
    // rows at most. Keep logits independent of the much larger prefill chunk.
    DALLOC(&d_logits_b, (size_t)g_cfg.vocab * 4 * 65);
    DALLOC(&d_argmax_arr, 65 * sizeof(int));
    // Trunk checkpoint belongs to the verifier, independently of MTP weights.
    {
      int ngdn = g_cfg.layers - g_cfg.layers / 4;
      ckpt_S = (size_t)ngdn * 48 * 128 * 128 * 4;
      ckpt_convst = (size_t)ngdn * 3 * 10240 * 4;
      ckpt_ring = (size_t)9 * 10240 * 4;
      ckpt_iraw = qsa_dense ? 0 : (size_t)(g_cfg.layers / 4) * 4 * 128 * 4;
      DALLOC(&d_ckpt, ckpt_S + ckpt_convst + ckpt_ring + ckpt_iraw);
    }
    DALLOC(&d_argmax, 4);
    DALLOC(&d_hnorm_mixer, 10240 * 4);

    // ---- prefill batch buffers (Phase 3a), allocated once at maxctx ----
    {
      size_t BP = (size_t)maxbatch_cap;
      DALLOC(&d_Rb, BP * 10240 * 4);
      DALLOC(&d_Rhatb, BP * 10240 * 4);
      DALLOC(&d_xb, BP * 2560 * 4);
      DALLOC(&d_yb, BP * 2560 * 4);
      DALLOC(&d_Gb, BP * 10240 * 4);
      DALLOC(&d_t320b, BP * 320 * 4);
      DALLOC(&d_w4b, BP * 4 * 4);
      DALLOC(&d_qkvb, BP * 10240 * 4);
      DALLOC(&d_zb, BP * 6144 * 4);
      DALLOC(&d_g48b, BP * 48 * 4);
      DALLOC(&d_beta48b, BP * 48 * 4);
      DALLOC(&d_qgb, BP * 12288 * 4);
      DALLOC(&d_qsb, BP * 6144 * 4);
      DALLOC(&d_gsb, BP * 6144 * 4);
      DALLOC(&d_kbb, BP * 512 * 4);
      DALLOC(&d_vbb, BP * 512 * 4);
      DALLOC(&d_ropecs, BP * 32 * sizeof(float2));
      DALLOC(&d_attnb, BP * 6144 * 4);
      DALLOC(&d_routerb, BP * 512 * 4);
      DALLOC(&d_topidsb, BP * 16 * 4);
      DALLOC(&d_topwb, BP * 16 * 4);
      // Fused down reuses the otherwise idle gate/up workspace for pair outputs.
      DALLOC(&d_guvb, BP * g_cfg.topk * (moe_deterministic ? 2560 : 1280) * 4);
      DALLOC(&d_hidb, BP * g_cfg.topk * 640 * 4);
      DALLOC(&d_moeaccb, BP * 2560 * 4);
      DALLOC(&d_eyb, BP * 2560 * 4);
      DALLOC(&d_sgb, BP * 4);
      DALLOC(&d_eb, BP * 2560 * 4);
      CK(hipHostMalloc((void**)&ple_eb, BP * 2560 * 4, hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ple_eb_dev, ple_eb, 0));
      DALLOC(&d_wbf16, (size_t)64 << 20);  // fits q_proj 12288x2560 bf16
      DALLOC(&d_xbf16, BP * 10240 * 2);
      DALLOC(&d_xbf16b, BP * 6144 * 2);  // largest non-cacheable K = 6144
      if (prefill_fused) DALLOC(&d_Rhatbf16, BP * 10240 * 2);
      DALLOC(&d_tokidx, BP * g_cfg.topk * 4);
      DALLOC(&d_pweight, BP * g_cfg.topk * 4);
      DALLOC(&d_eoff, (g_cfg.experts + 1) * 4);
      if (moe_deterministic) DALLOC(&d_pairids, BP * g_cfg.topk * 4);
      if (!moe_host_route && !moe_ordered) moe_routing.init(BP * g_cfg.topk, g_cfg.experts);
      if (!moe_untiled && !moe_ordered) {
        DALLOC(&d_moetiles, ((BP * g_cfg.topk + 63) / 64 + g_cfg.experts) * sizeof(MoeTile));
        DALLOC(&d_moentiles, sizeof(int));
      }
      if (moe_lt && g_cfg.topk == 10 && moe_deterministic && !moe_host_route && !moe_ordered) {
        // 4-expert dequant groups: scratch stays MALL-resident between dequant
        // and its GEMMs (measured in tools/moe_lt_bench.cu). GDEC_MOE_LT_OVL
        // doubles it for the A/B slot rotation of the overlap pipeline.
        DALLOC(&d_moexg, (size_t)BP * g_cfg.topk * g_cfg.d * 2);
        DALLOC(&d_moewup, (size_t)4 * 2 * g_cfg.moe_mid * g_cfg.d * 2 * (moe_lt_ovl ? 2 : 1));
        DALLOC(&d_moewdn, (size_t)4 * g_cfg.d * g_cfg.moe_mid * 2 * (moe_lt_ovl ? 2 : 1));
        if (moe_lt_cmp) {
          DALLOC(&d_hidb2, (size_t)BP * g_cfg.topk * g_cfg.moe_mid * 2);
          DALLOC(&d_pairs2, (size_t)BP * g_cfg.topk * g_cfg.d * 4);
          DALLOC(&d_cmpi, 8);
        }
      }
      DALLOC(&d_convb, BP * 10240 * 4);
      DALLOC(&d_gdn_ws, (size_t)48 * 2 * 64 * 128 * 4);
      if (!gdn_nosplit && !gdn_loop)  // streaming caps the workspace at window chunks (15.05 MiB @4)
        DALLOC(&d_gdn_split_ws,
                     (size_t)(gdn_stream ? std::min<size_t>(gdn_window_chunks, (BP + 63) / 64)
                                         : (BP + 63) / 64) * 48 * GDN_SPLIT_WS_FLOATS * 4);
      DALLOC(&d_keysb, BP * 10240 * 4);
      DALLOC(&d_plevb, BP * 2560 * 4);
      DALLOC(&d_qnb, BP * 10240 * 4);
      DALLOC(&d_Ub, BP * 10240 * 4);
      DALLOC(&d_Unb, BP * 10240 * 4);
      DALLOC(&d_convoutb, BP * 10240 * 4);
      // d_posarr/d_ringarr are indexed by ABSOLUTE position (qsa_loop debug,
      // ple_gpu_t): they stay maxctx-sized. d_tokarr only ever holds one
      // prefill chunk.
      DALLOC(&d_posarr, (size_t)maxctx * 4);
      DALLOC(&d_ringarr, (size_t)maxctx * 4);
      DALLOC(&d_tokarr, BP * 4);
      DALLOC(&d_img_inj, BP * sizeof(int2));  // vision injection map
      std::vector<int> h(maxctx);
      for (int i = 0; i < maxctx; i++) h[i] = i;
      CK(hipMemcpy(d_posarr, h.data(), (size_t)maxctx * 4, hipMemcpyHostToDevice));
      for (int i = 0; i < maxctx; i++) h[i] = i % 9;
      CK(hipMemcpy(d_ringarr, h.data(), (size_t)maxctx * 4, hipMemcpyHostToDevice));
      if (rocblas_create_handle(&rbh) != rocblas_status_success) {
        fprintf(stderr, "rocblas_create_handle failed\n");
        exit(1);
      }
      if (hipblasLtCreate(&lth) != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "hipblasLtCreate failed\n");
        exit(1);
      }
      DALLOC(&d_ltws, lt_ws_bytes);
    }

    // ---- upload small fp32 weights (norms, conv, A_log, dt_bias) ----
    char buf[128];
    std::vector<float> tmp;
    auto upload = [&](const char* name, float* dst, size_t n) {
      tmp.resize(n);
      ckpt.dequant(ckpt.at(name), tmp.data());
      CK(hipMemcpy(dst, tmp.data(), n * 4, hipMemcpyHostToDevice));
    };
    int gi = 0;
    for (int l = 0; l < g_cfg.layers; l++) {
      if (is_qsa(l)) {
        if (!qsa_dense) {
          snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.q_layernorm.weight", l);
          upload(buf, d_inorm + (l / 4) * 256, 128);
          snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.k_layernorm.weight", l);
          upload(buf, d_inorm + (l / 4) * 256 + 128, 128);
          if (index_f32) {
            snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
            upload(buf, d_iw + (size_t)(l / 4) * 640 * 2560, 640 * 2560);
          }
        }
        continue;
      }
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.conv1d.weight", l);
      upload(buf, d_convw + (size_t)gi * 10240 * 4, 10240 * 4);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.A_log", l);
      upload(buf, d_Alog + (size_t)gi * 48, 48);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.dt_bias", l);
      upload(buf, d_dtbias + (size_t)gi * 48, 48);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.norm.weight", l);
      upload(buf, d_gdnnorm + (size_t)gi * 128, 128);
      gi++;
    }
    upload("hyper_connection_mixer.hc_norm.weight", d_hnorm_mixer, 10240);
    upload("layers.1.ple.conv1d.weight", d_ple_convw, 10240 * 4);
    upload("layers.1.ple.norm_key.weight", d_ple_nk, 10240);
    upload("layers.1.ple.norm_query.weight", d_ple_nk + 10240, 10240);
    upload("layers.1.ple.norm_conv.weight", d_ple_nk + 20480, 10240);

    // PLE config tensors
    {
      const Tensor& mt = ckpt.at("layers.1.ple.ple_embedding.layer_multipliers");
      memcpy(ple_mult, mt.data, 24);
      const Tensor& vs = ckpt.at("layers.1.ple.ple_embedding.ngram_heads_vocab_sizes");
      memcpy(ple_vsz, vs.data, 128);
      const Tensor& vo = ckpt.at("layers.1.ple.ple_embedding.ngram_heads_offsets");
      memcpy(ple_voff, vo.data, 128);
      ple_table = &ckpt.at("layers.1.ple.ngram_embedding.weight");
      memcpy(&ple_tscale, ple_table->data + ple_table->numel(), 4);
      // io_uring path: file offset of the PLE table in the base .hgn. The fd
      // is the base file, so only the base mapping (maps from file offset 0)
      // is meaningful — if an overlay shadows the table, fall back.
#ifdef _WIN32
      if (g_ple_win.fh != INVALID_HANDLE_VALUE && !ckpt.mappings().empty()) {
#else
      if (g_ple_fd >= 0 && !ckpt.mappings().empty()) {
#endif
        const auto& mp0 = ckpt.mappings()[0];
        if (ple_table->data >= mp0.base && ple_table->data < mp0.base + mp0.len) {
          g_ple_foff = (uint64_t)(ple_table->data - mp0.base);
          g_ple_off_ok = true;
        }
      }
    }

    // ---- MTP draft head: detect weights, allocate own state ----
    if (ckpt.find("mtp.fc_hidden.weight")) {
      // moe_lt is compatible: it only activates at P >= moe_lt_min (4096) in
      // the trunk batch MoE, while MTP draft/ingest/verify run their own
      // fixed kernels at P <= gamma+1 <= 9. gr_bf16 taps are converted
      // bf16->f32 at capture/rollback/ingest (see spec_taps/restore_dR_row).
      bool moe_default = moe_deterministic && !moe_ordered && !moe_host_route &&
                         !moe_untiled && !moe_fp32_io;
      if (!moe_default) {
        fprintf(stderr, "mtp: disabled (moe_default=%d)\n", (int)moe_default);
      } else {
        // Validated wiring (mtp-test 0.855): branch-grouped norm + per-branch
        // fc_hidden + broadcast e. Env flips back for debugging only.
        const char* nm = getenv("GDEC_MTP_NORM");
        mtp_norm_mode = (nm && !strcmp(nm, "full")) ? 0 : 1;
        const char* hg = getenv("GDEC_MTP_HAGG");
        mtp_hagg_mode = (hg && !strcmp(hg, "sum")) ? 0 : 1;
        DALLOC(&d_mR, rb * 4);
        DALLOC(&d_mRhat, rb * 4);
        DALLOC(&d_mh, g_cfg.d * 4);
        DALLOC(&d_me, g_cfg.d * 4);
        DALLOC(&d_meo, g_cfg.d * 4);
        DALLOC(&d_mt, g_cfg.d * 4);
        DALLOC(&d_mtok, 4);
        DALLOC(&d_mpos, 4);
        DALLOC(&d_mnorm_e, g_cfg.d * 4);
        DALLOC(&d_mnorm_h, rb * 4);
        upload("mtp.pre_fc_norm_embedding.weight", d_mnorm_e, g_cfg.d);
        upload("mtp.pre_fc_norm_hidden.weight", d_mnorm_h, rb);
        if (qsa_kv_fp32) {
          DALLOC(&d_mkcf, (size_t)maxctx * 512 * 4);
          DALLOC(&d_mvcf, (size_t)maxctx * 512 * 4);
        } else {
          DALLOC(&d_mkcb, (size_t)maxctx * 512 * 2);
          DALLOC(&d_mvcb, (size_t)maxctx * 512 * 2);
        }
        DALLOC(&d_mRb, (size_t)maxbatch_cap * 10240 * 4);
        DALLOC(&d_mRhatb, (size_t)maxbatch_cap * 10240 * 4);
        DALLOC(&d_mtap, (size_t)maxbatch_cap * 10240 * 4);
        {
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          DALLOC(&d_mring_ckpt, 512 * 4);
          // Rollback snapshots: per-row GDN S / convst / indexer-proj state
          // captured during a spec verify chunk, indexed by accepted depth.
          // Spec chunks are gamma+1 <= 9 rows (gamma <= 8), not maxbatch.
          DALLOC(&d_Ssnap, (size_t)9 * ngdn * 48 * 128 * 128 * 4);
          DALLOC(&d_csnap, (size_t)9 * ngdn * 3 * 10240 * 4);
          DALLOC(&d_iproj_snap, (size_t)(g_cfg.layers / 4) * 9 * 128 * 4);
        }
        if (!qsa_dense) {
          DALLOC(&d_mik, (size_t)index_blocks * 128 * 4);
          DALLOC(&d_miraw, 4 * 128 * 4);
          CK(hipMemset(d_miraw, 0, 4 * 128 * 4));
          DALLOC(&d_minorm, 256 * 4);
          upload("mtp.layers.0.self_attn.indexer.q_layernorm.weight", d_minorm, 128);
          upload("mtp.layers.0.self_attn.indexer.k_layernorm.weight", d_minorm + 128,
                 128);
          if (index_f32) {
            DALLOC(&d_miw, (size_t)640 * 2560 * 4);
            upload("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", d_miw,
                   (size_t)640 * 2560);
          }
        }
        mtp_avail = true;
        fprintf(stderr,
                "mtp: draft head enabled (norm=%s hagg=%s kv=%s fc_hidden dtype=%d)\n",
                mtp_norm_mode ? "branch" : "full",
                mtp_hagg_mode ? "branch" : "sum",
                qsa_kv_fp32 ? "fp32" : "bf16",
                (int)ckpt.at("mtp.fc_hidden.weight").dtype);
      }
    }
    fprintf(stderr, "gdec: state allocated (maxctx=%d)\n", maxctx);
  }

  ~GpuModel() {
    if (rbh) rocblas_destroy_handle(rbh);
    if (lth) hipblasLtDestroy(lth);
#ifndef _WIN32
    // Windows 下 d_Rhatbf16 来自 devarena bump 分配，不能单独 hipFree
    if (d_Rhatbf16) CK(hipFree(d_Rhatbf16));
#endif
  }

  // GR read: R -> Rhat -> t320 -> G -> x. prefix like
  // "layers.3.attn_hyper_connection" or "hyper_connection_mixer".
  void gr_read(const std::string& prefix, float* x_out) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.hc_norm.weight", prefix.c_str());
    // hc_norm: BF16 10240 — needs fp32; we did NOT upload per-layer norms!
    // per-layer hc_norm weights are uploaded lazily on first use (see ensure_hcnorm)
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_R, hw, d_Rhat, 2560, g_cfg.norm_eps, 4);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_Rhat, d_t320);
    k_silu_scale<<<(320 + 255) / 256, 256, 0, g_str>>>(d_t320, 0.25f, 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_t320, d_G);
    k_sigmoid_inplace<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_G, 10240);
    k_gr_combine<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_G, d_Rhat, x_out, g_cfg.d,
                                              g_cfg.branches);
  }

  void gr_write(const std::string& prefix, const float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_Rhat, d_w4);
    k_gr_write<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_w4, d_R, y, g_cfg.d, g_cfg.branches);
  }

  // lazy cache of per-layer hc_norm + q/k_norm fp32 uploads
  std::unordered_map<std::string, float*> hcnorm_cache_;
  float* ensure_hcnorm(const std::string& prefix) {
    auto it = hcnorm_cache_.find(prefix);
    if (it != hcnorm_cache_.end()) return it->second;
    char buf[128];
    snprintf(buf, sizeof buf, "%s.hc_norm.weight", prefix.c_str());
    float* p;
    CK(hipMalloc(&p, 10240 * 4));
    std::vector<float> tmp(10240);
    ckpt.dequant(ckpt.at(buf), tmp.data());
    CK(hipMemcpy(p, tmp.data(), 10240 * 4, hipMemcpyHostToDevice));
    hcnorm_cache_[prefix] = p;
    return p;
  }
  float* ensure_norm256(const std::string& name) {
    auto it = hcnorm_cache_.find(name);
    if (it != hcnorm_cache_.end()) return it->second;
    float* p;
    CK(hipMalloc(&p, 256 * 4));
    std::vector<float> tmp(256);
    ckpt.dequant(ckpt.at(name), tmp.data());
    CK(hipMemcpy(p, tmp.data(), 256 * 4, hipMemcpyHostToDevice));
    hcnorm_cache_[name] = p;
    return p;
  }

  // ---- M-RoPE (vision) ----
  // set_mrope expands the request's image grid_thw triples into per-token
  // 3-row positions, replicating reference get_rope_index (tools/mrope_ref.py):
  // text runs take arange+current_pos on all 3 rows; an image run consumes
  // one grid (t, h, w) -> t*(h/2)*(w/2) placeholder tokens on a 3D grid
  // (t outer, h mid, w inner; each row offset by the run's start), and
  // current_pos advances only max(h, w)//2. rope_delta = max(pos)+1-N is the
  // decode offset (reference modeling_qwen4_exp.py:2121,2237-2241).
  bool set_mrope(const std::vector<int>& ids,
                 const std::vector<std::array<int, 3>>& grids) {
    clear_mrope();
    const int IMG = 248056, VID = 248057, MERGE = 2;
    const int N = (int)ids.size();
    mrope_pos_.assign(N, {0, 0, 0});
    auto type = [&](int id) { return id == IMG ? 1 : id == VID ? 2 : 0; };
    size_t gi = 0;
    int cur = 0;
    long long pmax = -1;
    for (int i = 0; i < N;) {
      int tp = type(ids[i]), j = i + 1;
      while (j < N && type(ids[j]) == tp) j++;
      if (tp == 0) {
        for (int k = i; k < j; k++) {
          int p = cur + (k - i);
          mrope_pos_[k] = {p, p, p};
        }
        pmax = std::max<long long>(pmax, cur + (j - i) - 1);
        cur += j - i;
      } else {
        if (gi >= grids.size()) return false;
        auto g = grids[gi++];
        if (g[0] < 1 || g[1] <= 0 || g[2] <= 0 || g[1] % MERGE || g[2] % MERGE)
          return false;
        int t = g[0], h = g[1] / MERGE, w = g[2] / MERGE;
        if ((long long)t * h * w != j - i) return false;
        int k = i;
        for (int tt = 0; tt < t; tt++)
          for (int hh = 0; hh < h; hh++)
            for (int ww = 0; ww < w; ww++, k++)
              mrope_pos_[k] = {cur + tt, cur + hh, cur + ww};
        pmax = std::max<long long>(pmax, cur + std::max({t - 1, h - 1, w - 1}));
        cur += std::max(g[1], g[2]) / MERGE;
      }
      i = j;
    }
    if (gi != grids.size()) return false;  // unconsumed grids
    rope_delta_ = (int)(pmax + 1 - N);
    mrope_active_ = true;
    mrope_grids_ = grids;  // kvsnap meta: pixel-agnostic placeholders need this
    mrope_gen_++;
    CK(hipMemcpy(d_rdelta, &rope_delta_, 4, hipMemcpyHostToDevice));
    return true;
  }

  void clear_mrope() {
    img_inj.clear();  // vision per-request state dies with the mrope request
    img_emb_rows = 0;
    img_hash_ = 0;
    mrope_grids_.clear();
    if (!mrope_active_ && rope_delta_ == 0) return;
    mrope_active_ = false;
    rope_delta_ = 0;
    mrope_pos_.clear();
    mrope_gen_++;
    CK(hipMemcpy(d_rdelta, &rope_delta_, 4, hipMemcpyHostToDevice));
  }

  // ---- Vision tower wiring (stage 4) ----
  void load_vision(const char* path) {
    vt.lth = lth;
    vt.rbh = rbh;
    vt.d_ltws = d_ltws;
    vt.lt_ws = lt_ws_bytes;
    vt.load(path);
    vision_on = true;
  }

  // Run the ViT over each image's patches and build the injection map.
  // grids/ids were already validated by set_mrope (same run-walking logic
  // here); images are t==1 only at this stage. Returns false on any
  // patches/grid mismatch (stderr carries the position).
  bool vision_embeds(const std::vector<int>& ids,
                     const std::vector<std::array<int, 3>>& grids,
                     const std::vector<std::vector<float>>& patches,
                     int cached = 0) {
    img_inj.clear();
    img_emb_rows = 0;
    const int IMG = 248056, VID = 248057, MERGE = 2;
    const int N = (int)ids.size();
    auto type = [&](int id) { return id == IMG ? 1 : id == VID ? 2 : 0; };
    size_t gi = 0, emb_off = 0;
    for (int i = 0; i < N;) {
      int tp = type(ids[i]), j = i + 1;
      while (j < N && type(ids[j]) == tp) j++;
      if (tp == 0) {
        i = j;
        continue;
      }
      if (tp != 1 || gi >= grids.size() || gi >= patches.size()) {
        fprintf(stderr, "vision: unsupported run at prompt pos %d\n", i);
        return false;
      }
      auto g = grids[gi];
      const std::vector<float>& px = patches[gi];
      gi++;
      int gh = g[1], gw = g[2];
      if (g[0] != 1 || gh <= 0 || gw <= 0 || gh % MERGE || gw % MERGE) {
        fprintf(stderr, "vision: bad grid t=%d h=%d w=%d at prompt pos %d\n",
                g[0], g[1], g[2], i);
        return false;
      }
      int P = gh * gw, M = P / 4;
      if ((int)px.size() != P * 1536 || j - i != M) {
        fprintf(stderr,
                "vision: patches/grid mismatch at prompt pos %d "
                "(patches=%zu floats, grid P=%d, run=%d tokens)\n",
                i, px.size(), P, j - i);
        return false;
      }
      if (j <= cached) {
        // Fully inside the reused live prefix: the KV already carries this
        // image's injected embeds. Shape validation above still applies;
        // skip the forward (and the injection rows, which would all fall
        // below the chunk base and be filtered out anyway).
        fprintf(stderr, "vision: image %zu grid=(%d,%d,%d) cached, forward "
                "skipped\n", gi, g[0], gh, gw);
        i = j;
        continue;
      }
      if (emb_off + (size_t)M > img_emb_cap) {
        size_t ncap = std::max<size_t>(emb_off + M, 4096);
        if (d_img_emb) (void)hipFree(d_img_emb);
        CK(hipMalloc(&d_img_emb, ncap * 2560 * 2));
        img_emb_cap = ncap;
      }
      auto t0 = std::chrono::steady_clock::now();
      // Debug hook (GDEC_VISION_EMBEDS_NPY=<f4 (M,2560) npy>): inject
      // precomputed embeds (host bf16 RNE) instead of running the ViT.
      // Isolates LLM/injection behavior from ViT numerics in A/B probes.
      const char* dbg_emb = getenv("GDEC_VISION_EMBEDS_NPY");
      if (dbg_emb) {
        VitNpy e = vit_npy_read(dbg_emb);
        if (e.shape.size() != 2 || e.shape[0] != (uint64_t)M ||
            e.shape[1] != 2560) {
          fprintf(stderr, "vision: embeds npy shape mismatch (want (%d,2560))\n", M);
          return false;
        }
        const float* ef = (const float*)e.data.data();
        std::vector<uint16_t> bf((size_t)M * 2560);
        for (size_t k = 0; k < bf.size(); k++) {
          uint32_t b;
          memcpy(&b, ef + k, 4);
          b += 0x7FFFu + ((b >> 16) & 1u);
          bf[k] = (uint16_t)(b >> 16);
        }
        CK(hipMemcpy(d_img_emb + emb_off * 2560, bf.data(),
                     (size_t)M * 2560 * 2, hipMemcpyHostToDevice));
      } else {
      int Mr = vt.forward(px.data(), P, gh, gw);
      if (Mr != M) return false;
      // forward only enqueues; the stitch copy must observe the finished
      // embeds (and the log line should report real forward time).
      CK(hipStreamSynchronize(g_str));
      CK(hipMemcpy(d_img_emb + emb_off * 2560, vt.d_emb_bf16,
                   (size_t)M * 2560 * 2, hipMemcpyDeviceToDevice));
      }
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
      fprintf(stderr, "vision: image %zu grid=(%d,%d,%d) P=%d M=%d "
              "forward %.0f ms\n", gi, g[0], gh, gw, P, M, ms);
      for (int k = 0; k < M; k++)
        img_inj.push_back(make_int2((int)emb_off + k, i + k));
      emb_off += M;
      i = j;
    }
    if (gi != grids.size() || gi != patches.size()) {
      fprintf(stderr, "vision: %zu grids/patches for unconsumed runs\n",
              grids.size() - gi);
      return false;
    }
    img_emb_rows = emb_off;
    img_hash_ = fnv1a_patches(patches);
    return true;
  }


  // ---- M-RoPE debug dump (GDEC_MROPE_DUMP=<file>; syncs the stream) ----
  // Format: "META rope_delta=<d>", then per table fill one line per row:
  //   ROW <base> <row> <pT> <pH> <pW> <cos0> <sin0> ... <cos31> <sin31>
  // and per decode step:
  //   DEC <tag> <rpos> <rdelta> <cos0> <sin0> ... <cos31> <sin31>
  // cos/sin as C %a hexfloats (exact binary value of the fp32 table entry).
  FILE* mrope_dump() {
    if (mrope_dump_ || !getenv("GDEC_MROPE_DUMP")) return mrope_dump_;
    mrope_dump_ = fopen(getenv("GDEC_MROPE_DUMP"), "w");
    if (mrope_dump_) fprintf(mrope_dump_, "# gdec-mrope-dump v1\n");
    return mrope_dump_;
  }

  void dump_rope_cs(int base, int P) {
    FILE* f = mrope_dump();
    if (!f) return;
    if (!mrope_dump_meta_) {
      fprintf(f, "META rope_delta=%d\n", rope_delta_);
      mrope_dump_meta_ = true;
    }
    std::vector<float2> h((size_t)P * 32);
    CK(hipMemcpy(h.data(), d_ropecs, h.size() * sizeof(float2),
                 hipMemcpyDeviceToHost));
    for (int r = 0; r < P; r++) {
      int pT = base + r, pH = base + r, pW = base + r;
      if (mrope_active_ && base + r < (int)mrope_pos_.size()) {
        pT = mrope_pos_[base + r][0];
        pH = mrope_pos_[base + r][1];
        pW = mrope_pos_[base + r][2];
      } else if (mrope_active_) {
        pT = pH = pW = base + r + rope_delta_;
      }
      fprintf(f, "ROW %d %d %d %d %d", base, r, pT, pH, pW);
      for (int i = 0; i < 32; i++)
        fprintf(f, " %a %a", (double)h[(size_t)r * 32 + i].x,
                (double)h[(size_t)r * 32 + i].y);
      fprintf(f, "\n");
    }
    fflush(f);
  }

  void dump_decode_step(int tag) {
    FILE* f = mrope_dump();
    if (!f || !d_cs1) return;
    if (!mrope_dump_meta_) {
      fprintf(f, "META rope_delta=%d\n", rope_delta_);
      mrope_dump_meta_ = true;
    }
    k_rope_dump<<<1, 32, 0, g_str>>>(d_cs1, d_rpos1, d_pos, d_rdelta,
                                     g_cfg.rope_theta);
    float2 h[32];
    int rpos = -1;
    CK(hipMemcpy(h, d_cs1, sizeof h, hipMemcpyDeviceToHost));
    CK(hipMemcpy(&rpos, d_rpos1, 4, hipMemcpyDeviceToHost));
    fprintf(f, "DEC %d %d %d", tag, rpos, rope_delta_);
    for (int i = 0; i < 32; i++)
      fprintf(f, " %a %a", (double)h[i].x, (double)h[i].y);
    fprintf(f, "\n");
    fflush(f);
  }

  // Refill the rope cos/sin table only when the chunk's (base, P) changes:
  // one compute per chunk, shared by all 12 QSA layers (stream-ordered).
  // M-RoPE: host fill with the request's per-token 3-row positions,
  // slot-interleaved; same fp64 expression as k_rope_cs (pow + libm cos/sin,
  // cast to float). Text requests keep the device-kernel fill untouched.
  void ensure_rope_cs(int base, int P) {
    if (base == ropecs_base_ && P == ropecs_P_ && mrope_gen_ == ropecs_gen_)
      return;
    if (mrope_active_) {
      double tab[32];
      for (int i = 0; i < 32; i++)
        tab[i] = pow(g_cfg.rope_theta, -2.0 * i / 64.0);
      h_ropecs_.resize((size_t)P * 32);
      const int mropeN = (int)mrope_pos_.size();
      for (int r = 0; r < P; r++)
        for (int i = 0; i < 32; i++) {
          // Spec verify runs with base=q+1 >= N: mrope_pos_ only holds the
          // prompt's N rows. Verified positions keep advancing from the last
          // mrope row by +1 (same linear sequence mrope_step_ produces), so
          // fall back to rope_delta_ + absolute position out of range.
          int p = (base + r < mropeN)
                      ? mrope_pos_[base + r][mrope_slot_row(i)]
                      : (base + r + rope_delta_);
          double ang = (double)p * tab[i];
          h_ropecs_[(size_t)r * 32 + i] =
              make_float2((float)cos(ang), (float)sin(ang));
        }
      CK(hipMemcpy(d_ropecs, h_ropecs_.data(),
                   (size_t)P * 32 * sizeof(float2), hipMemcpyHostToDevice));
    } else {
      k_rope_cs<<<(P * 32 + 255) / 256, 256, 0, g_str>>>(d_ropecs, base, P);
    }
    ropecs_base_ = base;
    ropecs_P_ = P;
    ropecs_gen_ = mrope_gen_;
    dump_rope_cs(base, P);
  }

  void gdn(int l, int gi, const float* x, float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_qkv.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_qkv);
    // conv + silu (state per layer)
    float* convst = d_convst + (size_t)gi * 3 * 10240;
    const float* cw = d_convw + (size_t)gi * 10240 * 4;
    k_gdn_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_qkv, cw, convst, d_qkv, 10240);
    // note: in-place (cur==out) is safe: read cur[ch] before write out[ch]
    // split: q [0,2048) k [2048,4096) v [4096,10240)
    float* qp = d_qkv;
    float* kp = d_qkv + 2048;
    float* vp = d_qkv + 4096;
    k_l2norm_qk<<<16, 128, 0, g_str>>>(qp, kp, 128, 1.f / sqrtf(128.f));
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_a.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_g48 /*tmp*/);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_b.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_beta48 /*tmp: raw b*/);
    k_gdn_gates<<<1, 64, 0, g_str>>>(d_g48, d_beta48, d_Alog + (size_t)gi * 48,
                           d_dtbias + (size_t)gi * 48, d_g48, d_beta48, 48, 48);
    k_gdn_step<<<48, 512, 0, g_str>>>(qp, kp, vp, d_g48, d_beta48,
                            d_S + (size_t)gi * 48 * 128 * 128, vp /*out in place*/,
                            128, 128, 48, 16);
    // out written over vp is safe: k_gdn_step reads vh before writing out
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_z);
    k_gdn_gatednorm<<<dim3(48, 1), 128, 0, g_str>>>(vp, d_z, d_gdnnorm + (size_t)gi * 128,
                                                    vp, 128, g_cfg.norm_eps, 10240,
                                                    6144);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
    gemv(wview(ckpt.at(buf)), vp, y);
  }

  // One decode step of QSA attention over the KV cache: flash-decoding split
  // over QSA_DEC_NSPLIT position slices plus a merge, for every token. Dense
  // tokens (< 2051) take the same path (ntok = token+1, identity source map);
  // the old 3-pass k_qsa_step dense kernel cost ~0.7ms->5.4ms/layer growing
  // linearly with pos (rocprofv3: 55.8% of all decode kernel time), while the
  // flash split stays flat. GDEC_QSA_DENSE keeps k_qsa_step as a debug fallback.
  template <typename KVT>
  void qsa_attn(const KVT* kc, const KVT* vc) {
    if (qsa_dense) {
      k_qsa_step<KVT><<<24, 256, 0, g_str>>>(d_qs, kc, vc, d_attn, d_pos, 256, 2, 24);
      return;
    }
    k_qsa_flash<true, false, KVT, true><<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
        d_qs, kc, vc, d_attn, 1, 256, 2, 24, d_iselected, 0, d_pos, d_pacc, d_pml);
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, d_attn, 12,
                                                       256, d_pos);
  }

  void qsa(int l, int qi, const float* x, float* y) {
    char buf[128];
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      float* keys = d_ik + (size_t)qi * index_blocks * 128;
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256, d_iq,
                                              1, g_cfg.norm_eps, g_cfg.rope_theta, d_pos,
                                              0, nullptr, d_rdelta);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256 + 128,
                                           d_iraw + qi * 512, keys, d_pos,
                                           g_cfg.norm_eps, g_cfg.rope_theta,
                                           d_rdelta);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(d_iq, keys, d_iscores,
                                                               index_blocks, d_pos);
      index_select(d_iscores, index_blocks, d_iselected, 0, 1, d_pos);
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_qg);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_kb);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.v_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_vb);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_norm.weight", l);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_norm.weight", l);
    float* knw = ensure_norm256(buf);
    // per-head: q[h] = zcnorm(qg[h*512 .. +256]); gate[h] = qg[h*512+256 ..]
    k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(d_qg, qnw, d_qs, d_gs, 256,
                                                 g_cfg.norm_eps);
    k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(d_kb, knw, d_kb, 256, g_cfg.norm_eps, 1);
    k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(d_qs, 24, 256, 64, d_pos,
                                                     g_cfg.rope_theta, d_rdelta);
    k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(d_kb, 2, 256, 64, d_pos,
                                                    g_cfg.rope_theta, d_rdelta);
    // append to cache at *d_pos
    if (qsa_kv_fp32) {
      float* kc = d_kc + (size_t)qi * maxctx * 512;
      float* vc = d_vc + (size_t)qi * maxctx * 512;
      k_store_kv<<<2, 256, 0, g_str>>>(d_kb, d_vb, kc, vc, d_pos);
      qsa_attn(kc, vc);
    } else {
      uint16_t* kc = d_kcb + (size_t)qi * maxctx * 512;
      uint16_t* vc = d_vcb + (size_t)qi * maxctx * 512;
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(d_kb, d_vb, kc, vc, d_pos);
      qsa_attn(kc, vc);
    }
    k_sigmoid_gate<<<(24 * 256 + 255) / 256, 256, 0, g_str>>>(d_attn, d_gs, d_attn, 24 * 256);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.o_proj.weight", l);
    gemv(wview(ckpt.at(buf)), d_attn, y);
  }


  // ---- 专家权重直读(零拷贝,gfx1151 统一内存;唯一形态,全量 arena 已移除)
  // load_arena 已把 .mlp.experts. 张量登记为 host mmap 指针;kernel 直读
  // 页缓存实测 224 GB/s(tools/tgv2_probe,与 GTT 同一片 DRAM),gather
  // kernel 按专家 id 索引全量张量,读到的字节与 arena 逐位一致 → 数值
  // 不变、kernel 零改动。decode 无 D2H routing 同步、无 H2D staging,
  // 指针全程静态,图捕获照常。
  // (2026-09-12 staging 方案曾记"裸指针 Memory Fault",与今日实测矛盾;
  //  当时 fault 应另有原因,本路径以 oracle 验证为准。)
  void moe(int l, const float* x, float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.mlp.gate.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_router);
    int k = g_cfg.topk;
    k_router_topk<<<1, 512, 0, g_str>>>(d_router, d_topids, d_topw, g_cfg.experts, k);
    if (dbg) {
      int hi[16];
      float hw[16];
      CK(hipMemcpy(hi, d_topids, k * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(hw, d_topw, k * 4, hipMemcpyDeviceToHost));
      fprintf(stderr, "gpu L%02d moe topk:", l);
      for (int i = 0; i < k; i++) fprintf(stderr, " %d:%.5f", hi[i], (double)hw[i]);
      fprintf(stderr, "\n");
    }

    CK(hipMemsetAsync(d_moeacc, 0, g_cfg.d * 4, g_str));
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.gate_up_proj.weight", l);
    WView gu = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.down_proj.weight", l);
    WView dn = wview(ckpt.at(buf));
    const int* topids = d_topids;  // 直读:ids 即专家号,免 D2H 同步
    // fused experts: one grouped launch each for gate_up / silu / down+accum
    {
      // v3 kernel: 2 rows per warp -> k * moe_mid warp-pairs
      uint64_t pairs = (uint64_t)k * g_cfg.moe_mid;
      k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data, x,
          d_guv, topids, (uint64_t)2 * g_cfg.moe_mid, gu.cols, gu.scale_stride, k,
          0 /*x_stride*/, 16 /*id_stride*/, k);
    }
    k_silu_mul_g<<<(k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(d_guv, d_hid, g_cfg.moe_mid, k);
    {
      // v3 kernel: 2 rows per warp -> k * (d/2) warp-pairs
      uint64_t pairs = (uint64_t)k * ((g_cfg.d + 1) / 2);
      k_q4cp_gemv_gd<<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data, d_hid,
          d_moeacc, topids, d_topw, (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
          (uint64_t)g_cfg.moe_mid, k, 16 /*id_stride*/, 0 /*acc_stride*/, k);
    }
    // shared expert
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.gate_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_guv);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.up_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_guv + 640);
    k_silu_mul<<<(640 + 255) / 256, 256, 0, g_str>>>(d_guv, d_guv + 640, d_hid, 640);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.down_proj.weight", l);
    gemv(wview(ckpt.at(buf)), d_hid, d_ey);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert_gate.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_sg);
    k_axpy_sg<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_moeacc, d_ey, d_sg, g_cfg.d,
                                                     g_cfg.d);
    CK(hipMemcpyAsync(y, d_moeacc, g_cfg.d * 4, hipMemcpyDeviceToDevice, g_str));
  }

  // PLE host phase: hash + read 16 rows into pinned staging (disk latency — overlapped).
  float* ple_e = nullptr;  // pinned
  // reads the 16 PLE rows for chist position t into dst (pure read of chist;
  // safe to call concurrently for distinct t/dst).
  void ple_rows(int t, float* dst) {
    auto cat = [&](int back) -> int64_t {
      int j = t - back;
      return j < 0 ? pad_id : chist[j];
    };
    const uint8_t* addrs[16];
    // pass 1: compute rows, kick async readahead on all 16 pages (concurrent I/O)
    for (int h = 0; h < 16; h++) {
      int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
      if (h >= 8) mix ^= cat(2) * ple_mult[2];
      uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
      const uint8_t* p = ple_table->data + row * 160;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      uint8_t stg[16 * 160];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, stg, (uint64_t)(uintptr_t)ple_table->data, 16);
      for (int i = 0; i < 16 * 160; i++)
        dst[i] = hgn::fp8e4m3_to_f32(stg[i]) * ple_tscale;
      return;
    }
#endif
    // pass 2: read (pages already in flight)
    for (int h = 0; h < 16; h++) {
      const uint8_t* p = addrs[h];
      for (int i = 0; i < 160; i++)
        dst[h * 160 + i] = hgn::fp8e4m3_to_f32(p[i]) * ple_tscale;
    }
  }
  // raw fp8 variant: staging keeps the 160 code bytes per head; dequant+scale
  // happens on GPU (k_ple_fp8_dequant). ~4x less host work per token.
  void ple_rows_fp8(int t, uint8_t* dst) {
    auto cat = [&](int back) -> int64_t {
      int j = t - back;
      return j < 0 ? pad_id : chist[j];
    };
    const uint8_t* addrs[16];
    for (int h = 0; h < 16; h++) {
      int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
      if (h >= 8) mix ^= cat(2) * ple_mult[2];
      uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
      const uint8_t* p = ple_table->data + row * 160;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, dst, (uint64_t)(uintptr_t)ple_table->data, 16);
      return;
    }
#endif
    for (int h = 0; h < 16; h++) memcpy(dst + h * 160, addrs[h], 160);
  }
  void ple_host(int token, float* dst) {
    chist.push_back(token);
    ple_rows((int)chist.size() - 1, dst);
  }

  // PLE device phase: upload + gate/conv kernels (must be enqueued after L0).
  void ple_gpu() {
    CK(hipMemcpyAsync(d_e, ple_e, 2560 * 4, hipMemcpyHostToDevice, g_str));
    gemv(wview(ckpt.at("layers.1.ple.key_proj.weight")), d_e, d_keys);
    gemv(wview(ckpt.at("layers.1.ple.value_proj.weight")), d_e, d_v);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_keys, d_ple_nk, d_kn, 2560,
                                                g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_R, d_ple_nk + 10240, d_qn, 2560,
                                                g_cfg.norm_eps, 4);
    k_ple_gate<<<4, 1024, 0, g_str>>>(d_kn, d_qn, d_v, d_U, 2560, 4);
    // ring[ringpos] = norm_conv(U)
    k_rmsnorm_zc_ring<<<4, 1024, 0, g_str>>>(d_U, d_ple_nk + 20480, d_ring, d_ringpos,
                                             2560, g_cfg.norm_eps);
    k_ple_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_ring, d_ringpos, d_ple_convw,
                                                       d_convout, 10240);
    k_add2<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_R, d_U, d_convout, 10240);
    // device ringpos advanced by k_step_incr (graph) or host upload (direct)
  }

  // one token; returns true + writes argmax id when want_token
  bool dbg = getenv("GDEC_DEBUG") != nullptr;
  bool perf = getenv("GDEC_PERF") != nullptr;
  bool qsa_loop = getenv("GDEC_QSA_LOOP") != nullptr;  // prefill QSA fallback
  bool qsa_global_v = getenv("GDEC_QSA_GLOBAL_V") != nullptr;
  bool moe_naive = getenv("GDEC_MOE_NAIVE") != nullptr;  // prefill MoE fallback
  // Small batches: per-pair naive gemv reads ~the same expert bytes as the
  // packed path (few shared experts) but at ~2x its effective bandwidth
  // (measured: P=8 chunk 0.19s -> 0.14s). 0 disables the auto switch.
  int moe_naive_max = [] {
    const char* e = getenv("GDEC_MOE_NAIVE_MAX");
    return e ? atoi(e) : 16;
  }();
  // Default small-P down path; GDEC_MOE_DOWN_ATOMIC restores the old
  // per-expert atomic accumulation for numerical/performance comparisons.
  bool moe_down_topk =
      moe_deterministic && getenv("GDEC_MOE_DOWN_ATOMIC") == nullptr;
  bool moe_ordered = getenv("GDEC_MOE_ORDERED") != nullptr;  // diagnostic: expert-order scatter
  bool moe_host_route = getenv("GDEC_MOE_HOST_ROUTE") != nullptr;
  bool moe_untiled = getenv("GDEC_MOE_UNTILED") != nullptr;
  bool moe_fp32_io = getenv("GDEC_MOE_FP32_IO") != nullptr;
  bool moe_up_legacy = getenv("GDEC_MOE_UP_LEGACY") != nullptr;
  bool moe_up_no_table = getenv("GDEC_MOE_UP_NO_TABLE") != nullptr;
  bool moe_down_no_table = getenv("GDEC_MOE_DOWN_NO_TABLE") != nullptr;
  bool moe_reduce_old = getenv("GDEC_MOE_REDUCE_OLD") != nullptr;
  // bf16 pair rows from the deterministic down kernel; the router weight is
  // applied in fp32 by k_moe_reduce_pw_bf16 instead of pre-multiplied in fp32.
  bool moe_pairs_bf16 = getenv("GDEC_MOE_PAIRS_BF16") != nullptr;
  bool moe_lt = getenv("GDEC_MOE_LT") != nullptr;  // prefill MoE: dequant + per-expert hipBLASLt
  // Validated PP dataflow changes. Unset each switch to restore its baseline.
  // Keep the P<=8 direct GEMV path's input precision and scratch lifetime.
  bool pp_moe_out = getenv("GDEC_PP_MOE_OUT") != nullptr;
  bool moe_lt_bf16 = getenv("GDEC_MOE_LT_BF16") != nullptr;  // bf16 GEMM outputs (LT path)
  bool moe_lt_cmp = getenv("GDEC_MOE_LT_CMP") != nullptr;  // debug: in-run fused reference diff
  bool moe_lt_ovl = getenv("GDEC_MOE_LT_OVL") != nullptr;  // overlap next group's dequant with current GEMMs (LT path)
  // LT loses below ~4K tokens (512 tiny per-expert launches + the d_eoff sync
  // cost more than the GEMM wins; measured 2026-09-10: +0.7s @2K, wash @8K).
  int moe_lt_min = getenv("GDEC_MOE_LT_MIN") ? atoi(getenv("GDEC_MOE_LT_MIN")) : 4096;
  bool gdn_loop = getenv("GDEC_GDN_LOOP") != nullptr;  // prefill GDN fallback
  bool gdn_nosplit = getenv("GDEC_GDN_NOSPLIT") != nullptr;  // single-kernel GDN fallback
  bool gdn_nostrip = getenv("GDEC_GDN_NOSTRIP") != nullptr;  // old inter (no strip)
  bool gdn_stream = getenv("GDEC_GDN_STREAM") != nullptr;  // four-chunk workspace window
  int gdn_window_chunks = [] {
    const char* e = getenv("GDEC_GDN_WINDOW_CHUNKS");
    return e ? std::max(1, atoi(e)) : 4;
  }();
  bool gdn_wave = getenv("GDEC_GDN_WAVE") != nullptr;  // column-local triangular solve
  bool gr_scat4 = getenv("GDEC_GR_SCAT4") != nullptr;  // one-block-per-token GR scatter+norm
  bool gdn_ut5 = getenv("GDEC_GDN_UT5") != nullptr;  // blocked triangular inverse
  bool ple_loop = getenv("GDEC_PLE_LOOP") != nullptr;  // prefill PLE fallback
  bool prof = getenv("GDEC_PROF") != nullptr;  // host-side phase timing
  double p_topk_wait = 0, p_bucket = 0, p_gemm = 0, p_pleh = 0,
         p_ple_wait = 0;
  // perf accumulators (ms): embed, ple, attn, moe, head
  double t_acc[5] = {0, 0, 0, 0, 0};
  double t_enq = 0, t_sync = 0;
  int t_n = 0;
  std::chrono::steady_clock::time_point tick() {
    if (!perf) return std::chrono::steady_clock::time_point{};
    CK(hipDeviceSynchronize());
    return std::chrono::steady_clock::now();
  }
  void tock(int slot, std::chrono::steady_clock::time_point t0) {
    if (!perf) return;
    CK(hipDeviceSynchronize());
    t_acc[slot] +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
  }
  void h_rms(const char* tag, int l, const char* sub, const float* p, int n) {
    if (!dbg) return;
    std::vector<float> h(n);
    CK(hipMemcpy(h.data(), (void*)p, n * 4, hipMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)h[i] * h[i];
    fprintf(stderr, "%s L%02d %s rms=%.4f [0..3]=%.5f %.5f %.5f %.5f\n", tag, l, sub,
            sqrt(s / n), h[0], h[1], h[2], h[3]);
  }
  // enqueue helpers — shared by direct execution and graph capture
  void enq_embed() {
    WView et = wview(ckpt.at("embed_tokens.weight"));
    const uint8_t* codes = et.data + 64;
    const uint8_t* scales = et.data + 64 + et.rows * et.cols / 2;
    k_q4cp_row<<<1, 128, 0, g_str>>>(codes, scales, (const float*)et.data, d_token,
                                     et.cols, et.scale_stride, d_emb);
    k_repeat4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_emb, d_R, g_cfg.d);
    h_rms("gpu", -1, "emb", d_emb, g_cfg.d);
  }
  void enq_layer(int l) {
    char prefix[128];
    snprintf(prefix, sizeof prefix, "layers.%d.attn_hyper_connection", l);
    auto t2 = tick();
    gr_read(prefix, d_x);
    if (is_qsa(l))
      qsa(l, l / 4, d_x, d_y);
    else
      gdn(l, l - (l + 1) / 4, d_x, d_y);
    h_rms("gpu", l, "attn-x", d_x, 2560);
    h_rms("gpu", l, "attn-y", d_y, 2560);
    gr_write(prefix, d_y);
    tock(2, t2);
    snprintf(prefix, sizeof prefix, "layers.%d.mlp_hyper_connection", l);
    auto t3 = tick();
    gr_read(prefix, d_x);
    moe(l, d_x, d_y);
    h_rms("gpu", l, "mlp-x", d_x, 2560);
    h_rms("gpu", l, "mlp-y", d_y, 2560);
    gr_write(prefix, d_y);
    tock(3, t3);
  }
  void enq_final() {
    gr_read("hyper_connection_mixer", d_x /*h*/);
    gemv(wview(ckpt.at("lm_head.weight")), d_x, d_logits);
    k_argmax<<<1, 1024, 0, g_str>>>(d_logits, g_cfg.vocab, d_argmax);
  }

  // =================== prefill batch (Phase 3a) ==============================
  // Batched GEMM: Y[P,N] = X[P,K] . W[N,K]^T. W dequanted to bf16 workspace,
  // X converted to bf16; rocBLAS col-major mapping: C_cm[N,P] with
  // A = W (transa=T, lda=K), B = X (transb=N, ldb=K), C = Y (ldc=N).
  void gemm(const WView& w, const float* x, float* y, int P, uint64_t xstride = 0,
            const uint16_t* xbf = nullptr) {
    auto pt0 = std::chrono::steady_clock::now();
    uint64_t N = w.rows, K = w.cols;
    if (!xstride) xstride = K;
    // Small P: direct multi-row q4 gemv reads the weight once (~0.56x bytes
    // of the dequant->bf16->LT path's traffic) and skips 3 launches/gemm.
    // xbf inputs are bf16 with row stride K (the LT B layout).
    bool mr_done =
        xbf ? gemv_multi_q4cp(w, xbf, y, P, K, true)
            : gemv_multi_q4cp(w, x, y, P, xstride);
    if (mr_done) {
      if (prof)
        p_gemm += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pt0)
                      .count();
      return;
    }
    static const bool gemm_dbg = getenv("GDEC_GEMM_DBG") != nullptr;
    if (gemm_dbg && P <= 8)
      fprintf(stderr, "gemm-LT: dtype=%d N=%llu K=%llu P=%d xbf=%d xs=%llu\n",
              w.dtype, (unsigned long long)N, (unsigned long long)K, P,
              (int)(xbf != nullptr), (unsigned long long)xstride);
    const uint16_t* wp = nullptr;
    if (w.dtype == 0) {
      wp = (const uint16_t*)w.data;
    } else if (w.dtype == 5) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 32;
      k_dequant_q4cp_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          w.data + 64, w.data + 64 + N * K / 2, (const float*)w.data, d_wbf16, N, K,
          w.scale_stride);
      wp = d_wbf16;
    } else if (w.dtype == 7) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 64;
      k_dequant_q8g64_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          w.data, d_wbf16, N, K);
      wp = d_wbf16;
    } else {
      throw std::runtime_error("gemm: unsupported dtype");
    }
    uint64_t tot = (uint64_t)P * K;
    bool residual_ready =
        !xbf && prefill_fused && x == d_Rhatb && K == 10240 && xstride == K;
    bool cacheable = (x == d_xb || x == d_Rhatb);
    // Non-cacheable inputs convert into d_xbf16b, leaving the d_xb/d_Rhatb
    // cache entries (xf16_*) untouched; xbf bypasses conversion entirely.
    uint16_t* dst = cacheable ? d_xbf16 : d_xbf16b;
    const uint16_t* xp = xbf ? xbf : residual_ready ? d_Rhatbf16 : dst;
    if (!xbf && !residual_ready &&
        !(cacheable && xf16_src == x && xf16_xs == xstride && xf16_P == P &&
          xf16_K == (int)K)) {
      if (K % 4 == 0 && xstride % 4 == 0 &&
          (((uintptr_t)x | (uintptr_t)dst) & 15) == 0)
        k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
            x, dst, xstride, P, (int)K);
      else
        k_f32_to_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
            x, dst, xstride, P, (int)K);
      if (cacheable) {
        xf16_src = x;
        xf16_xs = xstride;
        xf16_P = P;
        xf16_K = (int)K;
      }
    }
    float one = 1.f, zero = 0.f;
    if (getenv("GDEC_GEMM_RB")) {  // fallback: rocBLAS default algo
      rocblas_set_stream(rbh, g_str);
      rocblas_status st = rocblas_gemm_ex(
          rbh, rocblas_operation_transpose, rocblas_operation_none, (rocblas_int)N,
          (rocblas_int)P, (rocblas_int)K, &one, wp, rocblas_datatype_bf16_r,
          (rocblas_int)K, xp, rocblas_datatype_bf16_r, (rocblas_int)K, &zero, y,
          rocblas_datatype_f32_r, (rocblas_int)N, y, rocblas_datatype_f32_r,
          (rocblas_int)N, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
      if (st != rocblas_status_success) {
        fprintf(stderr, "rocblas_gemm_ex failed: %d (N=%llu K=%llu P=%d)\n", (int)st,
                (unsigned long long)N, (unsigned long long)K, P);
        exit(1);
      }
      if (prof)
        p_gemm += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pt0)
                      .count();
      return;
    }
    // Phase 3g: self-written bf16 WMMA GEMM (GDEC_GEMM_WMMA=1). Diverts the
    // four winning projection shapes at P >= 1024 to k_gemm_wmma — d3 config
    // (512t w2x8 f4x2, gm=16) for N=2560 K=6144, d9 (256t w2x4 f4x4, gm=4)
    // for N=6144/10240/12288 K=2560; every other shape and the default
    // flag-off state fall through to hipBLASLt unchanged. Purely async. The
    // shape list guarantees N % 256 == 0 and K % 32 == 0 (the kernel's
    // host-checked requirements); P tails are handled inside the kernel.
    // GDEC_GEMM_RB above takes precedence if both are set.
    static const bool gemm_wmma = getenv("GDEC_GEMM_WMMA") != nullptr;
    if (gemm_wmma && P >= 1024) {
      const bool gw_d3 = (N == 2560 && K == 6144);
      const bool gw_d9 = (K == 2560 && (N == 6144 || N == 10240 || N == 12288));
      if (gw_d3 || gw_d9) {
        const unsigned gw_grid = (unsigned)(((P + 127) / 128) * (int)(N / 256));
        if (gw_d3)
          k_gemm_wmma<128, 256, 2, 8, 4, 2, 32, 512>
              <<<gw_grid, 512, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N, 16);
        else
          k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256>
              <<<gw_grid, 256, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N, 4);
        if (prof)
          p_gemm += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pt0)
                        .count();
        return;
      }
    }
    // hipBLASLt path: ~30 TFLOPS vs ~5.6 for the rocBLAS default algo on
    // gfx1151. The heuristic's rank order is misleading — index 4 is the fast
    // tensile kernel for every projection shape measured (5-6x over index 0),
    // so use it after a validation run; fall back to a timing scan if the
    // list is short or index 4 fails. Full descriptor set cached per (N,K,P).
    uint64_t key = (N << 40) ^ (K << 20) ^ (uint64_t)P;
    auto it = lt_algos.find(key);
    if (it == lt_algos.end()) {
      LtEntry e;
      hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
      hipblasLtMatrixLayoutCreate(&e.Al, HIP_R_16BF, K, N, K);
      hipblasLtMatrixLayoutCreate(&e.Bl, HIP_R_16BF, K, P, K);
      hipblasLtMatrixLayoutCreate(&e.Cl, HIP_R_32F, N, P, N);
      hipblasLtMatrixLayoutCreate(&e.Dl, HIP_R_32F, N, P, N);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
          sizeof(lt_ws_bytes));
      hipblasLtMatmulHeuristicResult_t hres[8];
      int nres = 0;
      hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, e.Bl, e.Cl, e.Dl, pref,
                                      8, hres, &nres);
      auto try_run = [&](int i) {
        hipblasLtMatmul(lth, e.opd, &one, wp, e.Al, xp, e.Bl, &zero, y, e.Cl, y,
                        e.Dl, &hres[i].algo, d_ltws, lt_ws_bytes, g_str);
        if (hipStreamSynchronize(g_str) != hipSuccess) {
          (void)hipGetLastError();
          return false;
        }
        return true;
      };
      int bi = -1;
      if (nres > 4 && hres[4].state == HIPBLAS_STATUS_SUCCESS &&
          hres[4].workspaceSize <= lt_ws_bytes && try_run(4))
        bi = 4;
      else {  // fallback: time all candidates
        double best = 1e30;
        for (int i = 0; i < nres; i++) {
          if (hres[i].state != HIPBLAS_STATUS_SUCCESS) continue;
          if (hres[i].workspaceSize > lt_ws_bytes) continue;
          if (!try_run(i)) continue;
          auto t0 = std::chrono::steady_clock::now();
          for (int r = 0; r < 3; r++) try_run(i);
          double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
          if (ms < best) {
            best = ms;
            bi = i;
          }
        }
      }
      if (bi < 0) {
        fprintf(stderr, "hipBLASLt: no algo for N=%llu K=%llu P=%d\n",
                (unsigned long long)N, (unsigned long long)K, P);
        exit(1);
      }
      e.algo = hres[bi].algo;
      hipblasLtMatmulPreferenceDestroy(pref);
      it = lt_algos.emplace(key, e).first;
    }
    const LtEntry& e = it->second;
    hipblasStatus_t lst =
        hipblasLtMatmul(lth, e.opd, &one, wp, e.Al, xp, e.Bl, &zero, y, e.Cl, y,
                        e.Dl, &e.algo, d_ltws, lt_ws_bytes, g_str);
    if (lst != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "hipblasLtMatmul failed: %d (N=%llu K=%llu P=%d)\n", (int)lst,
              (unsigned long long)N, (unsigned long long)K, P);
      exit(1);
    }
    if (prof)
      p_gemm += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - pt0)
                    .count();
  }

  // gemm variant for inputs that are already bf16 was used by the 3a' grouped
  // MoE path; removed with it (3d-1 fused W4 kernels replace it).

  // bf16 A (weights, [N][K] k-contiguous -> op T) x bf16 B ([P][K] k-contiguous
  // -> op N) -> C[N,P] (n-contiguous), fp32 or bf16 out. Used by GDEC_MOE_LT
  // per-expert GEMMs. Algo cached per (N,K,P,outbf16); on miss trust heuristic
  // index 4 (the fast Tensile kernel for every projection shape measured in
  // tools/moe_lt_bench.cu scan mode; per-M validation syncs would stall the
  // per-layer expert loop). Bit 62 keeps keys disjoint from dense gemm's.
  void gemm_bf16(const __hip_bfloat16* A, const __hip_bfloat16* B, void* C,
                 int N, int K, int P, bool outbf16) {
    float one = 1.f, zero = 0.f;
    uint64_t key = (1ULL << 62) ^ ((uint64_t)outbf16 << 63) ^ ((uint64_t)N << 40) ^
                   ((uint64_t)K << 20) ^ (uint64_t)P;
    auto it = lt_algos.find(key);
    hipDataType dtout = outbf16 ? HIP_R_16BF : HIP_R_32F;
    if (it == lt_algos.end()) {
      LtEntry e;
      hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
      hipblasLtMatrixLayoutCreate(&e.Al, HIP_R_16BF, K, N, K);
      hipblasLtMatrixLayoutCreate(&e.Bl, HIP_R_16BF, K, P, K);
      hipblasLtMatrixLayoutCreate(&e.Cl, dtout, N, P, N);
      hipblasLtMatrixLayoutCreate(&e.Dl, dtout, N, P, N);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
          sizeof(lt_ws_bytes));
      hipblasLtMatmulHeuristicResult_t hres[8];
      int nres = 0;
      hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, e.Bl, e.Cl, e.Dl, pref,
                                      8, hres, &nres);
      if (nres <= 0) {
        fprintf(stderr, "moe_lt: no algo for N=%d K=%d P=%d\n", N, K, P);
        exit(1);
      }
      int bi = (nres > 4 && hres[4].state == HIPBLAS_STATUS_SUCCESS &&
                hres[4].workspaceSize <= lt_ws_bytes)
                   ? 4
                   : 0;
      e.algo = hres[bi].algo;
      hipblasLtMatmulPreferenceDestroy(pref);
      it = lt_algos.emplace(key, e).first;
    }
    const LtEntry& e = it->second;
    hipblasStatus_t lst =
        hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl, &zero, C, e.Cl, C,
                        e.Dl, &e.algo, d_ltws, lt_ws_bytes, g_str);
    if (lst != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "moe_lt: hipblasLtMatmul failed: %d (N=%d K=%d P=%d)\n",
              (int)lst, N, K, P);
      exit(1);
    }
  }

  void gr_mix_b(const std::string& prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_Rhatb, d_t320b, P, 0,
         gr_bf16 ? d_Rhatbf16 : nullptr);
    k_silu_scale<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(d_t320b, 0.25f, P * 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_t320b, d_Gb, P);
    if (prefill_fused) {
      if (gr_bf16)
        k_gr_combine_b_hc_bf16<<<P, 256, 0, g_str>>>(
            d_Gb, d_Rhatbf16, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16);
      else
        k_gr_combine_b<true><<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
            d_Gb, d_Rhatb, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16);
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
    } else {
      k_sigmoid_inplace<<<(P * 10240 + 255) / 256, 256, 0, g_str>>>(d_Gb, P * 10240);
      k_gr_combine_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_Gb, d_Rhatb, d_xb, g_cfg.d, g_cfg.branches, P);
      if (xf16_src == d_xb) xf16_src = nullptr;
    }
  }

  void gr_read_b(const std::string& prefix, int P) {
    float* hw = ensure_hcnorm(prefix);
    if (gr_bf16)
      k_rmsnorm_zc_grouped_bf16<<<4 * P, 1024, 0, g_str>>>(
          (const uint16_t*)d_Rb, hw, d_Rhatbf16, 2560, g_cfg.norm_eps, 4);
    else if (prefill_fused)
      k_rmsnorm_zc_grouped<true><<<4 * P, 1024, 0, g_str>>>(
          d_Rb, hw, d_Rhatb, 2560, g_cfg.norm_eps, 4, d_Rhatbf16);
    else
      k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(
          d_Rb, hw, d_Rhatb, 2560, g_cfg.norm_eps, 4);
    if (xf16_src == d_Rhatb) xf16_src = nullptr;  // d_Rhatb overwritten
    gr_mix_b(prefix, P);
  }

  void gr_write_b(const std::string& prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_Rhatb, d_w4b, P, 0,
         gr_bf16 ? d_Rhatbf16 : nullptr);
    if (gr_bf16)
      k_gr_write_b_hc_bf16<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_w4b, (uint16_t*)d_Rb, d_yb, g_cfg.d, g_cfg.branches, P);
    else
      k_gr_write_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, g_cfg.d, g_cfg.branches, P);
  }

  void gr_write_read_b(const std::string& write_prefix,
                       const std::string& read_prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight",
             write_prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_Rhatb, d_w4b, P, 0,
         gr_bf16 ? d_Rhatbf16 : nullptr);
    float* hw = ensure_hcnorm(read_prefix);
    if (gr_bf16)
      if (gr_scat4 && g_cfg.d <= 2560)
        k_gr_scatter_norm_b_hc_bf16_f4<<<P, 256, 0, g_str>>>(
            d_w4b, (uint16_t*)d_Rb, d_yb, hw, d_Rhatbf16, g_cfg.d,
            g_cfg.branches, P, g_cfg.norm_eps);
      else
        k_gr_scatter_norm_b_hc_bf16<<<4 * P, 256, 0, g_str>>>(
            d_w4b, (uint16_t*)d_Rb, d_yb, hw, d_Rhatbf16, g_cfg.d,
            g_cfg.branches, P, g_cfg.norm_eps);
    else if (prefill_fused)
      k_gr_scatter_norm_b<true><<<4 * P, 1024, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, hw, d_Rhatb, g_cfg.d, g_cfg.branches, P,
          g_cfg.norm_eps, d_Rhatbf16);
    else
      k_gr_scatter_norm_b<<<4 * P, 1024, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, hw, d_Rhatb, g_cfg.d, g_cfg.branches, P,
          g_cfg.norm_eps);
    if (xf16_src == d_Rhatb) xf16_src = nullptr;
    gr_mix_b(read_prefix, P);
  }

  // GDN: projections batched. Chunked path: conv/l2norm/gates batched, then
  // k_gdn_chunk (chunk=64, fp32) advances S and writes all outputs at once.
  // GDEC_GDN_LOOP=1 falls back to the per-token recurrence loop.
  void gdn_b(int l, int gi, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_qkv.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_qkvb, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_a.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_g48b, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_b.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_beta48b, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_zb, P);
    float* convst = d_convst + (size_t)gi * 3 * 10240;
    const float* cw = d_convw + (size_t)gi * 10240 * 4;
    const float* Alog = d_Alog + (size_t)gi * 48;
    const float* dtb = d_dtbias + (size_t)gi * 48;
    float* S = d_S + (size_t)gi * 48 * 128 * 128;
    const float* gnw = d_gdnnorm + (size_t)gi * 128;
    float qscale = 1.f / sqrtf(128.f);
    if (!gdn_loop) {
      int64_t tot = (int64_t)P * 10240 / 4;  // 4 channels per thread (float4)
      k_gdn_conv_b<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          d_qkvb, cw, convst, d_convb, P, 10240);
      if (spec_snap) {
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        k_convst_snap<<<P, 512, 0, g_str>>>(
            d_qkvb, convst, d_csnap + (size_t)gi * 3 * 10240, P, 10240,
            (size_t)ngdn * 3 * 10240);
      }
      k_convst_update<<<(10240 + 255) / 256, 256, 0, g_str>>>(convst, d_qkvb, P,
                                                              10240);
      k_l2norm_qk_b<<<dim3(16, P), 128, 0, g_str>>>(d_convb, 128, qscale, 10240, 16);
      k_gdn_gates<<<P, 64, 0, g_str>>>(d_g48b, d_beta48b, Alog, dtb, d_g48b, d_beta48b,
                                       48, 48);
      if (spec_snap) {
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        k_gdn_snap<<<48, 512, 0, g_str>>>(
            d_convb, d_g48b, d_beta48b, S, d_Ssnap + (size_t)gi * 48 * 128 * 128,
            P, 10240, (size_t)ngdn * 48 * 128 * 128);
      }
      if (ngram_verify_active && !ngram_legacy_gdn && P <= 65) {
        k_gdn_verify<<<48, 512, 0, g_str>>>(
            d_convb, d_g48b, d_beta48b, S, d_convb, P, 10240);
      } else if (gdn_nosplit) {
        k_gdn_chunk<<<48, GDN_NT, 0, g_str>>>(d_convb, d_g48b, d_beta48b, S, d_convb,
                                              d_gdn_ws, P, 10240);
      } else {
        // Keep producer and consumer close; S carries the exact recurrence
        // across windows and the last window alone can contain a padded chunk.
        int window = gdn_stream ? 64 * gdn_window_chunks : P;
        for (int first = 0; first < P; first += window) {
          int count = std::min(window, P - first), nchunks = (count + 63) / 64;
          float* conv = d_convb + (size_t)first * 10240;
          if (gdn_wave)
            if (gdn_ut5)
              k_gdn_intra<1024, true, true><<<dim3(nchunks, 48), 1024, 0, g_str>>>(
                  conv, d_g48b + first * 48, d_beta48b + first * 48,
                  d_gdn_split_ws, count, 10240);
            else
              k_gdn_intra<1024, true><<<dim3(nchunks, 48), 1024, 0, g_str>>>(
                  conv, d_g48b + first * 48, d_beta48b + first * 48,
                  d_gdn_split_ws, count, 10240);
          else
            k_gdn_intra<GDN_NT_INTRA><<<dim3(nchunks, 48), GDN_NT_INTRA, 0, g_str>>>(
                conv, d_g48b + first * 48, d_beta48b + first * 48,
                d_gdn_split_ws, count, 10240);
          if (gdn_nostrip)
            k_gdn_inter<GDN_NT_INTER><<<48, GDN_NT_INTER, 0, g_str>>>(
                conv, S, conv, d_gdn_split_ws, count, 10240);
          else
            k_gdn_inter_strip<<<dim3(4, 48), GDN_NT_STRIP, 0, g_str>>>(
                conv, S, conv, d_gdn_split_ws, count, 10240);
        }
      }
      k_gdn_gatednorm_b<true><<<dim3(48, (P + 31) / 32), 128, 0, g_str>>>(
          d_convb + 4096, d_zb, gnw, nullptr, g_cfg.norm_eps, 10240, 6144,
          d_xbf16b, P);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
      gemm(wview(ckpt.at(buf)), d_convb + 4096, d_yb, P, 10240 /*row stride*/,
           d_xbf16b);
      return;
    }
    for (int t = 0; t < P; t++) {
      float* qkv = d_qkvb + (size_t)t * 10240;
      k_gdn_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(qkv, cw, convst, qkv, 10240);
      k_l2norm_qk<<<16, 128, 0, g_str>>>(qkv, qkv + 2048, 128, qscale);
      k_gdn_gates<<<1, 64, 0, g_str>>>(d_g48b + t * 48, d_beta48b + t * 48, Alog, dtb,
                                       d_g48b + t * 48, d_beta48b + t * 48, 48, 48);
      k_gdn_step<<<48, 512, 0, g_str>>>(qkv, qkv + 2048, qkv + 4096, d_g48b + t * 48,
                                        d_beta48b + t * 48, S, qkv + 4096, 128, 128, 48,
                                        16);
      k_gdn_gatednorm<<<dim3(48, 1), 128, 0, g_str>>>(qkv + 4096,
                                                      d_zb + (size_t)t * 6144, gnw,
                                                      qkv + 4096, 128, g_cfg.norm_eps,
                                                      10240, 6144);
    }
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_qkvb + 4096, d_yb, P, 10240 /*row stride*/);
  }

  // QSA prefill flash over the KV cache: dense prefix plus the sparse tail.
  // base = absolute position of batch row 0 (0 for a fresh prefill; > 0 when
  // continuing from live KV/state). Dense rows are the batch-local prefix,
  // sparse queries are the absolute positions >= 2051 within the batch.
  template <typename KVT>
  void qsa_flash_b(const KVT* kc, const KVT* vc, int P, int dense_P, int base = 0,
                   const uint16_t* vc_transposed = nullptr) {
    if (dense_P > 0)
      k_qsa_flash<false, false, KVT><<<dim3((dense_P + 15) / 16, 24), 128, 0, g_str>>>(
          d_qsb, kc, vc, d_attnb, dense_P, 256, 2, 24, nullptr, 0, nullptr, nullptr,
          nullptr, base);
    int first_abs = std::max(base, 2051), nsparse = base + P - first_abs;
    if (!qsa_dense && nsparse > 0) {
      const int* sel = d_iselected + (size_t)first_abs * 512;
      if (qsa_global_v)
        k_qsa_flash<true, false, KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
            d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, nullptr, nullptr,
            nullptr, base);
      else if constexpr (sizeof(KVT) == 2) {
        if (qsa_wmma6 && vc_transposed)
          k_qsa_wmma6<<<dim3(nsparse, 2), 128, 0, g_str>>>(
              d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel,
              first_abs, base);
        else if (qsa_wmma_btv && vc_transposed)
          k_qsa_wmma<true><<<dim3(nsparse, 2), 256, 0, g_str>>>(
              d_qsb, kc, vc, vc_transposed, d_attnb, P, 256, 2, 24, sel, first_abs,
              base);
        else if (qsa_wmma)
          // GDEC_QSA_WMMA=1: WMMA bf16 sparse flash (256 thr), see k_qsa_wmma.
          k_qsa_wmma<false><<<dim3(nsparse, 2), 256, 0, g_str>>>(
              d_qsb, kc, vc, nullptr, d_attnb, P, 256, 2, 24, sel, first_abs, base);
        else
          // BF16 KV mode (!qsa_kv_fp32): bf16 LDS tiles + fdot2 (-61% at 32K).
          k_qsa_flash_bf16<KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
              d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, base);
      }
      else
        k_qsa_flash<true, true, KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
            d_qsb, kc, vc, d_attnb, P, 256, 2, 24, sel, first_abs, nullptr, nullptr,
            nullptr, base);
    }
  }

  // QSA: projections batched. Flash path: qsplit/norm/rope batched, KV rows
  // bulk-copied to cache slots 0..P-1, one causal flash kernel over all
  // positions. GDEC_QSA_LOOP=1 falls back to the per-token loop (debug).
  void qsa_b(int l, int qi, int P, int base = 0) {
    char buf[128];
    if (!qsa_dense) {
      // rope cos/sin table for the indexer kernels below (idempotent; the QSA
      // prep path re-ensures the same (base, P) later and hits the cache)
      ensure_rope_tab(g_cfg.rope_theta, 64);
      ensure_rope_cs(base, P);
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      if (index_f32) {
        float one = 1, zero = 0;
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose, rocblas_operation_none,
                                    640, P, 2560, &one, d_iw + (size_t)qi * 640 * 2560,
                                    2560, d_xb, 2560, &zero, d_iproj, 640);
        if (status != rocblas_status_success) {
          fprintf(stderr, "indexer fp32 projection failed: %d\n", (int)status);
          exit(1);
        }
      } else {
        gemm(wview(ckpt.at(buf)), d_xb, d_iproj, P);
      }
      float* keys = d_ik + (size_t)qi * index_blocks * 128;
      k_index_q<<<dim3(P, 4), 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256, d_iq,
                                              P, g_cfg.norm_eps, g_cfg.rope_theta,
                                              nullptr, base, d_ropecs);
      // absolute 4-token blocks [base/4, (base+P)/4); a left-straddling block
      // (base%4 != 0) is completed with history rows from the raw ring
      int nb = (base + P) / 4, b0 = base / 4;
      if (nb > b0)
        k_index_pool<<<nb - b0, 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256 + 128,
                                                 keys, g_cfg.norm_eps,
                                                 g_cfg.rope_theta, base,
                                                 d_iraw + qi * 512, d_ropecs);
      k_index_ring<<<min(P, 4), 128, 0, g_str>>>(d_iproj, d_iraw + qi * 512, P, base);
      if (spec_snap)
        k_iproj_snap<<<P, 128, 0, g_str>>>(
            d_iproj, d_iproj_snap + (size_t)qi * 9 * 128, P);
      const float one = 1.f, zero = 0.f;
      // NOTE: nb (m/ldc) is the absolute pooled-block count, matching the
      // shape a from-scratch prefill of the same total length would use —
      // the sgemm shape stays numerically locked across request splits.
      // Chunk starts are additionally aligned to the from-scratch grid
      // (2051 + k*INDEX_BATCH): with the same (m, n, k) every full chunk
      // replays bitwise-identical scores and selections, so a continuation
      // prefill can only diverge inside the leading ragged chunk [a0, grid).
      const int a0 = std::max(2051, base);
      const int grid =
          2051 + (a0 - 2051 + INDEX_BATCH - 1) / INDEX_BATCH * INDEX_BATCH;
      for (int first = a0; first < base + P;) {
        int nxt = first < grid ? min(grid, base + P)
                               : min(first + INDEX_BATCH, base + P);
        int count = nxt - first;
        if (index_fused2 && count >= 16 && getenv("GDEC_INDEX_OLDSEL") == nullptr) {
          k_index_scores_tiled<<<dim3((nb + 31) / 32, (count + 15) / 16), 256, 0, g_str>>>(
              d_iq + (size_t)(first - base) * 512, keys, d_iscores, nb, count, first);
          if ((first + count) / 4 <= 8192)
            k_index_select_rs<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, d_iselected + (size_t)first * 512, first);
          else if (index_stream_select)
            k_index_select_stream<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, d_iselected + (size_t)first * 512, first);
          else
            k_index_select<16, true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, d_iselected + (size_t)first * 512, first);
          first = nxt;
          continue;
        }
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose, rocblas_operation_none,
                                    nb, count * 4, 128, &one, keys, 128,
                                    d_iq + (size_t)(first - base) * 512, 128, &zero,
                                    d_iscores, nb);
        if (status != rocblas_status_success) {
          fprintf(stderr, "indexer sgemm failed: %d\n", (int)status);
          exit(1);
        }
        index_select(d_iscores, nb, d_iselected + (size_t)first * 512, first, count);
        first = nxt;
      }
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_qgb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_kbb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.v_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_vbb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_norm.weight", l);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_norm.weight", l);
    float* knw = ensure_norm256(buf);
    const size_t qsa_off = (size_t)qi * maxctx * 512;
    if (!qsa_loop) {
      ensure_rope_tab(g_cfg.rope_theta, 64);
      ensure_rope_cs(base, P);
      // fused: qsplit+rope (K1) and knorm+rope[+bf16 store] (K2), bit-exact
      // vs the previous qsplit / rmsnorm_zc / rope_b x2 / convert chain
      k_qsa_qsplit<true><<<dim3(24, P), 256, 0, g_str>>>(d_qgb, qnw, d_qsb, d_gsb,
                                                         256, g_cfg.norm_eps, base,
                                                         d_ropecs);
      int dense_P = qsa_dense ? P : min(P, std::max(0, 2051 - base));
      if (qsa_kv_fp32) {
        float* kc = d_kc + qsa_off;
        float* vc = d_vc + qsa_off;
        k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(d_kbb, knw, d_kbb,
                                                          nullptr, g_cfg.norm_eps,
                                                          base, d_ropecs);
        // absolute positions base..base+P-1 are contiguous: one bulk copy
        CK(hipMemcpyAsync(kc + (size_t)base * 512, d_kbb, (size_t)P * 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        CK(hipMemcpyAsync(vc + (size_t)base * 512, d_vbb, (size_t)P * 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        qsa_flash_b(kc, vc, P, dense_P, base);
      } else {
        uint16_t* kc = d_kcb + qsa_off;
        uint16_t* vc = d_vcb + qsa_off;
        k_qsa_kprep<true><<<dim3(2, P), 256, 0, g_str>>>(
            d_kbb, knw, d_kbb, kc + (size_t)base * 512, g_cfg.norm_eps, base,
            d_ropecs);
        // absolute positions base..base+P-1 are contiguous: one bulk convert
        k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256, 0,
                           g_str>>>(d_vbb, vc + (size_t)base * 512, 512, P, 512);
        const size_t bt_layer = qsa_vt_blocks * 2 * 256 * 4;
        uint16_t* vct = qsa_wmma_btv
                            ? d_vcbt + (size_t)qi * bt_layer
                            : nullptr;
        if (vct)
          k_f32_to_bf16_v4_bt<<<(unsigned)(((size_t)((base + P + 3) / 4 - base / 4) *
                                            512 + 255) /
                                               256),
                               256, 0, g_str>>>(d_vbb, vct, 512, P, 512, base);
        qsa_flash_b(kc, vc, P, dense_P, base, vct);
      }
      k_sigmoid_gate<<<(P * 6144 + 255) / 256, 256, 0, g_str>>>(d_attnb, d_gsb,
                                                                d_attnb, P * 6144);
    } else {
      for (int t = 0; t < P; t++) {
        const int* pabs = d_posarr + base + t;  // absolute position
        const float* qg = d_qgb + (size_t)t * 12288;
        float* qs = d_qsb + (size_t)t * 6144;
        float* gs = d_gsb + (size_t)t * 6144;
        float* kb = d_kbb + (size_t)t * 512;
        float* vb = d_vbb + (size_t)t * 512;
        float* att = d_attnb + (size_t)t * 6144;
        k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(qg, qnw, qs, gs, 256,
                                                     g_cfg.norm_eps);
        k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(kb, knw, kb, 256, g_cfg.norm_eps, 1);
        k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(qs, 24, 256, 64, pabs,
                                                         g_cfg.rope_theta);
        k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(kb, 2, 256, 64, pabs,
                                                        g_cfg.rope_theta);
        if (qsa_kv_fp32) {
          k_store_kv<<<2, 256, 0, g_str>>>(kb, vb, d_kc + qsa_off, d_vc + qsa_off,
                                           pabs);
          k_qsa_step<float><<<24, 256, 0, g_str>>>(
              qs, d_kc + qsa_off, d_vc + qsa_off, att, pabs, 256, 2, 24,
              qsa_dense ? nullptr : d_iselected + (size_t)(base + t) * 512);
        } else {
          k_store_kv_bf16<<<2, 256, 0, g_str>>>(kb, vb, d_kcb + qsa_off,
                                                d_vcb + qsa_off, pabs);
          k_qsa_step<uint16_t><<<24, 256, 0, g_str>>>(
              qs, d_kcb + qsa_off, d_vcb + qsa_off, att, pabs, 256, 2, 24,
              qsa_dense ? nullptr : d_iselected + (size_t)(base + t) * 512);
        }
        k_sigmoid_gate<<<(6144 + 255) / 256, 256, 0, g_str>>>(att, gs, att, 6144);
      }
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.o_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_attnb, d_yb, P);
  }

  // MoE: router batched; expert GEMVs grouped over P*k slots (naive: full
  // weight traffic per token — correct but slow; optimized in a later phase).
  void moe_b(int l, int P) {
    char buf[128];
    const bool direct_output = pp_moe_out && P > 8;
    float* const acc = direct_output ? d_yb : d_moeaccb;
    snprintf(buf, sizeof buf, "layers.%d.mlp.gate.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_routerb, P);
    int k = g_cfg.topk;
    k_router_topk<<<P, 512, 0, g_str>>>(d_routerb, d_topidsb, d_topwb, g_cfg.experts, k);
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.gate_up_proj.weight", l);
    WView gu = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.down_proj.weight", l);
    WView dn = wview(ckpt.at(buf));
    // fused kernels require exact tiling (guards would spill accumulators)
    bool fused_ok = g_cfg.moe_mid % MOE_CT == 0 && g_cfg.d % MOE_CT == 0 &&
                    g_cfg.d % FD_CT_DN == 0 &&
                    gu.cols % MOE_KC == 0 && dn.cols % MOE_KC == 0;
    bool naive_path = moe_naive || !fused_ok || P <= moe_naive_max;
    bool direct_down = naive_path && moe_down_topk && k == 10 && dn.cols == 640;
    // Direct-down and deterministic reductions overwrite every output element.
    // Atomic/ordered accumulation still requires a zero destination.
    if (!direct_down && (!direct_output || naive_path || moe_ordered || !moe_deterministic))
      CK(hipMemsetAsync(acc, 0, (size_t)P * g_cfg.d * 4, g_str));
    const int* topidsb = d_topidsb;
    if (naive_path) {
      // v3 kernels: 2 rows per warp -> halved warp counts
      uint64_t pairs = (uint64_t)P * k * g_cfg.moe_mid;
      k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
          d_xb, d_guvb, topidsb, (uint64_t)2 * g_cfg.moe_mid, gu.cols,
          gu.scale_stride, P * k, (uint64_t)g_cfg.d, 16, k);
      k_silu_mul_g<<<(P * k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(
          d_guvb, d_hidb, g_cfg.moe_mid, P * k);
      if (direct_down) {
        uint64_t row_pairs = (g_cfg.d + 1) / 2;
        constexpr uint64_t warps_per_block = 128 / 32;
        k_q4cp_gemv_gd_topk_h16<10><<<
            (unsigned)(((uint64_t)P * row_pairs + warps_per_block - 1) /
                       warps_per_block),
            128, 0, g_str>>>(
            dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
            (const float*)dn.data, d_hidb, acc, topidsb, d_topwb,
            (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
            (uint64_t)g_cfg.moe_mid, P, 16, (uint64_t)g_cfg.d);
      } else {
        uint64_t pairs2 = (uint64_t)P * k * ((g_cfg.d + 1) / 2);
        k_q4cp_gemv_gd<<<(unsigned)((pairs2 + 15) / 16), 512, 0, g_str>>>(
            dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
            (const float*)dn.data, d_hidb, acc, topidsb, d_topwb,
            (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
            (uint64_t)g_cfg.moe_mid, P * k, 16, (uint64_t)g_cfg.d, k);
      }
    } else {
      // Stable expert bucketing preserves token order and deterministic reduction.
      int E = g_cfg.experts, npairs = P * k;
      if (!moe_host_route && !moe_ordered) {
        moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff,
                        moe_deterministic ? d_pairids : nullptr, P, k, E);
      } else {
        std::vector<int> hids(P * 16);  // topids rows are 16 wide, first k valid
        std::vector<float> hws(P * 16);
        auto pt1 = std::chrono::steady_clock::now();
        CK(hipMemcpy(hids.data(), d_topidsb, (size_t)P * 16 * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hws.data(), d_topwb, (size_t)P * 16 * 4, hipMemcpyDeviceToHost));
        auto pt2 = std::chrono::steady_clock::now();
        std::vector<int> off(E + 1, 0);
        for (int t = 0; t < P; t++)
          for (int s = 0; s < k; s++) off[hids[t * 16 + s] + 1]++;
        for (int e = 0; e < E; e++) off[e + 1] += off[e];
        std::vector<int> cur(off.begin(), off.end() - 1);
        std::vector<int> tidx(npairs);
        std::vector<float> pw(npairs);
        for (int t = 0; t < P; t++)
          for (int s = 0; s < k; s++) {
            int slot = cur[hids[t * 16 + s]]++;
            tidx[slot] = t;
            pw[slot] = hws[t * 16 + s];
          }
        CK(hipMemcpy(d_tokidx, tidx.data(), npairs * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_pweight, pw.data(), npairs * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_eoff, off.data(), (E + 1) * 4, hipMemcpyHostToDevice));
        if (moe_deterministic && !moe_ordered) {
          std::vector<int> pairids(npairs), next(P, 0);
          for (int p = 0; p < npairs; p++) {
            int t = tidx[p];
            pairids[t * k + next[t]++] = p;
          }
          CK(hipMemcpy(d_pairids, pairids.data(), npairs * 4, hipMemcpyHostToDevice));
        }
        if (prof) {
          auto pt3 = std::chrono::steady_clock::now();
          p_topk_wait +=
              std::chrono::duration<double, std::milli>(pt2 - pt1).count();
          p_bucket += std::chrono::duration<double, std::milli>(pt3 - pt2).count();
        }
      }
      bool tiled = !moe_untiled && !moe_ordered;
      // Packed bf16 x/hid no longer requires the tiled grid: the untiled
      // expert-major grid resolves rows from eoff and is bit-identical.
      bool packed = !moe_ordered && moe_deterministic && !moe_fp32_io;
      int tasks = (npairs + 63) / 64 + E;
      bool lt_ran = false;
      if (moe_lt && k == 10 && moe_deterministic && !moe_host_route && !moe_ordered &&
          P >= moe_lt_min) {
        lt_ran = true;
        // Dequant + per-expert hipBLASLt path (see kernels' banner comment).
        // One host readback per layer to get expert offsets for the launch
        // loop. Async into a pinned buffer + event, issued BEFORE the
        // convert+gather launches: the host waits only for the 2KB copy
        // while the GPU works through gather (~16 ms), instead of the GPU
        // draining idle before a blocking sync (~5.5 ms/layer of GPU gap).
        const int mid = g_cfg.moe_mid, dd = g_cfg.d;
        static int* hoff_pin = nullptr;
        static hipEvent_t ev_eoff = nullptr;
        if (!hoff_pin) {
          CK(hipHostMalloc((void**)&hoff_pin, (size_t)(E + 1) * 4));
          CK(hipEventCreateWithFlags(&ev_eoff, hipEventDisableTiming));
        }
        CK(hipMemcpyAsync(hoff_pin, d_eoff, (E + 1) * 4, hipMemcpyDeviceToHost,
                          g_str));
        CK(hipEventRecord(ev_eoff, g_str));
        if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
          k_f32_to_bf16_v4<<<((size_t)P * dd / 4 + 255) / 256, 256, 0, g_str>>>(
              d_xb, d_xbf16, dd, P, dd);
          xf16_src = d_xb;
          xf16_xs = xf16_K = dd;
          xf16_P = P;
        }
        k_moe_gather_bf16<<<npairs, 256, 0, g_str>>>(
            d_moexg, (const __hip_bfloat16*)d_xbf16, d_tokidx, dd);
        CK(hipEventSynchronize(ev_eoff));
        const int* hoff = hoff_pin;
        bool b16 = moe_lt_bf16;
        __hip_bfloat16* guv16 = (__hip_bfloat16*)d_guvb;
        // up phase: group dequant -> per-expert GEMM -> segment silu
        // GDEC_MOE_LT_OVL pipeline: scratch is double-buffered (slots gi&1);
        // group gi+1's dequant runs on ovl_str while g_str runs group gi's
        // GEMMs. ovl_deq[b] = deq into slot b done; ovl_gemm[b] = GEMMs that
        // read slot b done. deq(gi+1) waits on the group gi-1 GEMMs (the
        // previous reader of its slot). g_str never touches slot b before
        // waiting ovl_deq[b], and by loop end has waited on every ovl_str
        // deq, so the next phase/layer (whose ovl_str work chains behind
        // these deqs and g_str's later GEMM event records) cannot race the
        // slots either. Prefill only runs outside graph capture.
        static hipStream_t ovl_str = nullptr;
        static hipEvent_t ovl_deq[2] = {nullptr, nullptr};
        static hipEvent_t ovl_gemm[2] = {nullptr, nullptr};
        if (moe_lt_ovl && !ovl_str) {
          // Non-blocking: g_str is the legacy default stream outside graph
          // capture, and a blocking side stream would implicitly serialize
          // with it (measured: -16% at 8K, no overlap at all).
          CK(hipStreamCreateWithFlags(&ovl_str, hipStreamNonBlocking));
          for (int i = 0; i < 2; i++) {
            CK(hipEventCreateWithFlags(&ovl_deq[i], hipEventDisableTiming));
            CK(hipEventCreateWithFlags(&ovl_gemm[i], hipEventDisableTiming));
          }
        }
        const int ngrp = (E + 3) / 4;
        auto deq_up = [&](int gi, hipStream_t st) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          int64_t total = (int64_t)g * 2 * mid * (dd / 32);
          k_moe_deq_bf16<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
              gu.data + 64 + (size_t)e0 * 2 * mid * (dd / 2),
              gu.data + 64 + gu.rows * gu.cols / 2 +
                  (size_t)e0 * 2 * mid * gu.scale_stride,
              (const float*)gu.data,
              d_moewup + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * 2 * mid * dd,
              g * 2 * mid, dd, gu.scale_stride);
        };
        auto deq_dn = [&](int gi, hipStream_t st) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          int64_t total = (int64_t)g * dd * (mid / 32);
          k_moe_deq_bf16<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
              dn.data + 64 + (size_t)e0 * dd * (mid / 2),
              dn.data + 64 + dn.rows * dn.cols / 2 +
                  (size_t)e0 * dd * dn.scale_stride,
              (const float*)dn.data,
              d_moewdn + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * dd * mid,
              g * dd, mid, dn.scale_stride);
        };
        if (moe_lt_ovl) {
          deq_up(0, ovl_str);
          CK(hipEventRecord(ovl_deq[0], ovl_str));
        }
        for (int gi = 0; gi < ngrp; gi++) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          __hip_bfloat16* wup =
              d_moewup + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * 2 * mid * dd;
          if (moe_lt_ovl) {
            if (gi + 1 < ngrp) {
              if (gi >= 1)
                CK(hipStreamWaitEvent(ovl_str, ovl_gemm[(gi + 1) & 1], 0));
              deq_up(gi + 1, ovl_str);
              CK(hipEventRecord(ovl_deq[(gi + 1) & 1], ovl_str));
            }
            CK(hipStreamWaitEvent(g_str, ovl_deq[gi & 1], 0));
          } else {
            deq_up(gi, g_str);
          }
          for (int e = e0; e < e0 + g; e++) {
            int n0 = hoff[e], m = hoff[e + 1] - n0;
            if (!m) continue;
            void* cup = b16 ? (void*)(guv16 + (size_t)n0 * 2 * mid)
                            : (void*)(d_guvb + (size_t)n0 * 2 * mid);
            gemm_bf16(wup + (size_t)(e - e0) * 2 * mid * dd,
                      d_moexg + (size_t)n0 * dd, cup, 2 * mid, dd, m, b16);
            int64_t seg = (int64_t)m * mid;
            if (b16)
              k_moe_silu_bf16<<<(unsigned)((seg + 255) / 256), 256, 0, g_str>>>(
                  guv16 + (size_t)n0 * 2 * mid,
                  (__hip_bfloat16*)d_hidb + (size_t)n0 * mid, mid, m);
            else
              k_moe_silu_f32_bf16<<<(unsigned)((seg + 255) / 256), 256, 0, g_str>>>(
                  d_guvb + (size_t)n0 * 2 * mid,
                  (__hip_bfloat16*)d_hidb + (size_t)n0 * mid, mid, m);
          }
          if (moe_lt_ovl) CK(hipEventRecord(ovl_gemm[gi & 1], g_str));
        }
        if (moe_lt_cmp && l == 0) {
          // Reference: fused packed up on identical inputs -> d_hidb2.
          k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E);
          k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0,
                                       g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2,
              (const float*)gu.data, d_xb, d_tokidx, d_eoff, nullptr, 2 * mid,
              gu.cols, mid, gu.scale_stride, d_moetiles, d_moentiles, d_xbf16,
              (uint16_t*)d_hidb2);
          CK(hipMemsetAsync(d_cmpi, 0, 8, g_str));
          int64_t nh = (int64_t)npairs * mid;
          k_cmp_absmax_bf16<<<(unsigned)std::min<int64_t>(1024, (nh + 255) / 256),
                              256, 0, g_str>>>(
              (const __hip_bfloat16*)d_hidb, (const __hip_bfloat16*)d_hidb2, nh,
              d_cmpi);
          int hc[2];
          CK(hipMemcpy(hc, d_cmpi, 8, hipMemcpyDeviceToHost));
          float fd, fr;
          memcpy(&fd, &hc[0], 4);
          memcpy(&fr, &hc[1], 4);
          fprintf(stderr, "moe_lt cmp up-hid: max|d|=%g max|ref|=%g\n", fd, fr);
        }
        // down phase: d_guvb is dead after silu; pair outputs alias it.
        if (moe_lt_ovl) {
          deq_dn(0, ovl_str);
          CK(hipEventRecord(ovl_deq[0], ovl_str));
        }
        for (int gi = 0; gi < ngrp; gi++) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          __hip_bfloat16* wdn =
              d_moewdn + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * dd * mid;
          if (moe_lt_ovl) {
            if (gi + 1 < ngrp) {
              if (gi >= 1)
                CK(hipStreamWaitEvent(ovl_str, ovl_gemm[(gi + 1) & 1], 0));
              deq_dn(gi + 1, ovl_str);
              CK(hipEventRecord(ovl_deq[(gi + 1) & 1], ovl_str));
            }
            CK(hipStreamWaitEvent(g_str, ovl_deq[gi & 1], 0));
          } else {
            deq_dn(gi, g_str);
          }
          for (int e = e0; e < e0 + g; e++) {
            int n0 = hoff[e], m = hoff[e + 1] - n0;
            if (!m) continue;
            void* cdn = b16 ? (void*)(guv16 + (size_t)n0 * dd)
                            : (void*)(d_guvb + (size_t)n0 * dd);
            gemm_bf16(wdn + (size_t)(e - e0) * dd * mid,
                      (const __hip_bfloat16*)d_hidb + (size_t)n0 * mid, cdn, dd,
                      mid, m, b16);
          }
          if (moe_lt_ovl) CK(hipEventRecord(ovl_gemm[gi & 1], g_str));
        }
        if (moe_lt_cmp && l == 0 && b16) {
          // Reference: fused packed down on LT's hid -> weighted fp32 pairs.
          k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks),
                                             256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
              (const float*)dn.data, nullptr, d_tokidx, d_pweight, d_eoff,
              d_pairs2, dd, dn.cols, dn.scale_stride, d_moetiles, d_moentiles,
              (const uint16_t*)d_hidb);
          CK(hipMemsetAsync(d_cmpi, 0, 8, g_str));
          int64_t npr = (int64_t)npairs * dd;
          k_cmp_absmax_pw<<<(unsigned)std::min<int64_t>(1024, (npr + 255) / 256),
                            256, 0, g_str>>>(
              d_pairs2, (const __hip_bfloat16*)d_guvb, d_pweight, npairs, dd,
              d_cmpi);
          int hc[2];
          CK(hipMemcpy(hc, d_cmpi, 8, hipMemcpyDeviceToHost));
          float fd, fr;
          memcpy(&fd, &hc[0], 4);
          memcpy(&fr, &hc[1], 4);
          fprintf(stderr, "moe_lt cmp down-pairs: max|d|=%g max|ref|=%g\n", fd,
                  fr);
        }
        if (b16)
          k_moe_reduce_pw_bf16<10><<<dim3((dd + 255) / 256, P), 256, 0, g_str>>>(
              (const __hip_bfloat16*)d_guvb, d_pairids, d_pweight, acc, P, dd);
        else
          k_moe_reduce_pw<10><<<dim3((dd + 255) / 256, P), 256, 0, g_str>>>(
              d_guvb, d_pairids, d_pweight, acc, P, dd);
      } else if (tiled) {
        k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E);
        if (packed) {
          if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
            k_f32_to_bf16_v4<<<((size_t)P * g_cfg.d / 4 + 255) / 256, 256, 0, g_str>>>(
                d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
            xf16_src = d_xb;
            xf16_xs = xf16_K = g_cfg.d;
            xf16_P = P;
          }
          // Hidden BF16 occupies the existing workspace until the shared expert rewrites it.
          if (moe_up_legacy)
            k_moe_w4_up<true, true, false, false><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
          else if (moe_up_no_table)
            k_moe_w4_up<true, true, true, false><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
          else
            k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
        } else {
          k_moe_w4_up<true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, d_hidb, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride, d_moetiles, d_moentiles);
        }
      } else {
        if (packed) {
          if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
            k_f32_to_bf16_v4<<<((size_t)P * g_cfg.d / 4 + 255) / 256, 256, 0, g_str>>>(
                d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
            xf16_src = d_xb;
            xf16_xs = xf16_K = g_cfg.d;
            xf16_P = P;
          }
          k_moe_w4_up<false, true><<<dim3(g_cfg.moe_mid / MOE_CT, E), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride, nullptr, nullptr, d_xbf16, (uint16_t*)d_hidb);
        } else {
          k_moe_w4_up<<<dim3(g_cfg.moe_mid / MOE_CT, E), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, d_hidb, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride);
        }
      }
      if (lt_ran) {
        // LT branch already produced acc (down + weighted reduce).
      } else if (moe_ordered) {
        for (int e = 0; e < E; e++)
          k_moe_w4_down<<<dim3(g_cfg.d / FD_CT_DN, 1), 256, 0, g_str>>>(
              dn.data + 64 + (size_t)e * g_cfg.d * dn.cols / 2,
              dn.data + 64 + dn.rows * dn.cols / 2 + (size_t)e * g_cfg.d * dn.scale_stride,
              (const float*)dn.data, d_hidb, d_tokidx, d_pweight, d_eoff + e,
              acc, g_cfg.d, dn.cols, dn.scale_stride);
      } else if (moe_deterministic) {
        // GDEC_MOE_PAIRS_BF16: down writes unweighted bf16 pair rows; the
        // router weight moves to the reduce (fp32 multiply). Requires the
        // fixed top-10 reduce below.
        bool pairs_bf16 = moe_pairs_bf16 && k == 10 && !moe_reduce_old;
        if (packed) {
          if (tiled) {
            if (moe_down_no_table) {
              if (pairs_bf16)
                k_moe_w4_down<false, true, true, false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
              else
                k_moe_w4_down<false, true, true, false><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
            } else {
              if (pairs_bf16)
                k_moe_w4_down<false, true, true, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
              else
                k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
            }
          } else {
            if (pairs_bf16)
              k_moe_w4_down<false, false, true, true, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                  dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                  nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                  dn.scale_stride, nullptr, nullptr, (const uint16_t*)d_hidb);
            else
              k_moe_w4_down<false, false, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                  dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                  nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                  dn.scale_stride, nullptr, nullptr, (const uint16_t*)d_hidb);
          }
        } else if (tiled) {
          if (pairs_bf16)
            k_moe_w4_down<false, true, false, false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride, d_moetiles, d_moentiles);
          else
            k_moe_w4_down<false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride, d_moetiles, d_moentiles);
        } else {
          if (pairs_bf16)
            k_moe_w4_down<false, false, false, false, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride);
          else
            k_moe_w4_down<false><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride);
        }
        if (pairs_bf16)
          k_moe_reduce_pw_bf16<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
              (const __hip_bfloat16*)d_guvb, d_pairids, d_pweight, acc, P, g_cfg.d);
        else if (k == 10 && !moe_reduce_old)
          k_moe_reduce_fast<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
              d_guvb, d_pairids, acc, P, g_cfg.d);
        else
          k_moe_reduce<<<(P * g_cfg.d + 255) / 256, 256, 0, g_str>>>(
              d_guvb, d_pairids, acc, P, k, g_cfg.d);
      } else {
        if (tiled) {
          k_moe_w4_down<true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
              d_hidb, d_tokidx, d_pweight, d_eoff, acc, g_cfg.d, dn.cols,
              dn.scale_stride, d_moetiles, d_moentiles);
        } else {
          k_moe_w4_down<<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
              d_hidb, d_tokidx, d_pweight, d_eoff, acc, g_cfg.d, dn.cols,
              dn.scale_stride);
        }
      }
    }
    // shared expert (gate/up scratch in d_guvb head, silu out in d_hidb head)
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.gate_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_guvb, P);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.up_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_guvb + (size_t)P * 640, P);
    k_silu_mul<<<(P * 640 + 255) / 256, 256, 0, g_str>>>(d_guvb, d_guvb + (size_t)P * 640,
                                                         d_hidb, P * 640);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.down_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_hidb, d_eyb, P);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert_gate.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_sgb, P);
    k_axpy_sg<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(acc, d_eyb, d_sgb,
                                                         g_cfg.d, P * g_cfg.d);
    if (!direct_output)
      CK(hipMemcpyAsync(d_yb, acc, (size_t)P * g_cfg.d * 4, hipMemcpyDeviceToDevice,
                        g_str));
  }

  // PLE device phase for one prefill token (embeddings already in d_eb).
  void ple_gpu_t(int t, int base = 0) {
    const float* e = d_eb + (size_t)t * 2560;
    gemv(wview(ckpt.at("layers.1.ple.key_proj.weight")), e, d_keys);
    gemv(wview(ckpt.at("layers.1.ple.value_proj.weight")), e, d_v);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_keys, d_ple_nk, d_kn, 2560,
                                                g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_Rb + (size_t)t * 10240,
                                                d_ple_nk + 10240, d_qn, 2560,
                                                g_cfg.norm_eps, 4);
    k_ple_gate<<<4, 1024, 0, g_str>>>(d_kn, d_qn, d_v, d_U, 2560, 4);
    k_rmsnorm_zc_ring<<<4, 1024, 0, g_str>>>(d_U, d_ple_nk + 20480, d_ring,
                                             d_ringarr + base + t, 2560,
                                             g_cfg.norm_eps);
    k_ple_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_ring, d_ringarr + base + t,
                                                       d_ple_convw, d_convout, 10240);
    k_add2<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_Rb + (size_t)t * 10240, d_U,
                                                   d_convout, 10240);
  }

  // Batched PLE over all P positions (embeddings already in d_eb). Bitwise
  // matches the ple_gpu_t loop: same per-block reductions, ring semantics of
  // k_ple_conv (tap back-9 aliases the current slot) folded into k_ple_conv_b.
  void ple_gpu_b(int P, int base = 0) {
    gemm(wview(ckpt.at("layers.1.ple.key_proj.weight")), d_eb, d_keysb, P);
    gemm(wview(ckpt.at("layers.1.ple.value_proj.weight")), d_eb, d_plevb, P);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_keysb, d_ple_nk, d_keysb, 2560,
                                                    g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_Rb, d_ple_nk + 10240, d_qnb,
                                                    2560, g_cfg.norm_eps, 4);
    k_ple_gate_b<<<dim3(4, P), 1024, 0, g_str>>>(d_keysb, d_qnb, d_plevb, d_Ub, 2560,
                                                 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_Ub, d_ple_nk + 20480, d_Unb,
                                                    2560, g_cfg.norm_eps, 4);
    int64_t tot = (int64_t)P * 10240;
    k_ple_conv_b<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
        d_Unb, d_ple_convw, d_convoutb, P, 10240, base, d_ring);
    k_add2<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(d_Rb, d_Ub, d_convoutb,
                                                             (int)tot);
    k_ple_ring_wr<<<9, 256, 0, g_str>>>(d_Unb, d_ring, P, 10240, base);
  }

  void enq_layer_b(int l, int P, bool fuse_prev_mlp = false,
                   bool flush_mlp = true, int base = 0) {
    char attn_prefix[128], mlp_prefix[128];
    snprintf(attn_prefix, sizeof attn_prefix,
             "layers.%d.attn_hyper_connection", l);
    if (fuse_prev_mlp) {
      snprintf(mlp_prefix, sizeof mlp_prefix,
               "layers.%d.mlp_hyper_connection", l - 1);
      gr_write_read_b(mlp_prefix, attn_prefix, P);
    } else {
      gr_read_b(attn_prefix, P);
    }
    if (is_qsa(l))
      qsa_b(l, l / 4, P, base);
    else
      gdn_b(l, l - (l + 1) / 4, P);
    if (dbg) {
      h_rms("gpu", l, "attn-x", d_xb + (size_t)(P - 1) * 2560, 2560);
      h_rms("gpu", l, "attn-y", d_yb + (size_t)(P - 1) * 2560, 2560);
    }
    snprintf(mlp_prefix, sizeof mlp_prefix,
             "layers.%d.mlp_hyper_connection", l);
    if (gr_scatter_norm)
      gr_write_read_b(attn_prefix, mlp_prefix, P);
    else {
      gr_write_b(attn_prefix, P);
      gr_read_b(mlp_prefix, P);
    }
    moe_b(l, P);
    if (dbg) {
      h_rms("gpu", l, "mlp-x", d_xb + (size_t)(P - 1) * 2560, 2560);
      h_rms("gpu", l, "mlp-y", d_yb + (size_t)(P - 1) * 2560, 2560);
    }
    if (flush_mlp) gr_write_b(mlp_prefix, P);
    if (!dbg && isatty(2)) {  // per-layer \r is terminal-only; logs use prefill-live
      fprintf(stderr, "\rprefill L%02d/%d", l + 1, g_cfg.layers);
      fflush(stderr);
    }
  }

  // Batched prefill: returns first generated token (argmax at last position).
  // Batched prefill: returns first generated token (argmax at last position).
  // base = absolute position of tokens[0] (KV-continuation); state handoff:
  // d_R = last row, pos = base+P, d_pos/d_ringpos absolute, chist/ring/KV/GDN
  // state fully advanced.
  // Chunked prefill: prompts longer than maxbatch are processed in
  // maxbatch-token continuations (the same path KV reuse uses), so batch
  // workspace stays chunk-sized regardless of maxctx.
  // 1s-cadence prefill progress: record a completion probe after layer l's
  // enqueue, and once a second report the highest COMPLETED layer as an
  // estimated token count. hipEventQuery is non-blocking (hipErrorNotReady
  // is the expected answer for in-flight layers), so this never serializes
  // the pipeline the way the GDEC_PHASE sync profiling does.
  void prog_print(int P) {
    auto now = std::chrono::steady_clock::now();
    double iv = std::chrono::duration<double>(now - prog_last).count();
    if (iv < 1.0) return;
    int done = -1;
    for (int k = g_cfg.layers - 1; k >= 0; --k)
      if (prog_ev[k] && hipEventQuery(prog_ev[k]) == hipSuccess) {
        done = k;
        break;
      }
    // done+1 of g_cfg.layers layers finished (L0 counts as one layer).
    double frac = double(done + 1) / g_cfg.layers;
    int64_t est = prog_base + (int64_t)(P * frac);
    double el = std::chrono::duration<double>(now - prog_t0).count();
    fprintf(stderr,
            "%s req %lld | prompt processing | n_tokens = %lld/%d (%.0f%%) "
            "| t = %.1f s | %.0f tok/s (avg %.0f)\n",
            log_ts(), (long long)prog_req, (long long)est, prog_total,
            100.0 * est / prog_total, el, (est - prog_last_est) / iv,
            el > 0 ? est / el : 0.0);
    prog_last = now;
    prog_last_est = est;
  }
  void prog_probe(int l, int P) {
    if (l < 0 || l >= 64) return;
    if (!prog_ev[l])
      CK(hipEventCreateWithFlags(&prog_ev[l], hipEventDisableTiming));
    CK(hipEventRecord(prog_ev[l], g_str));
    prog_print(P);
  }
  // The layer enqueue is fully asynchronous: every probe above fires within
  // the first fraction of a second of host time, then the host blocks on the
  // final readback. Real progress is therefore reported while WAITING for
  // the graph to drain — poll the last layer's event here at 1 s cadence.
  void prog_wait(int P) {
    if (!prog_ev[g_cfg.layers - 1]) return;
    while (hipEventQuery(prog_ev[g_cfg.layers - 1]) == hipErrorNotReady) {
      prog_print(P);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  int prefill_batch(const std::vector<int>& tokens, int base = 0, bool final = true,
                    bool quiet = false) {
    int N = (int)tokens.size();
    int id = -1;
    for (int off = 0; off < N;) {
      int rem = N - off;
      int n = std::min(maxbatch, rem);
      // Absorb a small tail into this chunk instead of running a tiny final
      // chunk whose per-chunk fixed cost (~0.9 s) dwarfs its token count.
      if (rem > maxbatch && rem <= maxbatch_cap) n = rem;
      id = prefill_chunk(
          std::vector<int>(tokens.begin() + off, tokens.begin() + off + n),
          base + off, final, quiet);
      if (mtp_tap_capture) {
        // Ingest this chunk's taps (chunk-local rows in d_mtap) into the MTP
        // layer before the next chunk overwrites them. MTP position i takes
        // token i+1, so the very last prompt row is left to the spec loop.
        int c0 = base + off;
        int count = std::min(n, base + N - 1 - c0);
        if (count > 0) {
          std::vector<int> shifted(tokens.begin() + off + 1,
                                   tokens.begin() + off + 1 + count);
          mtp_ingest_b(d_mtap, shifted, count, c0);
        }
      }
      off += n;
    }
    return id;
  }

  int prefill_chunk(const std::vector<int>& tokens, int base = 0, bool final = true,
                    bool quiet = false) {
    int P = (int)tokens.size();
    if (P < 1 || base + P > maxctx || base < 0 || P > maxbatch_cap) {
      fprintf(stderr, "prefill_chunk: base=%d P=%d out of range "
              "(maxctx=%d maxbatch_cap=%d)\n", base, P, maxctx, maxbatch_cap);
      exit(1);
    }
    static const bool phase_prof = getenv("GDEC_PHASE") != nullptr;
    auto phase = [&](const char* tag, std::chrono::steady_clock::time_point& tp) {
      if (!phase_prof) return;
      CK(hipStreamSynchronize(g_str));
      auto now = std::chrono::steady_clock::now();
      fprintf(stderr, "phase %-10s %8.1f ms\n", tag,
              std::chrono::duration<double, std::milli>(now - tp).count());
      tp = now;
    };
    xf16_src = nullptr;  // a new batch invalidates the bf16 conversion cache
#ifdef GDEC_PHASE_PROF
    {
      static const unsigned long long zeros[16] = {};
      CK(hipMemcpyToSymbol(g_ph, zeros, sizeof(zeros)));
    }
#endif
    auto T0 = std::chrono::steady_clock::now();
    auto TP = T0;
    // PLE rows depend only on token history, so start the existing page-fault
    // pipeline before L0. Its GPU consumer remains ordered after L0 below.
    std::vector<std::thread> ple_pool;
    std::chrono::steady_clock::time_point ple_start, ple_done;
    std::atomic<int> ple_workers{0};
    int ple_base0 = 0;
    uint8_t* ple_fp8 = nullptr;
    // two-phase PLE gather state (must outlive the async workers — every
    // variable the pool lambdas capture by reference lives at this scope;
    // declaring aliases/sizes inside the gather block is
    // stack-use-after-scope once the workers outlive that block. ASan-proven
    // in plegs v1/v2; the pre-plegs code had the same latent bug via
    // block-scope reference aliases — plfix removes them).
    std::vector<uint64_t> ple_addrs, ple_keys;
    std::atomic<int> ple_barrier{0};
    if (ple_on) {
      ple_start = std::chrono::steady_clock::now();
      for (int t = 0; t < P; t++) chist.push_back(tokens[t]);
      ple_base0 = (int)chist.size() - P;
      ple_fp8 = (uint8_t*)ple_eb;
      constexpr int nt = 16;
      ple_workers.store(nt);
      ple_pool.reserve(nt);
      if (!getenv("GDEC_PLE_NOSORT")) {
        ple_barrier.store(0, std::memory_order_relaxed);
        if (ple_uring_ready()) {
          // io_uring batch-pread gather: phase A is the same 16-thread row
          // address computation; worker 0 then pumps all 16*P rows in token
          // order through the 256-deep ring straight into pinned staging.
          ple_addrs.resize((size_t)P * 16);
          ple_workers.store(1);  // only the pump thread accounts for ple_done
          for (int w = 0; w < nt; w++)
            ple_pool.emplace_back([&, w] {
              const int t0 = (int)((size_t)w * P / nt), t1 = (int)((size_t)(w + 1) * P / nt);
              for (int t = t0; t < t1; t++) {
                const int at = ple_base0 + t;
                auto cat = [&](int back) -> int64_t {
                  int j = at - back;
                  return j < 0 ? pad_id : chist[j];
                };
                for (int h = 0; h < 16; h++) {
                  int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
                  if (h >= 8) mix ^= cat(2) * ple_mult[2];
                  uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
                  ple_addrs[(size_t)t * 16 + h] = (uint64_t)(uintptr_t)(ple_table->data + row * 160);
                }
              }
              ple_barrier.fetch_add(1, std::memory_order_acq_rel);
              if (w != 0) return;
              while (ple_barrier.load(std::memory_order_acquire) < nt)
                std::this_thread::yield();
              PLE_PUMP.run(ple_addrs.data(), ple_fp8,
                           (uint64_t)(uintptr_t)ple_table->data, (size_t)P * 16);
              if (ple_workers.fetch_sub(1) == 1)
                ple_done = std::chrono::steady_clock::now();
            });
        } else {
        // Two-phase gather: phase A computes all 16*P row addresses, sorts
        // each shard by page and issues madvise in ascending order (random
        // 4K reads become quasi-sequential for the disk scheduler, adjacent
        // pages dedup); phase B copies rows in token order with pages
        // already in flight. Output staging bytes are identical to the
        // per-token madvise+gather pipeline. The PLE table is mostly
        // page-cache-cold in steady state (124GB file > RAM), so submission
        // order is the cost driver (measured: 512K random madvise ~2.5s
        // serial-equivalent, warm gather ~0.01s).
        ple_addrs.resize((size_t)P * 16);
        ple_keys.resize((size_t)P * 16);
        for (int w = 0; w < nt; w++)
          ple_pool.emplace_back([&, w] {
            const int t0 = (int)((size_t)w * P / nt), t1 = (int)((size_t)(w + 1) * P / nt);
            for (int t = t0; t < t1; t++) {
              const int at = ple_base0 + t;
              auto cat = [&](int back) -> int64_t {
                int j = at - back;
                return j < 0 ? pad_id : chist[j];
              };
              for (int h = 0; h < 16; h++) {
                int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
                if (h >= 8) mix ^= cat(2) * ple_mult[2];
                uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
                ple_addrs[(size_t)t * 16 + h] = (uint64_t)(uintptr_t)(ple_table->data + row * 160);
              }
            }
            uint64_t* kb = ple_keys.data() + (size_t)t0 * 16;
            std::memcpy(kb, ple_addrs.data() + (size_t)t0 * 16, (size_t)(t1 - t0) * 16 * 8);
            std::sort(kb, kb + (size_t)(t1 - t0) * 16);
            uint64_t prev = ~0ull;
            for (int64_t i = 0; i < (int64_t)(t1 - t0) * 16; i++) {
              uint64_t pg = kb[i] & ~4095ull;
              if (pg != prev) {
                madvise((void*)(uintptr_t)pg, 4096, MADV_WILLNEED);
                prev = pg;
              }
            }
            ple_barrier.fetch_add(1, std::memory_order_acq_rel);
            while (ple_barrier.load(std::memory_order_acquire) < nt)
              std::this_thread::yield();
            for (int t = t0; t < t1; t++) {
              uint8_t* dst = ple_fp8 + (size_t)t * 2560;
              const uint64_t* ap = ple_addrs.data() + (size_t)t * 16;
              for (int h = 0; h < 16; h++)
                std::memcpy(dst + h * 160, (const void*)(uintptr_t)ap[h], 160);
            }
            if (ple_workers.fetch_sub(1) == 1)
              ple_done = std::chrono::steady_clock::now();
          });
        }
      } else {
        for (int w = 0; w < nt; w++)
          ple_pool.emplace_back([&, w] {
            for (int t = w; t < P; t += nt)
              ple_rows_fp8(ple_base0 + t, ple_fp8 + (size_t)t * 2560);
            if (ple_workers.fetch_sub(1) == 1)
              ple_done = std::chrono::steady_clock::now();
          });
      }
    }
    CK(hipMemcpyAsync(d_tokarr, tokens.data(), P * 4, hipMemcpyHostToDevice, g_str));
    // embed all tokens: gather rows then repeat to 4 branches (in-place safe)
    WView et = wview(ckpt.at("embed_tokens.weight"));
    const uint8_t* codes = et.data + 64;
    const uint8_t* scales = et.data + 64 + et.rows * et.cols / 2;
    if (gr_bf16)
      k_q4cp_row_b_hc_bf16<<<P, 128, 0, g_str>>>(
          codes, scales, (const float*)et.data, d_tokarr, et.cols,
          et.scale_stride, (uint16_t*)d_Rb);
    else {
      k_q4cp_row_b<<<P, 128, 0, g_str>>>(codes, scales, (const float*)et.data,
                                         d_tokarr, et.cols, et.scale_stride, d_Rb,
                                         10240);
      k_repeat4_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_Rb, d_Rb, g_cfg.d);
    }
    if (dbg && !gr_bf16)
      h_rms("gpu", -1, "emb", d_Rb + (size_t)(P - 1) * 10240, g_cfg.d);
    // Vision (stage 4): masked_scatter the ViT embeds over this chunk's image
    // placeholder rows (HF inputs_embeds semantics; bf16-quantized source).
    if (mrope_active_ && !img_inj.empty()) {
      h_img_inj.clear();
      for (const int2& e : img_inj)
        if (e.y >= base && e.y < base + P)
          h_img_inj.push_back(make_int2(e.x, e.y - base));
      if (!h_img_inj.empty()) {
        CK(hipMemcpyAsync(d_img_inj, h_img_inj.data(),
                          h_img_inj.size() * sizeof(int2),
                          hipMemcpyHostToDevice, g_str));
        if (gr_bf16)
          k_img_inject_bf16<<<(unsigned)h_img_inj.size(), 256, 0, g_str>>>(
              d_img_emb, d_img_inj, (uint16_t*)d_Rb);
        else
          k_img_inject_f32<<<(unsigned)h_img_inj.size(), 256, 0, g_str>>>(
              d_img_emb, d_img_inj, d_Rb);
      }
    }
    enq_layer_b(0, P, false, true, base);
    if (prog_total > 0) prog_probe(0, P);
    phase("embed+L0", TP);
    if (ple_on) {
      auto wait_start = std::chrono::steady_clock::now();
      for (auto& th : ple_pool) th.join();
      auto wait_done = std::chrono::steady_clock::now();
      if (prof) {
        p_pleh +=
            std::chrono::duration<double, std::milli>(ple_done - ple_start).count();
        p_ple_wait +=
            std::chrono::duration<double, std::milli>(wait_done - wait_start).count();
      }
      // dequant fp8 -> f32 d_eb on GPU (zero-copy read of pinned staging)
      size_t n16 = (size_t)P * 2560 / 16;
      k_ple_fp8_dequant<<<(unsigned)((n16 + 255) / 256), 256, 0, g_str>>>(
          ple_eb_dev, d_eb, ple_tscale, n16);
      float* hc_storage = d_Rb;
      if (gr_bf16) {
        uint64_t n = (uint64_t)P * 10240;
        k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
            (const uint16_t*)hc_storage, d_Rhatb, n);
        d_Rb = d_Rhatb;
      }
      if (ple_loop)
        for (int t = 0; t < P; t++) ple_gpu_t(t, base);
      else
        ple_gpu_b(P, base);
      if (gr_bf16) {
        uint64_t n = (uint64_t)P * 10240;
        k_f32_to_bf16_v4<<<(unsigned)((n / 4 + 255) / 256), 256, 0, g_str>>>(
            d_Rb, (uint16_t*)hc_storage, 10240, P, 10240);
        d_Rb = hc_storage;
      }
    }
    phase("ple", TP);
    for (int l = 1; l < g_cfg.layers; l++) {
      bool fuse_prev = gr_scatter_norm && l > 1;
      bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
      enq_layer_b(l, P, fuse_prev, flush_mlp, base);
      if (prog_total > 0) prog_probe(l, P);
      if (l == 24) phase("L01-24", TP);
      if (phase_prof && getenv("GDEC_PHASE")[0] == '2') {
        char lbuf[16];
        snprintf(lbuf, sizeof lbuf, "L%02d", l);
        phase(lbuf, TP);
      }
    }
    phase("L25-47", TP);
    // MTP tap capture: d_Rb now holds the final 48-layer stream for every row.
    if (mtp_tap_capture) {
      // Taps are stored at chunk-local rows (0..P) regardless of the absolute
      // base; prefill_batch ingests each chunk right after it completes, so
      // prompts longer than maxbatch work.
      if (P > maxbatch_cap) {
        fprintf(stderr, "mtp tap capture: unsupported (P=%d maxbatch_cap=%d)\n", P,
                maxbatch_cap);
        exit(1);
      }
      if (gr_bf16) {  // d_Rb rows are bf16; d_mtap stays fp32 for ingest
        uint64_t n = (uint64_t)P * 10240;
        k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
            (const uint16_t*)d_Rb, d_mtap, n);
      } else {
        CK(hipMemcpyAsync(d_mtap, d_Rb, (size_t)P * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
      }
    }
    // hand off last row to the single-token decode state
    if (gr_bf16)
      k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
          (const uint16_t*)d_Rb + (size_t)(P - 1) * 10240, d_R, 10240);
    else
      CK(hipMemcpyAsync(d_R, d_Rb + (size_t)(P - 1) * 10240, 10240 * 4,
                        hipMemcpyDeviceToDevice, g_str));
    pos = base + P;
    ple_ringpos = (base + P) % 9;
    CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
    // Verify consumes batch logits; do not compute and synchronously read
    // the last row's lm_head a second time.
    if (final) enq_final();
    phase("final", TP);
    int id = -1;
    if (final && prog_total > 0) prog_wait(P);
    if (final) CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    phase("argmax_d2h", TP);
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
    // quiet: spec-verify re-ingest chunks (<= gamma+1 tokens each round);
    // logging those would spam one line per spec round.
    if (!quiet)
      fprintf(stderr, "\n%s prefill: %d tokens in %.2f s = %.1f tok/s\n", log_ts(),
              P, sec, P / sec);
    if (prog_total > 0) prog_base += P;  // live progress: chunk complete
    if (prof)
      fprintf(stderr,
              "prof: gemm_host=%.0fms topk_wait=%.0fms bucket+h2d=%.0fms "
              "ple_host=%.0fms ple_wait=%.0fms\n",
              p_gemm, p_topk_wait, p_bucket, p_pleh, p_ple_wait);
#ifdef GDEC_PHASE_PROF
    if (getenv("GDEC_PHASE_PROF")) {
      CK(hipStreamSynchronize(g_str));
      unsigned long long h[16];
      CK(hipMemcpyFromSymbol(h, g_ph, sizeof(h)));
      hipDeviceProp_t dp;
      CK(hipGetDeviceProperties(&dp, 0));
      double cyc2ms = 1.0 / (dp.clockRate * 1e3);  // clockRate is kHz
      const char* names[11] = {"in_staging", "in_dots",   "in_solve", "in_vpkcd",
                               "in_attn2",   "st_Sstage", "st_vnew",  "st_out",
                               "st_Supd",    "in_TOTAL",  "st_TOTAL"};
      for (int s = 0; s < 11; ++s)
        fprintf(stderr, "ph %-10s %12llu cyc  %9.1f CU-ms\n", names[s], h[s],
                h[s] * cyc2ms);
    }
#endif
    return id;
  }

  // =================== MTP draft head (Phase A) ==============================
  // Wiring: MTP stream position p takes (trunk h_p, emb(t_{p+1})) and predicts
  // t_{p+2}. Input: h = sum_b fc_hidden(norm_h(tap)[b]), e =
  // fc_embedding(norm_e(embed(tok))), stream = repeat4(h + e). Norm mode is
  // GDEC_MTP_NORM=full (one 10240 RMSNorm, default) or =branch (4x2560).
  void mtp_input_1(const float* tap) {
    WView et = wview(ckpt.at("embed_tokens.weight"));
    k_q4cp_row<<<1, 128, 0, g_str>>>(et.data + 64,
                                     et.data + 64 + et.rows * et.cols / 2,
                                     (const float*)et.data, d_mtok, et.cols,
                                     et.scale_stride, d_emb);
    k_rmsnorm_zc_grouped<<<1, 1024, 0, g_str>>>(d_emb, d_mnorm_e, d_me, 2560,
                                                g_cfg.norm_eps, 1);
    gemv(wview(ckpt.at("mtp.fc_embedding.weight")), d_me, d_meo);
    if (mtp_norm_mode == 0)
      k_rmsnorm_zc_grouped<<<1, 1024, 0, g_str>>>(tap, d_mnorm_h, d_mRhat, 10240,
                                                  g_cfg.norm_eps, 1);
    else
      k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(tap, d_mnorm_h, d_mRhat, 2560,
                                                  g_cfg.norm_eps, 4);
    if (mtp_hagg_mode == 0) {
      // H-agg=sum: x = sum_b fc_hidden(norm_h[b]) + e, then repeat4.
      gemv(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat, d_mh);
      for (int b = 1; b < 4; b++) {
        gemv(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat + (size_t)b * 2560,
             d_mt);
        k_acc<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_mt, 2560);
      }
      k_acc<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_meo, 2560);
      k_repeat4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_mR, g_cfg.d);
    } else {
      // H-agg=branch: stream[b] = fc_hidden(norm_h[b]) + e (branch identity kept).
      for (int b = 0; b < 4; b++)
        gemv(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat + (size_t)b * 2560,
             d_mR + (size_t)b * 2560);
      k_acc_broadcast4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_meo, d_mR, 2560);
    }
  }

  // GR read/write on the MTP stream (mirrors gr_read/gr_write with d_mR).
  void mtp_gr_read(const std::string& prefix, float* x_out) {
    char buf[128];
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_mR, hw, d_mRhat, 2560,
                                                g_cfg.norm_eps, 4);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_mRhat, d_t320);
    k_silu_scale<<<(320 + 255) / 256, 256, 0, g_str>>>(d_t320, 0.25f, 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_t320, d_G);
    k_sigmoid_inplace<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_G, 10240);
    k_gr_combine<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_G, d_mRhat, x_out,
                                                        g_cfg.d, g_cfg.branches);
  }
  void mtp_gr_write(const std::string& prefix, const float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_mRhat, d_w4);
    k_gr_write<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_w4, d_mR, y, g_cfg.d,
                                                      g_cfg.branches);
  }

  template <typename KVT>
  void mtp_qsa_attn(const KVT* kc, const KVT* vc) {
    if (qsa_dense) {
      k_qsa_step<KVT><<<24, 256, 0, g_str>>>(d_qs, kc, vc, d_attn, d_mpos, 256, 2,
                                             24);
      return;
    }
    k_qsa_flash<true, false, KVT, true><<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
        d_qs, kc, vc, d_attn, 1, 256, 2, 24, d_iselected, 0, d_mpos, d_pacc,
        d_pml);
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, d_attn, 12,
                                                       256, d_mpos);
  }

  // One QSA attention step for the MTP layer (mirrors qsa()).
  void mtp_qsa(const float* x, float* y) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, 1,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               d_mpos, 0, nullptr, d_rdelta);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_minorm + 128, d_miraw,
                                           d_mik, d_mpos, g_cfg.norm_eps,
                                           g_cfg.rope_theta, d_rdelta);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(
          d_iq, d_mik, d_iscores, index_blocks, d_mpos);
      index_select(d_iscores, index_blocks, d_iselected, 0, 1, d_mpos);
    }
    snprintf(buf, sizeof buf, "%s.q_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_qg);
    snprintf(buf, sizeof buf, "%s.k_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_kb);
    snprintf(buf, sizeof buf, "%s.v_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_vb);
    snprintf(buf, sizeof buf, "%s.q_norm.weight", pfx);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "%s.k_norm.weight", pfx);
    float* knw = ensure_norm256(buf);
    k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(d_qg, qnw, d_qs, d_gs, 256,
                                                 g_cfg.norm_eps);
    k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(d_kb, knw, d_kb, 256,
                                               g_cfg.norm_eps, 1);
    k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(d_qs, 24, 256, 64, d_mpos,
                                                     g_cfg.rope_theta,
                                                     d_rdelta);
    k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(d_kb, 2, 256, 64, d_mpos,
                                                    g_cfg.rope_theta,
                                                    d_rdelta);
    if (qsa_kv_fp32) {
      k_store_kv<<<2, 256, 0, g_str>>>(d_kb, d_vb, d_mkcf, d_mvcf, d_mpos);
      mtp_qsa_attn(d_mkcf, d_mvcf);
    } else {
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(d_kb, d_vb, d_mkcb, d_mvcb, d_mpos);
      mtp_qsa_attn(d_mkcb, d_mvcb);
    }
    k_sigmoid_gate<<<(24 * 256 + 255) / 256, 256, 0, g_str>>>(d_attn, d_gs,
                                                              d_attn, 24 * 256);
    snprintf(buf, sizeof buf, "%s.o_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), d_attn, y);
  }

  // MTP MoE step (mirrors moe()).
  void mtp_moe(const float* x, float* y) {
    const char* pfx = "mtp.layers.0.mlp";
    char buf[128];
    snprintf(buf, sizeof buf, "%s.gate.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_router);
    int k = g_cfg.topk;
    k_router_topk<<<1, 512, 0, g_str>>>(d_router, d_topids, d_topw, g_cfg.experts,
                                        k);
    CK(hipMemsetAsync(d_moeacc, 0, g_cfg.d * 4, g_str));
    snprintf(buf, sizeof buf, "%s.experts.gate_up_proj.weight", pfx);
    WView gu = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "%s.experts.down_proj.weight", pfx);
    WView dn = wview(ckpt.at(buf));
    const int* topids = d_topids;  // 直读:ids 即专家号,免 D2H 同步
    uint64_t pairs = (uint64_t)k * g_cfg.moe_mid;
    k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
        gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
        x, d_guv, topids, (uint64_t)2 * g_cfg.moe_mid, gu.cols, gu.scale_stride,
        k, 0 /*x_stride*/, 16 /*id_stride*/, k);
    k_silu_mul_g<<<(k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(
        d_guv, d_hid, g_cfg.moe_mid, k);
    uint64_t pairs2 = (uint64_t)k * ((g_cfg.d + 1) / 2);
    k_q4cp_gemv_gd<<<(unsigned)((pairs2 + 15) / 16), 512, 0, g_str>>>(
        dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
        d_hid, d_moeacc, topids, d_topw, (uint64_t)g_cfg.d, dn.cols,
        dn.scale_stride, (uint64_t)g_cfg.moe_mid, k, 16 /*id_stride*/,
        0 /*acc_stride*/, k);
    snprintf(buf, sizeof buf, "%s.shared_expert.gate_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_guv);
    snprintf(buf, sizeof buf, "%s.shared_expert.up_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_guv + 640);
    k_silu_mul<<<(640 + 255) / 256, 256, 0, g_str>>>(d_guv, d_guv + 640, d_hid,
                                                     640);
    snprintf(buf, sizeof buf, "%s.shared_expert.down_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), d_hid, d_ey);
    snprintf(buf, sizeof buf, "%s.shared_expert_gate.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_sg);
    k_axpy_sg<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_moeacc, d_ey, d_sg,
                                                     g_cfg.d, g_cfg.d);
    CK(hipMemcpyAsync(y, d_moeacc, g_cfg.d * 4, hipMemcpyDeviceToDevice, g_str));
  }

  // One full MTP draft step: appends (tap, tok) at mtp_pos. Greedy callers get
  // the argmax; sampling callers request the full logits instead.
  int mtp_step(const float* tap, int tok,
               std::vector<float>* host_logits = nullptr) {
    CK(hipMemcpyAsync(d_mtok, &tok, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
    mtp_input_1(tap);
    mtp_gr_read("mtp.layers.0.attn_hyper_connection", d_x);
    mtp_qsa(d_x, d_y);
    mtp_gr_write("mtp.layers.0.attn_hyper_connection", d_y);
    mtp_gr_read("mtp.layers.0.mlp_hyper_connection", d_x);
    mtp_moe(d_x, d_y);
    mtp_gr_write("mtp.layers.0.mlp_hyper_connection", d_y);
    mtp_gr_read("mtp.hyper_connection_mixer", d_x);
    gemv(wview(ckpt.at("lm_head.weight")), d_x, d_logits);
    int id = -1;
    if (host_logits) {
      host_logits->resize(g_cfg.vocab);
      CK(hipMemcpy(host_logits->data(), d_logits, (size_t)g_cfg.vocab * 4,
                   hipMemcpyDeviceToHost));
    } else {
      k_argmax<<<1, 1024, 0, g_str>>>(d_logits, g_cfg.vocab, d_argmax);
      CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    }
    mtp_pos++;
    return id;
  }

  // ---- batch ingest: build MTP KV over a token range (prompt/accepted) ----
  // Row j takes (tap[j] = trunk h_{base+j}, toks[j] = t_{base+j+1}) at MTP
  // position base+j. No mixer/lm_head: only the layer state (KV) is built.
  void mtp_input_b(const float* taps, int P) {
    WView et = wview(ckpt.at("embed_tokens.weight"));
    k_q4cp_row_b<<<P, 128, 0, g_str>>>(et.data + 64,
                                       et.data + 64 + et.rows * et.cols / 2,
                                       (const float*)et.data, d_tokarr, et.cols,
                                       et.scale_stride, d_eb, 2560);
    k_rmsnorm_zc_grouped<<<P, 1024, 0, g_str>>>(d_eb, d_mnorm_e, d_eb, 2560,
                                                g_cfg.norm_eps, 1);
    gemm(wview(ckpt.at("mtp.fc_embedding.weight")), d_eb, d_qnb, P);
    if (mtp_norm_mode == 0)
      k_rmsnorm_zc_grouped<<<P, 1024, 0, g_str>>>(taps, d_mnorm_h, d_mRhatb,
                                                  10240, g_cfg.norm_eps, 1);
    else
      k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(taps, d_mnorm_h, d_mRhatb,
                                                      2560, g_cfg.norm_eps, 4);
    if (mtp_hagg_mode == 0) {
      gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhatb, d_Unb, P,
           10240 /*xstride*/);
      for (int b = 1; b < 4; b++) {
        gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhatb + (size_t)b * 2560,
             d_convoutb, P, 10240 /*xstride*/);
        k_acc<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
            d_Unb, d_convoutb, P * 2560);
      }
      k_acc<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
          d_Unb, d_qnb, P * 2560);
      k_row_head_copy<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0,
                        g_str>>>(d_Unb, d_mRb, 2560);
      k_repeat4_b<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
          d_mRb, d_mRb, g_cfg.d);
    } else {
      for (int b = 0; b < 4; b++) {
        gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhatb + (size_t)b * 2560,
             d_convoutb, P, 10240 /*xstride*/);
        k_row_branch_copy<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0,
                            g_str>>>(d_convoutb, d_mRb, 2560, b);
      }
      k_acc_broadcast4<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0,
                         g_str>>>(d_qnb, d_mRb, 2560);
    }
  }

  void mtp_gr_read_b(const std::string& prefix, int P) {
    char buf[128];
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_mRb, hw, d_mRhatb, 2560,
                                                    g_cfg.norm_eps, 4);
    // K=10240 gemms would overflow d_xbf16b (maxbatch*6144); convert into the
    // 10240-wide d_xbf16 ourselves and pass it as xbf. (d_xbf16 is clobbered
    // by later d_xb gemms, so gr_write_b reconverts.)
    uint64_t tot = (uint64_t)P * 10240;
    k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
        d_mRhatb, d_xbf16, 10240, P, 10240);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_mRhatb, d_t320b, P, 0, d_xbf16);
    k_silu_scale<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(d_t320b, 0.25f,
                                                           P * 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_t320b, d_Gb, P);
    k_sigmoid_inplace<<<(unsigned)(((size_t)P * 10240 + 255) / 256), 256, 0,
                        g_str>>>(d_Gb, P * 10240);
    k_gr_combine_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_Gb, d_mRhatb, d_xb, g_cfg.d, g_cfg.branches, P);
    if (xf16_src == d_xb) xf16_src = nullptr;
  }
  void mtp_gr_write_b(const std::string& prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    uint64_t tot = (uint64_t)P * 10240;
    k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
        d_mRhatb, d_xbf16, 10240, P, 10240);
    gemm(wview(ckpt.at(buf)), d_mRhatb, d_w4b, P, 0, d_xbf16);
    k_gr_write_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_w4b, d_mRb, d_yb,
                                                            g_cfg.d,
                                                            g_cfg.branches, P);
  }

  // Batch QSA for the MTP layer (mirrors qsa_b flash path).
  void mtp_qsa_b(int P, int base) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    // Table-driven rope (M-RoPE requests carry their 3-row positions in the
    // table; text gets the same absolute-position table, bit-identical to
    // the inline fp64 path it replaces).
    ensure_rope_tab(g_cfg.rope_theta, 64);
    ensure_rope_cs(base, P);
    if (!qsa_dense) {
      if (index_f32) {
        float one = 1, zero = 0;
        auto status =
            rocblas_sgemm(rbh, rocblas_operation_transpose,
                          rocblas_operation_none, 640, P, 2560, &one, d_miw,
                          2560, d_xb, 2560, &zero, d_iproj, 640);
        if (status != rocblas_status_success) {
          fprintf(stderr, "mtp indexer fp32 projection failed: %d\n", (int)status);
          exit(1);
        }
      } else {
        snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
        gemm(wview(ckpt.at(buf)), d_xb, d_iproj, P);
      }
      k_index_q<<<dim3(P, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, P,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               nullptr, base, d_ropecs);
      int nb = (base + P) / 4, b0 = base / 4;
      if (nb > b0)
        k_index_pool<<<nb - b0, 128, 0, g_str>>>(d_iproj, d_minorm + 128, d_mik,
                                                 g_cfg.norm_eps,
                                                 g_cfg.rope_theta, base, d_miraw,
                                                 d_ropecs);
      k_index_ring<<<min(P, 4), 128, 0, g_str>>>(d_iproj, d_miraw, P, base);
      const float one = 1.f, zero = 0.f;
      const int a0 = std::max(2051, base);
      const int grid =
          2051 + (a0 - 2051 + INDEX_BATCH - 1) / INDEX_BATCH * INDEX_BATCH;
      for (int first = a0; first < base + P;) {
        int nxt = first < grid ? min(grid, base + P)
                               : min(first + INDEX_BATCH, base + P);
        int count = nxt - first;
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose,
                                    rocblas_operation_none, nb, count * 4, 128,
                                    &one, d_mik, 128,
                                    d_iq + (size_t)(first - base) * 512, 128,
                                    &zero, d_iscores, nb);
        if (status != rocblas_status_success) {
          fprintf(stderr, "mtp indexer sgemm failed: %d\n", (int)status);
          exit(1);
        }
        index_select(d_iscores, nb, d_iselected + (size_t)first * 512, first,
                     count);
        first = nxt;
      }
    }
    snprintf(buf, sizeof buf, "%s.q_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_qgb, P);
    snprintf(buf, sizeof buf, "%s.k_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_kbb, P);
    snprintf(buf, sizeof buf, "%s.v_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_vbb, P);
    snprintf(buf, sizeof buf, "%s.q_norm.weight", pfx);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "%s.k_norm.weight", pfx);
    float* knw = ensure_norm256(buf);
    k_qsa_qsplit<true><<<dim3(24, P), 256, 0, g_str>>>(d_qgb, qnw, d_qsb, d_gsb,
                                                       256, g_cfg.norm_eps, base,
                                                       d_ropecs);
    k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(d_kbb, knw, d_kbb,
                                                      nullptr, g_cfg.norm_eps,
                                                      base, d_ropecs);
    int dense_P = qsa_dense ? P : min(P, std::max(0, 2051 - base));
    if (qsa_kv_fp32) {
      CK(hipMemcpyAsync(d_mkcf + (size_t)base * 512, d_kbb, (size_t)P * 512 * 4,
                        hipMemcpyDeviceToDevice, g_str));
      CK(hipMemcpyAsync(d_mvcf + (size_t)base * 512, d_vbb, (size_t)P * 512 * 4,
                        hipMemcpyDeviceToDevice, g_str));
      qsa_flash_b(d_mkcf, d_mvcf, P, dense_P, base);
    } else {
      k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256, 0,
                         g_str>>>(d_kbb, d_mkcb + (size_t)base * 512, 512, P,
                                  512);
      k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256, 0,
                         g_str>>>(d_vbb, d_mvcb + (size_t)base * 512, 512, P,
                                  512);
      qsa_flash_b(d_mkcb, d_mvcb, P, dense_P, base);
    }
    k_sigmoid_gate<<<(P * 6144 + 255) / 256, 256, 0, g_str>>>(d_attnb, d_gsb,
                                                              d_attnb, P * 6144);
    snprintf(buf, sizeof buf, "%s.o_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_attnb, d_yb, P);
  }

  // Batch MoE for the MTP layer: default deterministic packed path only
  // (mtp_init refuses non-default MoE flags).
  void mtp_moe_b(int P) {
    const char* pfx = "mtp.layers.0.mlp";
    char buf[128];
    snprintf(buf, sizeof buf, "%s.gate.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_routerb, P);
    int k = g_cfg.topk, E = g_cfg.experts, npairs = P * k;
    k_router_topk<<<P, 512, 0, g_str>>>(d_routerb, d_topidsb, d_topwb,
                                        g_cfg.experts, k);
    CK(hipMemsetAsync(d_moeaccb, 0, (size_t)P * g_cfg.d * 4, g_str));
    snprintf(buf, sizeof buf, "%s.experts.gate_up_proj.weight", pfx);
    const Tensor& gu_t = ckpt.at(buf);
    WView gu = wview(gu_t);
    snprintf(buf, sizeof buf, "%s.experts.down_proj.weight", pfx);
    const Tensor& dn_t = ckpt.at(buf);
    WView dn = wview(dn_t);
    moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff, d_pairids,
                    P, k, E);
    int tasks = (npairs + 63) / 64 + E;
    k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E);
    k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * g_cfg.d / 4 + 255) / 256), 256, 0,
                       g_str>>>(d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
    k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0,
                               g_str>>>(
        gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2,
        (const float*)gu.data, d_xb, d_tokidx, d_eoff, nullptr,
        2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid, gu.scale_stride, d_moetiles,
        d_moentiles, d_xbf16, (uint16_t*)d_hidb);
    k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0,
                                        g_str>>>(
        dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
        (const float*)dn.data, nullptr, d_tokidx, d_pweight, d_eoff, d_guvb,
        g_cfg.d, dn.cols, dn.scale_stride, d_moetiles, d_moentiles,
        (const uint16_t*)d_hidb);
    k_moe_reduce_fast<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
        d_guvb, d_pairids, d_moeaccb, P, g_cfg.d);
    snprintf(buf, sizeof buf, "%s.shared_expert.gate_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_guvb, P);
    snprintf(buf, sizeof buf, "%s.shared_expert.up_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_guvb + (size_t)P * 640, P);
    k_silu_mul<<<(P * 640 + 255) / 256, 256, 0, g_str>>>(
        d_guvb, d_guvb + (size_t)P * 640, d_hidb, P * 640);
    snprintf(buf, sizeof buf, "%s.shared_expert.down_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_hidb, d_eyb, P);
    snprintf(buf, sizeof buf, "%s.shared_expert_gate.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_sgb, P);
    k_axpy_sg<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_moeaccb, d_eyb, d_sgb,
                                                         g_cfg.d, P * g_cfg.d);
    CK(hipMemcpyAsync(d_yb, d_moeaccb, (size_t)P * g_cfg.d * 4,
                      hipMemcpyDeviceToDevice, g_str));
  }

  void mtp_ingest_b(const float* taps, const std::vector<int>& toks, int P,
                    int base) {
    if (P < 1) return;
    CK(hipMemcpyAsync(d_tokarr, toks.data(), P * 4, hipMemcpyHostToDevice,
                      g_str));
    mtp_input_b(taps, P);
    xf16_src = nullptr;
    mtp_gr_read_b("mtp.layers.0.attn_hyper_connection", P);
    mtp_qsa_b(P, base);
    mtp_gr_write_b("mtp.layers.0.attn_hyper_connection", P);
    mtp_gr_read_b("mtp.layers.0.mlp_hyper_connection", P);
    mtp_moe_b(P);
    mtp_gr_write_b("mtp.layers.0.mlp_hyper_connection", P);
    mtp_pos = base + P;
    CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
  }

  // Acceptance-rate harness: after prefill_batch(prompt) with tap capture,
  // ingest the prompt into the MTP layer, then walk K trunk steps; at each,
  // the draft (from the frozen pre-step d_R) must match the trunk's next
  // argmax. Prints top-1 acceptance.
  void mtp_accept_test(const std::vector<int>& prompt, int first_id, int K) {
    int P = (int)prompt.size();
    if (P > maxbatch) {
      fprintf(stderr, "mtp-test: prompt %d > maxbatch %d\n", P, maxbatch);
      exit(1);
    }
    std::vector<int> shifted(prompt.begin() + 1, prompt.end());
    mtp_ingest_b(d_mtap, shifted, P - 1, 0);
    int cur = first_id, hits = 0, n = 0;
    auto T0 = std::chrono::steady_clock::now();
    for (int g = 0; g < K; g++) {
      int draft = mtp_step(d_R, cur);  // predicts the token after next
      int nxt = forward(cur);          // trunk commits cur
      hits += (draft == nxt);
      n++;
      cur = nxt;
      if (g % 32 == 31 || g == K - 1) {
        double sec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - T0)
                         .count();
        fprintf(stderr, "\rmtp-test %d/%d acc=%.4f (%.1f steps/s)", n, K,
                (double)hits / n, n / sec);
        fflush(stderr);
      }
    }
    fprintf(stderr, "\nmtp-test: top-1 acceptance %d/%d = %.4f (norm=%s hagg=%s)\n",
            hits, n, (double)hits / n, mtp_norm_mode ? "branch" : "full",
            mtp_hagg_mode ? "branch" : "sum");
  }

  // =================== MTP speculative loop (Phase B) ========================
  // gr_bf16 shims: d_Rb rows are bf16 under GDEC_GR_BF16, but the MTP ingest
  // path and the decode state d_R are fp32. spec_taps stages rows 0..P-1 as
  // fp32 in d_Rhatb (scratch outside prefill; consumed by mtp_input_b before
  // anything else touches it).
  const float* spec_taps(int P) {
    if (!gr_bf16) return d_Rb;
    uint64_t n = (uint64_t)P * 10240;
    k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
        (const uint16_t*)d_Rb, d_Rhatb, n);
    return d_Rhatb;
  }
  void restore_dR_row(int d) {
    if (gr_bf16)
      k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
          (const uint16_t*)d_Rb + (size_t)d * 10240, d_R, 10240);
    else
      CK(hipMemcpyAsync(d_R, d_Rb + (size_t)d * 10240, 10240 * 4,
                        hipMemcpyDeviceToDevice, g_str));
  }
  // Spec checkpoints: only the position-mod rings (PLE ring + QSA indexer raw
  // ring) are needed — GDN S / convst are restored from per-row snapshots.
  void mtp_ckpt_save() {
    float* p = d_ckpt + (ckpt_S + ckpt_convst) / 4;
    CK(hipMemcpyAsync(p, d_ring, ckpt_ring, hipMemcpyDeviceToDevice, g_str));
    p += ckpt_ring / 4;
    if (ckpt_iraw)
      CK(hipMemcpyAsync(p, d_iraw, ckpt_iraw, hipMemcpyDeviceToDevice, g_str));
  }

  // Produce logits for every batch row's final stream. Rows remain live in
  // d_Rb after prefill_batch, and the multi-row lm_head streams weights once.
  void batch_logits_device(int P) {
    gr_read_b("hyper_connection_mixer", P);
    const WView& lw = wview(ckpt.at("lm_head.weight"));
    // The multi-row kernel is specialized for up to eight rows. Chunk larger
    // ngram proposals into groups of eight so lm_head weights are streamed
    // once per group instead of once per row.
    for (int off = 0; off < P; ) {
      const int q = std::min(8, P - off);
      const bool ok = gemv_multi_q4cp(
          lw, d_xb + (size_t)off * 2560,
          d_logits_b + (size_t)off * g_cfg.vocab, q, 2560);
      if (!ok) {
        for (int r = 0; r < q; r++) {
          gemv(lw, d_xb + (size_t)(off + r) * 2560,
               d_logits_b + (size_t)(off + r) * g_cfg.vocab);
        }
      }
      off += q;
    }
  }

  void batch_logits(int P, std::vector<float>& host) {
    batch_logits_device(P);
    host.resize((size_t)P * g_cfg.vocab);
    CK(hipMemcpy(host.data(), d_logits_b, host.size() * sizeof(float),
                 hipMemcpyDeviceToHost));
  }

  // Greedy reduction of the same batched logits.
  void batch_argmax(int P, int* host) {
    batch_logits_device(P);
    for (int r = 0; r < P; r++)
      k_argmax<<<1, 1024, 0, g_str>>>(d_logits_b + (size_t)r * g_cfg.vocab,
                                      g_cfg.vocab, d_argmax_arr + r);
    CK(hipMemcpy(host, d_argmax_arr, P * 4, hipMemcpyDeviceToHost));
  }

  std::vector<int> spec_loop_ngram(int cur, int gen, int gamma,
      const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, gamma), maxbatch - 1});
    int accepted = 0, fallback = 0;
    double verify_ms = 0, replay_ms = 0;
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto step = [&](int tok) {
      int nx = forward(tok);
      if (!ple_on) chist.push_back(tok);
      ple_ringpos = pos % 9;  // graph advances the device scalar, not the host mirror
      return nx;
    };
    auto verify = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    while ((int)out.size() < gen && pos < maxctx) {
      if (chist.size() != (size_t)pos)
        throw std::runtime_error("ngram: token history and trunk position disagree");
      ngram.observe(chist);
      // Leave one slot for the target-model continuation. Never verify past
      // the generation budget or context boundary, including the final round.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      std::vector<int> draft = want > 0 ? ngram.draft(chist, cur, want) : std::vector<int>{};
      if (draft.empty()) {
        int next = step(cur);
        ++fallback;
        if (emit && !emit(next)) break;
        out.push_back(next);
        cur = next;
        continue;
      }
      const int base = pos;
      const size_t history_size = chist.size();
      ngram_ckpt_save();
      std::vector<int> tokens{cur};
      tokens.insert(tokens.end(), draft.begin(), draft.end());
      auto t = std::chrono::steady_clock::now();
      verify(tokens, base);
      std::vector<int> targets(tokens.size());
      batch_argmax((int)tokens.size(), targets.data());
      verify_ms += elapsed(t);
      int a = 0;
      while (a < (int)draft.size() && draft[a] == targets[a]) ++a;
      int keep = a;
      bool stop = false;
      for (int j = 0; j < a; ++j) {
        if (emit && !emit(draft[j])) { keep = j; stop = true; break; }
        out.push_back(draft[j]);
      }
      if (!stop) {
        if (emit && !emit(targets[a])) stop = true;
        else out.push_back(targets[a]);
      }
      if (keep != (int)draft.size()) {
        // Restore every overwritten position-mod state, then replay exactly
        // [cur, accepted drafts]. Causal KV/indexer entries are overwritten
        // before they are visible. Replaying a batch preserves verify math.
        t = std::chrono::steady_clock::now();
        ngram_ckpt_restore();
        chist.resize(history_size);
        pos = base;
        ple_ringpos = base % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
        tokens.resize(keep + 1);
        verify(tokens, base);
        CK(hipStreamSynchronize(g_str));
        replay_ms += elapsed(t);
        ++last_spec_rollbacks;
      }
      cur = targets[keep];
      accepted += keep;
      ++last_spec_rounds;
      last_spec_proposed += draft.size();
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = accepted + last_spec_rounds;
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    last_spec_commit = (int)out.size();
    fprintf(stderr, "%s ngram: tokens=%d rounds=%d accepted=%d proposed=%d "
                    "fallback=%d rollbacks=%d verify=%.1fms replay=%.1fms\n",
            log_ts(), last_spec_commit, last_spec_rounds, accepted,
            last_spec_proposed, fallback, last_spec_rollbacks, verify_ms,
            replay_ms);
    return out;
  }

  void ngram_ckpt_save() {
    CK(hipMemcpyAsync(d_ckpt,d_S,ckpt_S,hipMemcpyDeviceToDevice,g_str));
    CK(hipMemcpyAsync(d_ckpt+ckpt_S/4,d_convst,ckpt_convst,hipMemcpyDeviceToDevice,g_str));
    float* p=d_ckpt+(ckpt_S+ckpt_convst)/4; CK(hipMemcpyAsync(p,d_ring,ckpt_ring,hipMemcpyDeviceToDevice,g_str)); p+=ckpt_ring/4;
    if(ckpt_iraw) CK(hipMemcpyAsync(p,d_iraw,ckpt_iraw,hipMemcpyDeviceToDevice,g_str));
  }
  void ngram_ckpt_restore() {
    CK(hipMemcpyAsync(d_S,d_ckpt,ckpt_S,hipMemcpyDeviceToDevice,g_str));
    CK(hipMemcpyAsync(d_convst,d_ckpt+ckpt_S/4,ckpt_convst,hipMemcpyDeviceToDevice,g_str));
    const float* p=d_ckpt+(ckpt_S+ckpt_convst)/4; CK(hipMemcpyAsync(d_ring,p,ckpt_ring,hipMemcpyDeviceToDevice,g_str)); p+=ckpt_ring/4;
    if(ckpt_iraw) CK(hipMemcpyAsync(d_iraw,p,ckpt_iraw,hipMemcpyDeviceToDevice,g_str));
  }

  // Speculative decode driver. cur = emitted token for position pos (trunk
  // state processed through pos-1, MTP KV covers 0..pos-2). Greedy accept:
  // draft j committed iff it equals the trunk argmax at its verify row; the
  // correction (or full-accept continuation) comes from the trunk.
  // emit (optional): called with each committed token as it lands; returning
  // false aborts the loop early (serve: eos/cancel/streaming).
  std::vector<int> spec_loop(int cur, int gen, int gamma,
                             const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(gen + 1);
    std::vector<int> drafts(gamma), v(gamma + 1);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int rounds = 0, rollbacks = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, t_ingest = 0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen) {
      int q = pos - 1;  // trunk processed through q
      int g = std::min(gamma, gen - (int)out.size());
      size_t chist_sz = chist.size();
      mtp_ckpt_save();
      auto t0 = std::chrono::steady_clock::now();
      drafts[0] = mtp_step(d_R, cur);
      // draft 1 used the committed (h_q, cur): its indexer ring rows are
      // truth. Later drafts pollute the ring; save/restore around them.
      CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                        g_str));
      for (int j = 1; j < g; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
      auto t1 = std::chrono::steady_clock::now();
      std::vector<int> vt;
      vt.reserve(g + 1);
      vt.push_back(cur);
      for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
      spec_snap = true;  // capture per-row rollback snapshots during verify
      prefill_batch(vt, q + 1, true, true);
      spec_snap = false;
      batch_argmax(g + 1, v.data());
      auto t2 = std::chrono::steady_clock::now();
      int a = 0;
      while (a < g && drafts[a] == v[a]) a++;
      for (int j = 0; j < g; j++) {
        depth_tot[j]++;
        if (j < a) depth_hits[j]++;
      }
      rounds++;
      live_spec_rounds = rounds;
      live_spec_proposed += g;
      live_spec_commits = (int)out.size();  // pre-emit; one round of lag is fine
      // emit() streams each committed token as it lands; returning false
      // stops the round and leaves THIS token out of the trunk state (eos,
      // or cancel after it was streamed — the next turn re-ingests it via
      // ordinary KV-reuse continuation). d = commit depth: keep
      // drafts[0..d-1]; keep_v = also keep the correction/continuation v[a].
      int d = a;
      bool stop = false, keep_v = true;
      if (emit) {
        for (int j = 0; j < a; j++)
          if (!emit(drafts[j])) { d = j; stop = true; keep_v = false; break; }
        if (!stop && !emit(v[a])) { stop = true; keep_v = false; }
      }
      for (int j = 0; j < d; j++) out.push_back(drafts[j]);
      if (keep_v) out.push_back(v[a]);
      auto t3 = std::chrono::steady_clock::now();
      if (!stop && d == g) {
        cur = v[g];  // full accept; state already at q+1+g
      } else {
        rollbacks++;
        // Install the state at commit depth d from the verify snapshots —
        // no re-prefill. KV pool entries and pooled indexer keys for the
        // accepted positions were written causally by the verify pass (and
        // rejected positions are rewritten before they can be read), so only
        // the position-mod ring slots polluted by rejected rows are restored
        // from the checkpoint. d_Rb rows 0..d hold the accepted tokens'
        // streams for the MTP ingest below.
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        const size_t sL = 48 * 128 * 128;
        for (int gi = 0; gi < ngdn; gi++)
          CK(hipMemcpyAsync(d_S + (size_t)gi * sL,
                            d_Ssnap + ((size_t)d * ngdn + gi) * sL, sL * 4,
                            hipMemcpyDeviceToDevice, g_str));
        CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)d * ngdn * 3 * 10240,
                          (size_t)ngdn * 3 * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (ckpt_iraw)
          k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
              d_iproj_snap, d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
              d_iraw, q + 1, q + d + 1, 9);
        const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
        for (int p = q + d + 2; p <= q + g + 1; p++)
          CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                            ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        chist.resize(chist_sz + d + 1);
        pos = q + d + 2;
        ple_ringpos = pos % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                          g_str));
        restore_dR_row(d);
        cur = v[d];
      }
      auto t4 = std::chrono::steady_clock::now();
      // Rebuild MTP KV over committed positions q+1..q+d with trunk taps
      // (d_Rb rows of the state-advancing pass) and committed tokens.
      CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4, hipMemcpyDeviceToDevice,
                        g_str));
      if (d > 0) {
        std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
        mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
      } else {
        mtp_pos = q + 1;
        CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
      }
      if (stop) break;
      auto t5 = std::chrono::steady_clock::now();
      t_draft += ms(t0, t1);
      t_verify += ms(t1, t2);
      t_roll += ms(t3, t4);
      t_ingest += ms(t4, t5);
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    fprintf(stderr,
            "\nspec: %d tokens in %.2f s = %.1f tok/s | rounds=%d commit/round=%.2f "
            "rollbacks=%d\n",
            (int)out.size(), sec, out.size() / sec, rounds,
            (double)out.size() / rounds, rollbacks);
    fprintf(stderr, "spec depth acc:");
    for (int j = 0; j < gamma && depth_tot[j]; j++)
      fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
    fprintf(stderr,
            "\nspec time: draft=%.0fms verify=%.0fms rollback+reprefill=%.0fms "
            "ingest=%.0fms\n",
            t_draft, t_verify, t_roll, t_ingest);
    last_spec_rounds = rounds;
    last_spec_commit = (int)out.size();
    last_spec_proposed = live_spec_proposed;
    last_spec_rollbacks = rollbacks;
    return out;
  }

  // Chain speculative driver (llama.cpp common/speculative.cpp style): every
  // round asks the ngram table first; an empty draft hands the round to MTP.
  // Both drafters stay live — ngram observes every round, and ngram rounds
  // patch the MTP KV over the committed positions — so either can draft the
  // next round. Round-start invariant (same as spec_loop): mtp_pos = pos-1,
  // MTP KV covers 0..pos-2, d_R = h_{pos-1} pending pairing with cur.
  std::vector<int> spec_loop_chain(int cur, int gen, int gamma,
      const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks =
        last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, ngram.n_max), maxbatch - 1});
    gamma = std::max(1, std::min(8, gamma));
    std::vector<int> drafts(8), v(9);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int accepted = 0, ng_rounds = 0, mtp_rounds = 0, ng_roll = 0, mtp_roll = 0;
    double verify_ms = 0, replay_ms = 0;
    // ngram verify chunking: GDEC_NGRAM_CHUNK=0 keeps the legacy single-pass
    // 65-row verify; N pins the chunk size; unset adapts within [8,64]
    // (init 16), doubling after fully-accepted rounds, halving after rejects.
    // The level is a Model member shared with the sampling chain and kept
    // across requests (workload property, not per-request).
    ng_chunk_init();
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto verify_ng = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen && pos < maxctx) {
      if (chist.size() != (size_t)pos)
        throw std::runtime_error(
            "chain: token history and trunk position disagree");
      ngram.observe(chist);
      // Same budget math as the pure-ngram loop: one slot stays for the
      // target-model continuation, never verify past gen or maxctx.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      std::vector<int> draft =
          want > 0 ? ngram.draft(chist, cur, want) : std::vector<int>{};
      bool stop = false;
      if (!draft.empty() && ng_legacy) {
        // ---- ngram round: long draft, whole-state checkpoint + replay ----
        const int base = pos;
        const size_t history_size = chist.size();
        // Stash h_{base-1} (d_R) for the MTP patch below: prefill_batch
        // overwrites d_R with its last row. d_mtap is idle outside prefill.
        CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        ngram_ckpt_save();
        std::vector<int> tokens{cur};
        tokens.insert(tokens.end(), draft.begin(), draft.end());
        auto t = std::chrono::steady_clock::now();
        verify_ng(tokens, base);
        std::vector<int> targets(tokens.size());
        batch_argmax((int)tokens.size(), targets.data());
        verify_ms += elapsed(t);
        int a = 0;
        while (a < (int)draft.size() && draft[a] == targets[a]) ++a;
        int keep = a;
        for (int j = 0; j < a; ++j) {
          if (emit && !emit(draft[j])) { keep = j; stop = true; break; }
          out.push_back(draft[j]);
        }
        if (!stop) {
          if (emit && !emit(targets[a])) stop = true;
          else out.push_back(targets[a]);
        }
        if (keep != (int)draft.size()) {
          // Restore every overwritten position-mod state, then replay exactly
          // [cur, accepted drafts] (same as spec_loop_ngram).
          t = std::chrono::steady_clock::now();
          ngram_ckpt_restore();
          chist.resize(history_size);
          pos = base;
          ple_ringpos = base % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          tokens.resize(keep + 1);
          verify_ng(tokens, base);
          CK(hipStreamSynchronize(g_str));
          replay_ms += elapsed(t);
          ++ng_roll;
        }
        cur = targets[keep];
        accepted += keep;
        ++ng_rounds;
        last_spec_proposed += (int)draft.size();
        // Patch the MTP KV over committed positions base-1..base+keep-1: tap
        // row 0 = stashed h_{base-1} paired with cur; rows 1..keep = d_Rb
        // rows 0..keep-1 (the committed tokens' trunk streams — the replay
        // re-generated them on partial accept). This restores the round-start
        // invariant (mtp_pos = pos-1), keeping MTP drafts live for later
        // rounds. d_R itself already holds h_{pos-1} (verify/replay leave the
        // last processed row's stream there; mtp_ingest_b does not touch it).
        if (keep > 0)
          CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                            (size_t)keep * 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        mtp_ingest_b(d_mtap,
                     std::vector<int>(tokens.begin(),
                                      tokens.begin() + keep + 1),
                     keep + 1, base - 1);
      } else if (!draft.empty()) {
        // ---- ngram round: chunked verify, tokens emit as each chunk ----
        // accepts. A rejection restores only this chunk's checkpoint and
        // replays its committed prefix (bounded by ng_cs), replacing the
        // legacy 65-row verify + keep-sized replay. ng_cs doubles after a
        // fully-accepted round (hot tables return to single-pass 65-row
        // throughput) and halves after a rejection. MTP KV is patched per
        // chunk while its d_Rb rows are still live; d_mtap row 0 carries the
        // h_{base-1} stash for the chunk about to verify (verify/replay never
        // touch d_mtap — mtp_tap_capture is off outside prefill).
        CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        int cur_c = cur, keep_round = 0;
        bool round_full = true;
        int di = 0;
        while (di < (int)draft.size()) {
          const int cs = std::min(ng_cs, (int)draft.size() - di);
          const int base = pos;
          const size_t history_size = chist.size();
          ngram_ckpt_save();
          std::vector<int> tokens{cur_c};
          tokens.insert(tokens.end(), draft.begin() + di,
                        draft.begin() + di + cs);
          auto t = std::chrono::steady_clock::now();
          verify_ng(tokens, base);
          std::vector<int> targets(tokens.size());
          batch_argmax((int)tokens.size(), targets.data());
          verify_ms += elapsed(t);
          int a = 0;
          while (a < cs && draft[di + a] == targets[a]) ++a;
          int keep = a;
          for (int j = 0; j < a; ++j) {
            if (emit && !emit(draft[di + j])) { keep = j; stop = true; break; }
            out.push_back(draft[di + j]);
          }
          if (!stop) {
            if (emit && !emit(targets[a])) stop = true;
            else out.push_back(targets[a]);
          }
          if (keep != cs) {
            // Mid-chunk rejection (or consumer stop): rows keep+1..cs
            // polluted the state — restore this chunk's checkpoint and
            // replay exactly its committed prefix. The round ends here.
            t = std::chrono::steady_clock::now();
            ngram_ckpt_restore();
            chist.resize(history_size);
            pos = base;
            ple_ringpos = base % 9;
            CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
            CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4,
                              hipMemcpyHostToDevice, g_str));
            tokens.resize(keep + 1);
            verify_ng(tokens, base);
            CK(hipStreamSynchronize(g_str));
            replay_ms += elapsed(t);
            ++ng_roll;
            round_full = false;
          }
          // MTP patch for this chunk's committed positions (same tap layout
          // as the legacy round: row 0 = stash, rows 1..keep = d_Rb rows).
          if (keep > 0)
            CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                              (size_t)keep * 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          mtp_ingest_b(d_mtap,
                       std::vector<int>(tokens.begin(),
                                        tokens.begin() + keep + 1),
                       keep + 1, base - 1);
          keep_round += keep;
          cur_c = targets[keep];
          if (!round_full || stop) break;
          // Chunk fully accepted: the continuation targets[cs] fills the
          // position the next draft token proposes. Chain into the next
          // chunk only when it equals that draft token (a mismatch is a
          // boundary rejection: the correction is already committed, the
          // state is clean, and the round ends without a replay).
          if (di + cs >= (int)draft.size()) break;  // bonus row: draft done
          if (targets[cs] != draft[di + cs]) { round_full = false; break; }
          ++keep_round;  // the splice committed draft[di+cs] as a draft
          di += cs + 1;
          // Stash h_{pos-1} (d_R = the accepted chunk's last row) as the next
          // chunk's tap pair before its verify overwrites d_R.
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        }
        cur = cur_c;
        accepted += keep_round;
        ++ng_rounds;
        last_spec_proposed += (int)draft.size();
        if (!ng_cs_fixed) {
          if (round_full && (int)draft.size() > ng_cs)
            ng_cs = std::min(64, ng_cs * 2);
          else if (!round_full)
            ng_cs = std::max(8, ng_cs / 2);
        }
      } else {
        // ---- MTP round (ngram table missed): short draft, snapshot rollback
        // ---- (same machinery as spec_loop)
        int q = pos - 1;
        int g = std::min({gamma, gen - (int)out.size(), maxctx - pos - 1});
        if (g < 1) break;  // context edge: no room for even one draft
        size_t chist_sz = chist.size();
        mtp_ckpt_save();
        auto t0 = std::chrono::steady_clock::now();
        drafts[0] = mtp_step(d_R, cur);
        // draft 1's indexer ring rows are truth; later drafts pollute them.
        CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        for (int j = 1; j < g; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
        std::vector<int> vt;
        vt.reserve(g + 1);
        vt.push_back(cur);
        for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
        spec_snap = true;  // capture per-row rollback snapshots during verify
        prefill_batch(vt, q + 1, true, true);
        spec_snap = false;
        batch_argmax(g + 1, v.data());
        verify_ms += elapsed(t0);
        int a = 0;
        while (a < g && drafts[a] == v[a]) a++;
        for (int j = 0; j < g; j++) {
          depth_tot[j]++;
          if (j < a) depth_hits[j]++;
        }
        ++mtp_rounds;
        last_spec_proposed += g;
        int d = a;
        bool keep_v = true;
        if (emit) {
          for (int j = 0; j < a; j++)
            if (!emit(drafts[j])) { d = j; stop = true; keep_v = false; break; }
          if (!stop && !emit(v[a])) { stop = true; keep_v = false; }
        }
        for (int j = 0; j < d; j++) out.push_back(drafts[j]);
        if (keep_v) out.push_back(v[a]);
        accepted += d;
        if (!stop && d == g) {
          cur = v[g];  // full accept; state already at q+1+g
        } else {
          ++mtp_roll;
          // Install the state at commit depth d from the verify snapshots —
          // identical to spec_loop's rollback.
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          const size_t sL = 48 * 128 * 128;
          for (int gi = 0; gi < ngdn; gi++)
            CK(hipMemcpyAsync(d_S + (size_t)gi * sL,
                              d_Ssnap + ((size_t)d * ngdn + gi) * sL, sL * 4,
                              hipMemcpyDeviceToDevice, g_str));
          CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)d * ngdn * 3 * 10240,
                            (size_t)ngdn * 3 * 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          if (ckpt_iraw)
            k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
                d_iproj_snap, d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
                d_iraw, q + 1, q + d + 1, 9);
          const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
          for (int p = q + d + 2; p <= q + g + 1; p++)
            CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                              ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          pos = q + d + 2;
          ple_ringpos = pos % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          restore_dR_row(d);
          cur = v[d];
        }
        // Committed-token history: rebuild explicitly so it is right whether
        // or not ple_on made the verify append (spec_loop relies on ple_on).
        chist.resize(chist_sz);
        chist.insert(chist.end(), vt.begin(), vt.begin() + d + 1);
        // MTP self-repair over committed positions q+1..q+d (as spec_loop).
        CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (d > 0) {
          std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
          mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
        } else {
          mtp_pos = q + 1;
          CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
        }
      }
      ++last_spec_rounds;
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = (int)out.size();
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    last_spec_commit = (int)out.size();
    last_spec_rollbacks = ng_roll + mtp_roll;
    fprintf(stderr,
            "%s chain: %d tokens in %.2f s = %.1f tok/s | rounds=%d (ngram=%d "
            "mtp=%d) commit/round=%.2f accepted=%d proposed=%d rollbacks=%d\n",
            log_ts(), (int)out.size(), sec, sec > 0 ? out.size() / sec : 0.0,
            last_spec_rounds, ng_rounds, mtp_rounds,
            last_spec_rounds ? (double)out.size() / last_spec_rounds : 0.0,
            accepted, last_spec_proposed, last_spec_rollbacks);
    if (mtp_rounds) {
      fprintf(stderr, "chain mtp depth acc:");
      for (int j = 0; j < gamma && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr, "\n");
    }
    return out;
  }

  // Sampling speculative decode. The sampler supplies normalized filtered
  // distributions, independent draft/accept RNG streams, and residual draws.
  // Accepted draft tokens use p/q rejection sampling; a rejection draws from
  // normalized max(p-q, 0), while full acceptance draws the verify continuation.
  template <typename Sampler>
  std::vector<int> spec_loop_sample(
      int cur, int gen, int gamma, Sampler& sampler,
      const std::function<bool(int, float)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(gen + 1);
    std::vector<int> drafts(gamma);
    std::vector<float> draft_lps(gamma), verify_logits, p_row(g_cfg.vocab);
    std::vector<std::vector<float>> q_rows(gamma);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int rounds = 0, rollbacks = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, t_ingest = 0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen) {
      int q = pos - 1;
      int g = std::min(gamma, gen - (int)out.size());
      size_t chist_sz = chist.size();
      mtp_ckpt_save();
      Sampler draft_sampler = sampler.fork_stream(
          0xd1b54a32d192ed03ull ^ (uint64_t)(q + 1));
      Sampler accept_sampler = sampler.fork_stream(
          0x3c6ef372fe94f82aull ^ (uint64_t)(q + 1));
      auto t0 = std::chrono::steady_clock::now();
      for (int j = 0; j < g; j++) {
        mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, &q_rows[j]);
        draft_sampler.prepare(q_rows[j]);
        float qlp = 0.0f;
        // Penalties are defined over committed/emitted tokens only. Provisional
        // draft tokens therefore do not enter the proposal sampler's history;
        // q_rows[j] records exactly the distribution that produced drafts[j].
        drafts[j] = draft_sampler.draw_prepared(q_rows[j], &qlp, false);
        if (j == 0) {
          // MTP indexer state after consuming cur is causal truth. Later
          // speculative draft steps are discarded before committed ingest.
          CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      std::vector<int> vt;
      vt.reserve(g + 1);
      vt.push_back(cur);
      for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
      spec_snap = true;
      prefill_batch(vt, q + 1, true, true);
      spec_snap = false;
      batch_logits(g + 1, verify_logits);

      Sampler target_view = sampler;
      int a = 0;
      for (; a < g; a++) {
        std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                    g_cfg.vocab, p_row.data());
        target_view.prepare(p_row);
        float px = p_row[drafts[a]];
        float qx = q_rows[a][drafts[a]];
        draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
        double threshold = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
        if (accept_sampler.uniform() >= threshold) break;
        target_view.remember(drafts[a]);
      }

      int next;
      float next_lp = 0.0f;
      target_view.rng = sampler.rng;
      if (a < g) {
        next = target_view.sample_residual(p_row, q_rows[a], &next_lp);
      } else {
        std::copy_n(verify_logits.data() + (size_t)g * g_cfg.vocab,
                    g_cfg.vocab, p_row.data());
        target_view.prepare(p_row);
        next = target_view.draw_prepared(p_row, &next_lp, false);
      }
      sampler.rng = target_view.rng;
      auto t2 = std::chrono::steady_clock::now();
      for (int j = 0; j < g; j++) {
        depth_tot[j]++;
        if (j < a) depth_hits[j]++;
      }
      rounds++;
      live_spec_rounds = rounds;
      live_spec_proposed += g;
      live_spec_commits = (int)out.size();  // pre-emit; one round of lag is fine

      int d = a, stopped_token = -1;
      bool stop = false, keep_next = true;
      if (emit) {
        for (int j = 0; j < a; j++)
          if (!emit(drafts[j], draft_lps[j])) {
            d = j;
            stopped_token = drafts[j];
            stop = true;
            keep_next = false;
            break;
          }
        if (!stop && !emit(next, next_lp)) {
          stop = true;
          keep_next = false;
        }
      }
      for (int j = 0; j < d; j++) {
        out.push_back(drafts[j]);
        sampler.remember(drafts[j]);
      }
      if (keep_next) {
        out.push_back(next);
        sampler.remember(next);
      }

      auto t3 = std::chrono::steady_clock::now();
      if (!stop && d == g) {
        cur = next;
      } else {
        rollbacks++;
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        const size_t sL = 48 * 128 * 128;
        for (int gi = 0; gi < ngdn; gi++)
          CK(hipMemcpyAsync(d_S + (size_t)gi * sL,
                            d_Ssnap + ((size_t)d * ngdn + gi) * sL, sL * 4,
                            hipMemcpyDeviceToDevice, g_str));
        CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)d * ngdn * 3 * 10240,
                          (size_t)ngdn * 3 * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (ckpt_iraw)
          k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
              d_iproj_snap, d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
              d_iraw, q + 1, q + d + 1, 9);
        const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
        for (int p = q + d + 2; p <= q + g + 1; p++)
          CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                            ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        chist.resize(chist_sz + d + 1);
        pos = q + d + 2;
        ple_ringpos = pos % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                          g_str));
        restore_dR_row(d);
        cur = stopped_token >= 0 ? stopped_token : next;
      }
      auto t4 = std::chrono::steady_clock::now();
      CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                        hipMemcpyDeviceToDevice, g_str));
      if (d > 0) {
        std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
        mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
      } else {
        mtp_pos = q + 1;
        CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
      }
      if (stop) break;
      auto t5 = std::chrono::steady_clock::now();
      t_draft += ms(t0, t1);
      t_verify += ms(t1, t2);
      t_roll += ms(t3, t4);
      t_ingest += ms(t4, t5);
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    fprintf(stderr,
            "\nspec-sample: %d tokens in %.2f s = %.1f tok/s | rounds=%d "
            "commit/round=%.2f rollbacks=%d\n",
            (int)out.size(), sec, out.size() / sec, rounds,
            rounds ? (double)out.size() / rounds : 0.0, rollbacks);
    fprintf(stderr, "spec-sample depth acc:");
    for (int j = 0; j < gamma && depth_tot[j]; j++)
      fprintf(stderr, " %d:%.3f", j + 1,
              (double)depth_hits[j] / depth_tot[j]);
    fprintf(stderr,
            "\nspec-sample time: draft=%.0fms verify=%.0fms "
            "rollback=%.0fms ingest=%.0fms\n",
            t_draft, t_verify, t_roll, t_ingest);
    last_spec_rounds = rounds;
    last_spec_commit = (int)out.size();
    last_spec_proposed = live_spec_proposed;
    last_spec_rollbacks = rollbacks;
    return out;
  }

  // Sampling chain (llama.cpp ngram semantics + rejection-sampled MTP leg).
  // ngram round: a draft token is committed only when it equals a fresh draw
  // from the target distribution at its verify row; the first mismatching
  // draw becomes the correction token. Every emitted token is therefore an
  // exact target-distribution sample — the draft only decides how many
  // verify rows stay usable. Rollback/replay is the greedy ngram round's
  // checkpoint machinery; the MTP patch over committed positions is the
  // greedy chain's. The MTP round is spec_loop_sample's round body with the
  // greedy chain's explicit chist rebuild. mtp_ok=false degrades empty-draft
  // rounds to plain serial sampling steps (MTP KV unavailable).
  template <typename Sampler>
  std::vector<int> spec_loop_chain_sample(
      int cur, int gen, int gamma, Sampler& sampler, bool mtp_ok,
      const std::function<bool(int, float)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks =
        last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, ngram.n_max), maxbatch - 1});
    gamma = std::max(1, std::min(8, gamma));
    std::vector<int> drafts(8);
    std::vector<float> draft_lps(8), verify_logits, p_row(g_cfg.vocab);
    std::vector<std::vector<float>> q_rows(8);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int accepted = 0, ng_rounds = 0, mtp_rounds = 0, ng_roll = 0,
        mtp_roll = 0, fallback = 0;
    double verify_ms = 0, replay_ms = 0;
    // ngram verify chunking: same env semantics as the greedy chain; the
    // level lives in the shared Model member (see ng_chunk_init).
    ng_chunk_init();
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto verify_ng = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen && pos < maxctx) {
      if (chist.size() != (size_t)pos)
        throw std::runtime_error(
            "chain-sample: token history and trunk position disagree");
      ngram.observe(chist);
      // Same budget math as the greedy chain: one slot stays for the
      // target-model continuation, never verify past gen or maxctx.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      std::vector<int> draft =
          want > 0 ? ngram.draft(chist, cur, want) : std::vector<int>{};
      bool stop = false;
      if (!draft.empty() && ng_legacy) {
        // ---- ngram round: long draft, whole-state checkpoint + replay ----
        const int base = pos;
        const size_t history_size = chist.size();
        if (mtp_ok)
          // Stash h_{base-1} (d_R) for the MTP patch below (as spec_loop_chain).
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        ngram_ckpt_save();
        std::vector<int> tokens{cur};
        tokens.insert(tokens.end(), draft.begin(), draft.end());
        auto t = std::chrono::steady_clock::now();
        verify_ng(tokens, base);
        batch_logits((int)tokens.size(), verify_logits);
        verify_ms += elapsed(t);
        // Walk the draft rows with a scratch sampler view: accept draft[a]
        // iff it equals this round's draw from row a, else the draw is the
        // correction. keep = committed draft count (emit may stop early).
        Sampler view = sampler;
        const size_t committed0 = out.size();
        int a = 0, next = -1;
        float next_lp = 0.0f;
        for (; a < (int)draft.size(); a++) {
          std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                      g_cfg.vocab, p_row.data());
          float lp_a = 0.0f;
          int s = view.sample(p_row, &lp_a);
          if (s != draft[a]) {
            next = s;
            next_lp = lp_a;
            break;
          }
          if (emit && !emit(draft[a], lp_a)) { stop = true; break; }
          out.push_back(draft[a]);
        }
        const int keep = a;
        if (!stop) {
          if (next < 0) {
            // Full accept: the bonus row continues the sequence.
            std::copy_n(verify_logits.data() +
                            (size_t)draft.size() * g_cfg.vocab,
                        g_cfg.vocab, p_row.data());
            next = view.sample(p_row, &next_lp);
          }
          if (emit && !emit(next, next_lp)) stop = true;
          else out.push_back(next);
        }
        sampler.rng = view.rng;
        for (size_t j = committed0; j < out.size(); j++)
          sampler.remember(out[j]);
        if (keep != (int)draft.size()) {
          // Restore every overwritten position-mod state, then replay exactly
          // [cur, accepted drafts] (same as the greedy ngram round).
          t = std::chrono::steady_clock::now();
          ngram_ckpt_restore();
          chist.resize(history_size);
          pos = base;
          ple_ringpos = base % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          tokens.resize(keep + 1);
          verify_ng(tokens, base);
          CK(hipStreamSynchronize(g_str));
          replay_ms += elapsed(t);
          ++ng_roll;
        }
        if (next >= 0) cur = next;
        accepted += keep;
        ++ng_rounds;
        last_spec_proposed += (int)draft.size();
        if (mtp_ok) {
          // Patch the MTP KV over committed positions (as spec_loop_chain):
          // tap row 0 = stashed h_{base-1} paired with cur; rows 1..keep the
          // committed tokens' trunk streams. Restores mtp_pos = pos-1.
          if (keep > 0)
            CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                              (size_t)keep * 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          mtp_ingest_b(d_mtap,
                       std::vector<int>(tokens.begin(),
                                        tokens.begin() + keep + 1),
                       keep + 1, base - 1);
        }
      } else if (!draft.empty()) {
        // ---- chunked ngram round, sampling semantics: the Sampler view ----
        // walks rows in draft order across chunks, so the draw stream matches
        // the single-pass form; tokens emit per chunk (see the greedy chain).
        if (mtp_ok)
          // Stash h_{base-1} (d_R) for the per-chunk MTP patch below.
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        Sampler view = sampler;
        const size_t committed0 = out.size();
        int cur_c = cur, keep_round = 0;
        bool round_full = true;
        int cont = -1;
        float cont_lp = 0.0f;
        int di = 0;
        while (di < (int)draft.size()) {
          const int cs = std::min(ng_cs, (int)draft.size() - di);
          const int base = pos;
          const size_t history_size = chist.size();
          ngram_ckpt_save();
          std::vector<int> tokens{cur_c};
          tokens.insert(tokens.end(), draft.begin() + di,
                        draft.begin() + di + cs);
          auto t = std::chrono::steady_clock::now();
          verify_ng(tokens, base);
          batch_logits((int)tokens.size(), verify_logits);
          verify_ms += elapsed(t);
          cont = -1;
          int a = 0;
          for (; a < cs; a++) {
            std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                        g_cfg.vocab, p_row.data());
            float lp_a = 0.0f;
            int s = view.sample(p_row, &lp_a);
            if (s != draft[di + a]) { cont = s; cont_lp = lp_a; break; }
            if (emit && !emit(draft[di + a], lp_a)) { stop = true; break; }
            out.push_back(draft[di + a]);
          }
          const int keep = a;
          if (!stop) {
            if (cont < 0) {
              // Full chunk accept: the bonus row continues the sequence.
              std::copy_n(verify_logits.data() + (size_t)cs * g_cfg.vocab,
                          g_cfg.vocab, p_row.data());
              cont = view.sample(p_row, &cont_lp);
            }
            if (emit && !emit(cont, cont_lp)) stop = true;
            else out.push_back(cont);
          }
          if (keep != cs) {
            // Mid-chunk rejection: restore + replay only this chunk's
            // committed prefix (the greedy block has the full commentary).
            t = std::chrono::steady_clock::now();
            ngram_ckpt_restore();
            chist.resize(history_size);
            pos = base;
            ple_ringpos = base % 9;
            CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
            CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4,
                              hipMemcpyHostToDevice, g_str));
            tokens.resize(keep + 1);
            verify_ng(tokens, base);
            CK(hipStreamSynchronize(g_str));
            replay_ms += elapsed(t);
            ++ng_roll;
            round_full = false;
          }
          if (mtp_ok) {
            if (keep > 0)
              CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                                (size_t)keep * 10240 * 4,
                                hipMemcpyDeviceToDevice, g_str));
            mtp_ingest_b(d_mtap,
                         std::vector<int>(tokens.begin(),
                                          tokens.begin() + keep + 1),
                         keep + 1, base - 1);
          }
          keep_round += keep;
          if (cont >= 0) cur_c = cont;
          if (!round_full || stop) break;
          // Full chunk accept: chain into the next chunk only when the drawn
          // continuation equals the next draft token (see the greedy block);
          // a mismatch is a clean boundary rejection — no replay needed.
          if (di + cs >= (int)draft.size()) break;  // bonus row: draft done
          if (cont != draft[di + cs]) { round_full = false; break; }
          ++keep_round;  // the splice committed draft[di+cs] as a draft
          di += cs + 1;
          if (mtp_ok)
            CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                              g_str));
        }
        sampler.rng = view.rng;
        for (size_t j = committed0; j < out.size(); j++)
          sampler.remember(out[j]);
        if (cont >= 0) cur = cur_c;
        accepted += keep_round;
        ++ng_rounds;
        last_spec_proposed += (int)draft.size();
        if (!ng_cs_fixed) {
          if (round_full && (int)draft.size() > ng_cs)
            ng_cs = std::min(64, ng_cs * 2);
          else if (!round_full)
            ng_cs = std::max(8, ng_cs / 2);
        }
      } else if (mtp_ok) {
        // ---- MTP round (ngram table missed): spec_loop_sample's round ----
        int q = pos - 1;
        int g = std::min({gamma, gen - (int)out.size(), maxctx - pos - 1});
        if (g < 1) break;  // context edge: no room for even one draft
        size_t chist_sz = chist.size();
        mtp_ckpt_save();
        Sampler draft_sampler = sampler.fork_stream(
            0xd1b54a32d192ed03ull ^ (uint64_t)(q + 1));
        Sampler accept_sampler = sampler.fork_stream(
            0x3c6ef372fe94f82aull ^ (uint64_t)(q + 1));
        auto t = std::chrono::steady_clock::now();
        for (int j = 0; j < g; j++) {
          mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, &q_rows[j]);
          draft_sampler.prepare(q_rows[j]);
          // Provisional drafts stay out of the penalty history (as
          // spec_loop_sample): q_rows[j] records the producing distribution.
          drafts[j] =
              draft_sampler.draw_prepared(q_rows[j], &draft_lps[j], false);
          if (j == 0) {
            // Draft 1's indexer ring rows are truth; later drafts pollute.
            CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          }
        }
        std::vector<int> vt;
        vt.reserve(g + 1);
        vt.push_back(cur);
        for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
        spec_snap = true;  // capture per-row rollback snapshots during verify
        prefill_batch(vt, q + 1, true, true);
        spec_snap = false;
        batch_logits(g + 1, verify_logits);
        verify_ms += elapsed(t);
        Sampler target_view = sampler;
        int a = 0;
        for (; a < g; a++) {
          std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                      g_cfg.vocab, p_row.data());
          target_view.prepare(p_row);
          float px = p_row[drafts[a]];
          float qx = q_rows[a][drafts[a]];
          draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
          double threshold = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
          if (accept_sampler.uniform() >= threshold) break;
          target_view.remember(drafts[a]);
        }
        int next;
        float next_lp = 0.0f;
        target_view.rng = sampler.rng;
        if (a < g) {
          next = target_view.sample_residual(p_row, q_rows[a], &next_lp);
        } else {
          std::copy_n(verify_logits.data() + (size_t)g * g_cfg.vocab,
                      g_cfg.vocab, p_row.data());
          target_view.prepare(p_row);
          next = target_view.draw_prepared(p_row, &next_lp, false);
        }
        sampler.rng = target_view.rng;
        for (int j = 0; j < g; j++) {
          depth_tot[j]++;
          if (j < a) depth_hits[j]++;
        }
        ++mtp_rounds;
        last_spec_proposed += g;
        int d = a, stopped_token = -1;
        bool keep_next = true;
        if (emit) {
          for (int j = 0; j < a; j++)
            if (!emit(drafts[j], draft_lps[j])) {
              d = j;
              stopped_token = drafts[j];
              stop = true;
              keep_next = false;
              break;
            }
          if (!stop && !emit(next, next_lp)) { stop = true; keep_next = false; }
        }
        for (int j = 0; j < d; j++) {
          out.push_back(drafts[j]);
          sampler.remember(drafts[j]);
        }
        if (keep_next) {
          out.push_back(next);
          sampler.remember(next);
        }
        accepted += d;
        if (!stop && d == g) {
          cur = next;  // full accept; state already at q+1+g
        } else {
          ++mtp_roll;
          // Install the state at commit depth d from the verify snapshots —
          // identical to spec_loop_sample's rollback.
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          const size_t sL = 48 * 128 * 128;
          for (int gi = 0; gi < ngdn; gi++)
            CK(hipMemcpyAsync(d_S + (size_t)gi * sL,
                              d_Ssnap + ((size_t)d * ngdn + gi) * sL, sL * 4,
                              hipMemcpyDeviceToDevice, g_str));
          CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)d * ngdn * 3 * 10240,
                            (size_t)ngdn * 3 * 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          if (ckpt_iraw)
            k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
                d_iproj_snap, d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
                d_iraw, q + 1, q + d + 1, 9);
          const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
          for (int p = q + d + 2; p <= q + g + 1; p++)
            CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                              ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          pos = q + d + 2;
          ple_ringpos = pos % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          restore_dR_row(d);
          cur = stopped_token >= 0 ? stopped_token : next;
        }
        // Committed-token history: rebuild explicitly so it is right whether
        // or not ple_on made the verify append (as spec_loop_chain).
        chist.resize(chist_sz);
        chist.insert(chist.end(), vt.begin(), vt.begin() + d + 1);
        // MTP self-repair over committed positions q+1..q+d.
        CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (d > 0) {
          std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
          mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
        } else {
          mtp_pos = q + 1;
          CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
        }
      } else {
        // ---- no draft, no MTP: plain serial sampling step ----
        ++fallback;
        forward(cur);
        if (!ple_on) chist.push_back(cur);
        ple_ringpos = pos % 9;  // graph advances the device scalar
        get_logits(p_row);
        float lp_s = 0.0f;
        int nx = sampler.sample(p_row, &lp_s);
        if (emit && !emit(nx, lp_s)) break;
        out.push_back(nx);
        cur = nx;
        continue;  // serial steps are not speculation rounds
      }
      ++last_spec_rounds;
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = (int)out.size();
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    last_spec_commit = (int)out.size();
    last_spec_rollbacks = ng_roll + mtp_roll;
    fprintf(stderr,
            "%s chain-sample: %d tokens in %.2f s = %.1f tok/s | rounds=%d "
            "(ngram=%d mtp=%d serial=%d) commit/round=%.2f accepted=%d "
            "proposed=%d rollbacks=%d\n",
            log_ts(), (int)out.size(), sec, sec > 0 ? out.size() / sec : 0.0,
            last_spec_rounds, ng_rounds, mtp_rounds, fallback,
            last_spec_rounds ? (double)out.size() / last_spec_rounds : 0.0,
            accepted, last_spec_proposed, last_spec_rollbacks);
    if (mtp_rounds) {
      fprintf(stderr, "chain-sample mtp depth acc:");
      for (int j = 0; j < gamma && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1,
                (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr, "\n");
    }
    return out;
  }

  // --- decode graphs: G1 = token upload + embed + L0; G2 = [PLE] + L1..end --
  hipGraphExec_t g1_exec = nullptr, g2_exec = nullptr;
  bool graphs_ok = false;
  void build_graphs() {
    if (dbg || perf || getenv("GDEC_NOGRAPH")) return;
    // pre-populate lazy caches (hipMalloc/hipMemcpy are illegal during capture)
    for (int l = 0; l < g_cfg.layers; l++) {
      char p[128];
      snprintf(p, sizeof p, "layers.%d.attn_hyper_connection", l);
      ensure_hcnorm(p);
      snprintf(p, sizeof p, "layers.%d.mlp_hyper_connection", l);
      ensure_hcnorm(p);
      if (is_qsa(l)) {
        snprintf(p, sizeof p, "layers.%d.self_attn.q_norm.weight", l);
        ensure_norm256(p);
        snprintf(p, sizeof p, "layers.%d.self_attn.k_norm.weight", l);
        ensure_norm256(p);
      }
    }
    ensure_hcnorm("hyper_connection_mixer");
    hipStream_t cap;
    CK(hipStreamCreate(&cap));
    g_str = cap;
    hipGraph_t g1, g2;
    CK(hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal));
    CK(hipMemcpyAsync(d_token, p_token, 4, hipMemcpyHostToDevice, cap));
    enq_embed();
    enq_layer(0);
    CK(hipStreamEndCapture(cap, &g1));
    CK(hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal));
    if (ple_on) ple_gpu();
    for (int l = 1; l < g_cfg.layers; l++) enq_layer(l);
    enq_final();
    k_step_incr<<<1, 1, 0, cap>>>(d_pos, d_ringpos);
    CK(hipStreamEndCapture(cap, &g2));
    g_str = nullptr;
    CK(hipGraphInstantiate(&g1_exec, g1, nullptr, nullptr, 0));
    CK(hipGraphInstantiate(&g2_exec, g2, nullptr, nullptr, 0));
    CK(hipStreamDestroy(cap));
    graphs_ok = true;
    fprintf(stderr, "decode graphs captured (ple=%d)\n", (int)ple_on);
  }

  // Zero all cross-request state so the next request behaves like a fresh
  // process (the constructor memsets d_S/d_convst; hipMalloc's fresh zero
  // pages cover d_ring — reproduce both here).
  void reset_state() {
    pos = 0;
    ple_ringpos = 0;
    chist.clear();
    xf16_src = nullptr;  // invalidate the prefill bf16 conversion cache
    clear_mrope();       // drop any M-RoPE request state (d_rdelta <- 0)
    int ngdn = g_cfg.layers - g_cfg.layers / 4;
    CK(hipMemsetAsync(d_pos, 0, 4, 0));
    CK(hipMemsetAsync(d_ringpos, 0, 4, 0));
    CK(hipMemsetAsync(d_S, 0, (size_t)ngdn * 48 * 128 * 128 * 4, 0));
    CK(hipMemsetAsync(d_convst, 0, (size_t)ngdn * 3 * 10240 * 4, 0));
    CK(hipMemsetAsync(d_ring, 0, (size_t)9 * 10240 * 4, 0));
    CK(hipDeviceSynchronize());
  }

  int forward(int token) {
    if (graphs_ok) {
      *p_token = token;
      auto g0 = std::chrono::steady_clock::now();
      CK(hipGraphLaunch(g1_exec, 0));
      dump_decode_step(pos);  // debug only (GDEC_MROPE_DUMP); syncs the stream
      double g1_ms;
      if (ple_on) {
        ple_host(token, ple_e);  // host hash+read overlaps G1 on GPU
        g1_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - g0)
                    .count();
      } else {
        g1_ms = 0;
      }
      CK(hipGraphLaunch(g2_exec, 0));
      pos++;
      int id;
      auto sync0 = std::chrono::steady_clock::now();
      CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
      double sync_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync0)
              .count();
      if (getenv("GDEC_TIME")) {
        t_enq += g1_ms;
        t_sync += sync_ms;
        if (t_n % 16 == 15)
          fprintf(stderr, "TIME: ple_host=%.2f ms/tok g2_sync=%.2f ms/tok\n",
                  t_enq / (t_n + 1), t_sync / (t_n + 1));
      }
      t_n++;
      return id;
    }
    // direct path (debug/perf): upload scalars, enqueue everything
    CK(hipMemcpyAsync(d_token, &token, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
    auto T0 = tick();
    enq_embed();
    tock(0, T0);
    // layer 0 enqueued first so PLE host reads overlap with GPU work
    enq_layer(0);
    dump_decode_step(pos);  // debug only (GDEC_MROPE_DUMP); syncs the stream
    if (ple_on) {
      auto t1 = tick();
      ple_host(token, ple_e);
      tock(1, t1);
      ple_gpu();
      ple_ringpos = (ple_ringpos + 1) % 9;
    }
    for (int l = 1; l < g_cfg.layers; l++) enq_layer(l);
    auto t4 = tick();
    auto enq0 = std::chrono::steady_clock::now();
    enq_final();
    pos++;
    int id;
    double enq_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - enq0)
            .count();
    auto sync0 = std::chrono::steady_clock::now();
    CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    double sync_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync0)
            .count();
    if (getenv("GDEC_TIME")) {
      t_enq += enq_ms;
      t_sync += sync_ms;
      if (t_n % 16 == 15)
        fprintf(stderr, "TIME: enq=%.2f ms/tok sync=%.2f ms/tok\n", t_enq / t_n,
                t_sync / t_n);
    }
    tock(4, t4);
    t_n++;
    return id;
  }

  void perf_report() {
    if (!perf || !t_n) return;
    fprintf(stderr,
            "PERF ms/token: embed=%.2f ple=%.2f attn=%.2f moe=%.2f head=%.2f total=%.2f "
            "(%d tokens)\n",
            t_acc[0] / t_n, t_acc[1] / t_n, t_acc[2] / t_n, t_acc[3] / t_n, t_acc[4] / t_n,
            (t_acc[0] + t_acc[1] + t_acc[2] + t_acc[3] + t_acc[4]) / t_n, t_n);
  }

  void get_logits(std::vector<float>& out) {
    out.resize(g_cfg.vocab);
    CK(hipMemcpy(out.data(), d_logits, (size_t)g_cfg.vocab * 4,
                 hipMemcpyDeviceToHost));
  }
};

// --- QSA q/gate split + per-head zc norm: qg (24, 512) -> qs (24,256), gs (24,256)
// Batched: grid.y = token row (qg row stride 24*2*dh, qs/gs row stride 24*dh).
// Rope=true additionally applies the rotary (64 dims, pairs i/i+32) in-kernel,
// bit-exact vs the k_qsa_qsplit + k_rope_b sequence: the normed value is staged
// in LDS (store/reload of a float is exact) and the cos/sin come from the
// k_rope_cs table (bit-identical to k_rope_b's inline fp64 math, but computed
// once per (token, i) instead of once per (head, token, i) — gfx1151 fp64 is
// too weak to pay 24x redundant transcendentals inside this kernel).
// Requires dh == 256, rotary == 64 (all QSA callers).
template <bool Rope>
__global__ void k_qsa_qsplit(const float* __restrict__ qg, const float* __restrict__ qnw,
                             float* __restrict__ qs, float* __restrict__ gs, int dh,
                             float eps, int rope_base, const float2* __restrict__ ropecs) {
  int h = blockIdx.x;
  int t = threadIdx.x;
  qg += (size_t)blockIdx.y * 48 * dh;
  qs += (size_t)blockIdx.y * 24 * dh;
  gs += (size_t)blockIdx.y * 24 * dh;
  __shared__ float sm[256];
  float v = (t < dh) ? qg[h * 2 * dh + t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / dh + eps);
  if (t < dh) gs[h * dh + t] = qg[h * 2 * dh + dh + t];
  if (!Rope) {
    if (t < dh) qs[h * dh + t] = v * inv * (1.f + qnw[t]);
    return;
  }
  float* q = qs + h * dh;
  if (t < 32) {
    // recompute the pair's normed values inline (same expressions, same bits;
    // the extra qg load is L2-hot) instead of staging them through LDS —
    // saves two __syncthreads and the sm round-trip per block.
    float2 cs = ropecs[(size_t)blockIdx.y * 32 + t];
    float v1 = qg[h * 2 * dh + t + 32];
    float x0 = v * inv * (1.f + qnw[t]);
    float x1 = v1 * inv * (1.f + qnw[t + 32]);
    // contraction pinned to k_rope_b's compiled forms (verified bit-identical
    // at -O2 and -O3): r0 = fma(x0,cs,-(x1*sn)), r1 = fma(x1,cs,x0*sn)
    q[t] = fmaf(x0, cs.x, -(x1 * cs.y));
    q[t + 32] = fmaf(x1, cs.x, x0 * cs.y);
  } else if (t >= 64 && t < dh) {
    q[t] = v * inv * (1.f + qnw[t]);
  }
  (void)rope_base;
}

// --- Fused QSA K prepare, one block per (kv head, token): zc rmsnorm over the
// 256-dim head (same tree, weight per head == k_rmsnorm_zc_grouped with
// ngroups=2 addressing) + rotary (cos/sin from the k_rope_cs table, bit-
// identical to k_rope_b) + optional bf16 store into the compact KV cache
// (same f2bf as k_f32_to_bf16_v4). Bit-exact vs the rmsnorm + k_rope_b
// [+ k_f32_to_bf16_v4] sequence.
// grid = dim3(2, P); kc (when Bf16) already includes the base*512 offset.
template <bool Bf16>
__global__ void k_qsa_kprep(const float* __restrict__ kb, const float* __restrict__ knw,
                            float* __restrict__ kout, uint16_t* __restrict__ kc,
                            float eps, int rope_base,
                            const float2* __restrict__ ropecs) {
  int h = blockIdx.x;
  int t = threadIdx.x;
  const float* xg = kb + ((size_t)blockIdx.y * 2 + h) * 256;
  const float* wg = knw;  // both K heads share the single 256-weight (knfix)
  float* og = kout + ((size_t)blockIdx.y * 2 + h) * 256;
  __shared__ float sm[256];
  float v = xg[t];
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / 256 + eps);
  if (t < 32) {
    // inline recompute instead of LDS staging (see k_qsa_qsplit)
    float2 cs = ropecs[(size_t)blockIdx.y * 32 + t];
    float v1 = xg[t + 32];
    float x0 = v * inv * (1.f + wg[t]);
    float x1 = v1 * inv * (1.f + wg[t + 32]);
    // contraction pinned to k_rope_b's compiled forms (see k_qsa_qsplit)
    float r0 = fmaf(x0, cs.x, -(x1 * cs.y)), r1 = fmaf(x1, cs.x, x0 * cs.y);
    og[t] = r0;
    og[t + 32] = r1;
    if (Bf16) {
      uint16_t* dst = kc + (size_t)blockIdx.y * 512 + h * 256;
      dst[t] = f2bf(r0);
      dst[t + 32] = f2bf(r1);
    }
  } else if (t >= 64) {
    float o = v * inv * (1.f + wg[t]);
    og[t] = o;
    if (Bf16) kc[(size_t)blockIdx.y * 512 + h * 256 + t] = f2bf(o);
  }
  (void)rope_base;
}

// --- acc[r] += sigmoid(sg[i/n]) * y[i]; flat over total = P*n
__global__ void k_axpy_sg(float* __restrict__ acc, const float* __restrict__ y,
                          const float* __restrict__ sg, int n, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    float s = 1.f / (1.f + expf(-sg[i / n]));
    acc[i] += s * y[i];
  }
}

// ---------------- serve mode (gdec wire protocol) ----------------------------
// Batch-1: one connection, one in-flight request. Verbs: PING / INFO / GEN / X.
// GEN accepts an optional SAMPLE suffix (temp>0, top_k/top_p/min_p/seed) with
// PENALTY/BIAS/LOGPROBS as sampler-only companions; drafter>0 is rejected.
// INFO reports sampling on, prompt cache on, kv_slots=1.
// The verb grammars below (parse_gen, INFO fields) are the specification.

// ---------------- KV snapshots: SSD prompt cache ---------------------------
// A snapshot is the full recurrent state at end-of-context: GDN d_S + conv
// state, PLE ring + chist, QSA KV caches (first ntok rows), indexer keys/raw,
// pos. Restoring one continues bit-exactly as if the prefix had been
// prefilled live (raw state bytes; A/B verified against live prefill).
//
// Layout under GDEC_KVSNAP_DIR (default data/kvsnap, relative to CWD):
//   tmp_<pid>_/  --write files-->  rename -->  s_<ntok>_<hash16>/  (atomic)
//     meta       "KVS1 ntok nqsa ngdn maxctx kvb index_blocks ringpos"
//                (KVS2 = same + "rope_delta img_hash k t0 h0 w0 ..." for
//                M-RoPE states; placeholder ids are pixel-agnostic, so a
//                vision snapshot only restores for the same grids+pixels)
//     tokens     int32[ntok]
//     s.bin      ngdn*48*128*128 fp32 | convst.bin ngdn*3*10240 fp32
//     ring.bin   9*10240 fp32         | iraw.bin   nqsa*512 fp32
//     ik.bin     nqsa * (ntok/4+2) * 128 fp32
//     kv.bin     K then V, nqsa * ntok * 512 * kvb each (kvb = 2 bf16 / 4 fp32)
// A crashed save never publishes (rename is the commit point). Snapshots are
// self-contained -- no prefix-chain dependencies -- so eviction is
// whole-snapshot LRU by directory mtime (a restore bumps it), capped by
// GDEC_KVSNAP_MAX_GB (default 20). States shorter than GDEC_KVSNAP_MIN
// (default 4096 tokens) are not worth the IO. GDEC_KVSNAP=0 disables.
// All errors are non-fatal: the cache is an accelerator, never a
// correctness dependency.
namespace kvsnap {

struct Cfg {
  std::string dir;
  uint64_t cap_bytes = 20ull << 30;
  size_t min_tokens = 4096;
};

static uint64_t fnv1a(const int* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= (uint32_t)p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static bool wr(int fd, const void* data, size_t n) {
  const char* p = (const char*)data;
  while (n) {
    ssize_t w = os_write(fd, p, n);
    if (w <= 0) return false;
    p += w;
    n -= (size_t)w;
  }
  return true;
}

// D2H into staging, then append to fd.
static bool d2h_wr(int fd, const void* dptr, void* staging, size_t n) {
  if (hipMemcpy(staging, dptr, n, hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  return wr(fd, staging, n);
}

static bool h2d_rd(const void* src, void* dptr, size_t n) {
  return hipMemcpy(dptr, src, n, hipMemcpyHostToDevice) == hipSuccess;
}

struct Sizes {
  int ntok, nqsa, ngdn, maxctx, kvb, iblocks, ringpos;
  // KVS2 (M-RoPE) extras: placeholder token ids are pixel-agnostic, so an
  // M-RoPE snapshot is only valid for a request with the same grids and
  // the same patch-bytes hash.
  bool mrope = false;
  int rope_delta = 0;
  uint64_t img_hash = 0;
  std::vector<std::array<int, 3>> grids;
};

static bool read_meta(const std::string& dir, Sizes& s) {
  std::ifstream f(dir + "/meta");
  char tag[8];
  if (!(f >> tag)) return false;
  const bool v2 = !strcmp(tag, "KVS2");
  if (!v2 && strcmp(tag, "KVS1")) return false;
  if (!(f >> s.ntok >> s.nqsa >> s.ngdn >> s.maxctx >> s.kvb >> s.iblocks >>
        s.ringpos))
    return false;
  if (!v2) return true;
  s.mrope = true;
  size_t ng = 0;
  if (!(f >> s.rope_delta >> s.img_hash >> ng) || ng > 64) return false;
  s.grids.resize(ng);
  for (auto& g : s.grids)
    if (!(f >> g[0] >> g[1] >> g[2])) return false;
  return true;
}

static size_t ik_blocks(int ntok) { return (size_t)ntok / 4 + 2; }

static uint64_t state_bytes(const Sizes& s) {
  uint64_t fixed = (uint64_t)s.ngdn * 48 * 128 * 128 * 4 +
                   (uint64_t)s.ngdn * 3 * 10240 * 4 + 9 * 10240 * 4 +
                   (uint64_t)s.nqsa * 512 * 4;
  return fixed + (uint64_t)s.nqsa * ik_blocks(s.ntok) * 128 * 4 +
         2ull * s.nqsa * s.ntok * 512 * s.kvb;
}

static void save(GpuModel& m, const std::vector<int>& toks, const Cfg& cfg) {
  namespace fs = std::filesystem;
  const int ntok = (int)toks.size();
  const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
  const int kvb = m.qsa_kv_fp32 ? 4 : 2;
  char name[64];
  snprintf(name, sizeof name, "s_%d_%016llx", ntok,
           (unsigned long long)fnv1a(toks.data(), toks.size()));
  std::string dst = cfg.dir + "/" + name, tmp = cfg.dir + "/tmp_" + name;
  std::error_code ec;
  if (fs::exists(dst, ec)) {  // same state already cached: just bump LRU time
    fs::last_write_time(dst, fs::file_time_type::clock::now(), ec);
    return;
  }
  fs::remove_all(tmp, ec);
  if (fs::create_directories(tmp, ec) == false && ec) return;
  const size_t s_b = (size_t)ngdn * 48 * 128 * 128 * 4,
               conv_b = (size_t)ngdn * 3 * 10240 * 4, ring_b = 9 * 10240 * 4,
               iraw_b = (size_t)nqsa * 512 * 4,
               ik_b = (size_t)nqsa * ik_blocks(ntok) * 128 * 4,
               kvl_b = (size_t)ntok * 512 * kvb;
  size_t stage_b = std::max({s_b, ik_b, kvl_b});
  void* st = malloc(stage_b);
  if (!st) return;
  // one blob file per device state region; any short write aborts the save
  auto blob = [&](const char* fn, const void* dptr, size_t n) {
    int fd = os_open_wr((tmp + "/" + fn).c_str());
    if (fd < 0) return false;
    bool ok = d2h_wr(fd, dptr, st, n);
    if (os_close(fd)) ok = false;
    return ok;
  };
  bool ok = true;
  {  // meta
    std::ofstream f(tmp + "/meta");
    f << (m.mrope_active_ ? "KVS2 " : "KVS1 ") << ntok << " " << nqsa << " "
      << ngdn << " " << m.maxctx << " " << kvb << " " << m.index_blocks << " "
      << m.ple_ringpos << "\n";
    if (m.mrope_active_) {
      f << m.rope_delta_ << " " << m.img_hash_ << " " << m.mrope_grids_.size();
      for (const auto& g : m.mrope_grids_)
        f << " " << g[0] << " " << g[1] << " " << g[2];
      f << "\n";
    }
  }
  {  // tokens (host data, not a device blob)
    int fd = os_open_wr((tmp + "/tokens").c_str());
    if (fd < 0 || !wr(fd, toks.data(), toks.size() * 4) || os_close(fd)) ok = false;
  }
  ok = ok && blob("s.bin", m.d_S, s_b) && blob("convst.bin", m.d_convst, conv_b) &&
       blob("ring.bin", m.d_ring, ring_b) && blob("iraw.bin", m.d_iraw, iraw_b) &&
       blob("ik.bin", m.d_ik, ik_b);
  if (ok) {  // kv.bin: K half then V half, per-layer slices out of the strided cache
    int fd = os_open_wr((tmp + "/kv.bin").c_str());
    if (fd < 0)
      ok = false;
    for (int half = 0; half < 2 && ok; half++)
      for (int qi = 0; qi < nqsa && ok; qi++) {
        const void* src =
            m.qsa_kv_fp32
                ? (const void*)((half ? m.d_vc : m.d_kc) + (size_t)qi * m.maxctx * 512)
                : (const void*)((half ? m.d_vcb : m.d_kcb) + (size_t)qi * m.maxctx * 512);
        ok = d2h_wr(fd, src, st, kvl_b);
      }
    if (fd >= 0 && os_close(fd)) ok = false;
  }
  if (ok) {
    fs::rename(tmp, dst, ec);  // atomic publish
    ok = !ec;
  }
  free(st);
  if (!ok) {
    fs::remove_all(tmp, ec);
    fprintf(stderr, "kvsnap: save of %d tokens failed (non-fatal)\n", ntok);
    return;
  }
  Sizes mine{ntok, nqsa, ngdn, m.maxctx, kvb, m.index_blocks, 0};
  fprintf(stderr, "kvsnap: saved %d tokens (%.1f MB) -> %s\n", ntok,
          state_bytes(mine) / 1048576.0, name);
  // LRU eviction over the cap: oldest directory mtime first, never the one
  // just written. Sizes derive from each snapshot's meta (no du(1) needed).
  struct Ent {
    fs::path p;
    fs::file_time_type t;
    uint64_t bytes;
  };
  std::vector<Ent> ents;
  uint64_t total = 0;
  for (auto& de : fs::directory_iterator(cfg.dir, ec)) {
    Sizes s;
    std::string fn = de.path().filename().string();
    if (fn.rfind("s_", 0) || !read_meta(de.path().string(), s)) continue;
    uint64_t b = state_bytes(s) + (uint64_t)s.ntok * 4;
    ents.push_back({de.path(), fs::last_write_time(de.path(), ec), b});
    total += b;
  }
  std::sort(ents.begin(), ents.end(),
            [](const Ent& a, const Ent& b) { return a.t < b.t; });
  for (auto& e : ents) {
    if (total <= cfg.cap_bytes) break;
    if (e.p.filename().string() == name) continue;  // protect the new one
    uint64_t b = e.bytes;
    fs::remove_all(e.p, ec);
    if (!ec) {
      total -= b;
      fprintf(stderr, "kvsnap: evicted %s (LRU, cap %.1f GB)\n",
              e.p.filename().string().c_str(), cfg.cap_bytes / 1073741824.0);
    }
  }
}

// map a blob read-only and H2D it into place; expected size must match exactly.
static bool rd_blob(const std::string& path, void* dptr, size_t n) {
  int fd = os_open_rd(path.c_str());
  if (fd < 0) return false;
  int64_t fsz = os_fsize(fd);
  os_close(fd);
  if (fsz < 0 || (uint64_t)fsz != n) return false;
  void* mp = os_map_ro(path.c_str(), n);
  if (!mp) return false;
  bool ok = h2d_rd(mp, dptr, n);
  os_unmap_ro(mp, n);
  return ok;
}

static bool restore(GpuModel& m, const std::string& dir,
                    std::vector<int>& toks_out) {
  Sizes s;
  const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
  const int kvb = m.qsa_kv_fp32 ? 4 : 2;
  if (!read_meta(dir, s) || s.nqsa != nqsa || s.ngdn != ngdn ||
      s.maxctx != m.maxctx || s.kvb != kvb || s.iblocks != m.index_blocks)
    return false;
  const int ntok = s.ntok;
  {  // tokens
    std::ifstream f(dir + "/tokens", std::ios::binary);
    toks_out.resize(ntok);
    if (!f.read((char*)toks_out.data(), (std::streamsize)ntok * 4)) return false;
  }
  if (!rd_blob(dir + "/s.bin", m.d_S, (size_t)ngdn * 48 * 128 * 128 * 4) ||
      !rd_blob(dir + "/convst.bin", m.d_convst, (size_t)ngdn * 3 * 10240 * 4) ||
      !rd_blob(dir + "/ring.bin", m.d_ring, 9 * 10240 * 4) ||
      !rd_blob(dir + "/iraw.bin", m.d_iraw, (size_t)nqsa * 512 * 4) ||
      !rd_blob(dir + "/ik.bin", m.d_ik,
               (size_t)nqsa * ik_blocks(ntok) * 128 * 4))
    return false;
  {  // kv.bin: K half then V half, per-layer slices into the maxctx-strided cache
    const std::string kvp = dir + "/kv.bin";
    int fd = os_open_rd(kvp.c_str());
    if (fd < 0) return false;
    size_t kvl_b = (size_t)ntok * 512 * kvb, tot = 2 * (size_t)nqsa * kvl_b;
    int64_t fsz = os_fsize(fd);
    os_close(fd);
    if (fsz < 0 || (uint64_t)fsz != tot) return false;
    void* mp = os_map_ro(kvp.c_str(), tot);
    if (!mp) return false;
    bool ok = true;
    for (int half = 0; half < 2 && ok; half++)
      for (int qi = 0; qi < nqsa && ok; qi++) {
        const char* src = (const char*)mp + ((size_t)half * nqsa + qi) * kvl_b;
        void* dst = m.qsa_kv_fp32
                        ? (void*)((half ? m.d_vc : m.d_kc) + (size_t)qi * m.maxctx * 512)
                        : (void*)((half ? m.d_vcb : m.d_kcb) + (size_t)qi * m.maxctx * 512);
        ok = h2d_rd(src, dst, kvl_b);
      }
    os_unmap_ro(mp, tot);
    if (!ok) return false;
  }
  // host-side and scalar state: pos/ringpos (both mirrors), PLE token history
  m.pos = ntok;
  m.ple_ringpos = s.ringpos;
  CK(hipMemcpy(m.d_pos, &m.pos, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(m.d_ringpos, &m.ple_ringpos, 4, hipMemcpyHostToDevice));
  m.chist.assign(toks_out.begin(), toks_out.end());
  // bump LRU time (self-contained snapshots: eviction never breaks others)
  std::error_code ec;
  std::filesystem::last_write_time(dir, std::filesystem::file_time_type::clock::now(), ec);
  return true;
}

// Longest snapshot whose token list is a strict prefix of `prompt` (strictly
// shorter: a full-length match has no logits to resume generation from).
// M-RoPE candidates additionally require the request's grids and patch-bytes
// hash to match the snapshot meta (req_grids != nullptr for vision requests).
static std::string best_prefix(const Cfg& cfg, const std::vector<int>& prompt,
                               size_t* ntok_out,
                               const std::vector<std::array<int, 3>>* req_grids =
                                   nullptr,
                               uint64_t req_img_hash = 0) {
  namespace fs = std::filesystem;
  std::error_code ec;
  std::string best;
  size_t bestn = 0;
  for (auto& de : fs::directory_iterator(cfg.dir, ec)) {
    std::string fn = de.path().filename().string();
    if (fn.rfind("s_", 0)) continue;
    Sizes s;
    if (!read_meta(de.path().string(), s) || (size_t)s.ntok >= prompt.size() ||
        (size_t)s.ntok <= bestn)
      continue;
    if (s.mrope && (!req_grids || s.grids != *req_grids ||
                    s.img_hash != req_img_hash))
      continue;  // same-shaped different image: not this snapshot
    std::ifstream f(de.path().string() + "/tokens", std::ios::binary);
    std::vector<int> t(s.ntok);
    if (!f.read((char*)t.data(), (std::streamsize)s.ntok * 4)) continue;
    if (std::equal(t.begin(), t.end(), prompt.begin())) {
      best = de.path().string();
      bestn = (size_t)s.ntok;
    }
  }
  *ntok_out = bestn;
  return best;
}

}  // namespace kvsnap

namespace serve {

struct LineBuf {
  sockfd_t fd;
  std::string buf;
  bool take(std::string& out) {
    size_t nl = buf.find('\n');
    if (nl == std::string::npos) return false;
    out = buf.substr(0, nl);
    buf.erase(0, nl + 1);
    return true;
  }
  // blocking read of one line; false on EOF/error
  bool readline(std::string& out) {
    for (;;) {
      if (take(out)) return true;
      char tmp[65536];
      ssize_t n = sock_recv(fd, tmp, sizeof tmp, false);
      if (n <= 0) return false;
      buf.append(tmp, n);
    }
  }
  // non-blocking: true + line iff a complete line is available right now
  bool try_readline(std::string& out) {
    for (;;) {
      if (take(out)) return true;
      char tmp[65536];
      ssize_t n = sock_recv(fd, tmp, sizeof tmp, true);
      if (n <= 0) return false;  // EAGAIN (or EOF: surfaced by next readline)
      buf.append(tmp, n);
    }
  }
  // Blocking read of exactly n bytes (binary frames after a GEN line).
  // Bytes already swallowed into the line buffer are drained first, so the
  // client may pipeline frame bytes right behind the newline. False on
  // EOF/error/timeout (SO_RCVTIMEO, armed by the caller for bulk reads).
  bool read_full(void* dst, size_t n) {
    char* p = (char*)dst;
    size_t head = std::min(n, buf.size());
    if (head) {
      memcpy(p, buf.data(), head);
      buf.erase(0, head);
      p += head;
      n -= head;
    }
    while (n > 0) {
      ssize_t r = sock_recv(fd, p, n, false);
      if (r <= 0) return false;
      p += r;
      n -= (size_t)r;
    }
    return true;
  }
};

static bool send_all(sockfd_t fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t n = sock_send(fd, s.data() + off, s.size() - off);
    if (n <= 0) return false;
    off += n;
  }
  return true;
}

struct GenReq {
  int64_t req = 0;
  int max_tokens = 0;
  std::vector<int> eos;
  std::vector<int> ids;
  int drafter = -1;  // omitted: engine default; 0 serial; 1 MTP; 3 ngram-mod; 4 chain
  bool reject = false;  // malformed or unsupported optional field
  // Sampler (SAMPLE suffix). sample=true implies temp>0; greedy otherwise.
  bool sample = false;
  float temp = 1.0f, top_p = 0.0f, min_p = 0.0f;  int top_k = 0;
  uint64_t seed = 0;
  // PENALTY (presence, frequency) / BIAS (tid,val pairs) / LOGPROBS: sampler-only.
  float presence = 0.0f, frequency = 0.0f;
  std::vector<std::pair<int, float>> bias;
  bool logprobs = false;
  // SNAP / SNAP2: frontend-hinted prefix lengths worth an SSD snapshot (the
  // next turn repeats them verbatim; see serve_api.py render_prompt).
  size_t snap = 0, snap2 = 0;
  // MROPE <k> <t> <h> <w> ...: per-image grid triples, expanded in-engine
  // (stage 1: fresh prefill + serial decode only; cont/spec are rejected).
  std::vector<std::array<int, 3>> mrope_grids;
  // VIMG <k>: k binary patch frames follow the GEN line (one per MROPE grid,
  // same order; stage 4). Frame: u32 magic 0x56494D31, u32 P, u64 nbytes,
  // then P*1536 fp32 LE patches in block-major (C,T,ph,pw) order.
  int vimg = 0;
};



// Host-side sampler: logits come back over unified memory (~1 MB), all filtering
// is O(vocab) or less. Filter order follows HF warpers: penalties/bias on raw
// logits, then temperature, top_k, top_p, min_p. top_p uses a binary-search
// threshold (mass(p >= t) >= top_p) instead of a full sort; ties keep all tied
// tokens, unlike sort-and-cut. RNG is splitmix64: deterministic per
// (seed, emitted sequence).
struct HostSampler {
  float temp, top_p, min_p;
  int top_k;
  uint64_t rng;
  float presence, frequency;
  std::vector<std::pair<int, float>> bias;
  std::unordered_map<int, int> hist;  // emitted token counts (not the prompt)
  std::vector<int> cand;              // scratch: candidate token ids
  std::vector<float> work;            // scratch: top-k preservation / residual

  static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  uint64_t next64() {  // splitmix64
    return mix64(rng += 0x9E3779B97F4A7C15ull);
  }

  double uniform() {
    return (double)(next64() >> 11) * (1.0 / 9007199254740992.0);
  }

  HostSampler fork_stream(uint64_t tag) const {
    HostSampler out = *this;
    out.rng = mix64(rng ^ tag);
    out.cand.clear();
    out.work.clear();
    return out;
  }

  void remember(int id) { hist[id]++; }

  // Convert raw logits in place to the normalized distribution after the
  // engine's penalty/bias -> temperature -> top-k -> top-p -> min-p pipeline.
  void prepare(std::vector<float>& logits) {
    const int n = (int)logits.size();
    for (auto& b : bias)
      if (b.first >= 0 && b.first < n) logits[b.first] += b.second;
    if (presence != 0.0f || frequency != 0.0f)
      for (auto& h : hist)
        if (h.first >= 0 && h.first < n)
          logits[h.first] -= presence + frequency * (float)h.second;
    float lmax = -INFINITY;
    for (int i = 0; i < n; i++) lmax = fmaxf(lmax, logits[i]);
    // unnormalized probs after temperature
    for (int i = 0; i < n; i++) logits[i] = expf((logits[i] - lmax) / temp);
    cand.clear();
    if (top_k > 0 && top_k < n) {
      cand.resize(n);
      for (int i = 0; i < n; i++) cand[i] = i;
      std::nth_element(cand.begin(), cand.begin() + top_k, cand.end(),
                       [&](int a, int b) { return logits[a] > logits[b]; });
      cand.resize(top_k);
      work.resize(cand.size());
      for (size_t j = 0; j < cand.size(); j++) work[j] = logits[cand[j]];
      std::fill(logits.begin(), logits.end(), 0.0f);
      for (size_t j = 0; j < cand.size(); j++) logits[cand[j]] = work[j];
    } else {
      cand.reserve(n);
      for (int i = 0; i < n; i++)
        if (logits[i] > 0.0f) cand.push_back(i);
    }
    if (top_p > 0.0f && top_p < 1.0f && !cand.empty()) {
      // largest threshold t with mass(p >= t) >= top_p * total
      double total = 0;
      for (int i : cand) total += logits[i];
      double lo = 0, hi = 1.0;  // probs are scaled so max == 1
      for (int it = 0; it < 50; it++) {
        double mid = 0.5 * (lo + hi), m = 0;
        for (int i : cand)
          if (logits[i] >= (float)mid) m += logits[i];
        if (m >= top_p * total) lo = mid; else hi = mid;
      }
      std::vector<int> keep;
      keep.reserve(cand.size());
      for (int i : cand)
        if (logits[i] >= (float)lo)
          keep.push_back(i);
        else
          logits[i] = 0.0f;
      cand.swap(keep);
    }
    if (min_p > 0.0f && !cand.empty()) {
      float pmax = 0;
      for (int i : cand) pmax = fmaxf(pmax, logits[i]);
      float thr = min_p * pmax;
      std::vector<int> keep;
      keep.reserve(cand.size());
      for (int i : cand)
        if (logits[i] >= thr) keep.push_back(i);
      if (!keep.empty()) {
        for (int i : cand)
          if (logits[i] < thr) logits[i] = 0.0f;
        cand.swap(keep);  // never empty the candidate set
      }
    }
    double z = 0;
    for (int i : cand) z += logits[i];
    for (int i : cand) logits[i] = (float)(logits[i] / z);
  }

  // Draw from a distribution produced by prepare(). Candidate order is kept
  // stable with the serial sampler so existing seeded behavior is unchanged.
  int draw_prepared(const std::vector<float>& probs, float* lp,
                    bool add_hist) {
    double z = 0;
    for (int i : cand) z += probs[i];
    double u = uniform() * z;
    int sel = cand.back();
    for (int i : cand) {
      u -= probs[i];
      if (u <= 0) { sel = i; break; }
    }
    if (lp) *lp = (float)(log((double)probs[sel]) - log(z));
    if (add_hist) remember(sel);
    return sel;
  }

  int sample(std::vector<float>& logits, float* lp) {
    prepare(logits);
    return draw_prepared(logits, lp, true);
  }

  // Standard speculative correction distribution: normalize max(p-q, 0).
  // The reported logprob remains the target probability p(token), matching
  // serial LOGPROBS semantics rather than the residual draw probability.
  int sample_residual(const std::vector<float>& p,
                      const std::vector<float>& q, float* lp) {
    const int n = (int)p.size();
    work.assign(n, 0.0f);
    cand.clear();
    double z = 0.0;
    for (int i = 0; i < n; i++) {
      float d = fmaxf(0.0f, p[i] - q[i]);
      if (d > 0.0f) {
        work[i] = d;
        cand.push_back(i);
        z += d;
      }
    }
    if (cand.empty() || z <= 0.0) {
      for (int i = 0; i < n; i++)
        if (p[i] > 0.0f) {
          work[i] = p[i];
          cand.push_back(i);
          z += p[i];
        }
    }
    double u = uniform() * z;
    int sel = cand.back();
    for (int i : cand) {
      u -= work[i];
      if (u <= 0.0) {
        sel = i;
        break;
      }
    }
    if (lp) *lp = p[sel] > 0.0f ? logf(p[sel]) : -INFINITY;
    return sel;
  }
};

// Default drafter for GEN lines that omit the field (the HTTP API always
// omits it). Unset: chain — ngram drafts every round it can and MTP
// takes the rounds where the table has nothing, for greedy and sampled
// requests alike (sampled ngram rounds re-draw from the target distribution
// and accept by token-id match, llama.cpp-style; only the pure-ngram
// default stays greedy-only, see parse_gen). GDEC_DRAFTER overrides:
// "serial"/"0" forces serial, "ngram"/"3" pure ngram, "chain"/"4" explicit
// chain, "mtp" the legacy MTP default.
static int env_default_drafter() {
  const char* e = getenv("GDEC_DRAFTER");
  if (!e) return 4;
  if (!strcmp(e, "serial") || !strcmp(e, "0")) return 0;
  if (!strcmp(e, "ngram") || !strcmp(e, "ngram-mod") || !strcmp(e, "3")) return 3;
  if (!strcmp(e, "chain") || !strcmp(e, "4")) return 4;
  return -1;
}

// GEN <req> <max_tokens> <n_eos> <eos...> <n_ids> <ids...> [drafter] [suffixes]
static GenReq parse_gen(std::istringstream& ss) {
  GenReq g;
  int n_eos = 0, n_ids = 0;
  if (!(ss >> g.req >> g.max_tokens >> n_eos) || n_eos < 0 || n_eos > 64 ||
      g.max_tokens <= 0) {
    g.reject = true;
    return g;
  }
  for (int i = 0; i < n_eos; i++) {
    int e;
    if (!(ss >> e)) { g.reject = true; return g; }
    g.eos.push_back(e);
  }
  if (!(ss >> n_ids) || n_ids <= 0) { g.reject = true; return g; }
  g.ids.resize(n_ids);
  for (int i = 0; i < n_ids; i++)
    if (!(ss >> g.ids[i])) { g.reject = true; return g; }
  std::string tok;
  while (ss >> tok) {
    if (tok == "SAMPLE") {
      // SAMPLE <temp> <top_k> <top_p> <min_p> <seed:u64>
      if (!(ss >> g.temp >> g.top_k >> g.top_p >> g.min_p >> g.seed) ||
          g.temp <= 0.0f || g.top_k < 0) {
        g.reject = true;
        return g;
      }
      g.sample = true;
    } else if (tok == "NGRAM") {
      g.drafter = 3;
    } else if (tok == "PENALTY") {
      if (!(ss >> g.presence >> g.frequency)) { g.reject = true; return g; }
    } else if (tok == "BIAS") {
      int nb;
      if (!(ss >> nb) || nb < 0 || nb > 20480) { g.reject = true; return g; }
      g.bias.resize(nb);
      for (int i = 0; i < nb; i++) {
        if (!(ss >> g.bias[i].first >> g.bias[i].second) ||
            g.bias[i].first < 0 || g.bias[i].first >= 248320) {
          g.reject = true;
          return g;
        }
      }
    } else if (tok == "LOGPROBS") {
      g.logprobs = true;
    } else if (tok == "MROPE") {
      int k;
      if (!(ss >> k) || k <= 0 || k > 64) { g.reject = true; return g; }
      for (int i = 0; i < k; i++) {
        std::array<int, 3> g3{0, 0, 0};
        if (!(ss >> g3[0] >> g3[1] >> g3[2])) { g.reject = true; return g; }
        g.mrope_grids.push_back(g3);
      }
    } else if (tok == "VIMG") {
      if (!(ss >> g.vimg) || g.vimg <= 0 || g.vimg > 16) {
        g.reject = true;
        return g;
      }
    } else if (tok == "SNAP" || tok == "SNAP2") {
      long long v;
      if (!(ss >> v) || v < 0) { g.reject = true; return g; }
      (tok == "SNAP" ? g.snap : g.snap2) = (size_t)v;
    } else if (isdigit(tok[0]) || tok[0] == '-') {
      int v = atoi(tok.c_str());
      if (v < 0 || v > 4) g.reject = true;
      else g.drafter = v;
    }
    // anything else: drop (lenient by design)
  }
  // Omitted drafter: resolve the GDEC_DRAFTER default. Pure ngram only
  // drafts greedy requests (sampled ones keep the legacy default); chain
  // covers greedy and sampled alike.
  if (g.drafter < 0) {
    static const int def = env_default_drafter();
    g.drafter = (def == 3 && g.sample) ? -1 : def;
  }
  // PENALTY/BIAS/LOGPROBS are sampler-only (protocol spec): no sampler, no-op
  // fields would silently mislead the client.
  if (!g.sample &&
      (g.presence != 0.0f || g.frequency != 0.0f || !g.bias.empty() || g.logprobs))
    g.reject = true;
  return g;
}

static int run(GpuModel& m, int port) {
#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);
#endif
  sockfd_t lfd = sock_tcp();
  int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (bind(lfd, (sockaddr*)&addr, sizeof addr) || listen(lfd, 4)) {
    fprintf(stderr, "serve: bind/listen failed on :%d: %s\n", port,
            sock_strerror(sock_errno()));
    return 1;
  }
  fprintf(stderr, "serve: listening on :%d (maxctx %d)\n", port, m.maxctx);
  kvsnap::Cfg kvs;
  kvs.dir = "data/kvsnap";
  if (const char* v = getenv("GDEC_KVSNAP_DIR")) kvs.dir = v;
  if (const char* v = getenv("GDEC_KVSNAP"))
    if (!strcmp(v, "0")) kvs.dir.clear();
  if (const char* v = getenv("GDEC_KVSNAP_MAX_GB"))
    kvs.cap_bytes = (uint64_t)(atof(v) * 1073741824.0);
  if (const char* v = getenv("GDEC_KVSNAP_MIN")) kvs.min_tokens = (size_t)atol(v);
  if (!kvs.dir.empty())
    fprintf(stderr, "kvsnap: dir %s, cap %.1f GB, min %zu tokens\n",
            kvs.dir.c_str(), kvs.cap_bytes / 1073741824.0, kvs.min_tokens);
  for (;;) {
    sockfd_t cfd = accept(lfd, nullptr, nullptr);
    if (cfd < 0) continue;
    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
    fprintf(stderr, "%s serve: client connected\n", log_ts());
    LineBuf lb{cfd, {}};
    std::string line;
    bool alive = true;
    // tokens INGESTED on this connection (prompt + every token actually fed
    // to forward). KV reuse: a GEN whose prompt extends hist exactly
    // continues from live state instead of re-prefilling.
    std::vector<int> hist;
    // MTP KV continuity: true after a spec-enabled turn (prefill tap capture
    // + spec decode keep the MTP layer's KV in lockstep with the trunk).
    // False after kvsnap restores and any plain (non-speculative) turn.
    bool mtp_live = false;
    while (alive && lb.readline(line)) {
      if (line.empty()) continue;
      std::istringstream ss(line);
      std::string verb;
      ss >> verb;
      if (verb == "PING") {
        alive = send_all(cfd, "PONG\n");
      } else if (verb == "INFO") {
        char ib[160];
        // fields: mtp draft_head ctx spec_rows default drafter_weights dflash2
        //         cache_mb cache_align kv_slots slot_ctx cache_mode sampling
        // The default drafter is the chain (ngram first, MTP fallback) unless
        // GDEC_DRAFTER overrides; draft_head is a separate shortlist readout
        // and is not part of this checkpoint.
        snprintf(ib, sizeof ib, "I %d 0 %d 8 %d %d 0 0 0 1 %d 1 1\n",
                 m.mtp_avail ? 1 : 0, m.maxctx,
                 env_default_drafter() >= 0 ? env_default_drafter()
                                            : (m.mtp_avail ? 1 : 0),
                 m.mtp_avail ? 1 : 0, m.maxctx);
        alive = send_all(cfd, ib);
      } else if (verb == "X") {
        // no GEN in flight at this point (mid-decode cancel is polled below)
        int64_t req = 0;
        ss >> req;
        char db[96];
        snprintf(db, sizeof db, "D %lld cancel 0 0 0.0 0.0\n", (long long)req);
        alive = send_all(cfd, db);
      } else if (verb == "GEN") {
        GenReq g = parse_gen(ss);
        if (g.reject || (int)g.ids.size() + g.max_tokens > m.maxctx) {
          char db[128];
          snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n", (long long)g.req,
                   g.ids.size());
          alive = send_all(cfd, db);
          continue;
        }
        // VIMG binary frames follow the GEN line immediately and must be
        // drained before ANY reply path below, or their bytes desync the
        // line protocol. 120 s ceiling on the bulk transfer (~50 MB per
        // 1080p image over loopback is sub-second; the ceiling only guards
        // against a stalled client wedging the batch-1 server).
        std::vector<std::vector<float>> vpatches;
        if (g.vimg > 0) {
          bool ok = true;
          sock_rcvtimeo(cfd, 120);
          vpatches.resize(g.vimg);
          for (int i = 0; i < g.vimg && ok; i++) {
            struct {
              uint32_t magic, P;
              uint64_t nbytes;
            } fh;
            ok = lb.read_full(&fh, sizeof fh) && fh.magic == 0x56494D31u &&
                 fh.P > 0 && fh.P <= 65536 &&
                 fh.nbytes == (uint64_t)fh.P * 1536 * 4;
            if (!ok) break;
            vpatches[i].resize((size_t)fh.P * 1536);
            ok = lb.read_full(vpatches[i].data(), fh.nbytes);
          }
          sock_rcvtimeo(cfd, 0);
          if (!ok) {
            fprintf(stderr, "vision: frame read failed for req %lld "
                    "(timeout/short/garbled); dropping the connection\n",
                    (long long)g.req);
            char db[128];
            snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n",
                     (long long)g.req, g.ids.size());
            send_all(cfd, db);
            break;  // protocol desync: cannot trust anything after this
          }
        }
        // Vision request validation (0.5.8 semantics), before state mutation:
        // patches need a loaded tower and one MROPE grid each; any image
        // placeholder without a grid is rejected (first position logged).
        if (g.vimg > 0) {
          const char* why = nullptr;
          if (!m.vision_on)
            why = "vision tower not loaded (start the engine with "
                  "--vision-tower <sidecar.hgn>)";
          else if ((int)g.mrope_grids.size() != g.vimg)
            why = "VIMG frame count != MROPE grid count";
          else
            for (int i = 0; i < g.vimg; i++) {
              auto gr = g.mrope_grids[i];
              if (gr[0] != 1 ||
                  (int)vpatches[i].size() != gr[0] * gr[1] * gr[2] * 1536) {
                why = "patches/grid size mismatch (or t != 1)";
                break;
              }
            }
          if (why) {
            fprintf(stderr, "vision: req %lld rejected: %s\n",
                    (long long)g.req, why);
            char db[128];
            snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n",
                     (long long)g.req, g.ids.size());
            alive = send_all(cfd, db);
            continue;
          }
        }
        if (g.mrope_grids.empty()) {
          size_t bad = g.ids.size();
          for (size_t i = 0; i < g.ids.size(); i++)
            if (g.ids[i] == 248056 || g.ids[i] == 248057) {
              bad = i;
              break;
            }
          if (bad < g.ids.size()) {
            fprintf(stderr, "vision: req %lld rejected: image placeholder at "
                    "prompt pos %zu without MROPE/VIMG\n", (long long)g.req,
                    bad);
            char db[128];
            snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n",
                     (long long)g.req, g.ids.size());
            alive = send_all(cfd, db);
            continue;
          }
        }
        // KV reuse: longest common prefix of live state and this prompt.
        // Continue only when hist is a STRICT prefix (mcp == hist.size() and
        // new tokens follow); anything else falls to the SSD snapshot cache
        // and only then to a full recompute.
        size_t mcp = 0;
        while (mcp < hist.size() && mcp < g.ids.size() && hist[mcp] == g.ids[mcp])
          mcp++;
        bool cont = !hist.empty() && mcp == hist.size() && mcp < g.ids.size();
        int n_cached = cont ? (int)mcp : 0;
        // M-RoPE restrictions, decided before any state mutation:
        // KV reuse (cont) IS sound for M-RoPE — prefix positions are
        // suffix-independent and RoPE is applied before K is cached, so the
        // cached prefix is identical to what a recompute would produce —
        // but only when the cached image KV came from the SAME pixels:
        // placeholder ids are pixel-agnostic, so the patch-bytes hash
        // guards cont; a mismatch (or GDEC_VISION_NOCONT) falls back to a
        // full recompute. MTP speculation IS allowed for M-RoPE: verify
        // positions come from the d_ropecs table (ensure_rope_cs handles
        // base=q+1 out-of-range rows with rope_delta_), and the drafter
        // paths rotate with *d_rdelta, so only acceptance rate is at stake.
        if (!g.mrope_grids.empty() && cont &&
            (fnv1a_patches(vpatches) != m.img_hash_ ||
             getenv("GDEC_VISION_NOCONT"))) {
          cont = false;
          n_cached = 0;
        }
        if (!cont && !kvs.dir.empty()) {
          size_t sn = 0;
          // M-RoPE requests: placeholder ids are pixel-agnostic, so a
          // snapshot only qualifies when grids AND patch-bytes hash match
          // (checked inside best_prefix against the KVS2 meta); a mismatch
          // is simply a cache miss and falls to a full recompute.
          const bool is_mrope = !g.mrope_grids.empty();
          const uint64_t req_ih = is_mrope ? fnv1a_patches(vpatches) : 0;
          std::string sp = kvsnap::best_prefix(
              kvs, g.ids, &sn, is_mrope ? &g.mrope_grids : nullptr, req_ih);
          if (!sp.empty()) {
            auto r0 = std::chrono::steady_clock::now();
            if (kvsnap::restore(m, sp, hist)) {  // hist := snapshot tokens
              mcp = hist.size();
              cont = true;
              n_cached = (int)mcp;
              double rs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - r0)
                              .count();
              fprintf(stderr, "%s kvsnap: restored %zu tokens from %s in %.1fs\n",
                      log_ts(), mcp, sp.c_str(), rs);
            }
          }
        }
        if (!cont) m.reset_state();
        // Ngram proposals are request-local. Clear learned suffixes so a
        // previous request cannot collide with this prompt's hash history.
        if (g.drafter == 3 || g.drafter == 4) m.ngram.reset();
        // Vision requests expand their grids in-engine (validated against the
        // prompt's image token runs); plain requests drop stale mrope state
        // (a cont turn skips reset_state).
        if (!g.mrope_grids.empty()) {
          if (!m.set_mrope(g.ids, g.mrope_grids)) {
            fprintf(stderr, "vision: req %lld rejected: grids do not cover "
                    "the image token runs\n", (long long)g.req);
            char db[128];
            snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n",
                     (long long)g.req, g.ids.size());
            alive = send_all(cfd, db);
            continue;
          }
          if (g.vimg > 0 &&
              !m.vision_embeds(g.ids, g.mrope_grids, vpatches, n_cached)) {
            char db[128];
            snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n",
                     (long long)g.req, g.ids.size());
            alive = send_all(cfd, db);
            continue;
          }
          vpatches.clear();
          vpatches.shrink_to_fit();  // ~50 MB/image: release before decode
        } else {
          m.clear_mrope();
        }
        std::unordered_set<int> eos(g.eos.begin(), g.eos.end());
        HostSampler smp;
        std::vector<float> lbuf;
        if (g.sample) {
          smp.temp = g.temp;
          smp.top_p = g.top_p;
          smp.min_p = g.min_p;
          smp.top_k = g.top_k;
          smp.rng = g.seed;
          smp.presence = g.presence;
          smp.frequency = g.frequency;
          smp.bias = g.bias;
          // penalties count GENERATED tokens only: hist starts empty and
          // sample() records each emitted token
        }
        auto t0 = std::chrono::steady_clock::now();
        int cur;
        float cur_lp = 0.0f;
        // MTP speculation is available for greedy and rejection-sampled
        // requests when its KV can be made complete. kvsnap restores lack MTP
        // KV for the prompt, so those requests stay on plain decode.
        static const int spec_gamma = std::max(
            1, std::min(8, getenv("GDEC_SPEC_GAMMA")
                               ? atoi(getenv("GDEC_SPEC_GAMMA"))
                               : 3));
        if (g.drafter == 1 && !m.mtp_avail) {
          char db[128];
          snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n",
                   (long long)g.req, g.ids.size());
          alive = send_all(cfd, db);
          continue;
        }
        const bool use_ngram = g.drafter == 3 && !g.sample;
        // Chain mode (llama.cpp-style fallback): ngram drafts every round it
        // can, MTP takes the rounds where the table has nothing. Sampled
        // requests run spec_loop_chain_sample (ngram rounds re-draw from the
        // target distribution; GDEC_NOSPEC_SAMPLE forces plain serial, same
        // as use_spec). The MTP leg requires a complete MTP KV: a fresh
        // prefill always has one; a cont/kvsnap turn only if the previous
        // request left MTP live.
        const bool use_chain = g.drafter == 4 &&
                               (!g.sample ||
                                getenv("GDEC_NOSPEC_SAMPLE") == nullptr);
        const bool chain_mtp = use_chain && m.mtp_avail &&
                               (!cont || mtp_live) &&
                               getenv("GDEC_NOSPEC") == nullptr;
        const bool use_spec = !use_ngram && !use_chain && m.mtp_avail &&
                              g.drafter != 0 && g.drafter != 3 &&
                              g.drafter != 4 &&
                              (!cont || mtp_live) &&
                              (!g.sample ||
                               getenv("GDEC_NOSPEC_SAMPLE") == nullptr) &&
                              getenv("GDEC_NOSPEC") == nullptr;
        m.last_spec_rounds = m.last_spec_commit = m.last_spec_rollbacks = 0;
        try {
          if (use_spec || chain_mtp) {
            m.mtp_tap_capture = true;  // prefill_batch ingests MTP per chunk
            if (cont) {
              // Boundary: MTP position mcp-1 pairs the live d_R (stream at
              // the last ingested position) with the first new token.
              std::vector<int> bt{g.ids[mcp]};
              m.mtp_ingest_b(m.d_R, bt, 1, (int)mcp - 1);
            }
          }
          // Frontend-hinted snapshot points (SNAP/SNAP2) that fall inside the
          // uncached range: pause the prefill there, snapshot, resume. The
          // next turn re-renders history as TEXT whose ids match the prompt
          // only up to these boundaries (the assistant opener and generated
          // text retokenize differently), so they are the cuts that hit.
          std::vector<size_t> cuts;
          if (!kvs.dir.empty())  // M-RoPE states snapshot fine (KVS2 meta)
            for (size_t c : {g.snap2, g.snap})
              if (c > mcp && c < g.ids.size() && c >= kvs.min_tokens)
                cuts.push_back(c);
          std::sort(cuts.begin(), cuts.end());
          if (!cont) hist.clear();
          size_t done_upto = cont ? mcp : 0;
          // Arm the 1s prefill progress reporter for the prompt ingest only
          // (spec-verify batches never see it armed). prog_base accumulates
          // across the kvsnap cut segments inside prefill_chunk.
          m.prog_total = (int)(g.ids.size() - done_upto);
          m.prog_base = 0;
          m.prog_last_est = 0;
          m.prog_req = g.req;
          m.prog_t0 = m.prog_last = std::chrono::steady_clock::now();
          // live spec counters reset per request, or a serial decode after a
          // speculative one would report the previous request's acceptance
          m.live_spec_rounds = m.live_spec_commits = m.live_spec_proposed = 0;
          for (size_t c : cuts) {
            std::vector<int> seg(g.ids.begin() + done_upto, g.ids.begin() + c);
            cur = m.prefill_batch(seg, (int)done_upto);
            hist.assign(g.ids.begin(), g.ids.begin() + c);
            kvsnap::save(m, hist, kvs);
            done_upto = c;
          }
          std::vector<int> seg(g.ids.begin() + done_upto, g.ids.end());
          cur = m.prefill_batch(seg, (int)done_upto);
          m.prog_total = 0;
          m.mtp_tap_capture = false;  // decode/verify must not capture taps
          hist = g.ids;  // prompt fully ingested on both paths
          if (use_spec || chain_mtp) mtp_live = true;  // MTP KV complete & in lockstep
          if (g.sample) {  // d_logits still holds the last position's logits
            m.get_logits(lbuf);
            cur = smp.sample(lbuf, &cur_lp);
          }
        } catch (const std::exception& e) {
          m.prog_total = 0;
          m.mtp_tap_capture = false;
          fprintf(stderr, "serve: prefill failed: %s\n", e.what());
          hist.clear();  // partial prefill left unknown state: force fresh
          char db[128];
          snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n", (long long)g.req,
                   g.ids.size());
          alive = send_all(cfd, db);
          continue;
        }
        double pre_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        // Prompt-boundary snapshot: chat templates re-render history as TEXT,
        // so the next turn's ids match this turn's PROMPT ids exactly but
        // diverge from the generated ids (BPE boundary effects). Caching the
        // end-of-prefill state is what makes multi-turn reuse actually hit.
        // save() dedups by content hash, so repeats cost nothing.
        // M-RoPE turns are cached too: the KVS2 meta records grids and the
        // patch-bytes hash so a different image with identical placeholder
        // ids can never restore this state.
        if (!kvs.dir.empty() && hist.size() >= kvs.min_tokens)
          kvsnap::save(m, hist, kvs);
        auto d0 = std::chrono::steady_clock::now();
        int ngen = 0;
        bool cancel = false, dead = false;
        std::string reason = "length";
        // mid-decode control poll: X cancels this request, PING stays alive
        auto poll_ctrl = [&]() {
          std::string cl;
          while (lb.try_readline(cl)) {
            std::istringstream cs(cl);
            std::string cv;
            int64_t cr = 0;
            cs >> cv;
            if (cv == "X" && (cs >> cr) && cr == g.req)
              cancel = true;
            else if (cv == "PING")
              send_all(cfd, "PONG\n");
            // other verbs mid-request: drop (batch-1)
          }
        };
        // 1s-cadence decode stats (2026-09-13): instantaneous + average t/s,
        // with the live MTP acceptance the spec loops bump every round.
        // committed drafts = commits - rounds (each round also emits the
        // verify/bonus token), acceptance = committed / proposed.
        auto dl_t0 = std::chrono::steady_clock::now();
        auto dl_last = dl_t0;
        int dl_last_ngen = 0;
        auto dl_tick = [&]() {
          auto now = std::chrono::steady_clock::now();
          double iv = std::chrono::duration<double>(now - dl_last).count();
          if (iv < 1.0) return;
          double el = std::chrono::duration<double>(now - dl_t0).count();
          if (m.live_spec_rounds > 0) {
            int acc_n = m.live_spec_commits - m.live_spec_rounds;
            fprintf(stderr,
                    "%s req %lld | n_decoded = %d | t = %.0f s | tg = %.1f "
                    "tok/s (avg %.1f) | accept %.1f%% (%d/%d)\n",
                    log_ts(), (long long)g.req, ngen, el,
                    (ngen - dl_last_ngen) / iv, el > 0 ? ngen / el : 0.0,
                    m.live_spec_proposed
                        ? 100.0 * acc_n / m.live_spec_proposed
                        : 0.0,
                    acc_n, m.live_spec_proposed);
          } else {
            fprintf(stderr,
                    "%s req %lld | n_decoded = %d | t = %.0f s | tg = %.1f "
                    "tok/s (avg %.1f)\n",
                    log_ts(), (long long)g.req, ngen, el,
                    (ngen - dl_last_ngen) / iv, el > 0 ? ngen / el : 0.0);
          }
          dl_last = now;
          dl_last_ngen = ngen;
        };
        if (use_ngram || use_chain) {
          // chain with a live MTP keeps it live (the loop patches its KV
          // every ngram round); pure/degraded ngram is trunk-only decode and
          // invalidates the previous MTP state.
          const bool chain = use_chain && chain_mtp;
          if (!chain) mtp_live = false;
          if (!m.ple_on) m.chist.assign(g.ids.begin(), g.ids.end());
          if (eos.count(cur)) { reason="done"; }
          else {
            char tb0[96];
            int tl0 = g.sample && g.logprobs
                          ? snprintf(tb0, sizeof tb0, "T %lld %d %.6f\n",
                                     (long long)g.req, cur, (double)cur_lp)
                          : snprintf(tb0, sizeof tb0, "T %lld %d\n",
                                     (long long)g.req, cur);
            if(!send_all(cfd,std::string(tb0,tl0))){dead=true;} else ngen++;
          }
          auto emit_ng = [&](int tok) -> bool {
            if (ngen >= g.max_tokens) return false;
            if (eos.count(tok)) { reason="done"; return false; }
            char tb2[96]; int tl2=snprintf(tb2,sizeof tb2,"T %lld %d\n",(long long)g.req,tok);
            if(!send_all(cfd,std::string(tb2,tl2))){dead=true;return false;} ngen++; poll_ctrl(); dl_tick(); return !cancel;
          };
          auto emit_ng_smp = [&](int tok, float lp) -> bool {
            if (ngen >= g.max_tokens) return false;
            if (eos.count(tok)) { reason="done"; return false; }
            char tb2[96];
            int tl2 = g.logprobs
                          ? snprintf(tb2, sizeof tb2, "T %lld %d %.6f\n",
                                     (long long)g.req, tok, (double)lp)
                          : snprintf(tb2, sizeof tb2, "T %lld %d\n",
                                     (long long)g.req, tok);
            if(!send_all(cfd,std::string(tb2,tl2))){dead=true;return false;} ngen++; poll_ctrl(); dl_tick(); return !cancel;
          };
          if (!dead && !eos.count(cur)) {
            if (g.sample)
              m.spec_loop_chain_sample(cur, g.max_tokens - 1, spec_gamma, smp,
                                       chain_mtp, emit_ng_smp);
            else if (chain) m.spec_loop_chain(cur, g.max_tokens - 1, spec_gamma, emit_ng);
            else m.spec_loop_ngram(cur,g.max_tokens-1,m.ngram.n_max,emit_ng);
          }
          hist.assign(m.chist.begin(),m.chist.end());
        } else if (use_spec) {
          // Stream each token as the spec loop commits it. eos/cancel/length
          // return false, rolling the round back so trunk and MTP stop at the
          // last token actually sent to the client.
          if (eos.count(cur)) {
            reason = "done";
          } else {
            char tb[96];
            int tl = g.sample && g.logprobs
                         ? snprintf(tb, sizeof tb, "T %lld %d %.6f\n",
                                    (long long)g.req, cur, (double)cur_lp)
                         : snprintf(tb, sizeof tb, "T %lld %d\n",
                                    (long long)g.req, cur);
            if (!send_all(cfd, std::string(tb, tl))) {
              dead = true;
            } else {
              ngen++;
              if (g.sample) {
                auto emit = [&](int tok, float lp) -> bool {
                  if (ngen >= g.max_tokens) return false;
                  if (eos.count(tok)) {
                    reason = "done";
                    return false;
                  }
                  char tb2[96];
                  int tl2 = g.logprobs
                                ? snprintf(tb2, sizeof tb2,
                                           "T %lld %d %.6f\n",
                                           (long long)g.req, tok, (double)lp)
                                : snprintf(tb2, sizeof tb2, "T %lld %d\n",
                                           (long long)g.req, tok);
                  if (!send_all(cfd, std::string(tb2, tl2))) {
                    dead = true;
                    return false;
                  }
                  ngen++;
                  poll_ctrl();
                  dl_tick();
                  return !cancel;
                };
                m.spec_loop_sample(cur, g.max_tokens - 1, spec_gamma, smp,
                                   emit);
              } else {
                auto emit = [&](int tok) -> bool {
                  if (ngen >= g.max_tokens) return false;
                  if (eos.count(tok)) {
                    reason = "done";
                    return false;
                  }
                  char tb2[96];
                  int tl2 = snprintf(tb2, sizeof tb2, "T %lld %d\n",
                                     (long long)g.req, tok);
                  if (!send_all(cfd, std::string(tb2, tl2))) {
                    dead = true;
                    return false;
                  }
                  ngen++;
                  poll_ctrl();
                  dl_tick();
                  return !cancel;
                };
                m.spec_loop(cur, g.max_tokens - 1, spec_gamma, emit);
              }
              // Trunk ingested exactly chist tokens; hist must match for the
              // next turn's KV-reuse prefix check.
              hist.assign(m.chist.begin(), m.chist.end());
            }
          }
        } else {
          mtp_live = false;  // plain decode leaves the MTP layer stale
        }
        if (!use_spec && !use_ngram && !use_chain)
        for (int i = 0; i < g.max_tokens; i++) {
          if (eos.count(cur)) {
            reason = "done";
            break;
          }
          char tb[96];
          int tl = g.logprobs
                       ? snprintf(tb, sizeof tb, "T %lld %d %.6f\n",
                                  (long long)g.req, cur, (double)cur_lp)
                       : snprintf(tb, sizeof tb, "T %lld %d\n",
                                  (long long)g.req, cur);
          if (!send_all(cfd, std::string(tb, tl))) {
            dead = true;
            break;
          }
          ngen++;
          if (i + 1 < g.max_tokens) {
            int fed = cur;  // the token forward() ingests (pre-sampling)
            cur = m.forward(fed);
            hist.push_back(fed);
            if (g.sample) {
              m.get_logits(lbuf);
              cur = smp.sample(lbuf, &cur_lp);
            }
          }
          poll_ctrl();
          dl_tick();
          if (cancel) break;
        }
        if (dead) break;
        double dec_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - d0)
                            .count();
        char db[192];
        if (cancel)
          snprintf(db, sizeof db, "D %lld cancel 0 0 0.0 0.0\n", (long long)g.req);
        else if ((use_spec || use_ngram || use_chain) && m.last_spec_rounds > 0)
          snprintf(db, sizeof db,
                   "D %lld %s %zu %d %.1f %.1f %d %d %d %d %d\n",
                   (long long)g.req, reason.c_str(), g.ids.size(), ngen, pre_ms,
                   ngen > 1 ? dec_ms : 0.0,
                   use_chain ? (chain_mtp ? 4 : 3) : (use_ngram ? 3 : 1),
                   m.last_spec_rounds,
                   m.last_spec_commit, n_cached, m.last_spec_proposed);
        else
          // trailing "0 0 0 <n_cached> 0": drafter/rounds/commit placeholders
          // so n_cached lands in the protocol's cache column (parts[10]);
          // the last 0 is the proposed-drafts column (parts[11])
          snprintf(db, sizeof db, "D %lld %s %zu %d %.1f %.1f 0 0 0 %d 0\n",
                   (long long)g.req, reason.c_str(), g.ids.size(), ngen, pre_ms,
                   ngen > 1 ? dec_ms : 0.0, n_cached);
        alive = send_all(cfd, db);
        {
          // llama.cpp-style per-request timing block (human-facing; the wire
          // D line above stays the machine-readable record).
          size_t n_proc =
              g.ids.size() - (n_cached > 0 ? (size_t)n_cached : 0);
          fprintf(stderr,
                  "%s req %lld | prompt eval = %.1f ms / %zu tokens "
                  "(%.2f ms/token, %.1f tok/s)%s\n",
                  log_ts(), (long long)g.req, pre_ms, n_proc,
                  n_proc ? pre_ms / n_proc : 0.0,
                  pre_ms > 0 ? n_proc / (pre_ms / 1000.0) : 0.0,
                  n_cached ? " | cached" : "");
          if (ngen > 0)
            fprintf(stderr,
                    "%s req %lld | decode eval = %.1f ms / %d tokens "
                    "(%.2f ms/token, %.1f tok/s) | %s\n",
                    log_ts(), (long long)g.req, dec_ms, ngen,
                    dec_ms > 0 ? dec_ms / ngen : 0.0,
                    dec_ms > 0 ? ngen / (dec_ms / 1000.0) : 0.0,
                    cancel ? "cancel" : reason.c_str());
          else
            fprintf(stderr, "%s req %lld | decode eval = no tokens | %s\n",
                    log_ts(), (long long)g.req,
                    cancel ? "cancel" : reason.c_str());
          if ((use_spec || use_ngram || use_chain) && m.last_spec_rounds > 0) {
            int acc_n = m.live_spec_commits - m.live_spec_rounds;
            fprintf(stderr,
                    "%s req %lld | draft acceptance = %.3f (%d accepted / %d "
                    "proposed), mean len = %.2f, drafter %s\n",
                    log_ts(), (long long)g.req,
                    m.live_spec_proposed
                        ? (double)acc_n / m.live_spec_proposed
                        : 0.0,
                    acc_n, m.live_spec_proposed,
                    m.last_spec_rounds
                        ? (double)m.live_spec_commits / m.last_spec_rounds
                        : 0.0,
                    use_chain ? (chain_mtp ? "chain" : "ngram")
                              : (use_ngram ? "ngram" : "mtp"));
          }
        }
        // SSD prompt cache: snapshot the end-of-turn state for future
        // sessions. Cancelled turns are valid too -- hist tracks exactly the
        // ingested tokens, and short states are skipped inside save().
        // M-RoPE states carry the KVS2 grids/img-hash meta (see above).
        if (!kvs.dir.empty() && hist.size() >= kvs.min_tokens)
          kvsnap::save(m, hist, kvs);
      }
      // unknown verbs: drop
    }
    sock_close(cfd);
    fprintf(stderr, "%s serve: client disconnected\n", log_ts());
  }
  return 0;
}

}  // namespace serve

// ---------------- main -------------------------------------------------------
int main(int argc, char** argv) {
#ifdef _WIN32
  if (!wsa_init()) {
    fprintf(stderr, "WSAStartup failed\n");
    return 1;
  }
#endif
  if (argc < 2) {
    fprintf(stderr,
            "usage: gdec <base.hgn> [overlay.hgn ...] --tokens 1,2,3 [--gen N] [--dump] "
            "[--no-ple] [--maxctx N]\n"
            "       gdec <base.hgn> [overlay.hgn ...] --serve [--port N] [--maxctx N] "
            "[--no-ple]\n"
            "       M-RoPE: [--mrope-grid t,h,w ...] with --tokens; "
            "--mrope-test K dumps rope tables/decode positions without weights "
            "(GDEC_MROPE_DUMP=<file>)\n"
            "       Vision: --vision-tower <sidecar.hgn> (serve: GEN suffix "
            "VIMG k + binary patch frames); --vision-test PREFIX "
            "--vision-tower S --patches x.npy --mrope-grid t,h,w (offline "
            "ViT parity dumps, no main checkpoint)\n");
    return 2;
  }
  std::string base = argv[1];
  std::vector<std::string> overlays;  // applied in order; later files win
  std::vector<int> tokens;
  int gen = 8, argi = 2, maxctx = 4096;
  std::vector<int> splits;  // --split K1,K2: KV-continuation self-test
  bool dump = false, ple_on = true, show_top = false;
  bool serve_mode = false, maxctx_set = false;
  int port = 8730, mtp_test = 0, spec_gen = 0, gamma = 7;
  // M-RoPE: per-image grid triples (repeatable), rope-only test mode.
  std::vector<std::array<int, 3>> mrope_grids;
  int mrope_test = -1;  // -1 off; else number of decode steps to dump
  // Vision (stage 4): sidecar tower weights; offline ViT parity mode.
  std::string vision_tower, vision_test, vision_patches;
  while (argi < argc && argv[argi][0] != '-') overlays.push_back(argv[argi++]);
  for (; argi < argc; argi++) {
    std::string a = argv[argi];
    auto next = [&]() { return argv[++argi]; };
    if (a == "--tokens") {
      char* s = strdup(next());
      for (char* p = strtok(s, ","); p; p = strtok(nullptr, ",")) tokens.push_back(atoi(p));
      free(s);
    } else if (a == "--tokens-file") {
      // one token id per line or comma-separated; for long prompts (ARG_MAX limit)
      std::ifstream f(next());
      std::string line;
      while (std::getline(f, line)) {
        char* s = strdup(line.c_str());
        for (char* p = strtok(s, ", \t"); p; p = strtok(nullptr, ", \t"))
          tokens.push_back(atoi(p));
        free(s);
      }
    } else if (a == "--gen")
      gen = atoi(next());
    else if (a == "--split") {
      char* s = strdup(next());
      for (char* p = strtok(s, ","); p; p = strtok(nullptr, ","))
        splits.push_back(atoi(p));
      free(s);
    } else if (a == "--dump")
      dump = true;
    else if (a == "--no-ple")
      ple_on = false;
    else if (a == "--show")
      show_top = true;
    else if (a == "--serve")
      serve_mode = true;
    else if (a == "--port")
      port = atoi(next());
    else if (a == "--maxctx") {
      maxctx = atoi(next());
      maxctx_set = true;
    } else if (a == "--mtp-test")
      mtp_test = atoi(next());
    else if (a == "--spec-gen")
      spec_gen = atoi(next());
    else if (a == "--gamma")
      gamma = atoi(next());
    else if (a == "--mrope-grid") {
      char* s = strdup(next());
      std::array<int, 3> g{0, 0, 0};
      int k = 0;
      for (char* p = strtok(s, ","); p && k < 3; p = strtok(nullptr, ","))
        g[k++] = atoi(p);
      free(s);
      if (k != 3) {
        fprintf(stderr, "--mrope-grid expects t,h,w\n");
        return 2;
      }
      mrope_grids.push_back(g);
    } else if (a == "--mrope-test")
      mrope_test = atoi(next());
    else if (a == "--vision-tower")
      vision_tower = next();
    else if (a == "--vision-test")
      vision_test = next();
    else if (a == "--patches")
      vision_patches = next();
  }
  if (!vision_test.empty()) {
    // Offline ViT parity mode (stage 4a): runs the INTEGRATED vision tower
    // standalone — no main checkpoint, no arena — and dumps vit.cpp-
    // compatible layers under <prefix>.layers/ for tools/vit_cmp.py.
    //   gdec <ignored-> --vision-test PREFIX --vision-tower S.hgn
    //        --patches x.npy --mrope-grid t,h,w
    if (vision_tower.empty() || vision_patches.empty() ||
        mrope_grids.empty()) {
      fprintf(stderr, "--vision-test needs --vision-tower, --patches and "
              "--mrope-grid t,h,w\n");
      return 2;
    }
    auto g = mrope_grids[0];
    VitNpy patches = vit_npy_read(vision_patches.c_str());
    if (patches.shape.size() != 2 || patches.shape[1] != 1536)
      throw std::runtime_error("patches npy must be <f4 (P,1536)");
    int P = (int)patches.shape[0];
    if (g[0] != 1 || g[1] * g[2] != P) {
      fprintf(stderr, "--vision-test: grid %d,%d,%d vs patches P=%d\n", g[0],
              g[1], g[2], P);
      return 2;
    }
    CK(hipSetDevice(0));
    hipblasLtHandle_t vlth;
    rocblas_handle vrbh;
    hipblasLtCreate(&vlth);
    rocblas_create_handle(&vrbh);
    void* vws = nullptr;
    const size_t vws_bytes = (size_t)64 << 20;
    CK(hipMalloc(&vws, vws_bytes));
    VisionTower vt;
    vt.lth = vlth;
    vt.rbh = vrbh;
    vt.d_ltws = vws;
    vt.lt_ws = vws_bytes;
    auto t0 = std::chrono::steady_clock::now();
    vt.load(vision_tower.c_str());
    int M = vt.forward((const float*)patches.data.data(), P, g[1], g[2],
                       vision_test.c_str());
    CK(hipStreamSynchronize(g_str));
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    if (M < 0) return 1;
    fprintf(stderr, "vision-test: P=%d M=%d total %.0f ms (load+forward, "
            "dumps at %s.layers/)\n", P, M, ms, vision_test.c_str());
    return 0;
  }
  if (mrope_test >= 0) {
    // M-RoPE rope-only verification (stage 1b/1c): drives the cos/sin table
    // fill and the decode rope position directly — no weight arena, no layer
    // forward. A second full engine cannot run next to the live 256K service
    // (the global pinned-page ceiling rejects a second 65 GiB expert
    // registration, tools/probe_reg3.cu),
    // and none of the rope machinery touches weights. Use with
    // GDEC_MROPE_DUMP=<file> and compare via tools/cmp_mrope.py.
    Checkpoint ckpt(base.c_str());
    for (const auto& o : overlays) ckpt.add_overlay(o.c_str());
    if (tokens.empty()) {
      fprintf(stderr, "need --tokens\n");
      return 2;
    }
    GpuModel m(ckpt, maxctx);
    if (!mrope_grids.empty() && !m.set_mrope(tokens, mrope_grids)) {
      fprintf(stderr, "mrope: grids do not match the image token runs\n");
      return 1;
    }
    const int N = (int)tokens.size();
    ensure_rope_tab(g_cfg.rope_theta, 64);
    m.ensure_rope_cs(0, N);
    for (int s = 0; s < mrope_test; s++) {
      int p = N + s;
      CK(hipMemcpy(m.d_pos, &p, 4, hipMemcpyHostToDevice));
      m.dump_decode_step(p);
    }
    // k_rope decode kernel probe: a deterministic vector rotated in place at
    // position N (+ rope_delta); bitwise-comparable across binaries.
    if (FILE* f = m.mrope_dump()) {
      std::vector<float> hv(24 * 256);
      for (size_t i = 0; i < hv.size(); i++)
        hv[i] = (float)(int)(((uint32_t)i * 2654435761u) >> 9) / 8388608.f - 1.f;
      float* d_vp = nullptr;
      CK(hipMalloc(&d_vp, hv.size() * 4));
      CK(hipMemcpy(d_vp, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(m.d_pos, &N, 4, hipMemcpyHostToDevice));
      k_rope<<<(24 * 32 + 127) / 128, 128>>>(d_vp, 24, 256, 64, m.d_pos,
                                             g_cfg.rope_theta, m.d_rdelta);
      CK(hipMemcpy(hv.data(), d_vp, hv.size() * 4, hipMemcpyDeviceToHost));
      CK(hipFree(d_vp));
      for (int h = 0; h < 24; h++) {
        fprintf(f, "KROPE %d", h);
        for (int i = 0; i < 64; i++)
          fprintf(f, " %a", (double)hv[(size_t)h * 256 + i]);
        fprintf(f, "\n");
      }
      fflush(f);
    }
    return 0;
  }
  if (serve_mode) {
    if (!maxctx_set) maxctx = 65536;  // service default; CLI one-shot stays 4096
    Checkpoint ckpt(base.c_str());
    ple_uring_open(base.c_str());
    for (const auto& o : overlays) ckpt.add_overlay(o.c_str());
    fprintf(stderr, "loaded %zu tensors (base+%zu overlay)\n", ckpt.tensor_count(),
            overlays.size());
    g_cfg.vocab = (int)ckpt.at("lm_head.weight").dims[0];
    devarena_init(ckpt, maxctx);
    load_arena(ckpt);
    GpuModel m(ckpt, maxctx);
    m.ple_on = ple_on;
    m.build_graphs();
    if (!vision_tower.empty()) m.load_vision(vision_tower.c_str());
    if (!getenv("GDEC_NOWARMUP")) {
      // one dummy chunk moves first-call costs (Tensile library load,
      // hipBLASLt algo selection, kernel module loads) out of the first
      // served prefill; reset_state is the KV-split self-test path, proven
      // output-identical to a fresh model.
      std::vector<int> warm((size_t)m.maxbatch, 0);
      m.prefill_batch(warm);
      m.reset_state();
    }
    return serve::run(m, port);
  }
  if (tokens.empty()) {
    fprintf(stderr, "need --tokens\n");
    return 2;
  }
  int need = mtp_test > 0 ? mtp_test : spec_gen > 0 ? spec_gen + gamma : gen;
  if (gamma < 1) gamma = 1;
  if (gamma > 8) gamma = 8;  // spec workspace holds at most 8 draft rows
  if ((int)tokens.size() + need > maxctx) {
    fprintf(stderr, "tokens+gen exceed maxctx\n");
    return 2;
  }

  Checkpoint ckpt(base.c_str());
  ple_uring_open(base.c_str());
  for (const auto& o : overlays) ckpt.add_overlay(o.c_str());
  fprintf(stderr, "loaded %zu tensors (base+%zu overlay)\n", ckpt.tensor_count(),
          overlays.size());
  g_cfg.vocab = (int)ckpt.at("lm_head.weight").dims[0];

  devarena_init(ckpt, maxctx);
  load_arena(ckpt);

  GpuModel m(ckpt, maxctx);
  m.ple_on = ple_on;
  m.build_graphs();
  if (mtp_test > 0 || spec_gen > 0) {
    if (mtp_test > 0 && !mrope_grids.empty()) {
      fprintf(stderr, "mrope: mtp-test not supported (acceptance harness "
              "measures text only)\n");
      return 1;
    }
    if (!m.mtp_avail) {
      fprintf(stderr, "mtp: no usable mtp.* weights in checkpoint\n");
      return 1;
    }
    m.mtp_tap_capture = true;  // prefill_chunk records per-row trunk streams
  }

  std::vector<int> all = tokens;
  std::vector<float> logits;

  if (!getenv("GDEC_NOWARMUP") && mtp_test == 0 && spec_gen == 0) {
    // warmup: one dummy chunk prefill, then reset (see serve path above).
    // Skipped for mtp/spec modes: tap capture would ingest warmup rows.
    std::vector<int> warm((size_t)m.maxbatch, 0);
    m.prefill_batch(warm);
    m.reset_state();
  }
  if (!mrope_grids.empty() && !m.set_mrope(tokens, mrope_grids)) {
    fprintf(stderr, "mrope: grids do not match the image token runs\n");
    return 1;
  }

  auto dump_pos = [&](const std::vector<float>& lg, int i, int next_tok) {
    int V = g_cfg.vocab;
    double mx = -1e300;
    for (int j = 0; j < V; j++) mx = std::max(mx, (double)lg[j]);
    double se = 0;
    for (int j = 0; j < V; j++) se += exp(lg[j] - mx);
    double lse = mx + log(se);
    std::vector<int> idx(V);
    for (int j = 0; j < V; j++) idx[j] = j;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    printf("pos\t%d\t", i);
    for (int k = 0; k < 5; k++)
      printf("%s%d,%.5f", k ? ";" : "", idx[k], lg[idx[k]] - (float)lse);
    if (next_tok >= 0) printf("\tnext=%.5f", lg[next_tok] - (float)lse);
    printf("\n");
    fflush(stdout);
  };

  int next_id = -1;
  if (getenv("GDEC_NOPREFILLBATCH")) {
    for (size_t i = 0; i < tokens.size(); i++) {
      next_id = m.forward(tokens[i]);
      if (dump) {
        m.get_logits(logits);
        dump_pos(logits, (int)i, i + 1 < tokens.size() ? tokens[i + 1] : -1);
      }
      fprintf(stderr, "\rprefill %zu/%zu", i + 1, tokens.size());
      fflush(stderr);
    }
    fprintf(stderr, "\n");
  } else {
    next_id = m.prefill_batch(tokens);
    if (dump) {  // per-position prefill dump unavailable; dump last position
      m.get_logits(logits);
      dump_pos(logits, (int)tokens.size() - 1, -1);
    }
  }

  if (mtp_test > 0) {
    m.mtp_accept_test(tokens, next_id, mtp_test);
    return 0;
  }
  if (spec_gen > 0) {
    m.mtp_tap_capture = false;  // verify/re-prefill must NOT capture taps
    // (MTP prompt ingest already happened per-chunk inside prefill_batch.)
    if (getenv("GDEC_NOPREFILLBATCH")) {
      fprintf(stderr, "spec-gen: GDEC_NOPREFILLBATCH unsupported (no tap "
              "capture without prefill_batch)\n");
      return 1;
    }
    std::vector<int> spec_out = m.spec_loop(next_id, spec_gen, gamma);
    printf("ids:");
    for (int t : tokens) printf(" %d", t);
    printf(" %d", next_id);
    for (int t : spec_out) printf(" %d", t);
    printf("\n");
    return 0;
  }

  auto show = [&](const std::vector<float>& lg, int step, int chosen) {
    int V = g_cfg.vocab;
    double mx = -1e300;
    for (int i = 0; i < V; i++) mx = std::max(mx, (double)lg[i]);
    double se = 0;
    for (int i = 0; i < V; i++) se += exp(lg[i] - mx);
    double lse = mx + log(se);
    std::vector<int> idx(V);
    for (int i = 0; i < V; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    printf("step %d chosen=%d top5:", step, chosen);
    for (int i = 0; i < 5; i++) printf(" [%d]=%.4f", idx[i], lg[idx[i]] - (float)lse);
    printf("\n");
    fflush(stdout);
  };

  int cur = next_id;
  auto loop_t0 = std::chrono::steady_clock::now();
  for (int g = 0; g < gen; g++) {
    if (dump) {
      m.get_logits(logits);
      dump_pos(logits, (int)tokens.size() + g, -1);
    } else if (show_top) {
      m.get_logits(logits);
      show(logits, g, cur);
    } else {
      printf("step %d chosen=%d\n", g, cur);
      fflush(stdout);
    }
    all.push_back(cur);
    if (g + 1 < gen) cur = m.forward(cur);
  }
  printf("ids:");
  for (int t : all) printf(" %d", t);
  printf("\n");
  if (gen > 1) {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          loop_t0)
                    .count();
    fprintf(stderr, "decode: %d tokens in %.0f ms = %.1f tok/s (%.1f ms/token)\n", gen - 1,
            ms, (gen - 1) * 1000.0 / ms, ms / (gen - 1));
  }

  // KV-continuation self-test: for each split K, reset, prefill [:K], continue
  // [K:] from live state, then decode — ids must match the one-shot baseline.
  for (int K : splits) {
    if (K <= 0 || K >= (int)tokens.size()) {
      fprintf(stderr, "split %d out of range\n", K);
      continue;
    }
    m.reset_state();
    std::vector<int> head(tokens.begin(), tokens.begin() + K);
    std::vector<int> tail(tokens.begin() + K, tokens.end());
    m.prefill_batch(head, 0);
    int cid = m.prefill_batch(tail, K);
    std::vector<int> sall = tokens;
    sall.push_back(cid);
    for (int g = 1; g < gen; g++) {
      cid = m.forward(cid);
      sall.push_back(cid);
    }
    bool ok = sall == all;
    printf("split %d: %s\n", K, ok ? "MATCH" : "DIFF");
    if (!ok) {
      for (size_t i = tokens.size(); i < all.size() && i < sall.size(); i++)
        if (all[i] != sall[i]) {
          printf("  first diff at gen step %zu: base=%d split=%d\n",
                 i - tokens.size(), all[i], sall[i]);
          break;
        }
    }
    fflush(stdout);
  }
  m.perf_report();
  return 0;
}
