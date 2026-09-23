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

// ---- Paged QSA KV (memory model Phase A) ----
// The trunk's QSA K/V rows and pooled indexer keys live in per-layer page
// pools; a page holds KV_PAGE consecutive logical tokens (= KV_PAGE/4 indexer
// blocks) and every layer uses the same physical page id for a given logical
// page. ptab[logical page] = physical page. ptab == nullptr is the legacy
// contiguous layout (identity), so every kernel below keeps its old addressing
// bit-for-bit when paging is off. Only addresses change, never values or
// accumulation order, so paged runs are bitwise identical to contiguous ones.
constexpr int KV_PAGE_SHIFT = 8;
constexpr int KV_PAGE = 1 << KV_PAGE_SHIFT;          // 256 tokens
constexpr int KV_PAGE_BLOCKS = KV_PAGE / 4;          // 64 indexer blocks
constexpr int KV_PAGE_BLOCK_SHIFT = KV_PAGE_SHIFT - 2;
__device__ __forceinline__ int kv_row(const int* __restrict__ ptab, int t) {
  return ptab ? (ptab[t >> KV_PAGE_SHIFT] << KV_PAGE_SHIFT) | (t & (KV_PAGE - 1))
              : t;
}
__device__ __forceinline__ int ik_blk(const int* __restrict__ ptab, int b) {
  return ptab ? (ptab[b >> KV_PAGE_BLOCK_SHIFT] << KV_PAGE_BLOCK_SHIFT) |
                    (b & (KV_PAGE_BLOCKS - 1))
              : b;
}

// ---- A2: host-side page allocator + per-sequence page table ----
// KvPagePool hands out physical pages [0, npages). Physical page `npages` is
// the GUARD page: every unmapped logical page points at it, it is never
// allocated and it stays zero, so a stray read sees finite data and a stray
// write is detectable (GpuModel::kv_guard_dirty). Refcounts are 0/1 while
// there is one live sequence (A2); A3 shares pages between the live sequence
// and RAM checkpoints (ref > 1).
//   order 1: lowest free page first -> a fresh sequence maps identity-like,
//            long physically contiguous runs (production order).
//   order 2: self-test -> scrambled initial order and FIFO reuse, so logical
//            pages land on non-monotonic physical pages and consecutive
//            sequences reuse pages that still hold the previous sequence's
//            data. Output must stay bitwise identical to order 1 / unpaged.
struct KvPagePool {
  int npages = 0, order = 1;
  std::deque<int> free_;
  std::vector<uint16_t> ref;
  void init(int n, int ord) {
    npages = n;
    order = ord;
    free_.clear();
    ref.assign((size_t)n, 0);
    if (ord == 2) {
      // p_i = (i * s + n / 3) mod n with gcd(s, n) == 1 -> a permutation
      int s = n / 2 + n / 8 + 1;
      auto gcd = [](int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; };
      while (n > 1 && gcd(s, n) != 1) s++;
      for (int i = 0; i < n; i++)
        free_.push_back((int)(((long long)i * s + n / 3) % n));
    } else {
      for (int i = 0; i < n; i++) free_.push_back(i);
    }
  }
  int guard() const { return npages; }
  int nfree() const { return (int)free_.size(); }
  int alloc() {  // -1 when exhausted
    if (free_.empty()) return -1;
    const int p = free_.front();
    free_.pop_front();
    ref[(size_t)p] = 1;
    return p;
  }
  void addref(int p) {
    if (p < 0 || p >= npages || ref[(size_t)p] == 0 || ref[(size_t)p] == 0xffff) {
      fprintf(stderr, "[kvpage] FATAL: addref(%d) on a free/invalid page\n", p);
      abort();
    }
    ref[(size_t)p]++;
  }
  // Drop one reference; true when the page went back to the free list.
  bool unref(int p) {
    if (p < 0 || p >= npages || ref[(size_t)p] == 0) {
      fprintf(stderr, "[kvpage] FATAL: unref(%d) on a free/invalid page\n", p);
      abort();
    }
    if (--ref[(size_t)p]) return false;
    if (order == 2)
      free_.push_back(p);
    else
      free_.insert(std::lower_bound(free_.begin(), free_.end(), p), p);
    return true;
  }
};

// One sequence's logical -> physical page table. Mapped logical pages are
// always a prefix [0, mapped); tab[lp] == guard for lp >= mapped. Writes only
// ever happen at rows >= the live pos, so growing / shrinking the tail is all
// a single sequence needs.
struct KvSeqTable {
  std::vector<int> tab;
  int mapped = 0;
  void init(int lpages, int guard) {
    tab.assign((size_t)lpages, guard);
    mapped = 0;
  }
  int rows() const { return mapped * KV_PAGE; }
  // Back rows [0, end). false: end beyond the table or pool exhausted (pages
  // mapped so far stay mapped; a later trim releases them).
  bool reserve(KvPagePool& pool, int end) {
    const int need = (end + KV_PAGE - 1) / KV_PAGE;
    if (need > (int)tab.size()) return false;
    while (mapped < need) {
      const int p = pool.alloc();
      if (p < 0) return false;
      tab[(size_t)mapped++] = p;
    }
    return true;
  }
  // Keep only the pages holding rows < pos; returns the number unmapped.
  int trim(KvPagePool& pool, int pos) {
    const int keep = std::max(0, (pos + KV_PAGE - 1) / KV_PAGE);
    int n = 0;
    while (mapped > keep) {
      pool.unref(tab[(size_t)--mapped]);
      tab[(size_t)mapped] = pool.guard();
      n++;
    }
    return n;
  }
};

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
                              const float2* cs_tab = nullptr,
                              const int* __restrict__ ptab = nullptr) {
  int b = blockIdx.x + (base / 4), d = threadIdx.x;  // absolute block id
  float v = 0.f;
  for (int j = 0; j < 4; j++) {
    int lt = b * 4 + j - base;  // batch-local row; < 0 -> history
    v += lt >= 0 ? proj[(size_t)lt * 640 + 512 + d]
                 : ring[(size_t)((b * 4 + j) % 4) * 128 + d];
  }
  index_norm_rope(v * 0.25f, norm, keys + (size_t)ik_blk(ptab, b) * 128, b * 4,
                  eps, theta, cs_tab, base);
}

__global__ void k_index_ring(const float* proj, float* ring, int P, int base = 0) {
  int t = max(0, P - 4) + (int)blockIdx.x;
  if (t < P)
    ring[((base + t) % 4) * 128 + threadIdx.x] = proj[(size_t)t * 640 + 512 + threadIdx.x];
}

__global__ void k_index_append(const float* proj, const float* norm, float* ring,
                                float* keys, const int* pos, float eps, double theta,
                                const int* __restrict__ d_rdelta = nullptr,
                                const int* __restrict__ ptab = nullptr) {
  int t = *pos, d = threadIdx.x;
  ring[(t % 4) * 128 + d] = proj[512 + d];
  __syncthreads();
  if (t % 4 != 3) return;
  float v = 0.f;
  for (int j = 0; j < 4; j++) v += ring[j * 128 + d];
  // d_rdelta (M-RoPE decode): pooled key's rope position is its first token's
  // cache position + request rope_delta (zero for text-only).
  int position = t - 3 + (d_rdelta ? *d_rdelta : 0);
  index_norm_rope(v * 0.25f, norm, keys + (size_t)ik_blk(ptab, t / 4) * 128,
                  position, eps, theta);
}

__global__ void k_index_scores(const float* queries, const float* keys, float* scores,
                                int stride, const int* pos,
                                const int* __restrict__ ptab = nullptr) {
  int b = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  if (*pos < 2051 || b >= (*pos + 1) / 4) return;
  const float* kb = keys + (size_t)ik_blk(ptab, b) * 128;
  for (int h = 0; h < 4; h++) {
    float sum = 0.f;
    for (int d = lane; d < 128; d += warpSize)
      sum += queries[h * 128 + d] * kb[d];
    for (int s = warpSize / 2; s; s >>= 1) sum += __shfl_down_sync(~0ull, sum, s);
    if (!lane) scores[h * stride + b] = sum;
  }
}

// Paged prefill: gather logical blocks [0, nb) of one layer's indexer keys
// into a contiguous scratch so the score sgemm / tiled kernel keep their
// exact shape (bit-identical to the unpaged layout). grid = nb, block = 128.
__global__ void k_ik_gather(const float* __restrict__ src,
                            const int* __restrict__ ptab,
                            float* __restrict__ dst, int nb) {
  int b = blockIdx.x;
  if (b >= nb) return;
  dst[(size_t)b * 128 + threadIdx.x] =
      src[(size_t)ik_blk(ptab, b) * 128 + threadIdx.x];
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

