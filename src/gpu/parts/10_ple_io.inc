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
// bytes per PLE row (160 codes): 160 = hgn fp8 (dtype 10), 90 = GGUF IQ4_NL
// (dtype 11, 5 x 18 B blocks). Staging holds 16 raw rows per token back to back.
static unsigned g_ple_rb = 160;

#ifndef _WIN32
static int g_ple_fd = -1;        // extra O_RDONLY handle on the base .hgn

static void ple_uring_open(const char* path) {
  if (g_ple_fd >= 0) close(g_ple_fd);  // re-pointed at the GGUF shard (dtype 11 table)
  g_ple_fd = -1;
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

  // Pump n preads of g_ple_rb B each: row i -> staging + i*rb, file offset
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
        s->addr = (uint64_t)(uintptr_t)(staging + sub * g_ple_rb);
        s->len = g_ple_rb;
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
        if (c->res != (int)g_ple_rb) {
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

// aggr 10-05: several rings pumped by parallel workers. One pump thread did
// all 16*P preads (16K prompt: 262K reads ≈ 276 ms on a warm page cache — the
// per-op syscall/copy cost, not the disk), while L0 on the GPU takes ~210 ms,
// so the GPU idled at the PLE join. QWENOX_PLE_RINGS (default 8, 1 = old).
static constexpr int kPleRingsMax = 16;
static PleUring g_ple_rings[kPleRingsMax];
static int g_ple_nrings = 0;
static bool g_ple_ring_tried = false;

static bool ple_uring_ready() {
  if (!getenv("QWENOX_PLE_URING")) return false;
  if (g_ple_fd < 0 || !g_ple_off_ok) return false;
  if (!g_ple_ring_tried) {
    g_ple_ring_tried = true;
    int want = 8;
    if (const char* e = getenv("QWENOX_PLE_RINGS")) want = atoi(e);
    want = std::max(1, std::min(kPleRingsMax, want));
    for (int i = 0; i < want; i++) {
      if (!g_ple_rings[i].setup(256)) {  // fixed depth: one structural decision
        if (i == 0)
          fprintf(stderr,
                  "ple_uring: io_uring_setup failed (%s); using madvise path\n",
                  strerror(g_ple_rings[0].err ? g_ple_rings[0].err : EINVAL));
        break;
      }
      g_ple_nrings = i + 1;
    }
  }
  return g_ple_nrings > 0;
}
static int ple_pump_count() { return g_ple_nrings; }
#define PLE_PUMP g_ple_rings[0]
#define PLE_PUMP_I(i) g_ple_rings[i]

#else  // _WIN32 ------------------------------------------------------------

// IOCP 批量读：与 PleUring::run 相同语义——n 个 160B 行从文件偏移
// g_ple_foff + (addrs[i] - table_va) 读入 staging + i*g_ple_rb；任何失败/短读
// 立即 fatal，不做半途回退。
struct PleWin {
  HANDLE fh = INVALID_HANDLE_VALUE;
  HANDLE iocp = nullptr;
  static constexpr size_t DEPTH = 512;
  std::string path;

  PleWin() = default;
  PleWin(const PleWin&) = delete;
  PleWin& operator=(const PleWin&) = delete;
  ~PleWin() { close(); }

  void close() {
    if (fh != INVALID_HANDLE_VALUE) CloseHandle(fh);
    if (iocp) CloseHandle(iocp);
    fh = INVALID_HANDLE_VALUE;
    iocp = nullptr;
  }

  bool ok() const { return iocp != nullptr && g_ple_off_ok; }

  bool setup(const char* filename) {
    close();
    fh = CreateFileA(filename, GENERIC_READ,
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
    path = filename;
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
        BOOL r = ReadFile(fh, staging + sub * g_ple_rb, g_ple_rb, nullptr, ov);
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
      if (!okr || got != g_ple_rb) {
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
static int ple_pump_count() { return 1; }
#define PLE_PUMP g_ple_win
#define PLE_PUMP_I(i) g_ple_win

struct PleWinPrefetch {
  struct Buffer {
    uint8_t* host = nullptr;
    uint8_t* device = nullptr;
    hipEvent_t consumed = nullptr;
    bool used = false;
  };
  PleWin reader;
  std::array<Buffer, 2> buffers;
  std::unique_ptr<ple_prefetch::Queue> queue;
  size_t slot = 0;
  int offset = 0, count = 0, next_count = 0;

  ~PleWinPrefetch() {
    queue.reset();
    for (auto& buffer : buffers) {
      if (buffer.used) CK(hipEventSynchronize(buffer.consumed));
      if (buffer.consumed) CK(hipEventDestroy(buffer.consumed));
      if (buffer.host) CK(hipHostFree(buffer.host));
    }
  }

  bool setup(int capacity, const std::vector<int64_t>& history,
             const std::vector<int>& tokens, ple_prefetch::Hash hash) {
    if (!reader.setup(g_ple_win.path.c_str())) return false;
    for (auto& buffer : buffers) {
      CK(hipHostMalloc((void**)&buffer.host,
                       (size_t)capacity * 16 * hash.row_bytes,
                       hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&buffer.device, buffer.host, 0));
      CK(hipEventCreateWithFlags(&buffer.consumed, hipEventDisableTiming));
    }
    queue = std::make_unique<ple_prefetch::Queue>(
        history, tokens, hash,
        std::array<uint8_t*, 2>{buffers[0].host, buffers[1].host},
        [this, hash](const uint64_t* addrs, uint8_t* staging, size_t rows) {
          reader.run(addrs, staging, hash.table, rows);
        },
        [this](size_t buffer_index) {
          Buffer& buffer = buffers[buffer_index];
          if (buffer.used) CK(hipEventSynchronize(buffer.consumed));
        });
    fprintf(stderr, "ple_prefetch: enabled (2 x %.1f MiB pinned staging)\n",
            (double)capacity * 16 * hash.row_bytes / (1024 * 1024));
    return true;
  }

  void select(int chunk_offset, int chunk_count, int following_count) {
    if (chunk_offset != 0) slot = 1 - slot;
    offset = chunk_offset;
    count = chunk_count;
    next_count = following_count;
  }

  void start_current() {
    if (!queue->pending(slot)) queue->start(slot, offset, count);
  }

  ple_prefetch::Queue::Timing wait() { return queue->wait(slot, offset, count); }

  void mark_consumed(hipStream_t stream) {
    CK(hipEventRecord(buffers[slot].consumed, stream));
    buffers[slot].used = true;
  }

  void start_next() {
    if (next_count > 0) queue->start(1 - slot, offset + count, next_count);
  }
};

#endif  // _WIN32

struct PleWinPrefetch;

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
  // A3: take a reference on the pages holding rows [0, pos) (a checkpoint).
  std::vector<int> pin(KvPagePool& pool, int pos) const {
    const int n = std::min(mapped, std::max(0, (pos + KV_PAGE - 1) / KV_PAGE));
    std::vector<int> v(tab.begin(), tab.begin() + n);
    for (int p : v) pool.addref(p);
    return v;
  }
  static void unpin(KvPagePool& pool, const std::vector<int>& pages) {
    for (size_t i = pages.size(); i-- > 0;) pool.unref(pages[i]);
  }
  // A3: map exactly `pages` (a pinned checkpoint prefix). The new references
  // are taken before the old mapping drops, so shared pages never hit the
  // free list in between.
  void adopt(KvPagePool& pool, const std::vector<int>& pages) {
    for (int p : pages) pool.addref(p);
    trim(pool, 0);
    for (size_t i = 0; i < pages.size(); i++) tab[i] = pages[i];
    mapped = (int)pages.size();
  }
  // A3 copy-on-write: logical page lp now maps fresh page p (already holding
  // a copy); the shared original loses the live reference.
  void replace(KvPagePool& pool, int lp, int p) {
    const int old = tab[(size_t)lp];
    tab[(size_t)lp] = p;
    pool.unref(old);
  }
};

// A3: physical pool size in pages. Default: one full-length sequence — pinned
// RAM checkpoints share it and are LRU-evicted when the live sequence needs
// pages. QWENOX_KV_POOL_TOKENS=<n> enlarges it (never below one sequence).
static inline int kv_pool_pages_for(int maxctx) {
  const long long lp = (maxctx + KV_PAGE - 1) / KV_PAGE;
  const char* v = getenv("QWENOX_KV_POOL_TOKENS");
  const long long t = v ? atoll(v) : 0;
  return (int)std::max(lp, (t + KV_PAGE - 1) / KV_PAGE);
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
                                const float2* cs_tab = nullptr, int tab_base = 0,
                                uint16_t* out16 = nullptr) {
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
      double angle = position * g_rope_tab[i];
      cs = (float)cos(angle), sn = (float)sin(angle);
    }
    v = d < 32 ? values[d] * cs - values[d + 32] * sn
                : values[d] * cs + values[d - 32] * sn;
  }
  out[d] = v;
  if (out16) out16[d] = __half_as_ushort(__float2half_rn(v));  // C1 f16 twin (RNE)
  (void)theta;
}

__global__ void k_index_q(const float* proj, const float* norm, float* queries,
                           int P, float eps, double theta, const int* pos = nullptr,
                           int base = 0, const float2* cs_tab = nullptr,
                           const int* __restrict__ d_rdelta = nullptr,
                           const int* __restrict__ d_vb = nullptr,
                           uint16_t* queries16 = nullptr) {
  int t = blockIdx.x, h = blockIdx.y;
  if (d_vb) base = *d_vb;  // P3-A: device-mirrored base (graph-replayable)
  // d_rdelta (M-RoPE decode): rope angle position += request rope_delta.
  // Only passed by decode callers (cs_tab == nullptr there); the prefill
  // cs_tab path must keep position-base == batch-local row.
  int position = pos ? *pos : base + t;
  if (d_rdelta) position += *d_rdelta;
  index_norm_rope(proj[(size_t)t * 640 + h * 128 + threadIdx.x], norm,
                  queries + (size_t)t * 512 + h * 128, position, eps,
                  theta, cs_tab, base,
                  queries16 ? queries16 + (size_t)t * 512 + h * 128 : nullptr);
}

// Pooled indexer keys: absolute 4-token blocks. base = absolute position of
// batch row 0; a block straddling the left edge (base % 4 != 0) reads its
// history rows from the raw ring (same values the decode path's
// k_index_append would see, summed in the same order -> bitwise identical).
__global__ void k_index_pool(const float* proj, const float* norm, float* keys,
                              float eps, double theta, int base = 0,
                              const float* ring = nullptr,
                              const float2* cs_tab = nullptr,
                              const int* __restrict__ ptab = nullptr,
                              const int* __restrict__ d_vb = nullptr,
                              uint16_t* keys16 = nullptr) {
  if (d_vb) base = *d_vb;  // P3-A: device-mirrored base (graph-replayable)
  int b = blockIdx.x + (base / 4), d = threadIdx.x;  // absolute block id
  float v = 0.f;
  for (int j = 0; j < 4; j++) {
    int lt = b * 4 + j - base;  // batch-local row; < 0 -> history
    v += lt >= 0 ? proj[(size_t)lt * 640 + 512 + d]
                 : ring[(size_t)((b * 4 + j) % 4) * 128 + d];
  }
  const size_t blk = (size_t)ik_blk(ptab, b) * 128;
  index_norm_rope(v * 0.25f, norm, keys + blk, b * 4,
                  eps, theta, cs_tab, base, keys16 ? keys16 + blk : nullptr);
}

__global__ void k_index_ring(const float* proj, float* ring, int P, int base = 0,
                             const int* __restrict__ d_vb = nullptr) {
  if (d_vb) base = *d_vb;  // P3-A: device-mirrored base (graph-replayable)
  int t = max(0, P - 4) + (int)blockIdx.x;
  if (t < P)
    ring[((base + t) % 4) * 128 + threadIdx.x] = proj[(size_t)t * 640 + 512 + threadIdx.x];
}

__global__ void k_index_append(const float* proj, const float* norm, float* ring,
                                float* keys, const int* pos, float eps, double theta,
                                const int* __restrict__ d_rdelta = nullptr,
                                const int* __restrict__ ptab = nullptr,
                                uint16_t* keys16 = nullptr) {
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
                  position, eps, theta, nullptr, 0,
                  keys16 ? keys16 + (size_t)ik_blk(ptab, t / 4) * 128 : nullptr);
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
// C1 f16 twin gather (same layout, 16-bit elements)
__global__ void k_ik_gather16(const uint16_t* __restrict__ src,
                              const int* __restrict__ ptab,
                              uint16_t* __restrict__ dst, int nb) {
  int b = blockIdx.x;
  if (b >= nb) return;
  dst[(size_t)b * 128 + threadIdx.x] =
      src[(size_t)ik_blk(ptab, b) * 128 + threadIdx.x];
}
// C1: rebuild f16 twin blocks [0, nb) from the fp32 indexer keys (kvsnap
// restore writes only fp32 pages; twins are derived state, not saved).
__global__ void k_ik_twin_build(const float* __restrict__ src,
                                const int* __restrict__ ptab,
                                uint16_t* __restrict__ dst, int nb) {
  int b = blockIdx.x;
  if (b >= nb) return;
  const size_t blk = (size_t)ik_blk(ptab, b) * 128 + threadIdx.x;
  dst[blk] = __half_as_ushort(__float2half_rn(src[blk]));
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

// k_index_select_rs / k_index_select_stream / k_index_scores_tiled and the
// faster bit-exact replacements live in 09_kernels_index.inc (standalone so
// tools/index_fast_test.cu can include them).

static void index_select(const float* scores, int stride, int* selected,
                         int first, int count, const int* pos = nullptr) {
  int n = pos ? stride : min(stride, (first + count) / 4);
  static bool old_sel = getenv("QWENOX_INDEX_OLDSEL") != nullptr;
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

__global__ void k_moe_tiles(const int* eoff, MoeTile* tiles, int* total, int experts, int cap) {
  using Scan = rocprim::block_scan<int, 512>;
  __shared__ Scan::storage_type storage;
  int e = threadIdx.x;
  int first = e < experts ? eoff[e] : 0;
  int ne = e < experts ? eoff[e + 1] - first : 0;
  int count = (ne + cap - 1) / cap, begin;
  Scan().exclusive_scan(count, begin, 0, storage);
  if (e == 511) *total = begin + count;
  for (int j = 0; j < count; j++) tiles[begin + j] = {e, first + j * cap, min(cap, ne - j * cap)};
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

