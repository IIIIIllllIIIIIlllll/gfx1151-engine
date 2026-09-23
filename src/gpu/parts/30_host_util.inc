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
      !getenv("GDEC_GDN_NOSPLIT") && !getenv("GDEC_GDN_LOOP") &&
      !getenv("GDEC_GDN_FUSED");
  const bool gdn_str = getenv("GDEC_GDN_STREAM") != nullptr;
  const bool gdn_pipe = getenv("GDEC_GDN_PIPE") != nullptr;
  const int gdn_window_chunks = [] {
    const char* e = getenv("GDEC_GDN_WINDOW_CHUNKS");
    return e ? std::max(1, atoi(e)) : 4;
  }();
  const size_t mc = (size_t)maxctx,
               BP = (size_t)(eff_maxbatch(maxctx) + eff_tail_slack());
  const size_t ngdn = g_cfg.layers - g_cfg.layers / 4, nqsa = g_cfg.layers / 4;
  const size_t ib = (mc + 3) / 4, rb = (size_t)g_cfg.branches * g_cfg.d;
  // GDEC_KV_PAGED：KV/ik（BF16 时含 d_vcbt）按整页（256 token）取整，另加页表与
  // ik gather 缓冲。BF16 KV 的可分页条件与构造函数一致（A1c）
  const char* kvpe = getenv("GDEC_KV_PAGED");
  const bool kv_bf16_pg_ok =
      qsa_dense || (getenv("GDEC_QSA_WMMA") && !(btv && getenv("GDEC_QSA_WMMA6")) &&
                    !(btv && getenv("GDEC_QSA_UNION")));
  const bool kv_paged = (kv_fp32 || kv_bf16_pg_ok) && kvpe && atoi(kvpe) != 0;
  const size_t kvpg = (mc + KV_PAGE - 1) / KV_PAGE;
  // A2/A3：物理页池 = kv_pool_pages_for(mc) 页（默认 kvpg，GDEC_KV_POOL_TOKENS 可加大）
  // + 1 个 guard 页；页表仍是每序列 kvpg 项
  const size_t kvpool = (size_t)kv_pool_pages_for((int)mc);
  const size_t kv_rows = kv_paged ? (kvpool + 1) * KV_PAGE : mc;
  const size_t ik_rows = kv_paged ? (kvpool + 1) * KV_PAGE_BLOCKS : ib;
  if (kv_paged) add(kvpg * 4);  // d_ptab
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
        (size_t)GDN_SPLIT_WS_FLOATS * 4 * (gdn_pipe ? 2 : 1));
  add((size_t)64 << 20);                    // d_wbf16
  add((size_t)64 << 20);                    // d_ltws
  if (!qsa_dense) {
    const size_t bc = std::min(mc, BP);     // maxbatch_cap（A1b：按 chunk 定尺寸）
    add(bc * 640 * 4);                      // d_iproj
    add(bc * 512 * 4);                      // d_iq
    add(nqsa * ik_rows * 128 * 4);          // d_ik
    if (kv_paged) add(ib * 128 * 4);        // d_ikg
    add(nqsa * 4 * 128 * 4);                // d_iraw
    add(nqsa * 256 * 4);                    // d_inorm
    if (index_f32) add(nqsa * 640 * 2560 * 4);  // d_iw
    add((size_t)std::min(mc, (size_t)INDEX_BATCH) * 4 * ib * 4);  // d_iscores
    add(bc * INDEX_BUDGET * 4);             // d_iselected
  }
  if (kv_fp32) {
    add(2 * nqsa * kv_rows * 512 * 4);      // d_kc + d_vc
  } else {
    add(2 * nqsa * kv_rows * 512 * 2);      // d_kcb + d_vcb
    if (btv) add(nqsa * ((kv_rows + 3) / 4) * 2 * 256 * 4 * 2);  // d_vcbt
    // GDEC_QSA_UNION: d_ublk + d_umask + d_ucount（按单 chunk 组数）
    if (btv && getenv("GDEC_QSA_UNION")) {
      const size_t ng = (size_t)BP / 4 + 1;
      add(ng * QSA_UCAP * 4 + ng * 4);
    }
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

