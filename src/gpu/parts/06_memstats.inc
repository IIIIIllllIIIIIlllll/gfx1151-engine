// ---------------------- engine memory accounting ---------------------------
// hipMemGetInfo/rocm-smi cannot attribute mmap + hipHostRegister consistently
// on an APU. Track the allocations made by this process and report the driver
// view alongside them. File-backed mmap pages that are not registered are
// intentionally excluded: they are pageable cache, not committed GPU memory.

struct EngineMemStats {
  uint64_t device_current = 0;
  uint64_t device_peak = 0;
  uint64_t registered_current = 0;
  uint64_t pinned_current = 0;
  uint64_t pinned_peak = 0;
  uint64_t committed_current = 0;
  uint64_t hip_free = 0;
  uint64_t hip_total = 0;
  uint64_t hip_used_since_start = 0;
  uint64_t process_rss = 0;
  uint64_t process_locked = 0;
};

static std::mutex g_memstats_mtx;
static std::unordered_map<void*, uint64_t> g_memstats_device;
static std::unordered_map<void*, uint64_t> g_memstats_pinned;
static std::unordered_map<void*, uint64_t> g_memstats_registered;
static uint64_t g_memstats_device_current = 0, g_memstats_device_peak = 0;
static uint64_t g_memstats_pinned_current = 0, g_memstats_pinned_peak = 0;
static uint64_t g_memstats_registered_current = 0;
static uint64_t g_memstats_hip_free_start = 0;

static void memstats_add(std::unordered_map<void*, uint64_t>& allocs,
                         uint64_t& current, uint64_t* peak, void* p,
                         uint64_t bytes) {
  std::lock_guard<std::mutex> lk(g_memstats_mtx);
  auto old = allocs.find(p);
  if (old != allocs.end()) {
    current -= old->second;
    old->second = bytes;
  } else {
    allocs.emplace(p, bytes);
  }
  current += bytes;
  if (peak) *peak = std::max(*peak, current);
}

static void memstats_remove(std::unordered_map<void*, uint64_t>& allocs,
                            uint64_t& current, void* p) {
  std::lock_guard<std::mutex> lk(g_memstats_mtx);
  auto it = allocs.find(p);
  if (it == allocs.end()) return;
  current -= it->second;
  allocs.erase(it);
}

static hipError_t mem_hip_malloc(void** p, size_t bytes) {
  hipError_t e = hipMalloc(p, bytes);
  if (e == hipSuccess && p && *p)
    memstats_add(g_memstats_device, g_memstats_device_current,
                 &g_memstats_device_peak, *p, bytes);
  return e;
}

static hipError_t mem_hip_free(void* p) {
  hipError_t e = hipFree(p);
  if (e == hipSuccess && p)
    memstats_remove(g_memstats_device, g_memstats_device_current, p);
  return e;
}

static hipError_t mem_hip_host_malloc(void** p, size_t bytes, unsigned flags) {
  hipError_t e = hipHostMalloc(p, bytes, flags);
  if (e == hipSuccess && p && *p)
    memstats_add(g_memstats_pinned, g_memstats_pinned_current,
                 &g_memstats_pinned_peak, *p, bytes);
  return e;
}

static hipError_t mem_hip_host_free(void* p) {
  hipError_t e = hipHostFree(p);
  if (e == hipSuccess && p)
    memstats_remove(g_memstats_pinned, g_memstats_pinned_current, p);
  return e;
}

static hipError_t mem_hip_host_register(void* p, size_t bytes, unsigned flags) {
  hipError_t e = hipHostRegister(p, bytes, flags);
  if (e == hipSuccess && p)
    memstats_add(g_memstats_registered, g_memstats_registered_current, nullptr,
                 p, bytes);
  return e;
}

static hipError_t mem_hip_host_unregister(void* p) {
  hipError_t e = hipHostUnregister(p);
  if (e == hipSuccess && p)
    memstats_remove(g_memstats_registered, g_memstats_registered_current, p);
  return e;
}

static void memstats_capture_hip_baseline() {
  size_t free_b = 0, total_b = 0;
  if (hipMemGetInfo(&free_b, &total_b) == hipSuccess)
    g_memstats_hip_free_start = free_b;
}

#ifndef _WIN32
static void memstats_proc(uint64_t* rss, uint64_t* locked) {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    const bool is_rss = line.rfind("VmRSS:", 0) == 0;
    const bool is_locked = line.rfind("VmLck:", 0) == 0;
    if (!is_rss && !is_locked) continue;
    std::istringstream ss(line.substr(6));
    uint64_t kb = 0;
    ss >> kb;
    if (is_rss) *rss = kb * 1024;
    if (is_locked) *locked = kb * 1024;
  }
}
#endif

static EngineMemStats memstats_snapshot() {
  EngineMemStats s;
  {
    std::lock_guard<std::mutex> lk(g_memstats_mtx);
    s.device_current = g_memstats_device_current;
    s.device_peak = g_memstats_device_peak;
    s.registered_current = g_memstats_registered_current;
    s.pinned_current = g_memstats_pinned_current;
    s.pinned_peak = g_memstats_pinned_peak;
  }
  s.committed_current =
      s.device_current + s.registered_current + s.pinned_current;
  size_t free_b = 0, total_b = 0;
  if (hipMemGetInfo(&free_b, &total_b) == hipSuccess) {
    s.hip_free = free_b;
    s.hip_total = total_b;
    if (g_memstats_hip_free_start > s.hip_free)
      s.hip_used_since_start = g_memstats_hip_free_start - s.hip_free;
  }
#ifndef _WIN32
  memstats_proc(&s.process_rss, &s.process_locked);
#endif
  return s;
}

static void memstats_log(const char* stage) {
  EngineMemStats s = memstats_snapshot();
  fprintf(stderr,
          "memory[%s]: device %.2f GiB (peak %.2f), registered mmap %.2f, "
          "pinned host %.2f (peak %.2f), gpu-accessible committed %.2f; "
          "HIP free %.2f/%.2f GiB, process RSS %.2f GiB\n",
          stage, s.device_current / 1073741824.0,
          s.device_peak / 1073741824.0,
          s.registered_current / 1073741824.0,
          s.pinned_current / 1073741824.0,
          s.pinned_peak / 1073741824.0,
          s.committed_current / 1073741824.0, s.hip_free / 1073741824.0,
          s.hip_total / 1073741824.0, s.process_rss / 1073741824.0);
}

// Wrap every subsequent engine allocation, including lazy vision buffers.
#define hipMalloc(p, n) mem_hip_malloc((void**)(p), (size_t)(n))
#define hipFree(p) mem_hip_free((void*)(p))
#define hipHostMalloc(p, n, f) \
  mem_hip_host_malloc((void**)(p), (size_t)(n), (unsigned)(f))
#define hipHostFree(p) mem_hip_host_free((void*)(p))
#define hipHostRegister(p, n, f) \
  mem_hip_host_register((void*)(p), (size_t)(n), (unsigned)(f))
#define hipHostUnregister(p) mem_hip_host_unregister((void*)(p))
