// perfsnap.h — 引擎内实时性能快照（dev-docs/HANDOFF-PERFSNAP.md）。
// dl_tick / prog_print 每秒写一次最近一次/进行中请求的 pp、tg、接受率；
// 行协议的 SNAP 动词把它吐给托盘轮询（127.0.0.1，无磁盘 IO、无日志解析）。
// 多 conn 线程并发写由互斥串行化，last-writer-wins（tooltip 不需要精确
// 多请求归属）；TTL 由读端判：ts_ms 超过 6s 未更新视为过期。
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>

struct PerfSnap {
  uint64_t ts_ms = 0;    // 最后一次更新（system_clock 毫秒墙钟）
  uint32_t state = 0;    // 0=idle 1=prefill 2=decode
  uint64_t req_id = 0;
  double pp_toks = 0;                       // prefill 瞬时吞吐 tok/s
  uint64_t prompt_done = 0, prompt_total = 0;  // prefill 进度
  double tg_inst = 0, tg_avg = 0;           // decode 瞬时/平均 tok/s
  double accept_pct = -1;                   // 接受率 %；<0 表示当前无投机数据
};

namespace perfsnap {

inline uint64_t now_ms() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline PerfSnap& slot() {
  static PerfSnap s;
  return s;
}

inline std::mutex& mtx() {
  static std::mutex m;
  return m;
}

template <typename F>
inline void update(F&& f) {
  std::lock_guard<std::mutex> lk(mtx());
  f(&slot());
  slot().ts_ms = now_ms();
}

inline void write_prefill(uint64_t req, double pp_toks, uint64_t done,
                          uint64_t total) {
  update([&](PerfSnap* p) {
    p->state = 1;
    p->req_id = req;
    p->pp_toks = pp_toks;
    p->prompt_done = done;
    p->prompt_total = total;
  });
}

inline void write_decode(uint64_t req, double tg_inst, double tg_avg,
                         double accept_pct) {
  update([&](PerfSnap* p) {
    p->state = 2;
    p->req_id = req;
    p->tg_inst = tg_inst;
    p->tg_avg = tg_avg;
    p->accept_pct = accept_pct;
  });
}

inline void write_idle() {
  update([](PerfSnap* p) { p->state = 0; });
}

inline PerfSnap snapshot() {
  std::lock_guard<std::mutex> lk(mtx());
  return slot();
}

}  // namespace perfsnap
