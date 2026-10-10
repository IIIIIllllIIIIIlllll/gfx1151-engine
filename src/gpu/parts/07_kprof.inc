// ---------------- QWENOX_KPROF: prefill 分段 GPU 计时（默认关） ----------------
// 设计：HANDOFF-V2-PREFILL.md §0。KP("tag") 在 g_str 上 hipEventRecord，标记一个
// 分段的开始；该分段持续到下一个 KP。KP_SUB("tag")/KP_SUB_END() 标记分段内部的
// 子区间（ht_deq、gdn:*/qsa:* 细分），按 tag 单独累加（其时间也算在外层分段里）；
// 子区间可嵌套（栈配对，外层含内层）。chunk 内不做任何同步；
// kprof_flush 每 chunk 调用一次（argmax D2H 之后），同步最后一个 event，用相邻
// event 的 hipEventElapsedTime 求差，按 tag 累加（48 层合计）后打一行到 stderr。
// 关时（QWENOX_KPROF 未设置）每个 KP 只剩一次全局 bool 检查，env 只读一次。
// g_kprof_mute 在 decode 图捕获期间置位：hipEventRecord 进捕获流会把 event 录进图里。
static bool g_kprof_on = [] {
  const char* e = getenv("QWENOX_KPROF");
  return e && !(e[0] == '0' && e[1] == 0);
}();
static bool g_kprof_mute = false;

struct KProfRing {
  static constexpr int CAP = 8192;
  hipEvent_t ev[CAP]{};
  const char* tag[CAP]{};
  uint8_t kind[CAP]{};  // 0 = 分段边界, 1 = 子区间开始, 2 = 子区间结束
  int n = 0;
  bool ready = false, overflow = false;
  void init() {
    if (ready) return;
    for (int i = 0; i < CAP; i++) CK(hipEventCreateWithFlags(&ev[i], 0));  // 要计时
    ready = true;
  }
  void rec(const char* t, int k) {
    if (g_kprof_mute) return;
    if (!ready) init();
    if (n >= CAP) {
      if (!overflow) {
        overflow = true;
        fprintf(stderr, "kprof: event ring (%d) full, tagging stopped\n", CAP);
      }
      return;
    }
    tag[n] = t;
    kind[n] = (uint8_t)k;
    CK(hipEventRecord(ev[n], g_str));
    n++;
  }
  // 每 chunk 一次（此刻主流已被 argmax D2H 排空；非 final chunk 则在此同步）。
  // quiet（spec verify 的小 chunk）只重置不打印，避免 decode 阶段刷屏。
  void flush(int base, int P, bool quiet) {
    if (!ready || n == 0) return;
    if (quiet) {
      n = 0;
      overflow = false;
      return;
    }
    CK(hipEventSynchronize(ev[n - 1]));
    std::vector<std::pair<const char*, double>> seg;
    auto add = [&](const char* t, double ms) {
      for (auto& p : seg)
        if (p.first == t) {
          p.second += ms;
          return;
        }
      seg.push_back({t, ms});
    };
    double total = 0, neg_clamp = 0;
    std::vector<std::pair<const char*, double>> sub;
    auto addsub = [&](const char* t, double ms) {
      for (auto& p : sub)
        if (p.first == t) {
          p.second += ms;
          return;
        }
      sub.push_back({t, ms});
    };
    // 子区间可嵌套（如 ht_deq 在 gdn:proj 的 gemm 内部）：用栈配对 begin/end；
    // 外层区间时间包含内层，各 tag 独立累加。
    int lastb = -1;
    int sub_stk[16], sub_top = 0;
    for (int i = 0; i < n; i++) {
      if (kind[i] == 0) {
        if (lastb >= 0) {
          float ms = 0;
          CK(hipEventElapsedTime(&ms, ev[lastb], ev[i]));
          // Windows/TheRock: 个别 event 的时间戳会回跳 ~200 ms（疑似 KMD 提交
          // 批边界处的时钟域差异），每 chunk 约一次。负区间钳为 0 并计入
          // neg_clamp，总量与归属误差据此可估。
          if (ms < 0) {
            neg_clamp += -ms;
            ms = 0;
          }
          add(tag[lastb], ms);
          total += ms;
        }
        lastb = i;
      } else if (kind[i] == 1) {
        if (sub_top < 16) sub_stk[sub_top++] = i;
      } else if (sub_top > 0) {
        int b = sub_stk[--sub_top];
        float ms = 0;
        CK(hipEventElapsedTime(&ms, ev[b], ev[i]));
        if (ms < 0) {
          neg_clamp += -ms;
          ms = 0;
        }
        addsub(tag[b], ms);
      }
    }
    std::sort(seg.begin(), seg.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    fprintf(stderr, "kprof base=%d P=%d segsum=%.1f ms", base, P, total);
    if (overflow) fprintf(stderr, " (ring overflowed, incomplete)");
    if (neg_clamp > 0) fprintf(stderr, " neg_clamp=%.1f ms", neg_clamp);
    for (auto& p : seg)
      fprintf(stderr, " | %s %.1f (%.1f%%)", p.first, p.second,
              total > 0 ? 100.0 * p.second / total : 0.0);
    for (auto& p : sub)
      fprintf(stderr, " | [%s %.1f (%.1f%%, 含在上列分段内)]", p.first, p.second,
              total > 0 ? 100.0 * p.second / total : 0.0);
    fprintf(stderr, "\n");
    n = 0;
    overflow = false;
  }
};
static KProfRing g_kprof_ring;

#define KP(t)                                     \
  do {                                            \
    if (g_kprof_on) g_kprof_ring.rec(t, 0);       \
  } while (0)
#define KP_SUB(t)                                 \
  do {                                            \
    if (g_kprof_on) g_kprof_ring.rec(t, 1);       \
  } while (0)
#define KP_SUB_END()                              \
  do {                                            \
    if (g_kprof_on) g_kprof_ring.rec("", 2);      \
  } while (0)
