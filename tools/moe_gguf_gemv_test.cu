// moe_gguf_gemv_test.cu — check + bench of the small-P GGUF routed-expert GEMV
// path (moe_gg_gemv in src/gpu/parts/26_kernels_moe_gguf.inc) on real GGUF
// expert weights.
//
//   moe_gguf_gemv_test FIRST_SHARD.gguf [--layer L] [--P N] [--iters N] [--tol X] [--share F]
//
// --share F: each slot of token t>0 reuses an expert of token t-1 with
// probability F (spec-verify tokens share experts). For P >= 2 the expert-
// dedup path (uq_meta/uq_pairs) is also run: it must match the plain path
// bit for bit, and both are timed.
//
// Random x (N(0,1)), random distinct top-10 ids per token in 16-wide rows
// (engine id_stride), random softmax-like weights. CPU reference in double
// with ggml dequant (gguf.h):
//   hid[t*10+s][m] = up * silu(gate)                       (checks gate/up)
//   out[t][r]      = sum_s w[t][s] * down_e(gpu hid)        (isolates down)
// FAIL if the max relative L2 error (per token row) exceeds --tol (1e-4).
// Also prints the time per call and effective GB/s (weight bytes touched).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../src/gguf.h"

struct MoeTile { int expert, first, count; };
#include "../src/gpu/parts/26_kernels_moe_gguf.inc"

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s FIRST_SHARD.gguf [--layer L] [--P N] [--iters N] [--tol X]\n", argv[0]);
    return 2;
  }
  std::string path = argv[1];
  int layer = 0, P = 1, iters = 200, topk = 10, IDS = 16;
  unsigned seed = 1;
  double tol = 1e-4, share = 0.0;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--layer") layer = atoi(argv[i + 1]);
    else if (a == "--P") P = atoi(argv[i + 1]);
    else if (a == "--iters") iters = atoi(argv[i + 1]);
    else if (a == "--seed") seed = (unsigned)atoi(argv[i + 1]);
    else if (a == "--tol") tol = atof(argv[i + 1]);
    else if (a == "--share") share = atof(argv[i + 1]);
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  gguf::File f(path);
  char nm[128];
  snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", layer);
  const gguf::Tensor& tg = f.at(nm);
  snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", layer);
  const gguf::Tensor& tu = f.at(nm);
  snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", layer);
  const gguf::Tensor& td = f.at(nm);
  const int D = (int)tg.ne[0], MID = (int)tg.ne[1], E = (int)tg.ne[2];
  printf("layer %d: gate/up %s, down %s, D=%d MID=%d E=%d P=%d\n", layer,
         gguf::type_name(tg.type), gguf::type_name(td.type), D, MID, E, P);

  uint8_t *dg, *du, *dd;
  CK(hipMalloc(&dg, tg.nbytes));
  CK(hipMalloc(&du, tu.nbytes));
  CK(hipMalloc(&dd, td.nbytes));
  CK(hipMemcpy(dg, tg.data, tg.nbytes, hipMemcpyHostToDevice));
  CK(hipMemcpy(du, tu.data, tu.nbytes, hipMemcpyHostToDevice));
  CK(hipMemcpy(dd, td.data, td.nbytes, hipMemcpyHostToDevice));

  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> ue(0, E - 1);
  std::vector<float> x((size_t)P * D), w((size_t)P * IDS, 0.f);
  std::vector<int> ids((size_t)P * IDS, 0);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::uniform_int_distribution<int> us(0, topk - 1);
  auto route = [&](int* id) {  // id: [P][IDS]
    for (int t = 0; t < P; t++)
      for (int s = 0; s < topk; s++) {
        int e;
        bool dup;
        do {
          e = (t > 0 && u01(rng) < share) ? id[(t - 1) * IDS + us(rng)] : ue(rng);
          dup = false;
          for (int q = 0; q < s; q++) dup |= id[t * IDS + q] == e;
        } while (dup);
        id[t * IDS + s] = e;
      }
  };
  for (auto& v : x) v = nd(rng);
  route(ids.data());
  {
    std::vector<int> v;
    for (int t = 0; t < P; t++)
      for (int s = 0; s < topk; s++) v.push_back(ids[t * IDS + s]);
    std::sort(v.begin(), v.end());
    printf("routing: %d unique experts of %d slots\n", (int)(std::unique(v.begin(), v.end()) - v.begin()), P * topk);
  }
  for (int t = 0; t < P; t++) {
    float sum = 0.f;
    for (int s = 0; s < topk; s++) {
      w[t * IDS + s] = expf(nd(rng));
      sum += w[t * IDS + s];
    }
    for (int s = 0; s < topk; s++) w[t * IDS + s] /= sum;
  }
  float *dx, *dw, *dh, *dout;
  int* di;
  CK(hipMalloc(&dx, x.size() * 4));
  CK(hipMalloc(&dw, w.size() * 4));
  CK(hipMalloc(&di, ids.size() * 4));
  CK(hipMalloc(&dh, (size_t)P * topk * MID * 4));
  CK(hipMalloc(&dout, (size_t)P * D * 4));
  CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(dw, w.data(), w.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(di, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemset(dh, 0xFF, (size_t)P * topk * MID * 4));   // NaN fill
  CK(hipMemset(dout, 0xFF, (size_t)P * D * 4));

  if (!moe_gg_gemv((int)tg.type, (int)td.type, dg, du, dd, dx, di, dw, dh, dout, P, D, MID, IDS, 0)) {
    printf("moe_gg_gemv: unsupported types\nFAIL\n");
    return 1;
  }
  CK(hipDeviceSynchronize());
  std::vector<float> hid((size_t)P * topk * MID), out((size_t)P * D);
  CK(hipMemcpy(hid.data(), dh, hid.size() * 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));

  // CPU reference
  std::map<int, std::vector<float>> G, U, Dn;  // dequantized experts
  auto deq = [&](const gguf::Tensor& t, int e, int rows, int cols) {
    std::vector<float> m((size_t)rows * cols);
    const size_t rb = t.row_bytes();
    for (int r = 0; r < rows; r++)
      gguf::dequant_row(t.type, t.data + ((size_t)e * rows + r) * rb, &m[(size_t)r * cols], cols);
    return m;
  };
  double worst_h = 0, worst_o = 0;
  bool nan = false;
  for (int t = 0; t < P; t++) {
    std::vector<double> o(D, 0.0);
    double hn = 0, he = 0;
    for (int s = 0; s < topk; s++) {
      const int e = ids[t * IDS + s];
      if (!G.count(e)) {
        G[e] = deq(tg, e, MID, D);
        U[e] = deq(tu, e, MID, D);
        Dn[e] = deq(td, e, D, MID);
      }
      const float* xt = &x[(size_t)t * D];
      const float* hg = &hid[((size_t)t * topk + s) * MID];
      for (int m = 0; m < MID; m++) {
        double g = 0, u = 0;
        for (int k = 0; k < D; k++) {
          g += (double)G[e][(size_t)m * D + k] * xt[k];
          u += (double)U[e][(size_t)m * D + k] * xt[k];
        }
        const double ref = u * g / (1.0 + exp(-g));
        if (!std::isfinite(hg[m])) nan = true;
        hn += ref * ref;
        he += (hg[m] - ref) * (hg[m] - ref);
      }
      for (int r = 0; r < D; r++) {
        double a = 0;
        for (int m = 0; m < MID; m++) a += (double)Dn[e][(size_t)r * MID + m] * hg[m];
        o[r] += (double)w[t * IDS + s] * a;
      }
    }
    double on = 0, oe = 0;
    for (int r = 0; r < D; r++) {
      const float g = out[(size_t)t * D + r];
      if (!std::isfinite(g)) nan = true;
      on += o[r] * o[r];
      oe += (g - o[r]) * (g - o[r]);
    }
    worst_h = std::max(worst_h, sqrt(he / std::max(hn, 1e-30)));
    worst_o = std::max(worst_o, sqrt(oe / std::max(on, 1e-30)));
  }
  printf("rel-L2 max: hid %.3e  out %.3e%s\n", worst_h, worst_o, nan ? "  (NaN/unwritten!)" : "");

  // expert-dedup path: must equal the plain path bit for bit
  int* dmeta = nullptr;
  float* dpairs = nullptr;
  bool uq_ok = true;
  const bool uq = P >= 2 && P <= moegg::kUqMaxP;
  if (uq) {
    CK(hipMalloc(&dmeta, moegg::kUqInts * 4));
    CK(hipMalloc(&dpairs, (size_t)P * topk * D * 4));
    CK(hipMemset(dh, 0xFF, (size_t)P * topk * MID * 4));
    CK(hipMemset(dout, 0xFF, (size_t)P * D * 4));
    if (!moe_gg_gemv((int)tg.type, (int)td.type, dg, du, dd, dx, di, dw, dh, dout, P, D, MID, IDS, 0,
                     dmeta, dpairs)) {
      printf("dedup: unsupported\nFAIL\n");
      return 1;
    }
    CK(hipDeviceSynchronize());
    std::vector<float> hid2(hid.size()), out2(out.size());
    CK(hipMemcpy(hid2.data(), dh, hid2.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(out2.data(), dout, out2.size() * 4, hipMemcpyDeviceToHost));
    const bool hb = memcmp(hid2.data(), hid.data(), hid.size() * 4) == 0;
    const bool ob = memcmp(out2.data(), out.data(), out.size() * 4) == 0;
    double md = 0;
    for (size_t i = 0; i < out.size(); i++) md = std::max(md, (double)fabsf(out2[i] - out[i]));
    printf("dedup vs plain: hid %s, out %s (max |diff| %.3e)\n", hb ? "bit-exact" : "DIFF",
           ob ? "bit-exact" : "DIFF", md);
    uq_ok = hb && ob;
  }

  // timing: rotate NSET random routings so the ~30 MB of experts per call does
  // not stay resident in the 32 MB MALL (production decode sees fresh experts)
  const int NSET = 64;
  std::vector<int> rids((size_t)NSET * P * IDS, 0);
  for (int q = 0; q < NSET; q++) route(&rids[(size_t)q * P * IDS]);
  int* dri;
  CK(hipMalloc(&dri, rids.size() * 4));
  CK(hipMemcpy(dri, rids.data(), rids.size() * 4, hipMemcpyHostToDevice));
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  auto timeit = [&](bool dedup) {
    auto call = [&](int i) {
      moe_gg_gemv((int)tg.type, (int)td.type, dg, du, dd, dx, dri + (size_t)(i % NSET) * P * IDS, dw,
                  dh, dout, P, D, MID, IDS, 0, dedup ? dmeta : nullptr, dedup ? dpairs : nullptr);
    };
    for (int i = 0; i < 8; i++) call(i);
    CK(hipEventRecord(a, 0));
    for (int i = 0; i < iters; i++) call(i);
    CK(hipEventRecord(b, 0));
    CK(hipEventSynchronize(b));
    float t = 0;
    CK(hipEventElapsedTime(&t, a, b));
    return t / iters;
  };
  const float ms = timeit(false);
  const float ms_uq = uq ? timeit(true) : 0.f;
  double bytes = 0;
  for (int q = 0; q < NSET; q++) {
    std::vector<int> u(rids.begin() + (size_t)q * P * IDS, rids.begin() + (size_t)(q + 1) * P * IDS);
    std::vector<int> v;
    for (int t = 0; t < P; t++)
      for (int s = 0; s < topk; s++) v.push_back(u[t * IDS + s]);
    std::sort(v.begin(), v.end());
    bytes += (double)(std::unique(v.begin(), v.end()) - v.begin());
  }
  bytes = bytes / NSET * (tg.row_bytes() * MID * 2 + td.row_bytes() * D);
  printf("time %.1f us/call  (avg %.1f MB unique weights, %.0f GB/s)\n", ms * 1e3, bytes / 1e6,
         bytes / (ms * 1e-3) / 1e9);
  if (uq)
    printf("dedup time %.1f us/call  (%.0f GB/s)  speedup %.2fx\n", ms_uq * 1e3,
           bytes / (ms_uq * 1e-3) / 1e9, ms / ms_uq);
  const bool ok = !nan && worst_h < tol && worst_o < tol && uq_ok;
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
