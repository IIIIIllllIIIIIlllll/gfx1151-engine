// moe_gguf_test.cu — standalone check + bench of the GGUF routed-expert WMMA
// kernels (src/gpu/parts/26_kernels_moe_gguf.inc) on real GGUF expert weights.
//
//   moe_gguf_test FIRST_SHARD.gguf [--layer L] [--P N] [--iters N]
//                 [--check-experts N] [--seed S] [--tol X]
//
// Routing is random top-10 (distinct experts per token, mildly skewed).
// Correctness: every output element is written (outputs are pre-filled with
// NaN), and for --check-experts experts ALL their slots are compared against
// a CPU reference that dequantizes with ggml semantics (gguf.h) in double:
//   up:   hid = up * silu(gate) from the F16-rounded x
//   down: pairs from the GPU's F16 hid (isolates the down kernel)
// Metric: per-slot relative L2 error; FAIL if max > --tol (default 1e-2).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static float h2f(__half h) { return __half2float(h); }

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s FIRST_SHARD.gguf [--layer L] [--P N] [--iters N] [--check-experts N]\n", argv[0]);
    return 2;
  }
  std::string path = argv[1];
  int layer = 0, P = 2048, iters = 10, check_e = 8, topk = 10;
  unsigned seed = 1;
  double tol = 1e-2;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--layer") layer = atoi(argv[i + 1]);
    else if (a == "--P") P = atoi(argv[i + 1]);
    else if (a == "--iters") iters = atoi(argv[i + 1]);
    else if (a == "--check-experts") check_e = atoi(argv[i + 1]);
    else if (a == "--seed") seed = (unsigned)atoi(argv[i + 1]);
    else if (a == "--tol") tol = atof(argv[i + 1]);
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
  if (tu.type != tg.type || td.ne[0] != (uint64_t)MID || td.ne[1] != (uint64_t)D || td.ne[2] != (uint64_t)E) {
    fprintf(stderr, "unexpected shapes/types\n");
    return 2;
  }
  printf("layer %d: gate/up %s [%d x %d x %d], down %s [%d x %d x %d], P=%d top%d\n", layer,
         gguf::type_name(tg.type), D, MID, E, gguf::type_name(td.type), MID, D, E, P, topk);

  // weights -> device
  uint8_t *dg, *du, *dd;
  CK(hipMalloc(&dg, tg.nbytes));
  CK(hipMalloc(&du, tu.nbytes));
  CK(hipMalloc(&dd, td.nbytes));
  CK(hipMemcpy(dg, tg.data, tg.nbytes, hipMemcpyHostToDevice));
  CK(hipMemcpy(du, tu.data, tu.nbytes, hipMemcpyHostToDevice));
  CK(hipMemcpy(dd, td.data, td.nbytes, hipMemcpyHostToDevice));

  // routing (mild skew: expert popularity ~ 1/(1+e/64))
  std::mt19937 rng(seed);
  std::vector<double> pop(E);
  for (int e = 0; e < E; e++) pop[e] = 1.0 / (1.0 + ((e * 2654435761u) % E) / 64.0);
  std::discrete_distribution<int> pick(pop.begin(), pop.end());
  std::vector<int> ids((size_t)P * topk);
  for (int t = 0; t < P; t++) {
    for (int s = 0; s < topk; s++) {
      int e;
      bool dup;
      do {
        e = pick(rng);
        dup = false;
        for (int q = 0; q < s; q++) dup |= ids[(size_t)t * topk + q] == e;
      } while (dup);
      ids[(size_t)t * topk + s] = e;
    }
  }
  const int npairs = P * topk;
  std::vector<int> eoff(E + 1, 0), tokidx(npairs), slot_e(npairs);
  for (int p = 0; p < npairs; p++) eoff[ids[p] + 1]++;
  for (int e = 0; e < E; e++) eoff[e + 1] += eoff[e];
  {
    std::vector<int> cur(eoff.begin(), eoff.end() - 1);
    for (int t = 0; t < P; t++)
      for (int s = 0; s < topk; s++) {
        int e = ids[(size_t)t * topk + s], sl = cur[e]++;
        tokidx[sl] = t;
        slot_e[sl] = e;
      }
  }
  std::vector<MoeTile> tiles;
  for (int e = 0; e < E; e++)
    for (int r = eoff[e]; r < eoff[e + 1]; r += 64) tiles.push_back({e, r, std::min(64, eoff[e + 1] - r)});
  const int ntiles = (int)tiles.size();
  const int max_tiles = (npairs + 63) / 64 + E;  // engine's grid bound

  // x: F16 [P][D]
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<__half> xh((size_t)P * D);
  for (auto& v : xh) v = __float2half(nd(rng));

  __half *d_x, *d_hid;
  float* d_pairs;
  int *d_tok, *d_nt;
  MoeTile* d_tiles;
  CK(hipMalloc(&d_x, xh.size() * 2));
  CK(hipMalloc(&d_hid, (size_t)npairs * MID * 2));
  CK(hipMalloc(&d_pairs, (size_t)npairs * D * 4));
  CK(hipMalloc(&d_tok, npairs * 4));
  CK(hipMalloc(&d_nt, 4));
  CK(hipMalloc(&d_tiles, (size_t)max_tiles * sizeof(MoeTile)));
  CK(hipMemcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_tok, tokidx.data(), npairs * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_nt, &ntiles, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_tiles, tiles.data(), ntiles * sizeof(MoeTile), hipMemcpyHostToDevice));
  CK(hipMemset(d_hid, 0xFF, (size_t)npairs * MID * 2));    // NaN
  CK(hipMemset(d_pairs, 0xFF, (size_t)npairs * D * 4));    // NaN

  auto run_up = [&]() {
    if (!moe_gg_up(tg.type, dg, du, d_x, d_tok, d_tiles, d_nt, max_tiles, d_hid, MID, D, 0)) {
      fprintf(stderr, "moe_gg_up: unsupported type %s\n", gguf::type_name(tg.type));
      exit(2);
    }
  };
  auto run_dn = [&]() {
    if (!moe_gg_down(td.type, dd, d_hid, d_tiles, d_nt, max_tiles, d_pairs, D, MID, 0)) {
      fprintf(stderr, "moe_gg_down: unsupported type %s\n", gguf::type_name(td.type));
      exit(2);
    }
  };
  run_up();
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  run_dn();
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());

  std::vector<__half> hid((size_t)npairs * MID);
  std::vector<float> pairs((size_t)npairs * D);
  CK(hipMemcpy(hid.data(), d_hid, hid.size() * 2, hipMemcpyDeviceToHost));
  CK(hipMemcpy(pairs.data(), d_pairs, pairs.size() * 4, hipMemcpyDeviceToHost));
  bool ok = true;
  size_t nan_h = 0, nan_p = 0;
  for (auto v : hid) nan_h += !std::isfinite(h2f(v));
  for (auto v : pairs) nan_p += !std::isfinite(v);
  printf("coverage: non-finite hid %zu / %zu, pairs %zu / %zu\n", nan_h, hid.size(), nan_p, pairs.size());
  if (nan_h || nan_p) ok = false;

  // reference on check_e experts (spread over the id range, non-empty)
  std::vector<int> ce;
  for (int i = 0; i < E && (int)ce.size() < check_e; i++) {
    int e = (int)(((uint64_t)i * 7919 + 13) % E);
    if (eoff[e + 1] > eoff[e]) ce.push_back(e);
  }
  double up_max = 0, up_sum = 0, dn_max = 0, dn_sum = 0;
  int nchk = 0;
  std::vector<float> wg((size_t)MID * D), wu((size_t)MID * D), wd((size_t)D * MID);
  const uint64_t rbg = tg.row_bytes(), rbd = td.row_bytes();
  for (int e : ce) {
    for (int r = 0; r < MID; r++) {
      gguf::dequant_row(tg.type, tg.data + ((size_t)e * MID + r) * rbg, &wg[(size_t)r * D], D);
      gguf::dequant_row(tu.type, tu.data + ((size_t)e * MID + r) * rbg, &wu[(size_t)r * D], D);
    }
    for (int r = 0; r < D; r++)
      gguf::dequant_row(td.type, td.data + ((size_t)e * D + r) * rbd, &wd[(size_t)r * MID], MID);
    for (int sl = eoff[e]; sl < eoff[e + 1]; sl++) {
      const __half* xr = &xh[(size_t)tokidx[sl] * D];
      std::vector<double> xf(D);
      for (int c = 0; c < D; c++) xf[c] = h2f(xr[c]);
      double num = 0, den = 0;
      for (int r = 0; r < MID; r++) {
        double g = 0, u = 0;
        const float* pg = &wg[(size_t)r * D];
        const float* pu = &wu[(size_t)r * D];
        for (int c = 0; c < D; c++) { g += pg[c] * xf[c]; u += pu[c] * xf[c]; }
        double ref = u * g / (1.0 + std::exp(-g));
        double got = h2f(hid[(size_t)sl * MID + r]);
        num += (got - ref) * (got - ref);
        den += ref * ref;
      }
      double rel = std::sqrt(num / std::max(den, 1e-30));
      up_max = std::max(up_max, rel);
      up_sum += rel;
      num = den = 0;
      std::vector<double> hf(MID);
      for (int c = 0; c < MID; c++) hf[c] = h2f(hid[(size_t)sl * MID + c]);
      for (int r = 0; r < D; r++) {
        double y = 0;
        const float* pd = &wd[(size_t)r * MID];
        for (int c = 0; c < MID; c++) y += pd[c] * hf[c];
        double got = pairs[(size_t)sl * D + r];
        num += (got - y) * (got - y);
        den += y * y;
      }
      rel = std::sqrt(num / std::max(den, 1e-30));
      dn_max = std::max(dn_max, rel);
      dn_sum += rel;
      nchk++;
    }
  }
  printf("check: %zu experts, %d slots | up rel-L2 max %.3e mean %.3e | down rel-L2 max %.3e mean %.3e (tol %.1e)\n",
         ce.size(), nchk, up_max, up_sum / std::max(nchk, 1), dn_max, dn_sum / std::max(nchk, 1), tol);
  if (!(up_max <= tol) || !(dn_max <= tol) || nchk == 0) ok = false;

  // timing
  hipEvent_t e0, e1, e2;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  CK(hipEventCreate(&e2));
  float t_up = 0, t_dn = 0;
  for (int it = 0; it < iters; it++) {
    CK(hipEventRecord(e0, 0));
    run_up();
    CK(hipEventRecord(e1, 0));
    run_dn();
    CK(hipEventRecord(e2, 0));
    CK(hipEventSynchronize(e2));
    float a, b;
    CK(hipEventElapsedTime(&a, e0, e1));
    CK(hipEventElapsedTime(&b, e1, e2));
    if (it > 0 || iters == 1) { t_up += a; t_dn += b; }
  }
  int nt = iters > 1 ? iters - 1 : 1;
  t_up /= nt;
  t_dn /= nt;
  double fl_up = 2.0 * npairs * 2 * MID * (double)D, fl_dn = 2.0 * npairs * (double)D * MID;
  printf("time: up %.3f ms (%.1f TFLOPS)  down %.3f ms (%.1f TFLOPS)  total %.3f ms/layer = %.4f ms/tok over 48 layers\n",
         t_up, fl_up / t_up / 1e9, t_dn, fl_dn / t_dn / 1e9, t_up + t_dn, (t_up + t_dn) * 48 / P);
  printf("%s\n", ok ? "RESULT PASS" : "RESULT FAIL");
  return ok ? 0 : 1;
}
