// moe_lut_test.cu — standalone check + bench of the LUT-decoded 4-bit routed
// expert WMMA kernels (src/gpu/parts/27_kernels_moe_lut.inc).
//
//   moe_lut_test --hgn W4B.hgn [--layer L]         real hgn q4cp experts
//   moe_lut_test --synth iq4nl|iq4xs [--E N]       synthetic GGUF IQ4 blocks
//                                                  (gate/up that type, down IQ4_NL)
//   common: [--P N] [--iters N] [--check-experts N] [--seed S] [--tol X]
//
// Routing: random top-10 (distinct experts per token, mildly skewed), like
// tools/moe_gguf_test.cu. Correctness: outputs are pre-filled with NaN and
// must be fully written; for --check-experts experts ALL their slots are
// compared with a CPU double reference built from the format's own
// dequantizer (hgn.h q4cp_row / gguf.h dequant_row):
//   up:   hid = up * silu(gate) from the F16-rounded x
//   down: pairs from the GPU's F16 hid (isolates the down kernel)
// Metric: per-slot relative L2 error; FAIL if max > --tol (default 1e-2).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../src/gguf.h"
#include "../src/hgn.h"

struct MoeTile { int expert, first, count; };
#include "../src/gpu/parts/26_kernels_moe_gguf.inc"
#include "../src/gpu/parts/27_kernels_moe_lut.inc"

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)

static float h2f(__half h) { return __half2float(h); }

static uint8_t* upload(const void* src, size_t n) {
  uint8_t* d;
  CK(hipMalloc(&d, n));
  CK(hipMemcpy(d, src, n, hipMemcpyHostToDevice));
  return d;
}

static uint16_t f2h_bits(float f) { return __builtin_bit_cast(uint16_t, __float2half(f)); }

// Synthetic IQ4 expert tensor [E][rows][cols] with plausible scales.
static std::vector<uint8_t> synth_iq4(int type, size_t E, size_t rows, size_t cols, std::mt19937& rng) {
  const size_t rb = type == moelut::kIQ4NL ? cols / 32 * 18 : cols / 256 * 136;
  std::vector<uint8_t> v(E * rows * rb);
  std::uniform_int_distribution<int> byte(0, 255), ls(0, 63);
  std::uniform_real_distribution<float> dd(0.2f, 1.0f);
  for (size_t r = 0; r < E * rows; r++) {
    uint8_t* p = v.data() + r * rb;
    if (type == moelut::kIQ4NL) {
      for (size_t b = 0; b < cols / 32; b++, p += 18) {
        uint16_t d = f2h_bits(dd(rng) * 2e-4f * (byte(rng) & 1 ? 1.f : -1.f));
        memcpy(p, &d, 2);
        for (int j = 0; j < 16; j++) p[2 + j] = (uint8_t)byte(rng);
      }
    } else {
      for (size_t b = 0; b < cols / 256; b++, p += 136) {
        uint16_t d = f2h_bits(dd(rng) * 1e-5f);
        memcpy(p, &d, 2);
        uint16_t sh = 0;
        uint8_t sl[4] = {0, 0, 0, 0};
        for (int ib = 0; ib < 8; ib++) {
          const int s = ls(rng);
          sl[ib / 2] |= (uint8_t)((s & 0xF) << (4 * (ib % 2)));
          sh |= (uint16_t)(((s >> 4) & 3) << (2 * ib));
        }
        memcpy(p + 2, &sh, 2);
        memcpy(p + 4, sl, 4);
        for (int j = 0; j < 128; j++) p[8 + j] = (uint8_t)byte(rng);
      }
    }
  }
  return v;
}

int main(int argc, char** argv) {
  std::string hgn_path, synth;
  int layer = 0, P = 2048, iters = 10, check_e = 8, topk = 10, E_synth = 128;
  unsigned seed = 1;
  double tol = 1e-2;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--hgn") hgn_path = argv[i + 1];
    else if (a == "--synth") synth = argv[i + 1];
    else if (a == "--layer") layer = atoi(argv[i + 1]);
    else if (a == "--P") P = atoi(argv[i + 1]);
    else if (a == "--E") E_synth = atoi(argv[i + 1]);
    else if (a == "--iters") iters = atoi(argv[i + 1]);
    else if (a == "--check-experts") check_e = atoi(argv[i + 1]);
    else if (a == "--seed") seed = (unsigned)atoi(argv[i + 1]);
    else if (a == "--tol") tol = atof(argv[i + 1]);
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (hgn_path.empty() == synth.empty()) {
    fprintf(stderr, "usage: %s (--hgn W4B.hgn [--layer L] | --synth iq4nl|iq4xs [--E N]) [--P N] ...\n", argv[0]);
    return 2;
  }
  std::mt19937 rng(seed);

  int D, MID, E, t_gu, t_dn_type;
  moelut::LutW pgu{}, pdn{};
  // CPU dequant of (expert, row) into out[cols]
  std::function<void(int, int, float*)> deq_gate, deq_up, deq_down;
  std::unique_ptr<hgn::Checkpoint> ck;
  std::vector<uint8_t> hg, hu, hd;  // synthetic host copies
  if (!hgn_path.empty()) {
    ck.reset(new hgn::Checkpoint(hgn_path.c_str()));
    char nm[128];
    snprintf(nm, sizeof nm, "layers.%d.mlp.experts.gate_up_proj.weight", layer);
    const hgn::Tensor& tgu = ck->at(nm);
    snprintf(nm, sizeof nm, "layers.%d.mlp.experts.down_proj.weight", layer);
    const hgn::Tensor& tdn = ck->at(nm);
    const auto qgu = hgn::Checkpoint::q4cp_parse(tgu);
    const auto qdn = hgn::Checkpoint::q4cp_parse(tdn);
    E = (int)tgu.dims[0];
    D = (int)qgu.cols;
    MID = (int)(qgu.rows / E / 2);
    if (qdn.cols != (uint64_t)MID || qdn.rows != (uint64_t)E * D) {
      fprintf(stderr, "unexpected shapes\n");
      return 2;
    }
    t_gu = t_dn_type = moelut::kQ4CP;
    const uint8_t* dgu = upload(tgu.data, tgu.data_size);
    const uint8_t* ddn = upload(tdn.data, tdn.data_size);
    pgu = lut_q4cp_view(dgu, qgu.rows, qgu.cols, qgu.scale_stride, 2 * MID, MID);
    pdn = lut_q4cp_view(ddn, qdn.rows, qdn.cols, qdn.scale_stride, D, 0);
    deq_gate = [=](int e, int r, float* o) { hgn::Checkpoint::q4cp_row(qgu, (uint64_t)e * 2 * MID + r, o); };
    deq_up = [=](int e, int r, float* o) { hgn::Checkpoint::q4cp_row(qgu, (uint64_t)e * 2 * MID + MID + r, o); };
    deq_down = [=](int e, int r, float* o) { hgn::Checkpoint::q4cp_row(qdn, (uint64_t)e * D + r, o); };
    printf("hgn layer %d: q4cp gate_up [%d][%d][%d], down [%d][%d][%d], P=%d top%d\n", layer, E, 2 * MID, D,
           E, D, MID, P, topk);
  } else {
    D = 2560;
    MID = 640;
    E = E_synth;
    if (synth == "iq4nl") t_gu = moelut::kIQ4NL;
    else if (synth == "iq4xs") t_gu = moelut::kIQ4XS;
    else { fprintf(stderr, "--synth iq4nl|iq4xs\n"); return 2; }
    t_dn_type = moelut::kIQ4NL;  // 640 is not a multiple of 256
    hg = synth_iq4(t_gu, E, MID, D, rng);
    hu = synth_iq4(t_gu, E, MID, D, rng);
    hd = synth_iq4(t_dn_type, E, D, MID, rng);
    pgu.w = upload(hg.data(), hg.size());
    pgu.w_up = upload(hu.data(), hu.size());
    pgu.e_rows = MID;
    pdn.w = upload(hd.data(), hd.size());
    pdn.e_rows = D;
    const size_t rbg = t_gu == moelut::kIQ4NL ? D / 32 * 18 : D / 256 * 136, rbd = MID / 32 * 18;
    const uint32_t tg = (uint32_t)t_gu;
    deq_gate = [&, rbg, tg](int e, int r, float* o) { gguf::dequant_row(tg, hg.data() + ((size_t)e * MID + r) * rbg, o, D); };
    deq_up = [&, rbg, tg](int e, int r, float* o) { gguf::dequant_row(tg, hu.data() + ((size_t)e * MID + r) * rbg, o, D); };
    deq_down = [&, rbd](int e, int r, float* o) { gguf::dequant_row(gguf::IQ4_NL, hd.data() + ((size_t)e * D + r) * rbd, o, MID); };
    printf("synthetic: gate/up %s [%d][%d][%d], down IQ4_NL [%d][%d][%d], P=%d top%d\n",
           t_gu == moelut::kIQ4NL ? "IQ4_NL" : "IQ4_XS", E, MID, D, E, D, MID, P, topk);
  }

  // routing (mild skew: expert popularity ~ 1/(1+e/64))
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
  std::vector<int> eoff(E + 1, 0), tokidx(npairs);
  for (int p = 0; p < npairs; p++) eoff[ids[p] + 1]++;
  for (int e = 0; e < E; e++) eoff[e + 1] += eoff[e];
  {
    std::vector<int> cur(eoff.begin(), eoff.end() - 1);
    for (int t = 0; t < P; t++)
      for (int s = 0; s < topk; s++) tokidx[cur[ids[(size_t)t * topk + s]]++] = t;
  }
  std::vector<MoeTile> tiles;
  for (int e = 0; e < E; e++)
    for (int r = eoff[e]; r < eoff[e + 1]; r += 64) tiles.push_back({e, r, std::min(64, eoff[e + 1] - r)});
  const int ntiles = (int)tiles.size();
  const int max_tiles = (npairs + 63) / 64 + E;  // engine's grid bound

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
  CK(hipMemset(d_hid, 0xFF, (size_t)npairs * MID * 2));  // NaN
  CK(hipMemset(d_pairs, 0xFF, (size_t)npairs * D * 4));  // NaN

  auto run_up = [&]() {
    if (!moe_lut_up(t_gu, pgu, d_x, d_tok, d_tiles, d_nt, max_tiles, d_hid, MID, D, 0)) {
      fprintf(stderr, "moe_lut_up: unsupported\n");
      exit(2);
    }
  };
  auto run_dn = [&]() {
    if (!moe_lut_down(t_dn_type, pdn, d_hid, d_tiles, d_nt, max_tiles, d_pairs, D, MID, 0)) {
      fprintf(stderr, "moe_lut_down: unsupported\n");
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

  std::vector<int> ce;
  for (int i = 0; i < E && (int)ce.size() < check_e; i++) {
    int e = (int)(((uint64_t)i * 7919 + 13) % E);
    if (eoff[e + 1] > eoff[e] && std::find(ce.begin(), ce.end(), e) == ce.end()) ce.push_back(e);
  }
  double up_max = 0, up_sum = 0, dn_max = 0, dn_sum = 0, ref_rms = 0;
  int nchk = 0;
  std::vector<float> wg((size_t)MID * D), wu((size_t)MID * D), wd((size_t)D * MID);
  for (int e : ce) {
    for (int r = 0; r < MID; r++) {
      deq_gate(e, r, &wg[(size_t)r * D]);
      deq_up(e, r, &wu[(size_t)r * D]);
    }
    for (int r = 0; r < D; r++) deq_down(e, r, &wd[(size_t)r * MID]);
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
      ref_rms += std::sqrt(den / MID);
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
  printf("check: %zu experts, %d slots | hid rms %.3e | up rel-L2 max %.3e mean %.3e | down rel-L2 max %.3e mean %.3e (tol %.1e)\n",
         ce.size(), nchk, ref_rms / std::max(nchk, 1), up_max, up_sum / std::max(nchk, 1), dn_max,
         dn_sum / std::max(nchk, 1), tol);
  if (!(up_max <= tol) || !(dn_max <= tol) || nchk == 0) ok = false;

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
