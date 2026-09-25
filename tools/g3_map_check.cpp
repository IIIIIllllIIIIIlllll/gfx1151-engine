// g3_map_check — CPU check of src/gguf_map.h against the production hgn stack.
//
//   g++ -O2 -std=c++17 -pthread -Isrc tools/g3_map_check.cpp -o build/g3_map_check
//   build/g3_map_check --gguf <main shard 1> [--mtp <mtp.gguf>] \
//       --hgn <base.hgn> [--hgn <overlay.hgn> ...]   (later files override)
//
// For sampled trunk layers + globals + MTP it builds the engine tensors from
// GGUF and compares each against the hgn dequant (rel-L2, cosine). hgn dense
// weights are 4-bit q4cp (rel-L2 ~0.1 vs the Q8_0 GGUF), so a correct mapping
// gives rel ~0.1 / cos > 0.99 and a wrong one rel >= 1 / cos ~0.
// Also: GDN v-head permutation A/B, norm +1-fold probe, coverage (every hgn
// non-expert tensor is produced), PLE table row sample (fp8 vs IQ4_NL).
// Ends with PASS / FAIL.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "gguf.h"
#include "gguf_map.h"
#include "hgn.h"

using hgn::Tensor;

static uint64_t cols_of(const Tensor& t) { return t.dims[t.ndims - 1]; }

static void row_deq(const Tensor& t, uint64_t r, float* out) {
  const uint64_t C = cols_of(t), n = t.numel();
  switch (t.dtype) {
    case 0: {
      const uint16_t* p = (const uint16_t*)t.data + r * C;
      for (uint64_t i = 0; i < C; i++) out[i] = hgn::bf16_to_f32(p[i]);
      return;
    }
    case 1:
      memcpy(out, t.data + r * C * 4, C * 4);
      return;
    case 5: {
      auto q = hgn::Checkpoint::q4cp_parse(t);
      hgn::Checkpoint::q4cp_row(q, r, out);
      return;
    }
    case 7: {
      const uint64_t st = C + C / 64 * 4;
      const uint8_t* rp = t.data + r * st;
      const uint16_t* sm = (const uint16_t*)(rp + C);
      for (uint64_t g = 0; g < C / 64; g++) {
        float s = hgn::fp16_to_f32(sm[2 * g]), m = hgn::fp16_to_f32(sm[2 * g + 1]);
        for (int j = 0; j < 64; j++) out[g * 64 + j] = rp[g * 64 + j] * s + m;
      }
      return;
    }
    case 8: {
      const int8_t* q = (const int8_t*)t.data + r * C;
      const uint16_t* s = (const uint16_t*)(t.data + n) + r * C / 32;
      for (uint64_t i = 0; i < C; i++) out[i] = q[i] * hgn::fp16_to_f32(s[i / 32]);
      return;
    }
    default:
      throw std::runtime_error("row_deq: dtype " + std::to_string(t.dtype) + " for " + t.name);
  }
}

struct Cmp {
  double rel = 0, cos = 0, rel_p1 = 0, rel_m1 = 0;
  uint64_t rows = 0;
};

// compare a (built) vs b (ref); sampled rows for big tensors
static Cmp compare(const Tensor& a, const Tensor& b, bool fold_probe) {
  const uint64_t C = cols_of(a), R = a.numel() / C;
  const uint64_t step = std::max<uint64_t>(1, a.numel() / (8u << 20));
  std::vector<float> x(C), y(C);
  double dd = 0, bb = 0, aa = 0, ab = 0, dp = 0, dm = 0;
  Cmp c;
  for (uint64_t r = 0; r < R; r += step) {
    row_deq(a, r, x.data());
    row_deq(b, r, y.data());
    for (uint64_t i = 0; i < C; i++) {
      double d = (double)x[i] - y[i];
      dd += d * d;
      bb += (double)y[i] * y[i];
      aa += (double)x[i] * x[i];
      ab += (double)x[i] * y[i];
      if (fold_probe) {
        dp += (d + 1) * (d + 1);
        dm += (d - 1) * (d - 1);
      }
    }
    c.rows++;
  }
  c.rel = bb > 0 ? sqrt(dd / bb) : sqrt(dd);
  c.cos = (aa > 0 && bb > 0) ? ab / sqrt(aa * bb) : (dd == 0 ? 1 : 0);
  c.rel_p1 = bb > 0 ? sqrt(dp / bb) : sqrt(dp);
  c.rel_m1 = bb > 0 ? sqrt(dm / bb) : sqrt(dm);
  return c;
}

static std::string kind_of(const std::string& n) {
  static const std::regex re("layers\\.[0-9]+\\.");
  return std::regex_replace(n, re, "layers.L.");
}

static bool is_norm(const std::string& n) {
  return n.find("norm") != std::string::npos && n.find("input_mix") == std::string::npos;
}

int main(int argc, char** argv) {
  std::string gpath, mpath;
  std::vector<std::string> hpaths;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--gguf") gpath = argv[i + 1];
    else if (a == "--mtp") mpath = argv[i + 1];
    else if (a == "--hgn") hpaths.push_back(argv[i + 1]);
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (gpath.empty() || hpaths.empty()) {
    fprintf(stderr, "usage: %s --gguf F [--mtp F] --hgn base [--hgn overlay ...]\n", argv[0]);
    return 2;
  }
  int fails = 0;
  try {
    hgn::Checkpoint ref(hpaths[0].c_str());
    for (size_t i = 1; i < hpaths.size(); i++) ref.add_overlay(hpaths[i].c_str());
    gguf::File g(gpath);
    std::unique_ptr<gguf::File> m;
    if (!mpath.empty()) m.reset(new gguf::File(mpath));
    printf("ref: %zu tensors (%zu files); gguf: %zu tensors%s\n", ref.tensor_count(),
           hpaths.size(), g.tensors().size(), m ? " + mtp" : "");

    const std::set<int> sample = {0, 1, 2, 3, 4, 7, 45, 46, 47};
    gguf_map::Opts o;
    o.layer = [&](int l) { return sample.count(l) > 0; };
    hgn::Checkpoint built;
    size_t nb = gguf_map::build(built, g, m.get(), o);
    printf("built %zu tensors (layers 0,1,2,3,4,7,45,46,47 + globals%s)\n", nb,
           m ? " + mtp" : "");

    // ---- per-tensor comparison, grouped by kind --------------------------
    struct Agg { int n = 0; double rel_max = 0, rel_sum = 0, cos_min = 2; bool bad = false;
                 std::string worst, hint; };
    std::map<std::string, Agg> agg;
    for (auto& kv : built.tensors()) {
      const Tensor& a = kv.second;
      const Tensor* b = ref.find(a.name);
      Agg& G = agg[kind_of(a.name)];
      G.n++;
      if (!b) { G.bad = true; G.hint = "missing in hgn"; continue; }
      if (a.numel() != b->numel()) {
        G.bad = true;
        G.hint = "numel " + std::to_string(a.numel()) + " vs " + std::to_string(b->numel());
        continue;
      }
      if (a.dtype == 4 || b->dtype == 4) {
        bool eq = a.data_size == b->data_size && !memcmp(a.data, b->data, a.data_size);
        if (!eq) { G.bad = true; G.hint = "u64 config differs"; }
        G.cos_min = std::min(G.cos_min, eq ? 1.0 : 0.0);
        continue;
      }
      if (b->dtype == 10) continue;
      Cmp c = compare(a, *b, is_norm(a.name));
      G.rel_sum += c.rel;
      if (c.rel > G.rel_max) { G.rel_max = c.rel; G.worst = a.name; }
      G.cos_min = std::min(G.cos_min, c.cos);
      if (c.rel > 0.35 || c.cos < 0.93) {
        G.bad = true;
        if (is_norm(a.name)) {
          char h[160];
          snprintf(h, sizeof h, "fold probe: rel(built+1)=%.3g rel(built-1)=%.3g", c.rel_p1,
                   c.rel_m1);
          G.hint = h;
        }
      }
    }
    printf("\n%-68s %3s %9s %9s %8s\n", "tensor kind", "n", "rel_mean", "rel_max", "cos_min");
    for (auto& kv : agg) {
      const Agg& G = kv.second;
      printf("%-68s %3d %9.4f %9.4f %8.5f %s\n", kv.first.c_str(), G.n,
             G.n ? G.rel_sum / G.n : 0, G.rel_max, G.cos_min, G.bad ? "FAIL" : "ok");
      if (G.bad) {
        fails++;
        if (!G.hint.empty()) printf("      %s\n", G.hint.c_str());
      }
    }

    // ---- GDN v-head permutation A/B (layer 0) ------------------------------
    {
      gguf_map::Opts o2;
      o2.layer = [](int l) { return l == 0; };
      o2.globals = false;
      o2.mtp = false;
      o2.vperm = false;
      hgn::Checkpoint alt;
      gguf_map::build(alt, g, nullptr, o2);
      printf("\nGDN v-head map, layer 0 (rel-L2 vs hgn):   %-10s %-10s\n", "tiled", "identity");
      for (const char* s : {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
                            "in_proj_b.weight", "out_proj.weight", "conv1d.weight", "A_log",
                            "dt_bias"}) {
        std::string n = std::string("layers.0.linear_attn.") + s;
        Cmp t = compare(built.at(n), ref.at(n), false), i = compare(alt.at(n), ref.at(n), false);
        printf("  %-40s %-10.4f %-10.4f\n", s, t.rel, i.rel);
      }
    }

    // ---- coverage ----------------------------------------------------------
    {
      static const std::regex lre("^layers\\.([0-9]+)\\.");
      int miss = 0;
      for (auto& kv : ref.tensors()) {
        const std::string& n = kv.first;
        if (n.find(".mlp.experts.") != std::string::npos) continue;
        if (n.find("ngram_embedding") != std::string::npos) continue;
        std::smatch mm;
        if (std::regex_search(n, mm, lre) && !sample.count(std::stoi(mm[1]))) continue;
        if (!m && n.compare(0, 4, "mtp.") == 0) continue;
        if (!built.find(n)) {
          if (miss < 20) printf("  not built: %s\n", n.c_str());
          miss++;
        }
      }
      printf("\ncoverage: %d hgn tensors (sampled layers/globals/mtp, excl. experts + PLE table) "
             "not produced%s\n", miss, miss ? "  FAIL" : "  ok");
      if (miss) fails++;
    }

    // ---- PLE table: hgn fp8 vs GGUF IQ4_NL, random rows ---------------------
    {
      const Tensor& pt = ref.at("layers.1.ple.ngram_embedding.weight");
      const gguf::Tensor& gt = g.at("per_layer_token_embd.weight");
      const uint64_t rows = pt.numel() / 160;
      float sc;
      memcpy(&sc, pt.data + pt.numel(), 4);
      printf("\nPLE table: hgn %llu rows (fp8, scale %g) vs gguf %llu rows (%s, %llu B/row)\n",
             (unsigned long long)rows, sc, (unsigned long long)gt.rows(),
             gguf::type_name(gt.type), (unsigned long long)gt.row_bytes());
      if (gt.rows() != rows || gt.ne[0] != 160) {
        printf("  row count / width mismatch  FAIL\n");
        fails++;
      } else {
        std::mt19937_64 rng(1234);
        double dd = 0, bb = 0, ab = 0, aa = 0, dx = 0;
        float x[160], y[160], z[160];
        for (int s = 0; s < 4096; s++) {
          uint64_t r = rng() % rows, r2 = rng() % rows;
          gguf::dequant_row(gt.type, gt.data + r * gt.row_bytes(), x, 160);
          gguf::dequant_row(gt.type, gt.data + r2 * gt.row_bytes(), z, 160);
          for (int i = 0; i < 160; i++) y[i] = hgn::fp8e4m3_to_f32(pt.data[r * 160 + i]) * sc;
          for (int i = 0; i < 160; i++) {
            double d = x[i] - y[i], e = z[i] - y[i];
            dd += d * d; dx += e * e; bb += (double)y[i] * y[i];
            aa += (double)x[i] * x[i]; ab += (double)x[i] * y[i];
          }
        }
        double rel = sqrt(dd / bb), cs = ab / sqrt(aa * bb), relx = sqrt(dx / bb);
        bool ok = rel < 0.35 && cs > 0.93;
        printf("  4096 random rows: rel-L2 %.4f cos %.5f (unrelated-row baseline rel %.3f)  %s\n",
               rel, cs, relx, ok ? "ok" : "FAIL");
        if (!ok) fails++;
      }
    }
  } catch (const std::exception& e) {
    printf("exception: %s\n", e.what());
    fails++;
  }
  printf("\n%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
