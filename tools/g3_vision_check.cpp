// g3_vision_check — CPU check of gguf_map::build_vision (mmproj GGUF ->
// visual.* tensors) against the hgn vision file.
//
//   g++ -O2 -std=c++17 -pthread -Isrc tools/g3_vision_check.cpp -o build/g3_vision_check
//   build/g3_vision_check --mmproj <mmproj-BF16.gguf> --hgn <qwen38-flash-next-vision.hgn>
//
// Every hgn tensor must be produced with the same dtype and dims; bf16 data is
// expected to be bit-exact (HF source is bf16, the GGUF F32 tensors are exact
// upcasts). Reports per-tensor mismatches (count, max abs diff) and coverage.
// Ends with PASS / FAIL.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"
#include "gguf_map.h"
#include "hgn.h"

int main(int argc, char** argv) {
  std::string mm, hp;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--mmproj") mm = argv[i + 1];
    else if (a == "--hgn") hp = argv[i + 1];
  }
  if (mm.empty() || hp.empty()) {
    fprintf(stderr, "usage: %s --mmproj F.gguf --hgn vision.hgn\n", argv[0]);
    return 2;
  }
  try {
    gguf::File g(mm);
    hgn::Checkpoint ref(hp.c_str());
    hgn::Checkpoint ck;
    const size_t n = gguf_map::build_vision(ck, g);
    printf("built %zu tensors from %s\n", n, mm.c_str());

    std::vector<std::string> names;
    for (const auto& kv : ref.tensors()) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    size_t ok = 0, missing = 0, shape = 0, diff = 0, other = 0;
    double worst = 0;
    std::string worst_n;
    for (const std::string& nm : names) {
      const hgn::Tensor& r = ref.at(nm);
      if (nm.compare(0, 7, "visual.") != 0) { other++; continue; }
      const hgn::Tensor* b = ck.find(nm);
      if (!b) { printf("  MISSING %s\n", nm.c_str()); missing++; continue; }
      bool same = b->dtype == r.dtype && b->ndims == r.ndims;
      for (uint32_t d = 0; same && d < r.ndims; d++) same = b->dims[d] == r.dims[d];
      if (!same || r.dtype != 0) {
        printf("  SHAPE %s: hgn dtype %u [", nm.c_str(), r.dtype);
        for (uint32_t d = 0; d < r.ndims; d++) printf("%s%llu", d ? "," : "", (unsigned long long)r.dims[d]);
        printf("] built dtype %u [", b->dtype);
        for (uint32_t d = 0; d < b->ndims; d++) printf("%s%llu", d ? "," : "", (unsigned long long)b->dims[d]);
        printf("]\n");
        shape++;
        continue;
      }
      const uint64_t ne = r.numel();
      const uint16_t* x = (const uint16_t*)r.data;
      const uint16_t* y = (const uint16_t*)b->data;
      if (memcmp(x, y, ne * 2) == 0) { ok++; continue; }
      uint64_t nd = 0;
      double md = 0, mr = 0;
      for (uint64_t i = 0; i < ne; i++) {
        if (x[i] == y[i]) continue;
        nd++;
        const double a = hgn::bf16_to_f32(x[i]), c = hgn::bf16_to_f32(y[i]);
        md = std::max(md, std::fabs(a - c));
        mr = std::max(mr, std::fabs(a));
      }
      printf("  DIFF %s: %llu/%llu elems differ, max|d| %.3g (max|ref| %.3g)\n", nm.c_str(),
             (unsigned long long)nd, (unsigned long long)ne, md, mr);
      if (md > worst) worst = md, worst_n = nm;
      diff++;
    }
    size_t extra = 0;
    for (const auto& kv : ck.tensors())
      if (!ref.find(kv.first)) { printf("  EXTRA %s\n", kv.first.c_str()); extra++; }
    printf("hgn visual.* %zu: bit-exact %zu, differ %zu, shape %zu, missing %zu; extra built %zu;"
           " non-visual hgn %zu\n",
           names.size() - other, ok, diff, shape, missing, extra, other);
    if (diff) printf("worst diff %.3g in %s\n", worst, worst_n.c_str());
    const bool pass = ok > 0 && diff == 0 && shape == 0 && missing == 0 && extra == 0;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    printf("error: %s\nFAIL\n", e.what());
    return 1;
  }
}
