// Unit tests for the Phase-3a batched kernels in gdec.cpp.
// gdec.cpp's main() is renamed via macro so we can link our own.
#define main gdec_real_main
#include "gdec.cpp"
#undef main

#include <cmath>
#include <random>
#include <vector>

static std::mt19937 rng(12345);
static float frand() { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); }

#define TCK(x)                                                          \
  do {                                                                  \
    hipError_t e_ = (x);                                                \
    if (e_ != hipSuccess) {                                             \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
              __FILE__, __LINE__);                                      \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

static int fails = 0;
static void check(const char* name, double maxrel, double tol) {
  bool ok = maxrel <= tol;
  printf("%-28s maxrel=%.3e tol=%.0e  %s\n", name, maxrel, tol, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

static void check_close(const char* name, const std::vector<float>& actual,
                        const std::vector<double>& expected) {
  // Atomic scatter can cancel large expert outputs. Use a tensor-scale absolute
  // floor (1e-6 of peak output), plus a per-element relative tolerance.
  constexpr double rtol = 1e-5;
  double scale = 1.0;
  for (double v : expected) scale = std::max(scale, fabs(v));
  const double atol = std::max(1e-4, 1e-6 * scale);
  double maxabs = 0, maxscaled = 0;
  bool ok = actual.size() == expected.size();
  if (ok)
    for (size_t i = 0; i < actual.size(); i++) {
      if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
        ok = false;
        break;
      }
      double error = fabs((double)actual[i] - expected[i]);
      maxabs = std::max(maxabs, error);
      maxscaled = std::max(maxscaled, error / (atol + rtol * fabs(expected[i])));
    }
  ok = ok && maxscaled <= 1.0;
  printf("%-28s maxabs=%.3e error/tol=%.3e (atol=%.0e rtol=%.0e)  %s\n",
         name, maxabs, maxscaled, atol, rtol, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

// CPU references -----------------------------------------------------------
static float bf16_round(float f) {  // RNE, same as f2bf
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  uint32_t r = u >> 16;
  u = r << 16;
  float out;
  memcpy(&out, &u, 4);
  return out;
}

static uint16_t bf16_bits(float f) {  // RNE bit pattern, same as f2bf
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

static float bf16_value(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float out;
  memcpy(&out, &u, 4);
  return out;
}

// MoE rounds codebook entries before scaling, not the dequantized weights.
// Keep the effective weights in fp64 to avoid an extra fp32 rounding here.
static std::vector<double> moe_ref_weights(const std::vector<uint8_t>& buf,
                                          int rows, int cols, int scale_stride) {
  float cb[16];
  memcpy(cb, buf.data(), sizeof cb);
  for (auto& v : cb) v = bf16_round(v);
  const uint8_t* codes = buf.data() + sizeof cb;
  const uint8_t* scales = codes + (size_t)rows * cols / 2;
  std::vector<double> weights((size_t)rows * cols);
  for (int r = 0; r < rows; r++)
    for (int k = 0; k < cols; k++) {
      size_t i = (size_t)r * cols + k;
      int code = (codes[i / 2] >> (4 * (i % 2))) & 15;
      __half hs;
      memcpy(&hs, scales + (size_t)r * scale_stride + (k / 32) * 2, sizeof hs);
      weights[i] = (double)cb[code] * __half2float(hs);
    }
  return weights;
}

// Q4C-P layout: [64B cb (16 f32)] [rows*cols/2 codes] [rows*scale_stride scales]
static std::vector<uint8_t> make_q4cp(int rows, int cols, int scale_stride,
                                      std::vector<float>& wref) {
  std::vector<uint8_t> buf(64 + (size_t)rows * cols / 2 + (size_t)rows * scale_stride);
  float* cb = (float*)buf.data();
  for (int i = 0; i < 16; i++) cb[i] = frand() * 2.f;
  uint8_t* codes = buf.data() + 64;
  uint8_t* scales = codes + (size_t)rows * cols / 2;
  wref.assign((size_t)rows * cols, 0.f);
  for (int r = 0; r < rows; r++) {
    for (int g = 0; g < cols / 32; g++) {
      float s = 0.5f + 0.5f * frand();
      __half hs = __float2half(s);
      memcpy(scales + r * scale_stride + g * 2, &hs, 2);
      s = __half2float(hs);  // kernel reads back the fp16 value
      for (int b = 0; b < 16; b++) {
        int lo = rng() % 16, hi = rng() % 16;
        codes[(size_t)r * cols / 2 + g * 16 + b] = (uint8_t)(lo | (hi << 4));
        wref[(size_t)r * cols + g * 32 + 2 * b] = cb[lo] * s;
        wref[(size_t)r * cols + g * 32 + 2 * b + 1] = cb[hi] * s;
      }
    }
  }
  return buf;
}

// q8g64 layout: per row [cols codes][cols/64 * (fp16 s, fp16 m)]
static std::vector<uint8_t> make_q8g64(int rows, int cols, std::vector<float>& wref) {
  int stride = cols + cols / 64 * 4;
  std::vector<uint8_t> buf((size_t)rows * stride);
  wref.assign((size_t)rows * cols, 0.f);
  for (int r = 0; r < rows; r++) {
    uint8_t* rp = buf.data() + (size_t)r * stride;
    for (int g = 0; g < cols / 64; g++) {
      float s = 0.01f + 0.005f * frand();
      float m = 0.5f * frand();
      __half hs = __float2half(s), hm = __float2half(m);
      __half* sm = (__half*)(rp + cols + g * 4);
      sm[0] = hs;
      sm[1] = hm;
      s = __half2float(hs);
      m = __half2float(hm);
      for (int j = 0; j < 64; j++) {
        uint8_t c = (uint8_t)(rng() % 256);
        rp[g * 64 + j] = c;
        wref[(size_t)r * cols + g * 64 + j] = c * s + m;
      }
    }
  }
  return buf;
}

static double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double mx = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
    double d = fabs((double)a[i] - b[i]);
    double denom = fabs((double)b[i]);
    if (denom > 1e-6) d /= denom;
    mx = std::max(mx, d);
  }
  return mx;
}

template <typename T>
static T* dup(const std::vector<T>& v) {
  T* p;
  TCK(hipMalloc(&p, v.size() * sizeof(T)));
  TCK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}
static float* dalloc(size_t n) {
  float* p;
  TCK(hipMalloc(&p, n * 4));
  TCK(hipMemset(p, 0, n * 4));
  return p;
}
static std::vector<float> dget(const float* p, size_t n) {
  std::vector<float> v(n);
  TCK(hipMemcpy(v.data(), p, n * 4, hipMemcpyDeviceToHost));
  // std::max can silently discard NaNs in the error summaries below.
  for (size_t i = 0; i < n; i++)
    if (!std::isfinite(v[i])) {
      fprintf(stderr, "FAIL: non-finite GPU output at element %zu\n", i);
      exit(1);
    }
  return v;
}

int main() {
  // ---- 1. k_rmsnorm_zc_grouped batched wrap (grid = 4*P, ngroups = 4) ----
  {
    const int P = 3, G = 4, N = 2560;
    std::vector<float> x(P * G * N), w(G * N);
    for (auto& v : x) v = frand();
    for (auto& v : w) v = frand();
    float *dx = dup(x), *dw = dup(w), *dy = dalloc(x.size());
    k_rmsnorm_zc_grouped<<<G * P, 1024>>>(dx, dw, dy, N, 1e-6f, G);
    std::vector<float> y = dget(dy, x.size()), ref(x.size());
    for (int t = 0; t < P * G; t++) {
      double ss = 0;
      for (int i = 0; i < N; i++) ss += (double)x[t * N + i] * x[t * N + i];
      float inv = 1.f / sqrtf((float)(ss / N) + 1e-6f);
      for (int i = 0; i < N; i++)
        ref[t * N + i] = x[t * N + i] * inv * (1.f + w[(t % G) * N + i]);
    }
    check("rmsnorm_zc_grouped_b", rel_diff(y, ref), 1e-5);
    TCK(hipFree(dx));
    TCK(hipFree(dw));
    TCK(hipFree(dy));
  }

  // ---- 1b. BF16 HC scatter+norm and mixed sigmoid combine ----
  {
    const int P = 3, G = 4, N = 2560;
    const float eps = 1e-6f;
    std::vector<uint16_t> R((size_t)P * G * N);
    std::vector<float> y((size_t)P * N), w((size_t)G * N), w4(P * G),
        gates((size_t)P * G * N);
    for (auto& v : R) v = bf16_bits(frand());
    std::vector<uint16_t> R0 = R;
    for (auto& v : y) v = frand() * 0.25f;
    for (auto& v : w) v = frand() * 0.25f;
    for (auto& v : w4) v = frand();
    for (auto& v : gates) v = frand();
    uint16_t *dR = dup(R), *dRh;
    TCK(hipMalloc(&dRh, R.size() * sizeof(uint16_t)));
    float *dy = dup(y), *dw = dup(w), *dw4 = dup(w4);
    k_gr_scatter_norm_b_hc_bf16<<<P * G, 256>>>(
        dw4, dR, dy, dw, dRh, N, G, P, eps);
    std::vector<uint16_t> Rgot(R.size()), Rhgot(R.size());
    TCK(hipMemcpy(Rgot.data(), dR, Rgot.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost));
    TCK(hipMemcpy(Rhgot.data(), dRh, Rhgot.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost));
    double rabs = 0, nhabs = 0;
    for (int t = 0; t < P; t++) {
      for (int b = 0; b < G; b++) {
        float inject = 2.f / (1.f + expf(-w4[t * G + b] * 0.25f));
        size_t base = ((size_t)t * G + b) * N;
        double ss = 0;
        for (int c = 0; c < N; c++) {
          float expected = bf16_round(bf16_value(R0[base + c]) +
                                      inject * y[(size_t)t * N + c]);
          rabs = std::max(rabs,
                          (double)fabs(bf16_value(Rgot[base + c]) - expected));
          float rv = bf16_value(Rgot[base + c]);
          ss += (double)rv * rv;
        }
        float inv = 1.f / sqrtf((float)(ss / N) + eps);
        for (int c = 0; c < N; c++) {
          float expected = bf16_round(bf16_value(Rgot[base + c]) * inv *
                                      (1.f + w[(size_t)b * N + c]));
          nhabs = std::max(nhabs,
                           (double)fabs(bf16_value(Rhgot[base + c]) - expected));
        }
      }
    }
    bool scatter_ok = rabs <= 8e-3 && nhabs <= 2e-2;
    printf("%-28s R_abs=%.3e norm_abs=%.3e %s\n", "gr_bf16_scatter_norm",
           rabs, nhabs, scatter_ok ? "PASS" : "FAIL");
    if (!scatter_ok) fails++;

    float* dg = dup(gates);
    float* dx = dalloc((size_t)P * N);
    uint16_t* dx16;
    TCK(hipMalloc(&dx16, (size_t)P * N * sizeof(uint16_t)));
    k_gr_combine_b_hc_bf16<<<P, 256>>>(dg, dRh, dx, N, G, P, dx16);
    std::vector<float> xgot = dget(dx, (size_t)P * N);
    std::vector<uint16_t> x16((size_t)P * N);
    TCK(hipMemcpy(x16.data(), dx16, x16.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost));
    double xabs = 0;
    bool packed_exact = true;
    for (int t = 0; t < P; t++)
      for (int c = 0; c < N; c++) {
        float acc = 0.f;
        for (int b = 0; b < G; b++) {
          size_t ri = ((size_t)t * G + b) * N + c;
          float gate = gates[ri];
          acc += (1.f / (1.f + expf(-gate))) * bf16_value(Rhgot[ri]);
        }
        float expected = acc / G;
        size_t i = (size_t)t * N + c;
        xabs = std::max(xabs, (double)fabs(xgot[i] - expected));
        packed_exact &= x16[i] == bf16_bits(xgot[i]);
      }
    bool combine_ok = xabs <= 2e-5 && packed_exact;
    printf("%-28s x_abs=%.3e packed_exact=%d %s\n", "gr_bf16_combine", xabs,
           (int)packed_exact, combine_ok ? "PASS" : "FAIL");
    if (!combine_ok) fails++;
    TCK(hipFree(dR));
    TCK(hipFree(dRh));
    TCK(hipFree(dy));
    TCK(hipFree(dw));
    TCK(hipFree(dw4));
    TCK(hipFree(dg));
    TCK(hipFree(dx));
    TCK(hipFree(dx16));
  }

  // ---- 2. k_router_topk batched (P blocks, ids/ws rows of 16) ----
  {
    const int P = 3, N = 512, K = 10;
    std::vector<float> lg(P * N);
    for (auto& v : lg) v = frand() * 4.f;
    float* dlg = dup(lg);
    int* dids;
    float* dws;
    TCK(hipMalloc(&dids, P * 16 * 4));
    TCK(hipMalloc(&dws, P * 16 * 4));
    k_router_topk<<<P, 512>>>(dlg, dids, dws, N, K);
    std::vector<int> ids(P * 16);
    std::vector<float> ws(P * 16);
    TCK(hipMemcpy(ids.data(), dids, P * 16 * 4, hipMemcpyDeviceToHost));
    TCK(hipMemcpy(ws.data(), dws, P * 16 * 4, hipMemcpyDeviceToHost));
    double mx = 0;
    for (int t = 0; t < P; t++) {
      std::vector<int> idx(N);
      for (int i = 0; i < N; i++) idx[i] = i;
      float m = *std::max_element(lg.begin() + t * N, lg.begin() + (t + 1) * N);
      std::vector<double> p(N);
      double se = 0;
      for (int i = 0; i < N; i++) {
        p[i] = exp((double)lg[t * N + i] - m);
        se += p[i];
      }
      std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                        [&](int a, int b) { return p[a] > p[b]; });
      double sw = 0;
      for (int j = 0; j < K; j++) sw += p[idx[j]] / se;
      for (int j = 0; j < K; j++) {
        if (ids[t * 16 + j] != idx[j]) {
          printf("  router t=%d j=%d: gpu=%d ref=%d\n", t, j, ids[t * 16 + j], idx[j]);
          mx = 1e9;
          continue;
        }
        double d = fabs(ws[t * 16 + j] - (p[idx[j]] / se) / sw);
        mx = std::max(mx, d);
      }
    }
    check("router_topk_b", mx, 1e-5);
    TCK(hipFree(dlg));
    TCK(hipFree(dids));
    TCK(hipFree(dws));
  }

  // ---- 3. k_q4cp_gemv_gg batched (slot -> tok/split decode) ----
  {
    const int E = 4, RP = 6, C = 64, P = 3, K = 2;  // experts, rows_per, cols
    const int SS = ((C / 32 * 2) + 15) & ~15;
    std::vector<float> wref;
    std::vector<uint8_t> w = make_q4cp(E * RP, C, SS, wref);
    std::vector<float> x(P * C);
    for (auto& v : x) v = frand();
    std::vector<int> ids(P * 16, 0);
    for (int t = 0; t < P; t++)
      for (int s = 0; s < K; s++) ids[t * 16 + s] = rng() % E;
    uint8_t* dw = dup(w);
    float* dx = dup(x);
    int* dids = dup(ids);
    float* dy = dalloc((size_t)P * K * RP);
    int nslots = P * K;
    uint64_t pairs = (uint64_t)nslots * ((RP + 1) / 2);  // v3: 2 rows/warp
    k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512>>>(
        dw + 64, dw + 64 + (size_t)E * RP * C / 2, (const float*)dw, dx, dy, dids, RP, C,
        SS, nslots, C /*x_stride*/, 16 /*id_stride*/, K);
    uint64_t total = (uint64_t)nslots * RP;
    std::vector<float> y = dget(dy, total), ref(total);
    for (int slot = 0; slot < nslots; slot++) {
      int tok = slot / K, s = slot % K;
      for (int r = 0; r < RP; r++) {
        double acc = 0;
        int row = ids[tok * 16 + s] * RP + r;
        for (int c = 0; c < C; c++) acc += (double)wref[(size_t)row * C + c] * x[tok * C + c];
        ref[slot * RP + r] = (float)acc;
      }
    }
    check("q4cp_gemv_gg_b", rel_diff(y, ref), 1e-4);
    TCK(hipFree(dw));
    TCK(hipFree(dx));
    TCK(hipFree(dids));
    TCK(hipFree(dy));
  }

  // ---- 4. k_q4cp_gemv_gd batched (weighted accumulate per token) ----
  {
    const int E = 4, RP = 5, C = 64, P = 3, K = 2;
    const int SS = ((C / 32 * 2) + 15) & ~15;
    std::vector<float> wref;
    std::vector<uint8_t> w = make_q4cp(E * RP, C, SS, wref);
    std::vector<float> x((size_t)P * K * C);  // per-slot x (x_stride = C)
    for (auto& v : x) v = frand();
    std::vector<int> ids(P * 16, 0);
    std::vector<float> ws(P * 16, 0);
    for (int t = 0; t < P; t++)
      for (int s = 0; s < K; s++) {
        ids[t * 16 + s] = rng() % E;
        ws[t * 16 + s] = frand();
      }
    uint8_t* dw = dup(w);
    float* dx = dup(x);
    int* dids = dup(ids);
    float* dws = dup(ws);
    float* dacc = dalloc((size_t)P * RP);
    int nslots = P * K;
    uint64_t pairs = (uint64_t)nslots * ((RP + 1) / 2);  // v3: 2 rows/warp
    k_q4cp_gemv_gd<<<(unsigned)((pairs + 15) / 16), 512>>>(
        dw + 64, dw + 64 + (size_t)E * RP * C / 2, (const float*)dw, dx, dacc, dids, dws,
        RP, C, SS, C /*x_stride*/, nslots, 16 /*id_stride*/, RP /*acc_stride*/, K);
    std::vector<float> acc = dget(dacc, (size_t)P * RP), ref((size_t)P * RP, 0.f);
    for (int slot = 0; slot < nslots; slot++) {
      int tok = slot / K, s = slot % K;
      for (int r = 0; r < RP; r++) {
        double a = 0;
        int row = ids[tok * 16 + s] * RP + r;
        for (int c = 0; c < C; c++)
          a += (double)wref[(size_t)row * C + c] * x[(size_t)slot * C + c];
        ref[tok * RP + r] += (float)(ws[tok * 16 + s] * a);
      }
    }
    check("q4cp_gemv_gd_b", rel_diff(acc, ref), 1e-4);
    TCK(hipFree(dw));
    TCK(hipFree(dx));
    TCK(hipFree(dids));
    TCK(hipFree(dws));
    TCK(hipFree(dacc));
  }

  // ---- 5. dequant kernels to bf16 ----
  {
    const int R = 8, C = 128;
    const int SS = ((C / 32 * 2) + 15) & ~15;
    std::vector<float> wref;
    std::vector<uint8_t> w = make_q4cp(R, C, SS, wref);
    uint8_t* dw = dup(w);
    uint16_t* dout;
    TCK(hipMalloc(&dout, R * C * 2));
    k_dequant_q4cp_bf16<<<(R * C / 32 + 255) / 256, 256>>>(
        dw + 64, dw + 64 + (size_t)R * C / 2, (const float*)dw, dout, R, C, SS);
    std::vector<uint16_t> o(R * C);
    TCK(hipMemcpy(o.data(), dout, R * C * 2, hipMemcpyDeviceToHost));
    double mx = 0;
    for (int i = 0; i < R * C; i++) {
      uint32_t u = (uint32_t)o[i] << 16;
      float f;
      memcpy(&f, &u, 4);
      double d = fabs(f - bf16_round(wref[i]));
      mx = std::max(mx, d);
    }
    check("dequant_q4cp_bf16", mx, 1e-6);
    TCK(hipFree(dw));
    TCK(hipFree(dout));
  }
  {
    const int R = 8, C = 128;
    std::vector<float> wref;
    std::vector<uint8_t> w = make_q8g64(R, C, wref);
    uint8_t* dw = dup(w);
    uint16_t* dout;
    TCK(hipMalloc(&dout, R * C * 2));
    k_dequant_q8g64_bf16<<<(R * C / 64 + 255) / 256, 256>>>(dw, dout, R, C);
    std::vector<uint16_t> o(R * C);
    TCK(hipMemcpy(o.data(), dout, R * C * 2, hipMemcpyDeviceToHost));
    double mx = 0;
    for (int i = 0; i < R * C; i++) {
      uint32_t u = (uint32_t)o[i] << 16;
      float f;
      memcpy(&f, &u, 4);
      double d = fabs(f - bf16_round(wref[i]));
      mx = std::max(mx, d);
    }
    check("dequant_q8g64_bf16", mx, 1e-6);
    TCK(hipFree(dw));
    TCK(hipFree(dout));
  }

  // ---- 6. k_gr_combine_b / k_gr_write_b / k_axpy_sg ----
  {
    const int P = 3, D = 64, B = 4;
    std::vector<float> G(P * B * D), Rh(P * B * D), R(P * B * D), y(P * D), w4(P * B),
        sg(P), ey(P * D), acc(P * D, 0.f);
    for (auto& v : G) v = frand();
    for (auto& v : Rh) v = frand();
    for (auto& v : R) v = frand();
    for (auto& v : y) v = frand();
    for (auto& v : w4) v = frand();
    for (auto& v : sg) v = frand();
    for (auto& v : ey) v = frand();
    float *dG = dup(G), *dRh = dup(Rh), *dR = dup(R), *dy = dup(y), *dw4 = dup(w4),
          *dsg = dup(sg), *dey = dup(ey), *dacc = dup(acc), *dx = dalloc(P * D);
    k_gr_combine_b<<<(P * D + 255) / 256, 256>>>(dG, dRh, dx, D, B, P);
    k_gr_write_b<<<(P * D + 255) / 256, 256>>>(dw4, dR, dy, D, B, P);
    k_axpy_sg<<<(P * D + 255) / 256, 256>>>(dacc, dey, dsg, D, P * D);
    std::vector<float> x = dget(dx, P * D), Rn = dget(dR, P * B * D),
                       accn = dget(dacc, P * D);
    std::vector<float> xref(P * D), Rref = R, acref(P * D);
    for (int t = 0; t < P; t++) {
      for (int i = 0; i < D; i++) {
        float a = 0;  // float, same order as the kernel
        for (int b = 0; b < B; b++)
          a += G[(t * B + b) * D + i] * Rh[(t * B + b) * D + i];
        xref[t * D + i] = a / B;
        float s = 1.f / (1.f + expf(-sg[t]));
        acref[t * D + i] = s * ey[t * D + i];
        for (int b = 0; b < B; b++) {
          float sc = 2.f / (1.f + expf(-w4[t * B + b] * 0.25f));
          Rref[(t * B + b) * D + i] += sc * y[t * D + i];
        }
      }
    }
    check("gr_combine_b", rel_diff(x, xref), 1e-4);  // FMA fusion noise
    check("gr_write_b", rel_diff(Rn, Rref), 1e-4);  // expf 1-ulp noise
    check("axpy_sg_b", rel_diff(accn, acref), 1e-6);
    TCK(hipFree(dG));
    TCK(hipFree(dRh));
    TCK(hipFree(dR));
    TCK(hipFree(dy));
    TCK(hipFree(dw4));
    TCK(hipFree(dsg));
    TCK(hipFree(dey));
    TCK(hipFree(dacc));
    TCK(hipFree(dx));
  }

  // ---- 7. k_f32_to_bf16 with row stride ----
  {
    const int P = 3, N = 100, XS = 160;  // N deliberately not multiple of anything
    std::vector<float> x(P * XS);
    for (auto& v : x) v = frand() * 10.f;
    float* dx = dup(x);
    uint16_t* dout;
    TCK(hipMalloc(&dout, P * N * 2));
    k_f32_to_bf16<<<(P * N + 255) / 256, 256>>>(dx, dout, XS, P, N);
    std::vector<uint16_t> o(P * N);
    TCK(hipMemcpy(o.data(), dout, P * N * 2, hipMemcpyDeviceToHost));
    double mx = 0;
    for (int t = 0; t < P; t++)
      for (int i = 0; i < N; i++) {
        uint32_t u = (uint32_t)o[t * N + i] << 16;
        float f;
        memcpy(&f, &u, 4);
        mx = std::max(mx, (double)fabs(f - bf16_round(x[t * XS + i])));
      }
    check("f32_to_bf16_stride", mx, 1e-6);
    TCK(hipFree(dx));
    TCK(hipFree(dout));
  }

  // ---- 7b. k_f32_to_bf16_v4: vectorized path, must be bit-exact vs scalar ----
  {
    const int P = 5, N = 104, XS = 160;  // N%4==0 and XS%4==0 required by v4
    std::vector<float> x(P * XS);
    for (auto& v : x) v = frand() * 10.f;
    float* dx = dup(x);
    uint16_t *dout, *dout4;
    TCK(hipMalloc(&dout, P * N * 2));
    TCK(hipMalloc(&dout4, P * N * 2));
    k_f32_to_bf16<<<(P * N + 255) / 256, 256>>>(dx, dout, XS, P, N);
    k_f32_to_bf16_v4<<<(P * N / 4 + 255) / 256, 256>>>(dx, dout4, XS, P, N);
    std::vector<uint16_t> o(P * N), o4(P * N);
    TCK(hipMemcpy(o.data(), dout, P * N * 2, hipMemcpyDeviceToHost));
    TCK(hipMemcpy(o4.data(), dout4, P * N * 2, hipMemcpyDeviceToHost));
    double mx = 0;
    for (int i = 0; i < P * N; i++)
      mx = std::max(mx, (double)fabs((int)o[i] - (int)o4[i]));
    check("f32_to_bf16_v4", mx, 1e-6);
    TCK(hipFree(dx));
    TCK(hipFree(dout));
    TCK(hipFree(dout4));
  }

  // ---- 8. k_rope_b: per-row positions, must match k_rope semantics ----
  {
    const int P = 37, NH = 24, DH = 256, ROT = 64;
    std::vector<float> v(P * NH * DH);
    for (auto& x : v) x = frand();
    float* dv = dup(v);
    ensure_rope_tab(1e7, ROT);
    k_rope_b<<<dim3((NH * 32 + 127) / 128, P), 128>>>(dv, NH, DH, ROT, 1e7);
    std::vector<float> out(P * NH * DH);
    TCK(hipMemcpy(out.data(), dv, out.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> ref = v;
    for (int t = 0; t < P; t++)
      for (int h = 0; h < NH; h++)
        for (int i = 0; i < ROT / 2; i++) {
          double ang = t * pow(1e7, -2.0 * i / ROT);
          float cs = (float)cos(ang), sn = (float)sin(ang);
          float* p = ref.data() + ((size_t)t * NH + h) * DH;
          float x0 = p[i], x1 = p[i + ROT / 2];
          p[i] = x0 * cs - x1 * sn;
          p[i + ROT / 2] = x0 * sn + x1 * cs;
        }
    double mabs = 0;  // abs diff: rel metric blows up on near-zero rotated values
    for (size_t i = 0; i < out.size(); i++)
      mabs = std::max(mabs, (double)fabs(out[i] - ref[i]));
    printf("rope_b                       maxabs=%.3e tol=1e-05  %s\n", mabs,
           mabs <= 1e-5 ? "PASS" : "FAIL");
    if (mabs > 1e-5) fails++;
    TCK(hipFree(dv));
  }

  // ---- 9. k_qsa_flash vs CPU two-pass softmax reference (tail block P=37) ----
  for (int P : {37, 64}) {
    const int HQ = 24, HKV = 2, DH = 256;
    std::vector<float> q((size_t)P * HQ * DH), kv((size_t)P * HKV * DH);
    for (auto& x : q) x = frand();
    for (auto& x : kv) x = frand();
    // kc/vc hold the same P rows (pos 0..P-1), interleaved by kv head
    float *dq = dup(q), *dkc = dup(kv), *dvc = dup(kv);
    float* dout = dalloc((size_t)P * HQ * DH);
    char nm[64];
    k_qsa_flash<<<dim3((P + 15) / 16, HQ), 128>>>(dq, dkc, dvc, dout, P, DH, HKV, HQ);
    std::vector<float> out = dget(dout, (size_t)P * HQ * DH);
    std::vector<float> ref((size_t)P * HQ * DH);
    float scale = 1.f / sqrtf((float)DH);
    for (int t = 0; t < P; t++)
      for (int h = 0; h < HQ; h++) {
        int kvh = h / (HQ / HKV);
        const float* qh = q.data() + ((size_t)t * HQ + h) * DH;
        float mx = -1e30f;
        std::vector<float> s(t + 1);
        for (int p = 0; p <= t; p++) {
          const float* kp = kv.data() + ((size_t)p * HKV + kvh) * DH;
          float a = 0;
          for (int i = 0; i < DH; i++) a += qh[i] * kp[i];
          s[p] = a * scale;
          mx = std::max(mx, s[p]);
        }
        float se = 0;
        for (int p = 0; p <= t; p++) se += expf(s[p] - mx);
        for (int i = 0; i < DH; i++) {
          float a = 0;
          for (int p = 0; p <= t; p++)
            a += expf(s[p] - mx) / se * kv[((size_t)p * HKV + kvh) * DH + i];
          ref[((size_t)t * HQ + h) * DH + i] = a;
        }
      }
    snprintf(nm, sizeof nm, "qsa_flash_P%d", P);
    double mabs = 0, mrel = 0;
    for (size_t i = 0; i < out.size(); i++) {
      double d = fabs((double)out[i] - ref[i]);
      mabs = std::max(mabs, d);
      if (fabs((double)ref[i]) > 1e-3) mrel = std::max(mrel, d / fabs((double)ref[i]));
    }
    printf("%-28s maxabs=%.3e maxrel=%.3e  %s\n", nm, mabs, mrel,
           (mabs <= 1e-4 && mrel <= 1e-3) ? "PASS" : "FAIL");
    if (mabs > 1e-4 || mrel > 1e-3) fails++;
    TCK(hipFree(dq));
    TCK(hipFree(dkc));
    TCK(hipFree(dvc));
    TCK(hipFree(dout));
  }

  // ---- 9b. k_qsa_flash over a BF16 KV cache (loads expand to fp32) ----
  for (int P : {37, 64}) {
    const int HQ = 24, HKV = 2, DH = 256;
    std::vector<float> q((size_t)P * HQ * DH), kv((size_t)P * HKV * DH);
    for (auto& x : q) x = frand();
    for (auto& x : kv) x = frand();
    // The GPU reads the rounded bits; the CPU reference uses the same values.
    std::vector<uint16_t> kvb(kv.size());
    std::vector<float> kvr(kv.size());
    for (size_t i = 0; i < kv.size(); i++) {
      kvb[i] = bf16_bits(kv[i]);
      kvr[i] = bf16_round(kv[i]);
    }
    float* dq = dup(q);
    uint16_t *dkc = dup(kvb), *dvc = dup(kvb);
    float* dout = dalloc((size_t)P * HQ * DH);
    char nm[64];
    k_qsa_flash<false, false, uint16_t>
        <<<dim3((P + 15) / 16, HQ), 128>>>(dq, dkc, dvc, dout, P, DH, HKV, HQ);
    std::vector<float> out = dget(dout, (size_t)P * HQ * DH);
    std::vector<float> ref((size_t)P * HQ * DH);
    float scale = 1.f / sqrtf((float)DH);
    for (int t = 0; t < P; t++)
      for (int h = 0; h < HQ; h++) {
        int kvh = h / (HQ / HKV);
        const float* qh = q.data() + ((size_t)t * HQ + h) * DH;
        float mx = -1e30f;
        std::vector<float> s(t + 1);
        for (int p = 0; p <= t; p++) {
          const float* kp = kvr.data() + ((size_t)p * HKV + kvh) * DH;
          float a = 0;
          for (int i = 0; i < DH; i++) a += qh[i] * kp[i];
          s[p] = a * scale;
          mx = std::max(mx, s[p]);
        }
        float se = 0;
        for (int p = 0; p <= t; p++) se += expf(s[p] - mx);
        for (int i = 0; i < DH; i++) {
          float a = 0;
          for (int p = 0; p <= t; p++)
            a += expf(s[p] - mx) / se * kvr[((size_t)p * HKV + kvh) * DH + i];
          ref[((size_t)t * HQ + h) * DH + i] = a;
        }
      }
    snprintf(nm, sizeof nm, "qsa_flash_bf16_P%d", P);
    double mabs = 0, mrel = 0;
    for (size_t i = 0; i < out.size(); i++) {
      double d = fabs((double)out[i] - ref[i]);
      mabs = std::max(mabs, d);
      if (fabs((double)ref[i]) > 1e-3) mrel = std::max(mrel, d / fabs((double)ref[i]));
    }
    printf("%-28s maxabs=%.3e maxrel=%.3e  %s\n", nm, mabs, mrel,
           (mabs <= 1e-4 && mrel <= 1e-3) ? "PASS" : "FAIL");
    if (mabs > 1e-4 || mrel > 1e-3) fails++;
    TCK(hipFree(dq));
    TCK(hipFree(dkc));
    TCK(hipFree(dvc));
    TCK(hipFree(dout));
  }

  // ---- 9c. k_qsa_step over FP32 and BF16 KV caches (dense, visible=100) ----
  for (int bf = 0; bf < 2; bf++) {
    const int HQ = 24, HKV = 2, DH = 256, VIS = 100;
    std::vector<float> q(HQ * DH), kv((size_t)VIS * HKV * DH);
    for (auto& x : q) x = frand();
    for (auto& x : kv) x = frand();
    std::vector<float> kvr = kv;
    if (bf)
      for (auto& x : kvr) x = bf16_round(x);
    float* dq = dup(q);
    float* dout = dalloc(HQ * DH);
    int pos = VIS - 1;
    int* dpos;
    TCK(hipMalloc(&dpos, 4));
    TCK(hipMemcpy(dpos, &pos, 4, hipMemcpyHostToDevice));
    if (bf) {
      std::vector<uint16_t> kvb(kv.size());
      for (size_t i = 0; i < kv.size(); i++) kvb[i] = bf16_bits(kv[i]);
      uint16_t *dkc = dup(kvb), *dvc = dup(kvb);
      k_qsa_step<uint16_t><<<HQ, 256>>>(dq, dkc, dvc, dout, dpos, DH, HKV, HQ);
      TCK(hipFree(dkc));
      TCK(hipFree(dvc));
    } else {
      float *dkc = dup(kv), *dvc = dup(kv);
      k_qsa_step<float><<<HQ, 256>>>(dq, dkc, dvc, dout, dpos, DH, HKV, HQ);
      TCK(hipFree(dkc));
      TCK(hipFree(dvc));
    }
    std::vector<float> out = dget(dout, HQ * DH);
    std::vector<float> ref(HQ * DH);
    float scale = 1.f / sqrtf((float)DH);
    for (int h = 0; h < HQ; h++) {
      int kvh = h / (HQ / HKV);
      const float* qh = q.data() + (size_t)h * DH;
      float mx = -1e30f;
      std::vector<float> s(VIS);
      for (int p = 0; p < VIS; p++) {
        const float* kp = kvr.data() + ((size_t)p * HKV + kvh) * DH;
        float a = 0;
        for (int i = 0; i < DH; i++) a += qh[i] * kp[i];
        s[p] = a * scale;
        mx = std::max(mx, s[p]);
      }
      float se = 0;
      for (int p = 0; p < VIS; p++) se += expf(s[p] - mx);
      for (int i = 0; i < DH; i++) {
        float a = 0;
        for (int p = 0; p < VIS; p++)
          a += expf(s[p] - mx) / se * kvr[((size_t)p * HKV + kvh) * DH + i];
        ref[(size_t)h * DH + i] = a;
      }
    }
    double mabs = 0, mrel = 0;
    for (size_t i = 0; i < out.size(); i++) {
      double d = fabs((double)out[i] - ref[i]);
      mabs = std::max(mabs, d);
      if (fabs((double)ref[i]) > 1e-3) mrel = std::max(mrel, d / fabs((double)ref[i]));
    }
    printf("%-28s maxabs=%.3e maxrel=%.3e  %s\n", bf ? "qsa_step_bf16" : "qsa_step_fp32",
           mabs, mrel, (mabs <= 1e-4 && mrel <= 1e-3) ? "PASS" : "FAIL");
    if (mabs > 1e-4 || mrel > 1e-3) fails++;
    TCK(hipFree(dq));
    TCK(hipFree(dout));
    TCK(hipFree(dpos));
  }

  // ---- 9d. k_store_kv_bf16: bit-exact RNE store, other slots untouched ----
  {
    const int NT = 8, POS = 3;
    std::vector<float> kb(512), vb(512);
    for (auto& x : kb) x = frand();
    for (auto& x : vb) x = frand();
    float* dkb = dup(kb);
    float* dvb = dup(vb);
    uint16_t *dkc, *dvc;
    TCK(hipMalloc(&dkc, (size_t)NT * 512 * 2));
    TCK(hipMalloc(&dvc, (size_t)NT * 512 * 2));
    TCK(hipMemset(dkc, 0xEE, (size_t)NT * 512 * 2));
    TCK(hipMemset(dvc, 0xEE, (size_t)NT * 512 * 2));
    int pos = POS;
    int* dpos;
    TCK(hipMalloc(&dpos, 4));
    TCK(hipMemcpy(dpos, &pos, 4, hipMemcpyHostToDevice));
    k_store_kv_bf16<<<2, 256>>>(dkb, dvb, dkc, dvc, dpos);
    TCK(hipGetLastError());
    std::vector<uint16_t> kc((size_t)NT * 512), vc((size_t)NT * 512);
    TCK(hipMemcpy(kc.data(), dkc, (size_t)NT * 512 * 2, hipMemcpyDeviceToHost));
    TCK(hipMemcpy(vc.data(), dvc, (size_t)NT * 512 * 2, hipMemcpyDeviceToHost));
    double bad = 0;
    for (int t = 0; t < NT; t++)
      for (int i = 0; i < 512; i++) {
        uint16_t ek = t == POS ? bf16_bits(kb[i]) : (uint16_t)0xEEEE;
        uint16_t ev = t == POS ? bf16_bits(vb[i]) : (uint16_t)0xEEEE;
        if (kc[(size_t)t * 512 + i] != ek) bad++;
        if (vc[(size_t)t * 512 + i] != ev) bad++;
      }
    check("qsa_store_bf16_bits", bad, 0);
    TCK(hipFree(dkb));
    TCK(hipFree(dvb));
    TCK(hipFree(dkc));
    TCK(hipFree(dvc));
    TCK(hipFree(dpos));
  }

  // ---- 9e. k_qsa_flash_bf16: sparse SharedV prefill over a BF16 KV cache ----
  // Mirrors the production BF16-mode launch in qsa_flash_b<uint16_t>: grid
  // (P-2051, 2), per-token top-512 block tables, 2048-slot block window plus
  // the incomplete tail (ntok = 2048 + visible%4; P=2056 covers tails 0..3).
  // Q is fp32 global (the kernel stages it to bf16 LDS with f2bf RNE), K/V
  // come from the bf16 cache, so the CPU reference computes in fp32 from the
  // same bf16-rounded q/k/v values, so the residual error is pure fp32
  // accumulation-order noise (measured maxabs 1.7e-7, maxrel 3.3e-5) and the
  // 9/9b tolerance applies. (The 2.8e-3 figure in the 32K bench was against
  // the fp32 kernel on *unrounded* inputs — a different comparison.)
  {
    const int P = 2056, HQ = 24, HKV = 2, DH = 256, FIRST = 2051;
    std::vector<float> q((size_t)P * HQ * DH), k((size_t)P * HKV * DH),
        v((size_t)P * HKV * DH);
    for (auto& x : q) x = frand();
    for (auto& x : k) x = frand();
    for (auto& x : v) x = frand();
    // The GPU reads the rounded bits; the CPU reference uses the same values.
    std::vector<float> qr(q.size()), kr(k.size()), vr(v.size());
    std::vector<uint16_t> kb(k.size()), vb(v.size());
    for (size_t i = 0; i < q.size(); i++) qr[i] = bf16_round(q[i]);
    for (size_t i = 0; i < k.size(); i++) {
      kb[i] = bf16_bits(k[i]);
      kr[i] = bf16_round(k[i]);
      vb[i] = bf16_bits(v[i]);
      vr[i] = bf16_round(v[i]);
    }
    std::vector<int> sel((size_t)P * 512, -1);
    for (int t = FIRST; t < P; t++) {
      int n = (t + 1) / 4;
      std::vector<int> blocks(n);
      for (int i = 0; i < n; i++) blocks[i] = i;
      std::shuffle(blocks.begin(), blocks.end(), rng);
      blocks.resize(512);
      std::sort(blocks.begin(), blocks.end());  // production ids are sorted
      for (int i = 0; i < 512; i++) sel[(size_t)t * 512 + i] = blocks[i];
    }
    float* dq = dup(q);
    uint16_t *dkc = dup(kb), *dvc = dup(vb);
    int* dsel = dup(sel);
    float* dout = dalloc((size_t)P * HQ * DH);
    k_qsa_flash_bf16<uint16_t><<<dim3(P - FIRST, HKV), 128>>>(
        dq, dkc, dvc, dout, P, DH, HKV, HQ, dsel + (size_t)FIRST * 512, FIRST);
    std::vector<float> out = dget(dout, (size_t)P * HQ * DH);
    std::vector<float> ref((size_t)P * HQ * DH, 0.f);
    float scale = 1.f / sqrtf((float)DH);
    for (int t = FIRST; t < P; t++) {
      int visible = t + 1;
      int ntok = 2048 + visible % 4;
      const int* blocks = sel.data() + (size_t)t * 512;
      for (int h = 0; h < HQ; h++) {
        int kvh = h / (HQ / HKV);
        const float* qh = qr.data() + ((size_t)t * HQ + h) * DH;
        float mx = -1e30f;
        std::vector<float> s(ntok);
        std::vector<int> src(ntok);
        for (int p = 0; p < ntok; p++) {
          src[p] = p < 2048 ? 4 * blocks[p / 4] + p % 4
                            : (visible / 4) * 4 + p - 2048;
          const float* kp = kr.data() + ((size_t)src[p] * HKV + kvh) * DH;
          float a = 0;
          for (int i = 0; i < DH; i++) a += qh[i] * kp[i];
          s[p] = a * scale;
          mx = std::max(mx, s[p]);
        }
        float se = 0;
        for (int p = 0; p < ntok; p++) se += expf(s[p] - mx);
        for (int i = 0; i < DH; i++) {
          float a = 0;
          for (int p = 0; p < ntok; p++)
            a += expf(s[p] - mx) / se * vr[((size_t)src[p] * HKV + kvh) * DH + i];
          ref[((size_t)t * HQ + h) * DH + i] = a;
        }
      }
    }
    double mabs = 0, mrel = 0;
    for (size_t i = 0; i < out.size(); i++) {
      double d = fabs((double)out[i] - ref[i]);
      mabs = std::max(mabs, d);
      if (fabs((double)ref[i]) > 1e-3) mrel = std::max(mrel, d / fabs((double)ref[i]));
    }
    printf("%-28s maxabs=%.3e maxrel=%.3e  %s\n", "qsa_flash_bf16lds_sp", mabs, mrel,
           (mabs <= 1e-4 && mrel <= 1e-3) ? "PASS" : "FAIL");
    if (mabs > 1e-4 || mrel > 1e-3) fails++;
    TCK(hipFree(dq));
    TCK(hipFree(dkc));
    TCK(hipFree(dvc));
    TCK(hipFree(dsel));
    TCK(hipFree(dout));
  }

  // ---- 9f. decode sparse flash: Split grid + combine vs unsplit reference ----
  // Same kernel, same per-chunk math; only the online-softmax grouping differs
  // (fp32 association), so expect ~1e-6, not bit-exact.
  {
    const int HQ = 24, HKV = 2, DH = 256, TOKEN = 2100;  // visible = 2101
    const int VIS = TOKEN + 1, NROW = 2112;
    std::vector<float> q(HQ * DH), k((size_t)NROW * HKV * DH),
        v((size_t)NROW * HKV * DH);
    for (auto& x : q) x = frand();
    for (auto& x : k) x = frand();
    for (auto& x : v) x = frand();
    std::vector<int> sel(512);
    for (int i = 0; i < 512; i++) sel[i] = rng() % 525;  // 4*524+3 = 2099 < VIS
    std::sort(sel.begin(), sel.end());
    float* dq = dup(q);
    float *dkc = dup(k), *dvc = dup(v);
    int* dsel = dup(sel);
    int* dpos = dup(std::vector<int>{TOKEN});
    float *dout1 = dalloc(HQ * DH), *dout2 = dalloc(HQ * DH);
    float* dpacc = dalloc((size_t)HKV * QSA_DEC_NSPLIT * (HQ / HKV) * DH);
    float* dpml = dalloc((size_t)HKV * QSA_DEC_NSPLIT * (HQ / HKV) * 2);
    k_qsa_flash<true, false, float><<<dim3(1, HKV), 128>>>(
        dq, dkc, dvc, dout1, 1, DH, HKV, HQ, dsel, 0, dpos);
    k_qsa_flash<true, false, float, true><<<dim3(QSA_DEC_NSPLIT, HKV), 128>>>(
        dq, dkc, dvc, dout2, 1, DH, HKV, HQ, dsel, 0, dpos, dpacc, dpml);
    k_qsa_flash_combine<<<dim3(HQ / HKV, HKV), 64>>>(dpml, dpacc, dout2,
                                                     HQ / HKV, DH, dpos);
    std::vector<float> o1 = dget(dout1, HQ * DH), o2 = dget(dout2, HQ * DH);
    double mabs = 0, mrel = 0;
    for (size_t i = 0; i < o1.size(); i++) {
      double d = fabs((double)o1[i] - o2[i]);
      mabs = std::max(mabs, d);
      if (fabs((double)o1[i]) > 1e-3) mrel = std::max(mrel, d / fabs((double)o1[i]));
    }
    printf("%-28s maxabs=%.3e maxrel=%.3e  %s\n", "qsa_flash_dec_split", mabs, mrel,
           (mabs <= 1e-4 && mrel <= 1e-3) ? "PASS" : "FAIL");
    if (mabs > 1e-4 || mrel > 1e-3) fails++;
    TCK(hipFree(dq));
    TCK(hipFree(dkc));
    TCK(hipFree(dvc));
    TCK(hipFree(dsel));
    TCK(hipFree(dpos));
    TCK(hipFree(dout1));
    TCK(hipFree(dout2));
    TCK(hipFree(dpacc));
    TCK(hipFree(dpml));
  }

  // ---- 9g. k_qsa_wmma6 (dim-split, 128 thr) vs k_qsa_wmma<true> + CPU ref ----
  // Same sparse setup as 9e, plus the transposed-V cache both WMMA kernels
  // read (vct[tok/4][kvh][dim][slot4]). Both kernels use bf16 P + WMMA
  // accumulation, so neither is bitwise fp32: v6 must land within 3x of v5's
  // measured error against the fp32 CPU reference (hard cap 1e-2).
  {
    const int P = 2056, HQ = 24, HKV = 2, DH = 256, FIRST = 2051;
    std::vector<float> q((size_t)P * HQ * DH), k((size_t)P * HKV * DH),
        v((size_t)P * HKV * DH);
    for (auto& x : q) x = frand();
    for (auto& x : k) x = frand();
    for (auto& x : v) x = frand();
    std::vector<float> qr(q.size()), kr(k.size()), vr(v.size());
    std::vector<uint16_t> kb(k.size()), vb(v.size());
    for (size_t i = 0; i < q.size(); i++) qr[i] = bf16_round(q[i]);
    for (size_t i = 0; i < k.size(); i++) {
      kb[i] = bf16_bits(k[i]);
      kr[i] = bf16_round(k[i]);
      vb[i] = bf16_bits(v[i]);
      vr[i] = bf16_round(v[i]);
    }
    std::vector<int> sel((size_t)P * 512, -1);
    for (int t = FIRST; t < P; t++) {
      int n = (t + 1) / 4;
      std::vector<int> blocks(n);
      for (int i = 0; i < n; i++) blocks[i] = i;
      std::shuffle(blocks.begin(), blocks.end(), rng);
      blocks.resize(512);
      std::sort(blocks.begin(), blocks.end());
      for (int i = 0; i < 512; i++) sel[(size_t)t * 512 + i] = blocks[i];
    }
    // transposed V cache, same bits the production k_f32_to_bf16_v4_bt emits
    std::vector<uint16_t> vct(((size_t)(P + 3) / 4) * HKV * DH * 4, 0);
    for (int t = 0; t < P; t++)
      for (int h = 0; h < HKV; h++)
        for (int d = 0; d < DH; d++)
          vct[(((size_t)(t / 4) * HKV + h) * DH + d) * 4 + t % 4] =
              vb[((size_t)t * HKV + h) * DH + d];
    float* dq = dup(q);
    uint16_t *dkc = dup(kb), *dvc = dup(vb), *dvct = dup(vct);
    int* dsel = dup(sel);
    float *dout5 = dalloc((size_t)P * HQ * DH), *dout6 = dalloc((size_t)P * HQ * DH),
          *dout7 = dalloc((size_t)P * HQ * DH);
    const int* sel0 = dsel + (size_t)FIRST * 512;
    k_qsa_wmma<true><<<dim3(P - FIRST, HKV), 256>>>(dq, dkc, dvc, dvct, dout5, P,
                                                    DH, HKV, HQ, sel0, FIRST);
    TCK(hipGetLastError());
    k_qsa_wmma6<<<dim3(P - FIRST, HKV), 128>>>(dq, dkc, dvc, dvct, dout6, P, DH,
                                               HKV, HQ, sel0, FIRST);
    TCK(hipGetLastError());
    // row-major register-transpose path (no vct)
    k_qsa_wmma<false><<<dim3(P - FIRST, HKV), 256>>>(dq, dkc, dvc, nullptr, dout7,
                                                     P, DH, HKV, HQ, sel0, FIRST);
    TCK(hipGetLastError());
    std::vector<float> o5 = dget(dout5, (size_t)P * HQ * DH),
                       o6 = dget(dout6, (size_t)P * HQ * DH),
                       o7 = dget(dout7, (size_t)P * HQ * DH);
    std::vector<float> ref((size_t)P * HQ * DH, 0.f);
    float scale = 1.f / sqrtf((float)DH);
    for (int t = FIRST; t < P; t++) {
      int visible = t + 1;
      int ntok = 2048 + visible % 4;
      const int* blocks = sel.data() + (size_t)t * 512;
      for (int h = 0; h < HQ; h++) {
        int kvh = h / (HQ / HKV);
        const float* qh = qr.data() + ((size_t)t * HQ + h) * DH;
        float mx = -1e30f;
        std::vector<float> s(ntok);
        std::vector<int> src(ntok);
        for (int p = 0; p < ntok; p++) {
          src[p] = p < 2048 ? 4 * blocks[p / 4] + p % 4
                            : (visible / 4) * 4 + p - 2048;
          const float* kp = kr.data() + ((size_t)src[p] * HKV + kvh) * DH;
          float a = 0;
          for (int i = 0; i < DH; i++) a += qh[i] * kp[i];
          s[p] = a * scale;
          mx = std::max(mx, s[p]);
        }
        float se = 0;
        for (int p = 0; p < ntok; p++) se += expf(s[p] - mx);
        for (int i = 0; i < DH; i++) {
          float a = 0;
          for (int p = 0; p < ntok; p++)
            a += expf(s[p] - mx) / se * vr[((size_t)src[p] * HKV + kvh) * DH + i];
          ref[((size_t)t * HQ + h) * DH + i] = a;
        }
      }
    }
    double mabs5 = 0, mrel5 = 0, mabs6 = 0, mrel6 = 0, mabs7 = 0, mrel7 = 0;
    for (size_t i = 0; i < ref.size(); i++) {
      double d5 = fabs((double)o5[i] - ref[i]), d6 = fabs((double)o6[i] - ref[i]),
             d7 = fabs((double)o7[i] - ref[i]);
      mabs5 = std::max(mabs5, d5);
      mabs6 = std::max(mabs6, d6);
      mabs7 = std::max(mabs7, d7);
      if (fabs((double)ref[i]) > 1e-3) {
        mrel5 = std::max(mrel5, d5 / fabs((double)ref[i]));
        mrel6 = std::max(mrel6, d6 / fabs((double)ref[i]));
        mrel7 = std::max(mrel7, d7 / fabs((double)ref[i]));
      }
    }
    // The 1e-2 absolute cap applies to maxabs; maxrel runs ~6e-2 for ALL
    // kernels (bf16 P on near-threshold ref elements), so its tolerance is
    // purely the 3x structural-bug bound relative to v5.
    double tola = std::min(3 * mabs5, 1e-2), tolr = 3 * mrel5;
    printf("%-28s maxabs=%.3e maxrel=%.3e  (reference row)\n", "qsa_wmma_btv", mabs5,
           mrel5);
    printf("%-28s maxabs=%.3e maxrel=%.3e tol=(%.1e, %.1e)  %s\n", "qsa_wmma6",
           mabs6, mrel6, tola, tolr,
           (mabs6 <= tola && mrel6 <= tolr) ? "PASS" : "FAIL");
    if (mabs6 > tola || mrel6 > tolr) fails++;
    printf("%-28s maxabs=%.3e maxrel=%.3e tol=(%.1e, %.1e)  %s\n",
           "qsa_wmma_rm", mabs7, mrel7, tola, tolr,
           (mabs7 <= tola && mrel7 <= tolr) ? "PASS" : "FAIL");
    if (mabs7 > tola || mrel7 > tolr) fails++;
    TCK(hipFree(dq));
    TCK(hipFree(dkc));
    TCK(hipFree(dvc));
    TCK(hipFree(dvct));
    TCK(hipFree(dsel));
    TCK(hipFree(dout5));
    TCK(hipFree(dout6));
    TCK(hipFree(dout7));
  }

  // ---- 10. k_qsa_qsplit batched (grid.y = token) ----
  {
    const int P = 5, DH = 256;
    std::vector<float> qg((size_t)P * 48 * DH), qnw(DH);
    for (auto& x : qg) x = frand();
    for (auto& x : qnw) x = frand();
    float* dqg = dup(qg);
    float* dqnw = dup(qnw);
    float* dqs = dalloc((size_t)P * 24 * DH);
    float* dgs = dalloc((size_t)P * 24 * DH);
    k_qsa_qsplit<<<dim3(24, P), 256>>>(dqg, dqnw, dqs, dgs, DH, 1e-6f);
    std::vector<float> qs = dget(dqs, (size_t)P * 24 * DH),
                       gs = dget(dgs, (size_t)P * 24 * DH);
    double mx = 0;
    for (int t = 0; t < P; t++)
      for (int h = 0; h < 24; h++) {
        const float* src = qg.data() + (size_t)t * 48 * DH + h * 2 * DH;
        float ss = 0;
        for (int i = 0; i < DH; i++) ss += src[i] * src[i];
        float inv = 1.f / sqrtf(ss / DH + 1e-6f);
        for (int i = 0; i < DH; i++) {
          size_t o = ((size_t)t * 24 + h) * DH + i;
          mx = std::max(mx, (double)fabs(qs[o] - src[i] * inv * (1.f + qnw[i])));
          mx = std::max(mx, (double)fabs(gs[o] - src[DH + i]));
        }
      }
    check("qsa_qsplit_b", mx, 1e-5);
    TCK(hipFree(dqg));
    TCK(hipFree(dqnw));
    TCK(hipFree(dqs));
    TCK(hipFree(dgs));
  }

  // ---- 10b. fused qsplit+rope (K1) must be bit-identical to the chain ----
  {
    const int P = 7, DH = 256, BASE = 13;
    std::vector<float> qg((size_t)P * 48 * DH), qnw(DH);
    for (auto& x : qg) x = frand();
    for (auto& x : qnw) x = frand();
    float* dqg = dup(qg);
    float* dqnw = dup(qnw);
    float* dqs1 = dalloc((size_t)P * 24 * DH);
    float* dgs1 = dalloc((size_t)P * 24 * DH);
    float* dqs2 = dalloc((size_t)P * 24 * DH);
    float* dgs2 = dalloc((size_t)P * 24 * DH);
    ensure_rope_tab(1e7, 64);
    float2* dcs;
    TCK(hipMalloc(&dcs, (size_t)P * 32 * sizeof(float2)));
    k_rope_cs<<<(P * 32 + 255) / 256, 256>>>(dcs, BASE, P);
    k_qsa_qsplit<<<dim3(24, P), 256>>>(dqg, dqnw, dqs1, dgs1, DH, 1e-6f);
    k_rope_b<<<dim3((24 * 32 + 127) / 128, P), 128>>>(dqs1, 24, DH, 64, 1e7, BASE);
    k_qsa_qsplit<true><<<dim3(24, P), 256>>>(dqg, dqnw, dqs2, dgs2, DH, 1e-6f, BASE,
                                             dcs);
    std::vector<float> qs1 = dget(dqs1, (size_t)P * 24 * DH),
                       qs2 = dget(dqs2, (size_t)P * 24 * DH),
                       gs1 = dget(dgs1, (size_t)P * 24 * DH),
                       gs2 = dget(dgs2, (size_t)P * 24 * DH);
    int bad = memcmp(qs1.data(), qs2.data(), qs1.size() * 4) |
              memcmp(gs1.data(), gs2.data(), gs1.size() * 4);
    printf("%-28s %s\n", "qsa_qsplit_rope_fused", bad ? "FAIL (bit diff)" : "PASS");
    fails += bad != 0;
    TCK(hipFree(dqg));
    TCK(hipFree(dqnw));
    TCK(hipFree(dqs1));
    TCK(hipFree(dgs1));
    TCK(hipFree(dqs2));
    TCK(hipFree(dgs2));
    TCK(hipFree(dcs));
  }

  // ---- 10c. fused kprep (K2) bit-identical to rmsnorm+rope[+convert] ----
  {
    const int P = 7, BASE = 13;
    std::vector<float> kb((size_t)P * 512), knw(512);  // 512: both heads defined
    for (auto& x : kb) x = frand();
    for (auto& x : knw) x = frand();
    float* dkb1 = dup(kb);
    float* dkb2 = dup(kb);
    float* dkb3 = dup(kb);
    float* dknw = dup(knw);
    uint16_t *dkc1, *dkc2;
    TCK(hipMalloc(&dkc1, (size_t)P * 512 * 2));
    TCK(hipMalloc(&dkc2, (size_t)P * 512 * 2));
    ensure_rope_tab(1e7, 64);
    float2* dcs;
    TCK(hipMalloc(&dcs, (size_t)P * 32 * sizeof(float2)));
    k_rope_cs<<<(P * 32 + 255) / 256, 256>>>(dcs, BASE, P);
    k_rmsnorm_zc_grouped<<<2 * P, 256>>>(dkb1, dknw, dkb1, 256, 1e-6f, 1);
    k_rope_b<<<dim3(1, P), 128>>>(dkb1, 2, 256, 64, 1e7, BASE);
    k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256>>>(
        dkb1, dkc1, 512, P, 512);
    k_qsa_kprep<true><<<dim3(2, P), 256>>>(dkb2, dknw, dkb2, dkc2, 1e-6f, BASE, dcs);
    std::vector<float> o1 = dget(dkb1, (size_t)P * 512),
                       o2 = dget(dkb2, (size_t)P * 512);
    std::vector<uint16_t> c1((size_t)P * 512), c2((size_t)P * 512);
    TCK(hipMemcpy(c1.data(), dkc1, c1.size() * 2, hipMemcpyDeviceToHost));
    TCK(hipMemcpy(c2.data(), dkc2, c2.size() * 2, hipMemcpyDeviceToHost));
    int bad = memcmp(o1.data(), o2.data(), o1.size() * 4) |
              memcmp(c1.data(), c2.data(), c1.size() * 2);
    printf("%-28s %s\n", "qsa_kprep_fused_bf16", bad ? "FAIL (bit diff)" : "PASS");
    fails += bad != 0;
    // fp32 KV variant: kprep<false> vs rmsnorm+rope only (o1 is post-chain kb)
    k_qsa_kprep<false><<<dim3(2, P), 256>>>(dkb3, dknw, dkb3, nullptr, 1e-6f, BASE,
                                            dcs);
    std::vector<float> o3 = dget(dkb3, (size_t)P * 512);
    bad = memcmp(o1.data(), o3.data(), o1.size() * 4);
    printf("%-28s %s\n", "qsa_kprep_fused_fp32", bad ? "FAIL (bit diff)" : "PASS");
    fails += bad != 0;
    TCK(hipFree(dkb1));
    TCK(hipFree(dkb2));
    TCK(hipFree(dkb3));
    TCK(hipFree(dknw));
    TCK(hipFree(dkc1));
    TCK(hipFree(dkc2));
    TCK(hipFree(dcs));
  }

  // ---- 11. MoE vs fp64 reference with BF16 operands (edge expert sizes) ----
  {
    const int E = 6, MID = 256, C = 256, D = 384, P = 65;
    const int counts[E] = {0, 1, 63, 64, 65, 2};  // 0/1/63/64/65 edges + tail
    const int NP = 195;                           // = sum(counts)
    const int SS = ((C / 32 * 2) + 15) & ~15;
    const int SSD = ((MID / 32 * 2) + 15) & ~15;
    std::vector<float> wup, wdn;
    std::vector<uint8_t> wu = make_q4cp(E * 2 * MID, C, SS, wup);
    std::vector<uint8_t> wd = make_q4cp(E * D, MID, SSD, wdn);
    auto wup_bf = moe_ref_weights(wu, E * 2 * MID, C, SS);
    auto wdn_bf = moe_ref_weights(wd, E * D, MID, SSD);
    std::vector<float> x((size_t)P * C);
    for (auto& v : x) v = frand();
    auto xbf = x;
    for (auto& v : xbf) v = bf16_round(v);
    std::vector<int> eoff(E + 1, 0), tokidx(NP);
    std::vector<float> pw(NP);
    for (int e = 0; e < E; e++) eoff[e + 1] = eoff[e] + counts[e];
    for (int p = 0; p < NP; p++) {
      tokidx[p] = rng() % P;
      pw[p] = 0.05f + 0.9f * fabsf(frand());
    }
    uint8_t *dwu = dup(wu), *dwd = dup(wd);
    float* dx = dup(x);
    int* dtok = dup(tokidx);
    float* dpw = dup(pw);
    int* deoff = dup(eoff);
    float* dhid = dalloc((size_t)NP * MID);
    float* dacc = dalloc((size_t)P * D);
    k_moe_w4_up<<<dim3(MID / MOE_CT, E), 256>>>(
        dwu + 64, dwu + 64 + (size_t)E * 2 * MID * C / 2, (const float*)dwu, dx, dtok,
        deoff, dhid, 2 * MID, C, MID, SS);
    std::vector<float> hid = dget(dhid, (size_t)NP * MID);
    std::vector<double> hidref((size_t)NP * MID), accref((size_t)P * D, 0.0),
                        pairref((size_t)NP * D);
    for (int e = 0; e < E; e++)
      for (int p = eoff[e]; p < eoff[e + 1]; p++) {
        const float* xt = xbf.data() + (size_t)tokidx[p] * C;
        for (int c = 0; c < MID; c++) {
          double g = 0, u = 0;
          for (int k = 0; k < C; k++) {
            g += wup_bf[((size_t)e * 2 * MID + c) * C + k] * xt[k];
            u += wup_bf[((size_t)e * 2 * MID + MID + c) * C + k] * xt[k];
          }
          hidref[(size_t)p * MID + c] = g / (1.0 + exp(-g)) * u;
        }
      }
    // Isolate down from up error; round its actual fp32 input to BF16 in the reference.
    std::vector<float> hidf(hidref.begin(), hidref.end());
    auto hidbf = hidf;
    for (auto& v : hidbf) v = bf16_round(v);
    TCK(hipMemcpy(dhid, hidf.data(), hidf.size() * 4, hipMemcpyHostToDevice));
    k_moe_w4_down<<<dim3(D / FD_CT_DN, E), 256>>>(
        dwd + 64, dwd + 64 + (size_t)E * D * MID / 2, (const float*)dwd, dhid, dtok,
        dpw, deoff, dacc, D, MID, SSD);
    std::vector<float> acc = dget(dacc, (size_t)P * D);
    for (int e = 0; e < E; e++)
      for (int p = eoff[e]; p < eoff[e + 1]; p++)
        for (int r = 0; r < D; r++) {
          double a = 0;
          for (int k = 0; k < MID; k++)
            a += wdn_bf[((size_t)e * D + r) * MID + k] * hidbf[(size_t)p * MID + k];
          pairref[(size_t)p * D + r] = pw[p] * a;
          accref[(size_t)tokidx[p] * D + r] += pairref[(size_t)p * D + r];
        }
    check_close("moe_w4_up", hid, hidref);
    check_close("moe_w4_down", acc, accref);
    const int tasks = (NP + 63) / 64 + E;
    MoeTile* dtiles;
    int* dntiles;
    TCK(hipMalloc(&dtiles, tasks * sizeof(MoeTile)));
    TCK(hipMalloc(&dntiles, sizeof(int)));
    k_moe_tiles<<<1, 512>>>(deoff, dtiles, dntiles, E);
    int ntiles;
    TCK(hipMemcpy(&ntiles, dntiles, sizeof(int), hipMemcpyDeviceToHost));
    std::vector<MoeTile> tiles(ntiles);
    TCK(hipMemcpy(tiles.data(), dtiles, ntiles * sizeof(MoeTile), hipMemcpyDeviceToHost));
    int ti = 0;
    bool layout_ok = true;
    for (int e = 0; e < E; e++)
      for (int p = eoff[e]; p < eoff[e + 1]; p += 64) {
        layout_ok &= ti < ntiles && tiles[ti].expert == e && tiles[ti].first == p &&
                     tiles[ti].count == std::min(64, eoff[e + 1] - p);
        ti++;
      }
    check("moe_tiles_layout", layout_ok && ti == ntiles ? 0 : 1, 0);
    float* dtiledhid = dalloc((size_t)NP * MID);
    k_moe_w4_up<true><<<dim3(MID / MOE_CT, tasks), 256>>>(
        dwu + 64, dwu + 64 + (size_t)E * 2 * MID * C / 2, (const float*)dwu, dx, dtok,
        deoff, dtiledhid, 2 * MID, C, MID, SS, dtiles, dntiles);
    check("moe_up_tiled_exact", hid == dget(dtiledhid, (size_t)NP * MID) ? 0 : 1, 0);
    float* dpairs = dalloc((size_t)NP * D);
    k_moe_w4_down<false><<<dim3(D / FD_CT_DN, E), 256>>>(
        dwd + 64, dwd + 64 + (size_t)E * D * MID / 2, (const float*)dwd, dhid, dtok,
        dpw, deoff, dpairs, D, MID, SSD);
    auto pairs = dget(dpairs, (size_t)NP * D);
    check_close("moe_w4_down_pairs", pairs, pairref);
    k_moe_w4_down<false, true><<<dim3(D / FD_CT_DN, tasks), 256>>>(
        dwd + 64, dwd + 64 + (size_t)E * D * MID / 2, (const float*)dwd, dhid, dtok,
        dpw, deoff, dpairs, D, MID, SSD, dtiles, dntiles);
    check("moe_down_tiled_exact", pairs == dget(dpairs, (size_t)NP * D) ? 0 : 1, 0);
    uint16_t *dx16, *dh16;
    TCK(hipMalloc(&dx16, (size_t)P * C * 2));
    TCK(hipMalloc(&dh16, (size_t)NP * MID * 2));
    k_f32_to_bf16<<<(P * C + 255) / 256, 256>>>(dx, dx16, C, P, C);
    k_moe_w4_up<true, true><<<dim3(MID / MOE_CT, tasks), 256>>>(
        dwu + 64, dwu + 64 + (size_t)E * 2 * MID * C / 2, (const float*)dwu, dx, dtok,
        deoff, nullptr, 2 * MID, C, MID, SS, dtiles, dntiles, dx16, dh16);
    std::vector<uint16_t> packed_hid((size_t)NP * MID), expected_hid(packed_hid.size());
    TCK(hipMemcpy(packed_hid.data(), dh16, packed_hid.size() * 2, hipMemcpyDeviceToHost));
    bool packed_finite = true;
    for (size_t i = 0; i < hid.size(); i++) {
      float rounded = bf16_round(hid[i]);
      uint32_t bits;
      memcpy(&bits, &rounded, 4);
      expected_hid[i] = bits >> 16;
      packed_finite &= std::isfinite(hid[i]);
    }
    check("moe_up_packed_exact", packed_finite && packed_hid == expected_hid ? 0 : 1, 0);
    k_moe_w4_up<true, true, false, false><<<dim3(MID / MOE_CT, tasks), 256>>>(
        dwu + 64, dwu + 64 + (size_t)E * 2 * MID * C / 2, (const float*)dwu, dx, dtok,
        deoff, nullptr, 2 * MID, C, MID, SS, dtiles, dntiles, dx16, dh16);
    TCK(hipMemcpy(packed_hid.data(), dh16, packed_hid.size() * 2, hipMemcpyDeviceToHost));
    check("moe_up_legacy_exact", packed_hid == expected_hid ? 0 : 1, 0);
    k_moe_w4_up<true, true, true, false><<<dim3(MID / MOE_CT, tasks), 256>>>(
        dwu + 64, dwu + 64 + (size_t)E * 2 * MID * C / 2, (const float*)dwu, dx, dtok,
        deoff, nullptr, 2 * MID, C, MID, SS, dtiles, dntiles, dx16, dh16);
    TCK(hipMemcpy(packed_hid.data(), dh16, packed_hid.size() * 2, hipMemcpyDeviceToHost));
    check("moe_up_row_only_exact", packed_hid == expected_hid ? 0 : 1, 0);
    k_f32_to_bf16<<<(NP * MID + 255) / 256, 256>>>(dhid, dh16, MID, NP, MID);
    k_moe_w4_down<false, true, true><<<dim3(D / FD_CT_DN, tasks), 256>>>(
        dwd + 64, dwd + 64 + (size_t)E * D * MID / 2, (const float*)dwd, nullptr, dtok,
        dpw, deoff, dpairs, D, MID, SSD, dtiles, dntiles, dh16);
    auto packed_pairs = dget(dpairs, (size_t)NP * D);
    bool pairs_finite = true;
    for (float value : packed_pairs) pairs_finite &= std::isfinite(value);
    check("moe_down_packed_exact", pairs_finite &&
              memcmp(pairs.data(), packed_pairs.data(), pairs.size() * 4) == 0 ? 0 : 1, 0);
    check_close("moe_w4_down_pairs", packed_pairs, pairref);
    k_moe_w4_down<false, true, true, false><<<dim3(D / FD_CT_DN, tasks), 256>>>(
        dwd + 64, dwd + 64 + (size_t)E * D * MID / 2, (const float*)dwd, nullptr, dtok,
        dpw, deoff, dpairs, D, MID, SSD, dtiles, dntiles, dh16);
    auto no_table_pairs = dget(dpairs, (size_t)NP * D);
    check("moe_down_no_table_exact", pairs_finite &&
              memcmp(packed_pairs.data(), no_table_pairs.data(), pairs.size() * 4) == 0 ? 0 : 1, 0);
    const int K = NP / P;
    std::vector<int> pairids(NP);
    std::vector<double> reducedref((size_t)P * D, 0.0);
    for (int t = 0; t < P; t++)
      for (int s = 0; s < K; s++) {
        int p = s * P + t;
        pairids[t * K + s] = p;
        for (int r = 0; r < D; r++) reducedref[t * D + r] += pairref[p * D + r];
      }
    auto* dmap = dup(pairids);
    k_moe_reduce<<<(P * D + 255) / 256, 256>>>(dpairs, dmap, dacc, P, K, D);
    auto reduced = dget(dacc, (size_t)P * D);
    check_close("moe_ordered_reduce", reduced, reducedref);
    k_moe_reduce_fast<K><<<dim3((D + 255) / 256, P), 256>>>(dpairs, dmap, dacc, P, D);
    auto fast_reduced = dget(dacc, (size_t)P * D);
    check("moe_reduce_fast_exact", memcmp(reduced.data(), fast_reduced.data(),
                                         reduced.size() * 4) == 0 ? 0 : 1, 0);
    k_moe_w4_down<false><<<dim3(D / FD_CT_DN, E), 256>>>(
        dwd + 64, dwd + 64 + (size_t)E * D * MID / 2, (const float*)dwd, dhid, dtok,
        dpw, deoff, dpairs, D, MID, SSD);
    k_moe_reduce<<<(P * D + 255) / 256, 256>>>(dpairs, dmap, dacc, P, K, D);
    check("moe_reduce_repeat_exact", reduced == dget(dacc, (size_t)P * D) ? 0 : 1, 0);
    TCK(hipFree(dpairs));
    TCK(hipFree(dx16)); TCK(hipFree(dh16));
    TCK(hipFree(dmap));
    TCK(hipFree(dtiledhid)); TCK(hipFree(dtiles)); TCK(hipFree(dntiles));
    TCK(hipFree(dwu));
    TCK(hipFree(dwd));
    TCK(hipFree(dx));
    TCK(hipFree(dtok));
    TCK(hipFree(dpw));
    TCK(hipFree(deoff));
    TCK(hipFree(dhid));
    TCK(hipFree(dacc));
  }

  // ---- 12. k_gdn_conv_b + k_convst_update vs per-token k_gdn_conv ----
  {
    const int P = 37, QKV = 10240;
    std::vector<float> x((size_t)P * QKV), cw(QKV * 4), st0(3 * QKV);
    for (auto& v : x) v = frand();
    for (auto& v : cw) v = frand() * 0.5f;
    for (auto& v : st0) v = frand();
    // reference: per-token in-place conv on a private copy
    std::vector<float> xref = x, stref = st0;
    float *dxref = dup(xref), *dcw = dup(cw), *dstref = dup(stref);
    for (int t = 0; t < P; t++)
      k_gdn_conv<<<(QKV + 255) / 256, 256>>>(dxref + (size_t)t * QKV, dcw, dstref,
                                             dxref + (size_t)t * QKV, QKV);
    // batched
    float* dx = dup(x);
    float* dst = dup(st0);
    float* dout = dalloc((size_t)P * QKV);
    int64_t tot = (int64_t)P * QKV;
    k_gdn_conv_b<<<(unsigned)((tot / 4 + 255) / 256), 256>>>(dx, dcw, dst, dout, P,
                                                             QKV);
    k_convst_update<<<(QKV + 255) / 256, 256>>>(dst, dx, P, QKV);
    std::vector<float> out = dget(dout, tot), ref = dget(dxref, tot),
                       stn = dget(dst, 3 * QKV), stre = dget(dstref, 3 * QKV);
    double cabs = 0;
    for (size_t i = 0; i < out.size(); i++)
      cabs = std::max(cabs, (double)fabs(out[i] - ref[i]));
    printf("gdn_conv_b                   maxabs=%.3e tol=1e-05  %s\n", cabs,
           cabs <= 1e-5 ? "PASS" : "FAIL");
    if (cabs > 1e-5) fails++;
    check("convst_update", rel_diff(stn, stre), 1e-6);
    TCK(hipFree(dxref));
    TCK(hipFree(dcw));
    TCK(hipFree(dstref));
    TCK(hipFree(dx));
    TCK(hipFree(dst));
    TCK(hipFree(dout));
  }

  // ---- 13. k_gdn_gates / k_l2norm_qk_b / k_gdn_gatednorm batched ----
  {
    const int P = 5, HV = 48;
    std::vector<float> a(P * HV), b(P * HV), al(HV), dt(HV);
    for (auto& v : a) v = frand() * 3.f;
    for (auto& v : b) v = frand() * 3.f;
    for (auto& v : al) v = frand();
    for (auto& v : dt) v = frand();
    float *da1 = dup(a), *db1 = dup(b), *dal = dup(al), *ddt = dup(dt);
    float *da2 = dup(a), *db2 = dup(b);
    k_gdn_gates<<<P, 64>>>(da1, db1, dal, ddt, da1, db1, HV, HV);
    for (int t = 0; t < P; t++)
      k_gdn_gates<<<1, 64>>>(da2 + t * HV, db2 + t * HV, dal, ddt, da2 + t * HV,
                             db2 + t * HV, HV, HV);
    std::vector<float> g1 = dget(da1, P * HV), g2 = dget(da2, P * HV);
    std::vector<float> b1 = dget(db1, P * HV), b2 = dget(db2, P * HV);
    check("gdn_gates_b", std::max(rel_diff(g1, g2), rel_diff(b1, b2)), 1e-6);
    TCK(hipFree(da1));
    TCK(hipFree(db1));
    TCK(hipFree(dal));
    TCK(hipFree(ddt));
    TCK(hipFree(da2));
    TCK(hipFree(db2));
  }
  {
    const int P = 5, QKV = 10240, DK = 128, HK = 16;
    std::vector<float> x((size_t)P * QKV);
    for (auto& v : x) v = frand() * 2.f;
    float* dx1 = dup(x);
    float* dx2 = dup(x);
    float qs = 1.f / sqrtf(128.f);
    k_l2norm_qk_b<<<dim3(HK, P), 128>>>(dx1, DK, qs, QKV, HK);
    for (int t = 0; t < P; t++)
      k_l2norm_qk<<<HK, 128>>>(dx2 + (size_t)t * QKV, dx2 + (size_t)t * QKV + 2048, DK,
                               qs);
    std::vector<float> o1 = dget(dx1, x.size()), o2 = dget(dx2, x.size());
    check("l2norm_qk_b", rel_diff(o1, o2), 1e-6);
    TCK(hipFree(dx1));
    TCK(hipFree(dx2));
  }
  {
    const int P = 5, HV = 48, DV = 128, QKV = 10240, ZS = 6144;
    std::vector<float> y((size_t)P * QKV, 0.f), z((size_t)P * ZS), w(DV);
    for (int t = 0; t < P; t++)
      for (int i = 0; i < HV * DV; i++) y[(size_t)t * QKV + 4096 + i] = frand();
    for (auto& v : z) v = frand();
    for (auto& v : w) v = frand();
    float* dy1 = dup(y);
    float* dy2 = dup(y);
    float* dz = dup(z);
    float* dw = dup(w);
    k_gdn_gatednorm<<<dim3(HV, P), 128>>>(dy1 + 4096, dz, dw, dy1 + 4096, DV, 1e-6f,
                                          QKV, ZS);
    for (int t = 0; t < P; t++)
      k_gdn_gatednorm<<<dim3(HV, 1), 128>>>(dy2 + (size_t)t * QKV + 4096,
                                            dz + (size_t)t * ZS, dw,
                                            dy2 + (size_t)t * QKV + 4096, DV, 1e-6f,
                                            QKV, ZS);
    std::vector<float> o1 = dget(dy1, y.size()), o2 = dget(dy2, y.size());
    check("gatednorm_b", rel_diff(o1, o2), 1e-6);
    // warp-per-token kernel vs the smem-tree kernel: same reduction order by
    // construction, so float out and bf16 out must be bit-identical.
    float* dy1b = dup(y);
    float* dy3 = dup(y);
    uint16_t *dbf1, *dbf3;
    TCK(hipMalloc(&dbf1, (size_t)P * HV * DV * 2));
    TCK(hipMalloc(&dbf3, (size_t)P * HV * DV * 2));
    k_gdn_gatednorm<true><<<dim3(HV, P), 128>>>(dy1b + 4096, dz, dw, dy1b + 4096, DV,
                                                1e-6f, QKV, ZS, dbf1);
    k_gdn_gatednorm_b<true><<<dim3(HV, (P + 31) / 32), 128>>>(
        dy3 + 4096, dz, dw, dy3 + 4096, 1e-6f, QKV, ZS, dbf3, P);
    std::vector<float> o1b = dget(dy1b, y.size()), o3 = dget(dy3, y.size());
    std::vector<uint16_t> bf1((size_t)P * HV * DV), bf3((size_t)P * HV * DV);
    TCK(hipMemcpy(bf1.data(), dbf1, bf1.size() * 2, hipMemcpyDeviceToHost));
    TCK(hipMemcpy(bf3.data(), dbf3, bf3.size() * 2, hipMemcpyDeviceToHost));
    double gabs = 0;
    int bfm = 0;
    for (size_t i = 0; i < o3.size(); i++)
      gabs = std::max(gabs, (double)fabs(o1b[i] - o3[i]));
    for (size_t i = 0; i < bf1.size(); i++) bfm += bf1[i] != bf3[i];
    printf("gatednorm_warp               maxabs=%.3e bfdiff=%d  %s\n", gabs, bfm,
           gabs == 0 && bfm == 0 ? "PASS" : "FAIL");
    if (gabs != 0 || bfm != 0) fails++;
    // null-out mode: fp32 store skipped (dead for the bf16-only consumer),
    // bf16 stream must stay bit-identical
    TCK(hipMemcpy(dy3, y.data(), y.size() * 4, hipMemcpyHostToDevice));
    k_gdn_gatednorm_b<true><<<dim3(HV, (P + 31) / 32), 128>>>(
        dy3 + 4096, dz, dw, nullptr, 1e-6f, QKV, ZS, dbf3, P);
    TCK(hipMemcpy(bf3.data(), dbf3, bf3.size() * 2, hipMemcpyDeviceToHost));
    bfm = 0;
    for (size_t i = 0; i < bf1.size(); i++) bfm += bf1[i] != bf3[i];
    printf("gatednorm_warp_nout          bfdiff=%d  %s\n", bfm,
           bfm == 0 ? "PASS" : "FAIL");
    if (bfm != 0) fails++;
    TCK(hipFree(dy1b));
    TCK(hipFree(dy3));
    TCK(hipFree(dbf1));
    TCK(hipFree(dbf3));
    TCK(hipFree(dy1));
    TCK(hipFree(dy2));
    TCK(hipFree(dz));
    TCK(hipFree(dw));
  }

  // ---- 14. k_gdn_chunk vs per-token k_gdn_step recurrence ----
  {
    hipFuncAttributes attr;
    TCK(hipFuncGetAttributes(&attr, (const void*)k_gdn_inter_strip));
    int maxb = 0;
    TCK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxb, (const void*)k_gdn_inter_strip, GDN_NT_STRIP, 0));
    printf("k_gdn_inter_strip: NT=%d regs=%d LDS=%zu maxBlocksPerCU=%d\n",
           GDN_NT_STRIP, attr.numRegs, attr.sharedSizeBytes, maxb);
  }
  for (int P : {37, 64, 128, 200, 257, 513}) {
    for (int nonzero_s : {0, 1}) {
      const int QKV = 10240, HV = 48, DK = 128;
      std::vector<float> x((size_t)P * QKV, 0.f), g((size_t)P * HV),
          be((size_t)P * HV), s0(HV * DK * DK);
      for (int t = 0; t < P; t++)
        for (int i = 0; i < QKV; i++) x[(size_t)t * QKV + i] = frand() * 0.5f;
      for (auto& v : g) v = -0.001f - 0.3f * fabsf(frand());
      for (auto& v : be) v = 0.05f + 0.9f * fabsf(frand());
      for (auto& v : s0) v = nonzero_s ? frand() * 0.1f : 0.f;
      // reference: per-token recurrence on private copies
      float* dxr = dup(x);
      float* dgr = dup(g);
      float* dbr = dup(be);
      float* dsr = dup(s0);
      for (int t = 0; t < P; t++)
        k_gdn_step<<<HV, 512>>>(dxr + (size_t)t * QKV, dxr + (size_t)t * QKV + 2048,
                                dxr + (size_t)t * QKV + 4096, dgr + (size_t)t * HV,
                                dbr + (size_t)t * HV, dsr, dxr + (size_t)t * QKV + 4096,
                                DK, DK, HV, 16);
      // chunked
      float* dxc = dup(x);
      float* dsc = dup(s0);
      float* dws = dalloc((size_t)HV * 2 * 64 * DK);
      k_gdn_chunk<<<HV, GDN_NT>>>(dxc, dgr, dbr, dsc, dxc, dws, P, QKV);
      std::vector<float> vr = dget(dxr, x.size()), vc = dget(dxc, x.size());
      std::vector<float> sr = dget(dsr, s0.size()), sc = dget(dsc, s0.size());
      // compare only v segments + S
      double mabs = 0, mrel = 0;
      for (int t = 0; t < P; t++)
        for (int i = 4096; i < QKV; i++) {
          double d = fabs((double)vr[(size_t)t * QKV + i] - vc[(size_t)t * QKV + i]);
          mabs = std::max(mabs, d);
          if (fabs((double)vr[(size_t)t * QKV + i]) > 1e-3)
            mrel = std::max(mrel, d / fabs((double)vr[(size_t)t * QKV + i]));
        }
      double sabs = 0;
      for (size_t i = 0; i < s0.size(); i++)
        sabs = std::max(sabs, (double)fabs(sr[i] - sc[i]));
      char nm[64];
      snprintf(nm, sizeof nm, "gdn_chunk_P%d_S%d", P, nonzero_s);
      printf("%-28s out_abs=%.3e out_rel=%.3e S_abs=%.3e  %s\n", nm, mabs, mrel, sabs,
             (mabs <= 2e-3 && mrel <= 2e-2 && sabs <= 2e-3) ? "PASS" : "FAIL");
      if (mabs > 2e-3 || mrel > 2e-2 || sabs > 2e-3) fails++;
      // split intra/inter path: same tolerances vs the recurrence, plus a
      // bitwise comparison against k_gdn_chunk (expected identical)
      {
        float* dxs = dup(x);
        float* dss = dup(s0);
        int nchunks = (P + 63) / 64;
        float* dws2 = dalloc((size_t)nchunks * HV * GDN_SPLIT_WS_FLOATS);
        k_gdn_intra<GDN_NT_INTRA><<<dim3(nchunks, HV), GDN_NT_INTRA>>>(dxs, dgr, dbr,
                                                                       dws2, P, QKV);
        k_gdn_inter<GDN_NT_INTER><<<HV, GDN_NT_INTER>>>(dxs, dss, dxs, dws2, P, QKV);
        std::vector<float> vs = dget(dxs, x.size()), ss = dget(dss, s0.size());
        double mabs2 = 0, mrel2 = 0;
        for (int t = 0; t < P; t++)
          for (int i = 4096; i < QKV; i++) {
            double d = fabs((double)vr[(size_t)t * QKV + i] - vs[(size_t)t * QKV + i]);
            mabs2 = std::max(mabs2, d);
            if (fabs((double)vr[(size_t)t * QKV + i]) > 1e-3)
              mrel2 = std::max(mrel2, d / fabs((double)vr[(size_t)t * QKV + i]));
          }
        double sabs2 = 0;
        for (size_t i = 0; i < s0.size(); i++)
          sabs2 = std::max(sabs2, (double)fabs(sr[i] - ss[i]));
        bool bitexact = vc == vs && sc == ss;
        snprintf(nm, sizeof nm, "gdn_split_P%d_S%d", P, nonzero_s);
        printf("%-28s out_abs=%.3e out_rel=%.3e S_abs=%.3e bitexact=%d  %s\n", nm,
               mabs2, mrel2, sabs2, (int)bitexact,
               (mabs2 <= 2e-3 && mrel2 <= 2e-2 && sabs2 <= 2e-3 && bitexact) ? "PASS"
                                                                            : "FAIL");
        if (mabs2 > 2e-3 || mrel2 > 2e-2 || sabs2 > 2e-3 || !bitexact) fails++;
        TCK(hipFree(dxs));
        TCK(hipFree(dss));
        TCK(hipFree(dws2));
      }
      // strip inter variant (8 column strips/head): bitwise vs k_gdn_chunk
      {
        float* dxt = dup(x);
        float* dst = dup(s0);
        int nchunks = (P + 63) / 64;
        float* dws3 = dalloc((size_t)nchunks * HV * GDN_SPLIT_WS_FLOATS);
        k_gdn_intra<GDN_NT_INTRA><<<dim3(nchunks, HV), GDN_NT_INTRA>>>(dxt, dgr, dbr,
                                                                       dws3, P, QKV);
        k_gdn_inter_strip<<<dim3(4, HV), GDN_NT_STRIP>>>(dxt, dst, dxt, dws3, P, QKV);
        std::vector<float> vt = dget(dxt, x.size()), st = dget(dst, s0.size());
        bool bitexact = vc == vt && sc == st;
        snprintf(nm, sizeof nm, "gdn_strip_P%d_S%d", P, nonzero_s);
        printf("%-28s bitexact=%d  %s\n", nm, (int)bitexact,
               bitexact ? "PASS" : "FAIL");
        if (!bitexact) fails++;
        TCK(hipFree(dxt));
        TCK(hipFree(dst));
        TCK(hipFree(dws3));
      }
      // Exercise window boundaries, partial chunks, nonzero state, and the
      // two independent stream/wave switches with byte-exact comparisons.
      for (bool wave : {false, true}) {
        float* dxw = dup(x);
        float* dsw = dup(s0);
        float* ws = dalloc((size_t)4 * HV * GDN_SPLIT_WS_FLOATS);
        for (int first = 0; first < P; first += 256) {
          int count = std::min(256, P - first);
          float* xx = dxw + (size_t)first * QKV;
          if (wave)
            k_gdn_intra<1024, true><<<dim3((count + 63) / 64, HV), 1024>>>(
                xx, dgr + first * HV, dbr + first * HV, ws, count, QKV);
          else
            k_gdn_intra<GDN_NT_INTRA><<<dim3((count + 63) / 64, HV), GDN_NT_INTRA>>>(
                xx, dgr + first * HV, dbr + first * HV, ws, count, QKV);
          k_gdn_inter_strip<<<dim3(4, HV), GDN_NT_STRIP>>>(xx, dsw, xx, ws, count, QKV);
        }
        auto vw = dget(dxw, x.size()), sw = dget(dsw, s0.size());
        bool exact = memcmp(vc.data(), vw.data(), vc.size() * sizeof(float)) == 0 &&
                     memcmp(sc.data(), sw.data(), sc.size() * sizeof(float)) == 0;
        snprintf(nm, sizeof nm, "gdn_window_P%d_S%d_W%d", P, nonzero_s, (int)wave);
        printf("%-28s bitexact=%d %s\n", nm, (int)exact, exact ? "PASS" : "FAIL");
        if (!exact) fails++;
        TCK(hipFree(dxw)); TCK(hipFree(dsw)); TCK(hipFree(ws));
      }
      // ut5 blocked solve: off-diagonal tiles reassociate in fp32, so compare
      // against the chunk reference with the gdn_chunk tolerances (no memcmp).
      {
        float* dxu = dup(x);
        float* dsu = dup(s0);
        float* ws = dalloc((size_t)4 * HV * GDN_SPLIT_WS_FLOATS);
        for (int first = 0; first < P; first += 256) {
          int count = std::min(256, P - first);
          float* xx = dxu + (size_t)first * QKV;
          k_gdn_intra<1024, true, true><<<dim3((count + 63) / 64, HV), 1024>>>(
              xx, dgr + first * HV, dbr + first * HV, ws, count, QKV);
          k_gdn_inter_strip<<<dim3(4, HV), GDN_NT_STRIP>>>(xx, dsu, xx, ws, count, QKV);
        }
        std::vector<float> vu = dget(dxu, x.size()), su = dget(dsu, s0.size());
        double mabs3 = 0, mrel3 = 0;
        for (int t = 0; t < P; t++)
          for (int i = 4096; i < QKV; i++) {
            double d = fabs((double)vc[(size_t)t * QKV + i] - vu[(size_t)t * QKV + i]);
            mabs3 = std::max(mabs3, d);
            if (fabs((double)vc[(size_t)t * QKV + i]) > 1e-3)
              mrel3 = std::max(mrel3, d / fabs((double)vc[(size_t)t * QKV + i]));
          }
        double sabs3 = 0;
        for (size_t i = 0; i < s0.size(); i++)
          sabs3 = std::max(sabs3, (double)fabs(sc[i] - su[i]));
        snprintf(nm, sizeof nm, "gdn_ut5_P%d_S%d", P, nonzero_s);
        printf("%-28s out_abs=%.3e out_rel=%.3e S_abs=%.3e  %s\n", nm, mabs3, mrel3,
               sabs3, (mabs3 <= 2e-3 && mrel3 <= 2e-2 && sabs3 <= 2e-3) ? "PASS" : "FAIL");
        if (mabs3 > 2e-3 || mrel3 > 2e-2 || sabs3 > 2e-3) fails++;
        TCK(hipFree(dxu)); TCK(hipFree(dsu)); TCK(hipFree(ws));
      }
      TCK(hipFree(dxr));
      TCK(hipFree(dgr));
      TCK(hipFree(dbr));
      TCK(hipFree(dsr));
      TCK(hipFree(dxc));
      TCK(hipFree(dsc));
      TCK(hipFree(dws));
    }
  }

  // ---- 15. PLE batched sequence vs per-token loop (incl. ring handoff) ----
  {
    const int P = 23, N = 10240, D = 2560;
    std::vector<float> keys((size_t)P * N), v((size_t)P * D), R((size_t)P * N),
        nk(3 * N), cw(N * 4);
    for (auto& x : keys) x = frand();
    for (auto& x : v) x = frand();
    for (auto& x : R) x = frand() * 2.f;
    for (auto& x : nk) x = frand() * 0.5f;
    for (auto& x : cw) x = frand() * 0.5f;
    std::vector<int> ringarr(P);
    for (int i = 0; i < P; i++) ringarr[i] = i % 9;
    const float eps = 1e-6f;
    // --- per-token loop path (mirrors ple_gpu_t) ---
    float *dkeys1 = dup(keys), *dv1 = dup(v), *dR1 = dup(R), *dnk = dup(nk),
          *dcw = dup(cw);
    int* dringarr = dup(ringarr);
    float* dring1 = dalloc(9 * N);
    float *dkey1 = dalloc(N), *dqn1 = dalloc(N), *dU1 = dalloc(N),
          *dco1 = dalloc(N), *dvt = dalloc(D), *dkn1 = dalloc(N);
    for (int t = 0; t < P; t++) {
      const float* kt = dkeys1 + (size_t)t * N;
      CK(hipMemcpy(dvt, dv1 + (size_t)t * D, D * 4, hipMemcpyDeviceToDevice));
      k_rmsnorm_zc_grouped<<<4, 1024>>>(kt, dnk, dkn1, D, eps, 4);
      k_rmsnorm_zc_grouped<<<4, 1024>>>(dR1 + (size_t)t * N, dnk + N, dqn1, D, eps, 4);
      k_ple_gate<<<4, 1024>>>(dkn1, dqn1, dvt, dU1, D, 4);
      k_rmsnorm_zc_ring<<<4, 1024>>>(dU1, dnk + 2 * N, dring1, dringarr + t, D, eps);
      k_ple_conv<<<(N + 255) / 256, 256>>>(dring1, dringarr + t, dcw, dco1, N);
      k_add2<<<(N + 255) / 256, 256>>>(dR1 + (size_t)t * N, dU1, dco1, N);
    }
    // --- batched path (mirrors ple_gpu_b post-GEMM stages) ---
    float *dkeys2 = dup(keys), *dv2 = dup(v), *dR2 = dup(R);
    float* dring2 = dalloc(9 * N);
    float *dqn2 = dalloc((size_t)P * N), *dU2 = dalloc((size_t)P * N),
          *dUn2 = dalloc((size_t)P * N), *dco2 = dalloc((size_t)P * N);
    k_rmsnorm_zc_grouped<<<4 * P, 1024>>>(dkeys2, dnk, dkeys2, D, eps, 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024>>>(dR2, dnk + N, dqn2, D, eps, 4);
    k_ple_gate_b<<<dim3(4, P), 1024>>>(dkeys2, dqn2, dv2, dU2, D, 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024>>>(dU2, dnk + 2 * N, dUn2, D, eps, 4);
    int64_t tot = (int64_t)P * N;
    k_ple_conv_b<<<(unsigned)((tot + 255) / 256), 256>>>(dUn2, dcw, dco2, P, N);
    k_add2<<<(unsigned)((tot + 255) / 256), 256>>>(dR2, dU2, dco2, (int)tot);
    k_ple_ring_wr<<<9, 256>>>(dUn2, dring2, P, N);
    std::vector<float> R1 = dget(dR1, tot), R2 = dget(dR2, tot);
    std::vector<float> g1 = dget(dring1, 9 * N), g2 = dget(dring2, 9 * N);
    double rabs = 0, gabs = 0;
    for (size_t i = 0; i < R1.size(); i++)
      rabs = std::max(rabs, (double)fabs(R1[i] - R2[i]));
    for (size_t i = 0; i < g1.size(); i++)
      gabs = std::max(gabs, (double)fabs(g1[i] - g2[i]));
    printf("ple_seq_b                    R_abs=%.3e ring_abs=%.3e tol=0  %s\n", rabs,
           gabs, (rabs == 0.0 && gabs == 0.0) ? "PASS" : "FAIL");
    if (rabs != 0.0 || gabs != 0.0) fails++;
    TCK(hipFree(dkeys1));
    TCK(hipFree(dv1));
    TCK(hipFree(dR1));
    TCK(hipFree(dnk));
    TCK(hipFree(dcw));
    TCK(hipFree(dringarr));
    TCK(hipFree(dring1));
    TCK(hipFree(dkey1));
    TCK(hipFree(dqn1));
    TCK(hipFree(dU1));
    TCK(hipFree(dco1));
    TCK(hipFree(dvt));
    TCK(hipFree(dkn1));
    TCK(hipFree(dkeys2));
    TCK(hipFree(dv2));
    TCK(hipFree(dR2));
    TCK(hipFree(dring2));
    TCK(hipFree(dqn2));
    TCK(hipFree(dU2));
    TCK(hipFree(dUn2));
    TCK(hipFree(dco2));
  }

  // ---- 16. PLE fp8 dequant: GPU vs host hgn::fp8e4m3_to_f32, bit-exact ----
  {
    const size_t n = 2560 * 37;  // 37 tokens, multiple of 16
    std::vector<uint8_t> bytes(n);
    for (size_t i = 0; i < n; i++) {
      uint8_t b = (uint8_t)(i * 167 + 13);  // sweeps all 256 codes
      if ((b & 0x7f) == 0x7f) b ^= 1;       // NaN codes: table has none; skip
      bytes[i] = b;
    }
    const float scale = 1.9945e-4f;
    auto ref_exact = [&](const std::vector<float>& got) {
      for (size_t i = 0; i < n; i++) {
        float ref = hgn::fp8e4m3_to_f32(bytes[i]) * scale;
        if (memcmp(&got[i], &ref, 4) != 0) return false;
      }
      return true;
    };
    const size_t n16 = n / 16;
    float* dout = dalloc(n);
    // device-buffer source
    uint8_t* db;
    TCK(hipMalloc(&db, n));
    TCK(hipMemcpy(db, bytes.data(), n, hipMemcpyHostToDevice));
    k_ple_fp8_dequant<<<(unsigned)((n16 + 255) / 256), 256>>>(db, dout, scale, n16);
    bool dev_ok = ref_exact(dget(dout, n));
    // pinned zero-copy source (production path)
    uint8_t *hb = nullptr, *hb_dev = nullptr;
    TCK(hipHostMalloc((void**)&hb, n, hipHostMallocDefault));
    TCK(hipHostGetDevicePointer((void**)&hb_dev, hb, 0));
    memcpy(hb, bytes.data(), n);
    k_ple_fp8_dequant<<<(unsigned)((n16 + 255) / 256), 256>>>(hb_dev, dout, scale,
                                                             n16);
    bool zc_ok = ref_exact(dget(dout, n));
    printf("%-28s dev=%d zerocopy=%d  %s\n", "ple_fp8_dequant", (int)dev_ok,
           (int)zc_ok, (dev_ok && zc_ok) ? "PASS" : "FAIL");
    if (!dev_ok || !zc_ok) fails++;
    TCK(hipFree(db));
    TCK(hipFree(dout));
    TCK(hipHostFree(hb));
  }

  // ---- 17. index select: radix-select vs sort-based, byte-identical ----
  {
    int fails17 = 0;
    auto run_case = [&](const char* tag, int n, int count, bool all_neg) {
      int stride = n + 3;  // exercise stride != n
      std::vector<float> sc((size_t)count * 4 * stride);
      std::mt19937 rr((unsigned)(n * 131 + count * 7));
      std::uniform_real_distribution<float> uf(-1.f, 1.f);
      for (auto& v : sc) v = floorf(uf(rr) * 50.f) / 50.f;  // coarse grid: ties
      if (all_neg)
        for (auto& v : sc) v = -fabsf(v);
      if (stride > 64)  // exact duplicate rows -> heavy threshold ties
        for (size_t t = 0; t < (size_t)count * 4; t++)
          for (int b = 0; b < 64; b++)
            sc[t * stride + 64 + b] = sc[t * stride + b];
      float* d_sc = dup(sc);
      int* sel_old = (int*)dalloc((size_t)count * 512);
      int* sel_new = (int*)dalloc((size_t)count * 512);
      int first = 4 * n - 1 - (count - 1);  // per-token n varies a little
      k_index_select<16><<<count, 256>>>(d_sc, stride, sel_old, first, nullptr);
      k_index_select_rs<<<count, 256>>>(d_sc, stride, sel_new, first, nullptr);
      std::vector<int> a((size_t)count * 512), b((size_t)count * 512);
      TCK(hipMemcpy(a.data(), sel_old, a.size() * 4, hipMemcpyDeviceToHost));
      TCK(hipMemcpy(b.data(), sel_new, b.size() * 4, hipMemcpyDeviceToHost));
      bool exact = memcmp(a.data(), b.data(), a.size() * 4) == 0;
      printf("%-28s n=%d count=%d allneg=%d exact=%d  %s\n", tag, n, count,
             (int)all_neg, (int)exact, exact ? "PASS" : "FAIL");
      if (!exact) fails17++;
      TCK(hipFree(d_sc));
      TCK(hipFree(sel_old));
      TCK(hipFree(sel_new));
    };
    run_case("index_select_rs", 300, 2, false);
    run_case("index_select_rs", 512, 2, false);
    run_case("index_select_rs", 513, 3, false);
    run_case("index_select_rs", 1000, 2, false);
    run_case("index_select_rs", 4096, 2, false);
    run_case("index_select_rs", 8192, 2, false);
    run_case("index_select_rs", 8192, 2, true);
    fails += fails17;
  }

  // ---- 18. ixrope: rope-table lookup in index_norm_rope is bit-identical to
  // the inline fp64 cos/sin, on both the Q path and the pooled-K path (incl.
  // the straddle block that falls back to inline below the table edge) ----
  {
    int fails19 = 0;
    auto run_ix = [&](int P, int BASE) {
      const float eps = 1e-6f;
      const double theta = 10000000.0;
      std::vector<float> proj((size_t)P * 640), norm(256);
      for (auto& v : proj) v = frand();
      for (auto& v : norm) v = frand() * 0.5f;
      float *dproj = dup(proj), *dnorm = dup(norm);
      float *dq1 = dalloc((size_t)P * 512), *dq2 = dalloc((size_t)P * 512);
      int nb = (BASE + P) / 4, b0 = BASE / 4, nblk = nb - b0;
      float *dk1 = dalloc((size_t)nb * 128), *dk2 = dalloc((size_t)nb * 128);
      float* draw = dalloc(512);
      ensure_rope_tab(theta, 64);
      float2* dcs;
      TCK(hipMalloc(&dcs, (size_t)P * 32 * sizeof(float2)));
      k_rope_cs<<<(P * 32 + 255) / 256, 256>>>(dcs, BASE, P);
      k_index_q<<<dim3(P, 4), 128>>>(dproj, dnorm, dq1, P, eps, theta, nullptr,
                                     BASE);
      k_index_q<<<dim3(P, 4), 128>>>(dproj, dnorm, dq2, P, eps, theta, nullptr,
                                     BASE, dcs);
      if (nblk > 0) {
        k_index_pool<<<nblk, 128>>>(dproj, dnorm + 128, dk1, eps, theta, BASE,
                                    draw);
        k_index_pool<<<nblk, 128>>>(dproj, dnorm + 128, dk2, eps, theta, BASE,
                                    draw, dcs);
      }
      std::vector<float> q1 = dget(dq1, (size_t)P * 512),
                         q2 = dget(dq2, (size_t)P * 512);
      std::vector<float> k1 = dget(dk1, (size_t)nb * 128),
                         k2 = dget(dk2, (size_t)nb * 128);
      int bad = memcmp(q1.data(), q2.data(), q1.size() * 4) |
                memcmp(k1.data(), k2.data(), k1.size() * 4);
      char nm[64];
      snprintf(nm, sizeof nm, "ixrope_P%d_base%d", P, BASE);
      printf("%-28s %s\n", nm, bad ? "FAIL (bit diff)" : "PASS");
      fails19 += bad != 0;
      TCK(hipFree(dproj));
      TCK(hipFree(dnorm));
      TCK(hipFree(dq1));
      TCK(hipFree(dq2));
      TCK(hipFree(dk1));
      TCK(hipFree(dk2));
      TCK(hipFree(draw));
      TCK(hipFree(dcs));
    };
    run_ix(517, 2051);  // base % 4 == 3: pool block 512 straddles, pos >= 2048
    run_ix(13, 0);      // position 0 covered, small ragged pool
    run_ix(8192, 8192); // aligned chunk shape
    fails += fails19;
  }

  // Short prefill/rejection replay must hand off the same convolution
  // history as sequential decode, including the next token's output.
  {
    constexpr int C = 10240;
    std::vector<float> hx(66 * C), hs(3 * C), hw(4 * C);
    for (auto& v : hx) v = frand();
    for (auto& v : hs) v = frand();
    for (auto& v : hw) v = frand();
    float *x = dup(hx), *w = dup(hw), *sr = dup(hs), *sg = dup(hs);
    float *yr = dalloc(C), *yg = dalloc(C);
    for (int P : {1,2,3,4,8,9,16,17,32,33,64,65}) {
      TCK(hipMemcpy(sr, hs.data(), hs.size()*4, hipMemcpyHostToDevice));
      TCK(hipMemcpy(sg, hs.data(), hs.size()*4, hipMemcpyHostToDevice));
      for (int t = 0; t < P; ++t)
        k_gdn_conv<<<(C+255)/256,256>>>(x+(size_t)t*C, w, sr, yr, C);
      k_convst_update<<<(C+255)/256,256>>>(sg, x, P, C);
      auto ref = dget(sr, 3*C), got = dget(sg, 3*C);
      bool pass = ref == got;
      k_gdn_conv<<<(C+255)/256,256>>>(x+(size_t)P*C, w, sr, yr, C);
      k_gdn_conv<<<(C+255)/256,256>>>(x+(size_t)P*C, w, sg, yg, C);
      pass = pass && dget(yr, C) == dget(yg, C);
      printf("convstate_handoff_P%-9d %s\n", P, pass ? "PASS" : "FAIL");
      fails += !pass;
    }
    for (float* ptr : {x, w, sr, sg, yr, yg}) TCK(hipFree(ptr));
  }

  printf(fails ? "== %d FAILURES ==\n" : "== ALL PASS ==\n", fails);
  return fails ? 1 : 0;
}
