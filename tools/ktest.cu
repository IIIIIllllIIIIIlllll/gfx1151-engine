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

// Verbatim copy of k_gr_scatter_norm_inj_bf16 before the B2-1 y==nullptr
// mode was added; the current kernel's y path must stay bit-identical to it.
template <int T>
__global__ void __launch_bounds__(256) k_gr_scatter_norm_inj_bf16_old(
    const float* __restrict__ pfull, const float* __restrict__ pin,
    uint16_t* __restrict__ R, const float* __restrict__ y,
    const float* __restrict__ w, uint16_t* __restrict__ Rhat,
    const uint16_t* __restrict__ winj, float* __restrict__ pout, int d, int P,
    float eps) {
  const int b = blockIdx.x & 3;
  const int t0 = (blockIdx.x >> 2) * T;
  const int lane = threadIdx.x;
  __shared__ float sm[256];
  __shared__ float red[8][4];
  __shared__ float s_inj[T];
  if (lane < T) {
    const int t = t0 + lane;
    float v = 0.f;
    if (t < P) {
      const float x = pin ? inj_psum(pin + (size_t)t * 16 + b) : pfull[(size_t)t * 4 + b];
      v = 2.f / (1.f + expf(-x * 0.25f));
    }
    s_inj[lane] = v;
  }
  const float* wg = w + (size_t)b * d;
  float wn[10], wr[4][10];
#pragma unroll
  for (int j = 0; j < 10; j++) {
    const int c = lane + j * 256;
    wn[j] = c < d ? 1.f + wg[c] : 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++)
      wr[k][j] = c < d ? bf2f(winj[(size_t)k * 4 * d + (size_t)b * d + c]) : 0.f;
  }
  __syncthreads();
  for (int i = 0; i < T; i++) {
    const int t = t0 + i;
    if (t >= P) break;  // block-uniform
    uint16_t* Rg = R + ((size_t)t * 4 + b) * d;
    const float* yg = y + (size_t)t * d;
    uint16_t* og = Rhat + ((size_t)t * 4 + b) * d;
    const float inject = s_inj[i];
    float values[10];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 10; j++) {
      int c = lane + j * 256;
      float rounded = 0.f;
      if (c < d) {
        uint16_t r16 = f2bf(bf2f(Rg[c]) + inject * yg[c]);
        Rg[c] = r16;
        rounded = bf2f(r16);
        ss += rounded * rounded;
      }
      values[j] = rounded;
    }
    sm[lane] = ss;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
      if (lane < off) sm[lane] += sm[lane + off];
      __syncthreads();
    }
    float inv = rsqrtf(sm[0] / d + eps);
    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 10; j++) {
      int c = lane + j * 256;
      if (c < d) {
        const uint16_t o = f2bf(values[j] * inv * wn[j]);
        og[c] = o;
        const float of = bf2f(o);
#pragma unroll
        for (int k = 0; k < 4; k++) a[k] += of * wr[k][j];
      }
    }
#pragma unroll
    for (int k = 0; k < 4; k++)
#pragma unroll
      for (int o = 16; o; o >>= 1) a[k] += __shfl_xor(a[k], o, 32);
    if ((lane & 31) == 0)
#pragma unroll
      for (int k = 0; k < 4; k++) red[lane >> 5][k] = a[k];
    __syncthreads();
    if (lane < 4) {
      float s = red[0][lane];
#pragma unroll
      for (int wv = 1; wv < 8; wv++) s += red[wv][lane];
      pout[((size_t)t * 4 + b) * 4 + lane] = s;
    }
  }
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
    k_gr_combine_b_hc_bf16<<<P, 256>>>(dg, dRh, dx, N, G, P, dx16, N);
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
    // column split (verify P<=8 path): dim3(P,4), cspan=N/4 must be bit-exact
    {
      float* dx2 = dalloc((size_t)P * N);
      uint16_t* dx16b;
      TCK(hipMalloc(&dx16b, (size_t)P * N * sizeof(uint16_t)));
      TCK(hipMemset(dx2, 0xCD, (size_t)P * N * 4));
      TCK(hipMemset(dx16b, 0xAB, (size_t)P * N * 2));
      k_gr_combine_b_hc_bf16<<<dim3(P, 4), 256>>>(dg, dRh, dx2, N, G, P, dx16b,
                                                  N / 4);
      std::vector<float> xgot2 = dget(dx2, (size_t)P * N);
      std::vector<uint16_t> x16b((size_t)P * N);
      TCK(hipMemcpy(x16b.data(), dx16b, x16b.size() * sizeof(uint16_t),
                    hipMemcpyDeviceToHost));
      size_t mism = 0;
      for (size_t i = 0; i < (size_t)P * N; i++)
        mism += (xgot2[i] != xgot[i]) + (x16b[i] != x16[i]);
      bool cs_ok = mism == 0;
      printf("%-28s mismatches=%zu %s\n", "gr_combine_colsplit", mism,
             cs_ok ? "PASS" : "FAIL");
      if (!cs_ok) fails++;
      TCK(hipFree(dx2));
      TCK(hipFree(dx16b));
    }
    TCK(hipFree(dR));
    TCK(hipFree(dRh));
    TCK(hipFree(dy));
    TCK(hipFree(dw));
    TCK(hipFree(dw4));
    TCK(hipFree(dg));
    TCK(hipFree(dx));
    TCK(hipFree(dx16));
  }

  // ---- 1c. k_f32_gemv_mr (iproj P<=8 decode path): vs fp64 CPU ref ----
  {
    const int rows = 640, cols = 2560;
    std::vector<float> w((size_t)rows * cols), x4((size_t)4 * cols);
    for (auto& v : w) v = frand();
    for (auto& v : x4) v = frand();
    float* dw = dup(w);
    float* dx4 = dup(x4);
    float* dy4 = dalloc((size_t)4 * rows);
    float* dy1 = dalloc((size_t)rows);
    k_f32_gemv_mr<4><<<(rows + 15) / 16, 512>>>(dw, dx4, dy4, rows, cols);
    k_f32_gemv_mr<1><<<(rows + 15) / 16, 512>>>(dw, dx4, dy1, rows, cols);
    std::vector<float> y4 = dget(dy4, (size_t)4 * rows);
    std::vector<float> y1 = dget(dy1, rows);
    double mx = 0;
    size_t p1mism = 0;
    for (int p = 0; p < 4; p++)
      for (int r = 0; r < rows; r++) {
        double ref = 0;
        for (int k = 0; k < cols; k++)
          ref += (double)w[(size_t)r * cols + k] * x4[(size_t)p * cols + k];
        mx = std::max(mx, fabs(y4[(size_t)p * rows + r] - ref) /
                              std::max(1.0, fabs(ref)));
        if (p == 0 && y1[r] != y4[r]) p1mism++;
      }
    printf("%-28s p1_vs_p4row0_mism=%zu %s\n", "f32_gemv_mr_p1", p1mism,
           p1mism == 0 ? "PASS" : "FAIL");
    if (p1mism) fails++;
    check("f32_gemv_mr", mx, 1e-4);
    TCK(hipFree(dw));
    TCK(hipFree(dx4));
    TCK(hipFree(dy4));
    TCK(hipFree(dy1));
  }

  // ---- 1d. k_q8g32_gemv_brow_r (row-tiled HC down): bit-exact vs brow + timing ----
  {
    const uint64_t rows = 320, cols = 10240, gpr = cols / 32;
    std::vector<int8_t> hq(rows * cols);
    for (auto& v : hq) v = (int8_t)(rng() % 256);
    std::vector<__half> hs(rows * gpr);
    for (auto& v : hs) v = __float2half(0.5f + 0.5f * frand());
    std::vector<uint16_t> hx(4 * cols);
    for (auto& v : hx) v = bf16_bits(frand());
    int8_t* dq = dup(hq);
    __half* ds = dup(hs);
    uint16_t* dx = dup(hx);
    float* dy0 = dalloc(4 * rows);
    float* dy1 = dalloc(4 * rows);
    auto timeit = [&](auto fn) {
      hipEvent_t e0, e1;
      TCK(hipEventCreate(&e0));
      TCK(hipEventCreate(&e1));
      fn(); fn();
      TCK(hipEventRecord(e0));
      for (int i = 0; i < 30; i++) fn();
      TCK(hipEventRecord(e1));
      TCK(hipEventSynchronize(e1));
      float ms = 0;
      TCK(hipEventElapsedTime(&ms, e0, e1));
      TCK(hipEventDestroy(e0)); TCK(hipEventDestroy(e1));
      return ms * 1000.0 / 30;
    };
    // timing: does brow P=4 cost >> P=1? (weights constant, only x traffic grows)
    double t_b1 = timeit([&] { k_q8g32_gemv_brow<1, true, 128><<<(unsigned)rows, 128>>>(dq, ds, dx, dy0, rows, cols, cols); });
    double t_b4 = timeit([&] { k_q8g32_gemv_brow<4, true, 128><<<(unsigned)rows, 128>>>(dq, ds, dx, dy0, rows, cols, cols); });
    double t_r2 = timeit([&] { k_q8g32_gemv_brow_r<4, 128, 2><<<(unsigned)((rows + 1) / 2), 128>>>(dq, ds, dx, dy1, rows, cols, cols); });
    double t_r4 = timeit([&] { k_q8g32_gemv_brow_r<4, 128, 4><<<(unsigned)((rows + 3) / 4), 128>>>(dq, ds, dx, dy1, rows, cols, cols); });
    printf("%-28s brow_p1=%.1f brow_p4=%.1f tiled_r2=%.1f tiled_r4=%.1f us\n",
           "brow_r_timing", t_b1, t_b4, t_r2, t_r4);
    // bit-exactness, full + tail rows
    size_t mism = 0;
    for (uint64_t rr : {rows, rows - 1}) {
      TCK(hipMemset(dy1, 0xCD, 4 * rows * 4));
      k_q8g32_gemv_brow<4, true, 128><<<(unsigned)rr, 128>>>(dq, ds, dx, dy0, rr, cols, cols);
      k_q8g32_gemv_brow_r<4, 128, 4><<<(unsigned)((rr + 3) / 4), 128>>>(dq, ds, dx, dy1, rr, cols, cols);
      std::vector<float> y0 = dget(dy0, 4 * rows), y1 = dget(dy1, 4 * rows);
      for (uint64_t i = 0; i < 4 * rr; i++) mism += (y0[i] != y1[i]);
    }
    printf("%-28s mismatches=%zu %s\n", "brow_r_bitexact", mism,
           mism == 0 ? "PASS" : "FAIL");
    if (mism) fails++;
    TCK(hipFree(dq)); TCK(hipFree(ds)); TCK(hipFree(dx));
    TCK(hipFree(dy0)); TCK(hipFree(dy1));
  }

  // ---- 1e. P3-A d_vb device-base plumbing: scalar base vs device-mirrored ----
  // base must give bit-identical outputs (same value, read from *d_vb).
  {
    ensure_rope_tab(1e7, 64);
    const int P = 5;
    int* dvb;
    TCK(hipMalloc(&dvb, 4));
    auto bit_eq = [&](const char* name, const float* a, const float* b,
                      size_t n) {
      std::vector<float> va = dget(a, n), vb2 = dget(b, n);
      size_t mism = memcmp(va.data(), vb2.data(), n * 4) ? 1 : 0;
      printf("%-28s bit_mism=%zu %s\n", name, mism, mism ? "FAIL" : "PASS");
      if (mism) fails++;
    };
    for (int base : {0, 8191}) {  // 8191: base%4==3 exercises pool straddle
      TCK(hipMemcpy(dvb, &base, 4, hipMemcpyHostToDevice));
      char nm[64];
      // k_rope_cs
      float2 *cs0, *cs1;
      TCK(hipMalloc(&cs0, (size_t)P * 32 * sizeof(float2)));
      TCK(hipMalloc(&cs1, (size_t)P * 32 * sizeof(float2)));
      k_rope_cs<<<(P * 32 + 255) / 256, 256>>>(cs0, base, P);
      k_rope_cs<<<(P * 32 + 255) / 256, 256>>>(cs1, 0, P, dvb);
      snprintf(nm, sizeof nm, "vbase_rope_cs@%d", base);
      bit_eq(nm, (const float*)cs0, (const float*)cs1, (size_t)P * 64);
      // shared inputs for the indexer kernels
      std::vector<float> proj((size_t)P * 640), nrm(256), ring(4 * 128);
      for (auto& v : proj) v = frand();
      for (auto& v : nrm) v = frand() * 0.1f;
      for (auto& v : ring) v = frand();
      float *dp = dup(proj), *dn = dup(nrm), *dr = dup(ring);
      const int nb = (base + P) / 4, b0 = base / 4;
      // k_index_q
      float *q0 = dalloc((size_t)P * 512), *q1 = dalloc((size_t)P * 512);
      k_index_q<<<dim3(P, 4), 128>>>(dp, dn, q0, P, 1e-6f, 1e7, nullptr, base,
                                     cs0);
      k_index_q<<<dim3(P, 4), 128>>>(dp, dn, q1, P, 1e-6f, 1e7, nullptr, 0,
                                     cs0, nullptr, dvb);
      snprintf(nm, sizeof nm, "vbase_index_q@%d", base);
      bit_eq(nm, q0, q1, (size_t)P * 512);
      // k_index_pool
      float *k0 = dalloc((size_t)nb * 128), *k1 = dalloc((size_t)nb * 128);
      if (nb > b0) {
        k_index_pool<<<nb - b0, 128>>>(dp, dn, k0, 1e-6f, 1e7, base, dr, cs0);
        k_index_pool<<<nb - b0, 128>>>(dp, dn, k1, 1e-6f, 1e7, 0, dr, cs0,
                                       nullptr, dvb);
      }
      snprintf(nm, sizeof nm, "vbase_index_pool@%d", base);
      bit_eq(nm, k0, k1, (size_t)nb * 128);
      // k_index_ring
      float *r0 = dalloc(4 * 128), *r1 = dalloc(4 * 128);
      k_index_ring<<<min(P, 4), 128>>>(dp, r0, P, base);
      k_index_ring<<<min(P, 4), 128>>>(dp, r1, P, 0, dvb);
      snprintf(nm, sizeof nm, "vbase_index_ring@%d", base);
      bit_eq(nm, r0, r1, 4 * 128);
      // k_ple_conv_b / k_ple_ring_wr (n=256 miniature)
      const int n = 256;
      std::vector<float> Un((size_t)P * n), cw((size_t)n * 4), pring(9 * n);
      for (auto& v : Un) v = frand();
      for (auto& v : cw) v = frand() * 0.2f;
      for (auto& v : pring) v = frand();
      float *dU = dup(Un), *dcw = dup(cw);
      float *o0 = dalloc((size_t)P * n), *o1 = dalloc((size_t)P * n);
      float *pr0 = dup(pring), *pr1 = dup(pring);
      k_ple_conv_b<<<(P * n + 255) / 256, 256>>>(dU, dcw, o0, P, n, base, pr0);
      k_ple_conv_b<<<(P * n + 255) / 256, 256>>>(dU, dcw, o1, P, n, 0, pr0,
                                                 dvb);
      snprintf(nm, sizeof nm, "vbase_ple_conv_b@%d", base);
      bit_eq(nm, o0, o1, (size_t)P * n);
      k_ple_ring_wr<<<9, 256>>>(dU, pr0, P, n, base);
      k_ple_ring_wr<<<9, 256>>>(dU, pr1, P, n, 0, dvb);
      snprintf(nm, sizeof nm, "vbase_ple_ring_wr@%d", base);
      bit_eq(nm, pr0, pr1, (size_t)9 * n);
      TCK(hipFree(cs0)); TCK(hipFree(cs1));
      TCK(hipFree(dp)); TCK(hipFree(dn)); TCK(hipFree(dr));
      TCK(hipFree(q0)); TCK(hipFree(q1));
      TCK(hipFree(k0)); TCK(hipFree(k1));
      TCK(hipFree(r0)); TCK(hipFree(r1));
      TCK(hipFree(dU)); TCK(hipFree(dcw));
      TCK(hipFree(o0)); TCK(hipFree(o1));
      TCK(hipFree(pr0)); TCK(hipFree(pr1));
    }
    TCK(hipFree(dvb));
  }

  // ---- 1e2. P3-B graph-capture kernels: scalar base vs device *dvb, and the
  // paged device-side scatter kernels vs the host per-run loop. All bit-eq. ----
  {
    std::mt19937 rng(0x1e2);  // private stream: never disturb later tests
    auto frand = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); };
    const int P = 5, N = 512;
    int* dvb;
    TCK(hipMalloc(&dvb, 4));
    auto bit_eq = [&](const char* name, const void* a, const void* b,
                      size_t nbytes) {
      std::vector<char> va(nbytes), vb2(nbytes);
      TCK(hipMemcpy(va.data(), a, nbytes, hipMemcpyDeviceToHost));
      TCK(hipMemcpy(vb2.data(), b, nbytes, hipMemcpyDeviceToHost));
      size_t mism = memcmp(va.data(), vb2.data(), nbytes) ? 1 : 0;
      printf("%-28s bit_mism=%zu %s\n", name, mism, mism ? "FAIL" : "PASS");
      if (mism) fails++;
    };
    for (int base : {0, 8189, 8191}) {  // 8189: base%4==1, union edges only
      TCK(hipMemcpy(dvb, &base, 4, hipMemcpyHostToDevice));
      char nm[64];
      const int rows = base + P;
      const int npg = (rows + KV_PAGE - 1) / KV_PAGE, prows = npg * KV_PAGE;
      std::vector<int> ptab(npg);
      for (int p = 0; p < npg; p++) ptab[p] = npg - 1 - p;  // reversed pages
      auto prow = [&](int t) {
        return ptab[t / KV_PAGE] * KV_PAGE + t % KV_PAGE;
      };
      int* dptab = dup(ptab);
      // shared source batch: P fp32 rows
      std::vector<float> src((size_t)P * N);
      for (auto& v : src) v = frand();
      float* dsrc = dup(src);
      // poison-filled outputs so untouched slots must match too
      std::vector<uint16_t> poi16((size_t)prows * N);
      for (auto& v : poi16) v = (uint16_t)rng();
      std::vector<float> poi32((size_t)prows * N);
      for (auto& v : poi32) v = frand();

      // k_f32_to_bf16_v4: scalar (offset out) vs dvb
      {
        uint16_t *o0 = dup(poi16), *o1 = dup(poi16);
        k_f32_to_bf16_v4<<<(P * 128 + 255) / 256, 256>>>(dsrc, o0 + (size_t)base * N, N, P, N);
        k_f32_to_bf16_v4<<<(P * 128 + 255) / 256, 256>>>(dsrc, o1, N, P, N, dvb);
        snprintf(nm, sizeof nm, "vbase_f2bf_v4@%d", base);
        bit_eq(nm, o0, o1, poi16.size() * 2);
        TCK(hipFree(o0)); TCK(hipFree(o1));
      }
      // k_f32_to_bf16_v4_bt: scalar vs dvb (unpaged)
      {
        const int nb = (rows + 3) / 4;
        std::vector<uint16_t> poib((size_t)(prows / 4) * 2 * 256 * 4);
        for (auto& v : poib) v = (uint16_t)rng();
        uint16_t *o0 = dup(poib), *o1 = dup(poib);
        const int nb0 = (base + P + 3) / 4 - base / 4;
        k_f32_to_bf16_v4_bt<<<(nb0 * N + 255) / 256, 256>>>(dsrc, o0, N, P, N, base);
        const int nbw = (P + 6) / 4 + 1;  // worst-case baked grid
        k_f32_to_bf16_v4_bt<<<(nbw * N + 255) / 256, 256>>>(dsrc, o1, N, P, N, 0, dvb);
        snprintf(nm, sizeof nm, "vbase_f2bf_v4_bt@%d", base);
        bit_eq(nm, o0, o1, poib.size() * 2);
        (void)nb;
        TCK(hipFree(o0)); TCK(hipFree(o1));
      }
      // paged scatters: device-side ptab+dvb vs host per-run emulation
      {
        uint16_t *o0 = dup(poi16), *o1 = dup(poi16);
        // reference: per-row scalar converts at the physical row
        for (int lr = 0; lr < P; lr++)
          k_f32_to_bf16_v4<<<(128 + 255) / 256, 256>>>(
              dsrc + (size_t)lr * N, o0 + (size_t)prow(base + lr) * N, N, 1, N);
        k_f32_to_bf16_v4_pd<<<(P * 128 + 255) / 256, 256>>>(dsrc, o1, dptab, P, N, dvb);
        snprintf(nm, sizeof nm, "vbase_f2bf_v4_pd@%d", base);
        bit_eq(nm, o0, o1, poi16.size() * 2);
        TCK(hipFree(o0)); TCK(hipFree(o1));
      }
      {
        float *o0 = dup(poi32), *o1 = dup(poi32);
        for (int lr = 0; lr < P; lr++)
          TCK(hipMemcpy(o0 + (size_t)prow(base + lr) * N, dsrc + (size_t)lr * N,
                        N * 4, hipMemcpyDeviceToDevice));
        k_kv_rows_store_pd<<<P, 128>>>(o1, dsrc, dptab, P, dvb);
        snprintf(nm, sizeof nm, "vbase_kv_rows_pd@%d", base);
        bit_eq(nm, o0, o1, poi32.size() * 4);
        TCK(hipFree(o0)); TCK(hipFree(o1));
      }
      {
        // paged bt: reference = production per-run loop on the host
        std::vector<uint16_t> poib((size_t)(prows / 4) * 2 * 256 * 4);
        for (auto& v : poib) v = (uint16_t)rng();
        uint16_t *o0 = dup(poib), *o1 = dup(poib);
        for (int lr = 0; lr < P; ) {
          const int pr0 = prow(base + lr);
          int n = 1;
          while (lr + n < P && prow(base + lr + n) == pr0 + n) n++;
          const int nb0 = (pr0 + n + 3) / 4 - pr0 / 4;
          k_f32_to_bf16_v4_bt<<<(nb0 * N + 255) / 256, 256>>>(
              dsrc + (size_t)lr * N, o0, N, n, N, pr0);
          lr += n;
        }
        const int nbw = (P + 6) / 4 + 1;
        k_f32_to_bf16_v4_bt_pd<<<(nbw * N + 255) / 256, 256>>>(dsrc, o1, dptab, P, N, dvb);
        snprintf(nm, sizeof nm, "vbase_f2bf_bt_pd@%d", base);
        bit_eq(nm, o0, o1, poib.size() * 2);
        TCK(hipFree(o0)); TCK(hipFree(o1));
      }
      // k_qsa_kprep<true>: scalar (kc offset by base) vs dvb
      {
        std::vector<float> kb((size_t)P * N), knw(256);
        std::vector<float2> cs((size_t)P * 32);
        for (auto& v : kb) v = frand();
        for (auto& v : knw) v = frand() * 0.1f;
        for (auto& v : cs) v = make_float2(frand(), frand());
        float *dkb = dup(kb), *dkn = dup(knw);
        float2* dcs = dup(cs);
        float *ko0 = dalloc((size_t)P * N), *ko1 = dalloc((size_t)P * N);
        uint16_t *kc0 = dup(poi16), *kc1 = dup(poi16);
        k_qsa_kprep<true><<<dim3(2, P), 256>>>(dkb, dkn, ko0, kc0 + (size_t)base * N,
                                               1e-6f, base, dcs);
        k_qsa_kprep<true><<<dim3(2, P), 256>>>(dkb, dkn, ko1, kc1, 1e-6f, base,
                                               dcs, dvb);
        snprintf(nm, sizeof nm, "vbase_kprep_out@%d", base);
        bit_eq(nm, ko0, ko1, (size_t)P * N * 4);
        snprintf(nm, sizeof nm, "vbase_kprep_kc@%d", base);
        bit_eq(nm, kc0, kc1, poi16.size() * 2);
        TCK(hipFree(dkb)); TCK(hipFree(dkn)); TCK(hipFree(dcs));
        TCK(hipFree(ko0)); TCK(hipFree(ko1)); TCK(hipFree(kc0)); TCK(hipFree(kc1));
      }
      // sparse attention kernels at this base (rows are batch-local, sel row 0)
      if (base >= 2051) {
        const int HQ = 24, HKV = 2, DH = 256;
        std::vector<float> q((size_t)P * HQ * DH);
        for (auto& v : q) v = frand();
        std::vector<int> sel((size_t)P * 512);
        for (int r = 0; r < P; r++)
          for (int i = 0; i < 512; i++) sel[(size_t)r * 512 + i] = i * 3;
        float* dq = dup(q);
        int* dsel = dup(sel);
        // fp32 KV
        std::vector<float> kf((size_t)rows * HKV * DH), vf(kf.size());
        for (auto& v : kf) v = frand();
        for (auto& v : vf) v = frand();
        float *dkf = dup(kf), *dvf = dup(vf);
        float *fo0 = dalloc((size_t)P * HQ * DH), *fo1 = dalloc((size_t)P * HQ * DH);
        k_qsa_flash<true, false, float><<<dim3(P, HKV), 128>>>(
            dq, dkf, dvf, fo0, P, DH, HKV, HQ, dsel, base, nullptr, nullptr,
            nullptr, base);
        k_qsa_flash<true, false, float><<<dim3(P, HKV), 128>>>(
            dq, dkf, dvf, fo1, P, DH, HKV, HQ, dsel, 0, nullptr, nullptr,
            nullptr, 0, nullptr, dvb);
        snprintf(nm, sizeof nm, "vbase_qsa_flash@%d", base);
        bit_eq(nm, fo0, fo1, (size_t)P * HQ * DH * 4);
        k_qsa_flash<true, true, float><<<dim3(P, HKV), 128>>>(
            dq, dkf, dvf, fo0, P, DH, HKV, HQ, dsel, base, nullptr, nullptr,
            nullptr, base);
        k_qsa_flash<true, true, float><<<dim3(P, HKV), 128>>>(
            dq, dkf, dvf, fo1, P, DH, HKV, HQ, dsel, 0, nullptr, nullptr,
            nullptr, 0, nullptr, dvb);
        snprintf(nm, sizeof nm, "vbase_qsa_flash_sv@%d", base);
        bit_eq(nm, fo0, fo1, (size_t)P * HQ * DH * 4);
        TCK(hipFree(dkf)); TCK(hipFree(dvf));
        // bf16 KV + transposed V, paged (reversed table)
        std::vector<uint16_t> kb((size_t)prows * HKV * DH, 0), vb(kb.size(), 0);
        std::vector<uint16_t> vct((size_t)(prows / 4) * HKV * DH * 4, 0);
        for (int t = 0; t < rows; t++) {
          const int pr = prow(t);
          for (int j = 0; j < HKV * DH; j++) {
            const uint16_t b = bf16_bits(frand());
            kb[(size_t)pr * HKV * DH + j] = b;
            vb[(size_t)pr * HKV * DH + j] = bf16_bits(frand());
            vct[((size_t)(pr / 4) * HKV * DH + j) * 4 + t % 4] = vb[(size_t)pr * HKV * DH + j];
          }
        }
        uint16_t *dkb = dup(kb), *dvb2 = dup(vb), *dvct = dup(vct);
        float *wo0 = dalloc((size_t)P * HQ * DH), *wo1 = dalloc((size_t)P * HQ * DH);
        k_qsa_wmma<true, true><<<dim3(P, HKV), 256>>>(dq, dkb, dvb2, dvct, wo0, P,
                                                      DH, HKV, HQ, dsel, base, base,
                                                      dptab);
        k_qsa_wmma<true, true><<<dim3(P, HKV), 256>>>(dq, dkb, dvb2, dvct, wo1, P,
                                                      DH, HKV, HQ, dsel, 0, 0, dptab,
                                                      dvb, 0);
        snprintf(nm, sizeof nm, "vbase_qsa_wmma@%d", base);
        bit_eq(nm, wo0, wo1, (size_t)P * HQ * DH * 4);
        // flash_bf16 (unpaged bf16 path)
        std::vector<uint16_t> kc16((size_t)rows * HKV * DH), vc16(kc16.size());
        for (int t = 0; t < rows; t++)
          for (int j = 0; j < HKV * DH; j++) {
            kc16[(size_t)t * HKV * DH + j] = kb[(size_t)prow(t) * HKV * DH + j];
            vc16[(size_t)t * HKV * DH + j] = vb[(size_t)prow(t) * HKV * DH + j];
          }
        uint16_t *dkc16 = dup(kc16), *dvc16 = dup(vc16);
        k_qsa_flash_bf16<uint16_t><<<dim3(P, HKV), 128>>>(dq, dkc16, dvc16, wo0, P,
                                                          DH, HKV, HQ, dsel, base,
                                                          base);
        k_qsa_flash_bf16<uint16_t><<<dim3(P, HKV), 128>>>(dq, dkc16, dvc16, wo1, P,
                                                          DH, HKV, HQ, dsel, 0, 0,
                                                          dvb);
        snprintf(nm, sizeof nm, "vbase_qsa_flash16@%d", base);
        bit_eq(nm, wo0, wo1, (size_t)P * HQ * DH * 4);
        // union: merge outputs + wmma_u + both ragged edges
        {
          const int ga = (base + 3) & ~3, ge = (base + P) & ~3;
          const int ng = (ge - ga) >> 2;
          int ngw = 0;
          for (int r = 0; r < 4; r++)
            ngw = std::max(ngw, (((r + P) & ~3) - ((r + 3) & ~3)) >> 2);
          uint16_t *ub0 = (uint16_t*)dalloc((size_t)ngw * QSA_UCAP * 2 / 4 + 4),
                   *ub1 = (uint16_t*)dalloc((size_t)ngw * QSA_UCAP * 2 / 4 + 4);
          uint16_t *um0 = (uint16_t*)dalloc((size_t)ngw * QSA_UCAP * 2 / 4 + 4),
                   *um1 = (uint16_t*)dalloc((size_t)ngw * QSA_UCAP * 2 / 4 + 4);
          int *uc0 = (int*)dalloc(ngw + 1), *uc1 = (int*)dalloc(ngw + 1);
          if (ng > 0)
            k_qsa_union_merge<<<ng, 128>>>(dsel, ga, ub0, um0, uc0, QSA_UCAP, base);
          if (ngw > 0)
            k_qsa_union_merge<<<ngw, 128>>>(dsel, 0, ub1, um1, uc1, QSA_UCAP, 0, P, dvb);
          snprintf(nm, sizeof nm, "vbase_union_merge@%d", base);
          bit_eq(nm, ub0, ub1, (size_t)ngw * QSA_UCAP * 2);
          snprintf(nm, sizeof nm, "vbase_union_mask@%d", base);
          bit_eq(nm, um0, um1, (size_t)ngw * QSA_UCAP * 2);
          snprintf(nm, sizeof nm, "vbase_union_cnt@%d", base);
          bit_eq(nm, uc0, uc1, (size_t)ngw * 4);
          if (ng > 0)
            k_qsa_wmma_u<true><<<dim3(ng, HKV), 256>>>(dq, dkb, dvct, wo0, DH, HKV,
                                                       HQ, ub0, um0, uc0, QSA_UCAP,
                                                       ga, base, dptab);
          if (ngw > 0)
            k_qsa_wmma_u<true><<<dim3(ngw, HKV), 256>>>(dq, dkb, dvct, wo1, DH, HKV,
                                                        HQ, ub1, um1, uc1, QSA_UCAP,
                                                        0, 0, dptab, P, dvb);
          snprintf(nm, sizeof nm, "vbase_wmma_u@%d", base);
          bit_eq(nm, wo0, wo1, (size_t)P * HQ * DH * 4);
          // edges: scalar explicit vs dvb edge kinds (grids baked at 3)
          const int ln = std::min(ga, base + P) - base;   // left tokens
          const int rn = base + P - std::max(ge, ga);     // right tokens
          if (ln > 0)
            k_qsa_wmma<true, true><<<dim3(ln, HKV), 256>>>(dq, dkb, dvb2, dvct, wo0,
                                                           P, DH, HKV, HQ, dsel,
                                                           base, base, dptab);
          if (rn > 0)
            k_qsa_wmma<true, true><<<dim3(rn, HKV), 256>>>(
                dq, dkb, dvb2, dvct, wo0, P, DH, HKV, HQ,
                dsel + (size_t)(ge - base) * 512, ge, base, dptab);
          k_qsa_wmma<true, true><<<dim3(3, HKV), 256>>>(dq, dkb, dvb2, dvct, wo1, P,
                                                        DH, HKV, HQ, dsel, 0, 0,
                                                        dptab, dvb, 2);
          k_qsa_wmma<true, true><<<dim3(3, HKV), 256>>>(dq, dkb, dvb2, dvct, wo1, P,
                                                        DH, HKV, HQ, dsel, 0, 0,
                                                        dptab, dvb, 1);
          snprintf(nm, sizeof nm, "vbase_wmma_edges@%d", base);
          bit_eq(nm, wo0, wo1, (size_t)P * HQ * DH * 4);
          TCK(hipFree(ub0)); TCK(hipFree(ub1)); TCK(hipFree(um0)); TCK(hipFree(um1));
          TCK(hipFree(uc0)); TCK(hipFree(uc1));
        }
        TCK(hipFree(dq)); TCK(hipFree(dsel));
        TCK(hipFree(dkb)); TCK(hipFree(dvb2)); TCK(hipFree(dvct));
        TCK(hipFree(dkc16)); TCK(hipFree(dvc16));
        TCK(hipFree(wo0)); TCK(hipFree(wo1));
        TCK(hipFree(fo0)); TCK(hipFree(fo1));
      }
      TCK(hipFree(dptab));
      TCK(hipFree(dsrc));
    }
    TCK(hipFree(dvb));
  }

  // ---- 1f. k_moe_sort_slots + k_q4cp_gemv_gg slot_order: 纯调度排列逐 bit ----
  {
    const int P = 5, k = 10, E = 32;  // E 小 → 强制大量重复专家
    // 独立随机源：不消耗共享 rng，否则其后所有测试的输入整体错位
    // （gr_combine_b 等按相对误差判定的测试会撞上抵消点而误报）。
    std::mt19937 rng(0x1f);
    auto frand = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); };
    const int n = P * k;
    std::vector<int> ids(P * 16, 0);
    for (int t = 0; t < P; t++)
      for (int s = 0; s < k; s++) ids[t * 16 + s] = rng() % E;
    int* dids = dup(ids);
    int* dord;
    TCK(hipMalloc(&dord, 256 * 4));
    k_moe_sort_slots<<<1, 128>>>(dids, P, k, 16, dord);
    std::vector<int> ord(n);
    TCK(hipMemcpy(ord.data(), dord, n * 4, hipMemcpyDeviceToHost));
    // 合法性：[0,n) 的排列 + 按 (expert, slot) 升序 + 确实打乱了顺序
    std::vector<char> seen(n, 0);
    int moved = 0;
    bool ok = true;
    for (int i = 0; i < n; i++) {
      if (ord[i] < 0 || ord[i] >= n || seen[ord[i]]) ok = false;
      else seen[ord[i]] = 1;
      if (ord[i] != i) moved++;
      if (i) {
        int e0 = ids[(ord[i - 1] / k) * 16 + ord[i - 1] % k];
        int e1 = ids[(ord[i] / k) * 16 + ord[i] % k];
        if (e0 > e1 || (e0 == e1 && ord[i - 1] > ord[i])) ok = false;
      }
    }
    printf("%-28s perm_ok=%d moved=%d %s\n", "moe_sort_slots", (int)ok, moved,
           ok && moved > 0 ? "PASS" : "FAIL");
    if (!ok || moved == 0) fails++;
    // gg 带 slot_order vs 不带：逐 bit
    const uint64_t rows_per = 8, cols = 1024, gpr = cols / 32;
    std::vector<uint8_t> codes((size_t)E * rows_per * cols / 2);
    for (auto& v : codes) v = (uint8_t)rng();
    std::vector<uint8_t> scales((size_t)E * rows_per * gpr * 2);
    __half* sh = (__half*)scales.data();
    for (size_t i = 0; i < (size_t)E * rows_per * gpr; i++)
      sh[i] = __float2half(0.01f + 0.005f * frand());
    std::vector<float> cb(16), x((size_t)P * cols);
    for (auto& v : cb) v = frand();
    for (auto& v : x) v = frand();
    uint8_t* dc = dup(codes);
    uint8_t* dsc = dup(scales);
    float *dcb = dup(cb), *dx = dup(x);
    float *y0 = dalloc((size_t)n * rows_per), *y1 = dalloc((size_t)n * rows_per);
    uint64_t pairs = (uint64_t)n * ((rows_per + 1) / 2);
    k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512>>>(
        dc, dsc, dcb, dx, y0, dids, rows_per, cols, gpr * 2, n, cols, 16, k);
    k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512>>>(
        dc, dsc, dcb, dx, y1, dids, rows_per, cols, gpr * 2, n, cols, 16, k,
        dord);
    std::vector<float> va = dget(y0, (size_t)n * rows_per),
                       vb2 = dget(y1, (size_t)n * rows_per);
    size_t mism = memcmp(va.data(), vb2.data(), va.size() * 4) ? 1 : 0;
    printf("%-28s bit_mism=%zu %s\n", "gg_slot_order", mism,
           mism ? "FAIL" : "PASS");
    if (mism) fails++;
    TCK(hipFree(dids)); TCK(hipFree(dord));
    TCK(hipFree(dc)); TCK(hipFree(dsc)); TCK(hipFree(dcb)); TCK(hipFree(dx));
    TCK(hipFree(y0)); TCK(hipFree(y1));
  }

  // ---- 1g. 行不变性审计（D0-M4，HANDOFF-CONCURRENCY §6.3）----
  // D2 合批 verify 把多序列拼成 R 行：按行合批的 kernel 必须对"同一批行在
  // P=R 与 P=P_s 下"逐 bit 一致（y 布局都是 [P, N]，前 Ps*N 个 float 即前
  // Ps 行）。独立 RNG（0x4d），不消耗共享 rng。
  {
    std::mt19937 rng(0x4d);
    auto frand = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); };
    auto bf16b = [&](float f) { return bf16_bits(f); };

    // 通用 family 审计：runP(P, dy) 启动该 kernel 的 P 模板实例；
    // 先跑 R=8 作基准，再与 Ps ∈ {1,3,5} 比前缀 Ps*N。
    auto audit = [&](const char* name, int N, auto runP, float* dy) {
      std::vector<float> yR((size_t)8 * N);
      runP(8, dy);
      TCK(hipMemcpy(yR.data(), dy, (size_t)8 * N * 4, hipMemcpyDeviceToHost));
      size_t m = 0;
      for (int Ps : {1, 3, 5}) {
        runP(Ps, dy);
        std::vector<float> yS = dget(dy, (size_t)Ps * N);
        for (size_t i = 0; i < (size_t)Ps * N; i++) m += (yR[i] != yS[i]);
      }
      printf("ri_%-23s mism=%zu %s\n", name, m, m ? "FAIL" : "PASS");
      if (m) fails++;
    };

    // ri_q4cp_mr：q4cp 稠密 GEMV（P<=8 档），float x 与 bf16 x 两条路
    for (int xbf = 0; xbf <= 1; xbf++) {
      const uint64_t rows = 128, cols = 1024, ss = (cols / 32) * 2;
      std::vector<uint8_t> w(64 + rows * cols / 2 + rows * ss);
      std::vector<float> cb(16);
      for (auto& v : cb) v = frand() * 2.f;
      memcpy(w.data(), cb.data(), 64);
      for (uint64_t i = 0; i < rows * cols / 2; i++) w[64 + i] = (uint8_t)rng();
      __half* hs = (__half*)(w.data() + 64 + rows * cols / 2);
      for (uint64_t i = 0; i < rows * cols / 32; i++)
        hs[i] = __float2half(0.01f + 0.005f * frand());
      std::vector<float> xf(8 * cols);
      for (auto& v : xf) v = frand();
      std::vector<uint16_t> xb(8 * cols);
      for (size_t i = 0; i < xb.size(); i++) xb[i] = bf16b(xf[i]);
      uint8_t* dw = dup(w);
      float* dxf = dup(xf);
      uint16_t* dxb = dup(xb);
      float* dy = dalloc((size_t)8 * rows);
      const unsigned blocks = (unsigned)((rows + 31) / 32);
      auto runP = [&](int P, float* y) {
        auto run = [&](auto Pc) {
          constexpr int PP = decltype(Pc)::value;
          if (xbf)
            k_q4cp_gemv_mr<PP, true><<<blocks, 512>>>(
                dw + 64, dw + 64 + rows * cols / 2, (const float*)dw, dxb, y,
                rows, cols, ss, cols);
          else
            k_q4cp_gemv_mr<PP, false><<<blocks, 512>>>(
                dw + 64, dw + 64 + rows * cols / 2, (const float*)dw, dxf, y,
                rows, cols, ss, cols);
        };
        switch (P) {
          case 1: run(std::integral_constant<int, 1>{}); break;
          case 2: run(std::integral_constant<int, 2>{}); break;
          case 3: run(std::integral_constant<int, 3>{}); break;
          case 4: run(std::integral_constant<int, 4>{}); break;
          case 5: run(std::integral_constant<int, 5>{}); break;
          case 6: run(std::integral_constant<int, 6>{}); break;
          case 7: run(std::integral_constant<int, 7>{}); break;
          case 8: run(std::integral_constant<int, 8>{}); break;
        }
      };
      audit(xbf ? "q4cp_mr_bx" : "q4cp_mr", (int)rows, runP, dy);
      TCK(hipFree(dw)); TCK(hipFree(dxf)); TCK(hipFree(dxb)); TCK(hipFree(dy));
    }

    // ri_bf16_mr / ri_bf16_row_mp（_bx）：bf16 稠密 GEMV 两条路径
    for (int kind = 0; kind < 3; kind++) {  // 0=mr 1=row_mp 2=row_mp bf16x
      const uint64_t rows = kind ? 48 : 64, cols = kind ? 2560 : 2048;
      std::vector<uint16_t> w(rows * cols);
      for (auto& v : w) v = bf16b(frand());
      std::vector<float> xf(8 * cols);
      for (auto& v : xf) v = frand();
      std::vector<uint16_t> xb(8 * cols);
      for (size_t i = 0; i < xb.size(); i++) xb[i] = bf16b(xf[i]);
      uint16_t* dw = dup(w);
      float* dxf = dup(xf);
      uint16_t* dxb = dup(xb);
      float* dy = dalloc((size_t)8 * rows);
      const unsigned blocks = (unsigned)((rows + 31) / 32);
      auto runP = [&](int P, float* y) {
        auto run = [&](auto Pc) {
          constexpr int PP = decltype(Pc)::value;
          if (kind == 0)
            k_bf16_gemv_mr<PP><<<blocks, 512>>>(dw, dxf, y, rows, cols, cols);
          else if (kind == 1)
            k_bf16_gemv_row_mp<PP, false><<<(unsigned)rows, 256>>>(dw, dxf, y,
                                                                   rows, cols,
                                                                   cols);
          else
            k_bf16_gemv_row_mp<PP, true><<<(unsigned)rows, 256>>>(dw, dxb, y,
                                                                  rows, cols,
                                                                  cols);
        };
        switch (P) {
          case 1: run(std::integral_constant<int, 1>{}); break;
          case 2: run(std::integral_constant<int, 2>{}); break;
          case 3: run(std::integral_constant<int, 3>{}); break;
          case 4: run(std::integral_constant<int, 4>{}); break;
          case 5: run(std::integral_constant<int, 5>{}); break;
          case 6: run(std::integral_constant<int, 6>{}); break;
          case 7: run(std::integral_constant<int, 7>{}); break;
          case 8: run(std::integral_constant<int, 8>{}); break;
        }
      };
      audit(kind == 0 ? "bf16_mr" : (kind == 1 ? "bf16_row_mp" : "bf16_row_mp_bx"),
            (int)rows, runP, dy);
      TCK(hipFree(dw)); TCK(hipFree(dxf)); TCK(hipFree(dxb)); TCK(hipFree(dy));
    }

    // ri_q8g64_mr：q8g64 稠密（float x）
    {
      const uint64_t rows = 64, cols = 1024;
      const uint64_t stride = cols + cols / 64 * 4;
      std::vector<uint8_t> w(rows * stride);
      for (uint64_t r = 0; r < rows; r++) {
        uint8_t* rp = w.data() + r * stride;
        for (uint64_t i = 0; i < cols; i++) rp[i] = (uint8_t)rng();
        for (uint64_t g = 0; g < cols / 64; g++) {
          __half* sm = (__half*)(rp + cols + g * 4);
          sm[0] = __float2half(0.01f + 0.005f * frand());
          sm[1] = __float2half(0.5f * frand());
        }
      }
      std::vector<float> xf(8 * cols);
      for (auto& v : xf) v = frand();
      uint8_t* dw = dup(w);
      float* dxf = dup(xf);
      float* dy = dalloc((size_t)8 * rows);
      const unsigned blocks = (unsigned)((rows + 31) / 32);
      auto runP = [&](int P, float* y) {
        auto run = [&](auto Pc) {
          constexpr int PP = decltype(Pc)::value;
          k_q8g64_gemv_mr<PP, false><<<blocks, 512>>>(dw, dxf, y, rows, cols,
                                                      cols);
        };
        switch (P) {
          case 1: run(std::integral_constant<int, 1>{}); break;
          case 2: run(std::integral_constant<int, 2>{}); break;
          case 3: run(std::integral_constant<int, 3>{}); break;
          case 4: run(std::integral_constant<int, 4>{}); break;
          case 5: run(std::integral_constant<int, 5>{}); break;
          case 6: run(std::integral_constant<int, 6>{}); break;
          case 7: run(std::integral_constant<int, 7>{}); break;
          case 8: run(std::integral_constant<int, 8>{}); break;
        }
      };
      audit("q8g64_mr", (int)rows, runP, dy);
      TCK(hipFree(dw)); TCK(hipFree(dxf)); TCK(hipFree(dy));
    }

    // ri_q8g32_multi（HQ 主路径 dtype8，lpr=16 档）/ ri_q8g32_brow_r（bf16 x、
    // 长行少行档）
    for (int kind = 0; kind < 2; kind++) {
      const uint64_t rows = kind ? 320 : 64, cols = kind ? 6144 : 2560;
      std::vector<int8_t> q(rows * cols);
      for (auto& v : q) v = (int8_t)rng();
      std::vector<__half> s(rows * (cols / 32));
      for (auto& v : s) v = __float2half(0.01f + 0.005f * frand());
      std::vector<float> xf(8 * cols);
      for (auto& v : xf) v = frand();
      std::vector<uint16_t> xb(8 * cols);
      for (size_t i = 0; i < xb.size(); i++) xb[i] = bf16b(xf[i]);
      int8_t* dq = dup(q);
      __half* ds = dup(s);
      float* dxf = dup(xf);
      uint16_t* dxb = dup(xb);
      float* dy = dalloc((size_t)8 * rows);
      auto runP = [&](int P, float* y) {
        auto run = [&](auto Pc) {
          constexpr int PP = decltype(Pc)::value;
          if (kind == 0)
            q8g32_gemv_multi<PP, false, 16, 256>(dq, ds, dxf, y, rows, cols,
                                                 cols, (hipStream_t)0);
          else
            k_q8g32_gemv_brow_r<PP, 128, 4>
                <<<(unsigned)((rows + 3) / 4), 128>>>(dq, ds, dxb, y, rows,
                                                      cols, cols);
        };
        switch (P) {
          case 1: run(std::integral_constant<int, 1>{}); break;
          case 2: run(std::integral_constant<int, 2>{}); break;
          case 3: run(std::integral_constant<int, 3>{}); break;
          case 4: run(std::integral_constant<int, 4>{}); break;
          case 5: run(std::integral_constant<int, 5>{}); break;
          case 6: run(std::integral_constant<int, 6>{}); break;
          case 7: run(std::integral_constant<int, 7>{}); break;
          case 8: run(std::integral_constant<int, 8>{}); break;
        }
      };
      audit(kind == 0 ? "q8g32_multi" : "q8g32_brow_r", (int)rows, runP, dy);
      TCK(hipFree(dq)); TCK(hipFree(ds)); TCK(hipFree(dxf)); TCK(hipFree(dxb));
      TCK(hipFree(dy));
    }

    // ri_f32_gemv_mr：iproj GEMV（GDEC_IPROJ_GEMV 默认开的路径）
    {
      const uint64_t rows = 640, cols = 2560;
      std::vector<float> w(rows * cols);
      for (auto& v : w) v = frand();
      std::vector<float> xf(8 * cols);
      for (auto& v : xf) v = frand();
      float* dw = dup(w);
      float* dxf = dup(xf);
      float* dy = dalloc((size_t)8 * rows);
      const unsigned blocks = (unsigned)((rows + 15) / 16);
      auto runP = [&](int P, float* y) {
        auto run = [&](auto Pc) {
          constexpr int PP = decltype(Pc)::value;
          k_f32_gemv_mr<PP><<<blocks, 512>>>(dw, dxf, y, rows, cols);
        };
        switch (P) {
          case 1: run(std::integral_constant<int, 1>{}); break;
          case 2: run(std::integral_constant<int, 2>{}); break;
          case 3: run(std::integral_constant<int, 3>{}); break;
          case 4: run(std::integral_constant<int, 4>{}); break;
          case 5: run(std::integral_constant<int, 5>{}); break;
          case 6: run(std::integral_constant<int, 6>{}); break;
          case 7: run(std::integral_constant<int, 7>{}); break;
          case 8: run(std::integral_constant<int, 8>{}); break;
        }
      };
      audit("f32_gemv_mr", (int)rows, runP, dy);
      TCK(hipFree(dw)); TCK(hipFree(dxf)); TCK(hipFree(dy));
    }

    // ri_moe_gg / ri_moe_gd_topk：MoE 槽数（nslots=P*k）不变性
    {
      const int k = 10, E = 32;
      const uint64_t rows_per = 8, cols = 1024, gpr = cols / 32;
      std::vector<uint8_t> codes((size_t)E * rows_per * cols / 2);
      for (auto& v : codes) v = (uint8_t)rng();
      std::vector<uint8_t> scales((size_t)E * rows_per * gpr * 2);
      __half* sh = (__half*)scales.data();
      for (size_t i = 0; i < (size_t)E * rows_per * gpr; i++)
        sh[i] = __float2half(0.01f + 0.005f * frand());
      std::vector<float> cb(16);
      for (auto& v : cb) v = frand();
      const int Pmax = 8;
      std::vector<int> ids(Pmax * 16);
      for (int t = 0; t < Pmax; t++)
        for (int s = 0; s < k; s++) ids[t * 16 + s] = (int)(rng() % E);
      std::vector<float> x((size_t)Pmax * k * cols);
      for (auto& v : x) v = frand();
      std::vector<float> ws(Pmax * 16);
      for (auto& v : ws) v = frand() + 1.1f;
      uint8_t* dc = dup(codes);
      uint8_t* dsc = dup(scales);
      float* dcb = dup(cb);
      float* dx = dup(x);
      int* dids = dup(ids);
      float* dws = dup(ws);
      float* dy = dalloc((size_t)Pmax * k * rows_per);
      size_t m = 0;
      {  // gg：槽 = gw/pairs_per_slot，槽内行对与总数无关
        uint64_t pairs = (uint64_t)Pmax * k * ((rows_per + 1) / 2);
        k_q4cp_gemv_gg<<<(unsigned)((pairs + 15) / 16), 512>>>(
            dc, dsc, dcb, dx, dy, dids, rows_per, cols, gpr * 2, Pmax * k,
            cols, 16, k);
        std::vector<float> yR = dget(dy, (size_t)Pmax * k * rows_per);
        for (int Ps : {1, 3, 5}) {
          uint64_t ps = (uint64_t)Ps * k * ((rows_per + 1) / 2);
          k_q4cp_gemv_gg<<<(unsigned)((ps + 15) / 16), 512>>>(
              dc, dsc, dcb, dx, dy, dids, rows_per, cols, gpr * 2, Ps * k,
              cols, 16, k);
          std::vector<float> yS = dget(dy, (size_t)Ps * k * rows_per);
          for (size_t i = 0; i < yS.size(); i++) m += (yR[i] != yS[i]);
        }
        printf("ri_%-23s mism=%zu %s\n", "moe_gg", m, m ? "FAIL" : "PASS");
        if (m) fails++;
      }
      m = 0;
      {  // gd_topk_h16<10>：每 token 独立，y [P, rows_per]
        uint64_t rp = (rows_per + 1) / 2;
        k_q4cp_gemv_gd_topk_h16<10><<<(unsigned)((Pmax * rp + 3) / 4), 128>>>(
            dc, dsc, dcb, dx, dy, dids, dws, rows_per, cols, gpr * 2, cols,
            Pmax, 16, rows_per);
        std::vector<float> yR = dget(dy, (size_t)Pmax * rows_per);
        for (int Ps : {1, 3, 5}) {
          k_q4cp_gemv_gd_topk_h16<10><<<(unsigned)((Ps * rp + 3) / 4), 128>>>(
              dc, dsc, dcb, dx, dy, dids, dws, rows_per, cols, gpr * 2, cols,
              Ps, 16, rows_per);
          std::vector<float> yS = dget(dy, (size_t)Ps * rows_per);
          for (size_t i = 0; i < yS.size(); i++) m += (yR[i] != yS[i]);
        }
        printf("ri_%-23s mism=%zu %s\n", "moe_gd_topk", m, m ? "FAIL" : "PASS");
        if (m) fails++;
      }
      TCK(hipFree(dc)); TCK(hipFree(dsc)); TCK(hipFree(dcb)); TCK(hipFree(dx));
      TCK(hipFree(dids)); TCK(hipFree(dws)); TCK(hipFree(dy));
    }

    // ri_argmax2：两段 argmax 的行不变性
    {
      const int n = 24832, R = 8;
      std::vector<float> x((size_t)R * n);
      for (auto& v : x) v = frand();
      float* dx = dup(x);
      unsigned long long* dcand;
      TCK(hipMalloc(&dcand, (size_t)R * 40 * 8));
      int* dout;
      TCK(hipMalloc(&dout, R * 4));
      k_argmax_p1<<<dim3(40, R), 256>>>(dx, n, dcand);
      k_argmax_p2<<<dim3(1, R), 32>>>(dcand, 40, dout);
      std::vector<int> oR(R);
      TCK(hipMemcpy(oR.data(), dout, R * 4, hipMemcpyDeviceToHost));
      size_t m = 0;
      for (int Ps : {1, 3, 5}) {
        k_argmax_p1<<<dim3(40, Ps), 256>>>(dx, n, dcand);
        k_argmax_p2<<<dim3(1, Ps), 32>>>(dcand, 40, dout);
        std::vector<int> oS(Ps);
        TCK(hipMemcpy(oS.data(), dout, Ps * 4, hipMemcpyDeviceToHost));
        for (int i = 0; i < Ps; i++) m += (oR[i] != oS[i]);
      }
      printf("ri_%-23s mism=%zu %s\n", "argmax2", m, m ? "FAIL" : "PASS");
      if (m) fails++;
      TCK(hipFree(dx)); TCK(hipFree(dcand)); TCK(hipFree(dout));
    }

    // ri_moe_reduce / ri_moe_reduce_fast：累加顺序固定 0..k-1，P 行不变
    for (int fast = 0; fast <= 1; fast++) {
      const int k = 10, D = 64, R = 8;
      std::vector<float> pairs((size_t)R * k * D);
      for (auto& v : pairs) v = frand();
      std::vector<int> pids(R * k);
      for (int t = 0; t < R; t++)
        for (int s = 0; s < k; s++) pids[t * k + s] = t * k + (s * 7 + 3) % k;
      float* dp = dup(pairs);
      int* dpi = dup(pids);
      float* dout = dalloc((size_t)R * D);
      std::vector<float> oR((size_t)R * D);
      if (fast)
        k_moe_reduce_fast<10><<<dim3(1, R), 256>>>(dp, dpi, dout, R, D);
      else
        k_moe_reduce<<<(R * D + 255) / 256, 256>>>(dp, dpi, dout, R, k, D);
      TCK(hipMemcpy(oR.data(), dout, (size_t)R * D * 4, hipMemcpyDeviceToHost));
      size_t m = 0;
      for (int Ps : {1, 3, 5}) {
        if (fast)
          k_moe_reduce_fast<10><<<dim3(1, Ps), 256>>>(dp, dpi, dout, Ps, D);
        else
          k_moe_reduce<<<(Ps * D + 255) / 256, 256>>>(dp, dpi, dout, Ps, k, D);
        std::vector<float> oS = dget(dout, (size_t)Ps * D);
        for (size_t i = 0; i < oS.size(); i++) m += (oR[i] != oS[i]);
      }
      printf("ri_%-23s mism=%zu %s\n", fast ? "moe_reduce_fast" : "moe_reduce",
             m, m ? "FAIL" : "PASS");
      if (m) fails++;
      TCK(hipFree(dp)); TCK(hipFree(dpi)); TCK(hipFree(dout));
    }
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

  // ---- 6b. B2-1 fused HC write-back (bitwise A/B, no tolerance) ----
  {
    const int P = 5, D = 2560, K = 10, S = P * K, B = 4;
    std::vector<float> pairs(S * D), pw(S), ey(P * D), sg(P), w4(P * B), svh(D),
        y(P * D), hw(B * D), pin(P * 16);
    std::vector<int> pid(P * K);
    std::vector<uint16_t> R0(P * B * D), winj(B * B * D);
    for (auto& v : pairs) v = frand();
    for (auto& v : pw) v = frand();
    for (auto& v : ey) v = frand();
    for (auto& v : sg) v = frand();
    for (auto& v : w4) v = frand();
    for (auto& v : svh) v = frand();
    for (auto& v : y) v = frand();
    for (auto& v : hw) v = frand();
    for (auto& v : pin) v = frand();
    for (int i = 0; i < P * K; i++) pid[i] = (i * 7 + 3) % S;
    for (auto& u : R0) u = bf16_bits(frand());
    for (auto& u : winj) u = bf16_bits(frand());
    float* dpairs = dup(pairs), *dpw = dup(pw), *dey = dup(ey), *dsg = dup(sg),
          *dw4 = dup(w4), *dsvh = dup(svh), *dy = dup(y), *dhw = dup(hw),
          *dpin = dup(pin);
    int* dpid = dup(pid);
    uint16_t* dwinj = dup(winj);
    float* dout = dalloc(P * D);
    const float eps = 1e-6f;
    auto getu16 = [&](const uint16_t* p, size_t n) {
      std::vector<uint16_t> v(n);
      TCK(hipMemcpy(v.data(), p, n * 2, hipMemcpyDeviceToHost));
      return v;
    };
    auto bitdiff = [&](const char* name, const std::vector<uint16_t>& a,
                       const std::vector<uint16_t>& b) {
      size_t mm = 0;
      for (size_t i = 0; i < a.size(); i++) mm += a[i] != b[i];
      check(name, (double)mm, 0);
    };
    auto bitdifff = [&](const char* name, const std::vector<float>& a,
                        const std::vector<float>& b) {
      size_t mm = 0;
      for (size_t i = 0; i < a.size(); i++) mm += memcmp(&a[i], &b[i], 4) != 0;
      check(name, (double)mm, 0);
    };
    auto mkR = [&] { return dup(R0); };

    // w4b: k_moe_reduce_pw_sg -> out + k_gr_write_b_hc_bf16 == *_hc
    uint16_t* dR1 = mkR();
    k_moe_reduce_pw_sg<10><<<dim3(D / 256, P), 256>>>(dpairs, dpid, dpw, dey, dsg,
                                                      dout, P, D);
    k_gr_write_b_hc_bf16<<<(P * D + 255) / 256, 256>>>(dw4, dR1, dout, D, B, P);
    uint16_t* dR2 = mkR();
    k_moe_reduce_pw_sg_hc<10><<<dim3(D / 256, P), 256>>>(dpairs, dpid, dpw, dey, dsg,
                                                         dw4, dR2, P, D);
    bitdiff("hcw_reduce_pw_sg", getu16(dR1, R0.size()), getu16(dR2, R0.size()));
    TCK(hipFree(dR1));
    TCK(hipFree(dR2));

    // v2: k_v2_reduce_rot -> out + k_gr_write_b_hc_bf16 == *_hc
    const unsigned waves = (unsigned)P * (D / 128);
    uint16_t* dR3 = mkR();
    k_v2_reduce_rot<10><<<(waves + 7) / 8, 256>>>(dpairs, dpid, dpw, dsvh, dey, dsg,
                                                  dout, P, D);
    k_gr_write_b_hc_bf16<<<(P * D + 255) / 256, 256>>>(dw4, dR3, dout, D, B, P);
    uint16_t* dR4 = mkR();
    k_v2_reduce_rot_hc<10><<<(waves + 7) / 8, 256>>>(dpairs, dpid, dpw, dsvh, dey,
                                                   dsg, dw4, dR4, P, D);
    bitdiff("hcw_v2_reduce_rot", getu16(dR3, R0.size()), getu16(dR4, R0.size()));
    TCK(hipFree(dR3));
    TCK(hipFree(dR4));

    // Scatter kernel: y path unchanged vs the pre-edit copy, and the split
    // flow (k_inj_psum -> pre-write -> scatter y=null) matches the old fused
    // scatter bit for bit (R, Rhat, pout).
    uint16_t *dRh1, *dRh2, *dRh3;
    float *dpo1, *dpo2, *dpo3;
    TCK(hipMalloc(&dRh1, R0.size() * 2));
    TCK(hipMalloc(&dRh2, R0.size() * 2));
    TCK(hipMalloc(&dRh3, R0.size() * 2));
    TCK(hipMalloc(&dpo1, pin.size() * 4));
    TCK(hipMalloc(&dpo2, pin.size() * 4));
    TCK(hipMalloc(&dpo3, pin.size() * 4));
    const unsigned sgrid = 4 * ((P + 3) / 4);
    uint16_t* dR5 = mkR();
    k_gr_scatter_norm_inj_bf16_old<4><<<sgrid, 256>>>(nullptr, dpin, dR5, dy, dhw,
                                                      dRh1, dwinj, dpo1, D, P, eps);
    uint16_t* dR6 = mkR();
    k_gr_scatter_norm_inj_bf16<4><<<sgrid, 256>>>(nullptr, dpin, dR6, dy, dhw, dRh2,
                                                  dwinj, dpo2, D, P, eps);
    bitdiff("hcw_scat_ypath_R", getu16(dR5, R0.size()), getu16(dR6, R0.size()));
    bitdiff("hcw_scat_ypath_Rhat", getu16(dRh1, R0.size()), getu16(dRh2, R0.size()));
    bitdifff("hcw_scat_ypath_pout", dget(dpo1, pin.size()), dget(dpo2, pin.size()));
    TCK(hipFree(dR5));
    TCK(hipFree(dR6));
    // split flow: dR7 pre-written with the psum inject gates, then norm-only
    uint16_t* dR7 = mkR();
    k_inj_psum<<<(P * 4 + 255) / 256, 256>>>(dpin, dw4, P);
    k_gr_write_b_hc_bf16<<<(P * D + 255) / 256, 256>>>(dw4, dR7, dy, D, B, P);
    k_gr_scatter_norm_inj_bf16<4, false><<<sgrid, 256>>>(nullptr, nullptr, dR7, nullptr,
                                                         dhw, dRh3, dwinj, dpo3, D, P, eps);
    uint16_t* dR8 = mkR();
    k_gr_scatter_norm_inj_bf16_old<4><<<sgrid, 256>>>(nullptr, dpin, dR8, dy, dhw,
                                                      dRh1, dwinj, dpo1, D, P, eps);
    bitdiff("hcw_scat_split_R", getu16(dR8, R0.size()), getu16(dR7, R0.size()));
    bitdiff("hcw_scat_split_Rhat", getu16(dRh1, R0.size()), getu16(dRh3, R0.size()));
    bitdifff("hcw_scat_split_pout", dget(dpo1, pin.size()), dget(dpo3, pin.size()));
    TCK(hipFree(dR7));
    TCK(hipFree(dR8));
    TCK(hipFree(dRh1));
    TCK(hipFree(dRh2));
    TCK(hipFree(dRh3));
    TCK(hipFree(dpo1));
    TCK(hipFree(dpo2));
    TCK(hipFree(dpo3));
    TCK(hipFree(dpairs));
    TCK(hipFree(dpw));
    TCK(hipFree(dey));
    TCK(hipFree(dsg));
    TCK(hipFree(dw4));
    TCK(hipFree(dsvh));
    TCK(hipFree(dy));
    TCK(hipFree(dhw));
    TCK(hipFree(dpin));
    TCK(hipFree(dpid));
    TCK(hipFree(dwinj));
    TCK(hipFree(dout));
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

  // ---- 9d2. BTV transposed V cache: decode store + snapshot rebuild ----
  // k_store_kv_bf16(vct) must write slot pos%4 of block pos/4 with the same
  // bits as the row-major cache; k_bf16_rows_to_bt must reproduce the
  // k_f32_to_bf16_v4_bt layout from the row-major cache for [r0, r1), zero
  // the slots >= r1 of the trailing partial block and leave the rest untouched.
  {
    const int NT = 13, NB = (NT + 3) / 4;
    std::vector<float> v((size_t)NT * 512), kb(512);
    for (auto& x : v) x = frand();
    for (auto& x : kb) x = frand();
    std::vector<uint16_t> vrow((size_t)NT * 512);
    for (size_t i = 0; i < vrow.size(); i++) vrow[i] = bf16_bits(v[i]);
    float* dv = dup(v);
    float* dkb = dup(kb);
    uint16_t *dkc, *dvc, *dref, *dst, *dreb;
    const size_t vt_n = (size_t)NB * 512 * 4;
    TCK(hipMalloc(&dkc, (size_t)NT * 512 * 2));
    TCK(hipMalloc(&dvc, (size_t)NT * 512 * 2));
    TCK(hipMalloc(&dref, vt_n * 2));
    TCK(hipMalloc(&dst, vt_n * 2));
    TCK(hipMalloc(&dreb, vt_n * 2));
    TCK(hipMemset(dref, 0xEE, vt_n * 2));
    TCK(hipMemset(dst, 0xEE, vt_n * 2));
    TCK(hipMemset(dreb, 0xEE, vt_n * 2));
    // reference: the production prefill writer over all NT rows
    k_f32_to_bf16_v4_bt<<<(unsigned)((NB * 512 + 255) / 256), 256>>>(dv, dref, 512,
                                                                     NT, 512, 0);
    TCK(hipGetLastError());
    // decode path: one k_store_kv_bf16 per position
    int* dpos;
    TCK(hipMalloc(&dpos, 4));
    for (int t = 0; t < NT; t++) {
      TCK(hipMemcpy(dpos, &t, 4, hipMemcpyHostToDevice));
      k_store_kv_bf16<<<2, 256>>>(dkb, dv + (size_t)t * 512, dkc, dvc, dpos, dst);
      TCK(hipGetLastError());
    }
    auto get16 = [](const uint16_t* p, size_t n) {
      std::vector<uint16_t> h(n);
      TCK(hipMemcpy(h.data(), p, n * 2, hipMemcpyDeviceToHost));
      return h;
    };
    std::vector<uint16_t> ref = get16(dref, vt_n), st = get16(dst, vt_n);
    std::vector<uint16_t> vcd = get16(dvc, (size_t)NT * 512);
    // slots >= NT of the trailing partial block: the prefill writer zeroes
    // them (k_qsa_wmma<true> WMMA output depends on masked V bits), the
    // decode store leaves them alone (next prefill rewrites from its base)
    double bad_st = 0;
    for (size_t i = 0; i < vt_n; i++)
      bad_st += (int)(i / 2048) * 4 + (int)(i % 4) < NT ? st[i] != ref[i] : ref[i] != 0;
    for (size_t i = 0; i < vcd.size(); i++) bad_st += vcd[i] != vrow[i];
    check("qsa_store_bf16_vct_bits", bad_st, 0);
    // rebuild a ragged sub-range [r0, r1) from the row-major cache
    const int r0 = 2, r1 = 11;
    uint16_t* dvrow = dup(vrow);
    const int nb = (r1 + 3) / 4 - r0 / 4;
    k_bf16_rows_to_bt<<<(unsigned)((nb * 512 + 255) / 256), 256>>>(dvrow, dreb, r0, r1);
    TCK(hipGetLastError());
    std::vector<uint16_t> reb = get16(dreb, vt_n);
    double bad_rb = 0;
    for (int b = 0; b < NB; b++)
      for (int i = 0; i < 512; i++)
        for (int s = 0; s < 4; s++) {
          const size_t o = ((size_t)b * 512 + i) * 4 + s;
          const int t = b * 4 + s;
          const uint16_t e = t < r0 ? (uint16_t)0xEEEE
                             : t < r1 ? ref[o]
                             : b == (r1 - 1) / 4 ? (uint16_t)0 : (uint16_t)0xEEEE;
          bad_rb += reb[o] != e;
        }
    check("qsa_bt_rebuild_bits", bad_rb, 0);
    TCK(hipFree(dv));
    TCK(hipFree(dkb));
    TCK(hipFree(dkc));
    TCK(hipFree(dvc));
    TCK(hipFree(dref));
    TCK(hipFree(dst));
    TCK(hipFree(dreb));
    TCK(hipFree(dvrow));
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
    // A1c: paged k_qsa_wmma<*, true> with a reversed page table over a
    // physically permuted copy of kc / vc / vct must be bitwise identical to
    // the contiguous launches above (only addresses change).
    {
      const int NPG = (P + KV_PAGE - 1) / KV_PAGE, PROWS = NPG * KV_PAGE;
      std::vector<int> ptab(NPG);
      for (int p = 0; p < NPG; p++) ptab[p] = NPG - 1 - p;
      auto prow = [&](int t) { return ptab[t / KV_PAGE] * KV_PAGE + t % KV_PAGE; };
      std::vector<uint16_t> kp((size_t)PROWS * HKV * DH, 0), vp(kp.size(), 0),
          vctp((size_t)(PROWS / 4) * HKV * DH * 4, 0);
      for (int t = 0; t < P; t++) {
        const int r = prow(t);
        for (int j = 0; j < HKV * DH; j++) {
          kp[(size_t)r * HKV * DH + j] = kb[(size_t)t * HKV * DH + j];
          vp[(size_t)r * HKV * DH + j] = vb[(size_t)t * HKV * DH + j];
          vctp[((size_t)(r / 4) * HKV * DH + j) * 4 + t % 4] =
              vb[(size_t)t * HKV * DH + j];
        }
      }
      uint16_t *dkp = dup(kp), *dvp = dup(vp), *dvctp = dup(vctp);
      int* dptab = dup(ptab);
      float *dp5 = dalloc((size_t)P * HQ * DH), *dp7 = dalloc((size_t)P * HQ * DH);
      k_qsa_wmma<true, true><<<dim3(P - FIRST, HKV), 256>>>(
          dq, dkp, dvp, dvctp, dp5, P, DH, HKV, HQ, sel0, FIRST, 0, dptab);
      TCK(hipGetLastError());
      k_qsa_wmma<false, true><<<dim3(P - FIRST, HKV), 256>>>(
          dq, dkp, dvp, nullptr, dp7, P, DH, HKV, HQ, sel0, FIRST, 0, dptab);
      TCK(hipGetLastError());
      std::vector<float> p5 = dget(dp5, (size_t)P * HQ * DH),
                         p7 = dget(dp7, (size_t)P * HQ * DH);
      size_t off = (size_t)FIRST * HQ * DH, cnt = (size_t)(P - FIRST) * HQ * DH;
      bool ok5 = memcmp(p5.data() + off, o5.data() + off, cnt * 4) == 0;
      bool ok7 = memcmp(p7.data() + off, o7.data() + off, cnt * 4) == 0;
      printf("%-28s %s\n", "qsa_wmma_btv_paged_bits", ok5 ? "PASS" : "FAIL");
      printf("%-28s %s\n", "qsa_wmma_rm_paged_bits", ok7 ? "PASS" : "FAIL");
      fails += !ok5;
      fails += !ok7;
      TCK(hipFree(dkp));
      TCK(hipFree(dvp));
      TCK(hipFree(dvctp));
      TCK(hipFree(dptab));
      TCK(hipFree(dp5));
      TCK(hipFree(dp7));
    }
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

  // ---- 9h. k_qsa_wmma_u（4-token 块并集, GDEC_QSA_UNION）vs 9g 基线 + CPU ref ----
  // Same sparse setup as 9g with P=2064: tokens 2052..2063 form three aligned
  // groups that go through k_qsa_union_merge + k_qsa_wmma_u. Group 1's last
  // token list deliberately excludes its own block to exercise the merge
  // append + causal-override path. The merge output is checked exactly against
  // a host-side union; attention output uses the same 3x bound as 9g anchored
  // on qsa_wmma_btv (the union stream's key order differs per query, so the
  // result is not bitwise-equal to the per-token kernel).
  {
    const int P = 2064, HQ = 24, HKV = 2, DH = 256, FIRST = 2051, G0 = 2052;
    const int NG = (P - G0) / 4;  // 3 groups: 2052..2063
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
      int own = t / 4;
      std::vector<int> blocks;
      // group 1's last token (2059): exclude its own block (append path)
      int nmax = (t == G0 + 7) ? own : n;
      for (int i = 0; i < nmax; i++) blocks.push_back(i);
      std::shuffle(blocks.begin(), blocks.end(), rng);
      blocks.resize(512);
      std::sort(blocks.begin(), blocks.end());
      for (int i = 0; i < 512; i++) sel[(size_t)t * 512 + i] = blocks[i];
    }
    std::vector<uint16_t> vct(((size_t)(P + 3) / 4) * HKV * DH * 4, 0);
    for (int t = 0; t < P; t++)
      for (int h = 0; h < HKV; h++)
        for (int d = 0; d < DH; d++)
          vct[(((size_t)(t / 4) * HKV + h) * DH + d) * 4 + t % 4] =
              vb[((size_t)t * HKV + h) * DH + d];
    float* dq = dup(q);
    uint16_t *dkc = dup(kb), *dvc = dup(vb), *dvct = dup(vct);
    int* dsel = dup(sel);
    float* dout5 = dalloc((size_t)P * HQ * DH);
    float* doutu = dalloc((size_t)P * HQ * DH);
    uint16_t *dublk, *dumask;
    int* ducount;
    TCK(hipMalloc(&dublk, (size_t)NG * QSA_UCAP * 2));
    TCK(hipMalloc(&dumask, (size_t)NG * QSA_UCAP * 2));
    TCK(hipMalloc(&ducount, NG * 4));
    const int* sel0 = dsel + (size_t)FIRST * 512;
    k_qsa_wmma<true><<<dim3(P - FIRST, HKV), 256>>>(dq, dkc, dvc, dvct, dout5, P,
                                                    DH, HKV, HQ, sel0, FIRST);
    TCK(hipGetLastError());
    k_qsa_union_merge<<<NG, 128>>>(dsel, G0, dublk, dumask, ducount, QSA_UCAP);
    TCK(hipGetLastError());
    k_qsa_wmma_u<<<dim3(NG, HKV), 256>>>(dq, dkc, dvct, doutu, DH, HKV, HQ,
                                         dublk, dumask, ducount, QSA_UCAP, G0, 0);
    TCK(hipGetLastError());
    // host 侧精确校验 merge 输出（升序并集 + 掩码 + 自身块补登 + pad）
    {
      std::vector<uint16_t> hblk((size_t)NG * QSA_UCAP),
          hmsk((size_t)NG * QSA_UCAP);
      std::vector<int> hcnt(NG);
      TCK(hipMemcpy(hblk.data(), dublk, hblk.size() * 2, hipMemcpyDeviceToHost));
      TCK(hipMemcpy(hmsk.data(), dumask, hmsk.size() * 2, hipMemcpyDeviceToHost));
      TCK(hipMemcpy(hcnt.data(), ducount, NG * 4, hipMemcpyDeviceToHost));
      int merge_bad = 0;
      for (int g = 0; g < NG; g++) {
        const int t0 = G0 + 4 * g, own = t0 / 4;
        std::map<int, uint32_t> u;
        for (int qi = 0; qi < 4; qi++)
          for (int i = 0; i < 512; i++) {
            int b = sel[(size_t)(t0 + qi) * 512 + i];
            u[b] |= 0xFu << (4 * qi);
          }
        std::vector<std::pair<int, uint32_t>> exp(u.begin(), u.end());
        if (u.find(own) == u.end() || !(u[own] & (0xFu << 12)))
          exp.push_back({own, 0});  // 补登（mask=0）
        while (exp.size() & 3) exp.push_back({0xFFFF, 0});
        if ((int)exp.size() > QSA_UCAP || hcnt[g] != (int)exp.size()) {
          merge_bad++;
          continue;
        }
        for (size_t i = 0; i < exp.size(); i++)
          if (hblk[(size_t)g * QSA_UCAP + i] != (uint16_t)exp[i].first ||
              hmsk[(size_t)g * QSA_UCAP + i] != (uint16_t)exp[i].second)
            merge_bad++;
      }
      printf("%-28s %s\n", "qsa_union_merge", merge_bad ? "FAIL" : "PASS");
      fails += merge_bad != 0;
    }
    std::vector<float> o5 = dget(dout5, (size_t)P * HQ * DH),
                       ou = dget(doutu, (size_t)P * HQ * DH);
    std::vector<float> ref((size_t)P * HQ * DH, 0.f);
    float scale = 1.f / sqrtf((float)DH);
    for (int t = G0; t < P; t++) {
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
    double mabs5 = 0, mrel5 = 0, mabsu = 0, mrelu = 0, mabsuv = 0;
    for (int t = G0; t < P; t++)
      for (size_t i = (size_t)t * HQ * DH; i < (size_t)(t + 1) * HQ * DH; i++) {
        double d5 = fabs((double)o5[i] - ref[i]), du = fabs((double)ou[i] - ref[i]);
        mabs5 = std::max(mabs5, d5);
        mabsu = std::max(mabsu, du);
        mabsuv = std::max(mabsuv, (double)fabs(o5[i] - ou[i]));
        if (fabs((double)ref[i]) > 1e-3) {
          mrel5 = std::max(mrel5, d5 / fabs((double)ref[i]));
          mrelu = std::max(mrelu, du / fabs((double)ref[i]));
        }
      }
    double tola = std::min(3 * mabs5, 1e-2), tolr = 3 * mrel5;
    printf("%-28s maxabs=%.3e maxrel=%.3e  (reference row)\n", "qsa_wmma_btv_grp",
           mabs5, mrel5);
    printf("%-28s maxabs=%.3e maxrel=%.3e |o5-ou|=%.3e tol=(%.1e, %.1e)  %s\n",
           "qsa_wmma_u", mabsu, mrelu, mabsuv, tola, tolr,
           (mabsu <= tola && mrelu <= tolr) ? "PASS" : "FAIL");
    if (mabsu > tola || mrelu > tolr) fails++;
    TCK(hipFree(dq));
    TCK(hipFree(dkc));
    TCK(hipFree(dvc));
    TCK(hipFree(dvct));
    TCK(hipFree(dsel));
    TCK(hipFree(dout5));
    TCK(hipFree(doutu));
    TCK(hipFree(dublk));
    TCK(hipFree(dumask));
    TCK(hipFree(ducount));
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
    k_moe_tiles<<<1, 512>>>(deoff, dtiles, dntiles, E, 64);
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

  // ---- A2: host page allocator (KvPagePool) + sequence table (KvSeqTable) ----
  {
    bool ok = true;
    auto expect = [&](bool c, const char* what) {
      if (!c) {
        printf("  kvpool: %s\n", what);
        ok = false;
      }
    };
    {  // order 1: lowest free page first
      KvPagePool pool;
      pool.init(10, 1);
      for (int i = 0; i < 10; i++) expect(pool.alloc() == i, "order1 alloc order");
      expect(pool.alloc() == -1, "order1 exhaustion");
      pool.unref(7);
      pool.unref(3);
      pool.unref(5);
      expect(pool.alloc() == 3 && pool.alloc() == 5 && pool.alloc() == 7,
             "order1 lowest-first reuse");
      pool.addref(4);
      expect(!pool.unref(4) && pool.unref(4) && pool.nfree() == 1, "refcount");
    }
    {  // order 2: scrambled permutation, FIFO reuse
      const int n = 160;
      KvPagePool pool;
      pool.init(n, 2);
      std::vector<int> got;
      for (int i = 0; i < n; i++) got.push_back(pool.alloc());
      expect(pool.alloc() == -1, "order2 exhaustion");
      std::vector<int> s = got;
      std::sort(s.begin(), s.end());
      bool perm = true, mono = true;
      for (int i = 0; i < n; i++) perm = perm && s[i] == i;
      for (int i = 1; i < n; i++) mono = mono && got[i] == got[i - 1] + 1;
      expect(perm, "order2 not a permutation");
      expect(!mono, "order2 not scrambled");
      pool.unref(got[5]);
      pool.unref(got[2]);
      expect(pool.alloc() == got[5] && pool.alloc() == got[2], "order2 FIFO reuse");
    }
    {  // sequence table: prefix mapping, trim, guard entries
      const int n = 160;
      KvPagePool pool;
      pool.init(n, 1);
      KvSeqTable seq;
      seq.init(n, pool.guard());
      expect(seq.reserve(pool, 1) && seq.mapped == 1, "reserve 1 row");
      expect(seq.reserve(pool, 256) && seq.mapped == 1, "reserve full page");
      expect(seq.reserve(pool, 257) && seq.mapped == 2, "reserve page+1");
      expect(seq.reserve(pool, n * KV_PAGE) && seq.mapped == n, "reserve all");
      expect(!seq.reserve(pool, n * KV_PAGE + 1), "reserve beyond table");
      bool ident = true;
      for (int i = 0; i < n; i++) ident = ident && seq.tab[i] == i;
      expect(ident, "order1 fresh sequence is identity");
      expect(seq.trim(pool, 300) == n - 2 && seq.mapped == 2 && pool.nfree() == n - 2,
             "trim keeps the page holding row pos-1");
      bool guard = true;
      for (int i = 2; i < n; i++) guard = guard && seq.tab[i] == pool.guard();
      expect(guard, "trimmed entries point at the guard page");
      expect(seq.trim(pool, 0) == 2 && seq.mapped == 0 && pool.nfree() == n, "trim to 0");
      expect(seq.reserve(pool, 1000) && seq.tab[0] == 0 && seq.tab[3] == 3,
             "order1 re-map after reset is identity again");
    }
    {  // order 2 rotation + conservation under a random reserve/trim workload
      const int n = 64;
      KvPagePool pool;
      pool.init(n, 2);
      KvSeqTable seq;
      seq.init(n, pool.guard());
      std::mt19937 wr(7);
      int first_prev = -1, rotations = 0;
      for (int it = 0; it < 500; it++) {
        const int end = (int)(wr() % (unsigned)(n * KV_PAGE + 1));
        if (wr() % 3 == 0) {
          const bool had = seq.mapped > 0;
          seq.trim(pool, 0);
          seq.reserve(pool, std::max(1, end));
          if (had && seq.tab[0] != first_prev) rotations++;
          first_prev = seq.tab[0];
        } else if (wr() % 2) {
          seq.trim(pool, end);
        } else {
          expect(seq.reserve(pool, end), "random reserve failed");
        }
        std::vector<char> seen((size_t)n, 0);
        bool dup = false;
        for (int i = 0; i < seq.mapped; i++) {
          const int p = seq.tab[i];
          if (p < 0 || p >= n || seen[(size_t)p]) dup = true;
          else seen[(size_t)p] = 1;
        }
        for (int i = seq.mapped; i < n; i++) dup = dup || seq.tab[i] != pool.guard();
        if (dup || pool.nfree() + seq.mapped != n) {
          expect(false, "conservation / double mapping");
          break;
        }
        if (seq.mapped == 0) first_prev = -1;
      }
      expect(rotations > 0, "order2 never rotated physical pages");
    }
    printf("%-28s %s\n", "kvpool_alloc_table", ok ? "PASS" : "FAIL");
    fails += !ok;
    // A3: checkpoint pin / adopt / copy-on-write bookkeeping
    ok = true;
    {
      const int n = 16;
      KvPagePool pool;
      pool.init(n, 1);
      KvSeqTable seq;
      seq.init(n, pool.guard());
      expect(seq.reserve(pool, 1000) && seq.mapped == 4, "a3 reserve");
      std::vector<int> ck = seq.pin(pool, 900);  // ceil(900/256) = 4 pages
      expect(ck.size() == 4 && pool.ref[3] == 2, "a3 pin takes a reference");
      seq.trim(pool, 0);  // reset_state: the pinned pages stay allocated
      expect(pool.nfree() == n - 4, "a3 pinned pages survive reset");
      expect(seq.reserve(pool, 600) && seq.tab[0] == 4, "a3 next sequence gets fresh pages");
      seq.adopt(pool, ck);  // rckpt restore
      expect(seq.mapped == 4 && seq.tab[3] == 3 && pool.nfree() == n - 4 &&
                 pool.ref[3] == 2 && seq.tab[4] == pool.guard(),
             "a3 adopt maps the checkpoint pages");
      const int p = pool.alloc();  // copy-on-write of the tail page
      seq.replace(pool, 3, p);
      expect(seq.tab[3] == p && pool.ref[3] == 1 && pool.ref[(size_t)p] == 1, "a3 replace");
      KvSeqTable::unpin(pool, ck);  // checkpoint evicted
      expect(pool.ref[3] == 0 && pool.ref[0] == 1 && pool.nfree() == n - 4, "a3 unpin");
      seq.trim(pool, 0);
      expect(pool.nfree() == n, "a3 conservation");
      std::vector<int> e = seq.pin(pool, 0);
      expect(e.empty(), "a3 pin of an empty prefix");
    }
    printf("%-28s %s\n", "kvpool_pin_cow", ok ? "PASS" : "FAIL");
    fails += !ok;
  }

  printf(fails ? "== %d FAILURES ==\n" : "== ALL PASS ==\n", fails);
  return fails ? 1 : 0;
}
