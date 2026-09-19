// ref.cpp — Phase 1 CPU reference forward for qwen3.8-flash-next.
// Token-by-token (decode-mode) forward; QSA computed as full causal attention
// (exact for sequences <= 2048+3 tokens, where the top-512-block budget covers
// everything). PLE n-gram layer implemented per official modeling code.
//
// Architecture ground truth: transformers 5.16.1 modeling_qwen4_exp.py
// (reference/ dir) + config.json. All former GUESS items resolved.
//
// Build: g++ -O3 -fopenmp -o ref ref.cpp
// Usage: ref <base.hgn> [overlay.hgn] --tokens 1,2,3 [--gen N] [--topk K]
//        [--theta T] [--dump] [--no-ple]

#include "hgn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using hgn::Checkpoint;
using hgn::Tensor;

// ---------------- config (derived from tensor shapes + tech report) --------
struct Cfg {
  int d = 2560;
  int layers = 48;
  int branches = 4;           // residual width 4*2560 = 10240
  // GDN
  int gdn_hk = 16, gdn_dk = 128;   // q,k heads
  int gdn_hv = 48, gdn_dv = 128;   // v heads
  int gdn_conv = 4;
  // QSA
  int qsa_hq = 24, qsa_dh = 256;   // q heads (q_proj = 24*512 fused [q|gate])
  int qsa_hkv = 2;
  int rotary_dim = 64;             // partial RoPE (tech report §2.1.2)
  // GR
  int gr_rank = 320;
  // MoE
  int experts = 512, topk = 10, moe_mid = 640;
  // misc
  float norm_eps = 1e-6f;          // rms_norm_eps
  double rope_theta = 1e7;         // rope_parameters.rope_theta
  int vocab = 248320;              // set from lm_head rows at runtime
};
static Cfg g_cfg;

static bool is_qsa(int layer) { return layer % 4 == 3; }

// ---------------- small ops -------------------------------------------------
static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
static inline float silu(float x) { return x / (1.f + expf(-x)); }
static inline float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// zero-centered RMSNorm (tech report §2.1.1): gain = (1 + w)
static void rmsnorm_zc(const float* x, const float* w, float* out, int n, float eps) {
  double ss = 0;
  for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
  float inv = (float)(1.0 / sqrt(ss / n + eps));
  for (int i = 0; i < n; i++) out[i] = x[i] * inv * (1.f + w[i]);
}

// y = W x, W [out,in] row-major; on-the-fly dequant per row.
static void matvec(const Checkpoint& ckpt, const Tensor& W, const float* x, float* y,
                   int rows, int cols) {
  if (W.dtype == 0) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
      const uint16_t* w = (const uint16_t*)W.data + (size_t)r * cols;
      double acc = 0;
      for (int c = 0; c < cols; c++) acc += (double)hgn::bf16_to_f32(w[c]) * x[c];
      y[r] = (float)acc;
    }
  } else if (W.dtype == 5) {
    auto q = Checkpoint::q4cp_parse(W);
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < (int64_t)rows; r++) {
      std::vector<float> wr(cols);
      Checkpoint::q4cp_row(q, r, wr.data());
      double acc = 0;
      for (int c = 0; c < cols; c++) acc += (double)wr[c] * x[c];
      y[r] = (float)acc;
    }
  } else if (W.dtype == 7) {  // q8g64: uint8 + (fp16 scale, fp16 min) per 64, per row
    size_t stride = (size_t)cols + cols / 64 * 4;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
      const uint8_t* codes = W.data + (size_t)r * stride;
      const uint16_t* sm = (const uint16_t*)(codes + cols);
      double acc = 0;
      for (int g = 0; g < cols / 64; g++) {
        float s = hgn::fp16_to_f32(sm[g * 2]), mn = hgn::fp16_to_f32(sm[g * 2 + 1]);
        double a2 = 0, csum = 0;
        for (int j = 0; j < 64; j++) {
          a2 += (double)codes[g * 64 + j] * x[g * 64 + j];
          csum += x[g * 64 + j];
        }
        acc += a2 * s + mn * csum;
      }
      y[r] = (float)acc;
    }
  } else {
    throw std::runtime_error("matvec: unsupported dtype " + std::to_string(W.dtype));
  }
}

static void load_vec(const Checkpoint& ckpt, const Tensor& t, float* out) {
  ckpt.dequant(t, out);  // small tensors only
}

static const Tensor& T(const Checkpoint& ckpt, const char* fmt, int layer, char* buf,
                       size_t bufsz) {
  snprintf(buf, bufsz, fmt, layer);
  return ckpt.at(buf);
}

// ---------------- model state ----------------------------------------------
struct GdnState {
  std::vector<float> conv;  // (conv_k-1) x 10240 history of qkv projections
  std::vector<float> S;     // 48 heads x 128 x 128
};
struct QsaState {
  std::vector<float> k, v;  // pos x (2*256)
};

// PLE (n-gram embedding, layers.1.ple) — official semantics from
// transformers modeling_qwen4_exp.py (Qwen4ExpTextPLELayer/NGramEmbedding).
struct PleState {
  std::vector<int64_t> chist;     // raw token id per position
  std::vector<float> ring;        // 9 x 10240 normed gated values (conv dilation 3)
  int ringpos = 0;
  // cached dequantized weights
  std::vector<float> Wk;          // (10240, 2560) key_proj, per-branch stacked
  std::vector<float> Wv;          // (2560, 2560) value_proj
  std::vector<float> convw;       // (10240, 4)
  std::vector<float> norm_key, norm_query, norm_conv;  // (10240) = 4 x 2560
  int64_t mult[3];
  int64_t vsz[16], voff[16];
  const Tensor* table = nullptr;  // FP8 (320M, 160) + trailing fp32 scale
  float tscale = 1.f;
  int pad_id = 248044;            // eos_token_id (<|endoftext|>)
  float eps = 1e-6f;              // rms_norm_eps
  int dilation = 3;               // conv_dilation = ngram_size
};

struct Model {
  const Checkpoint& ckpt;
  std::vector<GdnState> gdn_st;
  std::vector<QsaState> qsa_st;
  std::vector<float> R;  // branches x d residual stream
  int pos = 0;
  PleState ple;
  bool ple_on = true;

  explicit Model(const Checkpoint& c) : ckpt(c) {
    gdn_st.resize(g_cfg.layers);
    qsa_st.resize(g_cfg.layers);
    for (int l = 0; l < g_cfg.layers; l++) {
      if (is_qsa(l)) {
        qsa_st[l].k.reserve(4096 * g_cfg.qsa_hkv * g_cfg.qsa_dh);
        qsa_st[l].v.reserve(4096 * g_cfg.qsa_hkv * g_cfg.qsa_dh);
      } else {
        gdn_st[l].conv.assign((g_cfg.gdn_conv - 1) * 10240, 0.f);
        gdn_st[l].S.assign((size_t)g_cfg.gdn_hv * g_cfg.gdn_dk * g_cfg.gdn_dv, 0.f);
      }
    }
    R.assign(g_cfg.branches * g_cfg.d, 0.f);

    // PLE weights (layers.1.ple.*)
    char buf[128];
    auto L1 = [&](const char* name) -> const Tensor& {
      snprintf(buf, sizeof buf, "layers.1.ple.%s", name);
      return ckpt.at(buf);
    };
    ple.Wk.resize(10240 * 2560);
    ckpt.dequant(L1("key_proj.weight"), ple.Wk.data());
    ple.Wv.resize(2560 * 2560);
    ckpt.dequant(L1("value_proj.weight"), ple.Wv.data());
    ple.convw.resize(10240 * 4);
    ckpt.dequant(L1("conv1d.weight"), ple.convw.data());
    for (auto& kv : std::vector<std::pair<const char*, std::vector<float>*>>{
             {"norm_key.weight", &ple.norm_key},
             {"norm_query.weight", &ple.norm_query},
             {"norm_conv.weight", &ple.norm_conv}}) {
      kv.second->resize(10240);
      ckpt.dequant(L1(kv.first), kv.second->data());
    }
    {
      const Tensor& mt = L1("ple_embedding.layer_multipliers");
      memcpy(ple.mult, mt.data, 24);
      const Tensor& vs = L1("ple_embedding.ngram_heads_vocab_sizes");
      memcpy(ple.vsz, vs.data, 128);
      const Tensor& vo = L1("ple_embedding.ngram_heads_offsets");
      memcpy(ple.voff, vo.data, 128);
      ple.table = &L1("ngram_embedding.weight");
      if (ple.table->dtype != 10) throw std::runtime_error("ple table dtype");
      memcpy(&ple.tscale, ple.table->data + ple.table->numel(), 4);
      fprintf(stderr, "ple table scale = %.9g\n", ple.tscale);
    }
    ple.ring.assign(9 * 10240, 0.f);
  }

  // --- PLE: n-gram hash retrieval + context-aware gating ---------------------
  // Official: e = concat 16 heads' rows; k = Wk e (10240); per branch m:
  //   g = (norm_k(k_m)·norm_q(R_m))/sqrt(d); g = sign(g)*sqrt(max(|g|,1e-6))
  //   u_m = σ(g) * (Wv e);  out_m = u_m + SiLU(Conv_dil3(norm_conv(u_m)))
  //   R_m += out_m     (at layer 1 entry, before this layer's attn GR read)
  void ple_step(int token) {
    // Official modeling_qwen4_exp.py hashes RAW token ids (no text
    // canonicalization); shifts are padded with eos_token_id.
    ple.chist.push_back(token);
    int t = (int)ple.chist.size() - 1;
    auto cat = [&](int back) -> int64_t {  // token id `back` positions ago
      int j = t - back;
      if (j < 0) return ple.pad_id;
      return ple.chist[j];
    };
    // hashes: heads 0-7 order 2, heads 8-15 order 3
    float e[2560];
    for (int h = 0; h < 16; h++) {
      int64_t mix = cat(0) * ple.mult[0] ^ (cat(1) * ple.mult[1]);
      if (h >= 8) mix ^= cat(2) * ple.mult[2];
      uint64_t row = (uint64_t)(ple.voff[h] + (int64_t)(mix % ple.vsz[h]));
      if (dbg) fprintf(stderr, "  ple t=%d h=%d row=%llu\n", t, h, (unsigned long long)row);
      const uint8_t* p = ple.table->data + row * 160;
      for (int i = 0; i < 160; i++) e[h * 160 + i] = hgn::fp8e4m3_to_f32(p[i]) * ple.tscale;
    }
    if (dbg) {
      fprintf(stderr, "  ple e head0:");
      for (int i = 0; i < 8; i++) fprintf(stderr, " %.6f", e[i]);
      fprintf(stderr, "\n");
    }
    // v = Wv e ; keys = Wk e
    float v[2560], keys[10240];
    for (int r = 0; r < 2560; r++) {
      double a = 0;
      for (int c2 = 0; c2 < 2560; c2++) a += (double)ple.Wv[(size_t)r * 2560 + c2] * e[c2];
      v[r] = (float)a;
    }
    for (int r = 0; r < 10240; r++) {
      double a = 0;
      for (int c2 = 0; c2 < 2560; c2++) a += (double)ple.Wk[(size_t)r * 2560 + c2] * e[c2];
      keys[r] = (float)a;
    }
    // per-branch gate, value, conv
    float U[10240];
    for (int m = 0; m < 4; m++) {
      float kn[2560], qn[2560];
      ple_norm(keys + m * 2560, ple.norm_key.data() + m * 2560, kn, 2560);
      ple_norm(R.data() + m * 2560, ple.norm_query.data() + m * 2560, qn, 2560);
      double g = 0;
      for (int i = 0; i < 2560; i++) g += (double)kn[i] * qn[i];
      g /= sqrt(2560.0);
      // signed sqrt with clamp: sign(g)*sqrt(max(|g|,1e-6))
      g = (g < 0 ? -1.0 : 1.0) * sqrt(fmax(fabs(g), 1e-6));
      float alpha = sigmoid((float)g);
      if (dbg) fprintf(stderr, "  ple t=%d branch=%d g=%.6f alpha=%.6f v0=%.6f\n", t, m,
                       (double)g, alpha, v[0]);
      for (int i = 0; i < 2560; i++) U[m * 2560 + i] = alpha * v[i];
      // conv input: normed U per branch
      float un[2560];
      ple_norm(U + m * 2560, ple.norm_conv.data() + m * 2560, un, 2560);
      memcpy(ple.ring.data() + ((size_t)ple.ringpos * 10240) + m * 2560, un,
             2560 * sizeof(float));
    }
    // depthwise causal conv k=4 dilation=d over the ring.
    // F.conv1d semantics: tap j pairs with x[t-(3-j)*d] (j=3 = current).
    for (int m = 0; m < 4; m++) {
      for (int i = 0; i < 2560; i++) {
        int ch = m * 2560 + i;
        double a = 0;
        for (int j = 0; j < 4; j++) {
          int back = (3 - j) * ple.dilation;
          int rp = ((ple.ringpos - back) % 9 + 9) % 9;
          a += (double)ple.convw[ch * 4 + j] * ple.ring[(size_t)rp * 10240 + ch];
        }
        float co = silu((float)a);
        if (dbg && i < 4)
          fprintf(stderr, "  ple t=%d ch=%d conv=%.6f U=%.6f\n", t, ch, (float)a, U[ch]);
        R[ch] += U[ch] + co;
      }
    }
    ple.ringpos = (ple.ringpos + 1) % 9;
  }

  // PLE RMSNorm: zero-centered (1+w) like every Qwen4ExpTextRMSNorm,
  // eps = rms_norm_eps = 1e-6
  void ple_norm(const float* x, const float* w, float* out, int n) {
    rmsnorm_zc(x, w, out, n, ple.eps);
  }


  // Eq 30-32: per-branch zero-centered RMSNorm, gate G = unvec(σ(Wu SiLU(Wd vec/nr))),
  // x = mean_i G_i ⊙ R̂_i
  void gr_read(const char* prefix, int layer, const float* /*unused*/, float* x_out,
               std::vector<float>& Rhat) {
    char buf[128];
    const int rb = g_cfg.branches * g_cfg.d;
    Rhat.resize(rb);
    const Tensor& normt = T(ckpt, prefix, layer, buf, sizeof buf);  // hc_norm
    std::vector<float> gamma(rb);
    load_vec(ckpt, normt, gamma.data());
    for (int b = 0; b < g_cfg.branches; b++)
      rmsnorm_zc(R.data() + b * g_cfg.d, gamma.data() + b * g_cfg.d, Rhat.data() + b * g_cfg.d,
                 g_cfg.d, g_cfg.norm_eps);

    snprintf(buf, sizeof buf, "%s", "");  // reuse buf below
    // down/up weights live beside hc_norm: replace suffix
    std::string base(prefix);
    auto pos = base.find("hc_norm");
    base.replace(pos, 7, "input_mix_weight_%s");
    char b2[128];
    snprintf(b2, sizeof b2, base.c_str(), "down");
    std::vector<float> Wd(g_cfg.gr_rank * rb);
    load_vec(ckpt, ckpt.at(b2), Wd.data());
    snprintf(b2, sizeof b2, base.c_str(), "up");
    std::vector<float> Wu(rb * g_cfg.gr_rank);
    load_vec(ckpt, ckpt.at(b2), Wu.data());

    // t = (1/nr) Wd vec ; G = σ(Wu SiLU(t))
    std::vector<float> t(g_cfg.gr_rank, 0.f);
    for (int r = 0; r < g_cfg.gr_rank; r++) {
      double acc = 0;
      for (int c = 0; c < rb; c++) acc += (double)Wd[(size_t)r * rb + c] * Rhat[c];
      t[r] = (float)(acc / g_cfg.branches);
    }
    for (auto& v : t) v = silu(v);
    std::vector<float> G(rb);
    for (int r = 0; r < rb; r++) {
      double acc = 0;
      for (int c = 0; c < g_cfg.gr_rank; c++)
        acc += (double)Wu[(size_t)r * g_cfg.gr_rank + c] * t[c];
      G[r] = sigmoid((float)acc);
    }
    for (int i = 0; i < g_cfg.d; i++) {
      float acc = 0;
      for (int b = 0; b < g_cfg.branches; b++) acc += G[b * g_cfg.d + i] * Rhat[b * g_cfg.d + i];
      x_out[i] = acc / g_cfg.branches;
    }
  }

  // --- GR write: s = 2σ((1/nr) Ww vec(R̂)); R_i += s_i y --------------------
  void gr_write(const char* hc_prefix, int layer, const float* y, const std::vector<float>& Rhat) {
    std::string base(hc_prefix);
    auto pos = base.find("hc_norm");
    base.replace(pos, 7, "block_inject_weight");
    const Tensor& Ww_t = ckpt.at(base);
    std::vector<float> Ww(g_cfg.branches * g_cfg.branches * g_cfg.d);
    load_vec(ckpt, Ww_t, Ww.data());
    int rb = g_cfg.branches * g_cfg.d;
    for (int b = 0; b < g_cfg.branches; b++) {
      double acc = 0;
      for (int c = 0; c < rb; c++) acc += (double)Ww[(size_t)b * rb + c] * Rhat[c];
      float s = 2.f * sigmoid((float)(acc / g_cfg.branches));
      for (int i = 0; i < g_cfg.d; i++) R[b * g_cfg.d + i] += s * y[i];
    }
  }

  // --- GDN (tech report §2.1.1, Eq 1-11) ------------------------------------
  void gdn(int l, const float* x, float* y_out) {
    char buf[128];
    auto& st = gdn_st[l];
    const int qk = g_cfg.gdn_hk * g_cfg.gdn_dk;      // 2048
    const int vv = g_cfg.gdn_hv * g_cfg.gdn_dv;      // 6144
    const int qkv = 2 * qk + vv;                     // 10240

    std::vector<float> qkv_buf(qkv);
    matvec(ckpt, T(ckpt, "layers.%d.linear_attn.in_proj_qkv.weight", l, buf, sizeof buf), x,
           qkv_buf.data(), qkv, g_cfg.d);

    // causal depthwise conv k=4 over qkv channels, then SiLU
    std::vector<float> c(qkv);
    const Tensor& convt = T(ckpt, "layers.%d.linear_attn.conv1d.weight", l, buf, sizeof buf);
    std::vector<float> cw(qkv * g_cfg.gdn_conv);
    load_vec(ckpt, convt, cw.data());  // [10240, 1, 4]
    for (int ch = 0; ch < qkv; ch++) {
      double acc = 0;
      for (int j = 0; j < g_cfg.gdn_conv - 1; j++)
        acc += (double)cw[ch * g_cfg.gdn_conv + j] * st.conv[j * qkv + ch];
      acc += (double)cw[ch * g_cfg.gdn_conv + g_cfg.gdn_conv - 1] * qkv_buf[ch];
      c[ch] = (float)acc;
    }
    // shift conv history
    memmove(st.conv.data(), st.conv.data() + qkv, (g_cfg.gdn_conv - 2) * qkv * sizeof(float));
    memcpy(st.conv.data() + (g_cfg.gdn_conv - 2) * qkv, qkv_buf.data(), qkv * sizeof(float));

    for (auto& v : c) v = silu(v);
    const float* q = c.data();
    const float* k = c.data() + qk;
    const float* v = c.data() + qk * 2;

    // L2-normalize q,k per head: x * rsqrt(sum(x^2) + 1e-6) (fla convention);
    // q additionally scaled by 1/sqrt(head_k_dim) inside the kernel.
    std::vector<float> qn(qk), kn(qk);
    const float qscale = 1.f / sqrtf((float)g_cfg.gdn_dk);
    for (int h = 0; h < g_cfg.gdn_hk; h++) {
      double sq = 0, sk = 0;
      for (int i = 0; i < g_cfg.gdn_dk; i++) {
        sq += (double)q[h * g_cfg.gdn_dk + i] * q[h * g_cfg.gdn_dk + i];
        sk += (double)k[h * g_cfg.gdn_dk + i] * k[h * g_cfg.gdn_dk + i];
      }
      float iq = (float)(1.0 / sqrt(sq + 1e-6)), ik = (float)(1.0 / sqrt(sk + 1e-6));
      for (int i = 0; i < g_cfg.gdn_dk; i++) {
        qn[h * g_cfg.gdn_dk + i] = q[h * g_cfg.gdn_dk + i] * iq * qscale;
        kn[h * g_cfg.gdn_dk + i] = k[h * g_cfg.gdn_dk + i] * ik;
      }
    }

    // gates: α = exp(-exp(A) softplus(a + bα)), β = σ(b)   (Eq 9-10)
    std::vector<float> a(g_cfg.gdn_hv), b(g_cfg.gdn_hv);
    matvec(ckpt, T(ckpt, "layers.%d.linear_attn.in_proj_a.weight", l, buf, sizeof buf), x,
           a.data(), g_cfg.gdn_hv, g_cfg.d);
    matvec(ckpt, T(ckpt, "layers.%d.linear_attn.in_proj_b.weight", l, buf, sizeof buf), x,
           b.data(), g_cfg.gdn_hv, g_cfg.d);
    std::vector<float> A_log(g_cfg.gdn_hv), dt_bias(g_cfg.gdn_hv);
    load_vec(ckpt, T(ckpt, "layers.%d.linear_attn.A_log", l, buf, sizeof buf), A_log.data());
    load_vec(ckpt, T(ckpt, "layers.%d.linear_attn.dt_bias", l, buf, sizeof buf), dt_bias.data());

    // recurrence per value head (Eq 1-4); k/q shared in groups of hv/hk
    std::vector<float> y(vv);
    const int group = g_cfg.gdn_hv / g_cfg.gdn_hk;  // 3
    for (int h = 0; h < g_cfg.gdn_hv; h++) {
      float alpha = expf(-expf(A_log[h]) * softplus(a[h] + dt_bias[h]));
      float beta = sigmoid(b[h]);
      const float* kh = kn.data() + (h / group) * g_cfg.gdn_dk;
      const float* qh = qn.data() + (h / group) * g_cfg.gdn_dk;
      const float* vh = v + h * g_cfg.gdn_dv;
      float* S = st.S.data() + (size_t)h * g_cfg.gdn_dk * g_cfg.gdn_dv;
      std::vector<float> e(g_cfg.gdn_dv);
      for (int i = 0; i < g_cfg.gdn_dv; i++) {
        double acc = 0;
        for (int j = 0; j < g_cfg.gdn_dk; j++)
          acc += (double)S[(size_t)j * g_cfg.gdn_dv + i] * kh[j];
        e[i] = vh[i] - alpha * (float)acc;
      }
      for (int j = 0; j < g_cfg.gdn_dk; j++) {
        float bk = beta * kh[j];
        for (int i = 0; i < g_cfg.gdn_dv; i++)
          S[(size_t)j * g_cfg.gdn_dv + i] = alpha * S[(size_t)j * g_cfg.gdn_dv + i] + bk * e[i];
      }
      for (int i = 0; i < g_cfg.gdn_dv; i++) {
        double acc = 0;
        for (int j = 0; j < g_cfg.gdn_dk; j++)
          acc += (double)S[(size_t)j * g_cfg.gdn_dv + i] * qh[j];
        y[h * g_cfg.gdn_dv + i] = (float)acc;
      }
    }

    // output gate (RMSNormGated, activation = output_gate_type = sigmoid):
    // per-v-head plain-gain RMSNorm, then * sigmoid(z)
    std::vector<float> z(vv);
    matvec(ckpt, T(ckpt, "layers.%d.linear_attn.in_proj_z.weight", l, buf, sizeof buf), x,
           z.data(), vv, g_cfg.d);
    std::vector<float> nw(g_cfg.gdn_dv);
    load_vec(ckpt, T(ckpt, "layers.%d.linear_attn.norm.weight", l, buf, sizeof buf), nw.data());
    std::vector<float> og(vv);
    for (int h = 0; h < g_cfg.gdn_hv; h++) {
      std::vector<float> ny(g_cfg.gdn_dv);
      double ss = 0;
      for (int i = 0; i < g_cfg.gdn_dv; i++)
        ss += (double)y[h * g_cfg.gdn_dv + i] * y[h * g_cfg.gdn_dv + i];
      float inv = (float)(1.0 / sqrt(ss / g_cfg.gdn_dv + g_cfg.norm_eps));
      for (int i = 0; i < g_cfg.gdn_dv; i++)
        ny[i] = y[h * g_cfg.gdn_dv + i] * inv * nw[i];  // plain gain (not 1+w)
      for (int i = 0; i < g_cfg.gdn_dv; i++)
        og[h * g_cfg.gdn_dv + i] = sigmoid(z[h * g_cfg.gdn_dv + i]) * ny[i];
    }
    matvec(ckpt, T(ckpt, "layers.%d.linear_attn.out_proj.weight", l, buf, sizeof buf),
           og.data(), y_out, g_cfg.d, vv);
  }

  // --- QSA as full causal attention (exact for pos < ~2048) ------------------
  void qsa(int l, const float* x, float* y_out) {
    char buf[128];
    auto& st = qsa_st[l];
    const int hq = g_cfg.qsa_hq, dh = g_cfg.qsa_dh, hkv = g_cfg.qsa_hkv;

    std::vector<float> qg(hq * 2 * dh);  // per-head [q(256)|gate(256)] (chunk(2))
    matvec(ckpt, T(ckpt, "layers.%d.self_attn.q_proj.weight", l, buf, sizeof buf), x,
           qg.data(), hq * 2 * dh, g_cfg.d);
    std::vector<float> kb(hkv * dh), vb(hkv * dh);
    matvec(ckpt, T(ckpt, "layers.%d.self_attn.k_proj.weight", l, buf, sizeof buf), x,
           kb.data(), hkv * dh, g_cfg.d);
    matvec(ckpt, T(ckpt, "layers.%d.self_attn.v_proj.weight", l, buf, sizeof buf), x,
           vb.data(), hkv * dh, g_cfg.d);

    std::vector<float> qnw(dh), knw(dh);
    load_vec(ckpt, T(ckpt, "layers.%d.self_attn.q_norm.weight", l, buf, sizeof buf),
             qnw.data());
    load_vec(ckpt, T(ckpt, "layers.%d.self_attn.k_norm.weight", l, buf, sizeof buf),
             knw.data());

    // per-head qk norm (zero-centered), then partial RoPE on first rotary_dim
    auto rope = [&](float* vec, int nheads) {
      for (int h = 0; h < nheads; h++) {
        float* p = vec + h * dh;
        for (int i = 0; i < g_cfg.rotary_dim / 2; i++) {
          double ang = pos * pow(g_cfg.rope_theta, -2.0 * i / g_cfg.rotary_dim);
          float cs = (float)cos(ang), sn = (float)sin(ang);
          float x0 = p[i], x1 = p[i + g_cfg.rotary_dim / 2];
          p[i] = x0 * cs - x1 * sn;
          p[i + g_cfg.rotary_dim / 2] = x0 * sn + x1 * cs;
        }
      }
    };
    std::vector<float> qs(hq * dh), gs(hq * dh);
    for (int h = 0; h < hq; h++) {
      rmsnorm_zc(qg.data() + h * 2 * dh, qnw.data(), qs.data() + h * dh, dh, g_cfg.norm_eps);
      memcpy(gs.data() + h * dh, qg.data() + h * 2 * dh + dh, dh * sizeof(float));
    }
    std::vector<float> ks(hkv * dh);
    for (int h = 0; h < hkv; h++)
      rmsnorm_zc(kb.data() + h * dh, knw.data(), ks.data() + h * dh, dh, g_cfg.norm_eps);
    rope(qs.data(), hq);
    rope(ks.data(), hkv);

    st.k.insert(st.k.end(), ks.begin(), ks.end());
    st.v.insert(st.v.end(), vb.begin(), vb.end());

    // causal attention over all cached positions
    int ntok = (int)st.k.size() / (hkv * dh);
    float scale = 1.f / sqrtf((float)dh);
    std::vector<float> out(hq * dh, 0.f), attn(ntok);
    for (int h = 0; h < hq; h++) {
      const float* qh = qs.data() + h * dh;
      int kvh = h / (hq / hkv);  // repeat_kv grouping: 12 consecutive q heads per kv head
      double mx = -1e30;
      for (int t = 0; t < ntok; t++) {
        const float* kt = st.k.data() + ((size_t)t * hkv + kvh) * dh;
        double acc = 0;
        for (int i = 0; i < dh; i++) acc += (double)qh[i] * kt[i];
        attn[t] = (float)(acc * scale);
        mx = std::max(mx, (double)attn[t]);
      }
      double sum = 0;
      for (int t = 0; t < ntok; t++) {
        attn[t] = expf(attn[t] - (float)mx);
        sum += attn[t];
      }
      for (int t = 0; t < ntok; t++) {
        float w = (float)(attn[t] / sum);
        const float* vt = st.v.data() + ((size_t)t * hkv + kvh) * dh;
        for (int i = 0; i < dh; i++) out[h * dh + i] += w * vt[i];
      }
      // output gate σ(gate) ⊙ attn (Qiu et al., cited in tech report §3.1)
      for (int i = 0; i < dh; i++) out[h * dh + i] *= sigmoid(gs[h * dh + i]);
    }
    matvec(ckpt, T(ckpt, "layers.%d.self_attn.o_proj.weight", l, buf, sizeof buf),
           out.data(), y_out, g_cfg.d, hq * dh);
  }

  // --- MoE: softmax router, top-k=10, probs renormalized (norm_topk_prob) ----
  void moe(int l, const float* x, float* y_out) {
    char buf[128];
    std::vector<float> logits(g_cfg.experts);
    matvec(ckpt, T(ckpt, "layers.%d.mlp.gate.weight", l, buf, sizeof buf), x, logits.data(),
           g_cfg.experts, g_cfg.d);
    // official: fp32 softmax over all experts, top-k on probs, then / sum
    if (dbg)
      fprintf(stderr,
              "ref L%02d router logits[0..7]: %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f | [9]=%.5f [10]=%.5f [71]=%.5f [215]=%.5f\n",
              l, logits[0], logits[1], logits[2], logits[3], logits[4], logits[5], logits[6],
              logits[7], logits[9], logits[10], logits[71], logits[215]);
    std::vector<float> sc(g_cfg.experts);
    double mx = -1e300;
    for (int i = 0; i < g_cfg.experts; i++) mx = std::max(mx, (double)logits[i]);
    double se = 0;
    for (int i = 0; i < g_cfg.experts; i++) {
      sc[i] = (float)exp((double)logits[i] - mx);
      se += sc[i];
    }
    for (int i = 0; i < g_cfg.experts; i++) sc[i] = (float)(sc[i] / se);
    std::vector<int> idx(g_cfg.experts);
    for (int i = 0; i < g_cfg.experts; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + g_cfg.topk, idx.end(),
                      [&](int a, int b) { return sc[a] > sc[b]; });
    double wsum = 0;
    for (int i = 0; i < g_cfg.topk; i++) wsum += sc[idx[i]];
    if (dbg) {
      fprintf(stderr, "ref L%02d moe topk:", l);
      for (int i = 0; i < g_cfg.topk; i++)
        fprintf(stderr, " %d:%.5f", idx[i], (double)(sc[idx[i]] / wsum));
      fprintf(stderr, "\n");
    }

    std::vector<float> acc(g_cfg.d, 0.f);
    const Tensor& gu = T(ckpt, "layers.%d.mlp.experts.gate_up_proj.weight", l, buf, sizeof buf);
    const Tensor& dn = T(ckpt, "layers.%d.mlp.experts.down_proj.weight", l, buf, sizeof buf);
    // fused experts: flat [E*2*mid, d] and [E*d, mid] Q4C-P with one codebook each
    auto qgu = Checkpoint::q4cp_parse(gu);
    auto qdn = Checkpoint::q4cp_parse(dn);
    std::vector<float> guv(2 * g_cfg.moe_mid), hid(g_cfg.moe_mid), ey(g_cfg.d);
    std::vector<float> wr(std::max(g_cfg.d, g_cfg.moe_mid));
    for (int i = 0; i < g_cfg.topk; i++) {
      int e = idx[i];
      float w = (float)(sc[e] / wsum);
      for (int r = 0; r < 2 * g_cfg.moe_mid; r++) {
        Checkpoint::q4cp_row(qgu, (uint64_t)e * 2 * g_cfg.moe_mid + r, wr.data());
        double a2 = 0;
        for (int c = 0; c < g_cfg.d; c++) a2 += (double)wr[c] * x[c];
        guv[r] = (float)a2;
      }
      // gate_up_proj chunk(2): first half = gate (silu), second half = up
      for (int r = 0; r < g_cfg.moe_mid; r++)
        hid[r] = silu(guv[r]) * guv[g_cfg.moe_mid + r];
      for (int r = 0; r < g_cfg.d; r++) {
        Checkpoint::q4cp_row(qdn, (uint64_t)e * g_cfg.d + r, wr.data());
        double a2 = 0;
        for (int c = 0; c < g_cfg.moe_mid; c++) a2 += (double)wr[c] * hid[c];
        acc[r] += w * (float)a2;
      }
    }
    // shared expert with sigmoid scalar gate
    std::vector<float> sg(1);
    matvec(ckpt, T(ckpt, "layers.%d.mlp.shared_expert_gate.weight", l, buf, sizeof buf), x,
           sg.data(), 1, g_cfg.d);
    std::vector<float> g1(g_cfg.moe_mid), u1(g_cfg.moe_mid);
    matvec(ckpt, T(ckpt, "layers.%d.mlp.shared_expert.gate_proj.weight", l, buf, sizeof buf),
           x, g1.data(), g_cfg.moe_mid, g_cfg.d);
    matvec(ckpt, T(ckpt, "layers.%d.mlp.shared_expert.up_proj.weight", l, buf, sizeof buf), x,
           u1.data(), g_cfg.moe_mid, g_cfg.d);
    for (int i = 0; i < g_cfg.moe_mid; i++) g1[i] = silu(g1[i]) * u1[i];
    matvec(ckpt, T(ckpt, "layers.%d.mlp.shared_expert.down_proj.weight", l, buf, sizeof buf),
           g1.data(), ey.data(), g_cfg.d, g_cfg.moe_mid);
    float sgv = sigmoid(sg[0]);
    for (int i = 0; i < g_cfg.d; i++) y_out[i] = acc[i] + sgv * ey[i];
  }

  // --- one token ------------------------------------------------------------
  // returns logits (only computed when want_logits)
  static double rms_of(const float* p, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)p[i] * p[i];
    return sqrt(s / n);
  }
  bool dbg = getenv("REF_DEBUG") != nullptr;
  std::vector<float> forward(int token, bool want_logits) {
    char buf[128];
    std::vector<float> x(g_cfg.d), y(g_cfg.d), Rhat;
    // Official: hidden_states = embed_tokens(ids).repeat(1, 1, hc_count) —
    // all 4 residual branches initialized to the token embedding.
    {
      std::vector<float> emb(g_cfg.d);
      const Tensor& et = ckpt.at("embed_tokens.weight");
      auto q = Checkpoint::q4cp_parse(et);
      Checkpoint::q4cp_row(q, token, emb.data());
      for (int b = 0; b < g_cfg.branches; b++)
        memcpy(R.data() + b * g_cfg.d, emb.data(), g_cfg.d * sizeof(float));
      if (dbg) fprintf(stderr, "emb rms=%.4f\n", rms_of(emb.data(), g_cfg.d));
    }

    for (int l = 0; l < g_cfg.layers; l++) {
      // PLE n-gram layer at layer 1, before this layer's attn (official block order)
      if (l == 1 && ple_on) {
        if (dbg)
          fprintf(stderr, "L01 ple-in R[0..3]: %.6f %.6f %.6f %.6f\n", R[0], R[1], R[2],
                  R[3]);
        ple_step(token);
      }
      // token mixer sublayer
      snprintf(buf, sizeof buf, "layers.%d.attn_hyper_connection.hc_norm.weight", l);
      gr_read(buf, l, nullptr, x.data(), Rhat);
      if (is_qsa(l))
        qsa(l, x.data(), y.data());
      else
        gdn(l, x.data(), y.data());
      if (dbg)
        fprintf(stderr, "L%02d attn: x_rms=%.4f y_rms=%.4f\n", l,
                rms_of(x.data(), g_cfg.d), rms_of(y.data(), g_cfg.d));
      snprintf(buf, sizeof buf, "layers.%d.attn_hyper_connection.hc_norm.weight", l);
      gr_write(buf, l, y.data(), Rhat);

      // mlp sublayer
      snprintf(buf, sizeof buf, "layers.%d.mlp_hyper_connection.hc_norm.weight", l);
      gr_read(buf, l, nullptr, x.data(), Rhat);
      moe(l, x.data(), y.data());
      if (dbg)
        fprintf(stderr, "L%02d mlp : x_rms=%.4f y_rms=%.4f\n", l,
                rms_of(x.data(), g_cfg.d), rms_of(y.data(), g_cfg.d));
      gr_write(buf, l, y.data(), Rhat);
    }

    if (!want_logits) { pos++; return {}; }
    // final read via hyper_connection_mixer (no write)
    std::vector<float> h(g_cfg.d);
    gr_read("hyper_connection_mixer.hc_norm.weight", 0, nullptr, h.data(), Rhat);
    const Tensor& lh = ckpt.at("lm_head.weight");
    std::vector<float> logits(lh.dims[0]);
    matvec(ckpt, lh, h.data(), logits.data(), (int)lh.dims[0], g_cfg.d);
    pos++;
    return logits;
  }
};

// ---------------- main ------------------------------------------------------
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: ref <base.hgn> [overlay.hgn] --tokens 1,2,3 [--gen N] [--topk K] "
            "[--theta T]\n");
    return 2;
  }
  std::string base = argv[1];
  std::string overlay;
  std::vector<int> tokens;
  int gen = 8;
  int argi = 2;
  bool dump = false;
  bool ple_on = true;
  int ple_dil = 3, ple_pad = 248044;
  float ple_eps = 1e-6f;
  if (argi < argc && argv[argi][0] != '-') overlay = argv[argi++];
  for (; argi < argc; argi++) {
    std::string a = argv[argi];
    auto next = [&]() { return argv[++argi]; };
    if (a == "--tokens") {
      char* s = strdup(next());
      for (char* p = strtok(s, ","); p; p = strtok(nullptr, ",")) tokens.push_back(atoi(p));
      free(s);
    } else if (a == "--gen")
      gen = atoi(next());
    else if (a == "--topk")
      g_cfg.topk = atoi(next());
    else if (a == "--theta")
      g_cfg.rope_theta = atof(next());
    else if (a == "--dump")
      dump = true;
    else if (a == "--no-ple")
      ple_on = false;
    else if (a == "--ple-dil")
      ple_dil = atoi(next());
    else if (a == "--ple-pad")
      ple_pad = atoi(next());
    else if (a == "--ple-eps")
      ple_eps = atof(next());
  }
  if (tokens.empty()) {
    fprintf(stderr, "need --tokens\n");
    return 2;
  }

  Checkpoint ckpt(base.c_str());
  if (!overlay.empty()) ckpt.add_overlay(overlay.c_str());
  fprintf(stderr, "loaded %zu tensors (base+overlay)\n", ckpt.tensor_count());

  Model m(ckpt);
  m.ple_on = ple_on;
  m.ple.dilation = ple_dil;
  m.ple.pad_id = ple_pad;
  m.ple.eps = ple_eps;
  g_cfg.vocab = (int)ckpt.at("lm_head.weight").dims[0];
  fprintf(stderr, "vocab (lm_head rows) = %d\n", g_cfg.vocab);
  std::vector<int> all = tokens;
  std::vector<float> logits;

  // machine-friendly top-5 dump: "pos<TAB>i<TAB>id,lp;id,lp;...<TAB>next=<lp>"
  auto dump_pos = [&](const std::vector<float>& lg, int i, int next_tok) {
    int V = g_cfg.vocab;
    double mx = -1e300;
    for (int j = 0; j < V; j++) mx = std::max(mx, (double)lg[j]);
    double se = 0;
    for (int j = 0; j < V; j++) se += exp(lg[j] - mx);
    double lse = mx + log(se);
    std::vector<int> idx(V);
    for (int j = 0; j < V; j++) idx[j] = j;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    printf("pos\t%d\t", i);
    for (int k = 0; k < 5; k++)
      printf("%s%d,%.5f", k ? ";" : "", idx[k], lg[idx[k]] - (float)lse);
    if (next_tok >= 0) printf("\tnext=%.5f", lg[next_tok] - (float)lse);
    printf("\n");
    fflush(stdout);
  };

  for (size_t i = 0; i < tokens.size(); i++) {
    logits = m.forward(tokens[i], true);
    if (dump)
      dump_pos(logits, (int)i, i + 1 < tokens.size() ? tokens[i + 1] : -1);
    fprintf(stderr, "\rprefill %zu/%zu", i + 1, tokens.size());
    fflush(stderr);
  }
  fprintf(stderr, "\n");

  auto show = [&](const std::vector<float>& lg, int step, int chosen) {
    // log-softmax top-5 over logical vocab
    int V = g_cfg.vocab;
    double mx = -1e300;
    for (int i = 0; i < V; i++) mx = std::max(mx, (double)lg[i]);
    double se = 0;
    for (int i = 0; i < V; i++) se += exp(lg[i] - mx);
    double lse = mx + log(se);
    std::vector<int> idx(V);
    for (int i = 0; i < V; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    printf("step %d chosen=%d top5:", step, chosen);
    for (int i = 0; i < 5; i++) printf(" [%d]=%.4f", idx[i], lg[idx[i]] - (float)lse);
    printf("\n");
    fflush(stdout);
  };

  int cur = tokens.back();
  for (int g = 0; g < gen; g++) {
    int nxt = 0;
    int V = g_cfg.vocab;
    float best = -1e30f;
    for (int i = 0; i < V; i++)
      if (logits[i] > best) { best = logits[i]; nxt = i; }
    if (dump)
      dump_pos(logits, (int)tokens.size() + g, -1);
    else
      show(logits, g, nxt);
    cur = nxt;
    all.push_back(cur);
    logits = m.forward(cur, true);
  }
  printf("ids:");
  for (int t : all) printf(" %d", t);
  printf("\n");
  return 0;
}
