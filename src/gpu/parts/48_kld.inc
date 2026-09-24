// ---------------- KL divergence vs a llama.cpp logits file ------------------
// Port of llama.cpp tools/perplexity (--kl-divergence-base writes the file,
// --kl-divergence reads it), so engine numbers are directly comparable with
// llama-perplexity runs of GGUF quants against the same BF16 base file.
// File: "_logits_", u32 n_ctx, i32 n_vocab, i32 n_chunk, n_ctx*n_chunk i32
// tokens, then per chunk (n_ctx-1-n_ctx/2) records of nv = 2*((n_vocab+1)/2)+4
// uint16: float scale, float min_log_prob, n_vocab log-probs quantized over
// [max-16, max]. Every chunk starts from an empty state; positions
// n_ctx/2 .. n_ctx-2 are scored against token[pos+1]. The model's GGUF has
// add_bos_token=false, so tokens are fed verbatim (no BOS substitution).
//   gdec BASE OVL --kld-base REF [--kld-chunks N] [--kld-save OUT]
//   gdec BASE OVL --tokens-file IDS --kld-save OUT [--kld-ctx 512] [--kld-chunks N]
namespace kld {

static inline int nearest_int(float fval) {
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

// llama.cpp log_softmax(n_vocab, logits, uint16_t*, tok): one base record.
static double save_row(int n_vocab, const float* logits, uint16_t* log_prob, int tok) {
  float max_logit = logits[0], min_logit = logits[0];
  for (int i = 1; i < n_vocab; ++i) {
    max_logit = std::max(max_logit, logits[i]);
    min_logit = std::min(min_logit, logits[i]);
  }
  min_logit = std::max(min_logit, max_logit - 16);
  double sum_exp = 0.0;
  for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
  const float log_sum_exp = log(sum_exp);
  const float min_log_prob = min_logit - max_logit - log_sum_exp;
  const float scale = (max_logit - min_logit) / 65535.f;
  float* d = (float*)log_prob;
  d[0] = scale;
  d[1] = min_log_prob;
  log_prob += 4;
  if (scale) {
    const float inv_scale = 1 / scale;
    for (int i = 0; i < n_vocab; ++i)
      log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale * (logits[i] - min_logit)) : 0;
  } else {
    memset(log_prob, 0, n_vocab * sizeof(uint16_t));
  }
  return max_logit + log_sum_exp - logits[tok];
}

struct Row {
  float nll, nll_base, p_diff;
  double kld;
  bool same_top;
};

// llama.cpp log_softmax(n_vocab, logits, base_log_prob, tok, kld) per row.
static Row compare_row(int n_vocab, const float* logits, const uint16_t* base_log_prob, int tok) {
  float max_logit = logits[0];
  int imax = 0;
  for (int i = 1; i < n_vocab; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
      imax = i;
    }
  }
  double sum_exp = 0.0;
  for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
  const float log_sum_exp = log(sum_exp);
  const float* d = (const float*)base_log_prob;
  const float scale = d[0];
  const float min_log_prob = d[1];
  base_log_prob += 4;
  Row r;
  r.nll = max_logit + log_sum_exp - logits[tok];
  r.nll_base = -(scale * base_log_prob[tok] + min_log_prob);
  max_logit += log_sum_exp;
  double sum = 0;
  int imax_base = -1;
  float p_log_base_max = 0;
  for (int i = 0; i < n_vocab; ++i) {
    const float p_log_base = scale * base_log_prob[i] + min_log_prob;
    if (i == 0 || p_log_base > p_log_base_max) {
      p_log_base_max = p_log_base;
      imax_base = i;
    }
    if (p_log_base > -16.f) {
      const float p_base = expf(p_log_base);
      sum += p_base * (p_log_base - logits[i] + max_logit);
    }
  }
  r.kld = sum;
  r.same_top = imax == imax_base;
  r.p_diff = expf(-r.nll) - expf(-r.nll_base);
  return r;
}

// Runs fn(i) for i in [0, n) on all hardware threads.
static void parallel_rows(int n, const std::function<void(int)>& fn) {
  std::atomic<int> next{0};
  auto work = [&]() {
    for (int i; (i = next.fetch_add(1)) < n;) fn(i);
  };
  unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned t = 1; t < nt; t++) pool.emplace_back(work);
  work();
  for (auto& t : pool) t.join();
}

static std::pair<double, double> mean_unc(double sum, double sum2, size_t count) {
  if (count < 1) return {0., 0.};
  double f = sum / count;
  double df = sum2 / count - f * f;
  df = df > 0 && count > 10 ? sqrt(df / (count - 1)) : 0.;
  return {f, df};
}

static double covariance(double suma, double sumb, double sumab, size_t count) {
  if (count < 10) return 0.0;
  return (sumab / count - (suma / count) * (sumb / count)) / (count - 1);
}

static float percentile(const std::vector<float>& v, float fraction) {
  if (fraction <= 0) return v.front();
  if (fraction >= 1) return v.back();
  float p = fraction * (v.size() - 1);
  size_t ip = size_t(p);
  p -= ip;
  return (1 - p) * v[ip] + p * v[std::min(ip + 1, v.size() - 1)];
}

struct Opts {
  std::string base, save;
  int ctx = 512, chunks = -1;
};

static int run(GpuModel& m, std::vector<int> tokens, const Opts& o) {
  const int n_vocab = g_cfg.vocab;
  const int nv = 2 * ((n_vocab + 1) / 2) + 4;
  std::ifstream in;
  uint32_t n_ctx = (uint32_t)o.ctx;
  int n_chunk = 0;
  if (!o.base.empty()) {
    in.open(o.base, std::ios::binary);
    char check[9] = {};
    int fv = 0;
    if (!in || !in.read(check, 8) || strncmp(check, "_logits_", 8) ||
        !in.read((char*)&n_ctx, 4) || !in.read((char*)&fv, 4) ||
        !in.read((char*)&n_chunk, 4)) {
      fprintf(stderr, "kld: %s is not a llama.cpp logits file\n", o.base.c_str());
      return 2;
    }
    if (fv != n_vocab) {
      fprintf(stderr, "kld: base n_vocab %d != model %d\n", fv, n_vocab);
      return 2;
    }
    tokens.assign((size_t)n_ctx * n_chunk, 0);
    if (!in.read((char*)tokens.data(), tokens.size() * 4)) {
      fprintf(stderr, "kld: truncated token block in %s\n", o.base.c_str());
      return 2;
    }
  } else {
    n_chunk = (int)(tokens.size() / n_ctx);
  }
  if (o.chunks >= 0) n_chunk = std::min(n_chunk, o.chunks);
  if (n_ctx < 16 || (int)n_ctx > m.maxbatch || n_chunk < 1) {
    fprintf(stderr, "kld: need n_ctx in [16, maxbatch=%d] (got %u) and >= 1 chunk\n",
            m.maxbatch, n_ctx);
    return 2;
  }
  for (size_t i = 0; i < (size_t)n_ctx * n_chunk; i++) {
    if (tokens[i] < 0 || tokens[i] >= n_vocab) {
      fprintf(stderr, "kld: token %zu out of vocabulary: %d\n", i, tokens[i]);
      return 2;
    }
  }
  std::ofstream out;
  if (!o.save.empty()) {
    out.open(o.save, std::ios::binary);
    out.write("_logits_", 8);
    out.write((const char*)&n_ctx, 4);
    out.write((const char*)&n_vocab, 4);
    out.write((const char*)&n_chunk, 4);
    out.write((const char*)tokens.data(), (size_t)n_ctx * n_chunk * 4);
  }
  const int first = n_ctx / 2, nrow = n_ctx - 1 - first;
  const bool cmp = in.is_open();
  fprintf(stderr, "kld: %d chunks x %u ctx, %d scored/chunk, base=%s save=%s\n", n_chunk,
          n_ctx, nrow, cmp ? o.base.c_str() : "-", o.save.empty() ? "-" : o.save.c_str());
  std::vector<float> logits;
  std::vector<uint16_t> base((size_t)nrow * nv), saved;
  if (out.is_open()) saved.resize((size_t)nrow * nv);
  std::vector<Row> rows(nrow);
  std::vector<double> save_nll(nrow);
  std::vector<float> kld_values, p_diff_values;
  double s_nll = 0, s_nll2 = 0, s_b = 0, s_b2 = 0, s_nb = 0, s_k = 0, s_k2 = 0;
  double s_pd = 0, s_pd2 = 0, s_pd4 = 0;
  size_t same = 0, count = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int c = 0; c < n_chunk; c++) {
    m.reset_state();
    const int* tk = tokens.data() + (size_t)c * n_ctx;
    m.prefill_chunk(std::vector<int>(tk, tk + n_ctx), 0);
    m.batch_logits_rows((int)n_ctx, first, nrow, logits);
    const int* tgt = tk + first + 1;
    if (out.is_open()) {
      parallel_rows(nrow, [&](int i) {
        save_nll[i] = save_row(n_vocab, logits.data() + (size_t)i * n_vocab,
                               saved.data() + (size_t)i * nv, tgt[i]);
      });
      out.write((const char*)saved.data(), saved.size() * 2);
      if (!cmp)
        for (int i = 0; i < nrow; i++) {
          s_nll += save_nll[i];
          count++;
        }
    }
    if (cmp) {
      if (!in.read((char*)base.data(), base.size() * 2)) {
        fprintf(stderr, "kld: base file truncated at chunk %d\n", c);
        return 1;
      }
      parallel_rows(nrow, [&](int i) {
        rows[i] = compare_row(n_vocab, logits.data() + (size_t)i * n_vocab,
                              base.data() + (size_t)i * nv, tgt[i]);
      });
      for (const Row& r : rows) {
        if (!std::isfinite(r.kld) || !std::isfinite(r.nll)) {
          fprintf(stderr, "kld: non-finite value in chunk %d\n", c);
          return 1;
        }
        s_nll += r.nll;
        s_nll2 += r.nll * r.nll;
        s_b += r.nll_base;
        s_b2 += r.nll_base * r.nll_base;
        s_nb += r.nll * r.nll_base;
        s_k += r.kld;
        s_k2 += r.kld * r.kld;
        s_pd += r.p_diff;
        const double pd2 = r.p_diff * r.p_diff;  // float product, as llama.cpp
        s_pd2 += pd2;
        s_pd4 += pd2 * pd2;
        same += r.same_top;
        count++;
        kld_values.push_back((float)r.kld);
        p_diff_values.push_back(r.p_diff);
      }
    }
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (cmp)
      fprintf(stderr, "kld chunk %4d/%d  ppl=%.4f  base=%.4f  kld=%.5f  same_top=%.3f%%  %.1fs\n",
              c + 1, n_chunk, exp(s_nll / count), exp(s_b / count), s_k / count,
              100.0 * same / count, el);
    else
      fprintf(stderr, "kld chunk %4d/%d  ppl=%.4f  %.1fs\n", c + 1, n_chunk,
              exp(s_nll / count), el);
  }
  if (out.is_open()) {
    out.flush();
    if (!out) {
      fprintf(stderr, "kld: write to %s failed\n", o.save.c_str());
      return 1;
    }
  }
  if (!cmp) {
    printf("kld_saved\tchunks=%d\tn_ctx=%u\tcount=%zu\tppl=%.6f\n", n_chunk, n_ctx, count,
           exp(s_nll / count));
    return 0;
  }
  std::sort(kld_values.begin(), kld_values.end());
  std::sort(p_diff_values.begin(), p_diff_values.end());
  auto lp = mean_unc(s_nll, s_nll2, count);
  auto lb = mean_unc(s_b, s_b2, count);
  double cov = covariance(s_nll, s_b, s_nb, count);
  double lr_unc = sqrt(lp.second * lp.second + lb.second * lb.second - 2.0 * cov);
  auto kl = mean_unc(s_k, s_k2, count);
  auto pdm = mean_unc(s_pd2, s_pd4, count);
  double rms = sqrt(pdm.first), rms_unc = 0.5 / rms * pdm.second;
  double top = 1.0 * same / count, top_unc = sqrt(top * (1 - top) / (count - 1));
  size_t h = kld_values.size() / 2;
  float med = kld_values.size() % 2 ? kld_values[h] : 0.5f * (kld_values[h] + kld_values[h - 1]);
  printf("====== Perplexity statistics ======\n");
  printf("Mean PPL(Q)                   : %10.6f ± %10.6f\n", exp(lp.first), exp(lp.first) * lp.second);
  printf("Mean PPL(base)                : %10.6f ± %10.6f\n", exp(lb.first), exp(lb.first) * lb.second);
  printf("Cor(ln(PPL(Q)), ln(PPL(base))): %6.2f%%\n",
         100.0 * cov / (lp.second * lb.second));
  printf("Mean ln(PPL(Q)/PPL(base))     : %10.6f ± %10.6f\n", lp.first - lb.first, lr_unc);
  printf("====== KL divergence statistics ======\n");
  printf("Mean    KLD: %10.6f ± %10.6f\n", kl.first, kl.second);
  printf("Maximum KLD: %10.6f\n", kld_values.back());
  printf("99.9%%   KLD: %10.6f\n", percentile(kld_values, 0.999f));
  printf("99.0%%   KLD: %10.6f\n", percentile(kld_values, 0.990f));
  printf("95.0%%   KLD: %10.6f\n", percentile(kld_values, 0.950f));
  printf("90.0%%   KLD: %10.6f\n", percentile(kld_values, 0.900f));
  printf("Median  KLD: %10.6f\n", med);
  printf("Minimum KLD: %10.6f\n", kld_values.front());
  printf("====== Token probability statistics ======\n");
  printf("Mean    Δp: %6.3f ± %5.3f %%\n", 100.0 * s_pd / count,
         100.0 * mean_unc(s_pd, s_pd2, count).second);
  printf("Maximum Δp: %6.3f%%\n", 100.0 * p_diff_values.back());
  printf("99.9%%   Δp: %6.3f%%\n", 100.0 * percentile(p_diff_values, 0.999f));
  printf("99.0%%   Δp: %6.3f%%\n", 100.0 * percentile(p_diff_values, 0.990f));
  printf("Median  Δp: %6.3f%%\n", 100.0 * percentile(p_diff_values, 0.5f));
  printf("1.0%%    Δp: %6.3f%%\n", 100.0 * percentile(p_diff_values, 0.010f));
  printf("0.1%%    Δp: %6.3f%%\n", 100.0 * percentile(p_diff_values, 0.001f));
  printf("Minimum Δp: %6.3f%%\n", 100.0 * p_diff_values.front());
  printf("RMS Δp    : %6.3f ± %5.3f %%\n", 100.0 * rms, 100.0 * rms_unc);
  printf("Same top p: %6.3f ± %5.3f %%\n", 100.0 * top, 100.0 * top_unc);
  printf("kld_summary\tcount=%zu\tmean_kld=%.6f\tp999_kld=%.6f\tsame_top=%.3f\tppl_q=%.6f\tppl_base=%.6f\n",
         count, kl.first, percentile(kld_values, 0.999f), 100.0 * top, exp(lp.first),
         exp(lb.first));
  return 0;
}

}  // namespace kld
