#define QWENOX_NO_MAIN
#include "qwenox.cpp"

#include "../api/tokenizer.h"
#include "../api/tokenizer.cpp"

#include <cctype>
#include <cerrno>
#include <climits>
#include <new>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchConfig {
  std::filesystem::path root;
  std::unordered_map<std::string, std::string> values;
};

static std::string trim_copy(std::string text) {
  const char* ws = " \t\r\n";
  const size_t first = text.find_first_not_of(ws);
  if (first == std::string::npos) return {};
  const size_t last = text.find_last_not_of(ws);
  return text.substr(first, last - first + 1);
}

static std::string strip_comment(std::string text) {
  bool quoted = false;
  char quote = 0;
  for (size_t index = 0; index < text.size(); index++) {
    if (text[index] == '\'' || text[index] == '"') {
      if (!quoted) {
        quoted = true;
        quote = text[index];
      } else if (quote == text[index]) {
        quoted = false;
      }
    } else if (text[index] == '#' && !quoted &&
               (index == 0 || text[index - 1] != '\\')) {
      text.resize(index);
      break;
    }
  }
  return trim_copy(std::move(text));
}

static bool has_env_value(const std::string& name) {
  return std::getenv(name.c_str()) != nullptr;
}

static std::string env_value(const std::string& name) {
  const char* value = std::getenv(name.c_str());
  return value ? value : std::string();
}

static std::string unquote(std::string value) {
  value = trim_copy(std::move(value));
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

static std::string expand_shell_value(
    const std::string& input,
    const std::unordered_map<std::string, std::string>& values) {
  std::string out = input;
  for (int pass = 0; pass < 16; pass++) {
    std::string next;
    bool changed = false;
    for (size_t index = 0; index < out.size();) {
      if (out[index] == '$' && index + 1 < out.size() && out[index + 1] == '{') {
        const size_t closing = out.find('}', index + 2);
        if (closing == std::string::npos) {
          next.append(out, index, std::string::npos);
          break;
        }
        const std::string body = out.substr(index + 2, closing - index - 2);
        const size_t colon_dash = body.find(":-");
        const size_t dash = body.find('-');
        const bool colon_default = colon_dash != std::string::npos;
        const size_t default_pos = colon_default ? colon_dash : dash;
        const std::string name = default_pos == std::string::npos
                                     ? body
                                     : body.substr(0, default_pos);
        std::string replacement;
        const bool env_present = has_env_value(name);
        if (env_present) replacement = env_value(name);
        if (!env_present) {
          auto it = values.find(name);
          if (it != values.end()) replacement = it->second;
        }
        if (((colon_default && replacement.empty()) ||
             (!colon_default && !env_present && replacement.empty())) &&
            default_pos != std::string::npos) {
          replacement = body.substr(default_pos + (colon_default ? 2 : 1));
        }
        next += replacement;
        index = closing + 1;
        changed = true;
        continue;
      }
      if (out[index] == '$' && index + 1 < out.size() &&
          (std::isalpha((unsigned char)out[index + 1]) || out[index + 1] == '_')) {
        size_t end = index + 2;
        while (end < out.size() &&
               (std::isalnum((unsigned char)out[end]) || out[end] == '_'))
          end++;
        const std::string name = out.substr(index + 1, end - index - 1);
        std::string replacement = env_value(name);
        if (!has_env_value(name)) {
          auto it = values.find(name);
          if (it != values.end()) replacement = it->second;
        }
        next += replacement;
        index = end;
        changed = true;
        continue;
      }
      next.push_back(out[index++]);
    }
    out = std::move(next);
    if (!changed) break;
  }
  return out;
}

static BenchConfig read_service_config(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open service.conf: " + path.string());
  BenchConfig config;
  config.root = path.parent_path();
  std::string line;
  while (std::getline(file, line)) {
    line = strip_comment(std::move(line));
    if (line.empty()) continue;
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = trim_copy(line.substr(0, eq));
    if (key.empty() || key.find_first_not_of(
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_") !=
                           std::string::npos)
      continue;
    config.values[key] = expand_shell_value(unquote(line.substr(eq + 1)), config.values);
  }
  return config;
}

static std::filesystem::path find_root(const char* argv0) {
  std::vector<std::filesystem::path> starts;
  starts.push_back(std::filesystem::current_path());
  std::error_code ec;
  const auto exe = std::filesystem::absolute(argv0, ec);
  if (!ec) starts.push_back(exe.parent_path());
  for (const auto& start : starts) {
    auto current = start;
    for (int depth = 0; depth < 8 && !current.empty(); depth++) {
      if (std::filesystem::is_regular_file(current / "service.conf")) return current;
      const auto parent = current.parent_path();
      if (parent == current) break;
      current = parent;
    }
  }
  throw std::runtime_error("cannot locate service.conf from the current directory or executable");
}

static std::string config_value(const BenchConfig& config, const std::string& key,
                                const std::string& fallback = {}) {
  if (has_env_value(key)) return env_value(key);
  auto it = config.values.find(key);
  return it == config.values.end() ? fallback : it->second;
}

static int config_int(const BenchConfig& config, const std::string& key, int fallback) {
  const std::string value = config_value(config, key);
  if (value.empty()) return fallback;
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (errno || end == value.c_str() || *end != 0 || parsed < 0 || parsed > INT_MAX)
    throw std::runtime_error("invalid integer in service.conf: " + key + "=" + value);
  return (int)parsed;
}

static void set_env_value(const char* key, const std::string& value) {
#ifdef _WIN32
  if (_putenv_s(key, value.c_str()) != 0)
    throw std::runtime_error(std::string("cannot set environment variable ") + key);
#else
  if (setenv(key, value.c_str(), 1) != 0)
    throw std::runtime_error(std::string("cannot set environment variable ") + key + ": " +
                             std::strerror(errno));
#endif
}

static void unset_env_value(const char* key) {
#ifdef _WIN32
  if (_putenv_s(key, "") != 0)
    throw std::runtime_error(std::string("cannot clear environment variable ") + key);
#else
  unsetenv(key);
#endif
}

static std::filesystem::path rooted_path(const BenchConfig& config, const std::string& value) {
  if (value.empty()) return {};
  const std::filesystem::path path(value);
  return path.is_absolute() ? path : config.root / path;
}

static void require_file(const std::filesystem::path& path, const char* role) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec))
    throw std::runtime_error(std::string("missing ") + role + ": " + path.string());
}

static std::vector<int> read_token_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot read token file: " + path.string());
  std::vector<int> tokens;
  std::string line;
  while (std::getline(file, line)) {
    char* cursor = line.data();
    while (*cursor) {
      while (*cursor && (std::isspace((unsigned char)*cursor) || *cursor == ',')) cursor++;
      if (!*cursor) break;
      char* end = nullptr;
      errno = 0;
      const long id = std::strtol(cursor, &end, 10);
      if (errno || end == cursor || id < INT_MIN || id > INT_MAX)
        throw std::runtime_error("invalid token in " + path.string());
      tokens.push_back((int)id);
      cursor = end;
    }
  }
  if (tokens.empty()) throw std::runtime_error("empty token file: " + path.string());
  return tokens;
}

static std::vector<std::filesystem::path> token_files(const BenchConfig& config) {
  const auto dir = config.root / "data" / "qsa-oracle";
  std::vector<std::filesystem::path> files;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) throw std::runtime_error("cannot list token directory: " + dir.string());
    if (!entry.is_regular_file() || entry.path().extension() != ".tokens") continue;
    files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
    const auto parse = [](const std::filesystem::path& file_path) {
      try { return std::stoll(file_path.stem().string()); } catch (...) { return LLONG_MAX; }
    };
    const auto an = parse(a), bn = parse(b);
    return an == bn ? a.filename().string() < b.filename().string() : an < bn;
  });
  if (files.empty()) throw std::runtime_error("no .tokens files found in " + dir.string());
  return files;
}

static std::string choose_format(const BenchConfig& config) {
  std::string format = env_value("QWENOX_BENCH_FORMAT");
  if (format.empty()) format = config_value(config, "BENCH_FORMAT");
  if (!format.empty()) {
    for (char& character : format)
      character = (char)std::tolower((unsigned char)character);
    if (format == "hgn" || format == "gguf") return format;
    throw std::runtime_error("QWENOX_BENCH_FORMAT must be hgn or gguf");
  }
  const auto hgn = rooted_path(config, config_value(config, "MODEL_FILE"));
  const auto gguf = rooted_path(config, config_value(config, "GGUF_FILE"));
  std::error_code ec;
  if (std::filesystem::is_regular_file(hgn, ec)) return "hgn";
  if (std::filesystem::is_regular_file(gguf, ec)) return "gguf";
  throw std::runtime_error("neither configured MODEL_FILE nor GGUF_FILE exists");
}

static void configure_runtime(const BenchConfig& config) {
  const auto set_if_configured = [&](const char* env_key, const char* config_key) {
    const std::string value = config_value(config, config_key);
    if (!value.empty()) set_env_value(env_key, value);
  };
  set_env_value("QWENOX_QSA_KV_BF16", "1");
  set_env_value("QWENOX_QSA_WMMA", "1");
  set_env_value("QWENOX_QSA_WMMA_BTV", "1");
  set_env_value("QWENOX_GEMM_WMMA", "1");
  set_env_value("QWENOX_GDN_FUSED", "1");
  set_env_value("QWENOX_MOE_LT", "1");
  set_env_value("QWENOX_MOE_LT_BF16", "1");
  set_env_value("QWENOX_GR_BF16", "1");
  set_env_value("QWENOX_GDN_STREAM", "1");
  set_env_value("QWENOX_GDN_WAVE", "1");
  set_env_value("QWENOX_INDEX_FUSED2", "1");
  set_env_value("QWENOX_PP_MOE_OUT", "1");
  set_env_value("QWENOX_INDEX_STREAM_SELECT", "1");
  const int prefill_chunk = config_int(config, "PREFILL_CHUNK", 0);
  if (prefill_chunk > 0) {
    set_env_value("QWENOX_PREFILL_CHUNK", std::to_string(prefill_chunk));
  } else {
#ifdef _WIN32
    unset_env_value("QWENOX_PREFILL_CHUNK");
#else
    set_env_value("QWENOX_PREFILL_CHUNK", "16384");
#endif
  }
  set_env_value("QWENOX_KVSNAP", "0");
  set_env_value("QWENOX_NOWARMUP", "1");
  unset_env_value("QWENOX_CONC_PREFILL_CHUNK");
  set_env_value("QWENOX_PARALLEL", "1");
  if (config_int(config, "KV_PAGED", 1)) {
    set_env_value("QWENOX_KV_PAGED", "1");
    const int pool_tokens = config_int(config, "KV_POOL_TOKENS", 0);
    if (pool_tokens > 0) set_env_value("QWENOX_KV_POOL_TOKENS", std::to_string(pool_tokens));
    else unset_env_value("QWENOX_KV_POOL_TOKENS");
  } else {
    unset_env_value("QWENOX_KV_PAGED");
    unset_env_value("QWENOX_KV_POOL_TOKENS");
  }
  if (config_int(config, "PLE_URING", 1)) set_env_value("QWENOX_PLE_URING", "1");
  else unset_env_value("QWENOX_PLE_URING");
  set_if_configured("QWENOX_ROPE_FACTOR", "ROPE_FACTOR");
  set_if_configured("QWENOX_ROPE_ORIGINAL_CTX", "ROPE_ORIGINAL_CTX");
  set_if_configured("QWENOX_ROPE_BETA_FAST", "ROPE_BETA_FAST");
  set_if_configured("QWENOX_ROPE_BETA_SLOW", "ROPE_BETA_SLOW");
  set_if_configured("QWENOX_ROPE_ATTN_SCALE", "ROPE_ATTN_SCALE");
  const std::string gamma = config_value(config, "MTP_GAMMA");
  if (!gamma.empty() && gamma != "0") set_env_value("QWENOX_SPEC_GAMMA", gamma);
  else unset_env_value("QWENOX_SPEC_GAMMA");
}

struct ModelPaths {
  std::string format;
  std::filesystem::path base;
  std::vector<std::filesystem::path> overlays;
  std::filesystem::path mtp;
};

static ModelPaths configured_model(const BenchConfig& config) {
  ModelPaths result;
  result.format = choose_format(config);
  if (result.format == "hgn") {
    result.base = rooted_path(config, config_value(config, "MODEL_FILE"));
    require_file(result.base, "HGN model");
    const auto overlay = rooted_path(config, config_value(config, "OVERLAY_FILE"));
    if (!overlay.empty()) {  // optional: warn and skip when missing (e.g. v2 with a stale config)
      std::error_code ec;
      if (std::filesystem::is_regular_file(overlay, ec))
        result.overlays.push_back(overlay);
      else
        fprintf(stderr, "Warning: HGN overlay not found, skipped: %s (v2 weights need no overlay; "
                        "set OVERLAY_FILE=\"\" to silence)\n", overlay.string().c_str());
    }
    // PLE n-gram table: same file as MODEL_FILE for w4b, separate *-ngram.hgn for v2.
    const auto ngram = rooted_path(config, config_value(config, "NGRAM_FILE"));
    if (!ngram.empty()) {
      require_file(ngram, "HGN n-gram table");
      std::error_code ec;
      if (!std::filesystem::equivalent(ngram, result.base, ec)) result.overlays.push_back(ngram);
    }
    result.mtp = rooted_path(config, config_value(config, "MTP_FILE"));
    if (!result.mtp.empty()) {
      require_file(result.mtp, "HGN MTP weights");
      result.overlays.push_back(result.mtp);
    }
  } else {
    result.base = rooted_path(config, config_value(config, "GGUF_FILE"));
    require_file(result.base, "GGUF model shard");
    result.mtp = rooted_path(config, config_value(config, "GGUF_MTP_FILE"));
    if (!result.mtp.empty()) require_file(result.mtp, "GGUF MTP weights");
  }
  return result;
}

static void configure_gguf_mtp(const ModelPaths& model) {
  if (model.format == "gguf") {
    gguf_base_setup(model.base.string());
    set_env_value("QWENOX_GGUF_MTP", model.mtp.string());
  }
}

struct LoadedModel {
  std::unique_ptr<Checkpoint> checkpoint;
  std::unique_ptr<GpuModel> model;
};

static LoadedModel load_model(const ModelPaths& paths, int maxctx) {
  LoadedModel loaded;
  loaded.checkpoint = open_base(paths.base.string());
  Checkpoint& checkpoint = *loaded.checkpoint;
  lmhead_base_capture(checkpoint);
  for (const auto& overlay : paths.overlays)
    if (const char* why = checkpoint.add_overlay(overlay.string().c_str()))
      fprintf(stderr, "overlay %s: %s, skipped\n", overlay.string().c_str(), why);
  fprintf(stdout, "Loading %s weights...\n", paths.format == "hgn" ? "HGN" : "GGUF");
  fflush(stdout);
  gguf_dense_apply(checkpoint);
  g_cfg.vocab = (int)checkpoint.at("lm_head.weight").dims[0];
  memstats_capture_hip_baseline();
  devarena_init(checkpoint, maxctx);
  load_arena(checkpoint);
  v2_init(checkpoint);  // hgn v2 sidecars / codebook (no-op for v1 and GGUF)
  loaded.model = std::make_unique<GpuModel>(checkpoint, maxctx);
  loaded.model->setup_draft_lmhead(g_lmhead_base_ok ? &g_lmhead_base : nullptr);
  gguf_dense_release(checkpoint);
  loaded.model->ple_on = true;
  loaded.model->build_graphs();
  return loaded;
}

static double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct PrefillRow {
  std::string name;
  size_t tokens = 0;
  double rate = 0;
};

struct DecodeRow {
  std::string name;
  size_t prompt_tokens = 0;
  double rate = 0;
  double acceptance = 0;
};

struct BenchSummary {
  std::string format;
  std::string model_path;
  int maxctx = 0;
  int prefill_repeats = 0;
  std::vector<PrefillRow> prefill_rows;
  double prefill_avg = 0;
  bool decode_ran = false;
  int decode_gamma = 0;
  int decode_generated = 0;
  int decode_repeats = 0;
  std::vector<DecodeRow> decode_rows;
  double decode_avg = 0;
  double decode_accept_avg = 0;
};

static void print_summary(const BenchSummary& summary) {
  printf("\n================ Benchmark Summary ================\n");
  printf("Model       : %s (%s)\n", summary.model_path.c_str(), summary.format.c_str());
  printf("Max context : %d tokens\n", summary.maxctx);
  printf("\nPrefill (%d sample%s per file)\n", summary.prefill_repeats,
         summary.prefill_repeats == 1 ? "" : "s");
  printf("  %-18s %10s %12s\n", "token file", "tokens", "tok/s");
  for (const auto& row : summary.prefill_rows)
    printf("  %-18s %10zu %12.1f\n", row.name.c_str(), row.tokens, row.rate);
  printf("  %-18s %10s %12.1f  (weighted average over %zu files)\n", "TOTAL", "-",
         summary.prefill_avg, summary.prefill_rows.size());
  if (summary.decode_ran) {
    printf("\nDecode / TG (greedy, gamma=%d, %d generated tokens, %d sample%s per scenario)\n",
           summary.decode_gamma, summary.decode_generated, summary.decode_repeats,
           summary.decode_repeats == 1 ? "" : "s");
    printf("  %-18s %10s %12s %12s\n", "scenario", "prompt", "tok/s", "MTP accept");
    for (const auto& row : summary.decode_rows)
      printf("  %-18s %10zu %12.1f %11.1f%%\n", row.name.c_str(), row.prompt_tokens,
             row.rate, row.acceptance * 100.0);
    printf("  %-18s %10s %12.1f %11.1f%%  (weighted average)\n", "TOTAL", "-",
           summary.decode_avg, summary.decode_accept_avg * 100.0);
  } else {
    printf("\nDecode / TG : skipped (no usable MTP weights or tokenizer)\n");
  }
  printf("===================================================\n");
}

static void run_prefill(GpuModel& model, const BenchConfig& config, int maxctx,
                        BenchSummary& summary) {
  const int repeats = config_int(config, "BENCH_PREFILL_REPEATS", 2);
  if (repeats < 1) throw std::runtime_error("BENCH_PREFILL_REPEATS must be positive");
  const auto files = token_files(config);
  summary.prefill_repeats = repeats;
  double total_tokens = 0, total_seconds = 0;
  for (const auto& path : files) {
    const std::vector<int> tokens = read_token_file(path);
    if ((int)tokens.size() > maxctx)
      throw std::runtime_error("token file exceeds MAX_CONTEXT: " + path.string());
    for (size_t index = 0; index < tokens.size(); index++) {
      if (tokens[index] < 0 || tokens[index] >= g_cfg.vocab)
        throw std::runtime_error("token " + std::to_string(tokens[index]) +
                                 " at position " + std::to_string(index) +
                                 " is outside the model vocabulary");
    }
    double elapsed = 0;
    for (int sample = 0; sample < repeats; sample++) {
      model.mtp_tap_capture = false;
      model.reset_state();
      const auto start = Clock::now();
      model.prefill_batch(tokens, 0, true, /*quiet=*/true);
      CK(hipDeviceSynchronize());
      elapsed += seconds_since(start);
    }
    const double rate = tokens.size() * repeats / elapsed;
    printf("  %-18s %7zu tokens  %8.1f tok/s\n", path.stem().string().c_str(),
           tokens.size(), rate);
    summary.prefill_rows.push_back({path.stem().string(), tokens.size(), rate});
    total_tokens += (double)tokens.size() * repeats;
    total_seconds += elapsed;
  }
  summary.prefill_avg = total_tokens / total_seconds;
  printf("Prefill average: %.1f tok/s (weighted over %zu files, %d samples each)\n",
         summary.prefill_avg, files.size(), repeats);
}

struct DecodeCase {
  const char* name;
  const char* prompt;
};

static std::vector<int> encode_case(const qwenox::Tokenizer& tokenizer, const char* prompt) {
  const std::string formatted = std::string("<|im_start|>user\n") + prompt +
                                "<|im_end|>\n<|im_start|>assistant\n";
  return tokenizer.encode(formatted);
}

static int run_decode(GpuModel& model, const BenchConfig& config, int maxctx,
                      BenchSummary& summary) {
  if (!model.mtp_avail) {
    fprintf(stderr, "MTP decode benchmark skipped: no usable MTP weights are loaded.\n");
    return 1;
  }
  const std::filesystem::path tokenizer_dir =
      rooted_path(config, config_value(config, "TOKENIZER_DIR"));
  qwenox::Tokenizer tokenizer;
  std::string tokenizer_error;
  if (tokenizer_dir.empty() || !tokenizer.load(tokenizer_dir.string(), &tokenizer_error)) {
    fprintf(stderr, "MTP decode benchmark skipped: cannot load tokenizer from %s: %s\n",
            tokenizer_dir.string().c_str(), tokenizer_error.c_str());
    return 1;
  }
  const DecodeCase cases[] = {
      {"python-code", "Write a correct Python function that parses a CSV file, validates "
                       "an integer column, and returns the rows sorted by that column. "
                       "Explain the time complexity briefly."},
      {"creative-writing", "Write a short original science-fiction scene about a mechanic "
                          "repairing a weather satellite during an unexpected solar storm. "
                          "Use vivid but concise prose."},
      {"common-sense-qa", "Answer this common-sense question in two or three sentences: "
                         "Why should a person dry a wet floor before walking across it?"},
  };
  const int repeats = config_int(config, "BENCH_DECODE_REPEATS", 2);
  const int generated = config_int(config, "BENCH_DECODE_TOKENS", 128);
  if (repeats < 1 || generated < 1)
    throw std::runtime_error("BENCH_DECODE_REPEATS and BENCH_DECODE_TOKENS must be positive");
  int gamma = config_int(config, "MTP_GAMMA", 4);
  if (gamma == 0) gamma = 4;
  gamma = std::max(1, std::min(8, gamma));
  summary.decode_ran = true;
  summary.decode_gamma = gamma;
  summary.decode_generated = generated;
  summary.decode_repeats = repeats;
  printf("\nTG / MTP decode benchmark (greedy, gamma=%d, %d generated tokens)\n", gamma,
         generated);
  printf("  %-18s %7s %10s %12s %14s\n", "scenario", "prompt", "decode", "MTP accept", "samples");
  double total_tokens = 0, total_seconds = 0, total_proposed = 0, total_accepted = 0;
  for (const auto& test : cases) {
    const std::vector<int> prompt = encode_case(tokenizer, test.prompt);
    if ((int)prompt.size() + generated > maxctx)
      throw std::runtime_error(std::string("decode case exceeds MAX_CONTEXT: ") + test.name);
    double elapsed = 0, accepted = 0, proposed = 0, decoded = 0;
    for (int sample = 0; sample < repeats; sample++) {
      model.mtp_tap_capture = true;
      model.reset_state();
      const int first = model.prefill_batch(prompt, 0, true, /*quiet=*/true);
      CK(hipDeviceSynchronize());
      const auto start = Clock::now();
      const std::vector<int> output = model.spec_loop(first, generated, gamma);
      CK(hipDeviceSynchronize());
      elapsed += seconds_since(start);
      decoded += output.size();
      proposed += model.last_spec_proposed;
      accepted += std::max(0, model.last_spec_commit - model.last_spec_rounds);
    }
    const double rate = decoded / elapsed;
    const double acceptance = proposed > 0 ? accepted / proposed : 0;
    printf("  %-18s %7zu %10.1f %11.1f%% %14d\n", test.name, prompt.size(), rate,
           acceptance * 100.0, repeats);
    summary.decode_rows.push_back({test.name, prompt.size(), rate, acceptance});
    total_tokens += decoded;
    total_seconds += elapsed;
    total_proposed += proposed;
    total_accepted += accepted;
  }
  summary.decode_avg = total_tokens / total_seconds;
  summary.decode_accept_avg = total_proposed > 0 ? total_accepted / total_proposed : 0;
  printf("TG average: %.1f tok/s; MTP acceptance average: %.1f%% (weighted)\n",
         summary.decode_avg, summary.decode_accept_avg * 100.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) return 0;
  if (argc != 2 || (std::strcmp(argv[1], "run") != 0 &&
                    std::strcmp(argv[1], "--run") != 0)) {
    fprintf(stderr, "Usage: %s run\n", argv[0]);
    fprintf(stderr, "No arguments exits immediately; run starts the configured benchmark.\n");
    return 2;
  }
  try {
    const std::filesystem::path root = find_root(argv[0]);
    BenchConfig config = read_service_config(root / "service.conf");
    configure_runtime(config);
    const ModelPaths paths = configured_model(config);
    configure_gguf_mtp(paths);
    std::string rope_error;
    if (!rope_from_env(g_cfg.rope, rope_error))
      throw std::runtime_error("invalid RoPE configuration: " + rope_error);
    const int maxctx = config_int(config, "MAX_CONTEXT", 262144);
    if (maxctx < 1) throw std::runtime_error("MAX_CONTEXT must be positive");
    LoadedModel loaded = load_model(paths, maxctx);
    GpuModel& model = *loaded.model;
    std::vector<int> warm((size_t)model.maxbatch, 0);
    model.mtp_tap_capture = false;
    model.bench_quiet = true;
    model.prefill_batch(warm, 0, true, /*quiet=*/true);
    CK(hipDeviceSynchronize());
    model.reset_state();
    printf("Model load: PASS (%s)\n", paths.base.string().c_str());
    BenchSummary summary;
    summary.format = paths.format;
    summary.model_path = paths.base.string();
    summary.maxctx = maxctx;
    printf("\nPrefill benchmark\n");
    run_prefill(model, config, maxctx, summary);
    const int decode_result = run_decode(model, config, maxctx, summary);
    print_summary(summary);
    if (decode_result != 0) return decode_result;
    printf("\nBenchmark complete: PASS\n");
    return 0;
  } catch (const std::bad_alloc&) {
    fprintf(stderr, "Benchmark aborted: host memory allocation failed.\n");
    return 1;
  } catch (const std::exception& error) {
    fprintf(stderr, "Benchmark aborted: %s\n", error.what());
    return 1;
  }
}
