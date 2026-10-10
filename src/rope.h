#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

struct RopeConfig {
  double factor = 1.0;
  int original_max_position_embeddings = 262144;
  double beta_fast = 32.0;
  double beta_slow = 1.0;
  double attention_factor = 0.0;
};

inline double rope_attention_factor(const RopeConfig& rope) {
  if (rope.attention_factor > 0.0) return rope.attention_factor;
  return rope.factor > 1.0 ? 1.0 + 0.1 * std::log(rope.factor) : 1.0;
}

inline bool rope_config_valid(const RopeConfig& rope) {
  return std::isfinite(rope.factor) && rope.factor >= 1.0 &&
         std::isfinite(rope.attention_factor) && rope.attention_factor >= 0.0 &&
         std::isfinite(rope_attention_factor(rope)) &&
         rope.original_max_position_embeddings > 0 &&
         std::isfinite(rope.beta_fast) && std::isfinite(rope.beta_slow) &&
         rope.beta_fast >= rope.beta_slow && rope.beta_slow > 0.0;
}

inline double rope_inv_freq(const RopeConfig& rope, double theta, int rotary,
                            int index) {
  const double native = std::pow(theta, -2.0 * index / rotary);
  if (rope.factor == 1.0) return native;
  const double original_ctx = rope.original_max_position_embeddings;
  const double two_pi = 2.0 * std::acos(-1.0);
  const double denominator = 2.0 * std::log(theta);
  const double low = std::max(0.0, std::floor(
      rotary * std::log(original_ctx / (rope.beta_fast * two_pi)) / denominator));
  double high = std::min((double)rotary - 1, std::ceil(
      rotary * std::log(original_ctx / (rope.beta_slow * two_pi)) / denominator));
  if (low == high) high += 0.001;
  const double ramp = std::max(0.0, std::min(1.0, (index - low) / (high - low)));
  return native / rope.factor * ramp + native * (1.0 - ramp);
}

inline bool rope_set(RopeConfig& rope, const char* key, const char* text) {
  if (!text || !*text) return false;
  char* end = nullptr;
  errno = 0;
  if (!std::strcmp(key, "original_ctx")) {
    long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT_MAX) return false;
    rope.original_max_position_embeddings = (int)value;
    return true;
  }
  double value = std::strtod(text, &end);
  if (errno || end == text || *end || !std::isfinite(value)) return false;
  if (!std::strcmp(key, "factor")) rope.factor = value;
  else if (!std::strcmp(key, "beta_fast")) rope.beta_fast = value;
  else if (!std::strcmp(key, "beta_slow")) rope.beta_slow = value;
  else if (!std::strcmp(key, "attention_factor")) rope.attention_factor = value;
  else return false;
  return true;
}

inline const char* rope_cli_key(const std::string& option) {
  if (option == "--rope-factor") return "factor";
  if (option == "--rope-original-ctx") return "original_ctx";
  if (option == "--rope-beta-fast") return "beta_fast";
  if (option == "--rope-beta-slow") return "beta_slow";
  if (option == "--rope-attn-scale") return "attention_factor";
  return nullptr;
}

inline bool rope_from_env(RopeConfig& rope, std::string& error) {
  const char* names[][2] = {
      {"QWENOX_ROPE_FACTOR", "factor"},
      {"QWENOX_ROPE_ORIGINAL_CTX", "original_ctx"},
      {"QWENOX_ROPE_BETA_FAST", "beta_fast"},
      {"QWENOX_ROPE_BETA_SLOW", "beta_slow"},
      {"QWENOX_ROPE_ATTN_SCALE", "attention_factor"}};
  for (const auto& entry : names) {
    const char* text = std::getenv(entry[0]);
    if (text && !rope_set(rope, entry[1], text)) {
      error = std::string("invalid ") + entry[0] + "=" + text;
      return false;
    }
  }
  return true;
}
