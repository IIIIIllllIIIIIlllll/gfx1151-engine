// service_conf.h — service.conf 的共享读写层（header-only，纯 STL，跨平台）。
// 供 start_win.exe（launch_win.cpp / launch_panel.inc）与 qwenox-api 共同使用，
// 保证两边对 conf 的解析与写回行为完全一致。
//
// 格式：bash 可 source 的 KEY="${KEY:-默认}" 行（可选键用 ${KEY-值} 允许置空），
// 解析识别 KEY="v" / KEY='v' / KEY=v 与 ${KEY:-d} / ${KEY-d}，值内 $VAR/${VAR}
// 展开（取环境变量或 conf 中先解析的键）。取值优先级：环境变量 > conf > 内置默认。
#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace svcconf {

using Conf = std::map<std::string, std::string>;

// 面板/网页前端管理的 conf 键（也是环境变量覆盖提示的范围）。
// ROPE_ORIGINAL_CTX 刻意不在其中：不做 UI 行，save_conf 也不改写它。
inline const char* kManagedKeys[] = {
    "MODEL_FILE", "NGRAM_FILE", "OVERLAY_FILE", "MTP_FILE", "VISION_FILE",
    "GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE",
    "TOKENIZER_DIR", "MAX_CONTEXT", "PARALLEL", "KV_POOL_TOKENS", "KV_PAGED",
    "PREFILL_CHUNK", "ROPE_FACTOR",
    "KVSNAP_MAX_GB",
    "ENGINE_HOST", "ENGINE_PORT", "API_HOST", "API_PORT",
};
inline const size_t kManagedKeysCount = sizeof(kManagedKeys) / sizeof(kManagedKeys[0]);

// 可选文件键：写回时用 ${KEY-值}（无冒号，允许显式置空禁用），与 service.conf 现状一致。
inline const char* kOptionalFileKeys[] = {"OVERLAY_FILE", "MTP_FILE", "VISION_FILE",
                                          "GGUF_MTP_FILE", "GGUF_VISION_FILE"};

inline bool is_optional_file_key(const std::string& key) {
    for (const char* k : kOptionalFileKeys)
        if (key == k) return true;
    return false;
}

inline bool is_managed_key(const std::string& key) {
    for (size_t i = 0; i < kManagedKeysCount; ++i)
        if (key == kManagedKeys[i]) return true;
    return false;
}

inline bool read_text_file(const std::string& path, std::string* out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    fclose(f);
    return true;
}

inline std::string trim(const std::string& s) {
    // Windows 上 service.conf 常带 CRLF（如 git autocrlf=true 签出）：
    // 不去掉 \r 会让行尾引号剥不掉、"${KEY:-d}" 识别失败，值变成 ""\r 之类
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

inline std::string expand(const std::string& in, const Conf& conf) {
    std::string out;
    for (size_t i = 0; i < in.size();) {
        if (in[i] != '$') {
            out += in[i++];
            continue;
        }
        std::string name;
        size_t j = i + 1;
        if (j < in.size() && in[j] == '{') {
            size_t k = in.find('}', j);
            if (k == std::string::npos) {
                out += in[i++];
                continue;
            }
            name = in.substr(j + 1, k - j - 1);
            i = k + 1;
        } else {
            while (j < in.size() && (isalnum(static_cast<unsigned char>(in[j])) ||
                                     in[j] == '_'))
                name += in[j++];
            if (name.empty()) {
                out += in[i++];
                continue;
            }
            i = j;
        }
        const char* e = getenv(name.c_str());
        auto it = conf.find(name);
        if (e) out += e;
        else if (it != conf.end()) out += it->second;
    }
    return out;
}

inline void load_conf(const std::string& path, Conf* conf) {
    std::string text;
    if (!read_text_file(path, &text)) return;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        std::string line = text.substr(pos, eol == std::string::npos
                                              ? std::string::npos : eol - pos);
        pos = eol == std::string::npos ? text.size() : eol + 1;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        if (key.empty() ||
            key.find_first_not_of(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_") !=
                std::string::npos)
            continue;
        std::string rhs = trim(line.substr(eq + 1));
        if (rhs.size() >= 2 && (rhs[0] == '"' || rhs[0] == '\'') &&
            rhs.back() == rhs[0])
            rhs = rhs.substr(1, rhs.size() - 2);
        const char* env = getenv(key.c_str());
        const std::string colon_def = "${" + key + ":-";
        const std::string plain_def = "${" + key + "-";
        if (rhs.compare(0, colon_def.size(), colon_def) == 0 && rhs.back() == '}') {
            if (env && *env) continue;  // :- 语义：非空环境变量胜出
            (*conf)[key] = expand(rhs.substr(colon_def.size(),
                                             rhs.size() - colon_def.size() - 1), *conf);
        } else if (rhs.compare(0, plain_def.size(), plain_def) == 0 &&
                   rhs.back() == '}') {
            if (env) continue;  // - 语义：环境变量存在即胜出（置空也算）
            (*conf)[key] = expand(rhs.substr(plain_def.size(),
                                             rhs.size() - plain_def.size() - 1), *conf);
        } else {
            if (env) continue;  // 字面量：环境变量优先
            (*conf)[key] = expand(rhs, *conf);
        }
    }
}

inline std::string cfg(const Conf& conf, const char* key, const std::string& builtin) {
    const char* e = getenv(key);
    if (e && *e) return e;
    auto it = conf.find(key);
    if (it != conf.end() && !it->second.empty()) return it->second;
    return builtin;
}

// MTP_FILE / VISION_FILE 等可选项：显式置空（env 或 conf）即禁用。
inline std::string cfg_optional(const Conf& conf, const char* key,
                                const std::string& builtin) {
    const char* e = getenv(key);
    if (e) return e;
    auto it = conf.find(key);
    if (it != conf.end()) return it->second;
    return builtin;
}

// 严格整数解析（不退出）：空串、尾随垃圾都失败。
inline bool parse_int(const std::string& s, long* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const long n = strtol(s.c_str(), &end, 10);
    if (!end || *end) return false;
    *out = n;
    return true;
}

// 非负小数解析（不退出）：完整解析、有限、非负。
inline bool parse_double(const std::string& s, double* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const double n = strtod(s.c_str(), &end);
    if (!end || *end || end == s.c_str() || n != n || n < 0) return false;
    *out = n;
    return true;
}

// 读 "# <prefix><value>" 注释标记行（如 "# start_win: weights=v2"）；没有返回空串。
inline std::string conf_marker(const std::string& path, const char* prefix) {
    std::string text;
    if (!read_text_file(path, &text)) return "";
    const size_t p = text.find(prefix);
    if (p == std::string::npos) return "";
    const size_t b = p + strlen(prefix);
    const size_t e = text.find_first_of("\r\n", b);
    return trim(text.substr(b, e == std::string::npos ? std::string::npos : e - b));
}

// 写回 service.conf：注释/空行/未知键原样保留；已知键整行替换为
// KEY="${KEY:-值}"（可选文件键为 KEY="${KEY-值}"），保持 bash 启动器兼容。
// 键缺失时追加到文件末尾。写前备份为 <path>.bak。
// marker_prefix/marker_value：顺带 upsert 一行注释标记（如
// "# start_win: weights=" + "v2"），与键值同一次写入（同一份 .bak 备份）。
inline bool save_conf(const std::string& path,
                      const std::map<std::string, std::string>& vals,
                      std::string* error,
                      const char* marker_prefix = nullptr,
                      const std::string& marker_value = "") {
    std::string text;
    read_text_file(path, &text);  // 缺失视为空文件，后面全量追加
    std::map<std::string, bool> written;
    bool marker_written = false;
    std::string out;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t eol = text.find('\n', pos);
        const bool last = eol == std::string::npos;
        std::string line = text.substr(pos, last ? std::string::npos : eol - pos);
        pos = last ? text.size() + 1 : eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        bool replaced = false;
        // 注释标记行：已有该前缀行则整行改写
        if (marker_prefix && !marker_written &&
            line.compare(0, strlen(marker_prefix), marker_prefix) == 0) {
            out += std::string(marker_prefix) + marker_value;
            marker_written = true;
            replaced = true;
        }
        // 匹配 ^\s*KEY\s*=（KEY 为待写键）
        size_t i = line.find_first_not_of(" \t");
        if (!replaced && i != std::string::npos && line[i] != '#') {
            size_t eq = line.find('=', i);
            if (eq != std::string::npos) {
                std::string key = line.substr(i, eq - i);
                while (!key.empty() && (key.back() == ' ' || key.back() == '\t'))
                    key.pop_back();
                auto it = vals.find(key);
                if (it != vals.end() && !key.empty()) {
                    const char* sep = is_optional_file_key(key) ? "-" : ":-";
                    out += line.substr(0, i) + key + "=\"${" + key + sep + it->second +
                           "}\"";
                    replaced = true;
                    written[key] = true;
                }
            }
        }
        if (!replaced) out += line;
        if (!last) out += '\n';
    }
    for (const auto& kv : vals)
        if (!written[kv.first]) {
            const char* sep = is_optional_file_key(kv.first) ? "-" : ":-";
            out += kv.first + "=\"${" + kv.first + sep + kv.second + "}\"\n";
        }
    if (marker_prefix && !marker_written) {
        if (!out.empty() && out.back() != '\n') out += '\n';
        out += std::string(marker_prefix) + marker_value + "\n";
    }
    if (!text.empty()) {
        FILE* bak = fopen((path + ".bak").c_str(), "wb");
        if (bak) {
            fwrite(text.data(), 1, text.size(), bak);
            fclose(bak);
        }
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        if (error) *error = "cannot write " + path;
        return false;
    }
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return true;
}

// 非退出版配置校验（网页/面板保存用；启动器自己仍有 fail 版检查）。
// 校验的是"有效值"（env > conf > 内置默认解析后的 Conf），规则与
// launch_win.cpp 的启动检查一致；错误信息为英文，调用方自行翻译。
inline bool validate_conf(const Conf& conf, std::string* err) {
    auto get_int = [&](const char* key, long dflt, long lo, long hi, long* out) -> bool {
        const std::string v = cfg(conf, key, "");
        long n = dflt;
        if (!v.empty() && !parse_int(v, &n)) {
            *err = std::string(key) + " must be an integer (got \"" + v + "\")";
            return false;
        }
        if (n < lo || n > hi) {
            *err = std::string(key) + " must be in [" + std::to_string(lo) + ", " +
                   std::to_string(hi) + "]";
            return false;
        }
        *out = n;
        return true;
    };
    long engine_port, api_port, max_context, parallel, kv_paged, kv_pool_tokens,
        prefill_chunk, kvsnap_max_gb, rope_original;
    if (!get_int("ENGINE_PORT", 8730, 1, 65535, &engine_port)) return false;
    if (!get_int("API_PORT", 8731, 1, 65535, &api_port)) return false;
    if (!get_int("MAX_CONTEXT", 262144, 1024, 1 << 20, &max_context)) return false;
    if (!get_int("PARALLEL", 1, 1, 8, &parallel)) return false;
    if (!get_int("KV_PAGED", 1, 0, 1, &kv_paged)) return false;
    if (!get_int("KV_POOL_TOKENS", 0, 0, 1 << 24, &kv_pool_tokens)) return false;
    if (!get_int("PREFILL_CHUNK", 0, 0, 1 << 20, &prefill_chunk)) return false;
    if (!get_int("KVSNAP_MAX_GB", 20, 0, 1 << 16, &kvsnap_max_gb)) return false;
    if (!get_int("ROPE_ORIGINAL_CTX", 262144, 1, 1 << 30, &rope_original)) return false;
    (void)prefill_chunk;   // 只校验范围，值本身不再参与后续规则
    (void)kvsnap_max_gb;
    if (engine_port == api_port) {
        *err = "ENGINE_PORT and API_PORT must differ";
        return false;
    }
    if (parallel > 1 && !kv_paged) {
        *err = "PARALLEL>1 requires KV_PAGED=1";
        return false;
    }
    // Windows 设备内存是硬上限 95 GiB 的 arena：有效池 = max(KV_POOL_TOKENS,
    // MAX_CONTEXT)，512K（524288）以上任何配置都放不下。
    const long kv_pool_effective =
        kv_pool_tokens > max_context ? kv_pool_tokens : max_context;
    if (kv_pool_effective > 524288) {
        *err = "effective KV pool " + std::to_string(kv_pool_effective) +
               " tokens exceeds the 512K (524288) limit";
        return false;
    }
    double rope_factor, rope_fast, rope_slow;
    if (!parse_double(cfg(conf, "ROPE_FACTOR", "1"), &rope_factor)) {
        *err = "ROPE_FACTOR must be a non-negative number";
        return false;
    }
    if (!parse_double(cfg(conf, "ROPE_BETA_FAST", "32"), &rope_fast) ||
        !parse_double(cfg(conf, "ROPE_BETA_SLOW", "1"), &rope_slow)) {
        *err = "ROPE_BETA_FAST / ROPE_BETA_SLOW must be non-negative numbers";
        return false;
    }
    if (rope_factor < 1 || rope_fast < rope_slow || rope_slow <= 0) {
        *err = "ROPE_FACTOR must be >=1, beta_fast >= beta_slow > 0";
        return false;
    }
    if (rope_factor > 1 && max_context > rope_factor * rope_original) {
        *err = "MAX_CONTEXT exceeds ROPE_FACTOR*ROPE_ORIGINAL_CTX (YaRN limit)";
        return false;
    }
    return true;
}

}  // namespace svcconf
