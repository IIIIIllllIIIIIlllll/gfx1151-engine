// main.cpp — gdec-api: C++ OpenAI-compatible front-end.
//
// Independently implements the OpenAI-compatible front-end from public API
// schemas and project-owned black-box conformance fixtures. The tokenizer and
// chat template are verified by tools/tok_ab.py and tools/template_ab.py.
//
// KV reuse: text-only chat/responses requests send SNAPS hints (semantic
// message-boundary token cuts, see compute_snap_cuts); the engine keeps RAM
// checkpoints (rckpt) at those cuts, so edit-and-resend and multi-turn
// retokenization wobble hit the cache instead of re-prefilling. Vision
// requests and raw completions send no hints — the engine's `cont` strict-
// prefix reuse and the SSD kvsnap tier still apply.
#include <atomic>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include "../gpu/os_win32.h"  // wsa_init()
#endif

#include "chat_template.h"
#include "engine_client.h"
#include "http.h"
#include "json_py.h"
#include "reqstat.h"
#include "tokenizer.h"
#include "toolparse.h"
#include "vision.h"

namespace {

using json = nlohmann::ordered_json;

// ----------------------------------------------------------------- config --

struct Config {
    std::string tokenizer_dir = "models/tokenizer";
    std::string engine_addr = "127.0.0.1:8730";
    std::string listen = "0.0.0.0:8731";
    std::string model = "qwen3.8-flash-next";
    int context = 262144;
};

Config g_cfg;
gdec::Tokenizer g_tok;
gdec::EngineClient g_eng;
std::mutex g_conn_mtx;  // serialize (re)connects
std::atomic<long long> g_req_seq{0};

// The engine is batch-1 (INFO reports kv_slots = 1), so requests are
// serialized here rather than queued engine-side.
std::mutex g_slot_mtx;
int g_in_flight = 0;
std::chrono::steady_clock::time_point g_busy_since;

// generation_config.json of this checkpoint.
const std::vector<int> g_eos = {248046, 248044};

long long now_unix() {
    return (long long)std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string hex24() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* kHex = "0123456789abcdef";
    uint64_t v = rng();
    std::string s;
    s.reserve(24);
    for (int i = 0; i < 24; ++i) {
        if (i == 12) v = rng();
        s += kHex[(v >> (((i % 12) * 4) + 4)) & 0xf];
    }
    return s;
}

std::string make_id(const char* prefix) { return std::string(prefix) + hex24(); }

const char* kEngineRejected =
    "the engine rejected this request. Check /health for what this build "
    "supports, and that the prompt fits the context.";

// ------------------------------------------------------------- slot guard --

class SlotGuard {
  public:
    explicit SlotGuard(bool cache_probe = false) {
        std::lock_guard<std::mutex> lk(g_slot_mtx);
        if (g_in_flight != 0) {
            if (cache_probe)
                http::fail(500, "internal error: TimeoutError: ", "server_error",
                           "server_error");
            http::fail(503, "engine busy", "server_error", "engine_busy");
        }
        g_in_flight = 1;
        g_busy_since = std::chrono::steady_clock::now();
    }
    ~SlotGuard() {
        {
            std::lock_guard<std::mutex> lk(g_slot_mtx);
            g_in_flight = 0;
        }
    }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;
};

bool engine_ready(std::string* err) {
    std::lock_guard<std::mutex> lk(g_conn_mtx);
    if (g_eng.connected()) return true;
    return g_eng.connect(g_cfg.engine_addr, err);
}

// ------------------------------------------------------- incremental text --

// ------------------------------------------------------------------ utf8 --

const char* kReplacement = "\xEF\xBF\xBD";  // U+FFFD

// Largest cut <= n that does not split a UTF-8 character. Used wherever a
// look-ahead window is cut by a byte count (the </think> splitter): cutting at
// a lead-byte boundary keeps every emitted frame independently valid UTF-8,
// which per-frame JSON parsers require.
size_t utf8_boundary_at_or_before(const std::string& s, size_t n) {
    if (n > s.size()) n = s.size();
    while (n > 0 && n < s.size() && ((unsigned char)s[n] & 0xC0) == 0x80) --n;
    return n;
}

// Consumes well-formed text from `pending`, repairing ill-formed bytes to
// U+FFFD the same way the tokenizer's lossy decode does. Without `final`, a
// trailing incomplete character stays buffered.
std::string take_utf8(std::string* pending, bool final) {
    const std::string& s = *pending;
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            out += (char)c;
            ++i;
            continue;
        }
        size_t need = 0;
        unsigned lo2 = 0x80, hi2 = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) need = 2;
        else if (c >= 0xE0 && c <= 0xEF) {
            need = 3;
            if (c == 0xE0) lo2 = 0xA0;
            if (c == 0xED) hi2 = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 4;
            if (c == 0xF0) lo2 = 0x90;
            if (c == 0xF4) hi2 = 0x8F;
        }
        if (need == 0) {  // ill-formed lead byte
            out += kReplacement;
            ++i;
            continue;
        }
        if (i + need > s.size()) {
            if (!final) break;
            out += kReplacement;  // truncated tail at end of stream
            i = s.size();
            break;
        }
        size_t valid = 1;  // maximal subpart: the lead byte always counts
        for (size_t k = 1; k < need; ++k) {
            const unsigned char cc = (unsigned char)s[i + k];
            if (cc < (k == 1 ? lo2 : 0x80) || cc > (k == 1 ? hi2 : 0xBF)) break;
            ++valid;
        }
        if (valid == need) {
            out.append(s, i, need);
            i += need;
        } else {
            out += kReplacement;
            i += valid;
        }
    }
    pending->erase(0, i);
    return out;
}

// Streams token -> text, holding back a multi-byte character split across
// tokens until it is complete (decode() would otherwise emit U+FFFD first).
class Detokenizer {
  public:
    explicit Detokenizer(const gdec::Tokenizer* tok) : tok_(tok) {}
    std::string push(int id) {
        pending_ += tok_->decode_bytes(std::vector<int>{id}, /*skip_special=*/false);
        return take_utf8(&pending_, /*final=*/false);
    }
    std::string flush() { return take_utf8(&pending_, /*final=*/true); }

  private:
    const gdec::Tokenizer* tok_;
    std::string pending_;
};

// ------------------------------------------------------------ generation ---

struct GenSpec {
    std::vector<int> ids;
    int max_tokens = 0;  // 0 => context budget
    bool sample = false;
    float temp = 1.0f;
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    unsigned long long seed = 0;
    float presence = 0.0f;
    float frequency = 0.0f;
    std::vector<std::pair<int, float>> bias;
    bool logprobs = false;
    std::vector<std::string> stop;
    std::vector<std::array<int, 3>> mrope_grids;
    std::vector<std::vector<float>> patches;
    std::vector<long long> snaps;  // SNAPS hints: semantic boundary token cuts
};

struct GenOutcome {
    std::string text;
    std::string reason;  // OpenAI finish_reason
    long long n_prompt = 0;
    long long n_gen = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ttft_ms = 0.0;
    int n_cached = 0, rounds = 0, commit = 0, proposed = 0;
    int served_max_tokens = 0;
    int clamped_from = 0;
    std::vector<float> logprobs;
    bool hit_stop = false;
    bool client_gone = false;
};

// on_delta receives each incremental piece of text; returning false aborts the
// generation (stop sequence matched, or the HTTP client vanished). May be
// empty for non-streaming use.
GenOutcome run_generation(GenSpec& spec,
                          const std::function<bool(const std::string&)>& on_delta,
                          const std::string& log_tag) {
    SlotGuard slot;
    std::string err;
    if (!engine_ready(&err)) {
        fprintf(stderr, "REQ %s 502 engine connect failed: %s\n", log_tag.c_str(), err.c_str());
        http::fail(502, "engine not reachable: " + err, "server_error", "server_error");
    }

    const long long req_id = ++g_req_seq;

    gdec::GenParams p;
    p.req = req_id;
    p.ids = spec.ids;
    const long long budget = static_cast<long long>(g_cfg.context) -
                             static_cast<long long>(spec.ids.size());
    if (budget <= 0) http::fail(400, kEngineRejected);
    const long long requested = spec.max_tokens > 0 ? spec.max_tokens : budget;
    p.max_tokens = static_cast<int>(std::min(requested, budget));
    p.eos = g_eos;
    p.sample = spec.sample;
    p.temp = spec.temp;
    p.top_k = spec.top_k;
    p.top_p = spec.top_p;
    p.min_p = spec.min_p;
    p.seed = spec.seed;
    p.presence = spec.presence;
    p.frequency = spec.frequency;
    p.bias = spec.bias;
    p.logprobs = spec.logprobs;
    p.mrope_grids = spec.mrope_grids;
    p.patches = std::move(spec.patches);
    p.snaps = spec.snaps;

    GenOutcome out;
    out.served_max_tokens = p.max_tokens;
    if (spec.max_tokens > p.max_tokens) out.clamped_from = spec.max_tokens;
    Detokenizer detok(&g_tok);
    std::string full;
    std::string pending_emit;
    size_t max_stop_len = 0;
    for (const std::string& stop : spec.stop) max_stop_len = std::max(max_stop_len, stop.size());
    double ttft = 0.0;
    bool first_piece = true;
    long long emitted = 0;
    auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "REQ %lld start %s prompt=%zu max=%d %s\n", req_id, log_tag.c_str(),
            spec.ids.size(), p.max_tokens, spec.sample ? "sampled" : "greedy");

    auto deliver = [&](const std::string& piece) -> bool {
        if (piece.empty()) return true;
        if (first_piece) {
            ttft = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
            first_piece = false;
        }
        full += piece;
        if (on_delta && !on_delta(piece)) {
            out.client_gone = true;
            return false;
        }
        return true;
    };

    auto emit = [&](const std::string& piece) -> bool {
        pending_emit += piece;
        // Hold back enough suffix bytes that a stop sequence spanning two
        // decoded token pieces can still be removed before it reaches SSE.
        size_t cut = std::string::npos;
        for (const std::string& s : spec.stop) {
            size_t at = pending_emit.find(s);
            if (at != std::string::npos && (cut == std::string::npos || at < cut)) cut = at;
        }
        if (cut != std::string::npos) {
            std::string delta = pending_emit.substr(0, cut);
            pending_emit.clear();
            out.hit_stop = true;
            if (!deliver(delta)) out.client_gone = true;
            return false;
        }
        size_t safe = pending_emit.size();
        if (max_stop_len > 1) {
            const size_t hold = max_stop_len - 1;
            safe = safe > hold ? safe - hold : 0;
            safe = utf8_boundary_at_or_before(pending_emit, safe);
        }
        if (safe == 0) return true;
        std::string delta = pending_emit.substr(0, safe);
        pending_emit.erase(0, safe);
        return deliver(delta);
    };

    gdec::GenResult r = g_eng.generate(p, [&](int tok, float lp) {
        (void)lp;
        ++emitted;
        return emit(detok.push(tok));
    });
    out.ttft_ms = ttft;

    if (!r.transport_ok) {
        if (r.timed_out) http::fail(504, "engine timed out while generating", "server_error", "server_error");
        http::fail(502, "engine connection lost", "server_error", "server_error");
    }
    if (!out.hit_stop && !out.client_gone) {
        emit(detok.flush());
        if (!out.hit_stop && !out.client_gone && !pending_emit.empty()) {
            std::string tail = std::move(pending_emit);
            deliver(tail);
        }
    }

    out.text = full;
    out.n_prompt = r.n_prompt;
    // A cancelled request comes back as `D <req> cancel 0 0 0.0 0.0`, which
    // carries no counts, so fall back to what we actually emitted (a stop
    // sequence cancels the turn after the tokens that produced it).
    out.n_gen = r.n_gen > 0 ? r.n_gen : emitted;
    out.prefill_ms = r.prefill_ms;
    out.decode_ms = r.decode_ms;
    out.n_cached = r.n_cached;
    out.rounds = r.rounds;
    out.commit = r.commit;
    out.proposed = r.proposed;
    out.logprobs = std::move(r.logprobs);

    if (r.reason == "error") http::fail(400, kEngineRejected);
    if (out.hit_stop || r.reason == "done") out.reason = "stop";
    else if (r.reason == "length") out.reason = "length";
    else out.reason = "stop";  // "cancel": client gone, or stopped by us

    fprintf(stderr,
            "REQ %lld end finish=%s prompt=%lld gen=%lld ttft=%.0fms prefill=%.0fms "
            "decode=%.0fms cached=%d%s\n",
            req_id, out.reason.c_str(), out.n_prompt, out.n_gen, out.ttft_ms,
            out.prefill_ms, out.decode_ms, out.n_cached,
            out.hit_stop ? " (stop)" : (out.client_gone ? " (client gone)" : ""));
    if (out.rounds > 0)
        fprintf(stderr, "REQ %lld mtp rounds=%d commit=%d proposed=%d acc=%.1f%%\n", req_id,
                out.rounds, out.commit, out.proposed,
                out.proposed ? 100.0 * (out.commit - out.rounds) / out.proposed : 0.0);
    {
        auto us = [](double ms) -> uint32_t {
            double v = ms * 1000.0;
            return v >= 4294967295.0 ? 4294967295u : (uint32_t)(v > 0.0 ? v : 0.0);
        };
        reqstat::Entry e{};
        e.ts_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
        e.req_seq = (uint64_t)req_id;
        e.n_prompt = (uint32_t)std::min<long long>(out.n_prompt, UINT32_MAX);
        e.n_cached = (uint32_t)out.n_cached;
        e.n_gen = (uint32_t)std::min<long long>(out.n_gen, UINT32_MAX);
        e.proposed = (uint32_t)out.proposed;
        e.commit = (uint32_t)out.commit;
        e.rounds = (uint32_t)out.rounds;
        e.prefill_us = us(out.prefill_ms);
        e.decode_us = us(out.decode_ms);
        e.ttft_us = us(out.ttft_ms);
        const uint32_t finish = out.client_gone || r.reason == "cancel" ? 3u
                                : out.reason == "length"               ? 2u
                                                                       : 1u;
        e.flags = finish | ((uint32_t)(r.drafter & 0xF) << 4) |
                  (!p.patches.empty() ? 0x100u : 0u);
        reqstat::record(e);
    }
    return out;
}

// -------------------------------------------------------------- request io --

json parse_body(const http::Request& q) {
    if (q.body.empty()) return json::object();
    json body;
    try {
        body = json::parse(q.body);
    } catch (...) {
        http::fail(400, "1: JSON decode error");
    }
    if (!body.is_object()) http::fail(400, "request body must be a JSON object");
    return body;
}

bool bool_field(const json& body, const char* key, bool fallback) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return fallback;
    if (!it->is_boolean()) http::fail(400, std::string(key) + " must be a boolean");
    return it->get<bool>();
}

double number_field(const json& body, const char* key, double fallback) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return fallback;
    if (!it->is_number()) http::fail(400, std::string(key) + " must be a number");
    const double value = it->get<double>();
    if (!std::isfinite(value)) http::fail(400, std::string(key) + " must be finite");
    return value;
}

long long integer_value(const json& value, const char* key) {
    if (value.is_number_unsigned()) {
        const unsigned long long v = value.get<unsigned long long>();
        if (v > static_cast<unsigned long long>(std::numeric_limits<long long>::max()))
            http::fail(400, std::string(key) + " is too large");
        return static_cast<long long>(v);
    }
    if (value.is_number_integer()) return value.get<long long>();
    http::fail(400, std::string(key) + " must be an integer");
}

long long integer_field(const json& body, const char* key, long long fallback) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return fallback;
    return integer_value(*it, key);
}

void require_optional_boolean(const json& body, const char* key) {
    auto it = body.find(key);
    if (it != body.end() && !it->is_null() && !it->is_boolean())
        http::fail(400, std::string(key) + " must be a boolean");
}

bool request_wants_logprobs(const json& body) {
    if (!body.contains("logprobs") || body["logprobs"].is_null()) return false;
    const json& lp = body["logprobs"];
    if (lp.is_boolean()) return lp.get<bool>();
    if (lp.is_number_integer() || lp.is_number_unsigned()) {
        const long long value = integer_value(lp, "logprobs");
        if (value < 0) http::fail(400, "logprobs must be non-negative");
        return value > 0;
    }
    http::fail(400, "logprobs must be a boolean or integer");
}

// Fills the shared sampling / budget fields of a GenSpec.
void apply_sampling(const json& body, GenSpec* spec) {
    // temperature omitted => the checkpoint's Qwen defaults (sampled);
    // explicit 0 => greedy.
    const double temp = number_field(body, "temperature", 1.0);
    if (temp < 0.0 || temp > 2.0)
        http::fail(400, "temperature=" + json_py::py_float(temp) +
                            " is outside 0.0..2.0. Values past the range are NOT "
                            "clamped, because a clamped request is indistinguishable, "
                            "from the response, from one that was served as asked.");
    spec->sample = temp > 0.0;
    spec->temp = (float)temp;
    const long long top_k = integer_field(body, "top_k", 20);
    if (top_k < 0 || top_k > std::numeric_limits<int>::max())
        http::fail(400, "top_k must be between 0 and 2147483647");
    spec->top_k = static_cast<int>(top_k);
    spec->top_p = static_cast<float>(number_field(body, "top_p", 0.95));
    spec->min_p = static_cast<float>(number_field(body, "min_p", 0.0));
    if (spec->top_p < 0.0f || spec->top_p > 1.0f)
        http::fail(400, "top_p must be between 0 and 1");
    if (spec->min_p < 0.0f || spec->min_p > 1.0f)
        http::fail(400, "min_p must be between 0 and 1");
    spec->presence = static_cast<float>(number_field(body, "presence_penalty", 0.0));
    spec->frequency = static_cast<float>(number_field(body, "frequency_penalty", 0.0));
    if (spec->presence < -2.0f || spec->presence > 2.0f)
        http::fail(400, "presence_penalty must be between -2 and 2");
    if (spec->frequency < -2.0f || spec->frequency > 2.0f)
        http::fail(400, "frequency_penalty must be between -2 and 2");

    spec->logprobs = request_wants_logprobs(body);
    if (spec->logprobs && !spec->sample)
        http::fail(400,
                   "logprobs applies to the sampler, and this request decodes greedy "
                   "(explicit temperature: 0). Send temperature > 0 (or omit it for "
                   "the server-default sampling), or drop logprobs: a request silently "
                   "served without it would be indistinguishable from one that "
                   "honoured it.");

    if (body.contains("seed") && !body["seed"].is_null()) {
        const json& s = body["seed"];
        if (s.is_number_unsigned()) spec->seed = s.get<unsigned long long>();
        else if (s.is_number_integer())
            spec->seed = static_cast<unsigned long long>(s.get<long long>());
        else if (s.is_string()) {
            const std::string& text = s.get_ref<const std::string&>();
            unsigned long long value = 0;
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
                http::fail(400, "seed must be an integer or decimal integer string");
            spec->seed = value;
        } else {
            http::fail(400, "seed must be an integer or decimal integer string");
        }
    } else {
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        spec->seed = rng();  // the engine's RNG is counter-based: a seed is required
    }

    if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
        if (body["logit_bias"].size() > 20480)
            http::fail(400, "logit_bias has more than 20480 entries");
        for (auto it = body["logit_bias"].begin(); it != body["logit_bias"].end(); ++it) {
            int token = 0;
            const std::string& key = it.key();
            const auto parsed = std::from_chars(key.data(), key.data() + key.size(), token);
            if (parsed.ec != std::errc() || parsed.ptr != key.data() + key.size() || token < 0 ||
                token >= g_tok.vocab_size())
                http::fail(400, "logit_bias keys must be token ids");
            if (!it.value().is_number()) http::fail(400, "logit_bias values must be numbers");
            const double value = it.value().get<double>();
            if (!std::isfinite(value)) http::fail(400, "logit_bias values must be finite");
            if (value < -100.0 || value > 100.0)
                http::fail(400, "logit_bias values must be between -100 and 100");
            spec->bias.emplace_back(token, static_cast<float>(value));
        }
    } else if (body.contains("logit_bias") && !body["logit_bias"].is_null()) {
        http::fail(400, "logit_bias must be an object");
    }

    spec->max_tokens = 0;
    const char* selected_budget = nullptr;
    for (const char* k : {"max_tokens", "max_completion_tokens", "max_output_tokens"}) {
        if (body.contains(k) && !body[k].is_null()) {
            const long long value = integer_value(body[k], k);
            if (value <= 0 || value > std::numeric_limits<int>::max())
                http::fail(400, std::string(k) + " must be between 1 and 2147483647");
            if (selected_budget != nullptr && spec->max_tokens != value)
                http::fail(400, std::string(k) + " conflicts with " + selected_budget);
            spec->max_tokens = static_cast<int>(value);
            selected_budget = k;
        }
    }

    if (body.contains("stop") && !body["stop"].is_null()) {
        const json& s = body["stop"];
        if (s.is_string()) spec->stop.push_back(s.get<std::string>());
        else if (s.is_array()) {
            for (const auto& e : s) {
                if (!e.is_string()) http::fail(400, "stop entries must be strings");
                spec->stop.push_back(e.get<std::string>());
            }
        } else {
            http::fail(400, "stop must be a string or array of strings");
        }
        for (const std::string& stop : spec->stop)
            if (stop.empty()) http::fail(400, "stop strings must not be empty");
    }
}

json usage_json(const GenOutcome& o, const std::vector<int>& prompt_ids) {
    const long long pt = o.n_prompt > 0 ? o.n_prompt : (long long)prompt_ids.size();
    json u;
    u["prompt_tokens"] = pt;
    u["completion_tokens"] = o.n_gen;
    u["total_tokens"] = pt + o.n_gen;
    if (o.n_cached > 0) u["prompt_tokens_details"] = {{"cached_tokens", o.n_cached}};
    if (o.clamped_from > 0) u["clamped_from"] = o.clamped_from;
    return u;
}

json logprobs_json(const std::vector<float>& values) {
    json content = json::array();
    for (float value : values) {
        content.push_back({{"token", nullptr},
                           {"logprob", static_cast<double>(value)},
                           {"bytes", nullptr},
                           {"top_logprobs", json::array()}});
    }
    return json{{"content", std::move(content)}};
}

bool send_frame(http::Stream* st, const json& j) {
    return st->send_event(json_py::dumps(j, /*spaced=*/true));
}

bool stream_wants_usage(const json& body) {
    if (!body.contains("stream_options") || body["stream_options"].is_null()) return false;
    if (!body["stream_options"].is_object())
        http::fail(400, "stream_options must be an object");
    return bool_field(body["stream_options"], "include_usage", false);
}

const json* normalize_reasoning_effort(const json* value, json* storage) {
    if (value == nullptr || value->is_null()) return nullptr;
    if (!value->is_string()) http::fail(400, "reasoning_effort must be a string");
    const std::string& effort = value->get_ref<const std::string&>();
    if (effort == "high") {
        *storage = "xhigh";
        return storage;
    }
    if (effort == "minimal") {
        *storage = "low";
        return storage;
    }
    return value;
}

struct ToolSetup {
    json tools = json::array();
    toolparse::ToolChoice choice;

    bool enabled() const { return choice.tools_enabled() && !tools.empty(); }
};

ToolSetup parse_tool_setup(const json& body) {
    ToolSetup result;
    std::string error;
    const json* raw_tools = body.contains("tools") ? &body["tools"] : nullptr;
    if (!toolparse::normalize_tools(raw_tools, &result.tools, &error))
        http::fail(400, error);
    const json* raw_choice = body.contains("tool_choice") ? &body["tool_choice"] : nullptr;
    if (!toolparse::parse_tool_choice(raw_choice, result.tools, &result.choice, &error))
        http::fail(400, error);
    return result;
}

bool responses_media_item(const json& item) {
    if (!item.is_object()) return false;
    const std::string type =
        item.contains("type") && item["type"].is_string()
            ? item["type"].get<std::string>()
            : std::string();
    return type == "image" || type == "image_url" || type == "input_image" ||
           type == "computer_screenshot" || type == "video" ||
           type == "input_video" || item.contains("image") ||
           item.contains("image_url") || item.contains("video") ||
           item.contains("video_url");
}

json responses_message_content(const json& content, const char* field,
                               bool allow_media) {
    if (content.is_string()) return content;
    if (!content.is_array())
        http::fail(400, std::string(field) + " must be a string or a content array");
    json normalized = json::array();
    for (const auto& part : content) {
        if (!part.is_object())
            http::fail(400, std::string(field) + " content items must be objects");
        if (responses_media_item(part)) {
            if (!allow_media)
                http::fail(400, std::string(field) + " must contain only text items");
            normalized.push_back(part);
            continue;
        }
        if (!part.contains("text") || !part["text"].is_string())
            http::fail(400, std::string(field) + " contains an unsupported content item");
        normalized.push_back(part);
    }
    return normalized;
}

std::string responses_text_content(const json& content, const char* field) {
    const json normalized = responses_message_content(content, field, false);
    if (normalized.is_string()) return normalized.get<std::string>();
    std::string text;
    for (const auto& part : normalized) text += part["text"].get<std::string>();
    return text;
}

json normalize_responses_input(const json& body) {
    if (!body.contains("input") || body["input"].is_null())
        http::fail(400, "input is required");

    std::string system_text;
    bool has_system = false;
    if (body.contains("instructions") && !body["instructions"].is_null()) {
        if (!body["instructions"].is_string())
            http::fail(400, "instructions must be a string");
        system_text = body["instructions"].get<std::string>();
        has_system = true;
    }

    json ordinary = json::array();
    std::string pending_reasoning;
    bool has_pending_reasoning = false;
    auto append_system = [&](const json& content) {
        const std::string text = responses_text_content(content, "system content");
        if (has_system && !system_text.empty() && !text.empty()) system_text += "\n\n";
        system_text += text;
        has_system = true;
    };
    auto append_user_content = [&](json content) {
        ordinary.push_back(json{{"role", "user"}, {"content", std::move(content)}});
    };
    auto append_user_text = [&](const std::string& text) {
        append_user_content(text);
    };
    auto append_item = [&](const json& item) {
        if (item.is_string()) {
            append_user_text(item.get<std::string>());
            return;
        }
        if (!item.is_object())
            http::fail(400, "input items must be strings or message objects");

        const std::string type =
            item.contains("type") && item["type"].is_string()
                ? item["type"].get<std::string>()
                : std::string();
        if (type == "reasoning") {
            pending_reasoning.clear();
            if (item.contains("content") && !item["content"].is_null())
                pending_reasoning = responses_text_content(item["content"], "reasoning content");
            has_pending_reasoning = true;
            return;
        }
        if (type == "function_call") {
            const json* call_id = nullptr;
            if (item.contains("call_id")) call_id = &item["call_id"];
            else if (item.contains("id")) call_id = &item["id"];
            if (call_id == nullptr || !call_id->is_string() ||
                call_id->get_ref<const std::string&>().empty())
                http::fail(400, "function_call items require a string call_id");
            if (!item.contains("name") || !item["name"].is_string() ||
                item["name"].get_ref<const std::string&>().empty())
                http::fail(400, "function_call items require a string name");
            json arguments = item.contains("arguments") ? item["arguments"] : json("{}");
            json call = {{"id", *call_id},
                         {"type", "function"},
                         {"function", {{"name", item["name"]}, {"arguments", arguments}}}};
            if (!has_pending_reasoning && !ordinary.empty() && ordinary.back().is_object() &&
                ordinary.back().contains("role") && ordinary.back()["role"] == "assistant") {
                if (!ordinary.back().contains("tool_calls"))
                    ordinary.back()["tool_calls"] = json::array();
                ordinary.back()["tool_calls"].push_back(std::move(call));
            } else {
                json assistant = {{"role", "assistant"},
                                  {"content", ""},
                                  {"tool_calls", json::array({std::move(call)})}};
                if (has_pending_reasoning) {
                    assistant["reasoning_content"] = pending_reasoning;
                    pending_reasoning.clear();
                    has_pending_reasoning = false;
                }
                ordinary.push_back(std::move(assistant));
            }
            return;
        }
        if (type == "function_call_output") {
            if (!item.contains("call_id") || !item["call_id"].is_string() ||
                item["call_id"].get_ref<const std::string&>().empty())
                http::fail(400, "function_call_output items require a string call_id");
            if (!item.contains("output"))
                http::fail(400, "function_call_output items require output");
            ordinary.push_back(
                json{{"role", "tool"},
                     {"tool_call_id", item["call_id"]},
                     {"content", responses_text_content(item["output"], "function output")}});
            return;
        }
        if (type == "input_text") {
            if (!item.contains("text") || !item["text"].is_string())
                http::fail(400, "input_text items require a string text field");
            append_user_text(item["text"].get<std::string>());
            return;
        }
        if (!item.contains("role") || !item["role"].is_string())
            http::fail(400, "input message items require a string role");
        if (!item.contains("content"))
            http::fail(400, "input message items require content");

        const std::string role = item["role"].get<std::string>();
        if (role == "system" || role == "developer") {
            append_system(item["content"]);
            return;
        }
        if (role != "user" && role != "assistant")
            http::fail(400, "only user and assistant input messages are supported in this build");
        json content = responses_message_content(item["content"], "message content",
                                                 role == "user");
        json message = {{"role", role}, {"content", std::move(content)}};
        if (role == "assistant" && has_pending_reasoning) {
            message["reasoning_content"] = pending_reasoning;
            pending_reasoning.clear();
            has_pending_reasoning = false;
        }
        if (item.contains("tool_calls")) message["tool_calls"] = item["tool_calls"];
        ordinary.push_back(std::move(message));
    };

    const json& input = body["input"];
    if (input.is_string()) {
        append_user_text(input.get<std::string>());
    } else if (input.is_array()) {
        if (input.empty()) http::fail(400, "input must not be empty");
        const bool flat_content = input[0].is_object() && !input[0].contains("role") &&
                                  input[0].contains("type") && input[0]["type"].is_string() &&
                                  input[0]["type"] != "function_call" &&
                                  input[0]["type"] != "function_call_output" &&
                                  input[0]["type"] != "reasoning";
        if (flat_content)
            append_user_content(responses_message_content(input, "input", true));
        else
            for (const auto& item : input) append_item(item);
    } else {
        http::fail(400, "input must be a string or array");
    }

    json messages = json::array();
    if (has_system) messages.push_back(json{{"role", "system"}, {"content", system_text}});
    for (auto& item : ordinary) messages.push_back(std::move(item));
    std::string error;
    if (!toolparse::normalize_messages(&messages, &error)) http::fail(400, error);
    return messages;
}

// ------------------------------------------------------------------ routes --

void handle_models(const http::Request&, http::Response* r, http::Stream*) {
    json j;
    j["object"] = "list";
    j["data"] = json::array({json{{"id", g_cfg.model},
                                  {"object", "model"},
                                  {"owned_by", "local"},
                                  {"created", 0}}});
    r->body = json_py::dumps(j, /*spaced=*/false);
}

void handle_cache(const http::Request&, http::Response* r, http::Stream*) {
    SlotGuard slot(/*cache_probe=*/true);
    std::string line, err;
    {
        std::lock_guard<std::mutex> lk(g_conn_mtx);
        if (!g_eng.connected()) g_eng.connect(g_cfg.engine_addr, &err);
    }
    if (!g_eng.cstat(&line, &err)) {
        // The live service surfaces an engine-side probe failure this way.
        r->status = 500;
        r->body = http::error_json("internal error: TimeoutError: ", "server_error",
                                   "server_error");
        return;
    }
    json j;
    j["engine_cstat"] = line;
    r->body = json_py::dumps(j, /*spaced=*/false);
}

void handle_health(const http::Request&, http::Response* r, http::Stream*) {
    json j;
    j["status"] = "ok";
    j["model"] = g_cfg.model;
    j["endpoints"] = json::array(
        {"/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/responses"});
    j["context"] = g_cfg.context;
    j["rope_scaling"] = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_slot_mtx);
        j["busy"] = g_in_flight > 0;
        j["in_flight"] = g_in_flight;
        j["busy_for_s"] = g_in_flight > 0
                              ? (int)std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - g_busy_since)
                                    .count()
                              : 0;
    }
    j["slots"] = 1;
    j["slot_ctx"] = g_cfg.context;
    j["queued"] = 0;
    j["decode"] = std::string(
        "greedy at explicit temperature 0; Qwen-default sampling (temp 1.0, top_k 20, "
        "top_p 0.95) when temperature is omitted; sampled at temperature > 0");
    j["prompt_cache"] = {{"enabled", false}, {"cap_mb", 0}};
    j["tool_calls"] = {{"parsed", true},
                       {"template_injection", true},
                       {"wire_format", "qwen-xml (<function=>/<parameter=>)"},
                       {"streaming", true},
                       {"tool_choice", json::array({"auto", "none", "required", "named"})},
                       {"parallel_tool_calls", true},
                       {"constrained_decoding", false}};
    j["vision"] = {{"accepted", true},
                   {"chat_content_type", "image_url"},
                   {"responses_content_type", "input_image"},
                   {"inline_data_url", true},
                   {"bare_base64", true},
                   {"http_urls", false},
                   {"formats", json::array({"png", "jpeg", "webp"})},
                   {"max_images", 8},
                   {"video", false},
                   {"runtime", "native-cpp"}};
    j["supported"] = json::array(
        {"reasoning_effort", "enable_thinking", "preserve_thinking", "stop",
         "max_tokens", "max_completion_tokens", "max_output_tokens", "stream",
         "seed", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
         "frequency_penalty", "logit_bias", "logprobs", "tools", "tool_choice",
         "parallel_tool_calls", "image_url", "input_image", "add_vision_id"});
    j["token_budget_aliases"] =
        json::array({"max_tokens", "max_completion_tokens", "max_output_tokens"});
    j["max_tokens_default"] = "unbounded (budget = context minus prompt length)";
    j["error_format"] = "openai";
    j["accepted_but_ignored"] = json::array({"n", "drafter"});
    j["partial"] = json::array();
    j["sampling"] = {
        {"implemented", json::array({"temperature", "top_p", "top_k", "min_p", "seed",
                                     "presence_penalty", "frequency_penalty",
                                     "logit_bias", "logprobs"})}};
    j["max_tokens_cap"] = nullptr;
    j["reasoning_effort_values"] =
        json::array({"high", "low", "medium", "minimal", "xhigh"});
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// POST /v1/completions — raw prompt, no chat template.
void handle_completions(const http::Request& q, http::Response* r, http::Stream* st) {
    const json body = parse_body(q);
    GenSpec spec;
    if (body.contains("prompt")) {
        const json& p = body["prompt"];
        if (p.is_string()) spec.ids = g_tok.encode(p.get<std::string>());
        else if (p.is_array() && !p.empty() && p[0].is_string())
            spec.ids = g_tok.encode(p[0].get<std::string>());
        else
            http::fail(400, kEngineRejected);
    } else {
        http::fail(400, kEngineRejected);  // an empty prompt is an engine rejection
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    apply_sampling(body, &spec);

    const bool stream = bool_field(body, "stream", false) && st != nullptr;
    const bool include_usage = stream_wants_usage(body);
    if (stream && spec.logprobs)
        http::fail(400, "logprobs with stream=true is not supported in this build");
    const std::string id = make_id("cmpl-");
    const long long created = now_unix();
    auto frame_base = [&]() {
        json j;
        j["id"] = id;
        j["created"] = created;
        j["model"] = g_cfg.model;
        j["object"] = "text_completion";
        return j;
    };

    if (!stream) {
        GenOutcome o = run_generation(spec, nullptr, id);
        json j = frame_base();
        json choice{{"index", 0}, {"finish_reason", o.reason}, {"text", o.text}};
        if (spec.logprobs) choice["logprobs"] = logprobs_json(o.logprobs);
        j["choices"] = json::array({std::move(choice)});
        j["usage"] = usage_json(o, spec.ids);
        r->body = json_py::dumps(j, /*spaced=*/false);
        return;
    }

    r->sse = true;
    GenOutcome o = run_generation(
        spec,
        [&](const std::string& d) {
            json frame = frame_base();
            frame["choices"] = json::array(
                {json{{"index", 0}, {"finish_reason", nullptr}, {"text", d}}});
            return send_frame(st, frame);
        },
        id);
    json fin = frame_base();
    fin["choices"] =
        json::array({json{{"index", 0}, {"finish_reason", o.reason}, {"text", ""}}});
    send_frame(st, fin);
    if (include_usage) {
        json u = frame_base();
        u["choices"] = json::array();
        u["usage"] = usage_json(o, spec.ids);
        send_frame(st, u);
    }
    // The terminator is a bare payload, not JSON: send_event() wraps it as
    // `data: [DONE]`. Passing it through a `const json&` helper would quote it
    // into `data: "[DONE]"`, which streaming clients reject at end-of-stream.
    st->send_event("[DONE]");
}

// Splits a reply at </think>: before it is reasoning_content, after it content.
void split_reasoning(const std::string& text, bool thinking_enabled,
                     std::string* reasoning, std::string* content) {
    if (!thinking_enabled) {
        reasoning->clear();
        *content = text;
        return;
    }
    const size_t at = text.find("</think>");
    if (at == std::string::npos) {
        *reasoning = text;
        content->clear();
        while (!reasoning->empty() && reasoning->back() == '\n') reasoning->pop_back();
        return;
    }
    *reasoning = text.substr(0, at);
    while (!reasoning->empty() && reasoning->back() == '\n') reasoning->pop_back();
    *content = text.substr(at + 8);
    while (!content->empty() && (*content)[0] == '\n') content->erase(0, 1);
}

// Incremental version of the same split. A naive diff of split_reasoning()
// over the growing text shrinks the reasoning string once the marker lands
// (and re-emits it as content), so this holds back a partial-marker tail and
// keeps two independent cursors.
class ThinkSplitter {
  public:
    explicit ThinkSplitter(bool thinking_enabled) {
        if (!thinking_enabled) {
            split_ = true;
            content_started_ = true;
        }
    }

    void feed(const std::string& piece, std::string* reasoning_out,
              std::string* content_out) {
        buf_ += piece;
        if (!split_) {
            const size_t at = buf_.find(kMarker);
            if (at == std::string::npos) {
                // Hold back len(marker)-1 bytes in case the marker straddles
                // this piece and the next one, then snap that window to a
                // character boundary: a byte-count cut would split a
                // multi-byte character across two frames and make each frame
                // individually invalid UTF-8.
                size_t safe = buf_.size() >= kMarkerLen - 1 ? buf_.size() - (kMarkerLen - 1) : 0;
                safe = utf8_boundary_at_or_before(buf_, std::max(safe, rpos_));
                if (safe > rpos_) {
                    *reasoning_out = buf_.substr(rpos_, safe - rpos_);
                    rpos_ = safe;
                }
                return;
            }
            split_ = true;
            r_end_ = at;
            while (r_end_ > 0 && buf_[r_end_ - 1] == '\n') --r_end_;
            cpos_ = at + kMarkerLen;
        }
        if (rpos_ < r_end_) {
            *reasoning_out = buf_.substr(rpos_, r_end_ - rpos_);
            rpos_ = r_end_;
        }
        // Skip the newlines the template puts between </think> and the answer.
        // Only before the first content byte: afterwards a newline at the
        // cursor is real content ("```python\nprint(...") and must be emitted.
        // The marker can arrive before those bytes do, so this cannot be a
        // one-shot adjustment at split time -- it runs on every feed until
        // content starts, or the streamed text would keep a leading "\n\n"
        // that the non-streaming path strips.
        if (!content_started_) {
            while (cpos_ < buf_.size() && buf_[cpos_] == '\n') ++cpos_;
            if (cpos_ < buf_.size()) content_started_ = true;
        }
        if (content_started_ && buf_.size() > cpos_) {
            *content_out = buf_.substr(cpos_);
            cpos_ = buf_.size();
        }
    }

    // Flush the tail when the stream ends without a marker.
    void finish(std::string* reasoning_out, std::string* content_out) {
        if (!split_) {
            size_t end = buf_.size();
            while (end > 0 && buf_[end - 1] == '\n') --end;
            if (end > rpos_) {
                *reasoning_out = buf_.substr(rpos_, end - rpos_);
                rpos_ = end;
            }
            return;
        }
        feed("", reasoning_out, content_out);
    }

  private:
    static constexpr const char* kMarker = "</think>";
    static constexpr size_t kMarkerLen = 8;
    std::string buf_;
    size_t rpos_ = 0, r_end_ = 0, cpos_ = 0;
    bool split_ = false, content_started_ = false;
};

// Semantic boundary hints (SNAPS): token positions where a future
// re-rendered prompt is likely to share a prefix with this one — the end of
// every earlier message and the end of the last message's content (just
// before its closing tag). The engine keeps cheap RAM checkpoints at these
// cuts, which is what makes edit-and-resend and multi-turn retokenization
// wobble hit the cache instead of re-prefilling. A cut is only sent when its
// character boundary is an exact token start in the final encoding, so every
// hint is a real token boundary of this prompt.
std::vector<long long> compute_snap_cuts(const json& messages,
                                         chat_template::Options opts,
                                         const std::string& text,
                                         size_t n_ids) {
    std::vector<size_t> cut_chars;
    opts.add_generation_prompt = nullptr;  // prefix renders stop at message ends
    for (size_t k = 1; k < messages.size(); k++) {
        json prefix = json::array();
        for (size_t i = 0; i < k; i++) prefix.push_back(messages[i]);
        chat_template::RenderResult pr =
            chat_template::render_chat_template(&prefix, opts);
        // Only exact char prefixes of the full render are stable boundaries
        // (tool-role rendering depends on the NEXT message, so not every
        // message end qualifies).
        if (pr.ok && pr.text.size() < text.size() &&
            text.compare(0, pr.text.size(), pr.text) == 0)
            cut_chars.push_back(pr.text.size());
    }
    // End of the last message's content = just before its closing <|im_end|>.
    chat_template::RenderResult ng =
        chat_template::render_chat_template(&messages, opts);
    if (ng.ok && ng.text.size() <= text.size() &&
        text.compare(0, ng.text.size(), ng.text) == 0) {
        const size_t tail = ng.text.rfind("<|im_end|>");
        if (tail != std::string::npos) cut_chars.push_back(tail);
    }
    std::sort(cut_chars.begin(), cut_chars.end());
    cut_chars.erase(std::unique(cut_chars.begin(), cut_chars.end()),
                    cut_chars.end());
    std::vector<long long> cuts;
    if (cut_chars.empty()) return cuts;
    const std::vector<gdec::TokenSpan> spans = g_tok.encode_with_offsets(text);
    size_t ti = 0;
    for (size_t cc : cut_chars) {
        while (ti < spans.size() && spans[ti].start < cc) ti++;
        if (ti < spans.size() && spans[ti].start == cc && ti > 0 && ti < n_ids)
            cuts.push_back((long long)ti);
    }
    if (cuts.size() > 8) cuts.erase(cuts.begin(), cuts.end() - 8);
    return cuts;
}

// POST /v1/chat/completions
void handle_chat(const http::Request& q, http::Response* r, http::Stream* st) {
    const json body = parse_body(q);
    if (!body.contains("messages")) http::fail(400, "No messages provided.");
    if (!body["messages"].is_array() || body["messages"].empty())
        http::fail(400, "No messages provided.");
    json messages = body["messages"];
    std::string normalize_error;
    if (!toolparse::normalize_messages(&messages, &normalize_error))
        http::fail(400, normalize_error);
    std::vector<vision::Frame> frames;
    if (!vision::prepare_messages(messages, &frames, &normalize_error))
        http::fail(400, normalize_error);

    ToolSetup tool_setup = parse_tool_setup(body);
    chat_template::Options opts;
    if (tool_setup.enabled()) opts.tools = &tool_setup.tools;
    static const json kTrue = true;
    opts.add_generation_prompt = &kTrue;
    json normalized_effort;
    if (body.contains("reasoning_effort"))
        opts.reasoning_effort =
            normalize_reasoning_effort(&body["reasoning_effort"], &normalized_effort);
    require_optional_boolean(body, "enable_thinking");
    require_optional_boolean(body, "preserve_thinking");
    require_optional_boolean(body, "add_vision_id");
    const bool thinking_enabled = bool_field(body, "enable_thinking", true);
    if (body.contains("enable_thinking") && !body["enable_thinking"].is_null())
        opts.enable_thinking = &body["enable_thinking"];
    if (body.contains("preserve_thinking") && !body["preserve_thinking"].is_null())
        opts.preserve_thinking = &body["preserve_thinking"];
    if (body.contains("add_vision_id")) opts.add_vision_id = &body["add_vision_id"];

    chat_template::RenderResult rr = chat_template::render_chat_template(&messages, opts);
    if (!rr.ok) http::fail(400, rr.error);
    rr.text += tool_setup.choice.prompt_suffix(thinking_enabled);

    GenSpec spec;
    if (!vision::encode_prompt(rr.text, g_tok, frames, &spec.ids, &normalize_error))
        http::fail(400, normalize_error);
    for (auto& frame : frames) {
        spec.mrope_grids.push_back(frame.grid);
        spec.patches.push_back(std::move(frame.patches));
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    // Text-only: offsets across image pads are not meaningful, so vision
    // requests go without hints (plain `cont` reuse still applies).
    if (frames.empty())
        spec.snaps = compute_snap_cuts(messages, opts, rr.text, spec.ids.size());
    apply_sampling(body, &spec);

    const bool stream = bool_field(body, "stream", false) && st != nullptr;
    const bool include_usage = stream_wants_usage(body);
    if (stream && spec.logprobs)
        http::fail(400, "logprobs with stream=true is not supported in this build");
    const std::string id = make_id("chatcmpl-");
    const long long created = now_unix();
    auto chunk = [&](json delta, const char* finish) {
        json j;
        j["id"] = id;
        j["created"] = created;
        j["model"] = g_cfg.model;
        j["object"] = "chat.completion.chunk";
        j["choices"] = json::array(
            {json{{"index", 0},
                  {"finish_reason", finish != nullptr ? json(finish) : json(nullptr)},
                  {"delta", delta}}});
        return j;
    };
    auto text_chunk = [&](const char* field, const std::string& value) {
        json delta = json::object();
        delta[field] = value;
        return chunk(std::move(delta), nullptr);
    };

    if (!stream) {
        GenOutcome o = run_generation(spec, nullptr, id);
        std::string reasoning, answer;
        if (tool_setup.choice.forced())
            answer = o.text;
        else
            split_reasoning(o.text, thinking_enabled, &reasoning, &answer);
        toolparse::StreamParser parser(
            tool_setup.tools, []() { return make_id("call_"); }, tool_setup.enabled());
        parser.feed(tool_setup.choice.parser_prefix());
        parser.feed(answer);
        parser.finish();
        json msg;
        msg["role"] = "assistant";
        msg["content"] = parser.content();
        msg["reasoning_content"] = reasoning;
        if (!parser.calls().empty()) {
            msg["tool_calls"] = json::array();
            for (const auto& call : parser.calls()) {
                msg["tool_calls"].push_back(
                    json{{"id", call.id},
                         {"type", "function"},
                         {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
            }
        }
        json j;
        j["id"] = id;
        j["created"] = created;
        j["model"] = g_cfg.model;
        j["object"] = "chat.completion";
        const std::string finish = parser.calls().empty() ? o.reason : "tool_calls";
        json choice{{"index", 0}, {"finish_reason", finish}, {"message", msg}};
        if (spec.logprobs) choice["logprobs"] = logprobs_json(o.logprobs);
        j["choices"] = json::array({std::move(choice)});
        j["usage"] = usage_json(o, spec.ids);
        r->body = json_py::dumps(j, /*spaced=*/false);
        return;
    }

    r->sse = true;
    ThinkSplitter splitter(thinking_enabled);
    toolparse::StreamParser parser(
        tool_setup.tools, []() { return make_id("call_"); }, tool_setup.enabled());
    auto dispatch_tool_events = [&](const std::vector<toolparse::Event>& events) {
        for (const auto& event : events) {
            if (event.type == toolparse::EventType::Content) {
                if (!send_frame(st, text_chunk("content", event.data))) return false;
            } else if (event.type == toolparse::EventType::CallStart) {
                json tc;
                tc["index"] = event.index;
                tc["id"] = event.id;
                tc["type"] = "function";
                tc["function"] = {{"name", event.name}, {"arguments", ""}};
                json delta = json::object();
                delta["tool_calls"] = json::array({std::move(tc)});
                if (!send_frame(st, chunk(std::move(delta), nullptr))) return false;
            } else if (event.type == toolparse::EventType::ArgumentsDelta) {
                json tc;
                tc["index"] = event.index;
                tc["function"] = {{"arguments", event.data}};
                json delta = json::object();
                delta["tool_calls"] = json::array({std::move(tc)});
                if (!send_frame(st, chunk(std::move(delta), nullptr))) return false;
            }
        }
        return true;
    };
    if (!dispatch_tool_events(parser.feed(tool_setup.choice.parser_prefix()))) return;
    GenOutcome o = run_generation(
        spec,
        [&](const std::string& d) {
            if (st->disconnected()) return false;
            if (tool_setup.choice.forced()) return dispatch_tool_events(parser.feed(d));
            std::string rp, ct;
            splitter.feed(d, &rp, &ct);
            if (!rp.empty() && !send_frame(st, text_chunk("reasoning_content", rp)))
                return false;
            return dispatch_tool_events(parser.feed(ct));
        },
        id);
    if (o.client_gone) return;
    if (!tool_setup.choice.forced()) {
        std::string rp, ct;
        splitter.finish(&rp, &ct);
        if (!rp.empty() && !send_frame(st, text_chunk("reasoning_content", rp))) return;
        if (!dispatch_tool_events(parser.feed(ct))) return;
    }
    if (!dispatch_tool_events(parser.finish())) return;
    const std::string finish = parser.calls().empty() ? o.reason : "tool_calls";
    send_frame(st, chunk(json::object(), finish.c_str()));
    if (include_usage) {
        json u;
        u["id"] = id;
        u["created"] = created;
        u["model"] = g_cfg.model;
        u["object"] = "chat.completion.chunk";
        u["choices"] = json::array();
        u["usage"] = usage_json(o, spec.ids);
        send_frame(st, u);
    }
    // The terminator is a bare payload, not JSON: send_event() wraps it as
    // `data: [DONE]`. Passing it through a `const json&` helper would quote it
    // into `data: "[DONE]"`, which streaming clients reject at end-of-stream.
    st->send_event("[DONE]");
}

// POST /v1/responses -- stateless text and function-call subset.
void handle_responses(const http::Request& q, http::Response* r, http::Stream* st) {
    const json body = parse_body(q);
    json messages = normalize_responses_input(body);
    std::vector<vision::Frame> frames;
    std::string vision_error;
    if (!vision::prepare_messages(messages, &frames, &vision_error))
        http::fail(400, vision_error);
    const bool stream = bool_field(body, "stream", false) && st != nullptr;
    const bool parallel_tool_calls = bool_field(body, "parallel_tool_calls", true);
    require_optional_boolean(body, "enable_thinking");
    require_optional_boolean(body, "preserve_thinking");
    const bool thinking_enabled = bool_field(body, "enable_thinking", true);

    ToolSetup tool_setup = parse_tool_setup(body);
    chat_template::Options opts;
    static const json kTrue = true;
    opts.add_generation_prompt = &kTrue;
    if (tool_setup.enabled()) opts.tools = &tool_setup.tools;
    if (body.contains("enable_thinking") && !body["enable_thinking"].is_null())
        opts.enable_thinking = &body["enable_thinking"];
    if (body.contains("preserve_thinking") && !body["preserve_thinking"].is_null())
        opts.preserve_thinking = &body["preserve_thinking"];

    json normalized_effort;
    if (body.contains("reasoning") && !body["reasoning"].is_null()) {
        if (!body["reasoning"].is_object()) http::fail(400, "reasoning must be an object");
        const json& reasoning = body["reasoning"];
        if (reasoning.contains("effort"))
            opts.reasoning_effort =
                normalize_reasoning_effort(&reasoning["effort"], &normalized_effort);
        for (const char* key : {"summary", "generate_summary"}) {
            if (reasoning.contains(key) && !reasoning[key].is_null() &&
                !reasoning[key].is_string())
                http::fail(400, std::string("reasoning.") + key + " must be a string");
        }
    }

    chat_template::RenderResult rr = chat_template::render_chat_template(&messages, opts);
    if (!rr.ok) http::fail(400, rr.error);
    rr.text += tool_setup.choice.prompt_suffix(thinking_enabled);

    GenSpec spec;
    if (!vision::encode_prompt(rr.text, g_tok, frames, &spec.ids, &vision_error))
        http::fail(400, vision_error);
    for (auto& frame : frames) {
        spec.mrope_grids.push_back(frame.grid);
        spec.patches.push_back(std::move(frame.patches));
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    // Text-only: offsets across image pads are not meaningful, so vision
    // requests go without hints (plain `cont` reuse still applies).
    if (frames.empty())
        spec.snaps = compute_snap_cuts(messages, opts, rr.text, spec.ids.size());
    apply_sampling(body, &spec);

    const std::string id = make_id("resp_");
    const long long created = now_unix();
    auto output_part = [](const std::string& text) {
        return json{{"type", "output_text"},
                    {"text", text},
                    {"annotations", json::array()},
                    {"logprobs", json::array()}};
    };
    auto message_item = [&](const std::string& item_id, const std::string& status,
                            const std::string& text) {
        return json{{"id", item_id},
                    {"type", "message"},
                    {"status", status},
                    {"role", "assistant"},
                    {"content", json::array({output_part(text)})}};
    };
    auto function_item = [](const std::string& item_id, const std::string& call_id,
                            const std::string& status, const std::string& name,
                            const std::string& arguments) {
        return json{{"id", item_id},
                    {"type", "function_call"},
                    {"status", status},
                    {"call_id", call_id},
                    {"name", name},
                    {"arguments", arguments}};
    };
    auto response_object = [&](const std::string& status, const json& output,
                               const GenOutcome* outcome) {
        json response;
        response["id"] = id;
        response["object"] = "response";
        response["created_at"] = created;
        response["status"] = status;
        response["model"] = g_cfg.model;
        response["output"] = output;
        response["parallel_tool_calls"] = parallel_tool_calls;
        response["max_output_tokens"] =
            spec.max_tokens > 0 ? json(spec.max_tokens) : json(nullptr);
        if (outcome == nullptr) {
            response["usage"] = nullptr;
        } else {
            const long long pt = outcome->n_prompt > 0
                                     ? outcome->n_prompt
                                     : static_cast<long long>(spec.ids.size());
            response["usage"] = {{"input_tokens", pt},
                                 {"output_tokens", outcome->n_gen},
                                 {"total_tokens", pt + outcome->n_gen}};
            if (status == "incomplete")
                response["incomplete_details"] = {{"reason", "max_output_tokens"}};
        }
        return response;
    };

    if (!stream) {
        GenOutcome o = run_generation(spec, nullptr, id);
        std::string reasoning, answer;
        if (tool_setup.choice.forced())
            answer = o.text;
        else
            split_reasoning(o.text, thinking_enabled, &reasoning, &answer);
        toolparse::StreamParser parser(
            tool_setup.tools, []() { return make_id("call_"); }, tool_setup.enabled());
        parser.feed(tool_setup.choice.parser_prefix());
        parser.feed(answer);
        parser.finish();

        const bool incomplete = o.reason == "length" && parser.calls().empty();
        json output = json::array();
        if (!parser.content().empty() || (!incomplete && parser.calls().empty())) {
            output.push_back(message_item(make_id("msg_"),
                                          incomplete ? "incomplete" : "completed",
                                          parser.content()));
        }
        for (const auto& call : parser.calls()) {
            output.push_back(function_item(make_id("fc_"), call.id, "completed",
                                           call.name, call.arguments));
        }
        json response = response_object(incomplete ? "incomplete" : "completed", output, &o);
        r->body = json_py::dumps(response, /*spaced=*/false);
        return;
    }

    r->sse = true;
    long long sequence = 0;
    bool created_sent = false;
    int next_output_index = 0;
    json output_slots = json::array();
    auto put_output = [&](int index, json item) {
        while (output_slots.size() <= static_cast<size_t>(index))
            output_slots.push_back(nullptr);
        output_slots[index] = std::move(item);
    };
    auto event = [&](const char* type, json fields) {
        json e;
        e["type"] = type;
        e["sequence_number"] = sequence++;
        for (auto it = fields.begin(); it != fields.end(); ++it) e[it.key()] = it.value();
        return send_frame(st, e);
    };
    auto ensure_created = [&]() {
        if (created_sent) return true;
        created_sent = true;
        return event("response.created",
                     json{{"response", response_object("in_progress", json::array(), nullptr)}});
    };

    struct MessageState {
        bool open = false;
        int output_index = -1;
        std::string id;
        std::string content;
    } message;
    auto start_message = [&]() {
        if (message.open) return true;
        if (!ensure_created()) return false;
        message.open = true;
        message.output_index = next_output_index++;
        message.id = make_id("msg_");
        message.content.clear();
        if (!event("response.output_item.added",
                   json{{"output_index", message.output_index},
                        {"item", message_item(message.id, "in_progress", "")}}))
            return false;
        return event("response.content_part.added",
                     json{{"item_id", message.id},
                          {"output_index", message.output_index},
                          {"content_index", 0},
                          {"part", output_part("")}});
    };
    auto finish_message = [&]() {
        if (!message.open) return true;
        if (!event("response.output_text.done",
                   json{{"item_id", message.id},
                        {"output_index", message.output_index},
                        {"content_index", 0},
                        {"text", message.content},
                        {"logprobs", json::array()}}))
            return false;
        if (!event("response.content_part.done",
                   json{{"item_id", message.id},
                        {"output_index", message.output_index},
                        {"content_index", 0},
                        {"part", output_part(message.content)}}))
            return false;
        json item = message_item(message.id, "completed", message.content);
        if (!event("response.output_item.done",
                   json{{"output_index", message.output_index}, {"item", item}}))
            return false;
        put_output(message.output_index, std::move(item));
        message.open = false;
        return true;
    };
    auto emit_text = [&](const std::string& delta) {
        if (delta.empty()) return true;
        if (!start_message()) return false;
        message.content += delta;
        return event("response.output_text.delta",
                     json{{"item_id", message.id},
                          {"output_index", message.output_index},
                          {"content_index", 0},
                          {"delta", delta},
                          {"logprobs", json::array()}});
    };

    struct CallState {
        bool open = false;
        int output_index = -1;
        std::string item_id;
        std::string call_id;
        std::string name;
        std::string arguments;
    };
    std::vector<CallState> call_states;
    auto dispatch_tool_events = [&](const std::vector<toolparse::Event>& events) {
        for (const auto& parsed : events) {
            if (parsed.type == toolparse::EventType::Content) {
                if (!emit_text(parsed.data)) return false;
                continue;
            }
            if (parsed.index >= call_states.size()) call_states.resize(parsed.index + 1);
            CallState& call = call_states[parsed.index];
            if (parsed.type == toolparse::EventType::CallStart) {
                if (!finish_message() || !ensure_created()) return false;
                call.open = true;
                call.output_index = next_output_index++;
                call.item_id = make_id("fc_");
                call.call_id = parsed.id;
                call.name = parsed.name;
                json item = function_item(call.item_id, call.call_id, "in_progress",
                                          call.name, "");
                if (!event("response.output_item.added",
                           json{{"output_index", call.output_index}, {"item", item}}))
                    return false;
            } else if (parsed.type == toolparse::EventType::ArgumentsDelta) {
                call.arguments += parsed.data;
                if (!event("response.function_call_arguments.delta",
                           json{{"item_id", call.item_id},
                                {"output_index", call.output_index},
                                {"delta", parsed.data}}))
                    return false;
            } else if (parsed.type == toolparse::EventType::CallEnd) {
                call.arguments = parsed.data;
                if (!event("response.function_call_arguments.done",
                           json{{"item_id", call.item_id},
                                {"output_index", call.output_index},
                                {"arguments", call.arguments},
                                {"name", call.name}}))
                    return false;
                json item = function_item(call.item_id, call.call_id, "completed",
                                          call.name, call.arguments);
                if (!event("response.output_item.done",
                           json{{"output_index", call.output_index}, {"item", item}}))
                    return false;
                put_output(call.output_index, std::move(item));
                call.open = false;
            }
        }
        return true;
    };

    ThinkSplitter splitter(thinking_enabled);
    toolparse::StreamParser parser(
        tool_setup.tools, []() { return make_id("call_"); }, tool_setup.enabled());
    if (!dispatch_tool_events(parser.feed(tool_setup.choice.parser_prefix()))) return;
    GenOutcome o = run_generation(
        spec,
        [&](const std::string& delta) {
            if (st->disconnected()) return false;
            if (tool_setup.choice.forced()) return dispatch_tool_events(parser.feed(delta));
            std::string reasoning_delta, content_delta;
            splitter.feed(delta, &reasoning_delta, &content_delta);
            return dispatch_tool_events(parser.feed(content_delta));
        },
        id);
    if (o.client_gone) return;
    if (!tool_setup.choice.forced()) {
        std::string reasoning_delta, content_delta;
        splitter.finish(&reasoning_delta, &content_delta);
        if (!dispatch_tool_events(parser.feed(content_delta))) return;
    }
    if (!dispatch_tool_events(parser.finish())) return;
    if (!finish_message() || !ensure_created()) return;

    bool partial_call = parser.has_partial_call();
    for (CallState& call : call_states) {
        if (!call.open) continue;
        partial_call = true;
        json item = function_item(call.item_id, call.call_id, "incomplete",
                                  call.name, call.arguments);
        if (!event("response.output_item.done",
                   json{{"output_index", call.output_index}, {"item", item}}))
            return;
        put_output(call.output_index, std::move(item));
        call.open = false;
    }

    const bool incomplete = o.reason == "length" || partial_call;
    if (output_slots.empty() && !incomplete) {
        if (!start_message() || !finish_message()) return;
    }
    json final_response =
        response_object(incomplete ? "incomplete" : "completed", output_slots, &o);
    event(incomplete ? "response.incomplete" : "response.completed",
          json{{"response", std::move(final_response)}});
}

// -------------------------------------------------------------------- main --

void probe_engine() {
    std::string err, line;
    // Startup must not stall for the generation budget: if another front-end
    // currently owns the engine's single connection, fall back to the
    // configured defaults instead of blocking the listen socket for minutes.
    g_eng.set_timeouts(5.0, 5.0);
    if (!g_eng.connect(g_cfg.engine_addr, &err)) {
        fprintf(stderr, "gdec-api: engine connect failed: %s\n", err.c_str());
        g_eng.set_timeouts(1800.0, 300.0);
        return;
    }
    if (g_eng.info(&line, &err)) {
        // I mtp draft_head ctx spec_rows default drafter_weights dflash2
        //   cache_mb cache_align kv_slots slot_ctx cache_mode sampling
        std::vector<long long> f;
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;  // "I"
        long long v;
        while (ss >> v) f.push_back(v);
        if (f.size() >= 3) {
            g_cfg.context = (int)f[2];
        }
        fprintf(stderr, "gdec-api: engine INFO: %s\n", line.c_str());
    } else {
        fprintf(stderr, "gdec-api: engine INFO unavailable (%s); continuing with "
                        "ctx=%d\n", err.c_str(), g_cfg.context);
    }
    g_eng.set_timeouts(1800.0, 300.0);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    if (!wsa_init()) {
        std::fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }
#endif
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--tokenizer") g_cfg.tokenizer_dir = next();
        else if (a == "--engine") g_cfg.engine_addr = next();
        else if (a == "--port") {
            const std::string p = next();
            g_cfg.listen = g_cfg.listen.substr(0, g_cfg.listen.rfind(':') + 1) + p;
        } else if (a == "--host") {
            const std::string h = next();
            g_cfg.listen = h + ":" + g_cfg.listen.substr(g_cfg.listen.rfind(':') + 1);
        } else if (a == "--context") g_cfg.context = std::atoi(next().c_str());
        else if (a == "--model") g_cfg.model = next();
        else if (a == "--listen") g_cfg.listen = next();
        else {
            fprintf(stderr,
                    "usage: %s [--tokenizer DIR] [--engine H:P] [--listen H:P] "
                    "[--port N] [--host H] [--context N] [--model NAME]\n",
                    argv[0]);
            return 2;
        }
    }

    std::string err;
    if (!g_tok.load(g_cfg.tokenizer_dir, &err)) {
        fprintf(stderr, "gdec-api: tokenizer: %s\n", err.c_str());
        return 1;
    }
    probe_engine();

    http::Server srv;
    if (!srv.listen(g_cfg.listen, &err)) {
        fprintf(stderr, "gdec-api: listen: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "gdec-api: listening on :%d model=%s ctx=%d slots=1\n", srv.port(),
            g_cfg.model.c_str(), g_cfg.context);

    srv.on("GET", "/v1/models", handle_models);
    srv.on("GET", "/health", handle_health);
    srv.on("GET", "/cache", handle_cache);
    srv.on("POST", "/v1/completions", handle_completions);
    srv.on("POST", "/v1/chat/completions", handle_chat);
    srv.on("POST", "/v1/responses", handle_responses);

    srv.run();
    return 0;
}
