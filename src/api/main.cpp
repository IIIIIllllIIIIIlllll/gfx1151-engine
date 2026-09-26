// main.cpp — gdec-api: C++ OpenAI-compatible front-end.
//
// Independently implements the OpenAI-compatible front-end from public API
// schemas and project-owned black-box conformance fixtures. The tokenizer and
// chat template are verified by tools/tok_ab.py and tools/template_ab.py.
//
// KV reuse: prompts reuse the exact ids of earlier requests for their
// byte-identical prefix (TokenCache), so a history holding a generated reply
// stays a token prefix of what the engine holds. Chat/responses requests also
// send SNAPS hints (semantic message-boundary token cuts before the first
// image, see compute_snap_cuts); the engine keeps RAM checkpoints (rckpt) at
// those cuts, so edit-and-resend hits the cache instead of re-prefilling. Raw
// completions send no hints — the engine's `cont` strict-prefix reuse and the
// SSD kvsnap tier still apply.
#include <atomic>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
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
    // Server-side sampling/thinking overrides (see apply_overrides). Relative
    // to the working directory, which every launcher sets to the repo root.
    std::string overrides_path = "data/api-overrides.json";
    std::string admin_key;  // env GDEC_API_ADMIN_KEY; empty = POST open (LAN)
};

Config g_cfg;
gdec::Tokenizer g_tok;
std::mutex g_conn_mtx;  // serialize (re)connects
std::atomic<long long> g_req_seq{0};

// The engine runs INFO kv_slots sequences at once (GDEC_PARALLEL, sharing
// one KV pool; 1 = batch-1) and takes one GEN per connection.  Up to g_slots
// generations are in flight here, each on its own pooled engine connection;
// the rest wait in this front-end queue until a slot is released.  Control
// queries (MEM/CSTAT) use g_ctl and never wait for a generation slot.
int g_slots = 1;
std::vector<std::unique_ptr<gdec::EngineClient>> g_pool;  // g_slots clients
std::vector<gdec::EngineClient*> g_pool_free;             // under g_slot_mtx
gdec::EngineClient g_ctl;
std::mutex g_slot_mtx;
std::condition_variable g_slot_cv;
int g_in_flight = 0;
int g_queued = 0;
std::chrono::steady_clock::time_point g_busy_since;  // idle -> busy edge

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
const char* kEngineRejectedPar =
    "the engine rejected this request. Check /health for what this build "
    "supports, and that the prompt fits the context; the context is shared by "
    "concurrent requests, so it may also be full right now (retry later).";
const char* kEngineAborted =
    "the engine aborted this request: concurrent requests filled the shared "
    "context (earlier requests keep it). Retry when the server is less busy.";

// ------------------------------------------------------------- slot guard --

class SlotGuard {
  public:
    explicit SlotGuard(const std::function<bool()>& on_wait = {}) {
        std::unique_lock<std::mutex> lk(g_slot_mtx);
        const bool queued = g_in_flight >= g_slots;
        if (queued) ++g_queued;
        while (g_in_flight >= g_slots) {
            if (!on_wait) {
                g_slot_cv.wait(lk, [] { return g_in_flight < g_slots; });
                break;
            }
            // Do not hold the slot mutex while writing to the client.  The
            // callback also lets a disconnected stream leave the queue.
            lk.unlock();
            const bool keep_waiting = on_wait();
            lk.lock();
            if (!keep_waiting) {
                --g_queued;
                g_slot_cv.notify_all();
                return;
            }
            if (g_in_flight >= g_slots) g_slot_cv.wait_for(lk, std::chrono::seconds(1));
        }
        if (queued) --g_queued;
        if (g_in_flight++ == 0) g_busy_since = std::chrono::steady_clock::now();
        eng_ = g_pool_free.back();  // g_in_flight < g_slots: one is free
        g_pool_free.pop_back();
        acquired_ = true;
    }
    ~SlotGuard() {
        if (!acquired_) return;
        std::lock_guard<std::mutex> lk(g_slot_mtx);
        g_pool_free.push_back(eng_);
        --g_in_flight;
        g_slot_cv.notify_all();
    }
    bool acquired() const { return acquired_; }
    gdec::EngineClient& engine() const { return *eng_; }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;

  private:
    bool acquired_ = false;
    gdec::EngineClient* eng_ = nullptr;
};

bool engine_ready(gdec::EngineClient& eng, std::string* err) {
    std::lock_guard<std::mutex> lk(g_conn_mtx);
    if (eng.connected()) return true;
    return eng.connect(g_cfg.engine_addr, err);
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

// ------------------------------------------------------ transcript cache ---
// The engine reuses KV only for an exact token prefix of what it holds, but
// generated ids are not canonical BPE (the model may write "/meta" as
// 14+5317 where encode() gives 66862). Re-encoding a history that contains a
// long reply therefore diverges inside the reply and the next turn re-prefills
// everything. So every request's sent + generated ids are remembered, and a
// prompt whose text starts with the same bytes reuses those ids; only the rest
// is encoded. GDEC_API_TOKCACHE=<tokens> sizes the cache (0 = off).

constexpr int kImagePadId = 248056, kVideoPadId = 248057;
constexpr const char* kVisionPadSpan = "<|vision_start|><|image_pad|><|vision_end|>";

bool is_pad_id(int id) { return id == kImagePadId || id == kVideoPadId; }

class TokenCache {
  public:
    void set_capacity(size_t tokens) { cap_tokens_ = tokens; }
    bool enabled() const { return cap_tokens_ > 0; }

    // Remember what the engine saw for one request (prompt + generated ids).
    void record(std::vector<int> ids) {
        if (!enabled() || ids.empty() || ids.size() > cap_tokens_) return;
        auto t = build_entry(std::move(ids));
        {
            std::lock_guard<std::mutex> lk(mtx_);
            insert(std::move(t));
        }
        autosave();
    }

    // Persistence (GDEC_API_TOKCACHE_FILE, default data/tcache.bin, "" = off).
    // Layout, little-endian: magic "GDTC1\0\0\0" | u32 version(1) | u32 vocab
    // | u64 count | per entry u64 n_ids + n_ids*i32. Byte offsets and pad runs
    // are rebuilt from the ids at load, so the only compatibility guard is the
    // vocab size. Written tmp+rename; a torn file just means an empty cache.
    void set_file(std::string f) { file_ = std::move(f); }
    void set_save_interval(int s) { save_s_ = s; }

    void save_file() const {
        if (file_.empty()) return;
        std::vector<std::vector<int>> snap;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            snap.reserve(entries_.size());
            for (const auto& e : entries_) snap.push_back(e->ids);
        }
        const std::string tmp = file_ + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) return;
        const char magic[8] = "GDTC1";
        const uint32_t ver = 1, vocab = (uint32_t)g_tok.vocab_size();
        const uint64_t count = snap.size();
        bool ok = fwrite(magic, 1, 8, f) == 8 && fwrite(&ver, 4, 1, f) == 1 &&
                  fwrite(&vocab, 4, 1, f) == 1 && fwrite(&count, 8, 1, f) == 1;
        for (const auto& ids : snap) {
            const uint64_t n = ids.size();
            if (ok) ok = fwrite(&n, 8, 1, f) == 1;
            if (ok) ok = fwrite(ids.data(), 4, n, f) == n;
        }
        ok = fclose(f) == 0 && ok;
        if (!ok) {
            fprintf(stderr, "gdec-api: tcache: save %s failed: %s\n", tmp.c_str(),
                    strerror(errno));
            remove(tmp.c_str());
            return;
        }
        remove(file_.c_str());  // Windows rename(2) fails when the target exists
        rename(tmp.c_str(), file_.c_str());
    }

    void load_file() {
        if (file_.empty() || !enabled()) return;
        FILE* f = fopen(file_.c_str(), "rb");
        if (!f) return;  // first boot: no cache yet
        char magic[8];
        uint32_t ver = 0, vocab = 0;
        uint64_t count = 0;
        bool ok = fread(magic, 1, 8, f) == 8 && !memcmp(magic, "GDTC1\0\0\0", 8) &&
                  fread(&ver, 4, 1, f) == 1 && ver == 1 &&
                  fread(&vocab, 4, 1, f) == 1 && (int)vocab == g_tok.vocab_size() &&
                  fread(&count, 8, 1, f) == 1 && count <= kMaxEntries * 4;
        size_t entries = 0, tokens = 0;
        std::vector<std::shared_ptr<Entry>> loaded;
        for (uint64_t i = 0; ok && i < count; i++) {
            uint64_t n = 0;
            std::vector<int> ids;
            ok = fread(&n, 8, 1, f) == 1 && n > 0 && n <= cap_tokens_;
            if (!ok) break;
            ids.resize(n);
            ok = fread(ids.data(), 4, n, f) == n;
            if (!ok) break;
            loaded.push_back(build_entry(std::move(ids)));
            tokens += n;
        }
        fclose(f);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (auto& t : loaded) insert(std::move(t));
            entries = entries_.size();
        }
        if (!ok)
            fprintf(stderr, "gdec-api: tcache: %s truncated/mismatched, loaded what parsed\n",
                    file_.c_str());
        if (entries)
            fprintf(stderr, "gdec-api: tcache: loaded %zu entries (%zu tokens) from %s\n",
                    entries, tokens, file_.c_str());
    }

    struct Hit {
        std::vector<int> ids;        // reused ids
        std::vector<size_t> starts;  // their byte offsets in the text
        size_t bytes = 0;            // text bytes they cover
        size_t images = 0;           // image pad runs among them
    };

    // Longest reusable prefix of `text`. pad_counts = pad tokens per image of
    // this request, in order (an image run is only reused when it matches).
    Hit splice(const std::string& text, const std::vector<int>& pad_counts) {
        Hit best;
        if (!enabled()) return best;
        std::shared_ptr<Entry> pick;
        size_t pick_k = 0;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (const auto& t : entries_) {
                const size_t k = cut(*t, text, pad_counts);
                if (k == 0) continue;
                const size_t b = t->off[k];
                if (!pick || b > pick->off[pick_k] ||
                    (b == pick->off[pick_k] && t->used > pick->used)) {
                    pick = t;
                    pick_k = k;
                }
            }
            if (pick) pick->used = ++clock_;
        }
        if (!pick) return best;
        best.ids.assign(pick->ids.begin(), pick->ids.begin() + pick_k);
        best.starts.assign(pick->off.begin(), pick->off.begin() + pick_k);
        best.bytes = pick->off[pick_k];
        for (const auto& run : pick->runs)
            if (run.first < pick_k) best.images++;
        return best;
    }

  private:
    struct Entry {
        std::vector<int> ids;
        std::string bytes;           // what the ids decode to (a pad run = one pad)
        std::vector<uint32_t> off;   // off[i] = byte offset of ids[i]; off[n] = bytes.size()
        std::vector<std::pair<uint32_t, uint32_t>> runs;  // image pad runs: (first id, count)
        int vstart = -1;
        uint64_t used = 0;
    };

    // Reusable id count of `t` for `text` (0 = none).
    size_t cut(const Entry& t, const std::string& text, const std::vector<int>& pad_counts) const {
        const size_t n = t.ids.size();
        const size_t lim = std::min(t.bytes.size(), text.size());
        const size_t same = static_cast<size_t>(
            std::mismatch(t.bytes.begin(), t.bytes.begin() + lim, text.begin()).first -
            t.bytes.begin());
        // Last id boundary inside the equal bytes.
        size_t k = static_cast<size_t>(
            std::upper_bound(t.off.begin(), t.off.end(), static_cast<uint32_t>(same)) -
            t.off.begin()) - 1;
        // The whole transcript matched: the engine's `cont` resumes right at its
        // end, so cut there (unless the text continues inside a UTF-8 char).
        bool whole = k == n && !(same < text.size() && (uint8_t(text[same]) & 0xC0) == 0x80);
        for (;;) {
            // Otherwise the engine can only resume from a checkpoint below k
            // anyway, so cut right after an added token: the rest of the text
            // then encodes exactly as encode() would encode it in full. Never
            // inside or right after an image placeholder (its span must stay whole).
            if (!whole) {
                while (k > 0 && !(g_tok.is_added_token(t.ids[k - 1]) &&
                                  t.ids[k - 1] != t.vstart && !is_pad_id(t.ids[k - 1])))
                    k--;
            } else if (k > 0 && (t.ids[k - 1] == t.vstart || is_pad_id(t.ids[k - 1]))) {
                whole = false;
                k--;
                continue;
            }
            // Every image run inside the prefix must match this request's image.
            size_t bad = SIZE_MAX;
            for (size_t j = 0; j < t.runs.size() && t.runs[j].first < k; ++j) {
                if (j >= pad_counts.size() ||
                    static_cast<int>(t.runs[j].second) != pad_counts[j]) {
                    bad = t.runs[j].first;
                    break;
                }
            }
            if (bad == SIZE_MAX) return k;
            whole = false;
            k = bad;  // before the pad; the loop steps back past <|vision_start|>
        }
    }

    // Decode + offset bookkeeping of one recorded transcript.
    static std::shared_ptr<Entry> build_entry(std::vector<int> ids) {
        auto t = std::make_shared<Entry>();
        t->off.reserve(ids.size() + 1);
        const int vstart = g_tok.added_token_id("<|vision_start|>");
        for (size_t i = 0; i < ids.size(); ++i) {
            t->off.push_back(static_cast<uint32_t>(t->bytes.size()));
            // The prompt text holds one <|image_pad|> per image; the ids hold
            // pad_tokens() copies. Only the first copy of a run has bytes.
            const bool run_cont = i > 0 && is_pad_id(ids[i]) && is_pad_id(ids[i - 1]);
            if (is_pad_id(ids[i]) && !run_cont) {
                t->runs.push_back({static_cast<uint32_t>(i), 0});
            }
            if (is_pad_id(ids[i])) {
                t->runs.back().second++;
                if (run_cont) continue;
            }
            t->bytes += g_tok.decode_bytes(std::vector<int>{ids[i]}, /*skip_special=*/false);
        }
        t->off.push_back(static_cast<uint32_t>(t->bytes.size()));
        t->vstart = vstart;
        t->ids = std::move(ids);
        return t;
    }

    // Caller holds mtx_.
    void insert(std::shared_ptr<Entry> t) {
        t->used = ++clock_;
        // A multi-turn conversation re-sends its own history: the new
        // transcript supersedes every entry that is a prefix of it.
        for (auto it = entries_.begin(); it != entries_.end();) {
            const auto& e = (*it)->ids;
            if (e.size() <= t->ids.size() && std::equal(e.begin(), e.end(), t->ids.begin())) {
                total_ -= e.size();
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
        total_ += t->ids.size();
        entries_.push_back(std::move(t));
        while (entries_.size() > kMaxEntries || total_ > cap_tokens_) {
            auto lru = std::min_element(entries_.begin(), entries_.end(),
                                        [](const auto& a, const auto& b) { return a->used < b->used; });
            total_ -= (*lru)->ids.size();
            entries_.erase(lru);
        }
    }

    // Throttled persist after record(); 0 interval = save on every record.
    void autosave() {
        if (file_.empty()) return;
        const auto now = std::chrono::steady_clock::now();
        auto last = last_save_.load(std::memory_order_relaxed);
        if (save_s_ > 0 && now - last < std::chrono::seconds(save_s_)) return;
        if (!last_save_.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
        save_file();
    }

    static constexpr size_t kMaxEntries = 64;
    mutable std::mutex mtx_;  // mutable: save_file() is const
    std::vector<std::shared_ptr<Entry>> entries_;
    size_t total_ = 0;
    size_t cap_tokens_ = 4u << 20;
    uint64_t clock_ = 0;
    std::string file_;
    int save_s_ = 10;
    std::atomic<std::chrono::steady_clock::time_point> last_save_{
        std::chrono::steady_clock::time_point{}};
};

TokenCache g_tcache;

// Token ids the engine checkpoints its RAM state at mid-generation (CKPT
// hints on GEN). Mid-reply cuts happen right after these added tokens
// (<tool_call>, </think>): the TokenCache slices a next-turn prefix there, so
// a checkpoint at the same spot turns a template-roundtrip mismatch from a
// full reply re-prefill into re-decoding only the tool call + tool result.
// GDEC_CKPT_TOKENS=0 disables; the engine ignores unknown/empty hints.
std::vector<int> g_ckpt_ids;

std::vector<int> default_ckpt_tokens() {
    if (const char* e = std::getenv("GDEC_CKPT_TOKENS"))
        if (!strcmp(e, "0")) return {};
    std::vector<int> ids;
    for (const char* s : {"<tool_call>", "</think>"}) {
        const int id = g_tok.added_token_id(s);
        if (id >= 0) ids.push_back(id);
    }
    if (!ids.empty()) {
        std::string names;
        for (int id : ids) {
            if (!names.empty()) names += ",";
            names += std::to_string(id);
        }
        fprintf(stderr, "gdec-api: ckpt tokens: %s\n", names.c_str());
    }
    return ids;
}

// Encode a rendered prompt, reusing the transcript cache for its prefix.
// frames == nullptr: raw text (no image expansion, like encode()). starts
// receives each id's byte offset in `text`.
bool encode_spliced(const std::string& text, const std::vector<vision::Frame>* frames,
                    std::vector<int>* ids, std::vector<size_t>* starts, std::string* error) {
    std::vector<int> pad_counts;
    if (frames)
        for (const auto& f : *frames) pad_counts.push_back(f.pad_tokens());
    TokenCache::Hit hit = g_tcache.splice(text, pad_counts);
    // The reused text must hold exactly the images the reused ids hold.
    if (hit.bytes > 0) {
        size_t spans = 0;
        const std::string head = text.substr(0, hit.bytes);
        for (size_t at = 0; (at = head.find(kVisionPadSpan, at)) != std::string::npos;
             at += std::strlen(kVisionPadSpan))
            spans++;
        if (spans != hit.images || (!frames && hit.images > 0)) hit = TokenCache::Hit{};
    }
    const std::string tail_text = text.substr(hit.bytes);
    std::vector<int> tail;
    std::vector<size_t> tail_starts;
    if (frames) {
        std::vector<vision::Frame> rest;
        for (size_t i = hit.images; i < frames->size(); ++i) {
            vision::Frame f;
            f.grid = (*frames)[i].grid;
            rest.push_back(f);
        }
        if (!vision::encode_prompt(tail_text, g_tok, rest, &tail, error, &tail_starts))
            return false;
    } else {
        for (const auto& s : g_tok.encode_with_offsets(tail_text)) {
            tail.push_back(s.id);
            tail_starts.push_back(s.start);
        }
    }
    if (!hit.ids.empty())
        fprintf(stderr, "tcache: reused %zu ids (%zu of %zu bytes), encoded %zu\n",
                hit.ids.size(), hit.bytes, text.size(), tail.size());
    *ids = std::move(hit.ids);
    ids->insert(ids->end(), tail.begin(), tail.end());
    *starts = std::move(hit.starts);
    for (size_t s : tail_starts) starts->push_back(s + hit.bytes);
    return true;
}

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
                          const std::string& log_tag,
                          const std::function<bool()>& on_wait = {}) {
    GenOutcome out;
    std::function<bool()> wait_callback;
    if (on_wait) {
        wait_callback = [&]() {
            const bool keep_waiting = on_wait();
            if (!keep_waiting) out.client_gone = true;
            return keep_waiting;
        };
    }
    SlotGuard slot(wait_callback);
    if (!slot.acquired()) {
        out.client_gone = true;
        return out;
    }
    gdec::EngineClient& eng = slot.engine();
    std::string err;
    if (!engine_ready(eng, &err)) {
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
    p.ckpt = g_ckpt_ids;

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

    gdec::GenResult r = eng.generate(p, [&](int tok, float lp) {
        (void)lp;
        ++emitted;
        return emit(detok.push(tok));
    }, wait_callback);
    out.ttft_ms = ttft;
    if (r.transport_ok && r.reason != "error") {
        // Before the response ends, so a client's next turn always sees it.
        std::vector<int> seen = spec.ids;
        seen.insert(seen.end(), r.tokens.begin(), r.tokens.end());
        g_tcache.record(std::move(seen));
    }

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

    if (r.reason == "error") {
        // Concurrent requests share one KV pool; when it runs out the engine
        // fails the later request (the D line carries no reason).  After tokens
        // have streamed that is the only way a request can fail.
        if (emitted > 0) {
            fprintf(stderr, "REQ %lld aborted by the engine after %lld tokens (shared KV "
                            "context full)\n", req_id, emitted);
            http::fail(503, kEngineAborted, "server_error", "server_error");
        }
        http::fail(400, g_slots > 1 ? kEngineRejectedPar : kEngineRejected);
    }
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

// ------------------------------------------------------- server overrides --
// Operator policy for sampling / thinking, applied to the request body before
// any parsing so the normal validation still runs on the result. A client
// therefore cannot switch thinking or send a broken sampler by mistake.
// Each entry is {"mode": "force" | "default", "value": v}:
//   force   -- replaces what the client sent (max_tokens: caps it instead);
//   default -- used only when the client omitted the field.
// Fields without an entry follow the client. Edited from the dashboard via
// GET/POST /admin/overrides and persisted to g_cfg.overrides_path.

struct OverrideField {
    const char* key;
    char kind;  // b = bool, n = number, i = integer, e = reasoning effort
    double lo, hi;
};
const OverrideField kOverrideFields[] = {
    {"enable_thinking", 'b', 0, 0},
    {"preserve_thinking", 'b', 0, 0},
    {"reasoning_effort", 'e', 0, 0},
    {"temperature", 'n', 0.0, 2.0},
    {"top_p", 'n', 0.0, 1.0},
    {"top_k", 'i', 0, 2147483647.0},
    {"min_p", 'n', 0.0, 1.0},
    {"presence_penalty", 'n', -2.0, 2.0},
    {"frequency_penalty", 'n', -2.0, 2.0},
    {"max_tokens", 'i', 1, 2147483647.0},
};

std::mutex g_ovr_mtx;
json g_overrides = json::object();

// Checks a posted table and returns its canonical form (mode "client" and
// null entries dropped, effort aliases resolved like the request path does).
json validate_overrides(const json& in) {
    if (!in.is_object()) http::fail(400, "overrides must be a JSON object");
    json out = json::object();
    for (auto it = in.begin(); it != in.end(); ++it) {
        const std::string& k = it.key();
        const OverrideField* f = nullptr;
        for (const auto& c : kOverrideFields)
            if (k == c.key) f = &c;
        if (f == nullptr) http::fail(400, "unknown override field " + k);
        const json& e = it.value();
        if (e.is_null()) continue;
        if (!e.is_object() || !e.contains("mode") || !e["mode"].is_string())
            http::fail(400, k + ": expected {\"mode\": ..., \"value\": ...}");
        const std::string mode = e["mode"].get<std::string>();
        if (mode == "client") continue;
        if (mode != "force" && mode != "default")
            http::fail(400, k + ": mode must be force, default or client");
        if (!e.contains("value") || e["value"].is_null()) http::fail(400, k + ": value is required");
        json v = e["value"];
        if (f->kind == 'b') {
            if (!v.is_boolean()) http::fail(400, k + " must be a boolean");
        } else if (f->kind == 'e') {
            json storage;
            const json* n = normalize_reasoning_effort(&v, &storage);
            const std::string s = n->get<std::string>();
            if (s != "xhigh" && s != "medium" && s != "low")
                http::fail(400, k + " must be one of xhigh, high, medium, low, minimal");
            v = s;
        } else {
            const bool ok = f->kind == 'i' ? (v.is_number_integer() || v.is_number_unsigned())
                                           : v.is_number();
            const double d = ok ? v.get<double>() : 0.0;
            if (!ok || !std::isfinite(d) || d < f->lo || d > f->hi)
                http::fail(400, k + " must be " + (f->kind == 'i' ? "an integer" : "a number") +
                                    " in " + json_py::py_float(f->lo) + ".." +
                                    json_py::py_float(f->hi));
        }
        out[k] = json{{"mode", mode}, {"value", std::move(v)}};
    }
    return out;
}

bool save_overrides(const json& table, std::string* err) {
    if (g_cfg.overrides_path.empty()) return true;
    const std::string tmp = g_cfg.overrides_path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << table.dump(2) << "\n";
        if (!f.good()) {
            *err = "cannot write " + tmp;
            return false;
        }
    }
#ifdef _WIN32
    std::remove(g_cfg.overrides_path.c_str());  // rename() does not replace on Windows
#endif
    if (std::rename(tmp.c_str(), g_cfg.overrides_path.c_str()) != 0) {
        *err = "cannot rename " + tmp + " -> " + g_cfg.overrides_path;
        return false;
    }
    return true;
}

void load_overrides() {
    if (g_cfg.overrides_path.empty()) return;
    std::ifstream f(g_cfg.overrides_path, std::ios::binary);
    if (!f) return;  // no file yet: every field follows the client
    try {
        g_overrides = validate_overrides(json::parse(f));
        fprintf(stderr, "gdec-api: overrides from %s: %s\n", g_cfg.overrides_path.c_str(),
                g_overrides.dump().c_str());
    } catch (const http::Error& e) {
        fprintf(stderr, "gdec-api: ignoring %s: %s\n", g_cfg.overrides_path.c_str(),
                e.message.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "gdec-api: ignoring %s: %s\n", g_cfg.overrides_path.c_str(), e.what());
    }
}

enum class Api { Completions, Chat, Responses };

// Rewrites `body` per the override table and reports the fields it changed in
// the X-Gdec-Overrides response header.
void apply_overrides(json* body, Api api, http::Response* r) {
    json table;
    {
        std::lock_guard<std::mutex> lk(g_ovr_mtx);
        table = g_overrides;
    }
    if (table.empty()) return;
    std::vector<std::string> changed;
    auto put = [&](json& obj, const char* key, const json& e, const std::string& label) {
        const bool present = obj.contains(key) && !obj[key].is_null();
        if (present && (e["mode"] != "force" || obj[key] == e["value"])) return;
        obj[key] = e["value"];
        changed.push_back(label);
    };
    for (auto it = table.begin(); it != table.end(); ++it) {
        const std::string& k = it.key();
        const json& e = it.value();
        if (k == "enable_thinking" || k == "preserve_thinking") {
            if (api != Api::Completions) put(*body, k.c_str(), e, k);
        } else if (k == "reasoning_effort") {
            if (api == Api::Chat) {
                put(*body, "reasoning_effort", e, k);
            } else if (api == Api::Responses) {
                if (!body->contains("reasoning") || (*body)["reasoning"].is_null())
                    (*body)["reasoning"] = json::object();
                // A malformed `reasoning` is left for the normal 400.
                if ((*body)["reasoning"].is_object()) put((*body)["reasoning"], "effort", e, k);
            }
        } else if (k == "max_tokens") {
            const long long cap = e["value"].get<long long>();
            bool any = false;
            for (const char* a : {"max_tokens", "max_completion_tokens", "max_output_tokens"}) {
                if (!body->contains(a) || (*body)[a].is_null()) continue;
                any = true;
                json& v = (*body)[a];
                if (e["mode"] == "force" && v.is_number_integer() && v.get<long long>() > cap) {
                    v = cap;
                    changed.push_back(a);
                }
            }
            if (!any) {
                (*body)[api == Api::Responses ? "max_output_tokens" : "max_tokens"] = cap;
                changed.push_back("max_tokens");
            }
        } else {
            put(*body, k.c_str(), e, k);
        }
    }
    // A forced greedy decode cannot honour logprobs (apply_sampling rejects
    // the pair); the operator's choice wins, so the request loses logprobs.
    if (std::find(changed.begin(), changed.end(), "temperature") != changed.end() &&
        (*body)["temperature"] == 0 && body->contains("logprobs")) {
        body->erase("logprobs");
        body->erase("top_logprobs");
        changed.push_back("logprobs");
    }
    if (changed.empty()) return;
    std::string list;
    for (const auto& c : changed) list += (list.empty() ? "" : ",") + c;
    r->set("X-Gdec-Overrides", list);
}

bool admin_ok(const http::Request& q) {
    if (g_cfg.admin_key.empty()) return true;
    return q.header("authorization") == "Bearer " + g_cfg.admin_key ||
           q.header("x-admin-key") == g_cfg.admin_key;
}

json overrides_state() {
    json j;
    {
        std::lock_guard<std::mutex> lk(g_ovr_mtx);
        j["overrides"] = g_overrides;
    }
    j["persist_path"] = g_cfg.overrides_path;
    j["admin_key_required"] = !g_cfg.admin_key.empty();
    return j;
}

// GET /admin/overrides
void handle_overrides_get(const http::Request&, http::Response* r, http::Stream*) {
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(overrides_state(), /*spaced=*/false);
}

// POST /admin/overrides — replaces the whole table; {} clears it.
void handle_overrides_post(const http::Request& q, http::Response* r, http::Stream*) {
    if (!admin_ok(q))
        http::fail(401, "admin key required (Authorization: Bearer <key> or X-Admin-Key)");
    const json body = parse_body(q);
    json table = validate_overrides(body.contains("overrides") ? body["overrides"] : body);
    std::string err;
    {
        std::lock_guard<std::mutex> lk(g_ovr_mtx);
        g_overrides = table;
    }
    const bool saved = save_overrides(table, &err);
    fprintf(stderr, "gdec-api: overrides set by %s: %s%s\n", q.remote.c_str(),
            table.dump().c_str(), saved ? "" : (" (NOT persisted: " + err + ")").c_str());
    json j = overrides_state();
    j["saved"] = saved;
    if (!saved) j["save_error"] = err;
    r->body = json_py::dumps(j, /*spaced=*/false);
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
    std::string line, err;
    {
        std::lock_guard<std::mutex> lk(g_conn_mtx);
        if (!g_ctl.connected()) g_ctl.connect(g_cfg.engine_addr, &err);
    }
    if (!g_ctl.cstat(&line, &err)) {
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

void handle_memory(const http::Request&, http::Response* r, http::Stream*) {
    std::string line, err;
    {
        std::lock_guard<std::mutex> lk(g_conn_mtx);
        if (!g_ctl.connected()) g_ctl.connect(g_cfg.engine_addr, &err);
    }
    if (!g_ctl.memory(&line, &err)) {
        r->status = 500;
        r->body = http::error_json("engine memory query failed: " + err,
                                   "server_error", "server_error");
        return;
    }
    std::istringstream ss(line);
    std::string tag;
    unsigned long long version = 0, device = 0, device_peak = 0, registered = 0,
                       pinned = 0, pinned_peak = 0, committed = 0, hip_free = 0,
                       hip_total = 0, hip_delta = 0, rss = 0, locked = 0;
    if (!(ss >> tag >> version >> device >> device_peak >> registered >> pinned >>
          pinned_peak >> committed >> hip_free >> hip_total >> hip_delta >> rss >>
          locked) ||
        tag != "M" || version != 1) {
        r->status = 500;
        r->body = http::error_json("invalid engine memory reply", "server_error",
                                   "server_error");
        return;
    }
    json j;
    j["device_current_bytes"] = device;
    j["device_peak_bytes"] = device_peak;
    j["registered_mmap_bytes"] = registered;
    j["pinned_host_current_bytes"] = pinned;
    j["pinned_host_peak_bytes"] = pinned_peak;
    j["gpu_accessible_committed_bytes"] = committed;
    j["hip_free_bytes"] = hip_free;
    j["hip_total_bytes"] = hip_total;
    j["hip_used_since_engine_start_bytes"] = hip_delta;
    j["process_rss_bytes"] = rss;
    j["process_locked_bytes"] = locked;
    j["accounting"] =
        "engine-requested HIP allocations; unregistered pageable mmap is excluded";
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// GET / 和 GET /dashboard — static monitoring page (see dashboard_html.inc).
// 页面加载后用同源 fetch 轮询 /health 与 /memory，这里只负责回 HTML。
#include "dashboard_html.inc"

void handle_dashboard(const http::Request&, http::Response* r, http::Stream*) {
    r->content_type = "text/html; charset=utf-8";
    r->set("Cache-Control", "no-store");
    r->body = kDashboardHtml;
}

void handle_health(const http::Request&, http::Response* r, http::Stream*) {
    json j;
    j["status"] = "ok";
    j["model"] = g_cfg.model;
    j["endpoints"] = json::array(
        {"/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/responses",
         "/dashboard"});
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
    j["slots"] = g_slots;
    j["slot_ctx"] = g_cfg.context;
    {
        std::lock_guard<std::mutex> lk(g_slot_mtx);
        j["queued"] = g_queued;
    }
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
    {
        std::lock_guard<std::mutex> lk(g_ovr_mtx);
        j["server_overrides"] = g_overrides;
    }
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// POST /v1/completions — raw prompt, no chat template.
void handle_completions(const http::Request& q, http::Response* r, http::Stream* st) {
    json mutable_body = parse_body(q);
    apply_overrides(&mutable_body, Api::Completions, r);
    const json& body = mutable_body;
    GenSpec spec;
    if (body.contains("prompt")) {
        const json& p = body["prompt"];
        const json* text = p.is_string() ? &p
                           : (p.is_array() && !p.empty() && p[0].is_string()) ? &p[0]
                                                                               : nullptr;
        if (text == nullptr) http::fail(400, kEngineRejected);
        std::vector<size_t> starts;
        std::string err;
        encode_spliced(text->get_ref<const std::string&>(), nullptr, &spec.ids, &starts, &err);
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
        GenOutcome o = run_generation(
            spec, nullptr, id,
            [st]() { return st == nullptr || !st->disconnected(); });
        json j = frame_base();
        json choice{{"index", 0}, {"finish_reason", o.reason}, {"text", o.text}};
        if (spec.logprobs) choice["logprobs"] = logprobs_json(o.logprobs);
        j["choices"] = json::array({std::move(choice)});
        j["usage"] = usage_json(o, spec.ids);
        r->body = json_py::dumps(j, /*spaced=*/false);
        return;
    }

    r->sse = true;
    auto on_wait = [&]() {
        if (st->disconnected()) return false;
        json frame = frame_base();
        frame["choices"] = json::array(
            {json{{"index", 0}, {"finish_reason", nullptr}, {"text", ""}}});
        return send_frame(st, frame);
    };
    GenOutcome o = run_generation(
        spec,
        [&](const std::string& d) {
            json frame = frame_base();
            frame["choices"] = json::array(
                {json{{"index", 0}, {"finish_reason", nullptr}, {"text", d}}});
            return send_frame(st, frame);
        },
        id, on_wait);
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
// character boundary is an exact token start in the final ids (`starts` =
// each id's byte offset in `text`), so every hint is a real token boundary of
// this prompt. Cuts stop at the first image: a prefix without image tokens is
// the same KV for text and vision requests, one past it is not.
std::vector<long long> compute_snap_cuts(const json& messages,
                                         chat_template::Options opts,
                                         const std::string& text,
                                         const std::vector<int>& ids,
                                         const std::vector<size_t>& starts) {
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
    const size_t n_text = static_cast<size_t>(
        std::find_if(ids.begin(), ids.end(), is_pad_id) - ids.begin());
    size_t ti = 0;
    for (size_t cc : cut_chars) {
        while (ti < n_text && starts[ti] < cc) ti++;
        if (ti < n_text && starts[ti] == cc && ti > 0) cuts.push_back((long long)ti);
    }
    if (cuts.size() > 8) cuts.erase(cuts.begin(), cuts.end() - 8);
    return cuts;
}

// POST /v1/chat/completions
void handle_chat(const http::Request& q, http::Response* r, http::Stream* st) {
    json mutable_body = parse_body(q);
    apply_overrides(&mutable_body, Api::Chat, r);
    const json& body = mutable_body;
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
    std::vector<size_t> starts;
    if (!encode_spliced(rr.text, &frames, &spec.ids, &starts, &normalize_error))
        http::fail(400, normalize_error);
    for (auto& frame : frames) {
        spec.mrope_grids.push_back(frame.grid);
        spec.patches.push_back(std::move(frame.patches));
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    spec.snaps = compute_snap_cuts(messages, opts, rr.text, spec.ids, starts);
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
        GenOutcome o = run_generation(
            spec, nullptr, id,
            [st]() { return st == nullptr || !st->disconnected(); });
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
        const std::string finish = (o.reason == "length" || parser.has_partial_call())
                                       ? "length"
                                       : (parser.calls().empty() ? o.reason : "tool_calls");
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
    auto on_wait = [&]() {
        if (st->disconnected()) return false;
        return send_frame(st, text_chunk("content", ""));
    };
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
        id, on_wait);
    if (o.client_gone) return;
    if (!tool_setup.choice.forced()) {
        std::string rp, ct;
        splitter.finish(&rp, &ct);
        if (!rp.empty() && !send_frame(st, text_chunk("reasoning_content", rp))) return;
        if (!dispatch_tool_events(parser.feed(ct))) return;
    }
    if (!dispatch_tool_events(parser.finish())) return;
    const std::string finish = (o.reason == "length" || parser.has_partial_call())
                                   ? "length"
                                   : (parser.calls().empty() ? o.reason : "tool_calls");
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
    json mutable_body = parse_body(q);
    apply_overrides(&mutable_body, Api::Responses, r);
    const json& body = mutable_body;
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
    std::vector<size_t> starts;
    if (!encode_spliced(rr.text, &frames, &spec.ids, &starts, &vision_error))
        http::fail(400, vision_error);
    for (auto& frame : frames) {
        spec.mrope_grids.push_back(frame.grid);
        spec.patches.push_back(std::move(frame.patches));
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    spec.snaps = compute_snap_cuts(messages, opts, rr.text, spec.ids, starts);
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
                response["incomplete_details"] = {
                    {"reason", outcome->reason == "length" ? "max_output_tokens"
                                                              : "invalid_tool_call"}};
        }
        return response;
    };

    if (!stream) {
        GenOutcome o = run_generation(
            spec, nullptr, id,
            [st]() { return st == nullptr || !st->disconnected(); });
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

        const bool incomplete = o.reason == "length" || parser.has_partial_call();
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
    auto on_wait = [&]() {
        if (st->disconnected()) return false;
        // No response item exists until generation starts, so use a standard
        // SSE comment as a protocol-neutral keep-alive during queue/prefill.
        return st->send_raw(": keep-alive\n\n");
    };
    GenOutcome o = run_generation(
        spec,
        [&](const std::string& delta) {
            if (st->disconnected()) return false;
            if (tool_setup.choice.forced()) return dispatch_tool_events(parser.feed(delta));
            std::string reasoning_delta, content_delta;
            splitter.feed(delta, &reasoning_delta, &content_delta);
            return dispatch_tool_events(parser.feed(content_delta));
        },
        id, on_wait);
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
    // Startup must not stall for the generation budget: an unreachable or
    // wedged engine falls back to the configured defaults (and one slot)
    // instead of blocking the listen socket for minutes. The control
    // connection keeps the short timeouts: MEM/CSTAT answer at once.
    g_ctl.set_timeouts(5.0, 5.0);
    if (!g_ctl.connect(g_cfg.engine_addr, &err)) {
        fprintf(stderr, "gdec-api: engine connect failed: %s\n", err.c_str());
    } else if (g_ctl.info(&line, &err)) {
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
        if (f.size() >= 10 && f[9] >= 1 && f[9] <= 64) g_slots = (int)f[9];
        fprintf(stderr, "gdec-api: engine INFO: %s\n", line.c_str());
    } else {
        fprintf(stderr, "gdec-api: engine INFO unavailable (%s); continuing with "
                        "ctx=%d\n", err.c_str(), g_cfg.context);
    }
    for (int i = 0; i < g_slots; ++i) {
        g_pool.push_back(std::make_unique<gdec::EngineClient>());
        g_pool_free.push_back(g_pool.back().get());
    }
    fprintf(stderr, "gdec-api: %d concurrent generation slot%s (shared context %d)\n",
            g_slots, g_slots > 1 ? "s" : "", g_cfg.context);
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
        else if (a == "--overrides") g_cfg.overrides_path = next();  // "" = memory only
        else {
            fprintf(stderr,
                    "usage: %s [--tokenizer DIR] [--engine H:P] [--listen H:P] "
                    "[--port N] [--host H] [--context N] [--model NAME] "
                    "[--overrides FILE]\n",
                    argv[0]);
            return 2;
        }
    }
    // Environment, not argv: keeps the key out of `ps`.
    if (const char* k = std::getenv("GDEC_API_ADMIN_KEY")) g_cfg.admin_key = k;
    if (const char* t = std::getenv("GDEC_API_TOKCACHE"))
        g_tcache.set_capacity(std::strtoull(t, nullptr, 10));
    {
        // "" disables persistence; default keeps the cache across restarts.
        const char* f = std::getenv("GDEC_API_TOKCACHE_FILE");
        g_tcache.set_file(f ? f : "data/tcache.bin");
        if (const char* s = std::getenv("GDEC_API_TOKCACHE_SAVE_S"))
            g_tcache.set_save_interval(std::atoi(s));
    }
    load_overrides();

    std::string err;
    if (!g_tok.load(g_cfg.tokenizer_dir, &err)) {
        fprintf(stderr, "gdec-api: tokenizer: %s\n", err.c_str());
        return 1;
    }
    g_tcache.load_file();
    g_ckpt_ids = default_ckpt_tokens();
    probe_engine();

    http::Server srv;
    if (!srv.listen(g_cfg.listen, &err)) {
        fprintf(stderr, "gdec-api: listen: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "gdec-api: listening on :%d model=%s ctx=%d slots=%d\n", srv.port(),
            g_cfg.model.c_str(), g_cfg.context, g_slots);

    srv.on("GET", "/", handle_dashboard);
    srv.on("GET", "/dashboard", handle_dashboard);
    srv.on("GET", "/v1/models", handle_models);
    srv.on("GET", "/health", handle_health);
    srv.on("GET", "/memory", handle_memory);
    srv.on("GET", "/cache", handle_cache);
    srv.on("GET", "/admin/overrides", handle_overrides_get);
    srv.on("POST", "/admin/overrides", handle_overrides_post);
    srv.on("POST", "/v1/completions", handle_completions);
    srv.on("POST", "/v1/chat/completions", handle_chat);
    srv.on("POST", "/v1/responses", handle_responses);

    srv.run();
    return 0;
}
