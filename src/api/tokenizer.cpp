#include "tokenizer.h"

#include "unicode_tables.inc"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace {

using gdec_uni::CCCEntry;
using gdec_uni::CompEntry;
using gdec_uni::DecompEntry;

// ---------- Unicode table lookups ----------

bool in_intervals(const uint32_t tab[][2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < tab[mid][0]) hi = mid;
        else if (cp > tab[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

bool uni_is_L(uint32_t cp) { return in_intervals(gdec_uni::kCatL, gdec_uni::kCatLCount, cp); }
bool uni_is_M(uint32_t cp) { return in_intervals(gdec_uni::kCatM, gdec_uni::kCatMCount, cp); }
bool uni_is_N(uint32_t cp) { return in_intervals(gdec_uni::kCatN, gdec_uni::kCatNCount, cp); }
bool uni_is_ws(uint32_t cp) { return in_intervals(gdec_uni::kWhiteSpace, gdec_uni::kWhiteSpaceCount, cp); }

uint8_t uni_ccc(uint32_t cp) {
    size_t lo = 0, hi = gdec_uni::kCCCCount;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        const CCCEntry& e = gdec_uni::kCCC[mid];
        if (cp < e.lo) hi = mid;
        else if (cp > e.hi) lo = mid + 1;
        else return e.ccc;
    }
    return 0;
}

// ---------- NFC ----------

constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr uint32_t kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

const DecompEntry* find_decomp(uint32_t cp) {
    size_t lo = 0, hi = gdec_uni::kDecompCount;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < gdec_uni::kDecomp[mid].cp) hi = mid;
        else if (cp > gdec_uni::kDecomp[mid].cp) lo = mid + 1;
        else return &gdec_uni::kDecomp[mid];
    }
    return nullptr;
}

// A codepoint tagged with its source byte span in the original text piece.
struct SCP {
    uint32_t cp;
    size_t s0, s1;
};

void decompose_canonical(uint32_t cp, size_t s0, size_t s1, std::vector<SCP>* out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
        uint32_t si = cp - kSBase;
        out->push_back({kLBase + si / kNCount, s0, s1});
        out->push_back({kVBase + (si % kNCount) / kTCount, s0, s1});
        uint32_t t = si % kTCount;
        if (t) out->push_back({kTBase + t, s0, s1});
        return;
    }
    const DecompEntry* e = find_decomp(cp);
    if (!e) {
        out->push_back({cp, s0, s1});
        return;
    }
    for (uint16_t i = 0; i < e->len; ++i)
        decompose_canonical(gdec_uni::kDecompData[e->off + i], s0, s1, out);
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 &&
        b > kTBase && b < kTBase + kTCount)
        return a + (b - kTBase);
    size_t lo = 0, hi = gdec_uni::kCompCount;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        const CompEntry& e = gdec_uni::kComp[mid];
        if (a < e.a || (a == e.a && b < e.b)) hi = mid;
        else if (a == e.a && b == e.b) return e.c;
        else lo = mid + 1;
    }
    return 0;
}

std::vector<SCP> nfc(const std::vector<SCP>& in) {
    std::vector<SCP> d;
    d.reserve(in.size());
    for (const SCP& s : in) decompose_canonical(s.cp, s.s0, s.s1, &d);
    // Canonical ordering: stable sort by ccc within each non-starter run.
    for (size_t i = 1; i < d.size(); ++i) {
        uint8_t c = uni_ccc(d[i].cp);
        if (!c) continue;
        size_t j = i;
        while (j > 0) {
            uint8_t pc = uni_ccc(d[j - 1].cp);
            if (pc == 0 || pc <= c) break;
            std::swap(d[j], d[j - 1]);
            --j;
        }
    }
    // Canonical composition.
    std::vector<SCP> out;
    out.reserve(d.size());
    size_t starter = SIZE_MAX;
    int prev_cc = 0;
    for (const SCP& s : d) {
        int cc = uni_ccc(s.cp);
        if (starter != SIZE_MAX && (prev_cc == 0 || prev_cc < cc)) {
            uint32_t c = compose_pair(out[starter].cp, s.cp);
            if (c) {
                out[starter].cp = c;
                out[starter].s1 = s.s1;
                continue;
            }
        }
        if (cc == 0) starter = out.size();
        prev_cc = cc;
        out.push_back(s);
    }
    return out;
}

// ---------- UTF-8 helpers ----------

void append_utf8(uint32_t cp, std::string* out) {
    if (cp < 0x80) out->push_back((char)cp);
    else if (cp < 0x800) {
        out->push_back((char)(0xC0 | (cp >> 6)));
        out->push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out->push_back((char)(0xE0 | (cp >> 12)));
        out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out->push_back((char)(0xF0 | (cp >> 18)));
        out->push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back((char)(0x80 | (cp & 0x3F)));
    }
}

// Decode [s, e) of text into codepoints; invalid bytes become one U+FFFD each.
std::vector<SCP> decode_utf8(const std::string& text, size_t s, size_t e) {
    std::vector<SCP> out;
    size_t i = s;
    while (i < e) {
        uint8_t c = (uint8_t)text[i];
        uint32_t cp = 0xFFFD;
        size_t len = 1;
        if (c < 0x80) {
            cp = c;
        } else if (c >= 0xC2 && c <= 0xDF && i + 1 < e) {
            uint8_t c1 = (uint8_t)text[i + 1];
            if ((c1 & 0xC0) == 0x80) { cp = ((c & 0x1F) << 6) | (c1 & 0x3F); len = 2; }
        } else if (c >= 0xE0 && c <= 0xEF && i + 2 < e) {
            uint8_t c1 = (uint8_t)text[i + 1], c2 = (uint8_t)text[i + 2];
            if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 &&
                !(c == 0xE0 && c1 < 0xA0) && !(c == 0xED && c1 >= 0xA0)) {
                cp = ((c & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
                len = 3;
            }
        } else if (c >= 0xF0 && c <= 0xF4 && i + 3 < e) {
            uint8_t c1 = (uint8_t)text[i + 1], c2 = (uint8_t)text[i + 2], c3 = (uint8_t)text[i + 3];
            if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80 &&
                !(c == 0xF0 && c1 < 0x90) && !(c == 0xF4 && c1 >= 0x90)) {
                cp = ((c & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
                len = 4;
            }
        }
        out.push_back({cp, i, i + len});
        i += len;
    }
    return out;
}

// Lossy UTF-8 decode with the Unicode "maximal subpart" replacement rule
// (matches Rust String::from_utf8_lossy / Python errors='replace').
std::string utf8_lossy(const std::string& b) {
    std::string out;
    out.reserve(b.size());
    size_t i = 0, n = b.size();
    while (i < n) {
        uint8_t c = (uint8_t)b[i];
        if (c < 0x80) {
            out.push_back((char)c);
            ++i;
            continue;
        }
        int len = 0;
        uint8_t lo2 = 0x80, hi2 = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) len = 2;
        else if (c >= 0xE0 && c <= 0xEF) {
            len = 3;
            if (c == 0xE0) lo2 = 0xA0;
            if (c == 0xED) hi2 = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            len = 4;
            if (c == 0xF0) lo2 = 0x90;
            if (c == 0xF4) hi2 = 0x8F;
        } else {
            append_utf8(0xFFFD, &out);
            ++i;
            continue;
        }
        size_t valid = 1;  // lead byte counts as part of the subpart
        if (i + 1 < n) {
            uint8_t c1 = (uint8_t)b[i + 1];
            if (c1 >= lo2 && c1 <= hi2) {
                valid = 2;
                while (valid < (size_t)len && i + valid < n) {
                    uint8_t cx = (uint8_t)b[i + valid];
                    if (cx < 0x80 || cx > 0xBF) break;
                    ++valid;
                }
            }
        }
        if (valid == (size_t)len && i + (size_t)len <= n) {
            uint32_t cp = 0;
            if (len == 2) cp = ((c & 0x1F) << 6) | ((uint8_t)b[i + 1] & 0x3F);
            else if (len == 3)
                cp = ((c & 0x0F) << 12) | (((uint8_t)b[i + 1] & 0x3F) << 6) | ((uint8_t)b[i + 2] & 0x3F);
            else
                cp = ((c & 0x07) << 18) | (((uint8_t)b[i + 1] & 0x3F) << 12) |
                     (((uint8_t)b[i + 2] & 0x3F) << 6) | ((uint8_t)b[i + 3] & 0x3F);
            append_utf8(cp, &out);
            i += (size_t)len;
        } else {
            append_utf8(0xFFFD, &out);
            i += valid;
        }
    }
    return out;
}

// ---------- GPT-2 byte-level alphabet ----------

uint32_t byte_to_unicode_tab[256];
std::unordered_map<uint32_t, uint8_t> unicode_to_byte_tab;
bool alphabet_ready = false;

void init_alphabet() {
    if (alphabet_ready) return;
    bool direct[256] = {};
    for (int b = '!'; b <= '~'; ++b) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t ch = direct[b] ? (uint32_t)b : (uint32_t)(256 + n++);
        byte_to_unicode_tab[b] = ch;
        unicode_to_byte_tab[ch] = (uint8_t)b;
    }
    alphabet_ready = true;
}

// ---------- Pre-tokenizer ----------
// Hand implementation of the regex the deployment actually uses:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
// | [^\r\n\p{L}\p{N}]?\p{L}+
// | \p{N}
// |  ?[^\s\p{L}\p{N}]+[\r\n]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
// with behavior=Isolated (leftmost-first alternation, backtracking).
//
// This is the transformers v5 Qwen2Tokenizer pre-tokenizer, NOT the one stored
// in tokenizer.json. tokenizer.json (and tokenizer_config.json's
// `pretokenize_regex`) carry `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` and
// ` ?[^\s\p{L}\p{M}\p{N}]+`, but transformers v5 rebuilds the pipeline from its
// own class default, which keeps marks OUT of the letter run. That difference
// is observable: for "\u0308_s" tokenizer.json yields [136,230,625] while the
// live service (a black-box prompt_tokens probe) yields [136,230,62,82].
// See tools/tok_ab.py --ref=auto (the gate) and tools/probe_service.py.

enum CpFlags : uint8_t {
    F_L = 1, F_M = 2, F_N = 4, F_WS = 8, F_CR = 16, F_LF = 32,
};

struct CP {
    uint32_t cp;
    uint8_t flags;
    size_t byte;  // byte offset of this codepoint in the normalized string
};

// Unicode-aware simple case fold for the contraction alternation.
// Only folds what (?i:...) can match for s/t/r/e/v/m/l/d.
uint32_t fold_letter(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp == 0x017F) return 's';  // Latin small long s folds to 's'
    return cp;
}

size_t match_pretoken(const std::vector<CP>& s, size_t i) {
    const size_t n = s.size();
    const CP& c = s[i];

    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c.cp == '\'' && i + 1 < n) {
        uint32_t c1 = fold_letter(s[i + 1].cp);
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return i + 2;
        if ((c1 == 'r' || c1 == 'v' || c1 == 'l') && i + 2 < n) {
            uint32_t c2 = fold_letter(s[i + 2].cp);
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                (c1 == 'l' && c2 == 'l'))
                return i + 3;
        }
    }

    // 2. [^\r\n\p{L}\p{N}]?\p{L}+   (combining marks are NOT part of the run)
    if (c.flags & F_L) {
        size_t k = i;
        while (k < n && (s[k].flags & F_L)) ++k;
        return k;
    }
    if (!(c.flags & (F_L | F_N)) && c.cp != '\r' && c.cp != '\n') {
        size_t k = i + 1;
        while (k < n && (s[k].flags & F_L)) ++k;
        if (k > i + 1) return k;
    }

    // 3. \p{N}
    if (c.flags & F_N) return i + 1;

    // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    {
        size_t j = (c.cp == ' ') ? i + 1 : i;
        size_t k = j;
        while (k < n && !(s[k].flags & (F_WS | F_L | F_N))) ++k;
        if (k > j) {
            while (k < n && (s[k].cp == '\r' || s[k].cp == '\n')) ++k;
            return k;
        }
    }

    // 5/6/7. \s*[\r\n]+ | \s+(?!\S) | \s+
    if (c.flags & F_WS) {
        size_t wend = i;
        while (wend < n && (s[wend].flags & F_WS)) ++wend;
        // 5: match ends after the last CR/LF inside the whitespace run.
        for (size_t p = wend; p-- > i;) {
            if (s[p].cp == '\r' || s[p].cp == '\n') {
                size_t e = p + 1;
                while (e < n && (s[e].cp == '\r' || s[e].cp == '\n')) ++e;
                return e;
            }
        }
        // 6: \s+(?!\S) — whole run at EOS, else all but the last ws char.
        if (wend == n) return wend;
        if (wend >= i + 2) return wend - 1;
        // 7: \s+
        return wend;
    }

    return i;  // unreachable for valid input; caller skips one codepoint
}

}  // namespace

namespace gdec {

bool Tokenizer::load(const std::string& dir, std::string* err) {
    init_alphabet();
    std::ifstream f(dir + "/tokenizer.json");
    if (!f) {
        if (err) *err = "cannot open " + dir + "/tokenizer.json";
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ss.str());
    } catch (const std::exception& e) {
        if (err) *err = std::string("json parse: ") + e.what();
        return false;
    }

    try {
        // Vocab.
        int max_id = -1;
        for (auto& [tok, id] : j["model"]["vocab"].items()) {
            int i = id.get<int>();
            vocab_[tok] = i;
            max_id = std::max(max_id, i);
        }
        // Added tokens.
        for (auto& t : j["added_tokens"]) {
            AddedToken at{t["content"].get<std::string>(), t["id"].get<int>(),
                          t.value("special", false)};
            added_.push_back(at);
            added_special_[at.id] = at.special;
            added_ids_[at.content] = at.id;
            max_id = std::max(max_id, at.id);
        }
        id_to_token_.assign((size_t)max_id + 1, std::string());
        for (auto& [tok, id] : vocab_) id_to_token_[(size_t)id] = tok;
        for (auto& at : added_) id_to_token_[(size_t)at.id] = at.content;
        std::sort(added_.begin(), added_.end(), [](const AddedToken& a, const AddedToken& b) {
            return a.content.size() > b.content.size();
        });

        // Merges: "A B" (or ["A","B"]) in rank order.
        int rank = 0;
        for (auto& m : j["model"]["merges"]) {
            std::string a, b;
            if (m.is_string()) {
                std::string s = m.get<std::string>();
                size_t sp = s.find(' ');
                a = s.substr(0, sp);
                b = s.substr(sp + 1);
            } else {
                a = m[0].get<std::string>();
                b = m[1].get<std::string>();
            }
            auto ia = vocab_.find(a), ib = vocab_.find(b), im = vocab_.find(a + b);
            if (ia != vocab_.end() && ib != vocab_.end() && im != vocab_.end()) {
                uint64_t key = (uint64_t)(uint32_t)ia->second << 32 | (uint32_t)ib->second;
                merges_[key] = (uint64_t)(uint32_t)rank << 32 | (uint32_t)im->second;
            }
            ++rank;
        }
    } catch (const std::exception& e) {
        if (err) *err = std::string("tokenizer.json content: ") + e.what();
        return false;
    }

    auto it = added_ids_.find("<|im_end|>");
    if (it != added_ids_.end()) eos_id_ = it->second;
    it = added_ids_.find("<|endoftext|>");
    if (it != added_ids_.end()) pad_id_ = it->second;
    return true;
}

void Tokenizer::bpe(const std::string& alphabet_str, std::vector<int>* out,
                    std::vector<uint32_t>* token_char_lens) const {
    // Split the byte-alphabet string into single-character tokens.
    std::vector<int> word;
    std::vector<uint32_t> lens;
    for (size_t i = 0; i < alphabet_str.size();) {
        uint8_t c = (uint8_t)alphabet_str[i];
        size_t len = (c < 0x80) ? 1 : (c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4));
        auto it = vocab_.find(alphabet_str.substr(i, len));
        if (it == vocab_.end()) {  // should never happen (full byte coverage)
            i += len;
            continue;
        }
        word.push_back(it->second);
        lens.push_back(1);
        i += len;
    }
    while (word.size() > 1) {
        uint32_t best_rank = UINT32_MAX;
        int best_id = -1;
        for (size_t k = 0; k + 1 < word.size(); ++k) {
            uint64_t key = (uint64_t)(uint32_t)word[k] << 32 | (uint32_t)word[k + 1];
            auto it = merges_.find(key);
            if (it != merges_.end() && (uint32_t)(it->second >> 32) < best_rank) {
                best_rank = (uint32_t)(it->second >> 32);
                best_id = (int)(uint32_t)it->second;
            }
        }
        if (best_id < 0) break;
        std::vector<int> nw;
        std::vector<uint32_t> nl;
        nw.reserve(word.size());
        nl.reserve(word.size());
        for (size_t k = 0; k < word.size(); ++k) {
            if (k + 1 < word.size()) {
                uint64_t key = (uint64_t)(uint32_t)word[k] << 32 | (uint32_t)word[k + 1];
                auto it = merges_.find(key);
                if (it != merges_.end() && (uint32_t)(it->second >> 32) == best_rank) {
                    nw.push_back(best_id);
                    nl.push_back(lens[k] + lens[k + 1]);
                    ++k;
                    continue;
                }
            }
            nw.push_back(word[k]);
            nl.push_back(lens[k]);
        }
        word.swap(nw);
        lens.swap(nl);
    }
    *out = std::move(word);
    *token_char_lens = std::move(lens);
}

std::vector<TokenSpan> Tokenizer::encode_with_offsets(const std::string& text,
                                                      bool /*add_special_tokens*/) const {
    std::vector<TokenSpan> result;
    const size_t n = text.size();
    auto match_added = [&](size_t pos) -> const AddedToken* {
        for (const AddedToken& at : added_)  // sorted longest-first
            if (pos + at.content.size() <= n &&
                text.compare(pos, at.content.size(), at.content) == 0)
                return &at;
        return nullptr;
    };

    size_t pos = 0;
    while (pos < n) {
        if (const AddedToken* at = match_added(pos)) {
            result.push_back({at->id, pos, pos + at->content.size()});
            pos += at->content.size();
            continue;
        }
        size_t piece_end = pos + 1;
        while (piece_end < n && !match_added(piece_end)) ++piece_end;

        // NFC-normalize the piece, tracking source byte spans per codepoint.
        std::vector<SCP> cps = decode_utf8(text, pos, piece_end);
        std::vector<SCP> normed = nfc(cps);

        // Re-encode normalized codepoints; build norm-byte -> orig-byte map.
        std::string ns;
        std::vector<size_t> nb2orig;
        for (const SCP& s : normed) {
            size_t before = ns.size();
            append_utf8(s.cp, &ns);
            size_t len = ns.size() - before;
            size_t span = s.s1 > s.s0 ? s.s1 - s.s0 : 1;
            for (size_t t = 0; t < len; ++t) nb2orig.push_back(s.s0 + (t * span) / len);
        }

        // Codepoint view of the normalized string with category flags.
        std::vector<CP> view;
        for (size_t i = 0; i < ns.size();) {
            uint8_t c = (uint8_t)ns[i];
            size_t len = (c < 0x80) ? 1 : (c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4));
            uint32_t cp = 0;
            if (len == 1) cp = c;
            else if (len == 2) cp = ((c & 0x1F) << 6) | ((uint8_t)ns[i + 1] & 0x3F);
            else if (len == 3)
                cp = ((c & 0x0F) << 12) | (((uint8_t)ns[i + 1] & 0x3F) << 6) |
                     ((uint8_t)ns[i + 2] & 0x3F);
            else
                cp = ((c & 0x07) << 18) | (((uint8_t)ns[i + 1] & 0x3F) << 12) |
                     (((uint8_t)ns[i + 2] & 0x3F) << 6) | ((uint8_t)ns[i + 3] & 0x3F);
            uint8_t fl = 0;
            if (uni_is_L(cp)) fl |= F_L;
            if (uni_is_M(cp)) fl |= F_M;
            if (uni_is_N(cp)) fl |= F_N;
            if (uni_is_ws(cp)) fl |= F_WS;
            view.push_back({cp, fl, i});
            i += len;
        }

        // Split into pre-tokens, byte-level map, and BPE each.
        size_t i = 0;
        while (i < view.size()) {
            size_t j = match_pretoken(view, i);
            if (j <= i) { ++i; continue; }  // safety: Isolated drops gaps
            size_t bs = view[i].byte;
            size_t be = (j < view.size()) ? view[j].byte : ns.size();
            std::string atok;
            for (size_t b = bs; b < be; ++b)
                append_utf8(byte_to_unicode_tab[(uint8_t)ns[b]], &atok);
            std::vector<int> ids;
            std::vector<uint32_t> lens;
            bpe(atok, &ids, &lens);
            size_t cur = bs;
            for (size_t k = 0; k < ids.size(); ++k) {
                size_t ostart = nb2orig[cur];
                cur += lens[k];
                size_t oend = (cur >= ns.size()) ? piece_end : nb2orig[cur];
                result.push_back({ids[k], ostart, oend});
            }
            i = j;
        }
        pos = piece_end;
    }
    return result;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<TokenSpan> spans = encode_with_offsets(text, add_special_tokens);
    std::vector<int> ids;
    ids.reserve(spans.size());
    for (const TokenSpan& t : spans) ids.push_back(t.id);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    return utf8_lossy(decode_bytes(ids, skip_special_tokens));
}

std::string Tokenizer::decode_bytes(const std::vector<int>& ids,
                                    bool skip_special_tokens) const {
    init_alphabet();
    std::string bytes;
    for (int id : ids) {
        if (id < 0 || (size_t)id >= id_to_token_.size()) continue;
        auto sp = added_special_.find(id);
        if (sp != added_special_.end()) {
            if (skip_special_tokens && sp->second) continue;
            bytes += id_to_token_[(size_t)id];  // literal content (ASCII)
            continue;
        }
        const std::string& tok = id_to_token_[(size_t)id];
        for (size_t i = 0; i < tok.size();) {
            uint8_t c = (uint8_t)tok[i];
            size_t len = (c < 0x80) ? 1 : (c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4));
            uint32_t cp = 0;
            if (len == 1) cp = c;
            else if (len == 2) cp = ((c & 0x1F) << 6) | ((uint8_t)tok[i + 1] & 0x3F);
            else if (len == 3)
                cp = ((c & 0x0F) << 12) | (((uint8_t)tok[i + 1] & 0x3F) << 6) |
                     ((uint8_t)tok[i + 2] & 0x3F);
            else
                cp = ((c & 0x07) << 18) | (((uint8_t)tok[i + 1] & 0x3F) << 12) |
                     (((uint8_t)tok[i + 2] & 0x3F) << 6) | ((uint8_t)tok[i + 3] & 0x3F);
            auto it = unicode_to_byte_tab.find(cp);
            if (it != unicode_to_byte_tab.end()) bytes.push_back((char)it->second);
            i += len;
        }
    }
    return bytes;
}

int Tokenizer::added_token_id(const std::string& content) const {
    auto it = added_ids_.find(content);
    return it == added_ids_.end() ? -1 : it->second;
}

bool Tokenizer::is_special_token(int id) const {
    auto it = added_special_.find(id);
    return it != added_special_.end() && it->second;
}

}  // namespace gdec
