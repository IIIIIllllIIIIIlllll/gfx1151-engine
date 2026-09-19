// Clean-room byte-level BPE tokenizer (Qwen-family tokenizer.json format).
//
// Sources of truth: the model's own tokenizer.json (Apache-2.0) and public
// GPT-2/Qwen2 byte-level BPE semantics. Verified id-for-id against HF
// transformers (fast tokenizer) by tools/tok_ab.py.
//
// Behaviors confirmed against HF that are worth knowing:
//  - The reference is the transformers-v5 Qwen2Tokenizer, whose pre-tokenizer
//    is rebuilt from the library's class default rather than taken verbatim
//    from tokenizer.json. Both agree except that the library keeps `\p{M}`
//    out of the letter run; see the note above match_pretoken() in the .cpp.
//    tools/probe_service.py verifies by black-box prompt_tokens that the live
//    Python service follows the library, so that is what this file implements.
//  - The pre-tokenizer regex uses `\p{N}` (each digit its own pre-token), NOT
//    the classic Qwen2 `\p{N}{1,3}`. "12345" splits into five pre-tokens.
//  - The letter alternative `[^\r\n\p{L}\p{N}]?\p{L}+` allows ANY single
//    non-letter/non-digit/non-newline char (including a space, tab, or
//    combining mark) as prefix before a letter run: " abc" and "\tabc" are
//    each one pre-token, and so is "̈_s" up to the mark.
//  - `\s` follows the Rust regex crate = Unicode White_Space property
//    (25 codepoints: includes U+0085/U+00A0/U+1680/U+2000-200A/U+2028-2029/
//    U+202F/U+205F/U+3000; NOT U+001C-001F).
//  - `\s+(?!\S)` (greedy with backtrack + lookahead): a whitespace run at
//    end-of-string matches whole; a run before non-whitespace leaves its
//    last char for the next match ("  a" -> " " + " a"... actually
//    " " then " a" via the letter-prefix rule).
//  - Added (special) tokens are extracted longest-match-first from the raw
//    text before normalization/BPE, even when add_special_tokens=false
//    (that flag is a no-op here: the post-processor adds nothing).
//  - decode(skip_special_tokens=true) drops only added tokens flagged
//    special=true; non-special added tokens like `<tool_call>` survive as
//    literal text.
//  - decode replaces ill-formed UTF-8 with U+FFFD using the Unicode
//    "maximal subpart" rule (same as Rust String::from_utf8_lossy and
//    Python bytes.decode('utf-8', 'replace')).
//  - Normalizer is NFC only: canonical equivalents compose (e + U+0301 ->
//    U+00E9), compatibility variants (halfwidth katakana, circled digits)
//    are left untouched.
//  - Unicode data tables are Unicode 15.1.0 (see unicode_tables.inc). HF's
//    regex crate may use a newer Unicode; codepoints assigned after 15.1
//    can in principle classify differently (not exercised by the A/B corpus,
//    which only samples codepoints assigned in 15.1).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gdec {

struct TokenSpan {
    int id;
    size_t start;  // byte offset in the input text (inclusive)
    size_t end;    // byte offset in the input text (exclusive)
};

class Tokenizer {
  public:
    // Loads <dir>/tokenizer.json. Returns false on error (message in `err`).
    bool load(const std::string& dir, std::string* err = nullptr);

    // Encode text to token ids. `add_special_tokens` is accepted for API
    // parity with HF; this model's post-processor never adds tokens, and
    // added tokens appearing literally in the text are always recognized.
    std::vector<int> encode(const std::string& text,
                            bool add_special_tokens = false) const;

    // Like encode(), but also reports the byte span of each token in the
    // original input text. When NFC normalization rewrites a span, the
    // reported offsets are interpolated from the surrounding codepoints
    // (exact whenever normalization is the identity on that span).
    std::vector<TokenSpan> encode_with_offsets(
        const std::string& text, bool add_special_tokens = false) const;

    // Decode token ids back to text. skip_special_tokens drops added tokens
    // flagged special=true. Ill-formed UTF-8 byte runs become U+FFFD
    // (maximal-subpart rule, matching HF).
    std::string decode(const std::vector<int>& ids,
                       bool skip_special_tokens = true) const;

    // The raw byte stream the ids stand for, before UTF-8 lossy repair. This is
    // what an incremental streamer needs: a multi-byte character split across
    // tokens must be held back until it is complete, which is impossible once
    // decode() has already replaced the partial run with U+FFFD.
    std::string decode_bytes(const std::vector<int>& ids,
                             bool skip_special_tokens = true) const;

    // Look up an added token's id by literal content; -1 if not found.
    int added_token_id(const std::string& content) const;
    bool is_special_token(int id) const;
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int eos_token_id() const { return eos_id_; }   // <|im_end|>
    int pad_token_id() const { return pad_id_; }   // <|endoftext|>

  private:
    struct AddedToken {
        std::string content;
        int id;
        bool special;
    };

    std::unordered_map<std::string, int> vocab_;      // token str -> id
    std::vector<std::string> id_to_token_;            // id -> token str
    // Merge table: (left_id, right_id) -> (rank << 32) | merged_id.
    std::unordered_map<uint64_t, uint64_t> merges_;
    std::vector<AddedToken> added_;                   // sorted longest-first
    std::unordered_map<int, bool> added_special_;     // id -> special flag
    std::unordered_map<std::string, int> added_ids_;  // content -> id
    int eos_id_ = -1;
    int pad_id_ = -1;

    void bpe(const std::string& pretoken_alphabet, std::vector<int>* out,
             std::vector<uint32_t>* token_char_lens) const;
};

}  // namespace gdec
