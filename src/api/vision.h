#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer.h"

namespace vision {

using json = nlohmann::ordered_json;

struct Frame {
    std::array<int, 3> grid{1, 0, 0};
    std::vector<float> patches;

    int pad_tokens() const { return grid[0] * grid[1] * grid[2] / 4; }
};

// Decode a data URL or bare base64 payload. HTTP(S) is deliberately rejected.
bool decode_image_url(const std::string& value, std::vector<uint8_t>* bytes,
                      std::string* error);

// Decode PNG/JPEG/WebP, smart-resize it, normalize it, and build Qwen patches.
bool preprocess(const std::vector<uint8_t>& bytes, Frame* frame, std::string* error);

// Extract and preprocess image parts from normalized chat messages, in order.
bool prepare_messages(const json& messages, std::vector<Frame>* frames,
                      std::string* error);

// Expand only image-pad tokens inside complete vision placeholder spans.
// `starts` (optional) receives each id's byte offset in `prompt` (the copies
// of an expanded pad share the pad's offset). Only frames[i].pad_tokens() is
// read, so the frames may carry grids without patches.
bool encode_prompt(const std::string& prompt, const gdec::Tokenizer& tokenizer,
                   const std::vector<Frame>& frames, std::vector<int>* ids,
                   std::string* error, std::vector<size_t>* starts = nullptr);

}  // namespace vision
