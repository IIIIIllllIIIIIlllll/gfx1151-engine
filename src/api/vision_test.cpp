#include "vision.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}

std::vector<float> read_npy_f32(const std::string& path) {
    std::vector<uint8_t> bytes = read_bytes(path);
    if (bytes.size() < 12 || std::string(reinterpret_cast<char*>(bytes.data() + 1), 5) != "NUMPY")
        return {};
    const int major = bytes[6];
    size_t header = 0, data = 0;
    if (major == 1) {
        header = static_cast<size_t>(bytes[8]) | (static_cast<size_t>(bytes[9]) << 8);
        data = 10 + header;
    } else {
        header = static_cast<size_t>(bytes[8]) | (static_cast<size_t>(bytes[9]) << 8) |
                 (static_cast<size_t>(bytes[10]) << 16) |
                 (static_cast<size_t>(bytes[11]) << 24);
        data = 12 + header;
    }
    if (data > bytes.size() || (bytes.size() - data) % sizeof(float) != 0) return {};
    std::vector<float> result((bytes.size() - data) / sizeof(float));
    std::memcpy(result.data(), bytes.data() + data, result.size() * sizeof(float));
    return result;
}

std::string base64_encode(const std::vector<uint8_t>& input) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve((input.size() + 2) / 3 * 4);
    for (size_t i = 0; i < input.size(); i += 3) {
        const uint32_t value = static_cast<uint32_t>(input[i]) << 16 |
                               (i + 1 < input.size()
                                    ? static_cast<uint32_t>(input[i + 1]) << 8
                                    : 0u) |
                               (i + 2 < input.size() ? input[i + 2] : 0u);
        output.push_back(alphabet[(value >> 18) & 63]);
        output.push_back(alphabet[(value >> 12) & 63]);
        output.push_back(i + 1 < input.size() ? alphabet[(value >> 6) & 63] : '=');
        output.push_back(i + 2 < input.size() ? alphabet[value & 63] : '=');
    }
    return output;
}

void check(bool ok, const std::string& name, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " ", detail.c_str());
    if (!ok) std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: vision_test <data/vision-ref> <tokenizer-dir>\n");
        return 2;
    }
    const std::string root = argv[1];
    for (const char* name : {"text", "landscape", "table"}) {
        const std::vector<uint8_t> encoded = read_bytes(root + "/" + name + ".png");
        vision::Frame frame;
        std::string error;
        check(vision::preprocess(encoded, &frame, &error), std::string(name) + " preprocess", error);
        const std::vector<float> expected =
            read_npy_f32(root + "/" + name + ".patches.npy");
        check(frame.grid == std::array<int, 3>{1, 68, 120},
              std::string(name) + " grid");
        check(frame.patches.size() == expected.size(), std::string(name) + " shape");
        float max_diff = 0.0f;
        size_t differing = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            const float diff = std::abs(frame.patches[i] - expected[i]);
            max_diff = std::max(max_diff, diff);
            if (diff != 0.0f) ++differing;
        }
        check(max_diff <= 1e-6f, std::string(name) + " pixels",
              "max_diff=" + std::to_string(max_diff) +
                  " differing=" + std::to_string(differing));
    }

    const std::vector<uint8_t> png = read_bytes(root + "/text.png");
    const std::string encoded = base64_encode(png);
    std::vector<uint8_t> decoded;
    std::string error;
    check(vision::decode_image_url(encoded, &decoded, &error) && decoded == png,
          "bare-base64", error);
    check(vision::decode_image_url("data:image/png;base64," + encoded, &decoded, &error) &&
              decoded == png,
          "data-url", error);
    check(!vision::decode_image_url("https://example.invalid/image.png", &decoded, &error) &&
              error.find("no outbound requests") != std::string::npos,
          "http-url-rejected", error);
    check(!vision::decode_image_url("not base64", &decoded, &error),
          "bad-base64-rejected", error);

    using json = vision::json;
    const std::string data_url = "data:image/png;base64," + encoded;
    json messages = json::array(
        {json{{"role", "user"},
              {"content", json::array({json{{"type", "image_url"},
                                             {"image_url", {{"url", data_url}}}},
                                        json{{"type", "text"}, {"text", "describe"}}})}}});
    std::vector<vision::Frame> frames;
    check(vision::prepare_messages(messages, &frames, &error) && frames.size() == 1,
          "prepare-chat-image", error);

    json too_many = json::array({json{{"role", "user"}, {"content", json::array()}}});
    for (int i = 0; i < 9; ++i)
        too_many[0]["content"].push_back(
            json{{"type", "input_image"}, {"image_url", data_url}});
    check(!vision::prepare_messages(too_many, &frames, &error) &&
              error.find("at most 8") != std::string::npos,
          "image-count-rejected", error);
    json video = json::array(
        {json{{"role", "user"},
              {"content", json::array(
                              {json{{"type", "input_video"}, {"video_url", data_url}}})}}});
    check(!vision::prepare_messages(video, &frames, &error) &&
              error.find("video input") != std::string::npos,
          "video-rejected", error);
    json file_id = json::array(
        {json{{"role", "user"},
              {"content", json::array(
                              {json{{"type", "input_image"}, {"file_id", "file_123"}}})}}});
    check(!vision::prepare_messages(file_id, &frames, &error) &&
              error.find("file_id") != std::string::npos,
          "file-id-rejected", error);

    gdec::Tokenizer tokenizer;
    error.clear();
    check(tokenizer.load(argv[2], &error), "tokenizer-load", error);
    vision::Frame placeholder;
    placeholder.grid = {1, 4, 4};
    frames = {placeholder};
    std::vector<int> ids;
    const std::string prompt =
        "before<|vision_start|><|image_pad|><|vision_end|>after";
    error.clear();
    check(vision::encode_prompt(prompt, tokenizer, frames, &ids, &error),
          "placeholder-expand", error);
    check(std::count(ids.begin(), ids.end(), 248056) == placeholder.pad_tokens(),
          "placeholder-token-count");
    frames.push_back(placeholder);
    check(!vision::encode_prompt(prompt, tokenizer, frames, &ids, &error) &&
              error.find("placeholder count") != std::string::npos,
          "placeholder-mismatch-rejected", error);

    std::puts("RESULT PASS");
    return 0;
}
