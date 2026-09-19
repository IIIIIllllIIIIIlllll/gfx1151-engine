// Qwen image preprocessing derived from the public Apache-2.0 Transformers
// Qwen2-VL processor. Pillow-compatible bicubic resize is implemented locally.
#include "vision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>

#ifdef _WIN32
// Windows 构建不链 libpng/libjpeg：改用 stb_image（public domain 单头文件，
// 编译进 exe，零新增 DLL）解码 PNG/JPEG，保持产物自包含。WebP 无等价单头
// 实现，decode_image 里对其明确报错。
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "vendor/stb_image.h"
#else
#include <setjmp.h>

#include <jpeglib.h>
#include <png.h>
#include <webp/decode.h>
#endif

namespace vision {
namespace {

constexpr size_t kMaxEncodedBytes = 64u << 20;
constexpr uint64_t kMaxDecodedPixels = 100000000u;
constexpr int kMaxImages = 8;
constexpr int kPatch = 16;
constexpr int kMerge = 2;
constexpr int kFactor = kPatch * kMerge;
constexpr int kMinPixels = 256 * 256;
constexpr int kMaxPixels = 2560 * 1440;
constexpr int kImageToken = 248056;
constexpr const char* kPadSpan =
    "<|vision_start|><|image_pad|><|vision_end|>";

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

bool checked_rgb_size(int width, int height, size_t* bytes, std::string* error) {
    if (width <= 0 || height <= 0 ||
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > kMaxDecodedPixels) {
        *error = "image dimensions are invalid or too large";
        return false;
    }
    const uint64_t n = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 3u;
    if (n > std::numeric_limits<size_t>::max()) {
        *error = "image dimensions overflow memory limits";
        return false;
    }
    *bytes = static_cast<size_t>(n);
    return true;
}

#ifndef _WIN32
bool decode_png(const std::vector<uint8_t>& bytes, Image* out, std::string* error) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&image, bytes.data(), bytes.size())) {
        *error = "invalid PNG image";
        return false;
    }
    image.format = PNG_FORMAT_RGB;
    size_t size = 0;
    if (!checked_rgb_size(static_cast<int>(image.width), static_cast<int>(image.height),
                          &size, error)) {
        png_image_free(&image);
        return false;
    }
    out->width = static_cast<int>(image.width);
    out->height = static_cast<int>(image.height);
    out->rgb.resize(size);
    if (!png_image_finish_read(&image, nullptr, out->rgb.data(), 0, nullptr)) {
        *error = "failed to decode PNG image";
        png_image_free(&image);
        return false;
    }
    png_image_free(&image);
    return true;
}

struct JpegError {
    jpeg_error_mgr base;
    jmp_buf jump;
};

void jpeg_fail(j_common_ptr info) {
    auto* state = reinterpret_cast<JpegError*>(info->err);
    longjmp(state->jump, 1);
}

bool decode_jpeg(const std::vector<uint8_t>& bytes, Image* out, std::string* error) {
    jpeg_decompress_struct info{};
    JpegError state{};
    info.err = jpeg_std_error(&state.base);
    state.base.error_exit = jpeg_fail;
    if (setjmp(state.jump)) {
        jpeg_destroy_decompress(&info);
        *error = "invalid JPEG image";
        return false;
    }
    jpeg_create_decompress(&info);
    jpeg_mem_src(&info, bytes.data(), bytes.size());
    jpeg_read_header(&info, TRUE);
    info.out_color_space = JCS_RGB;
    jpeg_start_decompress(&info);
    size_t size = 0;
    if (!checked_rgb_size(static_cast<int>(info.output_width),
                          static_cast<int>(info.output_height), &size, error)) {
        jpeg_destroy_decompress(&info);
        return false;
    }
    out->width = static_cast<int>(info.output_width);
    out->height = static_cast<int>(info.output_height);
    out->rgb.resize(size);
    while (info.output_scanline < info.output_height) {
        JSAMPROW row = out->rgb.data() +
                       static_cast<size_t>(info.output_scanline) * out->width * 3u;
        jpeg_read_scanlines(&info, &row, 1);
    }
    jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    return true;
}

bool decode_webp(const std::vector<uint8_t>& bytes, Image* out, std::string* error) {
    int width = 0, height = 0;
    if (!WebPGetInfo(bytes.data(), bytes.size(), &width, &height)) {
        *error = "invalid WebP image";
        return false;
    }
    size_t size = 0;
    if (!checked_rgb_size(width, height, &size, error)) return false;
    out->width = width;
    out->height = height;
    out->rgb.resize(size);
    if (WebPDecodeRGBInto(bytes.data(), bytes.size(), out->rgb.data(), out->rgb.size(),
                          width * 3) == nullptr) {
        *error = "failed to decode WebP image";
        return false;
    }
    return true;
}

bool decode_image(const std::vector<uint8_t>& bytes, Image* out, std::string* error) {
    if (bytes.size() >= 8 && png_sig_cmp(bytes.data(), 0, 8) == 0)
        return decode_png(bytes, out, error);
    if (bytes.size() >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8)
        return decode_jpeg(bytes, out, error);
    if (bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 &&
        std::memcmp(bytes.data() + 8, "WEBP", 4) == 0)
        return decode_webp(bytes, out, error);
    *error = "unsupported image format; expected PNG, JPEG, or WebP";
    return false;
}
#else
bool decode_image(const std::vector<uint8_t>& bytes, Image* out, std::string* error) {
    if (bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 &&
        std::memcmp(bytes.data() + 8, "WEBP", 4) == 0) {
        *error = "WebP images are not supported in this Windows build; "
                 "send PNG or JPEG instead";
        return false;
    }
    int width = 0, height = 0;
    unsigned char* pixels =
        stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                              &width, &height, nullptr, 3);
    if (pixels == nullptr) {
        const char* reason = stbi_failure_reason();
        *error = std::string("failed to decode image (PNG/JPEG): ") +
                 (reason ? reason : "unknown error");
        return false;
    }
    size_t size = 0;
    if (!checked_rgb_size(width, height, &size, error)) {
        stbi_image_free(pixels);
        return false;
    }
    out->width = width;
    out->height = height;
    out->rgb.assign(pixels, pixels + size);
    stbi_image_free(pixels);
    return true;
}
#endif

int round_half_even(int value, int factor) {
    const int whole = value / factor;
    const int remainder = value % factor;
    if (remainder * 2 < factor) return whole;
    if (remainder * 2 > factor) return whole + 1;
    return (whole & 1) == 0 ? whole : whole + 1;
}

bool smart_resize(int height, int width, int* out_h, int* out_w, std::string* error) {
    const double ratio = static_cast<double>(std::max(height, width)) /
                         static_cast<double>(std::min(height, width));
    if (ratio > 200.0) {
        *error = "absolute image aspect ratio must be smaller than 200";
        return false;
    }
    int h = std::max(kFactor, round_half_even(height, kFactor) * kFactor);
    int w = std::max(kFactor, round_half_even(width, kFactor) * kFactor);
    if (static_cast<int64_t>(h) * w > kMaxPixels) {
        const double beta = std::sqrt(static_cast<double>(height) * width / kMaxPixels);
        h = std::max(kFactor,
                     static_cast<int>(std::floor(height / beta / kFactor)) * kFactor);
        w = std::max(kFactor,
                     static_cast<int>(std::floor(width / beta / kFactor)) * kFactor);
    } else if (h * w < kMinPixels) {
        const double beta = std::sqrt(static_cast<double>(kMinPixels) /
                                      (static_cast<double>(height) * width));
        h = static_cast<int>(std::ceil(height * beta / kFactor)) * kFactor;
        w = static_cast<int>(std::ceil(width * beta / kFactor)) * kFactor;
    }
    *out_h = h;
    *out_w = w;
    return true;
}

double bicubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a);
    return 0.0;
}

struct Coefficients {
    int kernel_size = 0;
    std::vector<int> bounds;
    std::vector<int32_t> weights;
};

Coefficients precompute_coefficients(int in_size, int out_size) {
    constexpr int kPrecision = 22;
    const double scale = static_cast<double>(in_size) / out_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    Coefficients result;
    result.kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    result.bounds.resize(static_cast<size_t>(out_size) * 2u);
    result.weights.assign(static_cast<size_t>(out_size) * result.kernel_size, 0);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        const int count = xmax - xmin;
        result.bounds[static_cast<size_t>(xx) * 2u] = xmin;
        result.bounds[static_cast<size_t>(xx) * 2u + 1u] = count;
        double sum = 0.0;
        std::vector<double> weights(static_cast<size_t>(count));
        for (int x = 0; x < count; ++x) {
            const double weight = bicubic((x + xmin - center + 0.5) / filter_scale);
            weights[static_cast<size_t>(x)] = weight;
            sum += weight;
        }
        if (sum == 0.0) continue;
        for (int x = 0; x < count; ++x) {
            const double scaled = weights[static_cast<size_t>(x)] / sum * (1 << kPrecision);
            result.weights[static_cast<size_t>(xx) * result.kernel_size + x] =
                static_cast<int32_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
        }
    }
    return result;
}

uint8_t clip8(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return static_cast<uint8_t>(value);
}

Image resize_bicubic(const Image& input, int out_w, int out_h) {
    constexpr int kPrecision = 22;
    const Coefficients x = precompute_coefficients(input.width, out_w);
    const Coefficients y = precompute_coefficients(input.height, out_h);
    std::vector<uint8_t> horizontal(static_cast<size_t>(out_w) * input.height * 3u);
    for (int row = 0; row < input.height; ++row) {
        for (int ox = 0; ox < out_w; ++ox) {
            const int first = x.bounds[static_cast<size_t>(ox) * 2u];
            const int count = x.bounds[static_cast<size_t>(ox) * 2u + 1u];
            const int32_t* weights =
                x.weights.data() + static_cast<size_t>(ox) * x.kernel_size;
            for (int c = 0; c < 3; ++c) {
                int64_t sum = 1 << (kPrecision - 1);
                for (int i = 0; i < count; ++i) {
                    const size_t src =
                        (static_cast<size_t>(row) * input.width + first + i) * 3u + c;
                    sum += static_cast<int64_t>(input.rgb[src]) * weights[i];
                }
                horizontal[(static_cast<size_t>(row) * out_w + ox) * 3u + c] =
                    clip8(static_cast<int>(sum >> kPrecision));
            }
        }
    }

    Image output;
    output.width = out_w;
    output.height = out_h;
    output.rgb.resize(static_cast<size_t>(out_w) * out_h * 3u);
    for (int oy = 0; oy < out_h; ++oy) {
        const int first = y.bounds[static_cast<size_t>(oy) * 2u];
        const int count = y.bounds[static_cast<size_t>(oy) * 2u + 1u];
        const int32_t* weights =
            y.weights.data() + static_cast<size_t>(oy) * y.kernel_size;
        for (int col = 0; col < out_w; ++col) {
            for (int c = 0; c < 3; ++c) {
                int64_t sum = 1 << (kPrecision - 1);
                for (int i = 0; i < count; ++i) {
                    const size_t src =
                        (static_cast<size_t>(first + i) * out_w + col) * 3u + c;
                    sum += static_cast<int64_t>(horizontal[src]) * weights[i];
                }
                output.rgb[(static_cast<size_t>(oy) * out_w + col) * 3u + c] =
                    clip8(static_cast<int>(sum >> kPrecision));
            }
        }
    }
    return output;
}

void patchify(const Image& image, Frame* frame) {
    constexpr float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    constexpr float stdev[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    const int gh = image.height / kPatch;
    const int gw = image.width / kPatch;
    frame->grid = {1, gh, gw};
    frame->patches.resize(static_cast<size_t>(gh) * gw * 3u * 2u * kPatch * kPatch);
    size_t out = 0;
    for (int block_h = 0; block_h < gh / kMerge; ++block_h) {
        for (int block_w = 0; block_w < gw / kMerge; ++block_w) {
            for (int merge_h = 0; merge_h < kMerge; ++merge_h) {
                for (int merge_w = 0; merge_w < kMerge; ++merge_w) {
                    for (int channel = 0; channel < 3; ++channel) {
                        for (int temporal = 0; temporal < 2; ++temporal) {
                            (void)temporal;
                            for (int ph = 0; ph < kPatch; ++ph) {
                                const int y = (block_h * kMerge + merge_h) * kPatch + ph;
                                for (int pw = 0; pw < kPatch; ++pw) {
                                    const int x = (block_w * kMerge + merge_w) * kPatch + pw;
                                    const uint8_t pixel =
                                        image.rgb[(static_cast<size_t>(y) * image.width + x) * 3u +
                                                  channel];
                                    const float scaled = static_cast<float>(pixel) / 255.0f;
                                    frame->patches[out++] =
                                        (scaled - mean[channel]) / stdev[channel];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

bool base64_decode(std::string_view input, std::vector<uint8_t>* output) {
    if (input.empty() || input.size() % 4 != 0 || input.size() / 4 * 3 > kMaxEncodedBytes)
        return false;
    output->clear();
    output->reserve(input.size() / 4 * 3);
    for (size_t i = 0; i < input.size(); i += 4) {
        int value[4] = {-1, -1, -1, -1};
        for (int j = 0; j < 4; ++j) {
            if (input[i + j] == '=') value[j] = -2;
            else value[j] = base64_value(static_cast<unsigned char>(input[i + j]));
        }
        if (value[0] < 0 || value[1] < 0 || value[2] == -1 || value[3] == -1) return false;
        const bool last = i + 4 == input.size();
        if (value[2] == -2 && (!last || value[3] != -2)) return false;
        if (value[3] == -2 && !last) return false;
        output->push_back(static_cast<uint8_t>((value[0] << 2) | (value[1] >> 4)));
        if (value[2] >= 0) {
            output->push_back(
                static_cast<uint8_t>(((value[1] & 15) << 4) | (value[2] >> 2)));
            if (value[3] >= 0)
                output->push_back(
                    static_cast<uint8_t>(((value[2] & 3) << 6) | value[3]));
        }
    }
    return output->size() <= kMaxEncodedBytes;
}

bool image_url_from_item(const json& item, std::string* url, bool* image,
                         std::string* error) {
    *image = false;
    if (!item.is_object()) return true;
    std::string type;
    if (item.contains("type") && item["type"].is_string()) type = item["type"].get<std::string>();
    if (type == "video" || type == "input_video" || item.contains("video") ||
        item.contains("video_url")) {
        *error = "video input is not supported";
        return false;
    }
    if (type == "input_image" && item.contains("file_id") && !item["file_id"].is_null()) {
        *error = "image file_id input is not supported; send an inline base64 image";
        return false;
    }
    const json* value = nullptr;
    if (item.contains("image_url")) value = &item["image_url"];
    else if (item.contains("image")) value = &item["image"];
    else if (type == "image" || type == "input_image") {
        *error = "image content requires image_url";
        return false;
    } else {
        return true;
    }
    if (value->is_string()) {
        *url = value->get<std::string>();
    } else if (value->is_object() && value->contains("url") && (*value)["url"].is_string()) {
        *url = (*value)["url"].get<std::string>();
    } else {
        *error = "image_url must be a string or an object with a string url";
        return false;
    }
    *image = true;
    return true;
}

}  // namespace

bool decode_image_url(const std::string& value, std::vector<uint8_t>* bytes,
                      std::string* error) {
    error->clear();
    if (value.rfind("http://", 0) == 0 || value.rfind("https://", 0) == 0) {
        *error = "http(s) image URLs are not fetched: this server makes no outbound requests. "
                 "Send the image inline as a data: URL (data:image/png;base64,...) or as bare "
                 "base64.";
        return false;
    }
    std::string_view encoded(value);
    if (value.rfind("data:", 0) == 0) {
        const size_t comma = value.find(',');
        if (comma == std::string::npos || value.substr(5, comma - 5).find(";base64") ==
                                               std::string::npos) {
            *error = "image data URLs must use base64 encoding";
            return false;
        }
        encoded = std::string_view(value).substr(comma + 1);
    }
    if (!base64_decode(encoded, bytes)) {
        *error = "invalid or oversized base64 image payload";
        return false;
    }
    return true;
}

bool preprocess(const std::vector<uint8_t>& bytes, Frame* frame, std::string* error) {
    Image decoded;
    if (!decode_image(bytes, &decoded, error)) return false;
    int height = 0, width = 0;
    if (!smart_resize(decoded.height, decoded.width, &height, &width, error)) return false;
    Image resized = (height == decoded.height && width == decoded.width)
                        ? std::move(decoded)
                        : resize_bicubic(decoded, width, height);
    patchify(resized, frame);
    return true;
}

bool prepare_messages(const json& messages, std::vector<Frame>* frames,
                      std::string* error) {
    static std::mutex preprocess_mutex;
    std::lock_guard<std::mutex> lock(preprocess_mutex);
    frames->clear();
    error->clear();
    if (!messages.is_array()) return true;
    std::vector<std::string> urls;
    for (const auto& message : messages) {
        if (!message.is_object() || !message.contains("content") ||
            !message["content"].is_array())
            continue;
        for (const auto& item : message["content"]) {
            std::string url;
            bool is_image = false;
            if (!image_url_from_item(item, &url, &is_image, error)) return false;
            if (!is_image) continue;
            urls.push_back(std::move(url));
            if (urls.size() > kMaxImages) {
                *error = "a request may contain at most 8 images";
                return false;
            }
        }
    }
    frames->reserve(urls.size());
    for (const std::string& url : urls) {
        std::vector<uint8_t> bytes;
        if (!decode_image_url(url, &bytes, error)) return false;
        Frame frame;
        if (!preprocess(bytes, &frame, error)) return false;
        frames->push_back(std::move(frame));
    }
    return true;
}

bool encode_prompt(const std::string& prompt, const gdec::Tokenizer& tokenizer,
                   const std::vector<Frame>& frames, std::vector<int>* ids,
                   std::string* error) {
    std::vector<size_t> pad_offsets;
    size_t at = 0;
    const size_t pad_in_span = std::string("<|vision_start|>").size();
    while ((at = prompt.find(kPadSpan, at)) != std::string::npos) {
        pad_offsets.push_back(at + pad_in_span);
        at += std::char_traits<char>::length(kPadSpan);
    }
    if (pad_offsets.size() != frames.size()) {
        *error = "image placeholder count does not match the number of supplied images";
        return false;
    }

    const std::vector<gdec::TokenSpan> spans = tokenizer.encode_with_offsets(prompt);
    ids->clear();
    size_t image_index = 0;
    for (const auto& token : spans) {
        if (image_index < pad_offsets.size() && token.id == kImageToken &&
            token.start == pad_offsets[image_index]) {
            const int count = frames[image_index].pad_tokens();
            ids->insert(ids->end(), static_cast<size_t>(count), kImageToken);
            ++image_index;
        } else {
            ids->push_back(token.id);
        }
    }
    if (image_index != frames.size()) {
        *error = "the tokenizer did not preserve an image placeholder token";
        return false;
    }
    return true;
}

}  // namespace vision
