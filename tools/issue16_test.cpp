// 验证 Issue #16 修复：极端宽高比图片应被 letterbox 而非 400 拒绝。
// 用库存储块（stored deflate）手工构造 zlib/PNG，零外部依赖。
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vision.h"

static void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x >> 24));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x));
}

static uint32_t crc32_of(const uint8_t* data, size_t n) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (0xedb88320u & (~(crc & 1u) + 1u));
    }
    return ~crc;
}

static uint32_t adler32_of(const uint8_t* data, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

static void chunk(std::vector<uint8_t>& png, const char tag[4],
                  const std::vector<uint8_t>& data) {
    put32(png, static_cast<uint32_t>(data.size()));
    const size_t at = png.size();
    png.insert(png.end(), tag, tag + 4);
    png.insert(png.end(), data.begin(), data.end());
    put32(png, crc32_of(png.data() + at, 4 + data.size()));
}

static std::vector<uint8_t> make_png(int w, int h) {
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(h) * (1 + 3 * w));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);  // filter: none
        for (int x = 0; x < w; ++x) {
            raw.push_back(80);
            raw.push_back(120);
            raw.push_back(160);
        }
    }
    // zlib: header + stored blocks + adler32
    std::vector<uint8_t> z = {0x78, 0x01};
    size_t off = 0;
    while (off < raw.size()) {
        const size_t n = std::min<size_t>(65535, raw.size() - off);
        const bool last = off + n == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n));
        z.push_back(static_cast<uint8_t>(n >> 8));
        z.push_back(static_cast<uint8_t>(~n));
        z.push_back(static_cast<uint8_t>(~n >> 8));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    }
    put32(z, adler32_of(raw.data(), raw.size()));

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<uint8_t> ihdr;
    put32(ihdr, static_cast<uint32_t>(w));
    put32(ihdr, static_cast<uint32_t>(h));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    chunk(png, "IHDR", ihdr);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", {});
    return png;
}

int main() {
    struct Case { int w, h; };
    const Case cases[] = {{448, 448}, {1200, 5}, {4, 900}, {2000, 5}};
    int fails = 0;
    for (const auto& c : cases) {
        const std::vector<uint8_t> png = make_png(c.w, c.h);
        vision::Frame frame;
        std::string error;
        const bool ok = vision::preprocess(png, &frame, &error);
        if (ok) {
            printf("%dx%d -> OK grid={%d,%d,%d} pad_tokens=%d patches=%zu\n", c.w,
                   c.h, frame.grid[0], frame.grid[1], frame.grid[2],
                   frame.pad_tokens(), frame.patches.size());
        } else {
            printf("%dx%d -> REJECTED: %s\n", c.w, c.h, error.c_str());
            ++fails;
        }
    }
    printf(fails == 0 ? "ALL PASS\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
