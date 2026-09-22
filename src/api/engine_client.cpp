// engine_client.cpp — see engine_client.h.
#include "engine_client.h"

#ifdef _WIN32
#include "../gpu/os_win32.h"  // winsock2/ws2tcpip + sock_* + MSG_NOSIGNAL=0 + ssize_t
#define poll WSAPoll
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>

namespace gdec {
namespace {

#ifdef _WIN32
int sock_close_fd(sock_t fd) { return closesocket((SOCKET)fd); }
bool sock_intr() { return WSAGetLastError() == WSAEINTR; }
std::string sock_errstr() { return sock_strerror(WSAGetLastError()); }
ssize_t platform_send(sock_t fd, const void* p, size_t n) {
    return sock_send(fd, p, n);
}
ssize_t platform_recv(sock_t fd, void* p, size_t n) {
    return sock_recv(fd, p, n, false);
}
#else
int sock_close_fd(sock_t fd) { return ::close(fd); }
bool sock_intr() { return errno == EINTR; }
std::string sock_errstr() { return std::strerror(errno); }
ssize_t platform_send(sock_t fd, const void* p, size_t n) {
    return ::send(fd, p, n, MSG_NOSIGNAL);
}
ssize_t platform_recv(sock_t fd, void* p, size_t n) { return ::recv(fd, p, n, 0); }
#endif

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// %.9g: the protocol's float form. Never rounds 0.9999999 up to 1, which
// would silently disable the engine's nucleus filter.
std::string g9(float v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.9g", (double)v);
    return b;
}

bool valid_vision_payload(const GenParams& p) {
    if (p.mrope_grids.size() != p.patches.size()) return false;
    for (size_t i = 0; i < p.mrope_grids.size(); ++i) {
        const auto& grid = p.mrope_grids[i];
        if (grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0) return false;
        const uint64_t patches = static_cast<uint64_t>(grid[0]) *
                                 static_cast<uint64_t>(grid[1]) *
                                 static_cast<uint64_t>(grid[2]);
        if (patches > std::numeric_limits<uint32_t>::max() ||
            patches > std::numeric_limits<size_t>::max() / 1536u ||
            p.patches[i].size() != static_cast<size_t>(patches) * 1536u)
            return false;
    }
    return true;
}

void put_le32(uint8_t* output, uint32_t value) {
    for (int i = 0; i < 4; ++i) output[i] = static_cast<uint8_t>(value >> (i * 8));
}

void put_le64(uint8_t* output, uint64_t value) {
    for (int i = 0; i < 8; ++i) output[i] = static_cast<uint8_t>(value >> (i * 8));
}

}  // namespace

EngineClient::~EngineClient() { close(); }

void EngineClient::close() {
    {
        std::lock_guard<std::mutex> lk(write_mtx_);
        const sock_t fd = fd_.exchange(-1, std::memory_order_acq_rel);
        if (fd >= 0) sock_close_fd(fd);
    }
    buf_.clear();
}

void EngineClient::invalidate(sock_t bad_fd) {
    if (bad_fd < 0) return;
    std::lock_guard<std::mutex> lk(write_mtx_);
    if (fd_.load(std::memory_order_acquire) != bad_fd) return;
    fd_.store(-1, std::memory_order_release);
    sock_close_fd(bad_fd);
}

bool EngineClient::connect(const std::string& host_port, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    size_t colon = host_port.rfind(':');
    if (colon == std::string::npos) return fail("expected host:port, got " + host_port);
    const std::string host = host_port.substr(0, colon);
    const std::string port = host_port.substr(colon + 1);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res); rc != 0)
        return fail("getaddrinfo(" + host_port + "): " + gai_strerror(rc));

    sock_t fd = -1;
    for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        fd = (sock_t)::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, ai->ai_addr, (int)ai->ai_addrlen) == 0) break;
        sock_close_fd(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) return fail("connect(" + host_port + ") failed: " + sock_errstr());

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
    {
        std::lock_guard<std::mutex> lk(write_mtx_);
        const sock_t old = fd_.exchange(fd, std::memory_order_acq_rel);
        if (old >= 0) sock_close_fd(old);
        buf_.clear();
    }
    return true;
}

bool EngineClient::write_all(const void* data, size_t n) {
    std::lock_guard<std::mutex> lk(write_mtx_);
    const sock_t fd = fd_.load(std::memory_order_acquire);
    if (fd < 0) return false;
    const char* p = static_cast<const char*>(data);
    size_t off = 0;
    while (off < n) {
        ssize_t w = platform_send(fd, p + off, n - off);
        if (w <= 0) {
            if (sock_intr()) continue;
            if (fd_.load(std::memory_order_acquire) == fd) {
                fd_.store(-1, std::memory_order_release);
                sock_close_fd(fd);
            }
            return false;
        }
        off += (size_t)w;
    }
    return true;
}

// Reads one '\n'-terminated line, honouring a wall-clock deadline. Any bytes
// already buffered are consumed first.
bool EngineClient::read_line(std::string* line, double timeout_s) {
    const double deadline = now_s() + timeout_s;
    read_timed_out_ = false;
    for (;;) {
        size_t nl = buf_.find('\n');
        if (nl != std::string::npos) {
            *line = buf_.substr(0, nl);
            buf_.erase(0, nl + 1);
            return true;
        }
        const sock_t fd = fd_.load(std::memory_order_acquire);
        if (fd < 0) return false;
        double left = deadline - now_s();
        if (left <= 0) {
            read_timed_out_ = true;
            return false;
        }
        pollfd pfd{};
        pfd.fd = (decltype(pfd.fd))fd;
        pfd.events = POLLIN;
        int pr = ::poll(&pfd, 1, (int)(left * 1000.0));
        if (pr == 0) {  // deadline elapsed with no readable data
            read_timed_out_ = true;
            return false;
        }
        if (pr < 0) {
            if (sock_intr()) continue;
            return false;
        }
        char tmp[65536];
        ssize_t r = platform_recv(fd, tmp, sizeof tmp);
        if (r == 0) {
            invalidate(fd);
            return false;
        }
        if (r < 0) {
            if (sock_intr()) continue;
            invalidate(fd);
            return false;
        }
        buf_.append(tmp, (size_t)r);
    }
}

bool EngineClient::send_ping_pong_c(const char* verb, std::string* line, std::string* err) {
    std::lock_guard<std::mutex> lk(gen_mtx_);
    if (!connected()) {
        if (err) *err = "not connected";
        return false;
    }
    std::string req = std::string(verb) + "\n";
    if (!write_all(req.data(), req.size())) {
        if (err) *err = "write failed";
        return false;
    }
    // Skip anything that is not our reply: a stray T/D from a cancelled
    // request can still be in flight.
    for (;;) {
        std::string l;
        if (!read_line(&l, next_timeout_)) {
            if (err) *err = "timeout waiting for " + std::string(verb) + " reply";
            close();
            return false;
        }
        if (verb[0] == 'P' && l == "PONG") return true;
        if (verb[0] == 'I' && !l.empty() && l[0] == 'I') {
            *line = l;
            return true;
        }
        if (verb[0] == 'M' && !l.empty() && l[0] == 'M') {
            *line = l;
            return true;
        }
        if (verb[0] == 'C' && !l.empty() && l[0] == 'C') {
            *line = l;
            return true;
        }
    }
}

bool EngineClient::ping(std::string* err) { return send_ping_pong_c("PING", nullptr, err); }

bool EngineClient::info(std::string* line, std::string* err) {
    return send_ping_pong_c("INFO", line, err);
}

bool EngineClient::memory(std::string* line, std::string* err) {
    return send_ping_pong_c("MEM", line, err);
}

bool EngineClient::cstat(std::string* line, std::string* err) {
    return send_ping_pong_c("CSTAT", line, err);
}

void EngineClient::cancel(long long req) {
    if (!connected()) return;
    char b[64];
    int n = std::snprintf(b, sizeof b, "X %lld\n", req);
    write_all(b, (size_t)n);
}

std::string build_gen_request(const GenParams& p) {
    std::string s;
    s.reserve(64 + p.ids.size() * 8 + p.bias.size() * 16);
    char head[96];
    std::snprintf(head, sizeof head, "GEN %lld %d %zu", p.req, p.max_tokens, p.eos.size());
    s += head;
    for (int e : p.eos) s += " " + std::to_string(e);
    s += " " + std::to_string(p.ids.size());
    for (int id : p.ids) s += " " + std::to_string(id);
    if (p.drafter >= 0) s += " " + std::to_string(p.drafter);
    if (p.sample) {
        // SAMPLE <temp> <top_k> <top_p> <min_p> <seed:u64>
        s += " SAMPLE " + g9(p.temp) + " " + std::to_string(p.top_k) + " " +
             g9(p.top_p) + " " + g9(p.min_p) + " " + std::to_string(p.seed);
    }
    if (p.presence != 0.0f || p.frequency != 0.0f)
        s += " PENALTY " + g9(p.presence) + " " + g9(p.frequency);
    if (!p.bias.empty()) {
        s += " BIAS " + std::to_string(p.bias.size());
        for (const auto& b : p.bias) s += " " + std::to_string(b.first) + " " + g9(b.second);
    }
    if (p.logprobs) s += " LOGPROBS";
    if (!p.mrope_grids.empty()) {
        s += " MROPE " + std::to_string(p.mrope_grids.size());
        for (const auto& g : p.mrope_grids)
            s += " " + std::to_string(g[0]) + " " + std::to_string(g[1]) + " " +
                 std::to_string(g[2]);
    }
    if (!p.patches.empty()) s += " VIMG " + std::to_string(p.patches.size());
    if (p.snap >= 0) s += " SNAP " + std::to_string(p.snap);
    if (p.snap2 >= 0) s += " SNAP2 " + std::to_string(p.snap2);
    if (!p.snaps.empty()) {
        s += " SNAPS " + std::to_string(p.snaps.size());
        for (long long c : p.snaps) s += " " + std::to_string(c);
    }
    s += "\n";
    return s;
}

GenResult EngineClient::generate(const GenParams& p, const TokenFn& on_token,
                                 const WaitFn& on_wait) {
    GenResult out;
    std::lock_guard<std::mutex> lk(gen_mtx_);
    if (!valid_vision_payload(p)) {
        out.transport_ok = true;
        out.reason = "error";
        return out;
    }
    if (!connected()) return out;

    const std::string s = build_gen_request(p);
    if (!write_all(s.data(), s.size())) return out;

    // VIMG frames must follow the newline immediately: their bytes are not
    // line-framed, and the engine drains them before replying at all.
    for (const auto& patch : p.patches) {
        const uint32_t patches = static_cast<uint32_t>(patch.size() / 1536u);
        std::array<uint8_t, 16> header{};
        put_le32(header.data(), 0x56494D31u);
        put_le32(header.data() + 4, patches);
        put_le64(header.data() + 8, static_cast<uint64_t>(patch.size()) * sizeof(float));
        if (!write_all(header.data(), header.size())) return out;
        if (!patch.empty() && !write_all(patch.data(), patch.size() * sizeof(float)))
            return out;
    }

    // --- read T lines until the D terminator ------------------------------
    bool saw_any = false;
    bool cancelled = false;
    bool heartbeat = static_cast<bool>(on_wait);
    // Poll in short slices when a heartbeat callback is present.  This keeps
    // long prefill gaps observable without changing the overall deadline.
    auto read_with_wait = [&](std::string* line, double timeout_s) {
        const double deadline = now_s() + timeout_s;
        for (;;) {
            const double left = deadline - now_s();
            if (left <= 0.0) {
                read_timed_out_ = true;
                return false;
            }
            const double slice = heartbeat ? std::min(left, 1.0) : left;
            if (read_line(line, slice)) return true;
            if (!read_timed_out_ || !heartbeat || now_s() >= deadline) return false;
            if (!on_wait()) {
                heartbeat = false;
                cancelled = true;
                cancel(p.req);
            }
        }
    };
    for (;;) {
        std::string l;
        if (!read_with_wait(&l, saw_any ? next_timeout_ : first_timeout_)) {
            out.timed_out = read_timed_out_;
            if (out.timed_out) cancel(p.req);
            close();
            return out;
        }
        if (l.empty()) continue;
        std::istringstream ss(l);
        std::string kind;
        ss >> kind;
        if (kind == "T") {
            long long req = 0;
            int tok = 0;
            if (!(ss >> req >> tok)) continue;
            if (req != p.req) continue;
            float lp = 0.0f;
            bool has_lp = (bool)(ss >> lp);
            saw_any = true;
            out.tokens.push_back(tok);
            out.logprobs.push_back(has_lp ? lp : 0.0f);
            if (!cancelled && on_token && !on_token(tok, has_lp ? lp : 0.0f)) {
                cancelled = true;
                cancel(p.req);  // engine acknowledges with its own D line
            }
        } else if (kind == "D") {
            long long req = 0;
            if (!(ss >> req) || req != p.req) continue;
            std::vector<std::string> f;
            std::string x;
            while (ss >> x) f.push_back(x);
            out.transport_ok = true;
            out.reason = f.empty() ? "error" : f[0];
            auto num = [&](size_t i, double dflt) -> double {
                return i < f.size() ? std::atof(f[i].c_str()) : dflt;
            };
            // Every shape carries the same numeric tail (zeros where a field
            // does not apply): `D <req> cancel 0 0 0.0 0.0`,
            // `D <req> error <n_prompt> 0 0 0`, and the success forms.
            out.n_prompt = (long long)num(1, 0);
            out.n_gen = (long long)num(2, 0);
            out.prefill_ms = num(3, 0);
            out.decode_ms = num(4, 0);
            out.drafter = (int)num(5, 0);
            out.rounds = (int)num(6, 0);
            out.commit = (int)num(7, 0);
            out.n_cached = (int)num(8, 0);
            out.proposed = (int)num(9, 0);
            return out;
        }
        // PONG / I / C / anything else: not ours, drop.
    }
}

}  // namespace gdec
