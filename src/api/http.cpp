// http.cpp — see http.h.
#include "http.h"

#ifdef _WIN32
#include "../gpu/os_win32.h"  // winsock2/ws2tcpip + sock_* + MSG_NOSIGNAL=0 + ssize_t
typedef int socklen_t;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

namespace http {
namespace {

#ifdef _WIN32
int sock_close_fd(sock_t fd) { return closesocket((SOCKET)fd); }
bool sock_intr() { return WSAGetLastError() == WSAEINTR; }
[[maybe_unused]] bool sock_wouldblock() { return WSAGetLastError() == WSAEWOULDBLOCK; }
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
[[maybe_unused]] bool sock_wouldblock() { return errno == EAGAIN || errno == EWOULDBLOCK; }
std::string sock_errstr() { return std::strerror(errno); }
ssize_t platform_send(sock_t fd, const void* p, size_t n) {
    return ::send(fd, p, n, MSG_NOSIGNAL);
}
ssize_t platform_recv(sock_t fd, void* p, size_t n) { return ::recv(fd, p, n, 0); }
#endif

constexpr size_t kMaxRequestBytes = 64u * 1024u * 1024u;  // 64 MB: base64 images
constexpr size_t kMaxHeaderBytes = 64u * 1024u;
constexpr size_t kMaxBufferedBytes = kMaxRequestBytes + kMaxHeaderBytes;
constexpr int kSocketTimeoutSec = 30;
constexpr int kMaxConnections = 128;

const char* status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 502: return "Bad Gateway";
        case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
        default: return "OK";
    }
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

std::string percent_decode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && std::isxdigit((unsigned char)s[i + 1]) &&
            std::isxdigit((unsigned char)s[i + 2])) {
            out.push_back((char)std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16));
            i += 2;
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

// Chunked-transfer SSE sink. Headers are written on the first frame so that a
// handler which throws before streaming can still return a JSON error.
class SseStream : public Stream {
  public:
    explicit SseStream(sock_t fd) : fd_(fd) {}

    bool send_event(const std::string& payload) override {
        return send_raw("data: " + payload + "\n\n");
    }

    bool send_raw(const std::string& bytes) override {
        if (closed_) return false;
        if (!started_ && !write_head()) return false;
        if (bytes.empty()) return true;
        char hdr[32];
        int n = std::snprintf(hdr, sizeof hdr, "%zx\r\n", bytes.size());
        if (!write_all(hdr, (size_t)n)) return false;
        if (!write_all(bytes.data(), bytes.size())) return false;
        return write_all("\r\n", 2);
    }

    bool finish() {
        if (closed_) return false;
        if (!started_ && !write_head()) return false;
        return write_all("0\r\n\r\n", 5);
    }

    bool disconnected() const override { return closed_ || peer_gone(); }
    bool started() const override { return started_; }

    std::string header(const std::string& name) const override {
        auto it = headers_.find(to_lower(name));
        return it == headers_.end() ? std::string() : it->second;
    }
    void set_headers(const std::map<std::string, std::string>& h) { headers_ = h; }
    // Headers the handler put on its Response before streaming (framing
    // headers are fixed here and skipped).
    void set_response_headers(const std::vector<std::pair<std::string, std::string>>* h) {
        extra_ = h;
    }

  private:
    bool write_head() {
        std::string head =
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
            "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n";
        if (extra_ != nullptr) {
            for (const auto& [k, v] : *extra_) {
                const std::string lk = to_lower(k);
                if (lk == "content-type" || lk == "cache-control" || lk == "connection" ||
                    lk == "content-length" || lk == "transfer-encoding")
                    continue;
                head += k + ": " + v + "\r\n";
            }
        }
        head += "Connection: ";
        head += keep_alive_ ? "keep-alive" : "close";
        head += "\r\n\r\n";
        if (!write_all(head.data(), head.size())) return false;
        started_ = true;
        return true;
    }

    bool write_all(const void* data, size_t n) {
        const char* p = (const char*)data;
        size_t off = 0;
        while (off < n) {
            ssize_t w = platform_send(fd_, p + off, n - off);
            if (w <= 0) {
                if (sock_intr()) continue;
                closed_ = true;
                return false;
            }
            off += (size_t)w;
        }
        return true;
    }

    // EOF on a stream where the client sends nothing else == disconnect.
    bool peer_gone() const {
        if (fd_ < 0) return true;
        char c;
#ifdef _WIN32
        // WinSock 的 recv 无 MSG_DONTWAIT：临时切非阻塞做 PEEK，再切回。
        // 错误码必须在切回阻塞的 ioctlsocket 之前取走，成功调用可能改写 last-error。
        u_long nb = 1;
        ioctlsocket((SOCKET)fd_, FIONBIO, &nb);
        int r = ::recv((SOCKET)fd_, &c, 1, MSG_PEEK);
        int err = (r == SOCKET_ERROR) ? WSAGetLastError() : 0;
        nb = 0;
        ioctlsocket((SOCKET)fd_, FIONBIO, &nb);
        if (r == 0) return true;
        if (r == SOCKET_ERROR && err != WSAEWOULDBLOCK && err != WSAEINTR) return true;
        return false;
#else
        ssize_t r = ::recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT);
        if (r == 0) return true;
        if (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            return true;
        return false;
#endif
    }

  public:
    void set_keep_alive(bool v) { keep_alive_ = v; }

  private:
    sock_t fd_;
    bool closed_ = false;
    bool started_ = false;
    bool keep_alive_ = true;
    std::map<std::string, std::string> headers_;
    const std::vector<std::pair<std::string, std::string>>* extra_ = nullptr;
};

}  // namespace

void fail(int status, std::string message, std::string type, std::string code) {
    throw Error{status, std::move(message), std::move(type), std::move(code)};
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool Request::keep_alive() const {
    const std::string c = header("connection");
    if (version == "HTTP/1.0") return to_lower(c) == "keep-alive";
    return to_lower(c) != "close";
}

std::string error_json(const std::string& message, const std::string& type,
                       const std::string& code) {
    nlohmann::ordered_json e;
    e["error"] = {{"message", message},
                  {"type", type},
                  {"param", nullptr},
                  {"code", code == "null" ? nlohmann::ordered_json(nullptr)
                                          : nlohmann::ordered_json(code)}};
    return e.dump();
}

Server::~Server() {
    if (lfd_ >= 0) sock_close_fd(lfd_);
}

bool Server::listen(const std::string& host_port, std::string* err) {
    size_t colon = host_port.rfind(':');
    const std::string host = colon == std::string::npos ? "0.0.0.0" : host_port.substr(0, colon);
    const std::string port = colon == std::string::npos ? host_port : host_port.substr(colon + 1);

    lfd_ = (sock_t)::socket(AF_INET, SOCK_STREAM, 0);
    if (lfd_ < 0) {
        if (err) *err = std::string("socket: ") + sock_errstr();
        return false;
    }
    int one = 1;
    setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)std::atoi(port.c_str()));
    if (host.empty() || host == "0.0.0.0") addr.sin_addr.s_addr = htonl(INADDR_ANY);
    else if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        if (err) *err = "bad listen host: " + host;
        return false;
    }
    if (::bind(lfd_, (sockaddr*)&addr, sizeof addr) != 0 || ::listen(lfd_, 64) != 0) {
        if (err) *err = std::string("bind/listen: ") + sock_errstr();
        return false;
    }
    sockaddr_in got{};
    socklen_t gl = sizeof got;
    if (getsockname(lfd_, (sockaddr*)&got, &gl) == 0) port_ = ntohs(got.sin_port);
    return true;
}

void Server::on(const std::string& method, const std::string& path, Handler h) {
    routes_.push_back(Route{method, path, std::move(h)});
}

void Server::set_not_found(Handler h) { not_found_ = std::move(h); }

void Server::run() {
    running_ = true;
    while (running_) {
        sockaddr_in peer{};
        socklen_t pl = sizeof peer;
        sock_t fd = (sock_t)::accept(lfd_, (sockaddr*)&peer, &pl);
        if (fd < 0) {
            if (sock_intr()) continue;
            break;
        }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
#ifdef _WIN32
        // WinSock 的 RCVTIMEO/SNDTIMEO 取 DWORD 毫秒，不是 timeval
        DWORD timeout = kSocketTimeoutSec * 1000;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof timeout);
#else
        timeval timeout{kSocketTimeoutSec, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout);
#endif
        char ip[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof ip);
        std::string remote = std::string(ip) + ":" + std::to_string(ntohs(peer.sin_port));
        const int prior = active_connections_.fetch_add(1, std::memory_order_acq_rel);
        if (prior >= kMaxConnections) {
            active_connections_.fetch_sub(1, std::memory_order_acq_rel);
            Request req;
            req.version = "HTTP/1.1";
            req.headers["connection"] = "close";
            Response res;
            res.status = 503;
            res.body = error_json("server connection limit reached", "server_error",
                                  "server_error");
            write_response(fd, req, res);
            sock_close_fd(fd);
            continue;
        }
        try {
            std::thread([this, fd, remote] {
                serve_connection(fd, remote);
                active_connections_.fetch_sub(1, std::memory_order_acq_rel);
            }).detach();
        } catch (const std::exception&) {
            active_connections_.fetch_sub(1, std::memory_order_acq_rel);
            sock_close_fd(fd);
        }
    }
}

bool Server::write_all(sock_t fd, const void* data, size_t n) {
    const char* p = (const char*)data;
    size_t off = 0;
    while (off < n) {
        ssize_t w = platform_send(fd, p + off, n - off);
        if (w <= 0) {
            if (sock_intr()) continue;
            return false;
        }
        off += (size_t)w;
    }
    return true;
}

bool Server::read_into(sock_t fd, std::string* buf, size_t at_least) {
    while (buf->size() < at_least) {
        char tmp[65536];
        ssize_t r = platform_recv(fd, tmp, sizeof tmp);
        if (r == 0) return false;
        if (r < 0) {
            if (sock_intr()) continue;
            return false;
        }
        if (buf->size() + (size_t)r > kMaxBufferedBytes) return false;
        buf->append(tmp, (size_t)r);
    }
    return true;
}

bool Server::read_request(sock_t fd, std::string* in, Request* req,
                          int* error_status, std::string* err) {
    *error_status = 400;
    size_t hdr_end;
    for (;;) {
        hdr_end = in->find("\r\n\r\n");
        if (hdr_end != std::string::npos) break;
        if (in->size() > kMaxHeaderBytes) {
            *err = "headers too large";
            *error_status = 413;
            return false;
        }
        if (!read_into(fd, in, in->size() + 1)) {
            *err = in->empty() ? "eof" : "short read";
            return false;
        }
    }
    if (hdr_end > kMaxHeaderBytes) {
        *err = "headers too large";
        *error_status = 413;
        return false;
    }
    std::istringstream hs(in->substr(0, hdr_end));
    std::string line;
    if (!std::getline(hs, line)) {
        *err = "empty request";
        return false;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    {
        std::istringstream rl(line);
        std::string extra;
        if (!(rl >> req->method >> req->path_target >> req->version) || (rl >> extra) ||
            (req->version != "HTTP/1.1" && req->version != "HTTP/1.0")) {
            *err = "bad request line";
            return false;
        }
    }
    while (std::getline(hs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        size_t c = line.find(':');
        if (c == std::string::npos) {
            *err = "bad header";
            return false;
        }
        const std::string name = to_lower(trim(line.substr(0, c)));
        const std::string value = trim(line.substr(c + 1));
        if (name.empty()) {
            *err = "bad header";
            return false;
        }
        auto old = req->headers.find(name);
        if (old == req->headers.end()) {
            req->headers[name] = value;
        } else if (name == "content-length") {
            if (old->second != value) {
                *err = "conflicting content-length";
                return false;
            }
        } else {
            old->second += "," + value;
        }
    }
    {
        const std::string t = req->path_target;
        size_t q = t.find('?');
        req->path = percent_decode(q == std::string::npos ? t : t.substr(0, q));
        req->query = q == std::string::npos ? "" : t.substr(q + 1);
    }
    size_t body_len = 0;
    const std::string cl = req->header("content-length");
    if (!cl.empty()) {
        unsigned long long parsed = 0;
        const char* begin = cl.data();
        const char* end = begin + cl.size();
        const auto result = std::from_chars(begin, end, parsed);
        if (result.ec != std::errc() || result.ptr != end) {
            *err = "bad content-length";
            return false;
        }
        if (parsed > kMaxRequestBytes) {
            *err = "body too large";
            *error_status = 413;
            return false;
        }
        body_len = static_cast<size_t>(parsed);
    }
    const std::string transfer_encoding = to_lower(trim(req->header("transfer-encoding")));
    if (!transfer_encoding.empty() && transfer_encoding != "identity") {
        *err = "transfer-encoding is not supported";
        return false;
    }
    if (body_len > kMaxRequestBytes) {
        *err = "body too large";
        *error_status = 413;
        return false;
    }
    // 100-continue: only acknowledge after the framing has been validated.
    if (to_lower(req->header("expect")) == "100-continue" &&
        !write_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25)) {
        *err = "continue write failed";
        return false;
    }
    const size_t total = hdr_end + 4 + body_len;
    if (in->size() < total && !read_into(fd, in, total)) {
        *err = "short body";
        return false;
    }
    req->body = in->substr(hdr_end + 4, body_len);
    in->erase(0, total);
    return true;
}

bool Server::write_response(sock_t fd, const Request& req, const Response& res) {
    std::string out = "HTTP/1.1 " + std::to_string(res.status) + " " +
                      status_text(res.status) + "\r\n";
    bool has_ct = false, has_cl = false, has_conn = false;
    for (const auto& [k, v] : res.headers) {
        std::string lk = to_lower(k);
        if (lk == "content-type") has_ct = true;
        if (lk == "content-length") has_cl = true;
        if (lk == "connection") has_conn = true;
        out += k + ": " + v + "\r\n";
    }
    if (!has_ct) out += "Content-Type: " + res.content_type + "\r\n";
    if (!has_cl) out += "Content-Length: " + std::to_string(res.body.size()) + "\r\n";
    if (!has_conn)
        out += std::string("Connection: ") + (req.keep_alive() ? "keep-alive" : "close") + "\r\n";
    out += "\r\n";
    out += res.body;
    return write_all(fd, out.data(), out.size());
}

void Server::serve_connection(sock_t fd, const std::string& remote) {
    std::string pending;
    for (;;) {
        Request req;
        std::string err;
        int read_status = 400;
        if (!read_request(fd, &pending, &req, &read_status, &err)) {
            if (err != "eof") {
                Request error_req;
                error_req.version = "HTTP/1.1";
                error_req.headers["connection"] = "close";
                Response error_res;
                error_res.status = read_status;
                error_res.body = error_json(read_status == 413 ? "Payload Too Large"
                                                               : "Bad Request");
                write_response(fd, error_req, error_res);
            }
            break;
        }
        req.remote = remote;

        const Route* hit = nullptr;
        bool method_exists = false;
        for (const Route& r : routes_) {
            if (r.path == req.path) {
                method_exists = true;
                if (r.method == req.method) {
                    hit = &r;
                    break;
                }
            }
        }

        Response res;
        SseStream st(fd);
        st.set_headers(req.headers);
        st.set_response_headers(&res.headers);
        st.set_keep_alive(req.keep_alive());

        // A handler that threw: if the SSE headers are already on the wire the
        // status line is spent, so the error becomes a final event; otherwise
        // it is an ordinary JSON response with a real status code.
        auto finish_with_error = [&](int status, const std::string& message,
                                     const std::string& type,
                                     const std::string& code) -> bool {
            if (st.started()) {
                st.send_event(error_json(message, type, code));
                return st.finish();
            }
            res.sse = false;
            res.status = status;
            res.body = error_json(message, type, code);
            return write_response(fd, req, res);
        };

        bool ok = true;
        try {
            if (hit != nullptr) {
                hit->handler(req, &res, &st);
            } else if (method_exists) {
                res.status = 405;
                res.body = error_json("Method Not Allowed");
            } else if (not_found_) {
                not_found_(req, &res, &st);
            } else {
                res.status = 404;
                res.body = error_json("Not Found", "not_found_error");
            }
            if (res.sse) ok = st.finish();
            else ok = write_response(fd, req, res);
        } catch (const Error& e) {
            ok = finish_with_error(e.status, e.message, e.type, e.code);
        } catch (const std::exception& e) {
            ok = finish_with_error(500, std::string("internal error: ") + e.what(),
                                   "server_error", "server_error");
        }
        if (!ok || !req.keep_alive()) break;
    }
    sock_close_fd(fd);
}

}  // namespace http
