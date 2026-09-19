// http.h — minimal HTTP/1.1 server (hand-written: no cpp-httplib on this box).
//
// Covers exactly what the OpenAI front-end needs: request line + headers +
// Content-Length body, keep-alive, chunked SSE writes with flush, and client
// disconnect detection (so a dropped stream can cancel the engine request).
// One thread per connection; each connection serves requests sequentially.
//
// A handler elects streaming by setting Response::sse = true and writing
// frames through the Stream it was given; the SSE status line and headers are
// emitted lazily on the first frame, so a handler that throws before streaming
// still produces an ordinary JSON error response with a real status code.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "sockt.h"

namespace http {

std::string to_lower(std::string s);

// Error a handler can throw to produce the service's OpenAI-style envelope.
struct Error {
    int status;
    std::string message;
    std::string type;  // "invalid_request_error" | "server_error" | ...
    std::string code;  // "null" => JSON null
};

[[noreturn]] void fail(int status, std::string message,
                       std::string type = "invalid_request_error",
                       std::string code = "null");

struct Request {
    std::string method;
    std::string path_target;  // raw request target, e.g. "/v1/models?x=1"
    std::string path;         // percent-decoded, without the query string
    std::string query;        // raw, after '?'
    std::string version;
    std::map<std::string, std::string> headers;  // lowercased keys
    std::string body;
    std::string remote;

    std::string header(const std::string& name) const {
        auto it = headers.find(to_lower(name));
        return it == headers.end() ? std::string() : it->second;
    }
    bool keep_alive() const;
};

struct Response {
    int status = 200;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
    std::string content_type = "application/json";
    bool sse = false;  // set by the handler to switch to text/event-stream

    void set(const std::string& k, const std::string& v) { headers.emplace_back(k, v); }
};

// SSE sink. send_event() writes one `data: <payload>\n\n` frame and flushes;
// false means the write failed or the client is gone.
class Stream {
  public:
    virtual ~Stream() = default;
    virtual bool send_event(const std::string& payload) = 0;
    virtual bool send_raw(const std::string& bytes) = 0;
    virtual bool disconnected() const = 0;
    virtual bool started() const = 0;
    virtual std::string header(const std::string& name) const = 0;
};

class Server {
  public:
    // The third argument is the SSE sink; ignore it for ordinary responses.
    using Handler = std::function<void(const Request&, Response*, Stream*)>;

    ~Server();

    bool listen(const std::string& host_port, std::string* err);
    void on(const std::string& method, const std::string& path, Handler h);
    void set_not_found(Handler h);
    void run();  // blocks until stop()

    void stop() { running_ = false; }
    int port() const { return port_; }

  private:
    void serve_connection(sock_t fd, const std::string& remote);
    bool read_request(sock_t fd, std::string* pending, Request* req,
                      int* error_status, std::string* err);
    bool write_response(sock_t fd, const Request& req, const Response& res);
    bool write_all(sock_t fd, const void* data, size_t n);
    bool read_into(sock_t fd, std::string* buf, size_t at_least);

    struct Route {
        std::string method, path;
        Handler handler;
    };
    std::vector<Route> routes_;
    Handler not_found_;
    sock_t lfd_ = -1;
    int port_ = 0;
    std::atomic<bool> running_{false};
    std::atomic<int> active_connections_{0};
};

// {"error":{"message":...,"type":...,"param":null,"code":null}}
std::string error_json(const std::string& message,
                       const std::string& type = "invalid_request_error",
                       const std::string& code = "null");

}  // namespace http
