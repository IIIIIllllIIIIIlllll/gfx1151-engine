// http_selftest.cpp — exercises the hand-written HTTP layer without an engine.
// Routes mirror the shapes main.cpp serves, so curl output is directly
// comparable with reference/GDEC-API-SHAPES.md.
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "http.h"

using nlohmann::ordered_json;

int main(int argc, char** argv) {
    http::Server srv;
    std::string err;
    const std::string addr = argc > 1 ? argv[1] : "127.0.0.1:8799";
    if (!srv.listen(addr, &err)) {
        std::fprintf(stderr, "listen: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "selftest listening on %s (port %d)\n", addr.c_str(), srv.port());

    srv.on("GET", "/v1/models", [](const http::Request&, http::Response* r, http::Stream*) {
        ordered_json j{{"object", "list"},
                       {"data", {{{"id", "qwen3.8-flash-next"},
                                  {"object", "model"}, {"owned_by", "local"},
                                  {"created", 0}}}}};
        r->body = j.dump();
    });

    srv.on("POST", "/echo", [](const http::Request& q, http::Response* r, http::Stream*) {
        ordered_json j{{"method", q.method},
                       {"path", q.path},
                       {"query", q.query},
                       {"content_type", q.header("content-type")},
                       {"body_len", q.body.size()},
                       {"body_head", q.body.substr(0, 64)}};
        r->body = j.dump();
    });

    srv.on("GET", "/error", [](const http::Request&, http::Response*, http::Stream*) {
        http::fail(400, "the engine rejected this request. Check /health for what "
                        "this build supports, and that the prompt fits the context.");
    });

    // SSE: 5 frames, 200 ms apart, then [DONE] — the shape main.cpp streams.
    srv.on("POST", "/sse", [](const http::Request&, http::Response* r, http::Stream* st) {
        r->sse = true;
        for (int i = 0; i < 5; i++) {
            ordered_json j{{"id", "chatcmpl-0123456789abcdef01234567"},
                           {"created", 1789371456},
                           {"object", "chat.completion.chunk"},
                           {"choices", {{{"index", 0}, {"finish_reason", nullptr},
                                         {"delta", {{"content", "tok" + std::to_string(i)}}}}}}};
            if (!st->send_event(j.dump())) return;  // client gone
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        ordered_json fin{{"id", "chatcmpl-0123456789abcdef01234567"},
                         {"created", 1789371456},
                         {"object", "chat.completion.chunk"},
                         {"choices", {{{"index", 0}, {"finish_reason", "length"},
                                       {"delta", ordered_json::object()}}}}};
        st->send_event(fin.dump());
        st->send_event("[DONE]");
    });

    // Slow stream: reports whether a client disconnect is observed, which is
    // what lets main.cpp send `X <req>` to the engine.
    srv.on("GET", "/slow", [](const http::Request&, http::Response* r, http::Stream* st) {
        r->sse = true;
        for (int i = 0; i < 50; i++) {
            ordered_json j{{"tick", i}, {"disconnected", st->disconnected()}};
            if (!st->send_event(j.dump())) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // Stream that fails after the headers are already out: must surface the
    // error as a final SSE event, not as a broken chunked body.
    srv.on("GET", "/sse-error", [](const http::Request&, http::Response* r, http::Stream* st) {
        r->sse = true;
        st->send_event("{\"tick\":0}");
        http::fail(400, "stream failed mid-flight");
    });

    srv.run();
    return 0;
}
