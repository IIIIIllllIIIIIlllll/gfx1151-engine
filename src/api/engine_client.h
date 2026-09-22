// engine_client.h — client for the gdec engine's line protocol (port 8730).
//
// Wire format is defined by the engine implementation in
// src/gpu/gdec.cpp:11220-11860 (its parse_gen + the GEN handler).
//
// One TCP connection, one in-flight request (the engine is batch-1: its INFO
// reply reports kv_slots = 1). The front-end therefore serializes generations
// with its own slot semaphore; cancelling is the exception -- `X <req>` is
// written out-of-band while a request is decoding, because the engine polls
// the same socket for it between tokens.
#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "sockt.h"

namespace gdec {

struct GenParams {
    long long req = 0;
    int max_tokens = 0;                  // must be > 0 (engine rejects <= 0)
    std::vector<int> eos;
    std::vector<int> ids;                // prompt; must be non-empty
    int drafter = -1;                    // -1 = omit; 0 serial / 1 mtp / 2 dflash2
    bool sample = false;                 // engine rejects SAMPLE with temp <= 0
    float temp = 0.0f;
    int top_k = 0;
    float top_p = 0.0f;
    float min_p = 0.0f;
    unsigned long long seed = 0;         // REQUIRED with SAMPLE
    float presence = 0.0f;
    float frequency = 0.0f;
    std::vector<std::pair<int, float>> bias;   // <= 20480, tid < 248320
    bool logprobs = false;               // sampler-only, like PENALTY/BIAS
    std::vector<std::array<int, 3>> mrope_grids;  // MROPE k t h w ...
    std::vector<std::vector<float>> patches;      // VIMG: one (P*1536) fp32 per grid
    long long snap = -1;                 // -1 = omit
    long long snap2 = -1;
    std::vector<long long> snaps;        // SNAPS <n> c1..cn, empty = omit
};

struct GenResult {
    bool transport_ok = false;   // false => the socket died, not a protocol error
    bool timed_out = false;      // transport_ok == false because a read deadline passed
    std::string reason;          // done | length | cancel | error
    std::vector<int> tokens;
    std::vector<float> logprobs; // parallel to tokens when LOGPROBS was set
    long long n_prompt = 0;
    long long n_gen = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int drafter = 0;
    int rounds = 0;
    int commit = 0;
    int n_cached = 0;
    int proposed = 0;
};

// Called for every T line. Return false to send `X <req>` and stop reading
// (the engine still emits its D line, which generate() consumes).
using TokenFn = std::function<bool(int /*token*/, float /*logprob*/)>;
// Called periodically while the engine has not produced the next T line.
// Returning false cancels the request, but generate() still drains its D line.
using WaitFn = std::function<bool()>;

// The `GEN ...` request line for these params, newline included. Exposed so the
// grammar can be checked without a live engine (tools/eng_cli --dump-req).
std::string build_gen_request(const GenParams& p);

class EngineClient {
  public:
    EngineClient() = default;
    ~EngineClient();
    EngineClient(const EngineClient&) = delete;
    EngineClient& operator=(const EngineClient&) = delete;

    // host_port is "host:port" (IPv4 or hostname). Returns false with *err set.
    bool connect(const std::string& host_port, std::string* err);
    void close();
    bool connected() const { return fd_.load(std::memory_order_acquire) >= 0; }

    // Send PING and wait for PONG.
    bool ping(std::string* err);
    // Send INFO and return the raw `I ...` line (for context/drafter probing).
    bool info(std::string* line, std::string* err);
    // Send MEM and return the raw `M ...` engine-memory line.
    bool memory(std::string* line, std::string* err);
    // Send CSTAT and return the raw `C ...` line.
    bool cstat(std::string* line, std::string* err);

    // Run one GEN to completion (or until `on_token` asks to stop). Blocks.
    // `on_wait` is called at roughly one-second intervals while waiting for
    // prefill or the next token, so streaming front-ends can send heartbeats.
    // `on_token` may be empty. On transport failure returns a result with
    // transport_ok = false; a protocol-level rejection (D ... error) returns
    // transport_ok = true with reason == "error".
    GenResult generate(const GenParams& p, const TokenFn& on_token,
                       const WaitFn& on_wait = {});

    // Write `X <req>` without disturbing an in-flight read. Safe to call from
    // another thread (e.g. when the HTTP client disconnects).
    void cancel(long long req);

    // Timeouts: time to the first T line, then between subsequent lines.
    void set_timeouts(double first_token_s, double between_tokens_s) {
        first_timeout_ = first_token_s;
        next_timeout_ = between_tokens_s;
    }

  private:
    bool write_all(const void* data, size_t n);
    bool read_line(std::string* line, double timeout_s);
    bool send_ping_pong_c(const char* verb, std::string* line, std::string* err);
    void invalidate(sock_t bad_fd);

    std::atomic<sock_t> fd_{-1};
    mutable std::mutex write_mtx_;   // guards fd_ writes only
    std::mutex gen_mtx_;             // one generation at a time
    std::string buf_;                // read buffer
    bool read_timed_out_ = false;    // set by read_line, consumed by generate()
    double first_timeout_ = 1800.0;
    double next_timeout_ = 300.0;
};

}  // namespace gdec
