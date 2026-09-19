// eng_cli.cpp — driver for the engine client.
//
//   eng_cli --dump-req      <json>   print the GEN request line (no socket)
//   eng_cli --host H:P ...  <json>   run the request against a live engine
//
// Each JSON request:
//   {"prompt":"...",            // tokenized with the model tokenizer
//    "ids":[...],               // ...or explicit token ids (wins over prompt)
//    "max_tokens":8, "temperature":0, "drafter":1, "seed":7,
//    "eos":[248046,248044], "logprobs":false, "req":1,
//    "snap":-1, "snap2":-1}
// Output: one JSON line with the reply tokens, decoded text and D-line stats.
#include <cstdio>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "engine_client.h"
#include "tokenizer.h"

using nlohmann::json;

static gdec::GenParams params_from_json(const json& q, gdec::Tokenizer* tok) {
    gdec::GenParams p;
    p.req = q.value("req", 1LL);
    p.max_tokens = q.value("max_tokens", 8);
    if (q.contains("eos")) p.eos = q["eos"].get<std::vector<int>>();
    else p.eos = {248046, 248044};
    if (q.contains("ids")) {
        p.ids = q["ids"].get<std::vector<int>>();
    } else if (q.contains("prompt") && tok != nullptr) {
        p.ids = tok->encode(q["prompt"].get<std::string>());
    }
    if (q.contains("drafter")) p.drafter = q["drafter"].get<int>();
    const double temp = q.value("temperature", 0.0);
    if (temp > 0.0) {
        p.sample = true;
        p.temp = (float)temp;
        p.top_k = q.value("top_k", 20);
        p.top_p = q.value("top_p", 0.95f);
        p.min_p = q.value("min_p", 0.0f);
        p.seed = q.value("seed", 0ULL);
    }
    p.presence = q.value("presence_penalty", 0.0f);
    p.frequency = q.value("frequency_penalty", 0.0f);
    if (q.contains("logit_bias")) {
        for (auto& [k, v] : q["logit_bias"].items())
            p.bias.emplace_back(std::stoi(k), v.get<float>());
    }
    p.logprobs = q.value("logprobs", false);
    if (q.contains("snap")) p.snap = q["snap"].get<long long>();
    if (q.contains("snap2")) p.snap2 = q["snap2"].get<long long>();
    return p;
}

int main(int argc, char** argv) {
    bool dump_only = false;
    std::string host = "127.0.0.1:8730";
    std::string tok_dir = "models/tokenizer";
    std::string req_line;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dump-req") dump_only = true;
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--tokenizer" && i + 1 < argc) tok_dir = argv[++i];
        else req_line = a;
    }
    if (req_line.empty()) {
        std::fprintf(stderr, "usage: %s [--dump-req] [--host H:P] '<json>'\n", argv[0]);
        return 2;
    }
    json q = json::parse(req_line);

    gdec::Tokenizer tok;
    std::string err;
    if (!tok.load(tok_dir, &err)) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
        return 1;
    }
    gdec::GenParams p = params_from_json(q, &tok);

    if (dump_only) {
        std::fputs(gdec::build_gen_request(p).c_str(), stdout);
        std::fprintf(stderr, "ids=%zu vimg=%zu\n", p.ids.size(), p.patches.size());
        return 0;
    }

    gdec::EngineClient eng;
    if (!eng.connect(host, &err)) {
        std::fprintf(stderr, "connect: %s\n", err.c_str());
        return 1;
    }
    std::string line;
    if (eng.info(&line, &err)) std::fprintf(stderr, "engine INFO: %s\n", line.c_str());
    else std::fprintf(stderr, "engine INFO failed: %s\n", err.c_str());

    gdec::GenResult r = eng.generate(p, nullptr);
    if (!r.transport_ok) {
        std::fprintf(stderr, "transport failure\n");
        return 1;
    }
    json out{
        {"reason", r.reason},
        {"prompt_tokens", r.n_prompt},
        {"gen_tokens", r.n_gen},
        {"prefill_ms", r.prefill_ms},
        {"decode_ms", r.decode_ms},
        {"drafter", r.drafter},
        {"rounds", r.rounds},
        {"commit", r.commit},
        {"n_cached", r.n_cached},
        {"proposed", r.proposed},
        {"ids", r.tokens},
        {"text", tok.decode(r.tokens, /*skip_special_tokens=*/false)},
    };
    std::printf("%s\n", out.dump().c_str());
    return r.reason == "error" ? 3 : 0;
}
