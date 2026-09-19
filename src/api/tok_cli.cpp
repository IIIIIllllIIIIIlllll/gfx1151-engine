// JSON-lines CLI driver for the tokenizer, used by tools/tok_ab.py.
// stdin:  {"op":"encode","text":"...","offsets":false}
//         {"op":"decode","ids":[...],"skip_special_tokens":true}
// stdout: {"ids":[...],("offsets":[[s,e],...])} or {"text":"..."} or {"error":...}
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "tokenizer.h"

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models/tokenizer";
    gdec::Tokenizer tok;
    std::string err;
    if (!tok.load(dir, &err)) {
        std::cerr << "load failed: " << err << "\n";
        return 1;
    }
    std::cerr << "tok_cli ready (dir=" << dir << ")\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        nlohmann::json out;
        try {
            nlohmann::json q = nlohmann::json::parse(line);
            std::string op = q["op"].get<std::string>();
            if (op == "encode") {
                bool want_offsets = q.value("offsets", false);
                auto spans = tok.encode_with_offsets(q["text"].get<std::string>());
                nlohmann::json ids = nlohmann::json::array();
                nlohmann::json offs = nlohmann::json::array();
                for (const auto& t : spans) {
                    ids.push_back(t.id);
                    if (want_offsets) offs.push_back({t.start, t.end});
                }
                out["ids"] = ids;
                if (want_offsets) out["offsets"] = offs;
            } else if (op == "decode") {
                bool skip = q.value("skip_special_tokens", true);
                out["text"] = tok.decode(q["ids"].get<std::vector<int>>(), skip);
            } else if (op == "specials") {
                out["eos_token_id"] = tok.eos_token_id();
                out["pad_token_id"] = tok.pad_token_id();
                out["image_pad_token_id"] = tok.added_token_id("<|image_pad|>");
                out["vocab_size"] = tok.vocab_size();
            } else {
                out["error"] = "unknown op: " + op;
            }
        } catch (const std::exception& e) {
            out["error"] = e.what();
        }
        std::cout << out.dump() << "\n" << std::flush;
    }
    return 0;
}
