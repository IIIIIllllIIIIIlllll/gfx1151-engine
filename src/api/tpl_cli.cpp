// tpl_cli.cpp — stdin/stdout driver for chat_template A/B testing.
// Reads one JSON object per stdin line:
//   {"messages": [...], "tools": ...|absent, "add_generation_prompt": bool,
//    "reasoning_effort": "...", "enable_thinking": bool,
//    "preserve_thinking": bool, "add_vision_id": bool}
// Any option may be absent (= template-undefined; distinct from JSON null).
// Prints one JSON line per request:
//   {"ok": true, "text": "..."}  or  {"ok": false, "error": "..."}
#include <iostream>
#include <string>

#include "chat_template.h"

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        nlohmann::json resp;
        try {
            auto req = chat_template::json::parse(line);
            auto pick = [&req](const char* key) -> const chat_template::json* {
                auto it = req.find(key);
                return it != req.end() ? &*it : nullptr;
            };
            chat_template::Options opts;
            opts.tools = pick("tools");
            opts.add_generation_prompt = pick("add_generation_prompt");
            opts.reasoning_effort = pick("reasoning_effort");
            opts.enable_thinking = pick("enable_thinking");
            opts.preserve_thinking = pick("preserve_thinking");
            opts.preserve_reasoning = pick("preserve_reasoning");
            opts.auto_disable_thinking_with_tools = pick("auto_disable_thinking_with_tools");
            opts.tool_call_format = pick("tool_call_format");
            opts.max_tool_arg_chars = pick("max_tool_arg_chars");
            opts.max_tool_response_chars = pick("max_tool_response_chars");
            opts.add_vision_id = pick("add_vision_id");

            auto result = chat_template::render_chat_template(pick("messages"), opts);
            if (result.ok) {
                resp = {{"ok", true},
                        {"text", result.text},
                        {"thinking_enabled", result.thinking_enabled}};
            } else {
                resp = {{"ok", false}, {"error", result.error}};
            }
        } catch (const std::exception& e) {
            resp = {{"ok", false}, {"error", std::string("cli: ") + e.what()}};
        }
        std::cout << resp.dump() << "\n" << std::flush;
    }
    return 0;
}
