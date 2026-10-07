// chat_template.cpp — see chat_template.h for the semantics notes.
#include "chat_template.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "json_py.h"

namespace chat_template {
namespace {

struct TemplateError {
    std::string message;
};

[[noreturn]] void raise(std::string message) { throw TemplateError{std::move(message)}; }

// ---------------------------------------------------------------------------
// Python type names (for error messages that must match CPython/jinja2).
// ---------------------------------------------------------------------------
const char* py_type_name(const json& v) {
    switch (v.type()) {
        case json::value_t::null: return "NoneType";
        case json::value_t::boolean: return "bool";
        case json::value_t::number_integer:
        case json::value_t::number_unsigned: return "int";
        case json::value_t::number_float: return "float";
        case json::value_t::string: return "str";
        case json::value_t::array: return "list";
        case json::value_t::object: return "dict";
        default: return "object";
    }
}

// ---------------------------------------------------------------------------
// UTF-8 helpers.
// ---------------------------------------------------------------------------
// Decodes the code point starting at s[i]; returns bytes consumed (>=1).
// Invalid bytes are consumed one at a time as Latin-1-ish placeholders;
// template inputs are JSON (guaranteed valid UTF-8), so this is a fallback.
size_t utf8_decode(const std::string& s, size_t i, uint32_t& cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { cp = c; return 1; }
    auto cont = [&](size_t k) -> uint32_t {
        return static_cast<unsigned char>(s[i + k]) & 0x3f;
    };
    if (c >= 0xc2 && c < 0xe0 && i + 1 < s.size()) {
        cp = ((c & 0x1f) << 6) | cont(1);
        return 2;
    }
    if (c >= 0xe0 && c < 0xf0 && i + 2 < s.size()) {
        cp = ((c & 0x0f) << 12) | (cont(1) << 6) | cont(2);
        return 3;
    }
    if (c >= 0xf0 && c < 0xf5 && i + 3 < s.size()) {
        cp = ((c & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3);
        return 4;
    }
    cp = c;
    return 1;
}

// ---------------------------------------------------------------------------
// Python str.strip() (header note 2).
// ---------------------------------------------------------------------------
bool py_isspace(uint32_t cp) {
    if (cp >= 0x09 && cp <= 0x0d) return true;
    if (cp >= 0x1c && cp <= 0x20) return true;
    if (cp >= 0x2000 && cp <= 0x200a) return true;
    switch (cp) {
        case 0x85: case 0xa0: case 0x1680: case 0x2028: case 0x2029:
        case 0x202f: case 0x205f: case 0x3000:
            return true;
        default:
            return false;
    }
}

std::string py_trim(const std::string& s) {
    size_t begin = 0, end = s.size();
    while (begin < end) {
        uint32_t cp;
        size_t n = utf8_decode(s, begin, cp);
        if (!py_isspace(cp)) break;
        begin += n;
    }
    while (end > begin) {
        size_t start = end - 1;
        while (start > begin && (static_cast<unsigned char>(s[start]) & 0xc0) == 0x80) --start;
        uint32_t cp;
        size_t n = utf8_decode(s, start, cp);
        if (start + n != end || !py_isspace(cp)) break;
        end = start;
    }
    return s.substr(begin, end - begin);
}

// Python repr/str and json.dumps formatting now live in json_py.cpp, shared
// with main.cpp's SSE frames and logprob fields so the two cannot drift.
// ---------------------------------------------------------------------------
// Jinja helpers.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Generic Jinja helpers.
// ---------------------------------------------------------------------------
bool truthy(const json* p) {
    if (p == nullptr) return false;
    switch (p->type()) {
        case json::value_t::null: return false;
        case json::value_t::boolean: return p->get<bool>();
        case json::value_t::number_integer: return p->get<int64_t>() != 0;
        case json::value_t::number_unsigned: return p->get<uint64_t>() != 0;
        case json::value_t::number_float: return p->get<double>() != 0.0;
        case json::value_t::string: return !p->get_ref<const std::string&>().empty();
        case json::value_t::array:
        case json::value_t::object: return !p->empty();
        default: return false;
    }
}

bool is_json_true(const json* p) { return p != nullptr && p->is_boolean() && p->get<bool>(); }

// Attribute access `obj.name`: only defined for dict key hits.
const json* get_attr(const json* obj, const char* key) {
    if (obj != nullptr && obj->is_object()) {
        auto it = obj->find(key);
        if (it != obj->end()) return &*it;
    }
    return nullptr;
}

bool str_eq(const json* p, const char* s) {
    return p != nullptr && p->is_string() && p->get_ref<const std::string&>() == s;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string option_string(const json* value, const char* fallback) {
    return value != nullptr && value->is_string() ? value->get_ref<const std::string&>()
                                                   : fallback;
}

std::string truncate_payload(const std::string& value, size_t limit) {
    if (!limit || value.size() <= limit) return value;
    return value.substr(0, limit) + "\n[TRUNCATED - original length " +
           std::to_string(value.size()) + " chars]";
}

bool option_true(const json* value) { return value != nullptr && is_json_true(value); }

size_t option_limit(const json* value) {
    if (value == nullptr || value->is_null()) return 0;
    if (value->is_number_unsigned()) return value->get<uint64_t>();
    if (value->is_number_integer()) {
        const auto n = value->get<int64_t>();
        return n > 0 ? static_cast<size_t>(n) : 0;
    }
    return 0;
}

bool is_tool_error(const std::string& content) {
    const std::string lower = lower_ascii(content);
    const std::string head = lower.substr(0, std::min<size_t>(120, lower.size()));
    const bool code_or_search = lower.find("throw new ") != std::string::npos ||
                                lower.find("throw error") != std::string::npos ||
                                lower.find("console.error") != std::string::npos ||
                                lower.find("logger.error") != std::string::npos ||
                                lower.find("logging.error") != std::string::npos ||
                                head.find("import ") != std::string::npos ||
                                head.find("def ") != std::string::npos ||
                                head.find("function ") != std::string::npos;
    const bool exit_zero = head.find("exit code: 0") != std::string::npos ||
                           head.find("process exited with code 0") != std::string::npos;
    const bool error_field_ok = head.find("\"error\": null") != std::string::npos ||
                                head.find("\"error\":null") != std::string::npos ||
                                head.find("\"error\": false") != std::string::npos ||
                                head.find("\"error\":false") != std::string::npos ||
                                head.find("\"error\": \"\"") != std::string::npos ||
                                head.find("\"error\":\"\"") != std::string::npos;
    const bool strong = (head.find("\"error\":") != std::string::npos && !error_field_ok) ||
                        head.find("\"status\": \"error\"") != std::string::npos ||
                        head.find("\"status\":\"error\"") != std::string::npos ||
                        head.find("traceback (most recent call last):") != std::string::npos ||
                        head.find("command not found") != std::string::npos ||
                        head.find("invalid syntax") != std::string::npos ||
                        head.find("fatal:") != std::string::npos ||
                        ((head.find("exit code: ") != std::string::npos ||
                          head.find("process exited with code") != std::string::npos) &&
                         !exit_zero) ||
                        head.rfind("exception:", 0) == 0 || head.rfind("failed to ", 0) == 0;
    const bool weak = head.find("error:") != std::string::npos ||
                      head.find("err!") != std::string::npos;
    const bool weak_suppressed = head.find("$ ") != std::string::npos ||
                                 head.find("took ") != std::string::npos || content.size() >= 600;
    return !code_or_search && (strong || (weak && !weak_suppressed));
}

bool structured_json_response(const std::string& content) {
    const std::string trimmed = py_trim(content);
    return !trimmed.empty() && (trimmed.front() == '{' || trimmed.front() == '[');
}

constexpr const char* kControlTags[] = {
    "<|think_off|>", "<|think_on|>", "<|think_low|>", "<|think_minimal|>",
    "<|think_medium|>", "<|think_high|>", "<|think_xhigh|>", "<|think_max|>",
    "<|think_ultracode|>", "<|think_extreme|>"};

std::string strip_control_tags(std::string value) {
    for (const char* tag : kControlTags) {
        size_t at = 0;
        const size_t len = std::strlen(tag);
        while ((at = value.find(tag, at)) != std::string::npos) value.erase(at, len);
    }
    return value;
}

void scan_control_text(const json& content, bool* thinking, std::string* effort) {
    std::vector<std::string> texts;
    if (content.is_string()) {
        texts.push_back(content.get_ref<const std::string&>());
    } else if (content.is_array()) {
        for (const auto& item : content) {
            if (item.is_string()) texts.push_back(item.get_ref<const std::string&>());
            else if (item.is_object() && item.contains("text") && item["text"].is_string())
                texts.push_back(item["text"].get_ref<const std::string&>());
        }
    }
    for (const auto& text : texts) {
        struct Marker { const char* tag; const char* effort; bool enabled; };
        static constexpr Marker markers[] = {
            {"<|think_off|>", "medium", false}, {"<|think_on|>", "medium", true},
            {"<|think_low|>", "low", true}, {"<|think_minimal|>", "low", true},
            {"<|think_medium|>", "medium", true}, {"<|think_high|>", "xhigh", true},
            {"<|think_xhigh|>", "xhigh", true}, {"<|think_max|>", "xhigh", true},
            {"<|think_ultracode|>", "xhigh", true}, {"<|think_extreme|>", "xhigh", true}};
        size_t pos = 0;
        while (pos < text.size()) {
            size_t best = std::string::npos;
            const Marker* hit = nullptr;
            for (const auto& marker : markers) {
                const size_t at = text.find(marker.tag, pos);
                if (at != std::string::npos && (best == std::string::npos || at < best)) {
                    best = at;
                    hit = &marker;
                }
            }
            if (!hit) break;
            *thinking = hit->enabled;
            *effort = hit->effort;
            pos = best + std::strlen(hit->tag);
        }
    }
}

void scan_control_tags(const json& messages, bool* thinking, std::string* effort) {
    for (const auto& message : messages) {
        if (!message.is_object()) continue;
        const auto role = get_attr(&message, "role");
        if (!str_eq(role, "system") && !str_eq(role, "developer") && !str_eq(role, "user"))
            continue;
        if (const auto* content = get_attr(&message, "content"))
            scan_control_text(*content, thinking, effort);
    }
}

struct ExtractedReasoning {
    std::string reasoning;
    std::string content;
};

ExtractedReasoning extract_reasoning(const std::string& raw, const json* explicit_value) {
    ExtractedReasoning result;
    result.content = raw;
    if (explicit_value != nullptr && !explicit_value->is_null()) {
        result.reasoning = explicit_value->is_string() ? explicit_value->get<std::string>()
                                                        : json_py::py_str(*explicit_value);
        const char* close = nullptr;
        if (result.content.rfind("<think>", 0) == 0) close = "</think>";
        else if (result.content.rfind("<thinking>", 0) == 0) close = "</thinking>";
        else if (result.content.rfind("</think>", 0) == 0) close = "</think>";
        else if (result.content.rfind("</thinking>", 0) == 0) close = "</thinking>";
        if (close != nullptr) {
            const size_t at = result.content.find(close);
            if (at != std::string::npos) result.content = result.content.substr(at + std::strlen(close));
        }
    } else {
        struct Close { const char* value; const char* open; };
        static constexpr Close closes[] = {{"</think>", "<think>"}, {"</thinking>", "<thinking>"},
                                           {"</ think>", "<think>"}, {"</think >", "<think>"}};
        size_t end = std::string::npos;
        const Close* hit = nullptr;
        for (const auto& candidate : closes) {
            const size_t at = result.content.find(candidate.value);
            if (at != std::string::npos && (end == std::string::npos || at < end)) {
                end = at;
                hit = &candidate;
            }
        }
        if (hit != nullptr) {
            result.reasoning = result.content.substr(0, end);
            if (result.reasoning.rfind(hit->open, 0) == 0)
                result.reasoning.erase(0, std::strlen(hit->open));
            result.content.erase(0, end + std::strlen(hit->value));
        }
    }
    result.reasoning = py_trim(result.reasoning);
    while (!result.content.empty() && result.content.front() == '\n') result.content.erase(0, 1);
    return result;
}

// Python `needle in item` for a string needle (header note 5).
bool py_in(const char* needle, const json& item) {
    if (item.is_string()) {
        return item.get_ref<const std::string&>().find(needle) != std::string::npos;
    }
    if (item.is_object()) return item.contains(needle);
    if (item.is_array()) {
        for (const auto& e : item) {
            if (e.is_string() && e.get_ref<const std::string&>() == needle) return true;
        }
        return false;
    }
    raise(std::string("argument of type '") + py_type_name(item) + "' is not iterable");
}

// ---------------------------------------------------------------------------
// Renderer — mirrors the template top to bottom.
// ---------------------------------------------------------------------------
class Renderer {
  public:
    explicit Renderer(const Options& opts) : opts_(opts) {
        add_vision_id_ = truthy(opts.add_vision_id);
    }

    std::string run(const json* messages) {
        std::string out;

        if (!truthy(messages)) raise("No messages provided.");
        if (!messages->is_array()) {
            // Not reachable for well-formed requests; Python would fail on
            // messages[0] / iteration in version-specific ways.
            raise("No messages provided.");
        }
        const auto& msgs = *messages;
        const size_t n = msgs.size();

        thinking_ = opts_.enable_thinking == nullptr || is_json_true(opts_.enable_thinking);
        effort_ = "medium";
        if (opts_.reasoning_effort != nullptr && !opts_.reasoning_effort->is_null()) {
            if (!opts_.reasoning_effort->is_string())
                raise("Unexpected reasoning effort " + json_py::py_str(*opts_.reasoning_effort) +
                      ". Supported types are none, off, minimal, low, medium, high, and xhigh.");
            const std::string effort = lower_ascii(
                opts_.reasoning_effort->get_ref<const std::string&>());
            if (effort == "none" || effort == "off") {
                thinking_ = false;
            } else if (effort == "minimal" || effort == "low") {
                effort_ = "low";
            } else if (effort == "high" || effort == "max" || effort == "ultracode" ||
                       effort == "extreme" || effort == "xhigh") {
                effort_ = "xhigh";
            } else if (effort != "medium") {
                raise("Unexpected reasoning effort " + json_py::py_str(*opts_.reasoning_effort) +
                      ". Supported types are none, off, minimal, low, medium, high, and xhigh.");
            }
        }
        if (option_true(opts_.auto_disable_thinking_with_tools) &&
            opts_.tools != nullptr && truthy(opts_.tools))
            thinking_ = false;
        scan_control_tags(msgs, &thinking_, &effort_);

        std::string reasoning_instructions;
        if (thinking_ && effort_ == "xhigh") {
            reasoning_instructions =
                "Reasoning effort is set to xhigh. Please think carefully through the task, "
                "validate key assumptions, consider plausible alternatives, and prioritize "
                "correctness, consistency, and clarity in the final answer.";
        } else if (thinking_ && effort_ == "low") {
            reasoning_instructions =
                "Reasoning effort is set to low. Keep your thinking brief and focused, "
                "moving directly to the conclusion without unnecessary elaboration.";
        }

        size_t head_count = 0;
        while (head_count < n) {
            const std::string role = get_attr(&msgs[head_count], "role") != nullptr &&
                                             get_attr(&msgs[head_count], "role")->is_string()
                                         ? get_attr(&msgs[head_count], "role")->get<std::string>()
                                         : std::string();
            if (role != "system" && role != "developer") break;
            ++head_count;
        }
        std::string system_content;
        for (size_t i = 0; i < head_count; ++i) {
            std::string part = strip_control_tags(py_trim(render_content(
                get_attr(&msgs[i], "content"), false, true)));
            if (part.empty()) continue;
            if (!system_content.empty()) system_content += "\n\n";
            system_content += part;
        }

        // --- system / tools header ----------------------------------------
        // `tools and tools is iterable and tools is not mapping`
        const json* tools = opts_.tools;
        const bool tools_block =
            tools != nullptr && truthy(tools) && (tools->is_array() || tools->is_string());

        if (tools_block) {
            out += "<|im_start|>system\n";
            if (!reasoning_instructions.empty()) {
                out += reasoning_instructions;
                out += "\n\n";
            }
            out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
            if (tools->is_array()) {
                for (const auto& tool : *tools) {
                    out += "\n";
                    out += json_py::dumps(tool, /*spaced=*/true);
                }
            } else {
                // A string is iterable too: `for tool in tools` yields its
                // characters (code points), each serialized with tojson.
                const std::string& s = tools->get_ref<const std::string&>();
                for (size_t i = 0; i < s.size();) {
                    uint32_t cp;
                    size_t len = utf8_decode(s, i, cp);
                    out += "\n";
                    out += json_py::dumps(json(s.substr(i, len)), /*spaced=*/true);
                    i += len;
                }
            }
            out += "\n</tools>";
            const std::string format = lower_ascii(option_string(opts_.tool_call_format, "xml"));
            if (format == "json") {
                out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n";
                if (thinking_) out += "<think>\nBrief explanation of tool call\n</think>\n";
                out += "<tool_call>\n{\"name\": \"example_function_name\", \"arguments\": {\"example_parameter\": \"value\"}}\n</tool_call>\n\n<IMPORTANT>\nReminder:\n";
                if (thinking_) {
                    out += "- You can use the <think></think> block to plan your next tool call OR to synthesize data and formulate your final response to the user.\n";
                    out += "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n";
                    out += "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after thinking, with NO conversational text before it.\n";
                } else {
                    out += "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with NO conversational text before it.\n";
                }
                out += "- Function calls MUST be exactly one JSON object with \"name\" and \"arguments\" keys inside <tool_call></tool_call>.\n- The <tool_call> tag starts at the beginning of a new line.\n- Output one completely closed block per function; never nest blocks.\n- If no tool is needed, answer directly without a tool call.\n</IMPORTANT>";
            } else {
                out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n";
                if (thinking_) out += "<think>\nBrief explanation of tool call\n</think>\n";
                out += "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n";
                if (thinking_) {
                    out += "- You can use the <think></think> block to plan your next tool call OR to synthesize data and formulate your final response to the user.\n";
                    out += "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n";
                    out += "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after thinking, with NO conversational text before it.\n";
                } else {
                    out += "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with NO conversational text before it.\n";
                }
                out += "- Function calls MUST contain one nested <function=...></function> block inside <tool_call></tool_call>.\n- The <tool_call> and <function> tags start at the beginning of a new line.\n- Output one completely closed block per function; never nest blocks.\n- If no tool is needed, answer directly without a tool call.\n</IMPORTANT>";
            }
            if (!system_content.empty()) {
                out += "\n\n";
                out += system_content;
            }
            out += "<|im_end|>\n";
        } else if (!system_content.empty() || !reasoning_instructions.empty()) {
            out += "<|im_start|>system\n";
            if (!reasoning_instructions.empty()) {
                out += reasoning_instructions;
                if (!system_content.empty()) out += "\n\n";
            }
            out += system_content;
            out += "<|im_end|>\n";
        }

        // --- multi_step_tool / last_query_index backward scan -------------
        bool multi_step_tool = true;
        size_t last_query_index = n - 1;
        for (size_t rev = 0; rev < n && multi_step_tool; ++rev) {
            const size_t index = (n - 1) - rev;
            const json& message = msgs[index];
            if (!str_eq(get_attr(&message, "role"), "user")) continue;
            std::string content =
                py_trim(render_content(get_attr(&message, "content"),
                                       /*do_vision_count=*/false,
                                       /*is_system_content=*/false));
            static const std::string kOpen = "<tool_response>";
            static const std::string kClose = "</tool_response>";
            const bool is_tool_response =
                content.compare(0, kOpen.size(), kOpen) == 0 &&
                content.size() >= kClose.size() &&
                content.compare(content.size() - kClose.size(), kClose.size(), kClose) == 0;
            if (!is_tool_response) {
                multi_step_tool = false;
                last_query_index = index;
            }
        }
        if (multi_step_tool) {
            // Tool-only replay histories are valid continuation context. Match
            // the fixed template's deterministic preservation boundary instead
            // of rejecting the prompt before it can reach the model.
            last_query_index = n > 51 ? n - 1 : 0;
        }

        // --- main loop ------------------------------------------------------
        size_t consecutive_tool_failures = 0;
        for (size_t index0 = 0; index0 < n; ++index0) {
            const json& message = msgs[index0];
            std::string content =
                py_trim(render_content(get_attr(&message, "content"),
                                       /*do_vision_count=*/true,
                                       /*is_system_content=*/false));
            content = strip_control_tags(content);
            const json* role = get_attr(&message, "role");

            if (str_eq(role, "system") || str_eq(role, "developer")) {
                if (index0 < head_count) continue;
                out += "<|im_start|>system\n";
                out += content;
                out += "<|im_end|>\n";
            } else if (str_eq(role, "user")) {
                consecutive_tool_failures = 0;
                out += "<|im_start|>user\n";
                out += content;
                out += "<|im_end|>\n";
            } else if (str_eq(role, "assistant")) {
                const json* rc = get_attr(&message, "reasoning_content");
                if (rc == nullptr) rc = get_attr(&message, "thinking");
                if (rc == nullptr) rc = get_attr(&message, "reasoning");
                ExtractedReasoning extracted = extract_reasoning(content, rc);
                content = extracted.content;
                const std::string reasoning_content = extracted.reasoning;
                const json* preserve_opt = opts_.preserve_reasoning != nullptr
                                               ? opts_.preserve_reasoning
                                               : opts_.preserve_thinking;
                const bool preserve = preserve_opt == nullptr || is_json_true(preserve_opt) ||
                                      index0 > last_query_index;
                const json* tool_calls = get_attr(&message, "tool_calls");
                const bool has_tool_calls =
                    tool_calls != nullptr && truthy(tool_calls) &&
                    (tool_calls->is_array() || tool_calls->is_string());
                // Skip turns that would render as an empty exemplar
                // (<|im_start|>assistant\n<|im_end|>). They arise when an
                // upstream strips reasoning_content from a thinking-only turn
                // (e.g. a response truncated to reasoning by a missing
                // </think>), and few-shot the model into ending its own turn
                // with no output.
                if (content.empty() && !has_tool_calls &&
                    (reasoning_content.empty() || !preserve)) {
                    continue;
                }
                if (preserve && !reasoning_content.empty()) {
                    out += "<|im_start|>assistant\n<think>\n";
                    out += reasoning_content;
                    out += "\n</think>\n\n";
                    out += content;
                } else {
                    out += "<|im_start|>assistant\n";
                    out += content;
                }

                if (has_tool_calls) {
                    if (tool_calls->is_string()) {
                        // Iterating chars: the first char's `.name` lookup fails.
                        raise("'str object' has no attribute 'name'");
                    }
                    size_t tc_index = 0;
                    for (const auto& tool_call_elem : *tool_calls) {
                        const json* tc = &tool_call_elem;
                        if (tc->is_object() && tc->contains("function")) {
                            tc = &(*tc)["function"];  // null counts as defined
                        }
                        const json* name = get_attr(tc, "name");
                        static const json kEmptyName = "";
                        if (name == nullptr || name->is_null()) name = &kEmptyName;
                        if (!name->is_string()) {
                            raise(std::string("can only concatenate str (not \"") +
                                  py_type_name(*name) + "\") to str");
                        }
                        const std::string format = lower_ascii(option_string(opts_.tool_call_format, "xml"));
                        if (format == "json") {
                            if (tc_index == 0 && !content.empty()) out += "\n\n";
                            else if (tc_index > 0) out += "\n";
                            std::string args = "{}";
                            if (const json* arg = get_attr(tc, "arguments")) {
                                if (arg->is_string()) {
                                    if (!arg->get_ref<const std::string&>().empty()) args = arg->get_ref<const std::string&>();
                                } else if (!arg->is_null()) {
                                    args = json_py::dumps(*arg, /*spaced=*/true);
                                }
                            }
                            // JSON tool calls must remain valid JSON; unlike XML
                            // text parameters, never slice the serialized object.
                            out += "<tool_call>\n{\"name\": ";
                            out += json_py::dumps(*name, /*spaced=*/true);
                            out += ", \"arguments\": " + args + "}\n</tool_call>";
                            ++tc_index;
                            continue;
                        }
                        if (tc_index == 0) {
                            if (!content.empty()) {
                                out += "\n\n<tool_call>\n<function=";
                            } else {
                                out += "<tool_call>\n<function=";
                            }
                        } else {
                            out += "\n<tool_call>\n<function=";
                        }
                        out += name->get_ref<const std::string&>();
                        out += ">\n";

                        const json* args = get_attr(tc, "arguments");
                        if (args != nullptr) {
                            // Python `!= ''`: everything but the exact string "".
                            const bool neq_empty =
                                !(args->is_string() &&
                                  args->get_ref<const std::string&>().empty());
                            if (neq_empty) {
                                if (args->is_string()) {
                                    out += truncate_payload(
                                        args->get_ref<const std::string&>(),
                                        option_limit(opts_.max_tool_arg_chars));
                                } else if (!args->is_object()) {
                                    out += json_py::dumps(*args, /*spaced=*/true);
                                } else {
                                    for (auto it = args->begin(); it != args->end(); ++it) {
                                        out += "<parameter=";
                                        out += it.key();
                                        out += ">\n";
                                            std::string value = it.value().is_string()
                                                                ? it.value().get_ref<const std::string&>()
                                                                : json_py::dumps(it.value(), /*spaced=*/true);
                                        out += truncate_payload(value, option_limit(opts_.max_tool_arg_chars));
                                        out += "\n</parameter>\n";
                                    }
                                }
                            }
                        }
                        out += "</function>\n</tool_call>";
                        ++tc_index;
                    }
                }
                out += "<|im_end|>\n";
            } else if (str_eq(role, "tool")) {
                if (is_tool_error(content)) {
                    ++consecutive_tool_failures;
                } else {
                    consecutive_tool_failures = 0;
                }
                const json* prev = index0 > 0 ? &msgs[index0 - 1] : nullptr;
                if (prev == nullptr || !str_eq(get_attr(prev, "role"), "tool")) {
                    out += "<|im_start|>user";
                }
                out += "\n<tool_response>\n";
                const size_t response_limit = option_limit(opts_.max_tool_response_chars);
                if (lower_ascii(option_string(opts_.tool_call_format, "xml")) == "json" &&
                    response_limit && !structured_json_response(content))
                    content = truncate_payload(content, response_limit);
                else if (response_limit && lower_ascii(option_string(opts_.tool_call_format, "xml")) != "json")
                    content = truncate_payload(content, response_limit);
                out += content;
                if (consecutive_tool_failures >= 2) {
                    out += "\n\nSYSTEM WARNING: ";
                    out += std::to_string(consecutive_tool_failures);
                    out += " consecutive tool errors detected. The previous approach is incorrect; use a fundamentally different approach or corrected arguments.";
                } else if (consecutive_tool_failures == 1) {
                    out += "\n\nSYSTEM WARNING: The previous tool call returned an error. Diagnose the failure and retry with completely corrected arguments.";
                }
                out += "\n</tool_response>";
                if (index0 + 1 < n) {
                    if (!str_eq(get_attr(&msgs[index0 + 1], "role"), "tool")) {
                        out += "<|im_end|>\n";
                    }
                } else {
                    out += "<|im_end|>\n";
                }
            } else {
                out += "<|im_start|>user\n[";
                out += role != nullptr && role->is_string() ? role->get_ref<const std::string&>() : "unknown";
                out += "]: " + content + "<|im_end|>\n";
            }
        }

        // --- generation prompt tail ----------------------------------------
        if (truthy(opts_.add_generation_prompt)) {
            out += "<|im_start|>assistant\n";
            if (!thinking_) {
                out += "<think>\n\n</think>\n\n";
            } else {
                out += "<think>\n";
            }
        }
        return out;
    }

  private:
    // The render_content macro.
    std::string render_content(const json* content, bool do_vision_count,
                               bool is_system_content) {
        if (content == nullptr || content->is_null()) return "";
        if (content->is_string()) return content->get_ref<const std::string&>();
            if (content->is_array()) {
            std::string out;
            for (const auto& item : *content) {
                if (is_image_item(item)) {
                    if (is_system_content) raise("System message cannot contain images.");
                    if (do_vision_count) ++image_count_;
                    if (add_vision_id_) {
                        out += "Picture " + std::to_string(image_count_) + ": ";
                    }
                    out += "<|vision_start|><|image_pad|><|vision_end|>";
                } else if (is_video_item(item)) {
                    if (is_system_content) raise("System message cannot contain videos.");
                    if (do_vision_count) ++video_count_;
                    if (add_vision_id_) {
                        out += "Video " + std::to_string(video_count_) + ": ";
                    }
                    out += "<|vision_start|><|video_pad|><|vision_end|>";
                } else if (item.is_object() && py_in("text", item)) {
                    if (const json* text = get_attr(&item, "text")) {
                        out += json_py::py_str(*text);
                    }
                } else if (item.is_string()) {
                    out += item.get_ref<const std::string&>();
                } else {
                    raise("Unexpected item type in content.");
                }
            }
            return out;
        }
        raise("Unexpected content type.");
    }

    bool is_image_item(const json& item) {
        return item.is_object() &&
               (item.contains("image") || item.contains("image_url") ||
                str_eq(get_attr(&item, "type"), "image"));
    }

    bool is_video_item(const json& item) {
        return item.is_object() &&
               (item.contains("video") || item.contains("video_url") ||
                str_eq(get_attr(&item, "type"), "video") ||
                str_eq(get_attr(&item, "type"), "video_url"));
    }

    const Options& opts_;
    bool add_vision_id_ = false;
    bool thinking_ = true;
    std::string effort_ = "medium";
    long image_count_ = 0;
    long video_count_ = 0;
};

}  // namespace

RenderResult render_chat_template(const json* messages, const Options& opts) {
    RenderResult result;
    try {
        result.text = Renderer(opts).run(messages);
        result.ok = true;
    } catch (const TemplateError& e) {
        result.error = e.message;
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

}  // namespace chat_template
