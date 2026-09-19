// chat_template.cpp — see chat_template.h for the semantics notes.
#include "chat_template.h"

#include <algorithm>
#include <cstdint>
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

        // --- reasoning instructions -------------------------------------
        std::string reasoning_instructions;
        // `enable_thinking is undefined or enable_thinking is true`
        if (opts_.enable_thinking == nullptr || is_json_true(opts_.enable_thinking)) {
            // `reasoning_effort|default('xhigh')`: default replaces undefined only.
            const json* resolved = opts_.reasoning_effort;
            const bool valid = resolved == nullptr || str_eq(resolved, "xhigh") ||
                               str_eq(resolved, "medium") || str_eq(resolved, "low");
            if (!valid) {
                raise("Unexpected reasoning effort " + json_py::py_str(*resolved) +
                      ". Supported types are xhigh (default), medium, and low.");
            }
            if (resolved == nullptr || str_eq(resolved, "xhigh")) {
                reasoning_instructions =
                    "Reasoning effort is set to xhigh. Please think carefully through the "
                    "task, validate key assumptions, consider plausible alternatives, and "
                    "prioritize correctness, consistency, and clarity in the final answer.";
            } else if (str_eq(resolved, "low")) {
                reasoning_instructions =
                    "Reasoning effort is set to low. Keep your thinking brief and focused, "
                    "moving directly to the conclusion without unnecessary elaboration.";
            }
            // medium: intentionally no instructions string.
        }

        const bool first_is_system = str_eq(get_attr(&msgs[0], "role"), "system");

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
            out +=
                "\n\nIf you choose to call a function ONLY reply in the following format "
                "with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
                "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                "<parameter=example_parameter_2>\nThis is the value for the second "
                "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
                "</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
                "specified format: an inner <function=...></function> block must be nested "
                "within <tool_call></tool_call> XML tags\n- Required parameters MUST be "
                "specified\n- You may provide optional reasoning for your function call in "
                "natural language BEFORE the function call, but NOT after\n- If there is "
                "no function call available, answer the question like normal with your "
                "current knowledge and do not tell the user about function calls\n"
                "</IMPORTANT>";
            if (first_is_system) {
                std::string content =
                    py_trim(render_content(get_attr(&msgs[0], "content"),
                                           /*do_vision_count=*/false,
                                           /*is_system_content=*/true));
                if (!content.empty()) {
                    out += "\n\n";
                    out += content;
                }
            }
            out += "<|im_end|>\n";
        } else if (first_is_system) {
            std::string content =
                py_trim(render_content(get_attr(&msgs[0], "content"),
                                       /*do_vision_count=*/false,
                                       /*is_system_content=*/true));
            if (!content.empty()) {
                out += "<|im_start|>system\n";
                if (!reasoning_instructions.empty()) {
                    out += reasoning_instructions;
                    out += "\n\n";
                }
                out += content;
                out += "<|im_end|>\n";
            } else if (!reasoning_instructions.empty()) {
                out += "<|im_start|>system\n";
                out += reasoning_instructions;
                out += "<|im_end|>\n";
            }
        } else if (!reasoning_instructions.empty()) {
            out += "<|im_start|>system\n";
            out += reasoning_instructions;
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
        if (multi_step_tool) raise("No user query found in messages.");

        // --- main loop ------------------------------------------------------
        for (size_t index0 = 0; index0 < n; ++index0) {
            const json& message = msgs[index0];
            std::string content =
                py_trim(render_content(get_attr(&message, "content"),
                                       /*do_vision_count=*/true,
                                       /*is_system_content=*/false));
            const json* role = get_attr(&message, "role");

            if (str_eq(role, "system")) {
                if (index0 != 0) raise("System message must be at the beginning.");
            } else if (str_eq(role, "user")) {
                out += "<|im_start|>user\n";
                out += content;
                out += "<|im_end|>\n";
            } else if (str_eq(role, "assistant")) {
                std::string reasoning_content;
                const json* rc = get_attr(&message, "reasoning_content");
                if (rc != nullptr && rc->is_string()) {
                    reasoning_content = rc->get_ref<const std::string&>();
                }
                reasoning_content = py_trim(reasoning_content);

                const bool preserve = opts_.preserve_thinking == nullptr ||
                                      is_json_true(opts_.preserve_thinking) ||
                                      index0 > last_query_index;
                if (preserve) {
                    out += "<|im_start|>assistant\n<think>\n";
                    out += reasoning_content;
                    out += "\n</think>\n\n";
                    out += content;
                } else {
                    out += "<|im_start|>assistant\n";
                    out += content;
                }

                const json* tool_calls = get_attr(&message, "tool_calls");
                if (tool_calls != nullptr && truthy(tool_calls) &&
                    (tool_calls->is_array() || tool_calls->is_string())) {
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
                        if (name == nullptr) {
                            raise(std::string("'") + py_type_name(*tc) +
                                  " object' has no attribute 'name'");
                        }
                        if (!name->is_string()) {
                            raise(std::string("can only concatenate str (not \"") +
                                  py_type_name(*name) + "\") to str");
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
                                if (!args->is_object()) {
                                    raise("Can only get item pairs from a mapping.");
                                }
                                for (auto it = args->begin(); it != args->end(); ++it) {
                                    out += "<parameter=";
                                    out += it.key();
                                    out += ">\n";
                                    if (it.value().is_string()) {
                                        out += it.value().get_ref<const std::string&>();
                                    } else {
                                        out += json_py::dumps(it.value(), /*spaced=*/true);
                                    }
                                    out += "\n</parameter>\n";
                                }
                            }
                        }
                        out += "</function>\n</tool_call>";
                        ++tc_index;
                    }
                }
                out += "<|im_end|>\n";
            } else if (str_eq(role, "tool")) {
                const json* prev = index0 > 0 ? &msgs[index0 - 1] : nullptr;
                if (truthy(prev) && !str_eq(get_attr(prev, "role"), "tool")) {
                    out += "<|im_start|>user";
                }
                out += "\n<tool_response>\n";
                out += content;
                out += "\n</tool_response>";
                if (index0 + 1 < n) {
                    if (!str_eq(get_attr(&msgs[index0 + 1], "role"), "tool")) {
                        out += "<|im_end|>\n";
                    }
                } else {
                    out += "<|im_end|>\n";
                }
            } else {
                raise("Unexpected message role.");
            }
        }

        // --- generation prompt tail ----------------------------------------
        if (truthy(opts_.add_generation_prompt)) {
            out += "<|im_start|>assistant\n";
            if (opts_.enable_thinking != nullptr &&
                opts_.enable_thinking->is_boolean() &&
                !opts_.enable_thinking->get<bool>()) {
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
                } else if (py_in("text", item)) {
                    // item.text: dict key value, or Undefined (prints as '')
                    // for string/array items (header note 5).
                    if (const json* text = get_attr(&item, "text")) {
                        out += json_py::py_str(*text);
                    }
                } else {
                    raise("Unexpected item type in content.");
                }
            }
            return out;
        }
        raise("Unexpected content type.");
    }

    bool is_image_item(const json& item) {
        return py_in("image", item) || py_in("image_url", item) ||
               str_eq(get_attr(&item, "type"), "image");
    }

    bool is_video_item(const json& item) {
        return py_in("video", item) || str_eq(get_attr(&item, "type"), "video");
    }

    const Options& opts_;
    bool add_vision_id_ = false;
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
