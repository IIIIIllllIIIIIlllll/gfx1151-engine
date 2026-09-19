// Clean-room Qwen XML tool-call adapter. The wire grammar and schema coercion
// semantics follow the public Apache-2.0 Transformers response parser.
#include "toolparse.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "json_py.h"

namespace toolparse {
namespace {

constexpr const char* kToolOpen = "<tool_call>";
constexpr const char* kFunctionOpen = "<function=";
constexpr const char* kParameterOpen = "<parameter=";
constexpr const char* kParameterClose = "</parameter>";
constexpr const char* kFunctionClose = "</function>";
constexpr const char* kToolClose = "</tool_call>";

bool ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

void trim_ascii(std::string* value) {
    size_t begin = 0;
    while (begin < value->size() && ascii_space((*value)[begin])) ++begin;
    size_t end = value->size();
    while (end > begin && ascii_space((*value)[end - 1])) --end;
    *value = value->substr(begin, end - begin);
}

bool is_prefix_of(std::string_view value, std::string_view token) {
    return value.size() <= token.size() && token.substr(0, value.size()) == value;
}

size_t suffix_prefix_len(std::string_view value, std::string_view token) {
    const size_t top = std::min(value.size(), token.size() - 1);
    for (size_t n = top; n > 0; --n) {
        if (value.substr(value.size() - n) == token.substr(0, n)) return n;
    }
    return 0;
}

std::string quote_json(const std::string& value) {
    return json_py::dumps(json(value), /*spaced=*/false);
}

std::string escape_json_fragment(const std::string& value) {
    std::string quoted = quote_json(value);
    if (quoted.size() < 2) return {};
    return quoted.substr(1, quoted.size() - 2);
}

void collect_schema_types(const json& schema, std::vector<std::string>* result) {
    if (!schema.is_object()) return;
    auto type = schema.find("type");
    if (type != schema.end()) {
        if (type->is_string()) result->push_back(type->get<std::string>());
        if (type->is_array()) {
            for (const auto& item : *type)
                if (item.is_string()) result->push_back(item.get<std::string>());
        }
    }
    auto any_of = schema.find("anyOf");
    if (any_of != schema.end() && any_of->is_array()) {
        for (const auto& item : *any_of) collect_schema_types(item, result);
    }
    auto nullable = schema.find("nullable");
    if (nullable != schema.end() && nullable->is_boolean() && nullable->get<bool>() &&
        std::find(result->begin(), result->end(), "null") == result->end())
        result->push_back("null");
}

json coerce_value(const std::string& raw, const std::vector<std::string>& types) {
    for (const std::string& type : types) {
        if (type == "integer") {
            errno = 0;
            char* end = nullptr;
            const long long value = std::strtoll(raw.c_str(), &end, 10);
            if (errno == 0 && end != raw.c_str() && *end == '\0') return value;
        } else if (type == "number") {
            errno = 0;
            char* end = nullptr;
            const double value = std::strtod(raw.c_str(), &end);
            if (errno == 0 && end != raw.c_str() && *end == '\0' && std::isfinite(value)) {
                if (std::floor(value) == value && raw.find('.') == std::string::npos &&
                    value >= static_cast<double>(std::numeric_limits<long long>::min()) &&
                    value <= static_cast<double>(std::numeric_limits<long long>::max()))
                    return static_cast<long long>(value);
                return value;
            }
        } else if (type == "boolean") {
            std::string value = raw;
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (value == "true" || value == "1") return true;
            if (value == "false" || value == "0") return false;
        } else if (type == "null") {
            if (raw == "null" || raw == "None") return nullptr;
        } else if (type == "object" || type == "array") {
            try {
                json value = json::parse(raw);
                if ((type == "object" && value.is_object()) ||
                    (type == "array" && value.is_array()))
                    return value;
            } catch (...) {
            }
        }
    }
    return raw;
}

bool function_exists(const json& tools, const std::string& name) {
    for (const auto& tool : tools) {
        if (tool.is_object() && tool.contains("function") && tool["function"].is_object() &&
            tool["function"].contains("name") && tool["function"]["name"].is_string() &&
            tool["function"]["name"].get_ref<const std::string&>() == name)
            return true;
    }
    return false;
}

bool append_function(const json& source, json* output, std::string* error) {
    json function;
    if (source.contains("function")) {
        if (!source["function"].is_object()) {
            *error = "function tools require a function object";
            return false;
        }
        function = source["function"];
    } else {
        for (const char* key : {"name", "description", "parameters", "strict"})
            if (source.contains(key)) function[key] = source[key];
    }
    if (!function.contains("name") || !function["name"].is_string() ||
        function["name"].get_ref<const std::string&>().empty()) {
        *error = "function tools require a non-empty string name";
        return false;
    }
    if (function.contains("description") && !function["description"].is_string()) {
        *error = "function tool description must be a string";
        return false;
    }
    if (function.contains("parameters") && !function["parameters"].is_object()) {
        *error = "function tool parameters must be an object";
        return false;
    }
    if (function.contains("strict") && !function["strict"].is_boolean()) {
        *error = "function tool strict must be a boolean";
        return false;
    }
    output->push_back(json{{"type", "function"}, {"function", std::move(function)}});
    return true;
}

bool append_tool(const json& tool, json* output, std::string* error) {
    if (!tool.is_object()) {
        *error = "tool entries must be objects";
        return false;
    }
    if (!tool.contains("type") || !tool["type"].is_string()) {
        *error = "tool entries require a string type";
        return false;
    }
    const std::string& type = tool["type"].get_ref<const std::string&>();
    if (type == "function") return append_function(tool, output, error);
    if (type != "namespace") return true;  // unsupported built-ins are ignored

    const json* nested = nullptr;
    if (tool.contains("tools")) nested = &tool["tools"];
    else if (tool.contains("functions")) nested = &tool["functions"];
    if (nested == nullptr) return true;
    if (!nested->is_array()) {
        *error = "namespace tools must be an array";
        return false;
    }
    for (const auto& child : *nested) {
        if (child.is_object() && !child.contains("type") && child.contains("name")) {
            if (!append_function(child, output, error)) return false;
        } else if (!append_tool(child, output, error)) {
            return false;
        }
    }
    return true;
}

std::string message_role(const json& message) {
    if (!message.is_object() || !message.contains("role") || !message["role"].is_string())
        return {};
    return message["role"].get<std::string>();
}

}  // namespace

std::string ToolChoice::parser_prefix() const {
    if (!forced()) return {};
    std::string result = "<tool_call>\n<function=";
    if (!name.empty()) result += name + ">\n";
    return result;
}

std::string ToolChoice::prompt_suffix(bool thinking_enabled) const {
    if (!forced()) return {};
    std::string result;
    if (thinking_enabled) result = "</think>\n\n";
    result += parser_prefix();
    return result;
}

bool normalize_tools(const json* input, json* output, std::string* error) {
    *output = json::array();
    error->clear();
    if (input == nullptr || input->is_null()) return true;
    if (!input->is_array()) {
        *error = "tools must be an array";
        return false;
    }
    for (const auto& tool : *input)
        if (!append_tool(tool, output, error)) return false;
    return true;
}

bool parse_tool_choice(const json* input, const json& tools, ToolChoice* output,
                       std::string* error) {
    *output = ToolChoice{};
    error->clear();
    if (input == nullptr || input->is_null()) return true;
    if (input->is_string()) {
        const std::string& value = input->get_ref<const std::string&>();
        if (value == "auto") return true;
        if (value == "none") {
            output->mode = ToolChoice::Mode::None;
            return true;
        }
        if (value == "required") {
            if (tools.empty()) {
                *error = "tool_choice='required' requires at least one function tool";
                return false;
            }
            output->mode = ToolChoice::Mode::Required;
            if (tools.size() == 1)
                output->name = tools[0]["function"]["name"].get<std::string>();
            return true;
        }
        *error = "tool_choice must be 'auto', 'none', 'required', or a named function";
        return false;
    }
    if (!input->is_object()) {
        *error = "tool_choice must be a string or object";
        return false;
    }
    const json* function = input;
    if (input->contains("function")) function = &(*input)["function"];
    if (!function->is_object() || !function->contains("name") ||
        !(*function)["name"].is_string() ||
        (*function)["name"].get_ref<const std::string&>().empty()) {
        *error = "named tool_choice requires a non-empty function name";
        return false;
    }
    output->mode = ToolChoice::Mode::Named;
    output->name = (*function)["name"].get<std::string>();
    if (!function_exists(tools, output->name)) {
        *error = "tool_choice names an unknown function: " + output->name;
        return false;
    }
    return true;
}

bool normalize_messages(json* messages, std::string* error) {
    error->clear();
    if (!messages->is_array()) {
        *error = "messages must be an array";
        return false;
    }

    for (auto& message : *messages) {
        if (!message.is_object()) continue;
        if (message_role(message) == "tool" && message.contains("tool_call_id") &&
            !message["tool_call_id"].is_string()) {
            *error = "tool_call_id must be a string";
            return false;
        }
        if (message_role(message) != "assistant" || !message.contains("tool_calls") ||
            message["tool_calls"].is_null())
            continue;
        if (!message["tool_calls"].is_array()) {
            *error = "assistant tool_calls must be an array";
            return false;
        }
        for (auto& call : message["tool_calls"]) {
            if (!call.is_object()) {
                *error = "assistant tool_calls entries must be objects";
                return false;
            }
            json* function = &call;
            if (call.contains("function")) {
                if (!call["function"].is_object()) {
                    *error = "assistant tool call function must be an object";
                    return false;
                }
                function = &call["function"];
            }
            if (!function->contains("name") || !(*function)["name"].is_string()) {
                *error = "assistant tool calls require a string function name";
                return false;
            }
            if (!function->contains("arguments") || (*function)["arguments"].is_null()) {
                (*function)["arguments"] = json::object();
            } else if ((*function)["arguments"].is_string()) {
                try {
                    json parsed = json::parse((*function)["arguments"].get<std::string>());
                    if (!parsed.is_object()) throw std::runtime_error("not object");
                    (*function)["arguments"] = std::move(parsed);
                } catch (...) {
                    *error = "assistant tool call arguments must be a JSON object string";
                    return false;
                }
            } else if (!(*function)["arguments"].is_object()) {
                *error = "assistant tool call arguments must be an object or JSON object string";
                return false;
            }
        }
    }

    json ordered = json::array();
    for (size_t i = 0; i < messages->size();) {
        json message = std::move((*messages)[i]);
        ordered.push_back(std::move(message));
        if (message_role(ordered.back()) != "assistant" ||
            !ordered.back().contains("tool_calls") ||
            !ordered.back()["tool_calls"].is_array()) {
            ++i;
            continue;
        }

        size_t end = i + 1;
        while (end < messages->size() && message_role((*messages)[end]) == "tool") ++end;
        std::vector<bool> used(end - (i + 1), false);
        for (const auto& call : ordered.back()["tool_calls"]) {
            if (!call.is_object() || !call.contains("id") || !call["id"].is_string()) continue;
            const std::string& id = call["id"].get_ref<const std::string&>();
            for (size_t j = i + 1; j < end; ++j) {
                const json& result = (*messages)[j];
                if (!used[j - (i + 1)] && result.contains("tool_call_id") &&
                    result["tool_call_id"].is_string() &&
                    result["tool_call_id"].get_ref<const std::string&>() == id) {
                    ordered.push_back(std::move((*messages)[j]));
                    used[j - (i + 1)] = true;
                    break;
                }
            }
        }
        for (size_t j = i + 1; j < end; ++j)
            if (!used[j - (i + 1)]) ordered.push_back(std::move((*messages)[j]));
        i = end;
    }
    *messages = std::move(ordered);
    return true;
}

StreamParser::StreamParser(json tools, IdFactory make_id, bool enabled)
    : tools_(std::move(tools)), make_id_(std::move(make_id)), enabled_(enabled) {}

std::vector<Event> StreamParser::feed(std::string_view piece) {
    buffer_.append(piece.data(), piece.size());
    return pump(false);
}

std::vector<Event> StreamParser::finish() { return pump(true); }

void StreamParser::emit_content(std::string text, std::vector<Event>* events) {
    if (text.empty()) return;
    content_ += text;
    events->push_back(Event{EventType::Content, 0, {}, {}, std::move(text)});
}

void StreamParser::emit_arguments(std::string text, std::vector<Event>* events) {
    if (text.empty()) return;
    current_.arguments += text;
    events->push_back(
        Event{EventType::ArgumentsDelta, current_.index, current_.id, current_.name,
              std::move(text)});
}

void StreamParser::start_call(std::string name, std::vector<Event>* events) {
    current_ = ToolCall{};
    current_.index = next_index_++;
    current_.id = make_id_ ? make_id_() : std::string();
    current_.name = std::move(name);
    argument_count_ = 0;
    events->push_back(Event{EventType::CallStart, current_.index, current_.id,
                            current_.name, {}});
}

void StreamParser::start_parameter(std::string key, std::vector<Event>* events) {
    parameter_key_ = std::move(key);
    parameter_types_ = schema_types(current_.name, parameter_key_);
    stream_parameter_ = !parameter_types_.empty() &&
                        std::all_of(parameter_types_.begin(), parameter_types_.end(),
                                    [](const std::string& type) {
                                        return type == "string" || type == "null";
                                    }) &&
                        std::find(parameter_types_.begin(), parameter_types_.end(), "string") !=
                            parameter_types_.end();
    value_started_ = false;
    if (stream_parameter_) {
        std::string prefix = argument_count_++ == 0 ? "{" : ",";
        prefix += quote_json(parameter_key_) + ":\"";
        emit_arguments(std::move(prefix), events);
    }
}

void StreamParser::finish_parameter(std::string raw, std::vector<Event>* events) {
    trim_ascii(&raw);
    if (stream_parameter_) {
        emit_arguments(escape_json_fragment(raw), events);
        emit_arguments("\"", events);
    } else {
        std::string encoded = argument_count_++ == 0 ? "{" : ",";
        encoded += quote_json(parameter_key_) + ":";
        encoded += json_py::dumps(coerce_value(raw, parameter_types_), /*spaced=*/false);
        emit_arguments(std::move(encoded), events);
    }
    parameter_key_.clear();
    parameter_types_.clear();
    stream_parameter_ = false;
    value_started_ = false;
}

void StreamParser::finish_call(std::vector<Event>* events) {
    emit_arguments(argument_count_ == 0 ? "{}" : "}", events);
    calls_.push_back(current_);
    events->push_back(Event{EventType::CallEnd, current_.index, current_.id,
                            current_.name, current_.arguments});
    current_ = ToolCall{};
    argument_count_ = 0;
}

std::vector<std::string> StreamParser::schema_types(
    const std::string& function, const std::string& parameter) const {
    for (const auto& tool : tools_) {
        if (!tool.is_object() || !tool.contains("function") ||
            !tool["function"].is_object())
            continue;
        const json& fn = tool["function"];
        if (!fn.contains("name") || !fn["name"].is_string() ||
            fn["name"].get_ref<const std::string&>() != function ||
            !fn.contains("parameters") || !fn["parameters"].is_object() ||
            !fn["parameters"].contains("properties") ||
            !fn["parameters"]["properties"].is_object() ||
            !fn["parameters"]["properties"].contains(parameter))
            continue;
        std::vector<std::string> result;
        collect_schema_types(fn["parameters"]["properties"][parameter], &result);
        return result;
    }
    return {};
}

std::vector<Event> StreamParser::pump(bool final) {
    std::vector<Event> events;
    if (!enabled_) {
        emit_content(std::move(buffer_), &events);
        buffer_.clear();
        return events;
    }

    while (true) {
        if (state_ == State::Broken) {
            partial_ = true;
            if (final) buffer_.clear();
            break;
        }

        if (state_ == State::Text) {
            const size_t marker = buffer_.find(kToolOpen);
            if (marker == std::string::npos) {
                if (final) {
                    emit_content(std::move(buffer_), &events);
                    buffer_.clear();
                } else {
                    const size_t partial = suffix_prefix_len(buffer_, kToolOpen);
                    size_t keep_at = buffer_.size() - partial;
                    while (keep_at > 0 && ascii_space(buffer_[keep_at - 1])) --keep_at;
                    emit_content(buffer_.substr(0, keep_at), &events);
                    buffer_.erase(0, keep_at);
                }
                break;
            }

            size_t content_end = marker;
            while (content_end > 0 && ascii_space(buffer_[content_end - 1])) --content_end;
            emit_content(buffer_.substr(0, content_end), &events);
            buffer_.erase(0, content_end);
            const size_t open_at = marker - content_end;
            size_t at = open_at + std::char_traits<char>::length(kToolOpen);
            while (at < buffer_.size() && ascii_space(buffer_[at])) ++at;
            const std::string_view rest(buffer_.data() + at, buffer_.size() - at);
            if (rest.size() < std::char_traits<char>::length(kFunctionOpen)) {
                if (!final && is_prefix_of(rest, kFunctionOpen)) break;
                emit_content(buffer_.substr(0, open_at + 1), &events);
                buffer_.erase(0, open_at + 1);
                continue;
            }
            if (rest.substr(0, std::char_traits<char>::length(kFunctionOpen)) != kFunctionOpen) {
                emit_content(buffer_.substr(0, open_at + 1), &events);
                buffer_.erase(0, open_at + 1);
                continue;
            }
            at += std::char_traits<char>::length(kFunctionOpen);
            const size_t close = buffer_.find('>', at);
            const size_t newline = buffer_.find_first_of("\r\n", at);
            if (close == std::string::npos) {
                if (!final && newline == std::string::npos) break;
                emit_content(buffer_.substr(0, open_at + 1), &events);
                buffer_.erase(0, open_at + 1);
                continue;
            }
            if (close == at || (newline != std::string::npos && newline < close)) {
                emit_content(buffer_.substr(0, open_at + 1), &events);
                buffer_.erase(0, open_at + 1);
                continue;
            }
            start_call(buffer_.substr(at, close - at), &events);
            buffer_.erase(0, close + 1);
            state_ = State::Between;
            continue;
        }

        if (state_ == State::Between) {
            size_t ws = 0;
            while (ws < buffer_.size() && ascii_space(buffer_[ws])) ++ws;
            buffer_.erase(0, ws);
            if (buffer_.empty()) {
                if (final) state_ = State::Broken;
                break;
            }
            if (buffer_.rfind(kFunctionClose, 0) == 0) {
                buffer_.erase(0, std::char_traits<char>::length(kFunctionClose));
                state_ = State::ToolTail;
                continue;
            }
            if (buffer_.rfind(kParameterOpen, 0) == 0) {
                const size_t begin = std::char_traits<char>::length(kParameterOpen);
                const size_t close = buffer_.find('>', begin);
                const size_t newline = buffer_.find_first_of("\r\n", begin);
                if (close == std::string::npos) {
                    if (!final && newline == std::string::npos) break;
                    state_ = State::Broken;
                    continue;
                }
                if (close == begin || (newline != std::string::npos && newline < close)) {
                    state_ = State::Broken;
                    continue;
                }
                std::string key = buffer_.substr(begin, close - begin);
                buffer_.erase(0, close + 1);
                start_parameter(std::move(key), &events);
                state_ = State::Parameter;
                continue;
            }
            if (!final && (is_prefix_of(buffer_, kFunctionClose) ||
                           is_prefix_of(buffer_, kParameterOpen)))
                break;
            state_ = State::Broken;
            continue;
        }

        if (state_ == State::Parameter) {
            const size_t close = buffer_.find(kParameterClose);
            if (close != std::string::npos) {
                std::string raw = buffer_.substr(0, close);
                buffer_.erase(0, close + std::char_traits<char>::length(kParameterClose));
                finish_parameter(std::move(raw), &events);
                state_ = State::Between;
                continue;
            }
            if (!stream_parameter_) {
                if (final) state_ = State::Broken;
                break;
            }

            if (!value_started_) {
                size_t begin = 0;
                while (begin < buffer_.size() && ascii_space(buffer_[begin])) ++begin;
                if (begin == buffer_.size()) break;
                buffer_.erase(0, begin);
                value_started_ = true;
            }
            if (final) {
                emit_arguments(escape_json_fragment(buffer_), &events);
                buffer_.clear();
                partial_ = true;
                break;
            }
            const size_t partial = suffix_prefix_len(buffer_, kParameterClose);
            size_t safe = buffer_.size() - partial;
            while (safe > 0 && ascii_space(buffer_[safe - 1])) --safe;
            if (safe == 0) break;
            emit_arguments(escape_json_fragment(buffer_.substr(0, safe)), &events);
            buffer_.erase(0, safe);
            continue;
        }

        if (state_ == State::ToolTail) {
            size_t ws = 0;
            while (ws < buffer_.size() && ascii_space(buffer_[ws])) ++ws;
            buffer_.erase(0, ws);
            if (buffer_.rfind(kToolClose, 0) == 0) {
                buffer_.erase(0, std::char_traits<char>::length(kToolClose));
                finish_call(&events);
                state_ = State::Text;
                continue;
            }
            if (!final && is_prefix_of(buffer_, kToolClose)) break;
            state_ = State::Broken;
            continue;
        }
    }
    if (state_ == State::Broken) partial_ = true;
    return events;
}

}  // namespace toolparse
