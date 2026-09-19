#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace toolparse {

using json = nlohmann::ordered_json;

struct ToolChoice {
    enum class Mode { Auto, None, Required, Named };

    Mode mode = Mode::Auto;
    std::string name;

    bool tools_enabled() const { return mode != Mode::None; }
    bool forced() const { return mode == Mode::Required || mode == Mode::Named; }
    std::string parser_prefix() const;
    std::string prompt_suffix(bool thinking_enabled) const;
};

// Convert Chat Completions or Responses tool definitions to the canonical
// {type:"function", function:{...}} shape used by the Qwen template.
bool normalize_tools(const json* input, json* output, std::string* error);

// Validate auto/none/required and named function choices against `tools`.
bool parse_tool_choice(const json* input, const json& tools, ToolChoice* output,
                       std::string* error);

// Parse JSON-string arguments in assistant history and put contiguous tool
// results back into the order declared by the preceding tool_calls array.
bool normalize_messages(json* messages, std::string* error);

struct ToolCall {
    size_t index = 0;
    std::string id;
    std::string name;
    std::string arguments;
};

enum class EventType { Content, CallStart, ArgumentsDelta, CallEnd };

struct Event {
    EventType type = EventType::Content;
    size_t index = 0;
    std::string id;
    std::string name;
    std::string data;
};

// Incremental parser for Qwen's
// <tool_call><function=name><parameter=key>value...</parameter>...</function>
// format. Declared string parameters stream before their closing tag arrives;
// structured values wait for closure so schema coercion cannot change bytes
// that were already sent to a client.
class StreamParser {
  public:
    using IdFactory = std::function<std::string()>;

    StreamParser(json tools, IdFactory make_id, bool enabled = true);

    std::vector<Event> feed(std::string_view piece);
    std::vector<Event> finish();

    const std::string& content() const { return content_; }
    const std::vector<ToolCall>& calls() const { return calls_; }
    bool has_partial_call() const { return partial_; }

  private:
    enum class State { Text, Between, Parameter, ToolTail, Broken };

    std::vector<Event> pump(bool final);
    void emit_content(std::string text, std::vector<Event>* events);
    void emit_arguments(std::string text, std::vector<Event>* events);
    void start_call(std::string name, std::vector<Event>* events);
    void start_parameter(std::string key, std::vector<Event>* events);
    void finish_parameter(std::string raw, std::vector<Event>* events);
    void finish_call(std::vector<Event>* events);
    std::vector<std::string> schema_types(const std::string& function,
                                          const std::string& parameter) const;

    json tools_;
    IdFactory make_id_;
    bool enabled_ = true;
    State state_ = State::Text;
    std::string buffer_;
    std::string content_;
    std::vector<ToolCall> calls_;
    ToolCall current_;
    size_t next_index_ = 0;
    size_t argument_count_ = 0;
    std::string parameter_key_;
    std::vector<std::string> parameter_types_;
    bool stream_parameter_ = false;
    bool value_started_ = false;
    bool partial_ = false;
};

}  // namespace toolparse
