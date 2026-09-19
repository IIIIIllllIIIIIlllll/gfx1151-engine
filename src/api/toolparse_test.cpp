#include "toolparse.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using toolparse::json;

namespace {

void check(bool ok, const char* name) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", name);
        std::exit(1);
    }
    std::printf("PASS %s\n", name);
}

json tools() {
    json properties = {
        {"path", {{"type", "string"}}},
        {"content", {{"type", "string"}}},
        {"mode", {{"type", "integer"}}},
        {"flag", {{"type", "boolean"}}},
        {"items", {{"type", "array"}}},
    };
    json result = json::array();
    result.push_back(
        {{"type", "function"},
         {"function",
          {{"name", "write_file"},
           {"parameters", {{"type", "object"}, {"properties", properties}}}}}});
    result.push_back(
        {{"type", "function"},
         {"function", {{"name", "read_file"}, {"parameters", {{"type", "object"}}}}}});
    return result;
}

}  // namespace

int main() {
    int ids = 0;
    auto make_id = [&]() { return "call_" + std::to_string(++ids); };
    const std::string raw =
        "answer before\n\n<tool_call>\n<function=write_file>\n"
        "<parameter=path>\n/tmp/a\n</parameter>\n"
        "<parameter=content>\nhello \\\"world\\\"\nline2\n</parameter>\n"
        "<parameter=mode>\n7\n</parameter>\n"
        "<parameter=flag>\nTRUE\n</parameter>\n"
        "<parameter=items>\n[1, 2]\n</parameter>\n"
        "</function>\n</tool_call>";

    toolparse::StreamParser whole(tools(), make_id);
    whole.feed(raw);
    whole.finish();
    check(whole.content() == "answer before", "content split");
    check(whole.calls().size() == 1, "one complete call");
    json args = json::parse(whole.calls()[0].arguments);
    check(args["path"] == "/tmp/a" && args["content"] == "hello \\\"world\\\"\nline2",
          "string arguments");
    check(args["mode"] == 7 && args["flag"] == true && args["items"].size() == 2,
          "schema coercion");

    ids = 0;
    toolparse::StreamParser bytewise(tools(), make_id);
    std::string streamed_args;
    bool opened_before_close = false;
    for (char c : raw) {
        for (const auto& event : bytewise.feed(std::string(1, c))) {
            if (event.type == toolparse::EventType::ArgumentsDelta) streamed_args += event.data;
            if (event.type == toolparse::EventType::CallStart &&
                raw.find("</function>") != std::string::npos)
                opened_before_close = true;
        }
    }
    bytewise.finish();
    check(opened_before_close, "call opens incrementally");
    check(bytewise.calls().size() == 1 && streamed_args == bytewise.calls()[0].arguments,
          "streamed arguments equal final JSON");

    ids = 0;
    toolparse::StreamParser partial(tools(), make_id);
    std::string partial_args;
    const std::string unfinished =
        "<tool_call>\n<function=write_file>\n<parameter=content>\n" +
        std::string(200, 'x');
    for (const auto& event : partial.feed(unfinished))
        if (event.type == toolparse::EventType::ArgumentsDelta) partial_args += event.data;
    for (const auto& event : partial.finish())
        if (event.type == toolparse::EventType::ArgumentsDelta) partial_args += event.data;
    check(partial.has_partial_call() && partial.calls().empty(), "incomplete call state");
    check(partial_args.size() > 150, "incomplete string streams early");

    json messages = json::array(
        {{{"role", "assistant"},
          {"tool_calls",
           json::array({{{"id", "a"},
                         {"function", {{"name", "write_file"}, {"arguments", "{\"mode\":7}"}}}},
                        {{"id", "b"},
                         {"function", {{"name", "read_file"}, {"arguments", "{}"}}}}})}},
         {{"role", "tool"}, {"tool_call_id", "b"}, {"content", "second"}},
         {{"role", "tool"}, {"tool_call_id", "a"}, {"content", "first"}}});
    std::string error;
    check(toolparse::normalize_messages(&messages, &error), "normalize history");
    check(messages[0]["tool_calls"][0]["function"]["arguments"].is_object(),
          "history arguments parsed");
    check(messages[1]["tool_call_id"] == "a" && messages[2]["tool_call_id"] == "b",
          "tool results reordered");

    toolparse::ToolChoice choice;
    json required = "required";
    check(toolparse::parse_tool_choice(&required, json::array({tools()[0]}), &choice, &error) &&
              choice.name == "write_file" && !choice.parser_prefix().empty(),
          "required single choice");
    json named = {{"type", "function"}, {"function", {{"name", "read_file"}}}};
    check(toolparse::parse_tool_choice(&named, tools(), &choice, &error) &&
              choice.name == "read_file",
          "named choice");

    std::puts("RESULT PASS");
    return 0;
}
