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

    const std::string code = "    if (ready) {\n        write();\n    }  \n\n";
    const std::string code_call =
        "<tool_call>\n<function=write_file>\n<parameter=content>\n" + code +
        "\n</parameter>\n</function>\n</tool_call>";
    toolparse::StreamParser code_whole(tools(), make_id);
    code_whole.feed(code_call);
    code_whole.finish();
    check(code_whole.calls().size() == 1 &&
              json::parse(code_whole.calls()[0].arguments)["content"] == code,
          "code indentation and trailing whitespace preserved");
    toolparse::StreamParser code_stream(tools(), make_id);
    std::string code_deltas;
    for (char character : code_call)
        for (const auto& event : code_stream.feed(std::string(1, character)))
            if (event.type == toolparse::EventType::ArgumentsDelta) code_deltas += event.data;
    for (const auto& event : code_stream.finish())
        if (event.type == toolparse::EventType::ArgumentsDelta) code_deltas += event.data;
    check(code_stream.calls().size() == 1 &&
              code_deltas == code_stream.calls()[0].arguments &&
              json::parse(code_deltas)["content"] == code,
          "bytewise code arguments match final content");

    json union_tools = tools();
    union_tools[0]["function"]["parameters"]["properties"]["content"]["type"] =
        json::array({"integer", "string"});
    toolparse::StreamParser union_parser(union_tools, make_id);
    union_parser.feed("<tool_call>\n<function=write_file>\n"
                      "<parameter=content>\n  text  \n</parameter>\n"
                      "</function>\n</tool_call>");
    union_parser.finish();
    check(union_parser.calls().size() == 1 &&
              json::parse(union_parser.calls()[0].arguments)["content"] == "  text  ",
          "union string argument retains whitespace");

    const std::string inline_tag =
        "<tool_call>\n<function=write_file>\n<parameter=content>\n"
        "const tag = \"</parameter>\";\n</parameter>\n</function>\n</tool_call>";
    toolparse::StreamParser inline_parser(tools(), make_id);
    inline_parser.feed(inline_tag);
    inline_parser.finish();
    check(inline_parser.calls().size() == 1 &&
              json::parse(inline_parser.calls()[0].arguments)["content"] ==
                  "const tag = \"</parameter>\";",
          "inline closing tag remains file content");

    const std::string ambiguous_tag =
        "<tool_call>\n<function=write_file>\n<parameter=content>\n"
        "before\n</parameter>\nafter\n</parameter>\n</function>\n</tool_call>";
    toolparse::StreamParser ambiguous(tools(), make_id);
    ambiguous.feed(ambiguous_tag);
    ambiguous.finish();
    check(ambiguous.has_partial_call() && ambiguous.calls().empty(),
          "ambiguous closing line does not return a truncated call");

    const std::string crlf_call =
        "<tool_call>\r\n<function=write_file>\r\n<parameter=content>\r\n"
        "  line\r\n\r\n</parameter>\r\n</function>\r\n</tool_call>";
    toolparse::StreamParser crlf(tools(), make_id);
    for (char character : crlf_call) crlf.feed(std::string(1, character));
    crlf.finish();
    check(crlf.calls().size() == 1 &&
              json::parse(crlf.calls()[0].arguments)["content"] == "  line\r\n",
          "CRLF framing preserves content");

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

    toolparse::StreamParser empty_partial(tools(), make_id);
    empty_partial.feed("<tool_call>\n<function=write_file>\n<parameter=content>\n");
    empty_partial.finish();
    check(empty_partial.has_partial_call() && empty_partial.calls().empty(),
          "empty unfinished parameter is incomplete");

    toolparse::StreamParser missing_tail(tools(), make_id);
    missing_tail.feed("<tool_call>\n<function=read_file>\n</function>\n");
    missing_tail.finish();
    check(missing_tail.has_partial_call() && missing_tail.calls().empty(),
          "missing tool close is incomplete");

    toolparse::StreamParser partial_second(tools(), make_id);
    partial_second.feed("<tool_call>\n<function=read_file>\n</function>\n</tool_call>"
                        "<tool_call>\n<function=write_file>\n<parameter=content>\npartial");
    partial_second.finish();
    check(partial_second.calls().size() == 1 && partial_second.has_partial_call(),
          "later incomplete call does not hide a partial result");

    json required_tools = tools();
    required_tools[0]["function"]["parameters"]["required"] =
        json::array({"path", "content"});
    toolparse::StreamParser missing_required(required_tools, make_id);
    missing_required.feed("<tool_call>\n<function=write_file>\n"
                          "<parameter=path>\n/tmp/a\n</parameter>\n"
                          "</function>\n</tool_call>");
    missing_required.finish();
    check(missing_required.has_partial_call() && missing_required.calls().empty(),
          "missing required argument is not returned as a complete call");

    toolparse::StreamParser unknown_tool(tools(), make_id);
    unknown_tool.feed("<tool_call>\n<function=delete_file>\n</function>\n</tool_call>");
    unknown_tool.finish();
    check(unknown_tool.has_partial_call() && unknown_tool.calls().empty(),
          "unknown function is not returned as a complete call");

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
