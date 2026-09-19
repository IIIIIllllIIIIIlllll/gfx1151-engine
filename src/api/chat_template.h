// chat_template.h — C++17 port of the model's chat template
// (chat_template.jinja, Apache-2.0).
//
// Hand-ported against public Jinja2 semantics and verified byte-exact
// against jinja2 3.1.6 (tools/template_ab.py). Jinja behaviors that
// differed from naive expectations, all verified empirically:
//
//  1. `|tojson` is `json.dumps(value, ensure_ascii=False)`: object keys stay
//     in insertion order, separators are (', ', ': '), and neither the
//     surrogate-pair escaping nor the htmlsafe `<>&'` rewriting that plain
//     Jinja2's built-in tojson filter applies. The deployment renders the
//     template through transformers' apply_chat_template, so that policy is
//     the one that matters -- a live-service probe
//     (tools/probe_service_chat.py: prompt_tokens for a tool schema with
//     unsorted keys and `<&>`) is what settled it. Inputs must therefore be
//     nlohmann::ordered_json. Markers other than `|tojson` are unaffected.
//     Known divergence: integers outside uint64 are parsed by nlohmann as
//     double (see note 9), so exact digits are lost where Python keeps them.
//  2. `|trim` = Python str.strip(): strips the full Unicode whitespace
//     set — \t-\r, \x1c-\x20, space, \x85, \xa0, U+1680, U+2000-U+200A,
//     U+2028/2029, U+202F, U+205F, U+3000 — but NOT U+200B (ZWSP) or
//     U+FEFF (BOM). Implemented as UTF-8-aware edge trimming.
//  3. `is true` is an identity test: enable_thinking/preserve_thinking
//     given as JSON null behave as FALSE, while absent (undefined)
//     behaves as true. `|default('xhigh')` only replaces undefined, so
//     reasoning_effort=null falls through to the error path and renders
//     as Python str(None) = "None" in the message.
//  4. `tool_call.arguments != ''` is Python inequality: anything except
//     the exact string "" is != '', including null/false/0/[]/{}; a
//     non-object then dies inside `|items` with TypeError
//     "Can only get item pairs from a mapping.".
//  5. `'image' in item` is Python `in`: substring test for string items,
//     key test for objects, membership test for arrays; scalars raise
//     TypeError "argument of type '<pytype>' is not iterable".
//  6. A missing `tool_call.name` raises jinja2.UndefinedError
//     "'dict object' has no attribute 'name'" (attribute-style access on
//     a dict); a non-string name raises TypeError
//     'can only concatenate str (not "<pytype>") to str'.
//  7. preserve_thinking is forced ON for assistant messages after the
//     last genuine user query (ns.last_query_index backward scan), even
//     when preserve_thinking=false.
//  8. Floats print as Python repr: shortest round-trip, ".0" appended
//     for integral values, exponent with sign and >=2 digits ("1e-05").
//  9. Known divergence: integers beyond uint64 range. Python is
//     arbitrary precision; nlohmann/json parses them as double, losing
//     exact digits. Not reachable from realistic tokenizer JSON.
// 10. Object key order: inputs must be nlohmann::ordered_json so that
//     `tool_call.arguments|items` iterates in client insertion order
//     (Python dict semantics). `|tojson` re-sorts keys recursively by
//     code point (= UTF-8 byte order).

#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace chat_template {

using json = nlohmann::ordered_json;

// Each pointer is nullptr when the corresponding template variable is
// UNDEFINED (absent from the request). A JSON null value is *not* the
// same as undefined — see note 3 above.
struct Options {
    const json* tools = nullptr;
    const json* add_generation_prompt = nullptr;
    const json* reasoning_effort = nullptr;
    const json* enable_thinking = nullptr;
    const json* preserve_thinking = nullptr;
    const json* add_vision_id = nullptr;
};

struct RenderResult {
    bool ok = false;
    std::string text;   // when ok
    std::string error;  // when !ok: the exact raise_exception / jinja2
                        // runtime error message the template would produce
};

// `messages` is nullptr when the request carries no "messages" key at all
// (template-undefined); an empty array raises "No messages provided.".
RenderResult render_chat_template(const json* messages, const Options& opts);

}  // namespace chat_template
