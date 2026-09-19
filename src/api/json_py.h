// json_py.h — serialization that matches CPython's output byte-for-byte.
//
// Needed in two places that must not drift apart:
//   * chat_template.cpp's `|tojson` (transformers renders via json.dumps);
//   * main.cpp's SSE frames and logprob fields, which the live service emits
//     with Python's default separators.
// Non-streaming HTTP bodies use compact separators instead (FastAPI's
// JSONResponse) — that is what the `spaced` flag selects.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace json_py {

using json = nlohmann::ordered_json;

// Python json.dumps(value, ensure_ascii=False), insertion order preserved.
//   spaced=true  -> separators (", ", ": ")   (Python's default)
//   spaced=false -> separators (",", ":")     (FastAPI JSONResponse)
// Only `"`, `\` and the C0 controls are escaped; non-ASCII passes through.
std::string dumps(const json& v, bool spaced);

// CPython repr(float): shortest round-trip digits, scientific when the decimal
// exponent is <= -4 or > 16, integral fixed output always gets a trailing ".0".
std::string py_float(double d);
std::string py_int(const json& v);
std::string py_repr(const json& v);   // repr(value)
std::string py_str(const json& v);    // str(value): identity for strings

}  // namespace json_py
