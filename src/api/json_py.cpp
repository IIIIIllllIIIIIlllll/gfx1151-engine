// json_py.cpp — see json_py.h.
#include "json_py.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace json_py {
namespace {

// Decodes the code point starting at s[i]; returns bytes consumed (>=1).
// Inputs are JSON strings (valid UTF-8); invalid bytes are consumed one at a
// time as Latin-1-ish placeholders.
size_t utf8_decode(const std::string& s, size_t i, uint32_t& cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        cp = c;
        return 1;
    }
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

void escape_string(const std::string& s, std::string& out) {
    static const char* kHex = "0123456789abcdef";
    out += '"';
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\b': out += "\\b"; continue;
            case '\t': out += "\\t"; continue;
            case '\n': out += "\\n"; continue;
            case '\f': out += "\\f"; continue;
            case '\r': out += "\\r"; continue;
            default: break;
        }
        if (c < 0x20) {
            out += "\\u00";
            out += kHex[c >> 4];
            out += kHex[c & 0xf];
            continue;
        }
        out += static_cast<char>(c);
    }
    out += '"';
}

// CPython repr(str): single quotes normally, double quotes when the value has
// a single quote but no double quote; escapes match PyUnicode_Repr.
void repr_string(const std::string& s, std::string& out) {
    const bool has_sq = s.find('\'') != std::string::npos;
    const bool has_dq = s.find('"') != std::string::npos;
    const char quote = (has_sq && !has_dq) ? '"' : '\'';
    out += quote;
    for (size_t i = 0; i < s.size();) {
        uint32_t cp;
        size_t n = utf8_decode(s, i, cp);
        if (cp == static_cast<uint32_t>(quote) || cp == '\\') {
            out += '\\';
            out += static_cast<char>(cp);
        } else if (cp == '\n') {
            out += "\\n";
        } else if (cp == '\r') {
            out += "\\r";
        } else if (cp == '\t') {
            out += "\\t";
        } else if (cp < 0x20 || (cp >= 0x7f && cp <= 0xa0)) {
            char tmp[8];
            std::snprintf(tmp, sizeof(tmp), "\\x%02x", cp);
            out += tmp;
        } else {
            out.append(s, i, n);
        }
        i += n;
    }
    out += quote;
}

void repr_rec(const json& v, std::string& out) {
    switch (v.type()) {
        case json::value_t::null: out += "None"; return;
        case json::value_t::boolean: out += v.get<bool>() ? "True" : "False"; return;
        case json::value_t::number_integer:
        case json::value_t::number_unsigned: out += py_int(v); return;
        case json::value_t::number_float: out += py_float(v.get<double>()); return;
        case json::value_t::string: repr_string(v.get_ref<const std::string&>(), out); return;
        case json::value_t::array: {
            out += '[';
            bool first = true;
            for (const auto& e : v) {
                if (!first) out += ", ";
                first = false;
                repr_rec(e, out);
            }
            out += ']';
            return;
        }
        case json::value_t::object: {
            out += '{';
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) out += ", ";
                first = false;
                repr_string(it.key(), out);
                out += ": ";
                repr_rec(it.value(), out);
            }
            out += '}';
            return;
        }
        default: out += "None"; return;
    }
}

void dumps_rec(const json& v, bool spaced, std::string& out) {
    const char* comma = spaced ? ", " : ",";
    const char* colon = spaced ? ": " : ":";
    switch (v.type()) {
        case json::value_t::null: out += "null"; return;
        case json::value_t::boolean: out += v.get<bool>() ? "true" : "false"; return;
        case json::value_t::number_integer:
        case json::value_t::number_unsigned: out += py_int(v); return;
        case json::value_t::number_float: out += py_float(v.get<double>()); return;
        case json::value_t::string:
            escape_string(v.get_ref<const std::string&>(), out);
            return;
        case json::value_t::array: {
            out += '[';
            bool first = true;
            for (const auto& e : v) {
                if (!first) out += comma;
                first = false;
                dumps_rec(e, spaced, out);
            }
            out += ']';
            return;
        }
        case json::value_t::object: {
            out += '{';
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) out += comma;
                first = false;
                escape_string(it.key(), out);
                out += colon;
                dumps_rec(it.value(), spaced, out);
            }
            out += '}';
            return;
        }
        default: return;
    }
}

}  // namespace

std::string py_float(double d) {
    if (d == 0.0) return std::signbit(d) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    size_t i = 0;
    bool neg = false;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
        neg = s[i] == '-';
        ++i;
    }
    std::string digits;
    for (; i < s.size() && s[i] != 'e' && s[i] != 'E'; ++i) {
        if (s[i] >= '0' && s[i] <= '9') digits += s[i];
    }
    const int exp10 = i < s.size() ? std::atoi(s.c_str() + i + 1) : 0;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int decpt = exp10 + 1;  // value == 0.<digits> * 10^decpt

    std::string out = neg ? "-" : "";
    if (decpt <= -4 || decpt > 16) {
        out += digits.substr(0, 1);
        if (digits.size() > 1) {
            out += '.';
            out += digits.substr(1);
        }
        out += 'e';
        const int e = exp10;
        out += e < 0 ? '-' : '+';
        std::string es = std::to_string(e < 0 ? -e : e);
        if (es.size() < 2) es.insert(es.begin(), '0');
        out += es;
    } else if (decpt <= 0) {
        out += "0.";
        out.append(static_cast<size_t>(-decpt), '0');
        out += digits;
    } else if (static_cast<size_t>(decpt) >= digits.size()) {
        out += digits;
        out.append(static_cast<size_t>(decpt) - digits.size(), '0');
        out += ".0";
    } else {
        out += digits.substr(0, static_cast<size_t>(decpt));
        out += '.';
        out += digits.substr(static_cast<size_t>(decpt));
    }
    return out;
}

std::string py_int(const json& v) {
    char buf[32];
    std::to_chars_result r;
    if (v.type() == json::value_t::number_unsigned) {
        r = std::to_chars(buf, buf + sizeof(buf), v.get<uint64_t>());
    } else {
        r = std::to_chars(buf, buf + sizeof(buf), v.get<int64_t>());
    }
    return std::string(buf, r.ptr);
}

std::string py_repr(const json& v) {
    std::string out;
    repr_rec(v, out);
    return out;
}

std::string py_str(const json& v) {
    if (v.is_string()) return v.get_ref<const std::string&>();
    return py_repr(v);
}

std::string dumps(const json& v, bool spaced) {
    std::string out;
    dumps_rec(v, spaced, out);
    return out;
}

}  // namespace json_py
