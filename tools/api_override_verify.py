#!/usr/bin/env python3
"""gdec-api 服务端参数覆盖（/admin/overrides）检查，由 tools/api_override_verify.sh 调用。

  python3 tools/api_override_verify.py main    --port 8733 --file F   # 功能检查（无管理密钥）
  python3 tools/api_override_verify.py persist --port 8733 --file F   # 重启 API 后覆盖表仍在
  python3 tools/api_override_verify.py auth    --port 8733 --key K    # 设了 GDEC_API_ADMIN_KEY 时的鉴权

每项打印 OK / FAIL，有 FAIL 时退出码 1。
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

BASE = ""
FAILS = []
PROMPT = [{"role": "user", "content": "用一句话介绍一下长城。"}]
PERSIST_TABLE = {
    "top_p": {"mode": "force", "value": 0.5},
    "reasoning_effort": {"mode": "default", "value": "medium"},
}


def check(name, cond, detail=""):
    print(("OK   " if cond else "FAIL ") + name + ("" if cond else "  -- " + str(detail)))
    if not cond:
        FAILS.append(name)


def http(method, path, body=None, headers=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def jpost(path, body, headers=None):
    st, h, text = http("POST", path, body, headers)
    try:
        j = json.loads(text)
    except ValueError:
        j = None
    return st, h, j, text


def ovr_header(h):
    for k, v in h.items():
        if k.lower() == "x-gdec-overrides":
            return set(v.split(","))
    return set()


def chat(body):
    """非流式 chat；返回 (status, 覆盖字段集合, content, reasoning, completion_tokens, 原文)"""
    b = {"messages": PROMPT}
    b.update(body)
    st, h, j, text = jpost("/v1/chat/completions", b)
    if st != 200 or not j:
        return st, ovr_header(h), None, None, None, text
    m = j["choices"][0]["message"]
    return st, ovr_header(h), m.get("content") or "", m.get("reasoning_content") or "", \
        j["usage"]["completion_tokens"], text


def chat_stream(body):
    b = {"messages": PROMPT, "stream": True, "stream_options": {"include_usage": True}}
    b.update(body)
    st, h, text = http("POST", "/v1/chat/completions", b)
    content, reasoning, usage = "", "", None
    for line in text.split("\n"):
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("usage"):
            usage = j["usage"]
        for c in j.get("choices", []):
            d = c.get("delta", {})
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
    return st, ovr_header(h), content, reasoning, usage


def set_table(t, headers=None):
    st, _, j, text = jpost("/admin/overrides", t, headers)
    return st, j, text


def run_main(file):
    st, _, html = http("GET", "/")
    check("dashboard 页面含覆盖面板和对话面板",
          st == 200 and 'id="ovr-body"' in html and 'id="c-send"' in html, st)
    st, _, text = http("GET", "/admin/overrides")
    j = json.loads(text) if st == 200 else {}
    check("GET /admin/overrides 初始为空表", st == 200 and j.get("overrides") == {}, text)
    check("持久化路径 = 测试文件", j.get("persist_path") == file, j.get("persist_path"))

    for name, bad in [
        ("temperature 超范围", {"temperature": {"mode": "force", "value": 3}}),
        ("top_k 非整数", {"top_k": {"mode": "force", "value": 1.5}}),
        ("reasoning_effort 非法", {"reasoning_effort": {"mode": "force", "value": "huge"}}),
        ("enable_thinking 非布尔", {"enable_thinking": {"mode": "force", "value": 1}}),
        ("未知字段", {"foo": {"mode": "force", "value": 1}}),
        ("未知模式", {"top_p": {"mode": "bogus", "value": 0.5}}),
        ("缺 value", {"top_p": {"mode": "force"}}),
        ("max_tokens = 0", {"max_tokens": {"mode": "force", "value": 0}}),
    ]:
        st, _, text = set_table(bad)
        check("拒绝非法覆盖：" + name + " -> 400", st == 400, "%s %s" % (st, text[:200]))
    st, _, text = http("GET", "/admin/overrides")
    check("非法 POST 不改动当前表", json.loads(text).get("overrides") == {}, text)

    # 参考：不覆盖，客户端自己传贪心 + 关思考 + 24 token
    st, ov, ref, rref, nref, raw = chat({"temperature": 0, "enable_thinking": False, "max_tokens": 24})
    check("参考请求成功", st == 200 and ref, raw[:300])
    check("参考请求无覆盖头", not ov, ov)
    check("参考请求无思考内容", rref == "", rref[:80] if rref else "")
    print("     参考输出：%r" % ref)

    forced = {
        "temperature": {"mode": "force", "value": 0},
        "enable_thinking": {"mode": "force", "value": False},
        "max_tokens": {"mode": "force", "value": 24},
        "reasoning_effort": {"mode": "force", "value": "high"},
    }
    st, j, text = set_table(forced)
    check("设置强制表成功并已持久化", st == 200 and j and j.get("saved") is True, text)
    check("high 规范化为 xhigh",
          j and j["overrides"].get("reasoning_effort", {}).get("value") == "xhigh", text)
    try:
        with open(file, encoding="utf-8") as f:
            on_disk = json.load(f)
    except (OSError, ValueError) as e:
        on_disk = repr(e)
    check("文件内容 = 服务端表", j and on_disk == j["overrides"], on_disk)
    st, _, text = http("GET", "/health")
    check("/health 报告 server_overrides",
          st == 200 and j and json.loads(text).get("server_overrides") == j["overrides"], text[:200])

    for seed in (1, 2):
        st, ov, out, rs, n, raw = chat({"temperature": 1.5, "top_k": 50, "enable_thinking": True,
                                        "reasoning_effort": "low", "max_tokens": 4000, "seed": seed})
        check("强制：客户端 temp=1.5/思考开/4000 token (seed %d) -> 与贪心参考一致" % seed,
              st == 200 and out == ref, "%s %r" % (st, out if out is not None else raw[:300]))
        check("强制：无思考内容 (seed %d)" % seed, rs == "", rs)
        check("强制：输出 <= 24 token (seed %d)" % seed, n is not None and n <= 24, n)
        check("强制：覆盖头列出 temperature/enable_thinking/max_tokens/reasoning_effort (seed %d)" % seed,
              {"temperature", "enable_thinking", "max_tokens", "reasoning_effort"} <= ov, ov)

    st, ov, out, rs, n, raw = chat({})
    check("强制：客户端什么都不传 -> 与参考一致", st == 200 and out == ref, "%s %r" % (st, out))
    check("强制：未传的 max_tokens 被补上", "max_tokens" in ov, ov)

    st, ov, out, rs, usage = chat_stream({"temperature": 0.9, "enable_thinking": True})
    check("强制（流式）：与参考一致", st == 200 and out == ref, "%s %r" % (st, out))
    check("强制（流式）：无思考内容", rs == "", rs[:80])
    check("强制（流式）：响应头带 X-Gdec-Overrides", "temperature" in ov, ov)

    st, ov, out, rs, n, raw = chat({"temperature": 0.8, "logprobs": True})
    check("强制贪心 + 客户端 logprobs -> 200，logprobs 被去掉",
          st == 200 and "logprobs" in ov, "%s %s %s" % (st, ov, raw[:200]))

    st, h, j, text = jpost("/v1/completions", {"prompt": "长城", "max_tokens": 100, "temperature": 1.2})
    check("completions：max_tokens 被压到 24",
          st == 200 and j["usage"]["completion_tokens"] <= 24 and "max_tokens" in ovr_header(h),
          "%s %s %s" % (st, ovr_header(h), text[:200]))
    check("completions：不应用思考字段", "enable_thinking" not in ovr_header(h), ovr_header(h))

    st, h, j, text = jpost("/v1/responses", {"input": "用一句话介绍一下长城。", "enable_thinking": True,
                                             "reasoning": {"effort": "low"}, "max_output_tokens": 500})
    ov = ovr_header(h)
    check("responses：覆盖 enable_thinking / reasoning_effort / temperature",
          st == 200 and {"enable_thinking", "reasoning_effort", "temperature"} <= ov,
          "%s %s %s" % (st, ov, text[:200]))
    check("responses：输出 <= 24 token",
          st == 200 and j["usage"]["output_tokens"] <= 24, text[:300])

    defaults = {
        "temperature": {"mode": "default", "value": 0},
        "enable_thinking": {"mode": "default", "value": False},
        "max_tokens": {"mode": "default", "value": 24},
    }
    st, j, text = set_table(defaults)
    check("设置默认值表成功", st == 200, text)
    st, ov, out, rs, n, raw = chat({})
    check("默认值：客户端不传 -> 用默认值，与参考一致", st == 200 and out == ref, "%s %r" % (st, out))
    check("默认值：覆盖头列出三项", {"temperature", "enable_thinking", "max_tokens"} <= ov, ov)
    st, ov, out, rs, n, raw = chat({"temperature": 0, "enable_thinking": False, "max_tokens": 24})
    check("默认值：客户端自己传了 -> 不改写（无覆盖头）", st == 200 and not ov and out == ref, "%s %s" % (st, ov))
    st, ov, out, rs, n, raw = chat({"enable_thinking": True})
    check("默认值：客户端传思考开 -> 保留，有思考内容", st == 200 and rs != "" and "enable_thinking" not in ov,
          "%s %s %r" % (st, ov, rs[:80] if rs else rs))
    st, ov, out, rs, n, raw = chat({"max_tokens": 8})
    check("默认值：客户端 max_tokens=8 不被放大", st == 200 and n is not None and n <= 8, n)

    st, j, text = set_table({"max_tokens": {"mode": "force", "value": 24}})
    st, ov, out, rs, n, raw = chat({"temperature": 0, "enable_thinking": False, "max_tokens": 8})
    check("强制上限：客户端更小的 max_tokens=8 保持不变", st == 200 and n <= 8 and "max_tokens" not in ov,
          "%s %s %s" % (st, n, ov))

    st, j, text = set_table({"overrides": PERSIST_TABLE})
    check("设置持久化测试表（{overrides: ...} 包装格式）",
          st == 200 and j and j["overrides"] == PERSIST_TABLE, text)


def run_persist(file):
    st, _, text = http("GET", "/admin/overrides")
    j = json.loads(text) if st == 200 else {}
    check("重启后覆盖表仍在", j.get("overrides") == PERSIST_TABLE, text)
    st, ov, out, rs, n, raw = chat({"top_p": 0.9, "temperature": 0.7, "max_tokens": 4})
    check("重启后强制 top_p 仍生效", st == 200 and "top_p" in ov, "%s %s" % (st, ov))


def run_auth(key):
    st, _, text = http("GET", "/admin/overrides")
    check("GET 显示需要管理密钥", st == 200 and json.loads(text).get("admin_key_required") is True, text)
    st, _, text = set_table({})
    check("无密钥 POST -> 401", st == 401, "%s %s" % (st, text[:200]))
    st, _, text = set_table({}, {"X-Admin-Key": key + "x"})
    check("错误密钥 POST -> 401", st == 401, st)
    st, _, text = set_table({"top_k": {"mode": "force", "value": 5}}, {"X-Admin-Key": key})
    check("X-Admin-Key 正确 -> 200", st == 200, "%s %s" % (st, text[:200]))
    st, _, text = set_table({}, {"Authorization": "Bearer " + key})
    check("Authorization: Bearer 正确 -> 200，表清空", st == 200 and json.loads(text)["overrides"] == {},
          "%s %s" % (st, text[:200]))


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "persist", "auth"])
    ap.add_argument("--port", type=int, default=8733)
    ap.add_argument("--file", default="")
    ap.add_argument("--key", default="")
    a = ap.parse_args()
    BASE = "http://127.0.0.1:%d" % a.port
    if a.cmd == "main":
        run_main(a.file)
    elif a.cmd == "persist":
        run_persist(a.file)
    else:
        run_auth(a.key)
    print("[%s] %s" % (a.cmd, "PASS" if not FAILS else "FAIL: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
