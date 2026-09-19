#!/usr/bin/env python3
"""Front-end regression checks against the deterministic api_fake_engine.py."""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request


def request(base, path, body):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, response.headers.get("content-type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("content-type", ""), error.read()


def check(condition, name, detail=""):
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}")


def parse_sse(raw):
    events = []
    for frame in raw.decode("utf-8").split("\n\n"):
        if not frame:
            continue
        check(frame.startswith("data: "), "sse-frame-prefix", frame[:60])
        payload = frame[6:]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18731")
    args = parser.parse_args()

    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
        "AQUBAScY42YAAAAASUVORK5CYII="
    )
    tiny_png_url = "data:image/png;base64," + tiny_png

    invalid = [
        ("temperature-type", "/v1/completions", {"prompt": "x", "temperature": "bad"}),
        ("top-k-type", "/v1/completions", {"prompt": "x", "top_k": "bad"}),
        ("stream-options-type", "/v1/completions", {"prompt": "x", "stream": True, "stream_options": "bad"}),
        ("stream-logprobs-rejected", "/v1/completions", {"prompt": "x", "stream": True, "logprobs": True}),
        ("budget-conflict", "/v1/completions", {"prompt": "x", "max_tokens": 2, "max_output_tokens": 3}),
        ("responses-input-type", "/v1/responses", {"input": 7}),
        ("responses-instructions-type", "/v1/responses", {"input": "x", "instructions": 7}),
        ("responses-reasoning-type", "/v1/responses", {"input": "x", "reasoning": "high"}),
        ("responses-effort-type", "/v1/responses", {"input": "x", "reasoning": {"effort": 7}}),
        ("responses-tools-type", "/v1/responses", {"input": "x", "tools": {}}),
        ("responses-malformed-image", "/v1/responses", {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}]}),
        ("responses-image-file-id", "/v1/responses", {"input": [{"type": "input_image", "file_id": "file_123"}]}),
        ("chat-http-image", "/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}]}]}),
        ("chat-video", "/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "input_video", "video_url": tiny_png_url}]}]}),
    ]
    for name, path, body in invalid:
        status, _, raw = request(args.base, path, body)
        payload = json.loads(raw)
        check(status == 400 and "error" in payload, name, raw[:200])

    status, _, raw = request(args.base, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": 4,
    })
    message = json.loads(raw)["choices"][0]["message"]
    check(status == 200 and message["content"] == "Hi" and message["reasoning_content"] == "",
          "thinking-disabled-content", raw[:300])

    vision_chat = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": tiny_png_url}},
            {"type": "text", "text": "describe"},
        ]}],
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": 4,
    }
    status, _, raw = request(args.base, "/v1/chat/completions", vision_chat)
    message = json.loads(raw)["choices"][0]["message"]
    check(status == 200 and message["content"] == "Hi", "chat-vision-nonstream", raw[:500])

    status, content_type, raw = request(
        args.base, "/v1/chat/completions", dict(vision_chat, stream=True))
    frames = parse_sse(raw)
    content = "".join(
        frame["choices"][0]["delta"].get("content", "")
        for frame in frames if isinstance(frame, dict) and frame.get("choices"))
    check(status == 200 and "text/event-stream" in content_type and content == "Hi" and
          frames[-1] == "[DONE]", "chat-vision-stream", raw[:800])

    vision_response = {
        "input": [
            {"type": "input_image", "image_url": tiny_png_url},
            {"type": "input_text", "text": "describe"},
        ],
        "enable_thinking": False,
        "temperature": 0,
        "max_output_tokens": 4,
    }
    status, _, raw = request(args.base, "/v1/responses", vision_response)
    response = json.loads(raw)
    check(status == 200 and response["output"][0]["content"][0]["text"] == "Hi",
          "responses-vision-flat-content", raw[:700])

    status, _, raw = request(args.base, "/v1/responses",
                             dict(vision_response, stream=True))
    events = parse_sse(raw)
    deltas = "".join(event.get("delta", "") for event in events
                     if event["type"] == "response.output_text.delta")
    check(status == 200 and deltas == "Hi" and events[-1]["type"] == "response.completed",
          "responses-vision-stream", raw[:1000])

    common = {"prompt": "x", "temperature": 0, "max_tokens": 4, "stop": "Paris"}
    status, _, raw = request(args.base, "/v1/completions", common)
    nonstream = json.loads(raw)["choices"][0]["text"]
    stream_body = dict(common, stream=True)
    status_stream, content_type, raw_stream = request(args.base, "/v1/completions", stream_body)
    frames = parse_sse(raw_stream)
    streamed = "".join(frame["choices"][0]["text"] for frame in frames
                       if isinstance(frame, dict) and frame.get("choices"))
    check(status == status_stream == 200 and "text/event-stream" in content_type and
          nonstream == streamed == " " and frames[-1] == "[DONE]",
          "cross-token-stop-stream-equivalence", raw_stream[:400])

    status, _, raw = request(args.base, "/v1/completions", {
        "prompt": "x", "temperature": 1, "seed": 7, "max_tokens": 2, "logprobs": True,
    })
    choice = json.loads(raw)["choices"][0]
    entries = choice["logprobs"]["content"]
    check(status == 200 and len(entries) == 2 and
          all(isinstance(entry["logprob"], (int, float)) for entry in entries),
          "sampled-logprobs-shape", raw[:500])

    response_body = {
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "instructions": "Be concise.",
        "reasoning": {"effort": "minimal"},
        "temperature": 0,
        "max_output_tokens": 4,
    }
    status, _, raw = request(args.base, "/v1/responses", response_body)
    response = json.loads(raw)
    check(status == 200 and response["status"] == "completed" and
          response["output"][0]["content"][0]["text"] == "Hi",
          "responses-nonstream", raw[:500])

    status, content_type, raw = request(args.base, "/v1/responses",
                                        dict(response_body, stream=True))
    events = parse_sse(raw)
    types = [event["type"] for event in events]
    expected = [
        "response.created", "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.done",
        "response.content_part.done", "response.output_item.done", "response.completed",
    ]
    deltas = "".join(event.get("delta", "") for event in events
                     if event["type"] == "response.output_text.delta")
    check(status == 200 and "text/event-stream" in content_type and types == expected and
          deltas == "Hi" and [event["sequence_number"] for event in events] == list(range(8)),
          "responses-sse", raw[:1200])

    tools = [
        {"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}}},
        }},
        {"type": "function", "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {
                "timezone": {"type": "string"}}},
        }},
    ]
    tool_body = {
        "messages": [{"role": "user", "content": "weather"}],
        "tools": tools,
        "temperature": 1,
        "seed": 9001,
        "max_tokens": 80,
    }
    status, _, raw = request(args.base, "/v1/chat/completions", tool_body)
    choice = json.loads(raw)["choices"][0]
    calls = choice["message"].get("tool_calls", [])
    check(status == 200 and calls, "chat-tool-present", raw[:1000])
    call = calls[0]
    call_args = json.loads(call["function"]["arguments"])
    check(status == 200 and choice["finish_reason"] == "tool_calls" and
          call["function"]["name"] == "get_weather" and
          call_args == {"city": "北京", "days": 3},
          "chat-tool-nonstream", raw[:1000])

    status, _, raw = request(args.base, "/v1/chat/completions",
                             dict(tool_body, stream=True))
    frames = parse_sse(raw)
    streamed_name, streamed_args = "", ""
    finish = None
    for frame in frames:
        if not isinstance(frame, dict) or not frame.get("choices"):
            continue
        item = frame["choices"][0]
        finish = item.get("finish_reason") or finish
        for tc in item.get("delta", {}).get("tool_calls", []):
            function = tc.get("function", {})
            streamed_name += function.get("name", "")
            streamed_args += function.get("arguments", "")
    check(finish == "tool_calls" and streamed_name == "get_weather" and
          json.loads(streamed_args) == {"city": "北京", "days": 3},
          "chat-tool-stream", raw[:1600])

    named_body = dict(tool_body, seed=9002,
                      tool_choice={"type": "function", "function": {"name": "get_time"}})
    status, _, raw = request(args.base, "/v1/chat/completions", named_body)
    named_calls = json.loads(raw)["choices"][0]["message"].get("tool_calls", [])
    check(status == 200 and named_calls, "chat-tool-named-present", raw[:1000])
    named_call = named_calls[0]
    check(status == 200 and named_call["function"]["name"] == "get_time" and
          json.loads(named_call["function"]["arguments"])["timezone"] == "Asia/Shanghai",
          "chat-tool-named", raw[:1000])

    required_body = dict(tool_body, seed=9003, tool_choice="required")
    status, _, raw = request(args.base, "/v1/chat/completions", required_body)
    required_calls = json.loads(raw)["choices"][0]["message"].get("tool_calls", [])
    check(status == 200 and required_calls, "chat-tool-required-present", raw[:1000])
    required_call = required_calls[0]
    check(status == 200 and required_call["function"]["name"] == "get_weather" and
          "上海" in json.loads(required_call["function"]["arguments"])["city"],
          "chat-tool-required-multiple", raw[:1000])

    response_tools = [
        {"type": "function", "name": "get_weather",
         "parameters": tools[0]["function"]["parameters"]},
        {"type": "function", "name": "get_time",
         "parameters": tools[1]["function"]["parameters"]},
    ]
    response_tool_body = {
        "input": "weather", "tools": response_tools, "temperature": 1,
        "seed": 9001, "max_output_tokens": 80,
    }
    status, _, raw = request(args.base, "/v1/responses", response_tool_body)
    response = json.loads(raw)
    function_calls = [item for item in response["output"] if item["type"] == "function_call"]
    check(status == 200 and function_calls, "responses-tool-present", raw[:1200])
    check(status == 200 and response["status"] == "completed" and
          function_calls[0]["name"] == "get_weather" and
          json.loads(function_calls[0]["arguments"])["days"] == 3,
          "responses-tool-nonstream", raw[:1200])

    status, _, raw = request(args.base, "/v1/responses",
                             dict(response_tool_body, stream=True))
    events = parse_sse(raw)
    event_types = [event["type"] for event in events]
    argument_deltas = "".join(event.get("delta", "") for event in events
                               if event["type"] == "response.function_call_arguments.delta")
    check("response.function_call_arguments.done" in event_types and
          event_types[-1] == "response.completed" and
          json.loads(argument_deltas)["city"] == "北京",
          "responses-tool-stream", raw[:1800])

    roundtrip = {
        "input": [
            {"type": "function_call", "call_id": "call_prev", "name": "get_weather",
             "arguments": "{\"city\":\"北京\"}"},
            {"type": "function_call_output", "call_id": "call_prev", "output": "sunny"},
            {"role": "user", "content": "summarize"},
        ],
        "tools": response_tools, "temperature": 0, "max_output_tokens": 4,
    }
    status, _, raw = request(args.base, "/v1/responses", roundtrip)
    check(status == 200, "responses-tool-roundtrip", raw[:1000])

    invalid_history = dict(tool_body)
    invalid_history["messages"] = [{"role": "assistant", "tool_calls": [{
        "id": "bad", "type": "function",
        "function": {"name": "get_weather", "arguments": "not json"},
    }]}]
    status, _, raw = request(args.base, "/v1/chat/completions", invalid_history)
    check(status == 400, "chat-tool-invalid-history", raw[:500])

    status, _, raw = request(args.base, "/v1/completions", {
        "prompt": "x", "temperature": 1, "seed": 424242, "max_tokens": 2,
    })
    check(status == 502, "engine-disconnect-first-request", raw[:300])
    status, _, raw = request(args.base, "/v1/completions", {
        "prompt": "x", "temperature": 0, "max_tokens": 2,
    })
    check(status == 200, "engine-auto-reconnect-next-request", raw[:300])

    slow_result = []
    slow = threading.Thread(target=lambda: slow_result.append(request(args.base, "/v1/completions", {
        "prompt": "x", "temperature": 1, "seed": 31337, "max_tokens": 2,
    })))
    slow.start()
    time.sleep(0.25)
    started = time.monotonic()
    status, _, raw = request(args.base, "/v1/completions", {
        "prompt": "x", "temperature": 0, "max_tokens": 2,
    })
    elapsed = time.monotonic() - started
    slow.join()
    check(status == 503 and elapsed < 1.0, "engine-busy-fast-503", raw[:300])
    check(slow_result and slow_result[0][0] == 200, "engine-busy-owner-completes")

    print("RESULT PASS")


if __name__ == "__main__":
    main()
