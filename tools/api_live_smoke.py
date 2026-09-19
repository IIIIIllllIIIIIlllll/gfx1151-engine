#!/usr/bin/env python3
"""Low-cost post-deploy checks for gdec-api against a real engine."""
import argparse
import json
import urllib.error
import urllib.request


def call(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.headers.get("content-type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("content-type", ""), error.read()


def sse(raw):
    events = []
    for frame in raw.decode("utf-8").split("\n\n"):
        if not frame:
            continue
        assert frame.startswith("data: "), frame[:80]
        payload = frame[6:]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def passed(name):
    print(f"PASS {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8731")
    args = parser.parse_args()

    status, _, raw = call(args.base, "GET", "/health")
    health = json.loads(raw)
    assert status == 200 and health["status"] == "ok" and health["slots"] == 1
    assert health["tool_calls"]["parsed"] is True
    passed("health")

    status, _, raw = call(args.base, "GET", "/v1/models")
    assert status == 200 and json.loads(raw)["data"]
    passed("models")

    status, _, raw = call(args.base, "POST", "/v1/completions",
                          {"prompt": "x", "temperature": "bad"})
    assert status == 400 and "error" in json.loads(raw)
    passed("invalid-type-400")

    chat_body = {
        "messages": [{"role": "user", "content": "Reply with exactly: Hi"}],
        "enable_thinking": False, "temperature": 0, "max_tokens": 8,
    }
    status, _, raw = call(args.base, "POST", "/v1/chat/completions", chat_body)
    message = json.loads(raw)["choices"][0]["message"]
    assert status == 200 and message["content"] and message["reasoning_content"] == ""
    passed("thinking-disabled-content")

    completion = {
        "prompt": "The capital of France is", "temperature": 0,
        "max_tokens": 8, "stop": "Paris",
    }
    status, _, raw = call(args.base, "POST", "/v1/completions", completion)
    nonstream = json.loads(raw)["choices"][0]["text"]
    status2, content_type, raw2 = call(
        args.base, "POST", "/v1/completions", dict(completion, stream=True))
    events = sse(raw2)
    streamed = "".join(event["choices"][0]["text"] for event in events
                       if isinstance(event, dict) and event.get("choices"))
    assert status == status2 == 200 and "text/event-stream" in content_type
    assert nonstream == streamed and "Paris" not in streamed and events[-1] == "[DONE]"
    passed("stop-stream-equivalence")

    status, _, raw = call(args.base, "POST", "/v1/completions", {
        "prompt": "hi", "temperature": 1.0, "seed": 1234,
        "max_tokens": 4, "logprobs": True,
    })
    response = json.loads(raw)
    logprobs = response["choices"][0]["logprobs"]["content"]
    assert status == 200 and len(logprobs) == response["usage"]["completion_tokens"]
    assert all(isinstance(item["logprob"], (int, float)) for item in logprobs)
    passed("sampled-logprobs")

    response_body = {
        "input": "Reply with exactly: Hi", "enable_thinking": False,
        "temperature": 0, "max_output_tokens": 8,
    }
    status, _, raw = call(args.base, "POST", "/v1/responses", response_body)
    response = json.loads(raw)
    assert status == 200 and response["status"] == "completed"
    assert response["output"][0]["content"][0]["text"]
    passed("responses-nonstream")

    status, content_type, raw = call(
        args.base, "POST", "/v1/responses", dict(response_body, stream=True))
    events = sse(raw)
    types = [event["type"] for event in events]
    required = [
        "response.created", "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.done",
        "response.content_part.done", "response.output_item.done", "response.completed",
    ]
    assert status == 200 and "text/event-stream" in content_type
    assert all(event_type in types for event_type in required)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    delta = "".join(event.get("delta", "") for event in events
                    if event["type"] == "response.output_text.delta")
    done = next(event["text"] for event in events
                if event["type"] == "response.output_text.done")
    assert delta == done and delta
    passed("responses-sse")

    print("RESULT PASS")


if __name__ == "__main__":
    main()
