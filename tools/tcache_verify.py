#!/usr/bin/env python3
"""KV prefix reuse across chat turns (driven by tools/tcache_verify.sh).

text   : a long sampled reply, then a follow-up that replays it (with
         reasoning_content, like a real client). Through the token-cache API
         (--on) the follow-up must reuse the whole first turn; the same
         follow-up through a GDEC_API_TOKCACHE=0 API (--off) is reported for
         comparison only (a reply that happens to be canonical BPE reuses
         fully there too).
vision : a text turn, then an image follow-up (the text prefix must be
         reused), then a second image added (prefix incl. image 1 reused),
         then image 1 swapped for other pixels (must NOT reuse past it).
Prints PASS/FAIL per check; exit code 0 only if all pass.
"""
import argparse
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib


def post(port, body, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_secs"] = time.time() - t0
    return out


def usage(out):
    u = out.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), cached


def assistant_msg(out):
    m = out["choices"][0]["message"]
    msg = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        msg["reasoning_content"] = m["reasoning_content"]
    return msg


def png(w, h, pixel):
    raw = b"".join(b"\0" + bytes(c for x in range(w) for c in pixel(x, y))
                   for y in range(h))
    def chunk(t, d):
        return (struct.pack(">I", len(d)) + t + d +
                struct.pack(">I", zlib.crc32(t + d) & 0xffffffff))
    data = (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def circle(x, y):
    inside = (x - 112) ** 2 + (y - 112) ** 2 < 70 ** 2
    return (220, 30, 30) if inside else (250, 250, 250)


def stripes(x, y):
    return (30, 60, 200) if (x // 28) % 2 else (240, 220, 40)


def check(ok, name, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    return ok


def run_text(a):
    ok = True
    system = ("你是资深 Linux 运维工程师，回答时给出完整可运行的脚本和逐行解释。" * 8)
    user1 = ("写一个 bash 脚本，巡检一台服务器：遍历 /etc /var/log /proc/meta "
             "/sys/class/net /usr/local/bin /opt/app/meta 等目录，检查权限、"
             "磁盘占用、最近修改的文件、systemd 服务状态、网卡与 URL "
             "https://example.com/api/v1/meta?id=42&x=/meta 的连通性，"
             "输出 JSON 报告。然后逐段解释。")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user1}]
    print(f"[text] 第 1 轮：采样生成长回复（max_tokens {a.max_tokens}，temperature 1.0）...",
          flush=True)
    o1 = post(a.on, {"messages": msgs, "max_tokens": a.max_tokens, "temperature": 1.0})
    p1, c1, _ = usage(o1)
    fin = o1["choices"][0].get("finish_reason")
    print(f"[text]   prompt {p1}, 生成 {c1} token，finish={fin}，{o1['_secs']:.0f}s", flush=True)
    msgs2 = msgs + [assistant_msg(o1), {"role": "user", "content": "把上面的脚本压缩成 5 行以内。"}]
    body2 = {"messages": msgs2, "max_tokens": 16, "temperature": 0}
    want = p1 + c1 - 32  # generated ids the engine ingested, minus seam slack
    on = post(a.on, body2)
    p2, _, k_on = usage(on)
    print(f"[text] 第 2 轮（token 缓存 ON）：prompt {p2}，cached {k_on}，{on['_secs']:.1f}s", flush=True)
    ok &= check(k_on >= want, "text-followup-reuse",
                f"cached {k_on} >= 第 1 轮 {p1}+{c1}-32 = {want}")
    off = post(a.off, body2)
    p3, _, k_off = usage(off)
    print(f"[text] 第 2 轮（token 缓存 OFF，对照）：prompt {p3}，cached {k_off}，"
          f"{off['_secs']:.1f}s", flush=True)
    if k_off >= want:
        print("INFO 对照组也整段复用：这次回复恰好是规范分词，A/B 无差异（不算失败）")
    else:
        print(f"INFO 对照组只复用 {k_off}/{p3}：旧路径在第一个非规范 token 处断开，"
              "引擎日志里有 'no live prefix' 行给出位置")
    return ok


def run_vision(a):
    ok = True
    img1, img2 = png(224, 224, circle), png(224, 224, stripes)
    system = ("你是一个认真细致的助手。" * 40)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "用三句话介绍一下 PNG 格式。"}]
    base = {"max_tokens": 200, "temperature": 0, "enable_thinking": False}
    print("[vision] V1：纯文本一轮 ...", flush=True)
    o1 = post(a.on, dict(base, messages=msgs))
    p1, c1, _ = usage(o1)
    msgs += [assistant_msg(o1)]

    def user_img(img, text):
        return {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img}},
            {"type": "text", "text": text}]}

    msgs2 = msgs + [user_img(img1, "图里画的是什么？一句话。")]
    o2 = post(a.on, dict(base, messages=msgs2, max_tokens=40))
    p2, c2, k2 = usage(o2)
    print(f"[vision] V2（文本历史 + 图 1）：prompt {p2}，cached {k2}", flush=True)
    ok &= check(k2 >= p1 + c1 - 32, "vision-text-prefix-reuse",
                f"cached {k2} >= V1 {p1}+{c1}-32")
    msgs3 = msgs2 + [assistant_msg(o2), user_img(img2, "这张新图和上一张有什么不同？一句话。")]
    o3 = post(a.on, dict(base, messages=msgs3, max_tokens=16))
    p3, _, k3 = usage(o3)
    print(f"[vision] V3（保留图 1 + 新增图 2）：prompt {p3}，cached {k3}", flush=True)
    ok &= check(k3 >= p2 + c2 - 32, "vision-add-image-reuse",
                f"cached {k3} >= V2 {p2}+{c2}-32")
    # Negative: image 1 replaced by other pixels (same size, so the same
    # grid and the same token ids as V2 — only the pixel hash tells them
    # apart). The first image pad sits ~6 tokens after V1's reply
    # (<|im_end|>\n<|im_start|>user\n<|vision_start|>).
    msgs4 = msgs + [user_img(img2, "图里画的是什么？一句话。")]
    o4 = post(a.on, dict(base, messages=msgs4, max_tokens=16))
    p4, _, k4 = usage(o4)
    print(f"[vision] V4（图 1 换成别的像素）：prompt {p4}，cached {k4}", flush=True)
    ok &= check(p4 == p2 and k4 <= p1 + c1 + 8, "vision-swapped-image-not-reused",
                f"prompt {p4} == V2 {p2}，cached {k4} <= 图片前文本 {p1}+{c1}+8")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", type=int, required=True, help="token-cache API port")
    ap.add_argument("--off", type=int, required=True, help="GDEC_API_TOKCACHE=0 API port")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--only", choices=["text", "vision"])
    a = ap.parse_args()
    ok = True
    if a.only in (None, "text"):
        ok &= run_text(a)
    if a.only in (None, "vision"):
        ok &= run_vision(a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
