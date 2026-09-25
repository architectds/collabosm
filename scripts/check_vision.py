"""Verify image input: it either works, or it is refused -- never silently dropped.

The bug this guards against is the one that shipped: an OpenAI-style image part was
rendered as text into the prompt, so the request "succeeded" and the picture was
ignored. Anything that accepts an image must therefore either embed it or say no.

    python scripts/check_vision.py --base http://127.0.0.1:8099/v1 --key sk-...
    python scripts/check_vision.py --base https://<tunnel>/v1 --key <key>

Also checks the two refusals that keep this safe on a tunnelled box: remote URLs
off by default, and never a private/loopback/metadata address, and never a
filesystem path.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import sys
import urllib.parse

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print("%-42s %s %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


def post(base, path, body, key=None, timeout=300):
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    raw = json.dumps(body).encode()
    headers["Content-Length"] = str(len(raw))
    conn.request("POST", u.path.rstrip("/") + path, body=raw, headers=headers)
    resp = conn.getresponse()
    payload = resp.read().decode("utf-8", "replace")
    conn.close()
    try:
        return resp.status, json.loads(payload)
    except Exception:
        return resp.status, {"_raw": payload[:300]}


def get(base, path, key=None):
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=60)
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    conn.request("GET", u.path.rstrip("/") + path, headers=headers)
    resp = conn.getresponse()
    payload = resp.read().decode("utf-8", "replace")
    conn.close()
    try:
        return resp.status, json.loads(payload)
    except Exception:
        return resp.status, {"_raw": payload[:300]}


def tiny_png_data_url():
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        # 1x1 PNG, no Pillow needed
        b64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
               "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==")
    return "data:image/png;base64," + b64


def err_type(body):
    return ((body or {}).get("error") or {}).get("type") or ""


def err_msg(body):
    return ((body or {}).get("error") or {}).get("message") or str(body)[:200]


def chat_image(base, key, url, model):
    return post(base, "/chat/completions", {
        "model": model, "max_tokens": 32,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What colour is this image? One word."},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
    }, key)


def responses_image(base, key, url, model):
    return post(base, "/responses", {
        "model": model, "max_output_tokens": 32,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "What colour is this image? One word."},
            {"type": "input_image", "image_url": url},
        ]}],
    }, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--key", default=None)
    ap.add_argument("--model", default="qwen3.8-flash-next-exl3")
    a = ap.parse_args()

    status, st = get(a.base, "/status", a.key)
    vision = (st or {}).get("vision") or {}
    image_input = (st or {}).get("image_input") or {}
    available = bool(vision.get("available"))
    record("GET /v1/status", status == 200, "vision=%s enabled=%s remote_urls=%s"
           % (available, vision.get("enabled"), image_input.get("remote_urls")))

    # 1. a real image, by data: URL -- must be embedded, or refused
    png = tiny_png_data_url()
    code, body = chat_image(a.base, a.key, png, a.model)
    if available:
        text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        record("chat + image data: URL is answered", code == 200 and bool(text.strip()),
               "HTTP %s %s" % (code, text.strip()[:40]))
    else:
        record("chat + image is refused, not dropped",
               code == 400 and err_type(body) == "vision_unavailable",
               "HTTP %s %s" % (code, err_msg(body)[:70]))

    # 2. the same image through the Responses dialect
    code, body = responses_image(a.base, a.key, png, a.model)
    if available:
        text = (body or {}).get("output_text") or ""
        record("responses + image data: URL is answered",
               code == 200 and bool(text.strip()),
               "HTTP %s %s" % (code, text.strip()[:40]))
    else:
        record("responses + image is refused, not dropped",
               code == 400 and err_type(body) == "vision_unavailable",
               "HTTP %s %s" % (code, err_msg(body)[:70]))

    # 3. remote URL: off by default, and never a private address
    loopback = "http://127.0.0.1:1/nope.png"
    code, body = chat_image(a.base, a.key, loopback, a.model)
    if image_input.get("remote_urls"):
        record("remote URL to a private address is refused",
               code == 400 and ("public address" in err_msg(body)),
               "HTTP %s %s" % (code, err_msg(body)[:70]))
    else:
        record("remote image URLs are refused by default",
               code == 400 and ("disabled" in err_msg(body)),
               "HTTP %s %s" % (code, err_msg(body)[:70]))

    # 4. a filesystem path must never be read for a caller
    code, body = chat_image(a.base, a.key, "C:/Windows/win.ini", a.model)
    record("filesystem path is refused",
           code == 400 and ("unsupported image reference" in err_msg(body)
                            or "disabled" in err_msg(body)),
           "HTTP %s %s" % (code, err_msg(body)[:70]))

    # 5. text-only still works (no regression from the image plumbing)
    code, body = post(a.base, "/chat/completions", {
        "model": a.model, "max_tokens": 24,
        "messages": [{"role": "user", "content": "Say pong."}],
    }, a.key)
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    record("text-only still answered", code == 200 and bool(text.strip()),
           "HTTP %s %s" % (code, text.strip()[:40]))

    bad = [n for n, ok, _ in RESULTS if not ok]
    print("\n%d checks, %d failed" % (len(RESULTS), len(bad)), flush=True)
    if bad:
        print("FAILED: " + ", ".join(bad))
    if not available:
        print("\nvision is not available on this server: the refusal path was verified. "
              "On the box, set VISION=1 and watch VRAM before making it the default.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())