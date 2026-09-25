"""Exercise the OpenAI surface the way a real client does, and judge the result.

Not a mock of the server: this talks HTTP to a running api_server.py (or
scripts/dev_stub.py) and asserts the things clients actually depend on --
incremental deltas, a terminal event, matching ids, monotonic sequence numbers.

    python scripts/check_surface.py                     # against dev_stub
    python scripts/check_surface.py --base http://127.0.0.1:8090/v1
    python scripts/check_surface.py --base https://x.trycloudflare.com/v1 --key sk-...
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.parse

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print("%-34s %s %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


def request(base, path, body=None, key=None, method=None, timeout=300):
    u = urllib.parse.urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    headers = {"Content-Type": "application/json", "Accept": "*/*"}
    if key:
        headers["Authorization"] = "Bearer " + key
    payload = json.dumps(body).encode() if body is not None else None
    if payload is not None:
        headers["Content-Length"] = str(len(payload))
    conn.request(method or ("POST" if payload is not None else "GET"),
                 u.path.rstrip("/") + path, body=payload, headers=headers)
    return conn, conn.getresponse()


def read_events(resp):
    """Yield (name, data, at) for each SSE event, timed from the caller's t0."""
    name = None
    t0 = time.time()
    while True:
        line = resp.readline()
        if not line:
            return
        line = line.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            raw = line.split(":", 1)[1].strip()
            if raw == "[DONE]":
                yield "__done__", None, time.time() - t0
                continue
            try:
                data = json.loads(raw)
            except Exception:
                data = {"_raw": raw}
            yield name, data, time.time() - t0
        elif line == "":
            name = None


def check_chat_stream(base, key, model):
    conn, resp = request(base, "/chat/completions",
                         {"model": model, "messages": [{"role": "user", "content": "hi"}],
                          "stream": True, "max_tokens": 64}, key)
    deltas, first, last, done = [], None, None, False
    for name, data, at in read_events(resp):
        if name == "__done__":
            done = True
            continue
        if first is None:
            first = at
        last = at
        if isinstance(data, dict):
            for ch in data.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    deltas.append(ch["delta"]["content"])
    conn.close()
    text = "".join(deltas)
    record("chat stream: [DONE] sent", done, "deltas=%d ttfb=%.3fs total=%.3fs"
           % (len(deltas), first or 0.0, last or 0.0))
    record("chat stream: incremental", len(deltas) >= 3, "n=%d" % len(deltas))
    record("chat stream: first byte before end",
           bool(first is not None and last is not None and first < last - 0.001),
           "%.3fs .. %.3fs" % (first or 0, last or 0))
    record("chat stream: non-empty text", bool(text.strip()), text[:48].replace("\n", " "))
    return text


def check_responses_stream(base, key, model):
    conn, resp = request(base, "/responses",
                         {"model": model, "input": "hi", "stream": True,
                          "max_output_tokens": 64}, key)
    events, seqs, deltas, terminal = [], [], [], None
    first_delta_at = None
    for name, data, at in read_events(resp):
        if name == "__done__":
            continue
        events.append(name)
        if isinstance(data, dict) and isinstance(data.get("sequence_number"), int):
            seqs.append(data["sequence_number"])
        if name == "response.output_text.delta":
            if first_delta_at is None:
                first_delta_at = at
            d = data or {}
            ok_ids = all(k in d for k in ("item_id", "output_index", "content_index"))
            deltas.append((d.get("delta") or "", ok_ids))
        if name in ("response.completed", "response.failed", "response.incomplete"):
            terminal = name
    conn.close()
    record("responses: terminal event", terminal == "response.completed",
           "terminal=%s events=%d" % (terminal, len(events)))
    record("responses: item announced before delta",
           "response.output_item.added" in events
           and events.index("response.output_item.added")
           < (events.index("response.output_text.delta")
              if "response.output_text.delta" in events else len(events)),
           "")
    record("responses: deltas carry item_id/index",
           bool(deltas) and all(ok for _, ok in deltas), "n=%d" % len(deltas))
    record("responses: sequence_number monotonic",
           bool(seqs) and seqs == sorted(seqs) and len(set(seqs)) == len(seqs),
           "n=%d" % len(seqs))
    text = "".join(t for t, _ in deltas)
    return text, events


def check_responses_nonstream(base, key, model):
    conn, resp = request(base, "/responses",
                         {"model": model, "input": "hi", "max_output_tokens": 64}, key)
    body = json.loads(resp.read().decode())
    conn.close()
    out = body.get("output") or []
    ids = [o.get("id") for o in out]
    text = body.get("output_text") or ""
    record("responses non-stream: status completed",
           body.get("status") == "completed", "id=%s" % body.get("id"))
    record("responses non-stream: output item ids unique",
           len(ids) == len(set(ids)) and all(ids), str(ids))
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--key", default=None)
    ap.add_argument("--model", default="qwen3.8-flash-next-exl3")
    a = ap.parse_args()

    # /health lives at the server root, not under /v1.
    root = a.base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        conn, resp = request(root, "/health", key=a.key)
        record("health", resp.status == 200, "status=%d" % resp.status)
        conn.close()
    except Exception as exc:
        record("health", False, repr(exc))

    try:
        conn, resp = request(a.base, "/models", key=a.key)
        body = json.loads(resp.read().decode())
        conn.close()
        ids = [m.get("id") for m in body.get("data") or []]
        record("models", bool(ids), str(ids))
    except Exception as exc:
        record("models", False, repr(exc))

    chat_text = ""
    resp_text = ""
    try:
        chat_text = check_chat_stream(a.base, a.key, a.model)
    except Exception as exc:
        record("chat stream", False, repr(exc))
    try:
        resp_text, _events = check_responses_stream(a.base, a.key, a.model)
    except Exception as exc:
        record("responses stream", False, repr(exc))
    try:
        check_responses_nonstream(a.base, a.key, a.model)
    except Exception as exc:
        record("responses non-stream", False, repr(exc))

    record("chat and responses both return text",
           bool(chat_text.strip()) and bool(resp_text.strip()),
           "chat=%d chars responses=%d chars" % (len(chat_text), len(resp_text)))
    # Two independent samples of a sampled model are expected to differ; the stub
    # (greedy) matches exactly, which is why this is informational, not a check.
    print("INFO  chat/Responses text %s"
          % ("identical" if chat_text.strip() == resp_text.strip() else "differs (sampling)"),
          flush=True)

    bad = [n for n, ok, _ in RESULTS if not ok]
    print("\n%d checks, %d failed" % (len(RESULTS), len(bad)), flush=True)
    if bad:
        print("FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())