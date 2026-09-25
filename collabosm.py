#!/usr/bin/env python3
"""collabosm client: configuration, a loopback proxy, and the web UI.

Zero dependencies (stdlib only), and deliberately narrow: it speaks exactly the
two dialects Codex uses -- ``/v1/responses`` and ``/v1/chat/completions`` -- and
it does not manage the session. Bringing the A100 up is ``scripts/up.sh``,
stopping it is ``scripts/down.sh``; this client only talks to whatever endpoint
you point it at.

Why a local proxy instead of pointing the browser straight at the endpoint:

* the bearer key never reaches the browser,
* the page is same-origin, so there is no CORS surface,
* the UI can be served from a cache and updated independently of the VM.

    python collabosm.py config --init            # write ~/.collabosm/config.toml
    python collabosm.py ui                       # http://127.0.0.1:8790
    python collabosm.py chat "hello"             # the same path, in a terminal
    python collabosm.py update-ui                # pull ui/ from the repo (cached)
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOME = os.path.join(os.path.expanduser("~"), ".collabosm")
CONFIG_PATH = os.path.join(HOME, "config.toml")
UI_CACHE = os.path.join(HOME, "ui")

DEFAULT_REPO = "architectds/collabosm"
DEFAULT_REF = "master"
DEFAULT_DIALECT = "responses"
DEFAULT_MODEL = "qwen3.8-flash-next-exl3"
DEFAULT_PORT = 8790

# Latency and token accounting. The OpenAI protocol has no field for prefill or
# decode throughput, which is why no off-the-shelf UI can display it: we time the
# stream as it passes and derive the two rates here.
#
#   prefill ~= prompt_tokens / time-to-first-token   (includes queueing + prefill)
#   decode  ~= output_tokens / (total - time-to-first-token)
#
# Token counts come from the upstream's own usage when it reports one, otherwise
# they are estimated from the payload and flagged -- a wrong number presented as a
# measurement is worse than no number.
class Metrics:
    def __init__(self, limit=50):
        self.lock = threading.Lock()
        self.last = None
        self.turns = []
        self.limit = limit
        self.live = None

    def begin(self, path, body):
        return Turn(self, path, body)

    def _push(self, rec):
        with self.lock:
            self.last = rec
            self.turns.append(rec)
            del self.turns[:-self.limit]
            self.live = None

    def snapshot(self):
        with self.lock:
            return {"last": self.last, "live": self.live, "turns": list(self.turns)}


class Turn:
    """Times one request as it goes past."""

    def __init__(self, metrics, path, body):
        self.m = metrics
        self.path = path
        self.t0 = time.time()
        self.ttft = None
        self.chars = 0
        self.events = 0
        self.prompt_tokens = None
        self.output_tokens = None
        self.estimated = False
        self.event_name = None
        if body:
            self.prompt_tokens = max(1, len(body) // 4)
            self.estimated = True
        self._live()

    def _live(self):
        with self.m.lock:
            self.m.live = {"path": self.path, "started": self.t0, "ttft": self.ttft,
                           "chars": self.chars, "events": self.events}

    def feed(self, line):
        """Watch one SSE line. The bytes are always forwarded untouched."""
        try:
            text = line.decode("utf-8", "replace").strip()
        except Exception:
            return
        if text.startswith("event:"):
            self.event_name = text.split(":", 1)[1].strip()
            return
        if not text.startswith("data:"):
            return
        raw = text.split(":", 1)[1].strip()
        if raw == "[DONE]":
            return
        try:
            ev = json.loads(raw)
        except Exception:
            return
        name = self.event_name
        self.events += 1
        piece = ""
        if name == "response.output_text.delta":
            piece = ev.get("delta") or ""
        elif name is None and isinstance(ev.get("choices"), list):
            d = ((ev.get("choices") or [{}])[0].get("delta") or {})
            piece = d.get("content") or ""
        if piece and self.ttft is None:
            self.ttft = time.time() - self.t0
        self.chars += len(piece)
        usage = ev.get("usage") or (ev.get("response") or {}).get("usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens") or usage.get("input_tokens")
            ct = usage.get("completion_tokens") or usage.get("output_tokens")
            if pt:
                self.prompt_tokens = pt
                self.estimated = False
            if ct:
                self.output_tokens = ct
            if pt or ct:
                self.estimated = False
        self._live()

    def end(self, payload=None):
        total = time.time() - self.t0
        if payload:
            try:
                body = json.loads(payload.decode("utf-8", "replace"))
            except Exception:
                body = {}
            usage = body.get("usage") or (body.get("response") or {}).get("usage") or {}
            if isinstance(usage, dict):
                pt = usage.get("prompt_tokens") or usage.get("input_tokens")
                ct = usage.get("completion_tokens") or usage.get("output_tokens")
                self.prompt_tokens = pt or self.prompt_tokens
                self.output_tokens = ct or self.output_tokens
                if pt or ct:
                    self.estimated = False
            if self.output_tokens is None:
                if isinstance(body.get("output_text"), str):
                    self.chars = len(body["output_text"])
                else:
                    try:
                        self.chars = len(((body.get("choices") or [{}])[0]
                                          .get("message") or {}).get("content") or "")
                    except Exception:
                        pass
        if self.output_tokens is None:
            self.output_tokens = max(0, round(self.chars / 4))
            self.estimated = True
        rec = {
            "path": self.path,
            "at": self.t0,
            "streamed": self.ttft is not None,
            "total_s": round(total, 3),
            "ttft_s": round(self.ttft, 3) if self.ttft is not None else None,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            # Without a first-token time there is no boundary between prefill and
            # decode, so neither rate is reported rather than one being invented.
            "prefill_tps": (round(self.prompt_tokens / self.ttft)
                            if self.prompt_tokens and self.ttft else None),
            "decode_tps": (round(self.output_tokens / (total - self.ttft), 1)
                           if self.output_tokens and self.ttft is not None
                           and total > self.ttft else None),
            "estimated": self.estimated,
        }
        self.m._push(rec)
        return rec


METRICS = Metrics()


# The two dialects this client implements, and the upstream path for each.
ROUTES = {
    "/v1/chat/completions": "/chat/completions",
    "/v1/responses": "/responses",
    "/v1/models": "/models",
}


# --------------------------------------------------------------------------- config
def write_toml(path, data):
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append("[%s]" % key)
            for k, v in value.items():
                lines.append("%s = %s" % (k, json.dumps(v)))
            lines.append("")
        else:
            lines.append("%s = %s" % (key, json.dumps(value)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    import tomllib
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def resolve(args):
    """CLI flag > environment > config file. Never prints the key."""
    cfg = load_config()
    endpoint = (getattr(args, "endpoint", None)
                or os.environ.get("COLLABOSM_ENDPOINT")
                or cfg.get("endpoint") or "")
    api_key = (getattr(args, "api_key", None)
               or os.environ.get("COLLABOSM_API_KEY")
               or cfg.get("api_key") or "")
    dialect = (getattr(args, "dialect", None)
               or cfg.get("dialect") or DEFAULT_DIALECT)
    model = (getattr(args, "model", None)
             or cfg.get("model") or DEFAULT_MODEL)
    if dialect not in ("responses", "chat"):
        raise SystemExit("dialect must be 'responses' or 'chat'")
    if endpoint:
        endpoint = endpoint.rstrip("/")
    elif not getattr(args, "allow_no_endpoint", False):
        raise SystemExit(
            "no endpoint configured.\n"
            "  python %s config --init --endpoint https://<host>/v1\n"
            "or pass --endpoint, or set COLLABOSM_ENDPOINT." % os.path.basename(sys.argv[0]))
    return {"endpoint": endpoint, "api_key": api_key, "dialect": dialect,
            "model": model, "ui_dir": None}


def default_ui_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "ui")
    if os.path.isdir(local):
        return local
    return os.path.join(UI_CACHE, "current")


# --------------------------------------------------------------------------- proxy
def upstream_url(endpoint, path):
    """Map a client path onto the configured endpoint, keeping one '/v1'."""
    tail = ROUTES[path]
    base = endpoint
    if base.endswith("/v1") and tail.startswith("/v1"):
        tail = tail[3:]
    return base + tail


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "collabosm-client"
    cfg = {}

    def log_message(self, fmt, *a):
        sys.stderr.write("[ui] %s\n" % (fmt % a))

    def _json(self, code, payload, close=False):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if close or code >= 400:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    # --- static -----------------------------------------------------------
    def _static_path(self):
        rel = urllib.parse.urlparse(self.path).path
        if rel in ("/", "/index.html"):
            rel = "/index.html"
        rel = os.path.normpath(rel).lstrip("/\\")
        if rel.startswith("..") or os.path.isabs(rel):
            return None
        full = os.path.join(self.cfg["ui_dir"], rel)
        return full if os.path.isfile(full) else None

    def _serve_static(self):
        full = self._static_path()
        if not full:
            return self._json(404, {"error": {"message": "not found: %s" % self.path}})
        ctype = {".html": "text/html; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8",
                 ".json": "application/json",
                 ".svg": "image/svg+xml"}.get(os.path.splitext(full)[1],
                                              "application/octet-stream")
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/local/metrics"):
            return self._json(200, METRICS.snapshot())
        if self.path.startswith("/health"):
            return self._json(200, {"ok": True,
                                    "endpoint": self.cfg["endpoint"],
                                    "dialect": self.cfg["dialect"],
                                    "ui_dir": self.cfg["ui_dir"]})
        if self.path.startswith("/local/config"):
            # Read-only, non-secret: what the page needs to label itself.
            return self._json(200, {
                "endpoint": self.cfg["endpoint"],
                "dialect": self.cfg["dialect"],
                "model": self.cfg["model"],
                "has_key": bool(self.cfg["api_key"]),
                "dialects": ["responses", "chat"],
            })
        if self.path.startswith("/v1/"):
            return self._proxy()
        return self._serve_static()

    def do_HEAD(self):
        return self.do_GET()

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/responses"):
            return self._json(404, {"error": {
                "message": "this client implements exactly two dialects: "
                           "/v1/responses and /v1/chat/completions"}})
        return self._proxy()

    # --- the actual passthrough ------------------------------------------
    def _proxy(self):
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path not in ROUTES:
            return self._json(404, {"error": {"message": "unsupported path: %s" % path}})
        if not self.cfg["endpoint"]:
            return self._json(503, {"error": {"message": "no endpoint configured"}})

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        url = upstream_url(self.cfg["endpoint"], path)
        parts = urllib.parse.urlparse(url)
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        headers = {"Accept": self.headers.get("Accept") or "*/*"}
        if body is not None:
            headers["Content-Type"] = self.headers.get("Content-Type") or "application/json"
            headers["Content-Length"] = str(len(body))
        if self.cfg["api_key"]:
            headers["Authorization"] = "Bearer " + self.cfg["api_key"]

        # Start timing before the upstream request: for a non-streaming call the
        # upstream sends headers only after the answer exists, so timing from
        # getresponse() would measure almost nothing.
        turn = METRICS.begin(path, body) if self.command == "POST" else None

        timeout = float(self.cfg.get("timeout") or 900)
        conn = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        up = conn(parts.hostname, parts.port, timeout=timeout)
        try:
            up.request(self.command, target, body=body, headers=headers)
            resp = up.getresponse()
        except Exception as exc:
            return self._json(502, {"error": {"message": "upstream unreachable: %r" % exc}})

        ctype = resp.getheader("Content-Type") or "application/json"
        streaming = "text/event-stream" in ctype

        if streaming:
            # Pass SSE straight through, line by line, chunked, flushed. Buffering
            # it would defeat the whole point and Content-Length cannot describe a
            # stream whose length is unknown.
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    if turn:
                        turn.feed(line)
                    self.wfile.write(b"%x\r\n" % len(line) + line + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
                self.close_connection = True
                up.close()
            if turn:
                turn.end()
            return

        data = resp.read()
        if turn:
            turn.end(data)
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        up.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# --------------------------------------------------------------------------- commands
def cmd_config(args):
    cfg = load_config()
    if args.endpoint:
        cfg["endpoint"] = args.endpoint.rstrip("/")
    if args.api_key:
        cfg["api_key"] = args.api_key
    if args.dialect:
        cfg["dialect"] = args.dialect
    if args.model:
        cfg["model"] = args.model
    cfg.setdefault("dialect", DEFAULT_DIALECT)
    cfg.setdefault("model", DEFAULT_MODEL)
    if args.init or not os.path.exists(CONFIG_PATH) or any(
            (args.endpoint, args.api_key, args.dialect, args.model)):
        write_toml(CONFIG_PATH, cfg)
        print("wrote %s" % CONFIG_PATH)
    shown = dict(cfg)
    if shown.get("api_key"):
        k = shown["api_key"]
        shown["api_key"] = "%s...%s (%d chars)" % (k[:12], k[-4:], len(k))
    print(json.dumps(shown, indent=2))


def cmd_ui(args):
    cfg = resolve(args)
    ui_dir = args.ui_dir or default_ui_dir()
    if not os.path.isdir(ui_dir):
        raise SystemExit("no UI bundle at %s\n  run: python %s update-ui"
                         % (ui_dir, os.path.basename(sys.argv[0])))
    cfg["ui_dir"] = ui_dir
    cfg["timeout"] = args.timeout
    Proxy.cfg = cfg
    srv = Server((args.host, args.port), Proxy)
    print("collabosm ui  ->  http://%s:%d" % (args.host, args.port))
    print("  endpoint : %s (%s dialect)" % (cfg["endpoint"], cfg["dialect"]))
    print("  ui bundle: %s" % ui_dir)
    print("  key      : %s (never leaves this process)"
          % ("set" if cfg["api_key"] else "MISSING"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


def _chat_once(cfg, prompt, stream=True, max_tokens=512):
    """One turn through the same proxy path, printed to the terminal."""
    if cfg["dialect"] == "responses":
        payload = {"model": cfg["model"], "input": prompt,
                   "max_output_tokens": max_tokens, "stream": stream}
        path = "/responses"
    else:
        payload = {"model": cfg["model"],
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "stream": stream}
        path = "/chat/completions"

    url = upstream_url(cfg["endpoint"], "/v1" + path)
    parts = urllib.parse.urlparse(url)
    headers = {"Content-Type": "application/json",
               "Accept": "text/event-stream" if stream else "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = "Bearer " + cfg["api_key"]
    body = json.dumps(payload).encode()
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = cls(parts.hostname, parts.port, timeout=float(cfg.get("timeout") or 900))
    conn.request("POST", parts.path, body=body, headers=headers)
    resp = conn.getresponse()

    if not stream:
        data = json.loads(resp.read().decode("utf-8", "replace"))
        if cfg["dialect"] == "responses":
            print(data.get("output_text") or json.dumps(data)[:400])
        else:
            print(((data.get("choices") or [{}])[0].get("message") or {}).get("content", ""))
        return resp.status

    name = None
    in_reasoning = False
    while True:
        line = resp.readline()
        if not line:
            break
        text = line.decode("utf-8", "replace").rstrip("\r\n")
        if text.startswith("event:"):
            name = text.split(":", 1)[1].strip()
            continue
        if not text.startswith("data:"):
            continue
        raw = text.split(":", 1)[1].strip()
        if raw == "[DONE]":
            break
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        if cfg["dialect"] == "responses":
            if name == "response.reasoning_text.delta":
                if not in_reasoning:
                    sys.stdout.write("\n[thinking] ")
                    in_reasoning = True
                sys.stdout.write(ev.get("delta", ""))
            elif name == "response.output_text.delta":
                if in_reasoning:
                    sys.stdout.write("\n\n")
                    in_reasoning = False
                sys.stdout.write(ev.get("delta", ""))
            elif name in ("response.completed", "response.failed"):
                if name == "response.failed":
                    sys.stdout.write("\n[failed] " + json.dumps(ev)[:300])
                break
        else:
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    if not in_reasoning:
                        sys.stdout.write("\n[thinking] ")
                        in_reasoning = True
                    sys.stdout.write(d["reasoning_content"])
                if d.get("content"):
                    if in_reasoning:
                        sys.stdout.write("\n\n")
                        in_reasoning = False
                    sys.stdout.write(d["content"])
        sys.stdout.flush()
    print()
    conn.close()
    return resp.status


def cmd_chat(args):
    cfg = resolve(args)
    cfg["timeout"] = args.timeout
    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read()
    if not prompt.strip():
        raise SystemExit("nothing to send")
    return _chat_once(cfg, prompt, stream=not args.no_stream, max_tokens=args.max_tokens)


def cmd_update_ui(args):
    """Pull ui/ out of the repo at a pinned ref, verify, cache, print the digest."""
    repo = args.repo
    ref = args.ref
    url = "https://github.com/%s/archive/%s.tar.gz" % (repo, ref)
    print("fetching %s" % url)
    tmp = os.path.join(UI_CACHE, "download.tar.gz")
    os.makedirs(UI_CACHE, exist_ok=True)
    urllib.request.urlretrieve(url, tmp)

    dest = os.path.join(UI_CACHE, "current")
    os.makedirs(dest, exist_ok=True)
    want = json.loads(args.expect_sha) if args.expect_sha else None
    digests = {}
    with tarfile.open(tmp, "r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/")
            if len(parts) < 3 or parts[1] != "ui" or not member.isfile():
                continue
            name = parts[-1]
            if name not in ("index.html", "app.js", "styles.css"):
                continue
            fh = tar.extractfile(member)
            data = fh.read()
            with open(os.path.join(dest, name), "wb") as out:
                out.write(data)
            digests[name] = hashlib.sha256(data).hexdigest()
    if not digests:
        raise SystemExit("archive had no ui/ bundle: %s" % url)
    os.remove(tmp)
    bundle = hashlib.sha256(
        "".join("%s:%s\n" % (k, digests[k]) for k in sorted(digests)).encode()
    ).hexdigest()
    print(json.dumps(digests, indent=2))
    print("bundle sha256 %s" % bundle)
    print("cached in %s" % dest)
    if want and want.get("bundle") != bundle:
        raise SystemExit("WARNING: bundle digest %s != expected %s"
                         % (bundle, want.get("bundle")))
    return 0


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="collabosm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--endpoint", help="e.g. https://<host>.trycloudflare.com/v1")
        p.add_argument("--api-key", help="bearer key (or COLLABOSM_API_KEY)")
        p.add_argument("--dialect", choices=["responses", "chat"])
        p.add_argument("--model")

    p = sub.add_parser("config", help="show or write ~/.collabosm/config.toml")
    common(p)
    p.add_argument("--init", action="store_true")

    p = sub.add_parser("ui", help="serve the UI and proxy the two dialects")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--ui-dir")
    p.add_argument("--timeout", type=float, default=900.0)

    p = sub.add_parser("chat", help="one turn in the terminal")
    common(p)
    p.add_argument("prompt", nargs="*")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=900.0)

    p = sub.add_parser("update-ui", help="pull ui/ from the repo and cache it")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--expect-sha", help='{"bundle": "<sha256>"}')

    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 1
    return {"config": cmd_config, "ui": cmd_ui, "chat": cmd_chat,
            "update-ui": cmd_update_ui}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())