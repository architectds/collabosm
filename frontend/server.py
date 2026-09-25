#!/usr/bin/env python3
"""collabosm frontend: our shell around an untouched llama.cpp WebUI.

Exactly two files here are ours: this server and shell.html. Everything under
upstream/ is a build of llama.cpp's tools/ui, vendored byte for byte -- see
UPSTREAM.md for the pinned commit, the build recipe and the checksums.

Routes
    /                       our shell; the WebUI is embedded as an iframe
    /?embed=1               the WebUI itself, served in the ROOT path space
    /control/status         state the shell's right rail renders
    /control/select         ask for a card+recipe (P0-d wires this to restore.py)
    /v1/*  /props  /slots  /tools  /models/*  /cors-proxy
                            proxied to the engine, with the model field rewritten

Why the WebUI is served at the root and not under a prefix: it fetches /v1/*,
/props and /tools with absolute paths, and its own assets with relative ones.
Under a prefix both break, and it does not report an error -- it disables the
composer with cursor-not-allowed, which reads as "cannot connect". Root serving
is the only configuration that works (measured, see UPSTREAM.md).

The chat UI never needs to know which card is live: this process owns that, and
rewrites the model field on the way out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")

# --------------------------------------------------------------------------- #
# control plane                                                               #
# --------------------------------------------------------------------------- #

RECIPES = [
    {"id": "a100-80g/qwen38-fn",
     "card": "A100-80G High-RAM", "model": "Qwen3.8-Flash-Next",
     "quant": "EXL3 4.05bpw + MTP", "cu_per_hour": 7.52, "vram_gb": 80, "eta_min": 10},
    {"id": "a100-40g/qwen38-27b",
     "card": "A100-40G", "model": "Qwen3.8-27B",
     "quant": "EXL3 3.5bpw + MTP", "cu_per_hour": 5.37, "vram_gb": 40, "eta_min": 4},
]

# Stage names and durations stand in for restore.py + bootstrap + load. The real
# timings are measured and pinned in manifest.json: request+shape check ~60 s,
# weights 280 s, launch+load 259.5 s for the 80 GB recipe.
STAGES = [("requesting", "正在请求实例并校验形状", 5.0),
          ("downloading", "从 Hugging Face 拉取权重", 20.0),
          ("loading", "写入显存、建 KV 页表", 25.0)]

STAGE_ZH = {"requesting": "申请中", "downloading": "下载权重", "loading": "加载中"}


class Control:
    """The seam P0-d replaces: status()/select() are all the shell needs."""

    def __init__(self, mock_speed: float = 4.0, idle_stop_min: int = 20):
        self.mock_speed = mock_speed
        self.lock = threading.Lock()
        self.state = {
            "stage": "idle", "stage_label": "", "progress": 0.0,
            "progress_note": "选一台卡开始", "selected": None,
            "live_model": None, "metrics": None, "cu_left": 168.0,
            "idle_stop_min": idle_stop_min, "recipes": RECIPES,
            "outgoing_model": None,
            "footnote": "shell 模式 · 上游 WebUI 原样（diff = 0）",
        }

    def status(self) -> dict:
        with self.lock:
            return dict(self.state)

    def select(self, recipe_id: str) -> bool:
        if not any(r["id"] == recipe_id for r in RECIPES):
            return False
        threading.Thread(target=self._provision, args=(recipe_id,), daemon=True).start()
        return True

    def _provision(self, recipe_id: str) -> None:
        r = next(x for x in RECIPES if x["id"] == recipe_id)
        with self.lock:
            self.state.update(selected=recipe_id, stage="provisioning", progress=0.0,
                              stage_label=STAGE_ZH[STAGES[0][0]],
                              progress_note=STAGES[0][1], metrics=None, live_model=None)
        total = sum(s[2] for s in STAGES)
        done = 0.0
        for key, note, dur in STAGES:
            with self.lock:
                self.state.update(stage_label=STAGE_ZH[key], progress_note=note)
            d = dur / self.mock_speed
            steps = max(4, int(d * 4))
            for _ in range(steps):
                time.sleep(d / steps)
                done += (dur / steps)
                with self.lock:
                    self.state["progress"] = min(99.0, round(100.0 * done / total, 1))
        with self.lock:
            self.state.update(stage="ready", progress=100.0, live_model=r["model"],
                              metrics={"prefill": 2806.0, "decode": 97.4},
                              footnote="卡已就绪 · 可以聊天")

    def note_outgoing_model(self, model: str | None) -> None:
        with self.lock:
            self.state["outgoing_model"] = model


# --------------------------------------------------------------------------- #
# HTTP                                                                        #
# --------------------------------------------------------------------------- #

CONTROL: Control
ARGS: argparse.Namespace
MIME = {"html": "text/html", "js": "text/javascript", "css": "text/css",
        "json": "application/json", "svg": "image/svg+xml", "png": "image/png",
        "ico": "image/x-icon", "webmanifest": "application/manifest+json",
        "txt": "text/plain", "map": "application/json"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "collabosm-frontend"

    def log_message(self, fmt, *a):
        sys.stderr.write("[fe] " + (fmt % a) + "\n")

    # -- helpers -----------------------------------------------------------
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, fs):
        ext = fs.rsplit(".", 1)[-1].lower()
        with open(fs, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        # The shell must never be cached; upstream immutable assets carry hashes.
        self.send_header("Cache-Control", "no-store" if ext == "html" else "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, body=None):
        req = urllib.request.Request(ARGS.backend + self.path, data=body, method=self.command)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "accept-encoding", "connection"):
                continue
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=ARGS.timeout) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read(1)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self._json(e.code, {"error": {"message": str(e), "type": "upstream_error"}})
        except Exception as e:
            self._json(502, {"error": {"message": repr(e), "type": "upstream_error"}})

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/control/status":
            return self._json(200, CONTROL.status())
        if path == "/props":
            st = CONTROL.status()
            name = st["live_model"] or "collabosm"
            return self._json(200, {"model_path": name, "n_ctx": ARGS.ctx,
                                    "default_generation_settings": {"n_ctx": ARGS.ctx}})
        if path == "/slots":
            st = CONTROL.status()
            return self._json(200, [{"id": 0, "is_processing": st["stage"] == "provisioning"}])
        if path == "/tools":
            return self._json(200, [])
        if path.startswith("/v1") or path.startswith("/models") or path == "/cors-proxy":
            return self._proxy()
        if path == "/healthz":
            return self._json(200, {"ok": True})

        # The WebUI lives at the root path space; the shell is the entry point.
        # Its assets resolve against upstream/, never against the repo root.
        if path in ("/", ""):
            rel = "upstream/index.html" if "embed=1" in self.path else "shell.html"
            fs = os.path.normpath(os.path.join(HERE, rel))
        elif path in ("/shell", "/shell/"):
            fs = os.path.join(HERE, "shell.html")
        else:
            fs = os.path.normpath(os.path.join(UPSTREAM, path.lstrip("/")))
        if not fs.startswith(HERE):
            return self._json(403, {"error": {"message": "path escapes the app root"}})
        if not os.path.isfile(fs):
            fs = os.path.join(UPSTREAM, "index.html")     # SPA fallback
        return self._file(fs)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        path = self.path.split("?")[0]
        if path == "/control/select":
            try:
                rid = json.loads(raw or b"{}").get("recipe")
            except Exception:
                return self._json(400, {"error": {"message": "bad json"}})
            if not CONTROL.select(rid):
                return self._json(404, {"error": {"message": "unknown recipe", "recipe": rid}})
            return self._json(200, {"ok": True, "selected": rid})
        if path.startswith("/v1") or path.startswith("/models"):
            if path.endswith("/chat/completions") and raw:
                try:
                    payload = json.loads(raw)
                    st = CONTROL.status()
                    if st["stage"] == "ready":
                        payload["model"] = st["live_model"]
                    CONTROL.note_outgoing_model(payload.get("model"))
                    raw = json.dumps(payload).encode()
                    self.log_message("model -> %r", payload.get("model"))
                except Exception as e:
                    self.log_message("model rewrite skipped: %r", e)
            return self._proxy(raw)
        return self._json(404, {"error": {"message": "no such route", "path": path}})


def main() -> int:
    global CONTROL, ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("FRONTEND_PORT", 3010)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", default=os.environ.get("FRONTEND_BACKEND", "http://127.0.0.1:8790"),
                    help="OpenAI-compatible engine (or the collabosm client proxy)")
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--mock-speed", type=float, default=4.0,
                    help="provisioning runs this much faster than real for demos")
    ARGS = ap.parse_args()
    CONTROL = Control(mock_speed=ARGS.mock_speed)
    if not os.path.isfile(os.path.join(UPSTREAM, "index.html")):
        print("!! upstream/index.html missing -- the vendored WebUI build is not here",
              file=sys.stderr)
        return 2
    print("[fe] shell      http://%s:%d/" % (ARGS.host, ARGS.port), flush=True)
    print("[fe] webui      http://%s:%d/?embed=1" % (ARGS.host, ARGS.port), flush=True)
    print("[fe] backend    %s" % ARGS.backend, flush=True)
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())