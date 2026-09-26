#!/usr/bin/env python3
"""collabosm frontend: our shell around an untouched llama.cpp WebUI.

Exactly two files here are ours: this server and shell.html. Everything under
upstream/ is a build of llama.cpp's tools/ui, vendored byte for byte -- see
UPSTREAM.md for the pinned commit, the build recipe and the checksums.

Routes
    /                       our shell; the WebUI is embedded as an iframe
    /?embed=1               the WebUI itself, served in the ROOT path space
    /control/status         state the shell's right rail renders
    /control/select         ask for a card+recipe; needs {"confirm": true} to bill
    /control/stop           stop the VM now (the point of the whole kit)
    /v1/*  /props  /slots  /tools  /models/*  /cors-proxy
                            proxied to the tunnel (key injected) or to --backend,
                            with the model field rewritten

The control plane is frontend/control.py: it drives scripts/up.sh over WSL,
keeps the CU ledger, refuses to start without an explicit confirmation, and
stops the box after `--idle-stop-min` of no chat traffic. `--mock` swaps in the
DemoControl below, which is the same contract at 1-minute-per-stage pacing, so
the shell can be driven with no card; `--fake-provision` keeps the real control
plane but runs frontend/fake_provision.py instead of touching Colab.

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
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from control import Control as ProvisionControl
except Exception as _exc:                        # pragma: no cover - import guard
    ProvisionControl = None
    _CONTROL_IMPORT_ERROR = _exc

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


class DemoControl:
    """The same contract as frontend/control.py, at demo pacing and no billing.

    `--mock` uses this: no WSL, no Colab, no CU, one stage per few seconds. It is
    what the rail is driven against when there is no card to spend, and it keeps
    the shell's contract honest -- status()/select()/stop() and nothing else.
    """

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

    def select(self, recipe_id: str, confirm: bool = False) -> dict:
        recipe = next((r for r in RECIPES if r["id"] == recipe_id), None)
        if recipe is None:
            return {"ok": False, "code": "unknown_recipe", "message": "没有这个配方"}
        if not confirm:
            return {"ok": True, "code": "confirm_required", "warning": {
                "recipe": recipe_id, "card": recipe["card"], "model": recipe["model"],
                "cu_per_hour": recipe["cu_per_hour"], "eta_min": recipe["eta_min"],
                "cu_estimate": round(recipe["cu_per_hour"] * (recipe["eta_min"] + 5) / 60, 2),
                "headline": "（演示）点“开始”开始计费：%s" % recipe["card"],
                "lines": ["这是 --mock 演示，不会真的申请实例。",
                          "真控制面是 frontend/control.py。"]}}
        threading.Thread(target=self._provision, args=(recipe_id,), daemon=True).start()
        return {"ok": True, "code": "started", "recipe": recipe_id}

    def stop(self, reason: str = "manual") -> dict:
        with self.lock:
            if self.state["stage"] in ("idle", "stopped"):
                return {"ok": True, "code": "nothing_to_stop"}
            self.state.update(stage="stopped", stage_label="已停机", progress=0.0,
                              live_model=None, metrics=None,
                              progress_note="已停机（%s）— 演示" % reason)
        return {"ok": True, "code": "stopping", "reason": reason}

    def backend_base(self):
        return None

    def api_key(self):
        return None

    def touch(self) -> None:
        return None

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

CONTROL = None                 # DemoControl or frontend.control.Control
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
        # Live tunnel if the box is up (the key is injected here and never
        # reaches the browser), otherwise --backend, which is the local stub or
        # the collabosm client proxy.
        live = CONTROL.backend_base()
        if live is None and not ARGS.mock and not ARGS.fake_provision:
            return self._json(503, {"error": {
                "message": "没有在跑的实例：在右栏选一个配方，等它变成“已就绪”再发消息",
                "type": "no_live_session"}})
        req = urllib.request.Request((live or ARGS.backend) + self.path,
                                     data=body, method=self.command)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "accept-encoding", "connection"):
                continue
            if live and k.lower() == "authorization":
                continue            # the VM's key is ours to hold, not the browser's
            req.add_header(k, v)
        if live and CONTROL.api_key():
            req.add_header("Authorization", "Bearer " + CONTROL.api_key())
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
            busy = st["stage"] in ("requesting", "uploading", "bootstrapping",
                                   "loading", "stopping")
            return self._json(200, [{"id": 0, "is_processing": busy}])
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
                body = json.loads(raw or b"{}")
            except Exception:
                return self._json(400, {"error": {"message": "bad json"}})
            res = CONTROL.select(body.get("recipe"), confirm=bool(body.get("confirm")))
            code = res.get("code")
            status = {"unknown_recipe": 404, "busy": 409, "budget": 409,
                      "unverified": 409, "stop_first": 409}.get(code, 200)
            return self._json(status, res)
        if path == "/control/stop":
            return self._json(200, CONTROL.stop("manual"))
        if path.startswith("/v1") or path.startswith("/models"):
            if path.endswith("/chat/completions") and raw:
                try:
                    payload = json.loads(raw)
                    st = CONTROL.status()
                    if st["stage"] == "ready":
                        payload["model"] = st["live_model"]
                    CONTROL.note_outgoing_model(payload.get("model"))
                    CONTROL.touch()
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
    ap.add_argument("--mock", action="store_true",
                    help="demo control plane: no WSL, no Colab, no CU")
    ap.add_argument("--mock-speed", type=float, default=4.0,
                    help="provisioning runs this much faster than real for demos")
    ap.add_argument("--fake-provision", action="store_true",
                    help="real control plane, but frontend/fake_provision.py instead of Colab")
    ap.add_argument("--session", default=os.environ.get("COLLABOSM_SESSION", "collabosm"))
    ap.add_argument("--wsl-distro", default=os.environ.get("COLLABOSM_WSL", "Ubuntu"))
    ap.add_argument("--budget-cu", type=float, default=200.0,
                    help="CU in the plan month; the rail shows what is left")
    ap.add_argument("--idle-stop-min", type=int, default=20,
                    help="stop the VM after this many minutes without chat traffic")
    ap.add_argument("--max-session-h", type=float, default=6.0)
    ap.add_argument("--state-dir", default=None,
                    help="where ledger.json lives (default ~/.collabosm)")
    ARGS = ap.parse_args()
    if ARGS.mock:
        CONTROL = DemoControl(mock_speed=ARGS.mock_speed)
        print("[fe] control    demo (--mock): nothing is billed", flush=True)
    else:
        if ProvisionControl is None:
            print("!! cannot import frontend/control.py: %r" % _CONTROL_IMPORT_ERROR,
                  file=sys.stderr)
            return 2
        CONTROL = ProvisionControl(os.path.dirname(HERE), session=ARGS.session,
                                   distro=ARGS.wsl_distro, budget_cu=ARGS.budget_cu,
                                   idle_stop_min=ARGS.idle_stop_min,
                                   max_session_h=ARGS.max_session_h,
                                   fake=ARGS.fake_provision, state_dir=ARGS.state_dir)
        print("[fe] control    %s -> wsl -d %s (idle stop %d min, budget %.0f CU)"
              % ("rehearsal" if ARGS.fake_provision else "colab", ARGS.wsl_distro,
                 ARGS.idle_stop_min, ARGS.budget_cu), flush=True)
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
