#!/usr/bin/env python3
"""collabosm frontend: our shell around an untouched llama.cpp WebUI.

The files here are ours -- this server, shell.html and the control plane in
control.py; everything under upstream/ is a build of llama.cpp's tools/ui,
vendored byte for byte -- see UPSTREAM.md for the pinned commit, the build
recipe and the checksums.

Routes
    /                       our shell; the WebUI is embedded as an iframe
    /?embed=1               the WebUI itself, served in the ROOT path space
    /control/status         state the shell's right rail renders
    /control/select         ask for a card+recipe; needs {"confirm": true} to bill
    /control/cancel         forget a pending confirmation
    /control/stop           stop the VM now (the point of the whole kit)
    /v1/*  /props  /slots  /tools  /models/*  /cors-proxy
                            proxied to the tunnel (key injected) or to --backend,
                            with the model field rewritten

The control plane is frontend/control.py: it drives scripts/up.sh over WSL,
keeps the CU ledger, refuses to start without an explicit confirmation, and
stops the box after `--idle-stop-min` of no chat traffic. `--fake-provision`
keeps that real control plane but runs frontend/fake_provision.py instead of
touching Colab; `--mock` is the same rehearsal with a ledger that is never
written, for looking at the shell. (There used to be a separate demo control
plane for --mock. It drifted from the real contract until the shell could no
longer start it, so the rehearsal is now the real code with one process swapped.)

Why the WebUI is served at the root and not under a prefix: it fetches /v1/*,
/props and /tools with absolute paths, and its own assets with relative ones.
Under a prefix both break, and it does not report an error -- it disables the
composer with cursor-not-allowed, which reads as "cannot connect". Root serving
is the only configuration that works (measured, see UPSTREAM.md).

Why every POST must be same-origin JSON
    /control/select starts billing and /v1/* spends GPU time with the VM key
    injected. Leaving out Access-Control-Allow-Origin stops a foreign page from
    READING our answers, not from SENDING a text/plain or form POST, and those
    need no preflight. So a POST must be application/json (a cross-origin JSON
    POST needs a preflight, which this server never grants), must come from this
    origin or from no browser at all, and must name this server in Host --
    which is what shuts out DNS rebinding.

The chat UI never needs to know which card is live: this process owns that, and
rewrites the model field on the way out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
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

CONTROL = None                 # frontend.control.Control
ARGS: argparse.Namespace
MIME = {"html": "text/html", "js": "text/javascript", "css": "text/css",
        "json": "application/json", "svg": "image/svg+xml", "png": "image/png",
        "ico": "image/x-icon", "webmanifest": "application/manifest+json",
        "txt": "text/plain", "map": "application/json"}

LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")
LANGS = ("en", "zh", "ja")

# The WebUI shows this when someone chats with no card up, so it follows the
# language the shell was set to (a cookie the shell writes).
NO_LIVE_SESSION = {
    "en": "No instance is running: pick a recipe in the right column and wait for "
          "Ready before sending.",
    "zh": "没有在跑的实例：在右栏选一个配方，等它变成“已就绪”再发消息",
    "ja": "稼働中のインスタンスがありません。右の列でレシピを選び、「準備完了」に"
          "なってから送信してください。",
}


def _names_this_server(hostport: str) -> bool:
    """True if `hostport` (a Host header, or an Origin's host[:port]) is this server."""
    try:
        parts = urllib.parse.urlsplit("//" + hostport)
        host, port = parts.hostname, parts.port
    except ValueError:
        return False
    return ((port or 80) == ARGS.port
            and (host in LOOPBACK_NAMES or host == ARGS.host))


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

    def _lang(self):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "collabosm_lang" and v in LANGS:
                return v
        return "en"

    def _refuse_cross_site(self):
        """None if this POST may proceed, else (status, message) -- see the docstring."""
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return 415, "POST bodies must be application/json"
        if not _names_this_server(self.headers.get("Host") or ""):
            return 403, "Host does not name this server"
        origin = self.headers.get("Origin")
        if origin is not None:
            u = urllib.parse.urlsplit(origin)
            if u.scheme != "http" or not _names_this_server(u.netloc):
                return 403, "cross-origin POST refused"
        site = (self.headers.get("Sec-Fetch-Site") or "").lower()
        if site and site not in ("same-origin", "none"):
            return 403, "cross-site POST refused"
        return None

    def _proxy(self, body=None):
        # Live tunnel if the box is up (the key is injected here and never
        # reaches the browser), otherwise --backend, which is the local stub or
        # the collabosm client proxy.
        live = CONTROL.backend_base()
        if live is None and not ARGS.mock and not ARGS.fake_provision:
            return self._json(503, {"error": {"message": NO_LIVE_SESSION[self._lang()],
                                              "type": "no_live_session"}})
        req = urllib.request.Request((live or ARGS.backend) + self.path,
                                     data=body, method=self.command)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "accept-encoding", "connection", "cookie"):
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
        refused = self._refuse_cross_site()
        if refused:
            code, message = refused
            self.log_message("refused POST %s: %s", path, message)
            return self._json(code, {"error": {"message": message,
                                               "type": "cross_site_refused"}})
        if path == "/control/select":
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                return self._json(400, {"error": {"message": "bad json"}})
            res = CONTROL.select(body.get("recipe"), confirm=bool(body.get("confirm")))
            status = {"unknown_recipe": 404, "busy": 409, "budget": 409, "unverified": 409,
                      "stop_first": 409, "stale_ledger": 409,
                      "attached": 409}.get(res.get("code"), 200)
            return self._json(status, res)
        if path == "/control/cancel":
            return self._json(200, CONTROL.cancel())
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
    ap.add_argument("--port", type=int, default=int(os.environ.get("FRONTEND_PORT", 3020)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", default=os.environ.get("FRONTEND_BACKEND", "http://127.0.0.1:8790"),
                    help="OpenAI-compatible engine (or the collabosm client proxy)")
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--mock", action="store_true",
                    help="rehearsal with a ledger that is never written: no WSL, no Colab, no CU")
    ap.add_argument("--mock-speed", type=float, default=1.0,
                    help="with --mock, provisioning runs this much faster than the ~24 s rehearsal")
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
    ap.add_argument("--external-endpoint", default=os.environ.get("COLLABOSM_EXTERNAL"),
                    help="proxy /v1/* straight to an endpoint that is already running "
                         "(no Colab, no CU, no auto-stop): e.g. http://127.0.0.1:11435/v1")
    ap.add_argument("--external-key", default=os.environ.get("COLLABOSM_EXTERNAL_KEY"))
    ap.add_argument("--external-model", default=None,
                    help="label to show in the rail for --external-endpoint")
    ARGS = ap.parse_args()
    if ProvisionControl is None:
        print("!! cannot import frontend/control.py: %r" % _CONTROL_IMPORT_ERROR,
              file=sys.stderr)
        return 2
    if ARGS.mock:
        # fake_provision.py reads its pacing from the environment it inherits
        os.environ.setdefault("COLLABOSM_FAKE_SECONDS",
                              "%.1f" % (24.0 / max(0.1, ARGS.mock_speed)))
    CONTROL = ProvisionControl(os.path.dirname(HERE), session=ARGS.session,
                               distro=ARGS.wsl_distro, budget_cu=ARGS.budget_cu,
                               idle_stop_min=ARGS.idle_stop_min,
                               max_session_h=ARGS.max_session_h,
                               fake=ARGS.mock or ARGS.fake_provision,
                               state_dir=ARGS.state_dir, persist_ledger=not ARGS.mock,
                               external=ARGS.external_endpoint,
                               external_key=ARGS.external_key,
                               external_model=ARGS.external_model)
    mode = ("mock (nothing billed, nothing written)" if ARGS.mock
            else "rehearsal" if ARGS.fake_provision else "colab")
    print("[fe] control    %s -> wsl -d %s (idle stop %d min, budget %.0f CU)"
          % (mode, ARGS.wsl_distro, ARGS.idle_stop_min, ARGS.budget_cu), flush=True)
    if ARGS.external_endpoint:
        print("[fe] external   %s (nothing billed, nothing auto-stopped)"
              % ARGS.external_endpoint, flush=True)
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
