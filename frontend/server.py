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
    /control/couple         find the running service on our session and attach to it
    /control/key            the VM's key, for the rail's copy button (never in /control/status)
    /control/colab/<act>    the Colab guide: install | connect | cancel | check | disconnect
    /control/quit           close this app (the page's Quit; a running GPU is not stopped)
    /v1/*  /props  /slots  /tools  /models/*  /cors-proxy
                            proxied to the tunnel (key injected), or in a rehearsal to --backend,
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
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
ICONS = os.path.join(HERE, "icon")
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


class Meter:
    """Times one proxied turn from the bytes that pass through, in either dialect:
    /v1/chat/completions chunks, or /v1/responses events (what Codex and ModelDock
    send when they are pointed at this address).

    Rates come from the server's own `timings` (llama.cpp's field, which
    api_server sends on the last chat chunk); without them only the proxy's
    first-token time and the token counts are kept -- no tokens-per-second is
    invented from characters. A Responses turn carries no timings, and its rates
    are filled in from /v1/status's last_timings (control.py _absorb_server).
    """

    # the Responses events that carry the first generated text
    FIRST = ("response.output_text.delta", "response.reasoning_text.delta",
             "response.custom_tool_call_input.delta", "response.function_call_arguments.delta")

    def __init__(self):
        self.t0 = time.time()
        self.ttft = None
        self.timings = None
        self.usage = None

    def feed(self, line: bytes) -> None:
        if not line.startswith(b"data: {"):
            return
        try:
            ev = json.loads(line[6:])
        except ValueError:
            return
        self._take(ev)
        if self.ttft is None:
            d = ((ev.get("choices") or [{}])[0] or {}).get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or (
                    ev.get("type") in self.FIRST and ev.get("delta")):
                self.ttft = time.time() - self.t0

    def whole(self, data: bytes) -> None:
        try:
            self._take(json.loads(data))
        except ValueError:
            pass

    def _take(self, ev) -> None:
        if isinstance(ev, dict):
            self.timings = ev.get("timings") or self.timings
            self.usage = ev.get("usage") or self.usage
            if isinstance(ev.get("response"), dict):       # response.completed / .incomplete
                self.usage = ev["response"].get("usage") or self.usage

    def result(self):
        if self.ttft is None and not self.timings:
            return None                    # an error or an empty turn: nothing measured
        t, u = self.timings or {}, self.usage or {}
        # usage in either dialect: prompt/completion_tokens, or input/output_tokens
        total = u.get("prompt_tokens", u.get("input_tokens"))
        cached = (u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}).get("cached_tokens")
        new = (total - (cached or 0)) if isinstance(total, int) else None

        def rate(n, ms, given):
            if given:
                return given
            return round(n / ms * 1000.0, 1) if n and ms else None
        return {"at": int(time.time()), "source": "server" if t else "proxy",
                "ttft_s": round(self.ttft, 3) if self.ttft is not None else None,
                "total_s": round(time.time() - self.t0, 2),
                "prefill": rate(t.get("prompt_n"), t.get("prompt_ms"), t.get("prompt_per_second")),
                "decode": rate(t.get("predicted_n"), t.get("predicted_ms"),
                               t.get("predicted_per_second")),
                # llama.cpp's prompt_n leaves out the cached prefix; so does this one
                "prompt_n": t.get("prompt_n", new),
                "prompt_total": (((t.get("prompt_n") or 0) + (t.get("cache_n") or 0)) if t else total),
                "cache_n": t.get("cache_n", cached),
                "predicted_n": t.get("predicted_n", u.get("completion_tokens", u.get("output_tokens")))}


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
        if sys.stderr is not None:                        # pythonw.exe has none
            sys.stderr.write("[fe] " + (fmt % a) + "\n")

    def log_request(self, code="-", size="-"):
        # The page asks for /control/status every 1.5 s and the WebUI checks its service
        # worker every minute: a line for each buried everything else. Kept: every POST
        # (a start, a stop, a chat) and anything that failed.
        try:
            quiet = self.command in ("GET", "HEAD") and int(code) < 400
        except (TypeError, ValueError):
            quiet = False
        if not quiet:
            super().log_request(code, size)

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
        # reaches the browser); in a rehearsal, --backend (a local stub) instead.
        live = CONTROL.backend_base()
        if live is None and not ARGS.mock and not ARGS.fake_provision:
            return self._json(503, {"error": {"message": NO_LIVE_SESSION[self._lang()],
                                              "type": "no_live_session"}})
        target = live or ARGS.backend
        if not target:
            return self._json(503, {"error": {
                "message": "a rehearsal with nothing to chat with: start scripts/dev_stub.py "
                           "and pass --backend http://127.0.0.1:8099",
                "type": "no_backend"}})
        req = urllib.request.Request(target + self.path, data=body, method=self.command)
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "accept-encoding", "connection", "cookie"):
                continue
            if live and k.lower() == "authorization":
                continue            # the VM's key is ours to hold, not the browser's
            req.add_header(k, v)
        if live and CONTROL.api_key():
            req.add_header("Authorization", "Bearer " + CONTROL.api_key())
        meter = (Meter() if self.command == "POST"
                 and self.path.split("?")[0].endswith(("/chat/completions", "/responses")) else None)
        try:
            with urllib.request.urlopen(req, timeout=ARGS.timeout) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    # api_server answers with `Access-Control-Allow-Origin: *`, which is
                    # right for a keyed endpoint and wrong here: this proxy adds the key
                    # itself, so passing it on would let any web page read /v1/* through us
                    if k.lower().startswith("access-control-"):
                        continue
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                if "text/event-stream" in (resp.headers.get("Content-Type") or ""):
                    # Line by line: every SSE line goes out the moment it arrives,
                    # and the meter reads the same bytes on the way past.
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                        if meter:
                            meter.feed(line)
                else:
                    data = resp.read()
                    self.wfile.write(data)
                    if meter:
                        meter.whole(data)
            measured = meter.result() if meter else None
            if measured:
                CONTROL.note_metrics(measured)
        except urllib.error.HTTPError as e:
            self._json(e.code, {"error": {"message": str(e), "type": "upstream_error"}})
        except Exception as e:
            self._json(502, {"error": {"message": repr(e), "type": "upstream_error"}})

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        # DNS rebinding: a page on an attacker's name that resolves to 127.0.0.1 would
        # otherwise read the rail's state and, through the proxy, the model's -- a GET
        # needs this machine in Host just as a POST does
        if not _names_this_server(self.headers.get("Host") or ""):
            return self._json(403, {"error": {"message": "Host does not name this server",
                                              "type": "cross_site_refused"}})
        if path == "/control/status":
            return self._json(200, CONTROL.status())
        if path == "/props":
            # What the WebUI believes about the model: its context meter uses n_ctx,
            # and `modalities` decides whether it offers image upload at all.
            st = CONTROL.status()
            server = st.get("server") or {}
            n_ctx = webui_ctx(server, ARGS.ctx)
            vision = bool((server.get("vision") or {}).get("available"))
            name = st["live_model"] or "collabosm"
            return self._json(200, {"model_path": name, "n_ctx": n_ctx,
                                    "default_generation_settings": {"n_ctx": n_ctx},
                                    "modalities": {"vision": vision, "audio": False}})
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
        elif path in ("/collabosm-icon.svg", "/collabosm-icon.png"):
            # the rail's mark, for the page's tab: the WebUI's favicon.ico/svg are the
            # WebUI's own (vendored byte for byte), and the shell used to show them
            fs = os.path.join(ICONS, "collabosm" + path[-4:])
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
            status = {"unknown_recipe": 404, "busy": 409, "budget": 409, "placeholder": 409,
                      "stop_first": 409, "stale_ledger": 409, "colab_setup": 409,
                      "colab_native": 409, "attached": 409}.get(res.get("code"), 200)
            return self._json(status, res)
        if path.startswith("/control/colab/"):
            # the rail's Colab guide: each answers at once, the work runs behind it
            act = {"install": CONTROL.colab_install, "connect": CONTROL.colab_connect,
                   "cancel": CONTROL.colab_cancel, "check": CONTROL.colab_check,
                   "disconnect": CONTROL.colab_disconnect}.get(path.rsplit("/", 1)[-1])
            if act is None:
                return self._json(404, {"error": {"message": "no such route", "path": path}})
            res = act()
            return self._json(200 if res.get("ok") else 409, res)
        if path == "/control/cancel":
            return self._json(200, CONTROL.cancel())
        if path == "/control/stop":
            return self._json(200, CONTROL.stop("manual"))
        if path == "/control/quit":
            # The page's Quit -- the only close button a windowless start has. Answer
            # first, then stop serving (from another thread: shutdown() waits for the
            # serving loop). A running GPU is the page's to warn about; it is not stopped.
            self.log_message("quit from the page")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return self._json(200, {"ok": True, "code": "quitting"})
        if path == "/control/key":
            # only on a click, only same-origin JSON (the POST guard above): a foreign page
            # can neither send this nor read the answer
            res = CONTROL.reveal_key()
            return self._json(200 if res.get("ok") else 409, res)
        if path == "/control/couple":
            # WSL round trips take tens of seconds: answer now, let the rail watch
            threading.Thread(target=CONTROL.couple, args=("manual",), daemon=True).start()
            return self._json(200, {"ok": True, "code": "coupling"})
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


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # A browser that drops a kept-alive connection (a closed tab, a reload) is not
        # an error; a traceback for each one buried the log. Everything else still is.
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def main() -> int:
    global CONTROL, ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("FRONTEND_PORT", 3020)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", default=os.environ.get("FRONTEND_BACKEND"),
                    help="rehearsals only (--mock, --fake-provision): an OpenAI-compatible "
                         "engine to chat with, e.g. scripts/dev_stub.py at http://127.0.0.1:8099")
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--mock", action="store_true",
                    help="rehearsal with a ledger that is never written: no WSL, no Colab, no CU")
    ap.add_argument("--mock-speed", type=float, default=1.0,
                    help="with --mock, provisioning runs this much faster than the ~24 s rehearsal")
    ap.add_argument("--fake-provision", action="store_true",
                    help="real control plane, but frontend/fake_provision.py instead of Colab")
    ap.add_argument("--fake-vm", default=None, metavar="URL",
                    help="with --mock/--fake-provision: the rehearsal VM serves here (e.g. "
                         "scripts/dev_stub.py), so coupling, the machine block and link loss "
                         "can be rehearsed too")
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
    ap.add_argument("--no-browser", action="store_true",
                    default=os.environ.get("COLLABOSM_NO_BROWSER") == "1",
                    help="do not open the page in the browser (a real start opens it)")
    ap.add_argument("--browser", action="store_true",
                    help="open the page even in a rehearsal (--mock, --fake-provision)")
    ap.add_argument("--no-shortcut", action="store_true",
                    default=os.environ.get("COLLABOSM_NO_SHORTCUT") == "1",
                    help="do not put a desktop shortcut down on the first real start")
    ARGS = ap.parse_args()
    if os.name == "nt" and sys.stdout is None:
        # pythonw.exe -- the desktop shortcut: the server runs windowless (run_windowless)
        return run_windowless(ARGS)
    if ProvisionControl is None:
        print("!! cannot import frontend/control.py: %r" % _CONTROL_IMPORT_ERROR,
              file=sys.stderr)
        return 2
    if ARGS.mock:
        # fake_provision.py reads its pacing from the environment it inherits
        os.environ.setdefault("COLLABOSM_FAKE_SECONDS",
                              "%.1f" % (24.0 / max(0.1, ARGS.mock_speed)))
    if ARGS.fake_vm and (ARGS.mock or ARGS.fake_provision):
        os.environ["COLLABOSM_FAKE_VM_URL"] = ARGS.fake_vm
    CONTROL = ProvisionControl(os.path.dirname(HERE), session=ARGS.session,
                               distro=ARGS.wsl_distro, budget_cu=ARGS.budget_cu,
                               idle_stop_min=ARGS.idle_stop_min,
                               max_session_h=ARGS.max_session_h,
                               fake=ARGS.mock or ARGS.fake_provision,
                               state_dir=ARGS.state_dir, persist_ledger=not ARGS.mock,
                               external=ARGS.external_endpoint,
                               external_key=ARGS.external_key,
                               external_model=ARGS.external_model,
                               # rehearsal only: where the Colab guide starts (missing,
                               # no_python, signed_out, expired, connected)
                               fake_colab=os.environ.get("COLLABOSM_FAKE_COLAB"))
    mode = ("mock (nothing billed, nothing written)" if ARGS.mock
            else "rehearsal" if ARGS.fake_provision else "colab")
    print("[fe] control    %s (idle stop %d min, budget %.0f CU)"
          % (mode, ARGS.idle_stop_min, ARGS.budget_cu), flush=True)
    if ARGS.external_endpoint:
        print("[fe] external   %s (nothing billed, nothing auto-stopped)"
              % ARGS.external_endpoint, flush=True)
    if not os.path.isfile(os.path.join(UPSTREAM, "index.html")):
        print("!! upstream/index.html missing -- the vendored WebUI build is not here",
              file=sys.stderr)
        return 2
    url = page_url(ARGS)
    rehearsal = ARGS.mock or ARGS.fake_provision
    if os.name == "nt":
        try:
            # started from a terminal, its window is "collabosm" in the taskbar, not the
            # interpreter's path (the desktop shortcut's start has no window at all)
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW("collabosm")
        except Exception:                                 # noqa: BLE001 - cosmetic
            pass
    if app_answers(url):
        # A second start -- the desktop shortcut clicked again: the page is what was
        # wanted, and a second server on the port would be the wrong answer
        print("[fe] collabosm is already running at %s" % url, flush=True)
        if wants_browser(ARGS, rehearsal):
            webbrowser.open(url)
        return 0
    print("[fe] shell      %s   <- open this in your browser" % url, flush=True)
    print("[fe] webui      %s?embed=1" % url, flush=True)
    if ARGS.backend:
        print("[fe] backend    %s (rehearsal chat)" % ARGS.backend, flush=True)
    httpd = Server((ARGS.host, ARGS.port), Handler)
    # bound, so the page can be asked for now; it is answered once serve_forever runs
    if not ARGS.no_shortcut and not rehearsal and not ARGS.external_endpoint:
        import shortcut
        threading.Thread(target=shortcut.ensure_once, args=(CONTROL.state_dir,),
                         kwargs={"log": lambda line: print(line, flush=True)},
                         daemon=True).start()
    if wants_browser(ARGS, rehearsal):
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    httpd.serve_forever()                  # returns only for the page's Quit
    httpd.server_close()
    print("[fe] closed from the page (Quit)", flush=True)
    return 0


def run_windowless(args) -> int:
    """Windows, started by pythonw.exe -- what the desktop shortcut runs: no console,
    and so no window, which is the point. The server is started from here as python.exe
    with CREATE_NO_WINDOW. It keeps a console nobody sees, and the wsl.exe and
    PowerShell calls it makes share that console, so none of them flashes a window
    (under pythonw each would open its own). Its output goes to server.log in the state
    folder, as the macOS app's does; the previous start's is kept as server.log.1. It is
    closed from the page (Quit), or in Task Manager. A second start while it runs only
    opens the page."""
    url = page_url(args)
    if app_answers(url):
        if wants_browser(args, args.mock or args.fake_provision):
            webbrowser.open(url)
        return 0
    state = args.state_dir or os.path.join(os.path.expanduser("~"), ".collabosm")
    os.makedirs(state, exist_ok=True)
    log = os.path.join(state, "server.log")
    try:
        if os.path.exists(log):
            os.replace(log, log + ".1")
    except OSError:
        pass                                   # still held open somewhere: append to it
    exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
    if not os.path.exists(exe):
        exe = sys.executable
    with open(log, "a", encoding="utf-8") as out:
        subprocess.Popen([exe, os.path.abspath(__file__)] + sys.argv[1:],
                         cwd=os.path.dirname(HERE), stdin=subprocess.DEVNULL,
                         stdout=out, stderr=subprocess.STDOUT,
                         env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"),
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    return 0


def webui_ctx(server: dict, default: int) -> int:
    """The context the WebUI's meter shows: one conversation's ceiling. A cache larger
    than the model's position window (native, or YaRN's) holds more conversations, not
    a longer one -- the 27B recipe keeps two 400K conversations in an 819K cache."""
    n_ctx = int((server or {}).get("cache_max_tokens") or default)
    window = ((server or {}).get("context") or {}).get("max_positions")
    return min(n_ctx, int(window)) if window else n_ctx


def page_url(args) -> str:
    """The page's address: a server bound to every interface is still opened on loopback."""
    host = "127.0.0.1" if args.host in ("", "0.0.0.0", "::") else args.host
    return "http://%s:%d/" % (host, args.port)


def wants_browser(args, rehearsal: bool) -> bool:
    """A real start opens the page -- that is what starting collabosm, or clicking its
    shortcut, is for. A rehearsal does not unless --browser says so (tests start many),
    nor does --no-browser, nor a Linux session without a display."""
    if args.no_browser:
        return False
    if rehearsal and not args.browser:
        return False
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY")
                                                 or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return True


def app_answers(url: str) -> bool:
    """collabosm itself, already on this address: its /control/status answers, with the
    shape only this app gives it (another program on the port is not us)."""
    try:
        with urllib.request.urlopen(url + "control/status", timeout=1.5) as r:
            data = json.loads(r.read())
        return isinstance(data, dict) and "stage" in data and "colab" in data
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
