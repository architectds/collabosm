#!/usr/bin/env python3
"""The Colab connector's first step: is the Colab CLI here, is it signed in, as whom.

A new user has none of it, so the rail's Colab guide walks through it in order, one
button per stage:

    missing      Install     google-colab-cli, into ~/.collabosm/colab-cli (a venv of its own)
    signed_out   Connect     one Google consent in the browser; the token stays on this PC
    expired      Reconnect   revoked, expired, unreadable, or missing the Colab scope
    connected    the account the token belongs to

and the stages between them: checking, installing, install_failed, no_python (nothing
here can build the venv) and authorizing (the consent page is open).

Everything that talks to Google runs with the CLI's own interpreter, through
scripts/colab_auth.py -- this process stays stdlib-only -- and uses the CLI's own
OAuth client, scopes and token file. So a token made here is the one the CLI uses,
and a CLI that is already signed in is simply found connected.

Where the CLI lives
    wsl     ~/.local/share/uv/tools/google-colab-cli inside WSL, the setup this kit
            grew up on: looked at first, when that distro exists (asked without
            booting WSL), so a machine already set up keeps its sign-in.
    native  a Python on this machine that imports colab_cli: $COLLABOSM_COLAB_PY, ours
            (~/.collabosm/colab-cli), a `uv tool install`, or this interpreter --
            Windows without WSL, macOS, Linux. The `colab` command dies on Windows
            (its console imports termios); scripts/colab_cmd.py stands in for that.
Whichever it is, everything runs as Python with that interpreter: the sign-in,
the balance, the keep-alive and scripts/provision.py (up, down, fetch, sessions).

Rehearsal
    fake=<stage> plays the guide without touching anything: an install takes a few
    seconds, the consent approves itself, and the account is rehearsal@example.com
    (server.py --mock with COLLABOSM_FAKE_COLAB=missing|no_python|signed_out|expired).
    COLLABOSM_COLAB_WSL=0 skips WSL for real, to walk a fresh machine's path.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

# The version scripts/restore.py, colab_ccu.py and colab_keepalive.py are written
# against: they use the CLI's internals, which a newer release may move.
PIN = "google-colab-cli==0.6.0"
WSL_PY = "$HOME/.local/share/uv/tools/google-colab-cli/bin/python"
FAKE_ACCOUNT = "rehearsal@example.com"
LOGIN_TIMEOUT_S = 300
LOG_KEEP = 14
NO_WINDOW = {"creationflags": 0x08000000} if os.name == "nt" else {}

# colab_auth.py's verdict on the token file -> the guide's stage
TOKEN_STAGE = {"none": "signed_out", "ok": "connected", "unverified": "connected",
               "expired": "expired", "invalid": "expired", "scopes": "expired"}
FAKE_STAGES = ("missing", "no_python", "signed_out", "expired", "connected")


def _wsl_path(win_path: str) -> str:
    """E:\\models\\collabosm -> /mnt/e/models/collabosm (WSL sees the same files)."""
    p = os.path.abspath(win_path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    return "/mnt/%s/%s" % (m.group(1).lower(), m.group(2)) if m else p


def _env() -> dict:
    # helper output is parsed as UTF-8; wsl.exe otherwise prints UTF-16 for itself
    return dict(os.environ, PYTHONIOENCODING="utf-8", WSL_UTF8="1", PIP_NO_INPUT="1",
                PIP_DISABLE_PIP_VERSION_CHECK="1")


def run(argv: list, timeout: float = 120.0) -> str:
    """argv -> its stdout and stderr as text; the reason instead when it cannot run."""
    try:
        res = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout, env=_env(),
                             stdin=subprocess.DEVNULL, **NO_WINDOW)
        return (res.stdout or "") + (res.stderr or "")
    except Exception as exc:
        return "!! %r" % exc


class ColabSetup:
    """probe / install / connect / cancel / disconnect -- and snapshot() for the rail."""

    def __init__(self, root: str, *, distro: str = "Ubuntu", home: str | None = None,
                 fake: str | None = None, runner=None, log=None, on_connected=None,
                 use_wsl: bool | None = None):
        self.root = os.path.abspath(root)
        self.distro = distro
        self.home = home or os.path.join(os.path.expanduser("~"), ".collabosm")
        self.venv = os.path.join(self.home, "colab-cli")
        self.fake = (fake if fake in FAKE_STAGES else "connected") if fake else None
        self.runner = runner or run
        self._log = log or (lambda line: None)
        self.on_connected = on_connected
        # COLLABOSM_COLAB_WSL=0 skips WSL: a dev machine that has it can walk the guide
        # the way a fresh one does (native install, native sign-in)
        self.use_wsl = ((os.name == "nt" and os.environ.get("COLLABOSM_COLAB_WSL", "1") != "0")
                        if use_wsl is None else use_wsl)
        self.lock = threading.RLock()
        self.settled_event = threading.Event()
        self.where = None             # "wsl" | "native": where the CLI was found
        self.py = None                # its interpreter (for wsl, a path inside WSL)
        self.proc = None              # the sign-in in flight
        self.cancelled = False
        self.before_auth = "signed_out"
        self.state = {"stage": "checking", "where": None, "version": None, "account": None,
                      "verified": None, "auth_url": None, "error": None, "log": [],
                      "distro": distro, "checked_at": None}
        if self.fake:
            self._fake_arrive(self.fake)

    # ---- what the rail reads -------------------------------------------- #

    def snapshot(self) -> dict:
        with self.lock:
            snap = dict(self.state)
            snap["log"] = list(self.state["log"])
            return snap

    def stage(self) -> str:
        with self.lock:
            return self.state["stage"]

    def ready(self) -> bool:
        return self.stage() == "connected"

    def settled(self, timeout: float = 15.0) -> dict:
        """The snapshot once the first look is over (or `timeout` passed): a Start
        pressed in the first seconds is judged on an answer, not on "checking"."""
        self.settled_event.wait(timeout)
        return self.snapshot()

    def _set(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)
            if self.state["stage"] != "checking":
                self.settled_event.set()

    def _tail(self, line: str) -> None:
        with self.lock:
            self.state["log"] = (self.state["log"] + [line.rstrip()])[-LOG_KEEP:]

    # ---- running things where the CLI is ---------------------------------- #

    def _wsl(self, cmd: str, timeout: float) -> str:
        return self.runner(["wsl.exe", "-d", self.distro, "--", "bash", "-lc", cmd], timeout)

    def script_argv(self, name: str, args=()) -> list:
        """argv for scripts/<name> under the CLI's interpreter: natively, or inside WSL
        (also the answer before the first probe -- the setup provisioning uses)."""
        path = os.path.join(self.root, "scripts", name)
        with self.lock:
            where, py = self.where, self.py
        if where == "native" and py:
            return [py, path] + list(args)
        return ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                "%s %s %s" % (WSL_PY, shlex.quote(_wsl_path(path)),
                              " ".join(shlex.quote(str(a)) for a in args))]

    def run_script(self, name: str, args=(), timeout: float = 90.0) -> str:
        return self.runner(self.script_argv(name, args), timeout)

    def _helper_argv(self, args) -> list:
        """scripts/colab_auth.py, unbuffered: AUTH_URL has to arrive while it waits."""
        path = os.path.join(self.root, "scripts", "colab_auth.py")
        with self.lock:
            where, py = self.where, self.py
        if where == "native" and py:
            return [py, "-u", path] + list(args)
        return ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                "%s -u %s %s" % (WSL_PY, shlex.quote(_wsl_path(path)),
                                 " ".join(shlex.quote(str(a)) for a in args))]

    # ---- finding the CLI, and whom it is signed in as ---------------------- #

    def _native_candidates(self) -> list:
        sub, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        uv_tools = os.environ.get("UV_TOOL_DIR") or (
            os.path.join(os.environ.get("APPDATA") or "", "uv", "tools") if os.name == "nt"
            else os.path.expanduser("~/.local/share/uv/tools"))
        out, seen = [], set()
        for p in (os.environ.get("COLLABOSM_COLAB_PY"), os.path.join(self.venv, sub, exe),
                  os.path.join(uv_tools, "google-colab-cli", sub, exe), sys.executable):
            key = os.path.normcase(os.path.abspath(p)) if p else None
            if p and key not in seen and os.path.exists(p):
                seen.add(key)
                out.append(p)
        return out

    def _wsl_has_distro(self) -> bool:
        """Is there a WSL distro of this name to look in? Asked without booting WSL --
        `--list` reads what is installed, while `-d <distro> -- ...` starts its VM
        (~40 s cold) -- so a machine without WSL, or without this distro, costs
        nothing here: not installed, no distros, another name all read as no."""
        if not self.use_wsl or not shutil.which("wsl.exe"):
            return False
        out = self.runner(["wsl.exe", "--list", "--quiet"], 30)
        names = {line.replace("\x00", "").strip() for line in out.splitlines()}
        names.discard("")
        if not names:
            return True                        # nothing to read: the probe itself decides
        if self.distro not in names:
            self._log("[colab] no WSL distro %r here (%s) -- looking for the Colab CLI natively"
                      % (self.distro, ", ".join(sorted(names))[:120]))
            return False
        return True

    def _locate(self):
        """(where, interpreter) of the Colab CLI, or (None, None)."""
        if self._wsl_has_distro():
            out = self._wsl("test -x %s && echo COLAB_OK || echo COLAB_MISSING" % WSL_PY, 120)
            if "COLAB_OK" in out:
                return "wsl", WSL_PY
        for py in self._native_candidates():
            out = self.runner([py, "-c", "import colab_cli.auth; print('COLAB_IMPORT_OK')"], 60)
            if "COLAB_IMPORT_OK" in out:
                return "native", py
        return None, None

    def probe(self) -> dict:
        """Find the CLI and ask it who is signed in. Blocking: call it from a thread."""
        with self.lock:
            if self.fake or self.state["stage"] in ("installing", "authorizing"):
                return self.snapshot()           # each of those ends in a probe of its own
        where, py = self._locate()
        now = time.time()
        if where is None:
            with self.lock:
                self.where = self.py = None
            self._set(stage="missing", where=None, version=None, account=None, verified=None,
                      auth_url=None, checked_at=now, error=None)
            return self.snapshot()
        with self.lock:
            self.where, self.py = where, py
        out = self.runner(self._helper_argv(["status"]), 90)
        m = re.search(r"^COLAB_AUTH (\{.*\})\s*$", out, re.M)
        info = {}
        if m:
            try:
                info = json.loads(m.group(1))
            except ValueError:
                info = {}
        token = info.get("token")
        why = (info.get("error") or (out.strip().splitlines() or ["no answer"])[-1])[:160]
        if token == "error":
            # found, but it will not import (a broken venv): as good as missing
            self._set(stage="missing", where=where, version=info.get("version"), account=None,
                      verified=None, auth_url=None, checked_at=now,
                      error={"k": "broken", "a": {"why": why}})
            return self.snapshot()
        # No answer at all (the helper could not run) is "connected, unverified", not
        # a wall: the CLI is there, and a sign-in that really is missing still shows
        # up the moment anything asks Colab (up.sh exits 5).
        stage = TOKEN_STAGE.get(token, "connected")
        err = None
        if stage == "expired":
            err = {"k": "expired", "a": {"why": token}}
        elif token != "ok" and stage == "connected":
            err = {"k": "unverified", "a": {"why": why}}
        self._set(stage=stage, where=where, version=info.get("version"),
                  account=info.get("account"),
                  verified=(token == "ok") if stage == "connected" else None,
                  auth_url=None, checked_at=now, error=err)
        return self.snapshot()

    def check(self) -> dict:
        with self.lock:
            stage = self.state["stage"]
            if stage in ("installing", "authorizing"):
                return {"ok": False, "code": "colab_busy", "stage": stage}
        threading.Thread(target=self._after_change, daemon=True).start()
        return {"ok": True, "code": "colab_checking"}

    def _after_change(self) -> None:
        """Probe again after an install, a sign-in or a sign-out, and tell the control
        plane when that made the CLI usable."""
        with self.lock:
            was = self.state["stage"]
            if not self.fake:
                # out of installing/authorizing first: probe() leaves those alone
                self.state["stage"] = "checking"
        snap = self.probe()
        if snap["stage"] == "connected" and was != "connected" and self.on_connected:
            try:
                self.on_connected()
            except Exception as exc:          # noqa: BLE001 - the guide must not die of it
                self._log("!! [colab] after connecting: %r" % exc)

    # ---- install ------------------------------------------------------------- #

    def install(self) -> dict:
        with self.lock:
            stage = self.state["stage"]
            if stage not in ("missing", "install_failed", "no_python"):
                return {"ok": False, "code": "colab_busy", "stage": stage}
            self.state.update(stage="installing", error=None, log=[])
        threading.Thread(target=self._install, daemon=True).start()
        return {"ok": True, "code": "colab_installing"}

    def _base_python(self):
        """What can build the venv: a Python >= 3.12 (google-colab-cli needs it), or
        uv, which fetches one. None when neither is on this machine."""
        if sys.version_info >= (3, 12) and not getattr(sys, "frozen", False):
            return "python", sys.executable
        if os.name == "nt" and shutil.which("py"):
            for v in ("3.13", "3.12"):
                out = self.runner(["py", "-" + v, "-c", "import sys; print('PY=' + sys.executable)"], 30)
                m = re.search(r"^PY=(.+)$", out, re.M)
                if m and os.path.exists(m.group(1).strip()):
                    return "python", m.group(1).strip()
        for name in ("python3.13", "python3.12"):
            if shutil.which(name):
                return "python", shutil.which(name)
        if shutil.which("uv"):
            return "uv", shutil.which("uv")
        return None

    def _stream(self, argv: list) -> int:
        """Run one install step, its output into the guide's log as it comes."""
        self._tail("$ " + " ".join(os.path.basename(a) if i == 0 else a for i, a in enumerate(argv)))
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                                    errors="replace", env=_env(), **NO_WINDOW)
        except Exception as exc:
            self._tail("!! %r" % exc)
            return 127
        for line in proc.stdout:
            if line.strip():
                self._tail(line)
        return proc.wait()

    def _install(self) -> None:
        if self.fake:
            return self._fake_install()
        base = self._base_python()
        if base is None:
            self._set(stage="no_python", error={"k": "no_python",
                                                "a": {"have": "%d.%d" % sys.version_info[:2]}})
            self._log("!! [colab] no Python 3.12+ and no uv here: the Colab CLI cannot be installed")
            return
        kind, exe = base
        sub, name = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        py = os.path.join(self.venv, sub, name)
        if kind == "uv":
            steps = [("venv", [exe, "venv", "--python", "3.12", self.venv]),
                     ("pip", [exe, "pip", "install", "--python", py, PIN])]
        else:
            steps = [("venv", [exe, "-m", "venv", self.venv]),
                     ("pip", [py, "-m", "pip", "install", PIN])]
        self._log("[colab] installing %s into %s" % (PIN, self.venv))
        for step, argv in steps:
            rc = self._stream(argv)
            if rc != 0:
                self._set(stage="install_failed", error={"k": "install", "a": {"step": step, "rc": rc}})
                self._log("!! [colab] install failed at %s (exit %s)" % (step, rc))
                return
        self._log("[colab] installed %s" % PIN)
        self._after_change()

    # ---- sign in, sign out ---------------------------------------------------- #

    def connect(self) -> dict:
        with self.lock:
            stage = self.state["stage"]
            if stage == "authorizing":
                return {"ok": True, "code": "colab_authorizing"}
            if stage not in ("signed_out", "expired"):
                return {"ok": False, "code": "colab_busy", "stage": stage}
            self.cancelled = False
            self.before_auth = stage
            self.state.update(stage="authorizing", auth_url=None, error=None)
        threading.Thread(target=self._login, daemon=True).start()
        return {"ok": True, "code": "colab_authorizing"}

    def _open(self, url: str) -> None:
        """The consent page, in the user's own browser: this process runs where it is
        (from WSL the helper could not open it). The rail also shows the link."""
        try:
            webbrowser.open(url, new=2)
        except Exception as exc:              # noqa: BLE001
            self._log("!! [colab] could not open the browser (%r) -- use the link in the rail" % exc)

    def _login(self) -> None:
        if self.fake:
            return self._fake_login()
        argv = self._helper_argv(["login", "--timeout", str(LOGIN_TIMEOUT_S)])
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                                    errors="replace", env=_env(), **NO_WINDOW)
        except Exception as exc:
            self._set(stage=self.before_auth, error={"k": "auth", "a": {"why": repr(exc)[:160]}})
            return
        with self.lock:
            self.proc = proc
            cancelled = self.cancelled
        if cancelled:                         # a Cancel that came before the process did
            self._stop_helper(proc)
        ok = why = None
        for raw in proc.stdout:
            line = raw.strip()
            if line.startswith("AUTH_URL "):
                url = line[len("AUTH_URL "):].strip()
                with self.lock:
                    if self.cancelled:        # never open a consent page nobody wants
                        continue
                self._set(auth_url=url)
                self._log("[colab] waiting for Google sign-in in the browser")
                self._open(url)
            elif line.startswith("AUTH_OK"):
                ok = line[len("AUTH_OK"):].strip() or "?"
            elif line.startswith("AUTH_ERROR"):
                why = line[len("AUTH_ERROR"):].strip()
        proc.wait()
        with self.lock:
            self.proc = None
            cancelled = self.cancelled
        if cancelled:
            self._set(stage=self.before_auth, auth_url=None, error=None)
            return
        if ok:
            self._log("[colab] signed in as %s" % ok)
            self._set(auth_url=None)
            self._after_change()
            return
        why = why or "exit %s" % proc.returncode
        self._log("!! [colab] sign-in did not complete: %s" % why)
        self._set(stage=self.before_auth, auth_url=None, error={"k": "auth", "a": {"why": why}})

    def cancel(self) -> dict:
        with self.lock:
            if self.state["stage"] != "authorizing":
                return {"ok": True, "code": "colab_cancelled"}
            self.cancelled = True
            proc = self.proc
        if proc is not None:                  # else _login sees `cancelled` when it starts one
            self._stop_helper(proc)
        return {"ok": True, "code": "colab_cancelled"}

    def _stop_helper(self, proc) -> None:
        """End a sign-in helper. Terminating wsl.exe does not reach the Linux process,
        which would keep its loopback port (and a consent page) live until its own
        timeout, so inside WSL it is ended by name as well."""
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        if self.where == "wsl" and not self.fake:
            threading.Thread(target=self._wsl, args=("pkill -f 'colab_auth.py login' || true", 60),
                             daemon=True).start()

    def disconnect(self) -> dict:
        with self.lock:
            stage = self.state["stage"]
            if stage not in ("connected", "expired"):
                return {"ok": False, "code": "colab_busy", "stage": stage}
            self.state["stage"] = "checking"
        threading.Thread(target=self._logout, daemon=True).start()
        return {"ok": True, "code": "colab_disconnected"}

    def _logout(self) -> None:
        if self.fake:
            self._set(stage="signed_out", account=None, verified=None, error=None)
            return
        out = self.runner(self._helper_argv(["logout"]), 60)
        if "LOGOUT ok" in out:
            self._log("[colab] signed out%s" % (" (Google did not confirm the revoke)"
                                                if "revoke_failed" in out else ""))
        else:
            self._log("!! [colab] sign-out: %s" % ((out.strip().splitlines() or ["no answer"])[-1][:160]))
        self._after_change()

    # ---- rehearsal -------------------------------------------------------------- #

    def _fake_arrive(self, stage: str) -> None:
        found = {"where": "native", "version": "0.6.0", "checked_at": time.time()}
        if stage == "missing":
            self._set(stage="missing", checked_at=time.time())
        elif stage == "no_python":
            self._set(stage="no_python", checked_at=time.time(),
                      error={"k": "no_python", "a": {"have": "3.11"}})
        elif stage == "signed_out":
            self._set(stage="signed_out", **found)
        elif stage == "expired":
            self._set(stage="expired", error={"k": "expired", "a": {"why": "expired"}}, **found)
        else:
            self._set(stage="connected", account=FAKE_ACCOUNT, verified=True, **found)
        with self.lock:
            self.where = "native" if stage not in ("missing", "no_python") else None

    def _fake_install(self) -> None:
        if self.fake == "no_python":
            time.sleep(0.6)
            self._set(stage="no_python", error={"k": "no_python", "a": {"have": "3.11"}})
            return
        for line in ("$ python -m venv ~/.collabosm/colab-cli", "$ python -m pip install " + PIN,
                     "Collecting google-colab-cli==0.6.0",
                     "Installing collected packages: google-colab-cli",
                     "Successfully installed google-colab-cli-0.6.0"):
            time.sleep(0.6)
            self._tail(line)
        with self.lock:
            self.where = "native"
        self._set(stage="signed_out", where="native", version="0.6.0", checked_at=time.time())

    def _fake_login(self) -> None:
        for _ in range(25):                    # the consent "takes" 2.5 s
            time.sleep(0.1)
            with self.lock:
                if self.cancelled:
                    self.state.update(stage=self.before_auth, error=None, auth_url=None)
                    return
        self._set(stage="connected", account=FAKE_ACCOUNT, verified=True, error=None,
                  auth_url=None, checked_at=time.time())
        if self.on_connected:
            try:
                self.on_connected()
            except Exception as exc:           # noqa: BLE001
                self._log("!! [colab] after connecting: %r" % exc)
