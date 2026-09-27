#!/usr/bin/env python3
"""collabosm on this machine, the way CI checks it on Windows without WSL, macOS and Linux.

No GPU, no Google account, nothing billed:

  1. every Python file here compiles;
  2. the app starts for real (not a rehearsal) and serves its page, the chat page and its
     icon, and its Colab guide finds no CLI on a fresh machine;
  3. asked to, it installs google-colab-cli into its own folder -- a real pip install --
     and then finds it, signed out; the kit's `colab` wrapper (scripts/colab_cmd.py, which
     stands in for the termios Windows lacks) runs under that interpreter;
  4. a rehearsed GPU start runs its course: the confirmation with the cost, the
     provisioning stages, ready, stop, and the session in the ledger;
  5. chat through the app's /v1 reaches a stub engine (scripts/dev_stub.py: the shipping
     API server with a scripted model), in both dialects, streaming and not;
  6. the desktop shortcut is written for this system, into a scratch folder;
  7. the page's Quit closes the app.

It runs in a throwaway home folder (HOME, USERPROFILE and APPDATA), so ~/.collabosm, the
CLI's venv and any sign-in are its own, never the user's. COLLABOSM_COLAB_WSL=0 makes a
Windows machine that has WSL walk the no-WSL path too; --wsl-as-is leaves that to the
machine (CI's Windows runners have no WSL, so they take the same path for real).

    python scripts/check_platform.py                # 1-5 min, most of it pip
    python scripts/check_platform.py --no-install   # without step 3's install
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import py_compile
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RESULTS = []
PROCS = []          # (process, its log file)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("%-68s %s%s" % (name, "PASS" if ok else "FAIL",
                          "" if ok else "  " + str(detail)[:400]), flush=True)
    return bool(ok)


def section(title):
    print("\n==== " + title, flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for(fn, timeout, every=0.5):
    """fn() until it answers something truthy; its last answer (None on a timeout)."""
    end = time.time() + timeout
    while True:
        try:
            got = fn()
        except (OSError, ValueError, http.client.HTTPException):
            got = None
        if got or time.time() > end:
            return got
        time.sleep(every)


def call(port, method, path, body=None, headers=None, timeout=60, stream=False):
    """(status, text) -- or (status, response) with stream=True."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    h = {"Content-Type": "application/json"} if body is not None else {}
    h.update(headers or {})
    c.request(method, path, body=None if body is None else json.dumps(body).encode(), headers=h)
    r = c.getresponse()
    if stream:
        return r.status, r
    text = r.read().decode("utf-8", "replace")
    c.close()
    return r.status, text


def status(port):
    code, text = call(port, "GET", "/control/status", timeout=10)
    return json.loads(text) if code == 200 else None


def spawn(args, env, name, tmp):
    log = os.path.join(tmp, name + ".log")
    fh = open(log, "w", encoding="utf-8")
    p = subprocess.Popen([PY] + args, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    PROCS.append((p, log, name))
    return p


def up(port, timeout=60):
    return wait_for(lambda: status(port), timeout)


def quit_app(port, proc, name):
    try:
        code, text = call(port, "POST", "/control/quit", {}, timeout=15)
        answered = code == 200 and json.loads(text).get("code") == "quitting"
    except (OSError, ValueError, http.client.HTTPException) as exc:
        code, text, answered = None, repr(exc), False
    try:
        rc = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        rc = None
    return check("%s: Quit closes it (exit %s)" % (name, rc), answered and rc == 0,
                 (code, text[:200]))


def home_env(home, wsl_as_is):
    e = dict(os.environ)
    e.update(HOME=home, USERPROFILE=home, PYTHONIOENCODING="utf-8",
             COLLABOSM_NO_BROWSER="1", COLLABOSM_NO_SHORTCUT="1")
    if os.name == "nt":
        e.update(APPDATA=os.path.join(home, "AppData", "Roaming"),
                 LOCALAPPDATA=os.path.join(home, "AppData", "Local"))
        os.makedirs(e["APPDATA"], exist_ok=True)
        os.makedirs(e["LOCALAPPDATA"], exist_ok=True)
    if not wsl_as_is:
        e["COLLABOSM_COLAB_WSL"] = "0"
    for k in ("COLLABOSM_COLAB_PY", "UV_TOOL_DIR", "COLLABOSM_EXTERNAL", "FRONTEND_BACKEND",
              "COLLABOSM_FAKE_COLAB", "COLLABOSM_FAKE_VM_URL"):
        e.pop(k, None)
    return e


# --------------------------------------------------------------------------------- #

def step_compile():
    section("1. every Python file compiles")
    bad = []
    n = 0
    for folder in ("frontend", "scripts"):
        for dirpath, dirs, files in os.walk(os.path.join(ROOT, folder)):
            dirs[:] = [d for d in dirs if d not in ("upstream", "__pycache__", "icon")]
            for f in files:
                if f.endswith(".py"):
                    n += 1
                    try:
                        py_compile.compile(os.path.join(dirpath, f), doraise=True,
                                           cfile=os.path.join(tempfile.gettempdir(), "cp.pyc"))
                    except py_compile.PyCompileError as exc:
                        bad.append("%s: %s" % (f, exc.msg.strip().splitlines()[-1]))
    check("%d files" % n, not bad and n > 20, bad)


def step_app(env, tmp, install):
    section("2. the app starts for real, and serves what the browser asks for")
    port = free_port()
    proc = spawn(["frontend/server.py", "--port", str(port)], env, "app", tmp)
    st = up(port)
    if not check("control plane answers on 127.0.0.1:%d" % port, st, "no answer in 60 s"):
        return
    check("  idle, with the recipes", st.get("stage") == "idle"
          and any(r.get("id") == "a100-40g/qwen38-27b" for r in st.get("recipes") or []),
          st.get("stage"))
    code, page = call(port, "GET", "/")
    check("GET /: the shell", code == 200 and "collabosm" in page, code)
    code, webui = call(port, "GET", "/?embed=1")
    check("GET /?embed=1: the chat page (llama.cpp WebUI)", code == 200 and "<html" in webui.lower(), code)
    code, _ = call(port, "GET", "/collabosm-icon.svg")
    check("GET /collabosm-icon.svg", code == 200, code)

    colab = wait_for(lambda: (status(port) or {}).get("colab", {}).get("stage") not in
                     (None, "checking") and status(port)["colab"], 180)
    stage = (colab or {}).get("stage")
    check("Colab guide settles on a fresh machine: %s" % stage,
          stage in ("missing", "signed_out"), colab)

    if install and stage == "missing":
        section("3. the Colab CLI installs into the app's folder, and loads")
        code, text = call(port, "POST", "/control/colab/install", {})
        check("install accepted", code == 200 and json.loads(text).get("ok"), text[:200])
        t0 = time.time()
        colab = wait_for(lambda: (status(port) or {}).get("colab", {}).get("stage") in
                         ("signed_out", "install_failed", "no_python", "missing")
                         and status(port)["colab"], 900, every=2)
        stage = (colab or {}).get("stage")
        check("installed in %.0f s, found signed out" % (time.time() - t0),
              stage == "signed_out", (colab or {}).get("error") or (colab or {}).get("log", [])[-5:])
        check("  natively (no WSL), version %s" % (colab or {}).get("version"),
              (colab or {}).get("where") == "native" and (colab or {}).get("version") == "0.7.4",
              colab)
        sub, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        venv_py = os.path.join(env["USERPROFILE" if os.name == "nt" else "HOME"],
                               ".collabosm", "colab-cli", sub, exe)
        if os.path.exists(venv_py):
            p = subprocess.run([venv_py, os.path.join(ROOT, "scripts", "colab_cmd.py"), "--help"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env=env, timeout=120)
            out = p.stdout + p.stderr
            check("`colab --help` through scripts/colab_cmd.py", p.returncode == 0
                  and "sessions" in out, out[-400:])
        else:
            check("the CLI's own interpreter is where the app put it", False, venv_py)
    elif install:
        print("(the CLI was already importable here -- nothing to install)", flush=True)

    section("7a. Quit")
    quit_app(port, proc, "the app")


def step_rehearsal(env, tmp):
    section("4. a rehearsed GPU start: confirm, stages, ready, stop, ledger")
    port = free_port()
    state = os.path.join(tmp, "state-rehearsal")
    renv = dict(env, COLLABOSM_FAKE_SECONDS="2")
    proc = spawn(["frontend/server.py", "--fake-provision", "--port", str(port),
                  "--state-dir", state], renv, "rehearsal", tmp)
    if not check("rehearsal app answers", up(port), "no answer in 60 s"):
        return
    recipe = "a100-40g/qwen38-27b"
    code, text = call(port, "POST", "/control/select", {"recipe": recipe})
    ans = json.loads(text) if code == 200 else {}
    warn = ans.get("warning") or {}
    check("select: confirmation first, with the cost (%s CU/h)" % warn.get("cu_per_hour"),
          ans.get("code") == "confirm_required" and warn.get("cu_per_hour"), text[:300])
    code, text = call(port, "POST", "/control/select", {"recipe": recipe, "confirm": True})
    check("confirm: it starts", code == 200, text[:300])
    seen = set()

    def ready():
        s = status(port) or {}
        seen.add(s.get("stage"))
        return s.get("stage") in ("ready", "failed") and s

    st = wait_for(ready, 120)
    check("reached ready through %s" % ", ".join(sorted(x for x in seen if x)),
          (st or {}).get("stage") == "ready", (st or {}).get("note"))
    code, text = call(port, "POST", "/control/stop", {})
    st = wait_for(lambda: (status(port) or {}).get("stage") in ("stopped", "idle") and status(port), 60)
    check("stop: stopped", (st or {}).get("stage") in ("stopped", "idle"), (st or {}).get("stage"))
    ledger = os.path.join(state, "ledger.json")
    entries, data = [], {}
    if os.path.exists(ledger):
        with open(ledger, encoding="utf-8") as fh:
            data = json.load(fh)
        entries = data.get("entries") or []
    check("the session is in the ledger, closed", entries and not data.get("open"),
          entries[-1:] if entries else ledger)
    quit_app(port, proc, "rehearsal app")


def step_chat(env, tmp):
    section("5. chat through the app's /v1 to a stub engine, both dialects")
    sport, port = free_port(), free_port()
    stub = spawn(["scripts/dev_stub.py", "--port", str(sport), "--delay", "0"], env, "stub", tmp)
    ok = wait_for(lambda: call(sport, "GET", "/health", timeout=5)[0] == 200, 60)
    if not check("stub engine (the shipping API server) answers", ok):
        return
    proc = spawn(["frontend/server.py", "--fake-provision", "--fake-vm",
                  "http://127.0.0.1:%d" % sport, "--port", str(port),
                  "--state-dir", os.path.join(tmp, "state-chat")], env, "chat-app", tmp)
    st = wait_for(lambda: (status(port) or {}).get("stage") == "ready" and status(port), 90)
    if check("the app finds the stub box and is ready", st, (status(port) or {}).get("note")):
        key = {"Authorization": "Bearer sk-local"}
        code, text = call(port, "GET", "/v1/models", headers=key)
        check("GET /v1/models", code == 200 and json.loads(text).get("data"), text[:200])
        msg = {"model": "any", "messages": [{"role": "user", "content": "Say hi."}],
               "max_tokens": 32}
        code, text = call(port, "POST", "/v1/chat/completions", msg, headers=key)
        answer = json.loads(text)["choices"][0]["message"].get("content") if code == 200 else None
        check("chat/completions: an answer", bool(answer), text[:300])
        code, r = call(port, "POST", "/v1/chat/completions", dict(msg, stream=True),
                       headers=key, stream=True)
        body = r.read().decode("utf-8", "replace")
        check("chat/completions, streaming: deltas and [DONE]",
              code == 200 and body.count("data: ") > 2 and "[DONE]" in body, body[-300:])
        code, r = call(port, "POST", "/v1/responses",
                       {"model": "any", "input": "Say hi.", "stream": True}, headers=key, stream=True)
        body = r.read().decode("utf-8", "replace")
        check("responses, streaming: deltas and response.completed",
              code == 200 and "response.output_text.delta" in body
              and "response.completed" in body, body[-300:])
    quit_app(port, proc, "chat app")
    stub.terminate()


def step_shortcut(env, tmp):
    section("6. the desktop shortcut, for this system")
    desk = os.path.join(tmp, "Desktop")
    os.makedirs(desk, exist_ok=True)
    code = ("import sys; sys.path.insert(0, 'frontend'); import shortcut; "
            "print('SHORTCUT', shortcut.create(sys.argv[1]))")
    p = subprocess.run([PY, "-c", code, desk], cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    path = next((line.split(" ", 1)[1] for line in p.stdout.splitlines()
                 if line.startswith("SHORTCUT ")), None)
    if not check("written: %s" % (os.path.basename(path) if path else None),
                 p.returncode == 0 and path and os.path.exists(path), (p.stdout + p.stderr)[-400:]):
        return
    if os.name == "nt":
        ps = "(New-Object -ComObject WScript.Shell).CreateShortcut($env:CO_LNK).TargetPath"
        t = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           env=dict(env, CO_LNK=path), capture_output=True, text=True,
                           timeout=60).stdout.strip()
        check("  it runs pythonw.exe: no window", t.lower().endswith("pythonw.exe"), t)
    elif sys.platform == "darwin":
        run = os.path.join(path, "Contents", "MacOS", "collabosm")
        check("  an .app whose program is executable", os.access(run, os.X_OK), run)
    else:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        check("  a .desktop entry that runs server.py", "Exec=" in text and "server.py" in text,
              text[:300])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-install", action="store_true", help="skip the Colab CLI install")
    ap.add_argument("--wsl-as-is", action="store_true",
                    help="let a Windows machine's WSL be found (CI's runners have none)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch folder")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="collabosm-check-")
    home = os.path.join(tmp, "home")
    os.makedirs(home)
    env = home_env(home, a.wsl_as_is)
    print("collabosm platform check: %s, Python %s, scratch %s"
          % (sys.platform, sys.version.split()[0], tmp), flush=True)
    try:
        step_compile()
        step_app(env, tmp, install=not a.no_install)
        step_rehearsal(env, tmp)
        step_chat(env, tmp)
        step_shortcut(env, tmp)
    finally:
        for p, _log, _name in PROCS:
            if p.poll() is None:
                p.kill()
        failed = [n for n, ok in RESULTS if not ok]
        if failed:
            for _p, log, name in PROCS:
                try:
                    with open(log, encoding="utf-8", errors="replace") as fh:
                        tail = fh.read()[-3000:]
                except OSError:
                    tail = "(no log)"
                print("\n---- %s log (tail) ----\n%s" % (name, tail), flush=True)
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(failed),
                                       (": " + "; ".join(failed)) if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
