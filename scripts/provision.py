#!/usr/bin/env python3
"""collabosm up / down / fetch / sessions: bring a recipe up on Colab, stop it, read the VM.

One implementation for every machine -- Windows with WSL or without, macOS, Linux --
in Python, run with the Colab CLI's own interpreter. It drives the Colab CLI step by
step (`colab upload`, `colab exec`, `colab stop`, ...) through scripts/colab_cmd.py,
the CLI itself minus its Unix-only console: so it runs where there is no bash, no WSL,
or -- macOS -- no GNU `timeout`. scripts/up.sh, down.sh and vm_fetch.sh are one-line
shims onto this, so the commands in the docs keep working.

    python scripts/provision.py up [--recipe ID] [--session NAME]
    python scripts/provision.py down [--session NAME] [--endpoint EP]
    python scripts/provision.py fetch <session> <remote path> [timeout]
    python scripts/provision.py sessions

up -- the recipe decides everything (scripts/recipe.py): the card, the model and how
it loads; a variable the caller sets wins for that run (SESSION=mybox CACHE_SIZE=...).
The expensive steps come last:
  1. restore/verify the box    (a box below the recipe's VRAM is rejected in ~1 min)
  2. upload the kit and the launch environment
  3. bootstrap on the VM: the runtime, the weights from Hugging Face; serve.sh after it
  4. wait until the API is healthy AND the tunnel URL is published
Exits 0 READY | 1 upload/bootstrap failed | 2 a recipe or argument that cannot run |
3-6 from restore.py | 7 timed out | 8 serve.sh failed | 9 healthy on the VM, no URL.

down -- stops the session: on a metered plan, the most important command here. Given
the endpoint, it first puts back a session record the CLI dropped: `colab stop` cannot
stop what it has no record of, and would "work" while the VM kept billing.

The frontend (frontend/control.py) reads these log lines to draw its stages: keep them.
COLAB names another Colab CLI executable, COLAB_PY another interpreter for restore.py
and the keep-alive -- the test harness replaces both.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import recipe as registry  # noqa: E402

# what serve.sh launches on the VM, and what it reads its state with; api_server.py
# above all -- the kit once shipped without it, and a fresh clone waited 50 paid
# minutes for a /health that nothing could answer
KIT = ("api_server.py", "bootstrap.sh", "serve.sh", "strata.sh", "status.py", "probe_gpu.py")

# Runs on the VM. A re-attached box still has the last run's STATUS and endpoint.json:
# they go first, so nothing can take the old service for the new one. serve.sh is
# chained after bootstrap.sh, so it runs only if bootstrap succeeded -- for a long time
# nothing started it at all. `exec` matters: serve.sh stops older copies of itself by
# matching "bash /content/serve.sh", and this launcher's own command line would match.
START_BOOTSTRAP = '''import subprocess
print(subprocess.run("rm -f /content/STATUS /content/endpoint.json /content/url.txt; "
                     "nohup bash -lc 'source /content/collabosm_env.sh && bash /content/bootstrap.sh "
                     "&& exec bash /content/serve.sh' > /content/bootstrap.log 2>&1 & echo BOOTSTRAPPING",
                     shell=True, capture_output=True, text=True).stdout)
'''

DOWN_TRAILER = """
[down] If an assignment is still listed above, it is still billing.
       A100-80GB High-RAM costs about %s CU/h (~$0.68/h). 200 CU/month ≈ 29.5 h.
       Nothing here keeps a box alive: the frontend pings Colab only while a box is in use."""


# ---- running the Colab CLI ------------------------------------------------------- #

def colab_argv() -> list:
    """The Colab CLI: $COLAB when it names one, else the CLI itself through
    colab_cmd.py under this interpreter -- which works where `colab` does not."""
    exe = os.environ.get("COLAB")
    return [exe] if exe else [sys.executable, os.path.join(HERE, "colab_cmd.py")]


def cli_python() -> str:
    """The interpreter for restore.py and colab_keepalive.py: the CLI's own, this one."""
    return os.environ.get("COLAB_PY") or sys.executable


def _env() -> dict:
    # unbuffered: restore.py's "[restore] box: ..." has to reach the rail when it happens
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")


def run(argv: list, timeout: float | None = None) -> tuple[int, str]:
    """argv -> (exit code, stdout and stderr). A timeout is exit 124, as with
    GNU timeout; a command that cannot start at all is 127."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
                           env=_env())
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as exc:
        part = exc.stdout or ""
        if isinstance(part, bytes):
            part = part.decode("utf-8", "replace")
        return 124, part + "\n[timed out after %ss]" % timeout
    except OSError as exc:
        return 127, "!! cannot run %s: %r" % (argv[0], exc)


def colab(args: list, timeout: float | None = None) -> tuple[int, str]:
    return run(colab_argv() + list(args), timeout)


def stream(argv: list) -> int:
    """Run a step whose output belongs in this log as it happens (restore.py)."""
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                                errors="replace", env=_env())
    except OSError as exc:
        print("!! cannot run %s: %r" % (argv[0], exc), flush=True)
        return 127
    for line in proc.stdout:
        print(line.rstrip("\r\n"), flush=True)
    return proc.wait()


def say(msg: str) -> None:
    print("[up %s] %s" % (time.strftime("%H:%M:%S", time.gmtime()), msg), flush=True)


def indented(text: str) -> None:
    for line in (text or "").splitlines() or [""]:
        print("  " + line, flush=True)


def _recipe(cmd: str, rid: str) -> tuple[int, str]:
    """scripts/recipe.py's own `env` / `vmenv`, in-process: its stdout, and its code
    (2 unknown, 3 placeholder -- it says why on stderr, which is left alone)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = registry.main([cmd, rid])
        except Exception as exc:              # a registry that does not load says so
            print("[recipe] !! %r" % exc, file=sys.stderr)
            rc = 2
    return rc, buf.getvalue()


def _lf_copy(path: str, work: str) -> str:
    """The file as the VM needs it: a Windows editor's CRLF would break bash there
    (.gitattributes keeps a checkout LF; this keeps an edited file LF too)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if b"\r\n" not in data:
        return path
    out = os.path.join(work, os.path.basename(path))
    with open(out, "wb") as fh:
        fh.write(data.replace(b"\r\n", b"\n"))
    return out


def _minutes(x: float) -> str:
    return ("%d" % x) if float(x).is_integer() else ("%g" % x)


# ---- up ---------------------------------------------------------------------------- #

def cmd_up(a) -> int:
    session = a.session or os.environ.get("SESSION") or "collabosm"
    rid = a.recipe or os.environ.get("RECIPE") or registry.DEFAULT
    E = os.environ
    E["SESSION"], E["RECIPE"] = session, rid
    E.setdefault("RUNTIME", "wheel")             # wheel | source (manifest.json)
    E.setdefault("PORT", "8090")
    try:
        wait_min = float(E.get("WAIT_MIN") or 50)
        poll_s = float(E.get("COLLABOSM_POLL_S") or 45)
    except ValueError:
        say("!! WAIT_MIN / COLLABOSM_POLL_S must be numbers")
        return 2

    # ---- 0. the recipe: whatever the caller did not set comes from recipes.json
    rc, _ = _recipe("env", rid)
    if rc != 0:
        say("!! recipe %s cannot run (see above; list them with: python scripts/recipe.py list)" % rid)
        return 2
    for k, v in registry.launch_env(registry.load(), rid).items():
        E.setdefault(k, v)                       # a variable set by the caller wins
    say("recipe %s: %s shape=%s (>= %s GiB) · %s@%s"
        % (rid, E["ACCELERATOR"], E["SHAPE"], E["MIN_VRAM_GIB"], E["MODEL_REPO"],
           E["MODEL_REVISION"][:12]))
    if E.get("ENGINE") == "strata":
        say("  Strata %s @ %s: %s, context=%s kv=%s vision=%s, CUDA %s%s"
            % (E.get("STRATA_REPO", "?").rsplit("/", 2)[-2], E.get("STRATA_COMMIT", "?")[:12],
               E.get("STRATA_MODEL") or "?", E.get("STRATA_CONTEXT") or "?", E.get("STRATA_KV") or "?",
               E.get("STRATA_VISION") or "none", E.get("STRATA_CUDA") or "?",
               ", engine " + E["STRATA_ENGINE_ARGS"] if E.get("STRATA_ENGINE_ARGS") else ""))
    else:
        say("  cache=%s cq=%s ccs=%sGB rcs=%sGB ndt=%s gcs=%s vision=%s yarn=%s"
            % (E.get("CACHE_SIZE") or "?", E.get("CACHE_QUANT") or "?", E.get("CPU_CACHE_GB") or "0",
               E.get("RECURRENT_CACHE_GB") or "?", E.get("NDT") or "?", E.get("GCS") or "?",
               E.get("VISION") or "0", E.get("YARN_FACTOR") or "0"))

    # ---- 1. the box
    say("restoring/creating the %s box, shape %s (session: %s)" % (E["ACCELERATOR"], E["SHAPE"], session))
    rc = stream([cli_python(), os.path.join(HERE, "restore.py"), "-n", session,
                 "--accelerator", E["ACCELERATOR"], "--shape", E["SHAPE"],
                 "--min-vram-gib", E["MIN_VRAM_GIB"]])
    if rc != 0:
        say("!! restore failed (rc=%s). 6 = a box below %s GiB was drawn and stopped; just re-run."
            % (rc, E["MIN_VRAM_GIB"]))
        return rc

    with tempfile.TemporaryDirectory(prefix="collabosm-up-") as work:
        # ---- 2. the kit
        say("uploading the toolkit")
        for f in KIT:
            src = _lf_copy(os.path.join(HERE, f), work)
            rc, _ = colab(["upload", "-s", session, src, "/content/" + f], timeout=180)
            if rc != 0:
                say("  FAIL %s (upload) - refusing to continue" % f)
                return 1
            say("  ok   %s" % f)

        # ---- 3. bootstrap it
        say("bootstrapping (runtime=%s)" % E["RUNTIME"])
        # the recipe's launch settings as they stand now, overrides included
        rc, vmenv = _recipe("vmenv", rid)
        if rc != 0:
            say("!! could not write the launch environment for %s" % rid)
            return 2
        env_file = os.path.join(work, "collabosm_env.sh")
        with open(env_file, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(vmenv)
        # Checked: without it bootstrap never starts on a fresh box (and this would
        # wait its full 50 paid minutes), and on a re-attached box the previous run's
        # file is sourced, launching the old recipe under the new one's name.
        rc, _ = colab(["upload", "-s", session, env_file, "/content/collabosm_env.sh"], timeout=120)
        if rc != 0:
            say("!! could not upload the launch environment - refusing to continue")
            return 1
        starter = os.path.join(work, "start_bootstrap.py")
        with open(starter, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(START_BOOTSTRAP)
        rc, out = colab(["exec", "-s", session, "--timeout", "150", "-f", starter], timeout=200)
        if "BOOTSTRAPPING" not in out:
            say("!! the bootstrap did not start on the VM - refusing to wait %s min for it"
                % _minutes(wait_min))
            return 1
        print("BOOTSTRAPPING", flush=True)

    # ---- 4. wait for the endpoint
    say("waiting up to %s min for the endpoint" % _minutes(wait_min))
    deadline = time.time() + wait_min * 60
    while time.time() < deadline:
        time.sleep(poll_s)
        rc, out = colab(["exec", "-s", session, "--timeout", "60", "-f",
                         os.path.join(HERE, "status.py")], timeout=120)
        indented(out)
        m = re.search(r"^stage: *stage=([a-z_]*)", out, re.M)
        stage = m.group(1) if m else ""
        # READY needs the published URL as well as a healthy API: /health answers a few
        # seconds before serve.sh has the tunnel hostname, and a READY without a URL
        # leaves every client -- and the frontend -- with nothing to point at.
        if stage == "ready" and "health:         200" in out:
            say("READY")
            for line in out.splitlines():
                if line.startswith(("tunnel:", "models:")):
                    print("  " + line, flush=True)
            say("when you are done: python %s down   (stopping is the whole point)"
                % os.path.join(HERE, "provision.py"))
            return 0
        if stage in ("serve_failed", "serve_timeout"):
            say("!! serve.sh reported stage=%s - see /content/serve.log on the VM" % stage)
            return 8
        if stage == "ready_no_tunnel":
            say("!! the API is healthy on the VM but no tunnel URL was published - see /content/tunnel.log")
            return 9
        rc, out = colab(["exec", "-s", session, "--timeout", "60", "-f",
                         os.path.join(HERE, "bootstrap_ok.py")], timeout=120)
        if "bootstrap_failed" in out:
            # the log's last lines, here: a failed start is stopped right after this, and
            # its /content/bootstrap.log goes with the box
            indented(out.split("bootstrap_failed", 1)[1])
            say("!! bootstrap reported failure - the end of /content/bootstrap.log is above")
            return 1
    say("!! timed out after %s min" % _minutes(wait_min))
    return 7


# ---- down, fetch, sessions ------------------------------------------------------------ #

def cmd_down(a) -> int:
    session = a.session or os.environ.get("SESSION") or "collabosm"
    endpoint = a.endpoint or os.environ.get("ENDPOINT")
    if endpoint:
        # the record first, without a ping: nothing is assigned, nothing is held
        rc, out = run([cli_python(), os.path.join(HERE, "colab_keepalive.py"), "--no-ping",
                       session, endpoint], timeout=120)
        for line in out.splitlines():
            print("[down] " + line, flush=True)
    print("[down] stopping '%s' ..." % session, flush=True)
    rc, out = colab(["stop", "-s", session], timeout=300)
    print(out.rstrip("\n"), flush=True)
    if rc != 0:
        print("[down] stop returned nonzero", flush=True)
    print("[down] remaining server-side assignments:", flush=True)
    rc, out = colab(["sessions"], timeout=120)
    indented(out)
    print(DOWN_TRAILER % (os.environ.get("CU_PER_HOUR") or "6.77"), flush=True)
    return 0


def cmd_fetch(a) -> int:
    """One small file from the VM between __BEGIN__ / __END__, through the contents
    API (`colab download`) -- never `colab exec`, which goes through the kernel and
    was seen to hang for minutes. Nothing at all when there is no such file."""
    with tempfile.TemporaryDirectory(prefix="collabosm-fetch-") as work:
        dest = os.path.join(work, "file")
        colab(["download", "-s", a.session, a.remote, dest], timeout=a.timeout)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            with open(dest, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            sys.stdout.write("__BEGIN__\n" + text + "\n__END__\n")
            sys.stdout.flush()
    return 0


def cmd_sessions(a) -> int:
    rc, out = colab(["sessions"], timeout=a.timeout)
    sys.stdout.write(out)
    sys.stdout.flush()
    return rc


def main(argv: list) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="restore or create the box, then bootstrap and serve the recipe")
    up.add_argument("--recipe", help="recipe id (default $RECIPE, else %s)" % registry.DEFAULT)
    up.add_argument("--session", help="local session name (default $SESSION, else collabosm)")
    dn = sub.add_parser("down", help="stop the session")
    dn.add_argument("--session", help="default $SESSION, else collabosm")
    dn.add_argument("--endpoint", help="the session's endpoint, to put back a dropped record first")
    ft = sub.add_parser("fetch", help="print one small file from the VM")
    ft.add_argument("session")
    ft.add_argument("remote")
    ft.add_argument("timeout", nargs="?", type=float, default=90.0)
    ss = sub.add_parser("sessions", help="what is assigned on the account (colab sessions)")
    ss.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args(argv)
    return {"up": cmd_up, "down": cmd_down, "fetch": cmd_fetch, "sessions": cmd_sessions}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
