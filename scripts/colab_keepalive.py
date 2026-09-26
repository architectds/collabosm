#!/usr/bin/env python3
"""Tell Colab our box is in use:
`python colab_keepalive.py [--no-ping] [--heartbeat] <session> [<endpoint>]`.

Chat reaches the VM through the Cloudflare tunnel, which Colab does not see: a
box whose service is up and answering looks unattended. What Colab does count is
the notebook kernel. Two boxes on 2026-09-26 were reclaimed 22-25 minutes after
their last kernel execution while chat went on -- the second although this
script had sent the CLI's keep-alive ping every 3 minutes, the last one 40
seconds before the box went (the CLI's history and the frontend's own record).
Boxes whose kernel ran something at least every ~15 minutes lived for hours. So
the ping alone does not hold a box. `--heartbeat` also runs scripts/heartbeat.py
on the box's kernel -- the use Colab counts -- which appends one line of the
box's health (GPU, RAM, disk, model server, tunnel) to /content/heartbeat.jsonl
and hands the same line back here.

The CLI's own answer is a keep-alive daemon, but a daemon inside WSL dies whenever
WSL shuts its VM down, and restore.py deliberately starts none. The frontend runs
this instead, every few minutes and only while the box is in use (a chat inside
its idle-stop window). An idle box gets neither ping nor heartbeat and is left to the
frontend's idle stop -- or to Colab.

It also heals the CLI's local record. The CLI drops the record when
list_assignments briefly leaves the endpoint out, and when a call to the VM is
refused (401/404) -- which is also what a runtime-proxy token past its expiry
gets: the record keeps the token it was registered with, and RuntimeProxyInfo
carries tokenExpiresInSeconds. Every long run so far lost its record about once
an hour with the VM intact, and until it is back every `colab download -s
<session>` fails -- which is how the frontend reads the VM's endpoint.json. So
the record's url and token are refreshed from the live assignment on every call,
and a record already dropped is registered again from the assignment itself.
Nothing here assigns: no VM is ever created.

`--no-ping` only heals the record (coupling needs the record, not the hold).

Prints `KEEPALIVE ok <endpoint>` (with ` reattached` when the record was put
back), `KEEPALIVE_GONE` when Colab lists no such assignment, or
`KEEPALIVE_ERROR <reason>`. With `--heartbeat`, an ok is followed by a second
line, `HEARTBEAT <json>` or `HEARTBEAT_ERROR <reason>`. Run it with the CLI's own
interpreter, like scripts/colab_ccu.py.
"""
import glob
import json
import os
import sys
import threading
import time

for cand in glob.glob(os.path.expanduser(
        "~/.local/share/uv/tools/google-colab-cli/lib/python3*/site-packages")):
    if cand not in sys.path:
        sys.path.insert(0, cand)

HEARTBEAT_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heartbeat.py")
HEARTBEAT_MARK = "HEARTBEAT_LINE "
HEARTBEAT_EXEC_S = 40   # heartbeat.py's own probes are bounded at ~35 s; it usually takes ~1 s
HEARTBEAT_LIMIT_S = 55  # connect included; with the listing and the ping, inside the frontend's 90 s


def _stdout(outputs):
    """The text a kernel printed, and the error it raised if it raised one."""
    text, err = [], None
    for o in outputs or []:
        if not isinstance(o, dict):
            continue
        if o.get("output_type") == "error":
            err = "%s: %s" % (o.get("ename"), o.get("evalue"))
        elif o.get("output_type") == "stream" or "text" in o:
            t = o.get("text")
            text.append("".join(t) if isinstance(t, list) else str(t or ""))
    return "".join(text), err


def heartbeat(state, name):
    """scripts/heartbeat.py on the box's kernel, the way `colab exec` runs a file:
    on the record's kernel, whose id -- or a new kernel's, when the record was just
    put back and has none -- is written back to the record. (ok, line or reason)."""
    from colab_cli.runtime import ColabRuntime

    with open(HEARTBEAT_PY, encoding="utf-8") as fh:
        code = fh.read()

    rec = state.store.get(name)
    if rec is None:
        return False, "no local record"

    def keep(attr):
        def on(value):
            setattr(rec, attr, value)
            state.store.add(rec)
        return on

    runtime = ColabRuntime(rec.url, rec.token, kernel_id=rec.kernel_id,
                           session_id=rec.session_id,
                           on_kernel_started=keep("kernel_id"),
                           on_session_started=keep("session_id"))
    t0 = time.time()
    try:
        outputs = runtime.execute_code(code, timeout=HEARTBEAT_EXEC_S)
    finally:
        try:
            runtime.stop()
        except Exception:                                # noqa: BLE001 - best effort
            pass
    ms = int((time.time() - t0) * 1000)
    text, err = _stdout(outputs)
    for line in text.splitlines():
        if line.startswith(HEARTBEAT_MARK):
            try:
                json.loads(line[len(HEARTBEAT_MARK):])
            except ValueError:
                break
            return True, line[len(HEARTBEAT_MARK):].strip()
    return False, (err or "the kernel printed no heartbeat (%d ms)" % ms)[:200]


def heartbeat_within(state, name, limit=HEARTBEAT_LIMIT_S):
    """heartbeat(), but never longer than `limit`: a kernel websocket that hangs
    must not take the frontend's whole wait with it."""
    box = {}

    def run():
        try:
            box["r"] = heartbeat(state, name)
        except Exception as exc:                          # noqa: BLE001 - it is a probe
            box["r"] = (False, repr(exc)[:200])

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(limit)
    return box.get("r") or (False, "no answer from the kernel within %d s" % limit)


def main(argv):
    ping = "--no-ping" not in argv
    beat = ping and "--heartbeat" in argv
    args = [a for a in argv if a not in ("--no-ping", "--heartbeat")]
    if not args:
        print("KEEPALIVE_ERROR usage: colab_keepalive.py [--no-ping] [--heartbeat] <session> [<endpoint>]")
        return 2
    name, hint = args[0], (args[1] if len(args) > 1 else None)
    try:
        from colab_cli.common import state
        from colab_cli.state import SessionState

        def listing():
            return {a.endpoint: a for a in state.client.list_assignments()}

        live = listing()
        rec = state.store.get(name)
        candidates = [e for e in ((rec.endpoint if rec is not None else None), hint) if e]
        if not candidates:
            print("KEEPALIVE_ERROR no local record for %s and no endpoint given" % name)
            return 1
        # list_assignments has been seen to leave out a live box for a few seconds
        # (the very flake that drops the CLI's record): ask again before saying GONE,
        # which makes the frontend stop holding -- and watching -- a box that bills
        for _ in range(2):
            if any(e in live for e in candidates):
                break
            time.sleep(3)
            live = listing()
        endpoint = next((e for e in candidates if e in live), None)
        if endpoint is None:
            print("KEEPALIVE_GONE")
            return 0
        if ping:
            state.client.keep_alive_assignment(endpoint)
        a = live[endpoint]
        healed = ""
        if rec is None or rec.endpoint != endpoint:
            # two names on one VM is how the wrong machine gets stopped
            claimed = {s.endpoint for n, s in state.store.list().items() if n != name}
            if endpoint in claimed:
                print("KEEPALIVE_ERROR %s is registered under another session name" % endpoint)
                return 1
            state.store.add(SessionState(
                name=name, token=a.runtime_proxy_info.token, url=a.runtime_proxy_info.url,
                endpoint=endpoint, variant="GPU",
                accelerator=getattr(a.accelerator, "value", str(a.accelerator))))
            try:
                state.history.log_event(name, "session_registered",
                                        {"endpoint": endpoint, "by": "colab_keepalive"})
            except Exception:
                pass
            healed = " reattached"
        elif (rec.token, rec.url) != (a.runtime_proxy_info.token, a.runtime_proxy_info.url):
            # the token the record was registered with runs out; the assignment's is
            # current. The kernel and session ids stay: they are the VM's, not the token's.
            rec.token, rec.url = a.runtime_proxy_info.token, a.runtime_proxy_info.url
            state.store.add(rec)
        print("KEEPALIVE ok %s%s" % (endpoint, healed), flush=True)
        if not beat:
            return 0
        ok, detail = heartbeat_within(state, name)
        try:
            # the CLI's history outlives the VM: the post-mortem of the last box came from it
            state.history.log_event(name, "heartbeat", {"ok": ok, "by": "colab_keepalive",
                                                        ("line" if ok else "error"): detail})
        except Exception:
            pass
        print(("HEARTBEAT %s" if ok else "HEARTBEAT_ERROR %s") % detail, flush=True)
        if not ok:
            os._exit(0)     # a hung kernel websocket thread must not hold the exit
        return 0
    except Exception as exc:                              # noqa: BLE001 - it is a probe
        print("KEEPALIVE_ERROR %r" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
