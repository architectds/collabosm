#!/usr/bin/env python3
"""Tell Colab our box is in use: `python colab_keepalive.py [--no-ping] <session> [<endpoint>]`.

Colab counts two things as use: the notebook kernel (`colab exec`) and the
keep-alive ping this sends. Chat reaches the VM through the Cloudflare tunnel
and is neither -- so a box whose service is up and answering looks unattended,
and Colab reclaimed one within 25 minutes of the last `colab exec`
(2026-09-26). The CLI's own answer is a keep-alive daemon, but a daemon inside
WSL dies whenever WSL shuts its VM down, and restore.py deliberately starts
none. The frontend runs this instead, every few minutes and only while the box
is in use (a chat inside its idle-stop window). An idle box gets no ping and is
left to the frontend's idle stop -- or to Colab.

It also heals the CLI's local record. The CLI drops a `sessions.json` entry
whenever list_assignments briefly leaves its endpoint out (seen mid-run, VM
intact), and every `colab download -s <session>` fails until the entry is back
-- which is how the frontend reads the VM's endpoint.json. When the entry is
gone but the endpoint is still assigned, it is registered again from the
assignment itself. Nothing here assigns: no VM is ever created.

`--no-ping` only heals the record (coupling needs the record, not the hold).

Prints one line: `KEEPALIVE ok <endpoint>` (with ` reattached` when the record
was healed), `KEEPALIVE_GONE` when Colab lists no such assignment, or
`KEEPALIVE_ERROR <reason>`. Run it with the CLI's own interpreter, like
scripts/colab_ccu.py.
"""
import glob
import os
import sys
import time

for cand in glob.glob(os.path.expanduser(
        "~/.local/share/uv/tools/google-colab-cli/lib/python3*/site-packages")):
    if cand not in sys.path:
        sys.path.insert(0, cand)


def main(argv):
    ping = "--no-ping" not in argv
    args = [a for a in argv if a != "--no-ping"]
    if not args:
        print("KEEPALIVE_ERROR usage: colab_keepalive.py [--no-ping] <session> [<endpoint>]")
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
        healed = ""
        if rec is None or rec.endpoint != endpoint:
            # two names on one VM is how the wrong machine gets stopped
            claimed = {s.endpoint for n, s in state.store.list().items() if n != name}
            if endpoint in claimed:
                print("KEEPALIVE_ERROR %s is registered under another session name" % endpoint)
                return 1
            a = live[endpoint]
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
        print("KEEPALIVE ok %s%s" % (endpoint, healed))
        return 0
    except Exception as exc:                              # noqa: BLE001 - it is a probe
        print("KEEPALIVE_ERROR %r" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
