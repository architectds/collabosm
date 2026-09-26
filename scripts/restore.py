#!/usr/bin/env python3
"""collabosm - restore (or create) the Colab A100-80GB High-RAM session.

Why this script exists
----------------------
1. The local session record is disposable. `~/.config/colab-cli/sessions.json` is
   pruned whenever the runtime proxy token lapses, and the CLI then reports
   "No active sessions found" while the VM is still running and still billing.
   This re-attaches from server truth (`list_assignments`) instead of re-creating.
   Observed twice on 2026-09-24, once mid-run with /content fully intact.

2. `colab new --gpu A100` never sends `shape`, so it is a lottery between
   80 GB High-RAM (167 GB RAM, 6.77 CU/h as Colab reports it) and 40 GB standard (83 GB RAM,
   5.37 CU/h by Colab's published figure).
   The 40 GB box cannot load this model: the 4.05 bpw pack is ~100 GiB on disk with
   ~63.6 GiB of weights that must be VRAM-resident. Eleven consecutive unpatched
   attempts gave 40 GB. This requests HIGH_RAM explicitly (google-colab-cli#47:
   Shape.HIGH_RAM exists and machineShape is parsed, but is never sent).

3. A 40 GB box is worse than no box, because you pay for the probe. So the shape is
   verified immediately and an unsuitable runtime is STOPPED before anything is
   downloaded -- about 1 minute, ~0.13 CU, instead of ~10 CU of wasted setup.

Deliberately NOT done here: no keep-alive daemon is started. On a metered plan the
right default is "use it, then stop it"; a keep-alive turns a 2 h session into 24 h.
Pass --keepalive only when you are actively working and want the VM held open.

Exit codes: 0 ok | 2 bad args | 3 too many assignments | 4 assign failed
            5 cannot read assignments | 6 unsuitable box (stopped) | 7 timeout
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.expanduser("~/.config/colab-cli/sessions.json")


def _import_colab_cli():
    """The CLI lives in a uv tool venv; make it importable from any interpreter."""
    for cand in glob.glob(os.path.expanduser(
            "~/.local/share/uv/tools/google-colab-cli/lib/python3*/site-packages")):
        if cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        from colab_cli.common import state
        from colab_cli.state import SessionState
        return state, SessionState
    except Exception as exc:
        raise SystemExit(
            "cannot import google-colab-cli (%r).\n"
            "Install it with: uv tool install google-colab-cli\n"
            "Run this script with that tool venv python, e.g.\n"
            "  ~/.local/share/uv/tools/google-colab-cli/bin/python scripts/restore.py" % exc)


def local_registry():
    try:
        with open(REGISTRY) as fh:
            return json.load(fh)
    except Exception:
        return {}


def claimed_endpoints(except_name=None):
    """Endpoints already owned by another local session name.

    Two names pointing at one VM is how you end up stopping the wrong machine.
    """
    out = set()
    for name, rec in local_registry().items():
        if name == except_name:
            continue
        if isinstance(rec, dict) and rec.get("endpoint"):
            out.add(rec["endpoint"])
    return out


def attached_shape(assignment):
    return str(getattr(getattr(assignment, "machine_shape", None), "name", "?"))


def describe(assignment):
    accel = getattr(getattr(assignment, "accelerator", None), "value", "?")
    return "%s accel=%s shape=%s" % (assignment.endpoint, accel, attached_shape(assignment))


def register(state, SessionState, name, endpoint, token, url, accelerator, shape_note=""):
    state.store.add(SessionState(name=name, token=token, url=url, endpoint=endpoint,
                                 variant="GPU", accelerator=accelerator))
    try:
        state.history.log_event(name, "session_registered",
                                {"endpoint": endpoint, "accelerator": accelerator,
                                 "shape": shape_note})
    except Exception:
        pass


def maybe_keepalive(state, name, endpoint, want):
    if not want:
        return
    try:
        from colab_cli.commands.session import spawn_keep_alive
        pid = spawn_keep_alive(endpoint, name, auth_provider=state.auth_provider,
                               config_path=state.config_path)
        rec = state.store.get(name)
        if rec is not None:
            rec.keep_alive_pid = pid
            state.store.add(rec)
        print("[restore] keep-alive daemon started (pid %s) -- remember to stop it" % pid)
    except Exception as exc:
        print("[restore] could not start keep-alive: %r" % exc)


def _variant():
    from colab_cli.commands.session import Variant
    return Variant.GPU


def create_high_ram(state, SessionState, name, accelerator, shape_code):
    """Assign a new runtime, actually sending `shape=hm` (the upstream gap).

    `st` sends no shape at all: Colab's default is the standard 40 GB shape (every
    unpatched draw in colab.log came back machineShape 0), while `shape=st` itself
    has never been sent and is not known to be accepted."""
    from urllib.parse import urljoin

    import requests
    from colab_cli import client as C
    from colab_cli.client import TUN_ENDPOINT, TooManyAssignmentsError, uuid_to_web_safe_base64

    orig = C.Client._build_assign_url

    def _build(self, notebook_hash, variant=None, accelerator=None, shape=None):
        url = urljoin(self.colab_domain, "%s/assign" % TUN_ENDPOINT)
        params = {"nbh": uuid_to_web_safe_base64(notebook_hash)}
        if variant:
            params["variant"] = variant.value
        if accelerator:
            params["accelerator"] = accelerator.value
        if shape_code == "hm":
            params["shape"] = shape_code      # <-- the only added field
        return requests.Request("GET", url, params=params).prepare().url

    C.Client._build_assign_url = _build
    try:
        res = state.client.assign(uuid.uuid4(), variant=_variant(), accelerator=accelerator)
    except TooManyAssignmentsError:
        raise SystemExit(3)
    except Exception as exc:
        print("[restore] assign failed: %s: %s" % (type(exc).__name__, exc))
        raise SystemExit(4)
    finally:
        C.Client._build_assign_url = orig

    register(state, SessionState, name, res.endpoint, res.runtime_proxy_info.token,
             res.runtime_proxy_info.url, accelerator.value, shape_code)
    granted = "?"
    try:
        for a in state.client.list_assignments():
            if a.endpoint == res.endpoint:
                granted = attached_shape(a)
    except Exception:
        pass
    print("[restore] created %s endpoint=%s granted shape=%s" % (name, res.endpoint, granted))
    return res.endpoint, granted


def probe(colab, session, attempts=6):
    """Run scripts/probe_gpu.py on the VM and parse its one-line JSON answer."""
    probe_path = os.path.join(HERE, "probe_gpu.py")
    for i in range(1, attempts + 1):
        try:
            out = subprocess.run([colab, "exec", "-s", session, "--timeout", "180",
                                  "-f", probe_path],
                                 capture_output=True, text=True, timeout=300).stdout
        except Exception:
            out = ""
        for line in out.splitlines():
            if line.startswith("PROBE "):
                try:
                    return json.loads(line[len("PROBE "):])
                except Exception:
                    pass
        if i < attempts:
            print("[restore] probe %d/%d did not answer yet; waiting 30 s" % (i, attempts))
            time.sleep(30)
    return None


def stop_session(colab, session):
    subprocess.run([colab, "stop", "-s", session], capture_output=True, text=True, timeout=300)


def main():
    ap = argparse.ArgumentParser(description="restore or create the collabosm A100-80GB box")
    ap.add_argument("-n", "--name", default="collabosm")
    ap.add_argument("--accelerator", default="A100")
    ap.add_argument("--shape", choices=["hm", "st"], default="hm",
                    help="hm = HIGH_RAM (80 GB VRAM / 167 GB RAM), st = standard (40 GB)")
    ap.add_argument("--min-vram-gib", type=float, default=70.0,
                    help="refuse and stop the box below this (a 40 GB box measures ~39)")
    ap.add_argument("--colab", default=os.environ.get("COLAB", "colab"))
    ap.add_argument("--keepalive", action="store_true",
                    help="hold the VM open (bad on a metered plan)")
    ap.add_argument("--force-new", action="store_true",
                    help="create even if an unclaimed assignment is live")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    state, SessionState = _import_colab_cli()

    try:
        live = list(state.client.list_assignments())
    except Exception as exc:
        print("[restore] cannot read assignments: %r" % exc)
        return 5

    mine = state.store.get(args.name)
    caught = claimed_endpoints(except_name=args.name)
    # Our record names a box the listing does not show: list_assignments has been seen
    # to leave out a live box for a few seconds. Creating a VM on that one snapshot
    # would leave the first one billing with no record, so ask twice more first.
    for _ in range(2):
        if not mine or any(a.endpoint == mine.endpoint for a in live):
            break
        print("[restore] %s is not listed; asking again in 5 s before treating it as gone"
              % mine.endpoint)
        time.sleep(5)
        try:
            live = list(state.client.list_assignments())
        except Exception as exc:
            print("[restore] cannot read assignments: %r" % exc)
            return 5

    # ---- 1. refresh our own registration, if the VM is still ours
    if mine and any(a.endpoint == mine.endpoint for a in live):
        a = next(a for a in live if a.endpoint == mine.endpoint)
        state.client.keep_alive_assignment(mine.endpoint)
        register(state, SessionState, args.name, a.endpoint, a.runtime_proxy_info.token,
                 a.runtime_proxy_info.url, args.accelerator, attached_shape(a))
        print("[restore] re-issued token for existing registration: %s" % describe(a))
        maybe_keepalive(state, args.name, a.endpoint, args.keepalive)
        endpoint, granted, created = a.endpoint, attached_shape(a), False
    else:
        # ---- 2. adopt an orphaned assignment instead of creating another one
        adopt = [a for a in live
                 if a.endpoint not in caught
                 and getattr(getattr(a, "accelerator", None), "value", "") == args.accelerator]
        if adopt and not args.force_new:
            a = adopt[0]
            state.client.keep_alive_assignment(a.endpoint)
            register(state, SessionState, args.name, a.endpoint, a.runtime_proxy_info.token,
                     a.runtime_proxy_info.url, args.accelerator, attached_shape(a))
            print("[restore] adopted orphaned assignment %s (no new VM, no new billing)"
                  % describe(a))
            maybe_keepalive(state, args.name, a.endpoint, args.keepalive)
            endpoint, granted, created = a.endpoint, attached_shape(a), False
        else:
            for a in live:
                mark = " (claimed by another local session)" if a.endpoint in caught else ""
                print("[restore] live: %s%s" % (describe(a), mark))
            from colab_cli.commands.session import Accelerator
            acc = getattr(Accelerator, args.accelerator.upper(), None)
            if acc is None:
                print("[restore] unknown accelerator %s" % args.accelerator)
                return 2
            endpoint, granted = create_high_ram(state, SessionState, args.name, acc, args.shape)
            created = True

    result = {"name": args.name, "endpoint": endpoint, "granted_shape": granted,
              "created": created}

    # ---- 3. verify the box before it costs anything
    if not args.no_verify:
        info = probe(args.colab, args.name)
        if info is None:
            print("[restore] !! the VM did not answer the probe; the kernel may still be starting")
            result["probe"] = None
        else:
            result["probe"] = info
            vram = info.get("vram_GiB")
            print("[restore] box: %s | vram %.1f GiB | ram %s GiB | cc %s | python %s | disk %s free"
                  % (info.get("device"), vram or -1, info.get("ram_GiB"), info.get("cc"),
                     info.get("python"), info.get("disk_free")))
            if vram is not None and vram < args.min_vram_gib:
                print("[restore] !! %.1f GiB VRAM is below the %.1f GiB this recipe needs."
                      % (vram, args.min_vram_gib))
                print("[restore]    Its weights and cache must be VRAM-resident, so this box")
                print("[restore]    cannot load it at all (Flash-Next: ~63.6 GiB of weights).")
                if created or args.shape == "hm":
                    print("[restore]    stopping %s now, before any download; exit 6" % args.name)
                    stop_session(args.colab, args.name)
                    result["stopped"] = True
                print("[restore]    re-run this script to draw a new box; the lottery is real.")
                if args.json:
                    print(json.dumps(result))
                return 6

    if args.json:
        print(json.dumps(result))
    print("[restore] ready: session '%s' -> %s" % (args.name, endpoint))
    return 0


if __name__ == "__main__":
    sys.exit(main())