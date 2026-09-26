#!/usr/bin/env python3
"""Rehearsal for the shell's rail: the log lines up.sh emits, in ~24 s, no card.

This exists so the provisioning path can be driven and screenshotted without
spending CU. It prints exactly the shapes frontend/control.py parses -- the
`[up ...]` stage markers, the `[restore] box:` line, the relayed
scripts/status.py rows (`stage: stage=weights`, `gpu_MiB: ...`), the tunnel URL
and the final READY. Nothing here touches Colab, WSL, Docker or the network.

    python frontend/fake_provision.py            # ~24 s
    COLLABOSM_FAKE_SECONDS=3 python frontend/fake_provision.py
    COLLABOSM_FAKE_FAIL=serve python frontend/fake_provision.py   # model never comes up

The failure mode exits the way up.sh does when serve.sh reports failure (exit 8),
so the rail's failure path -- including the automatic stop -- can be rehearsed too.
"""
from __future__ import annotations

import os
import time

SCALE = float(os.environ.get("COLLABOSM_FAKE_SECONDS", "24")) / 24.0
FAIL = os.environ.get("COLLABOSM_FAKE_FAIL", "")
# the address the rehearsal VM "publishes": the fake VM when one serves (server.py
# --fake-vm), so the rail couples to what the log said instead of "moving" to it
URL = os.environ.get("COLLABOSM_FAKE_VM_URL") or "https://rehearsal-collabosm.trycloudflare.com"


def say(text: str) -> None:
    print("[up 00:00:00] %s" % text, flush=True)


def row(text: str) -> None:
    print("  " + text, flush=True)


def hold(seconds: float) -> None:
    time.sleep(max(0.0, seconds * SCALE))


say("restoring/creating the A100-80GB box (session: collabosm)")
hold(2.0)
print("[restore] re-issued token for existing registration: "
      "abc123 accel=A100 shape=HIGH_RAM", flush=True)
hold(1.0)
print("[restore] box: Tesla A100-SXM4-80GB | vram 79.3 GiB | ram 167 GiB | cc 8.0 "
      "| python 3.11 | disk 235.7 GB free", flush=True)
hold(2.0)

say("uploading the toolkit")
for name in ("api_server.py", "bootstrap.sh", "serve.sh", "status.py", "probe_gpu.py"):
    hold(0.4)
    row("ok   %s" % name)
hold(1.0)

say("bootstrapping (runtime=wheel)")
print("BOOTSTRAPPING", flush=True)
hold(4.0)

say("waiting up to 50 min for the endpoint")
row("stage:          stage=probing")
row("health:         000")
row("engine_running: False")
row("gpu_MiB:        0, 81920")
hold(3.0)
row("stage:          stage=weights")
row("gpu_MiB:        0, 81920")
hold(4.0)
row("stage:          stage=bootstrapped")
row("stage:          stage=loading")
row("gpu_MiB:        41234, 81920")
hold(4.0)
if FAIL == "serve":
    row("stage:          stage=serve_failed")
    row("health:         000")
    say("!! serve.sh reported stage=serve_failed - see /content/serve.log on the VM")
    raise SystemExit(8)
row("stage:          stage=ready url=%s port=8090" % URL)
row("health:         200")
row("engine_running: True")
row("gpu_MiB:        76481, 81920")
row("tunnel:         %s" % URL)
row('models:         {"data":[{"id":"Qwen3.8-Flash-Next"}]}')
hold(1.0)

say("READY")
say("when you are done: bash scripts/down.sh   (stopping is the whole point)")
