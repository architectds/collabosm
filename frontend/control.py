#!/usr/bin/env python3
"""The control plane behind the shell's right rail -- the real one.

The shell asks five questions and nothing else:

    status()                     what should the rail draw
    select(recipe, confirm)      start paying for a card
    cancel()                     forget a confirmation that was never given
    stop(reason)                 stop paying for a card
    couple()                     find the service that is running, and attach to it

Everything expensive lives behind that seam: the WSL bridge to the Colab CLI,
the CU ledger, the confirmation gate, the idle auto-stop, and the endpoint the
chat page is proxied to once the box is up.

Why WSL
    google-colab-cli is Linux-only (`uv tool install` has no Windows build), so
    the Windows frontend cannot import it. It shells out instead: every Colab
    action is `wsl.exe -d <distro> -- bash -lc ...`, which is also how
    scripts/up.sh, restore.py and down.sh are already driven by hand.

Why the confirmation gate
    Selecting a recipe is the moment billing starts. A bare click therefore
    returns `confirm_required` with the numbers (CU/h, ETA, cost of the load,
    what is left this month); only `confirm: true` starts the job.

Why an idle auto-stop, a keep-alive only while in use, and a stop on failure
    A100 High-RAM is 6.77 CU/h and the plan is ~200 CU/month, so 29.5 h. An
    idle session left open overnight is a month of work, so `idle_stop_min`
    (default 20) after the last chat request the VM is stopped and the reason
    is written into the ledger. The opposite failure is just as real: Colab
    counts only the kernel (`colab exec`) and the CLI's keep-alive ping as use,
    never chat through the tunnel, and it reclaimed a box whose service was up
    within 25 minutes of the last exec. So while the box is in use -- a chat
    inside that idle window -- it is kept alive every 3 minutes
    (scripts/colab_keepalive.py), and never otherwise: no daemon, nothing that
    outlives this process, and an idle box is left to the idle stop or to
    Colab. A job that fails after `assign` is stopped too: up.sh stops nothing
    on its way out, and a failed run whose VM keeps billing is worse than no
    run at all.

Coupling to a service that is already running
    A box can be up without this process having started it -- up.sh by hand, a
    frontend restart, another tool. couple() finds it: `colab sessions` says
    whether our session is live, and the VM's own files say where the service
    is and what its key is. Those files come through `colab download` (the
    contents API), never `colab exec`: exec goes through the kernel and was
    seen to hang for minutes. Everything after that is HTTP through the tunnel
    -- /health every 20 s, /v1/status every 60 s (launch parameters, activity,
    and the box's VRAM/RAM/disk) -- so a coupled rail needs no WSL on its hot
    path. When the tunnel stops answering the VM is asked again (a restarted
    serve.sh means a new quick-tunnel hostname), and when the session is gone
    its billing is closed.

    An adopted box is billed from the VM's own uptime, not from the moment we
    noticed it. Its idle auto-stop follows the server's request log when the
    server keeps one, because a client talking to the tunnel directly is
    activity this process never sees; a server that keeps no log gets no idle
    stop at all. Stopping a box someone is using is worse than showing its cost.

Adopt, never duplicate
    Provisioning goes through scripts/up.sh -> scripts/restore.py, which
    re-attaches to an existing assignment rather than creating a second one
    (that is what restore.py exists for: a pruned local session record once cost
    a duplicate VM).

Words are the shell's job
    The rail is drawn in English, Chinese or Japanese, so nothing here writes a
    sentence for it: `note` and `foot` are {"k": key, "a": args} and answers
    carry a `code`. shell.html owns every string in all three languages. Log
    lines stay raw -- they are what up.sh printed.

Rehearsal
    `fake=True` swaps the WSL command for frontend/fake_provision.py, which
    emits the same log lines in ~24 s. That is how this file is exercised
    without a card: server.py --fake-provision, or --mock with a ledger that is
    never written. COLLABOSM_FAKE_VM_URL=<a local endpoint> makes the rehearsal
    VM "serve" there, so coupling, link loss and rediscovery can be rehearsed too.
"""
from __future__ import annotations

import collections
import ctypes
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------- #
# recipes                                                                     #
# --------------------------------------------------------------------------- #

# The registry (recipes.json, read through scripts/recipe.py) is the one list of
# what can run where: the rail offers exactly its entries, and a pick sends only the
# recipe id to up.sh, which resolves everything else from the same file. Each
# number carries {v, measured, src}, so the rail can tell a measurement from an
# estimate without a word of it living here.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import recipe as registry  # noqa: E402

REGISTRY = registry.load()
RECIPES = registry.flat(REGISTRY)
GPUS = [{k: g.get(k) for k in ("id", "card", "accelerator", "shape", "vram_gb", "ram_gb",
                               "cu_per_hour", "facts")} for g in REGISTRY["gpus"]]
MODELS = [{k: m.get(k) for k in ("id", "name", "params", "quant", "native_ctx", "vision",
                                 "bytes")} for m in REGISTRY["models"]]

# --------------------------------------------------------------------------- #
# the provisioning stages, and how the rail should pace them                   #
# --------------------------------------------------------------------------- #

# (progress floor, progress ceiling) per stage, and the seconds it usually takes
STAGE_SPAN = {"requesting": (0.0, 8.0), "uploading": (8.0, 14.0),
              "bootstrapping": (14.0, 60.0), "loading": (60.0, 98.0)}
STAGE_SECS = {"requesting": 120.0, "uploading": 120.0,
              "bootstrapping": 330.0, "loading": 300.0}

# /content/STATUS stages (bootstrap.sh, then serve.sh) -> the rail stage they belong to
VM_STAGES = {"probing": "bootstrapping", "runtime": "bootstrapping",
             "weights": "bootstrapping", "env": "bootstrapping",
             "bootstrapped": "bootstrapping", "loading": "loading", "ready": "loading",
             "serve_failed": "loading", "serve_timeout": "loading",
             "ready_no_tunnel": "loading"}

# stages only move forward, so a relayed `stage=probing` line cannot drag the
# rail back from loading to bootstrapping while progress stays at 60%
STAGE_ORDER = ["idle", "requesting", "uploading", "bootstrapping", "loading", "ready"]

# A ledger session no frontend closed is billed up to the last heartbeat plus this.
# An unattended VM was reclaimed within 25 minutes (2026-09-26), but Colab's
# documented idle limit is ~90: a ledger may overstate a dead session, never
# understate it (docs/RUNBOOK.md).
STALE_GRACE_S = 90 * 60

# coupling cadence: cheap HTTP through the tunnel, WSL only when that fails
LINK_EVERY_S = 20            # /health while an endpoint is live
STATUS_EVERY_S = 60          # /v1/status: launch, activity, the box's VRAM/RAM/disk
LINK_FAILS_TO_ASK_VM = 3     # then ask the VM where the service is now
SESSIONS_EVERY_S = 300       # `colab sessions` while nothing is coupled
NO_ADOPT_AFTER_STOP_S = 120  # a box we just stopped is not re-adopted while it goes
BALANCE_EVERY_S = 300        # the account's real CU balance, from Colab
KEEPALIVE_EVERY_S = 180      # while in use; Colab reclaimed an unheld box within 25 min
KEEPALIVE_FIRST_S = 30       # a fresh session's first ping, or a retry until one succeeds
# `colab sessions` has been seen to leave out a live box for a few seconds (the CLI then
# prunes its record: docs/MEASURED.md), so one listing without our box is not proof it
# is gone. Two, this far apart, are.
GONE_AFTER_S = 60

NO_ENDPOINT = {"base": None, "key": None, "key_pending": False}
NO_LINK = {"ok": None, "latency_ms": None, "fails": 0, "checked_at": None}
NO_HOLD = {"holding": False, "at": None, "ok": None, "err": None}
ENDPOINT_ID = re.compile(r"^[A-Za-z0-9-]+$")

SESSION_LINE = re.compile(
    r"^\s*\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)\s*\|\s*Hardware:\s*(?P<hw>[^|\n]+)", re.M)


def _note(key: str, **args) -> dict:
    """A sentence for the rail, as data -- shell.html words it in the viewer's language."""
    return {"k": key, "a": args}


def _sanitize(text: str) -> str:
    return text.replace("\x00", "").replace("\r", "")


def _wsl_path(win_path: str) -> str:
    """E:\\models\\collabosm -> /mnt/e/models/collabosm (WSL sees the same files)."""
    p = os.path.abspath(win_path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    return "/mnt/%s/%s" % (m.group(1).lower(), m.group(2)) if m else p


def _root_of(base: str) -> str:
    """`http://host/v1` -> `http://host`.

    The rail shows the endpoint the way a human writes it (with /v1), but the
    proxy appends the *incoming* path, which already starts with /v1 -- so the
    base it forwards to must be the root, or every request becomes /v1/v1/...
    """
    trimmed = base.rstrip("/")
    return trimmed[:-3] if trimmed.endswith("/v1") else trimmed


def _is_loopback(url: str) -> bool:
    return (urllib.parse.urlsplit(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


def _http_json(url: str, key: str | None = None, timeout: float = 10.0):
    """GET from the live endpoint -> (HTTP status or 0, parsed JSON or None, ms)."""
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            ms = (time.time() - t0) * 1000.0
            try:
                return r.status, json.loads(body), ms
            except ValueError:
                return r.status, None, ms
    except urllib.error.HTTPError as e:
        return e.code, None, (time.time() - t0) * 1000.0
    except Exception:
        return 0, None, None


def _parse_sessions(out: str) -> list:
    """`colab sessions` prints `[name] endpoint | Hardware: A100 | Variant: GPU`.

    The old pattern only counted lines that *start* with an accelerator name, so
    a live A100 read as 0 billing instances -- the one number that must not lie.
    """
    found = [{"name": m.group("name"), "endpoint": m.group("endpoint"),
              "hardware": m.group("hw").strip()} for m in SESSION_LINE.finditer(out)]
    if not found:        # an older CLI printed one accelerator per line
        found = [{"name": None, "endpoint": None, "hardware": m.group(1)}
                 for m in re.finditer(r"^\s*(A100|T4|L4|H100|G4)\b", out, re.M)]
    return found


def _num(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None                      # nvidia-smi says "[N/A]"


def _local_machine(path: str) -> dict:
    """This PC, for an endpoint that runs on it (a local llama-server): the same
    shape api_server reports for its own box."""
    d = {"at": int(time.time()), "local": True, "gpu": None, "ram": None, "disk": None,
         "uptime_s": None}
    try:
        kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
        row = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, **kwargs).stdout.strip().splitlines()[0]
        name, used, total, util, temp, power = [x.strip() for x in row.split(",")]
        d["gpu"] = {"name": name, "used_mib": _num(used), "total_mib": _num(total),
                    "util_pct": _num(util), "temp_c": _num(temp), "power_w": _num(power)}
    except Exception:
        pass
    try:
        if os.name == "nt":
            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MemoryStatus()
            ms.dwLength = ctypes.sizeof(ms)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            total, avail = ms.ullTotalPhys, ms.ullAvailPhys
        else:
            info = {}
            with open("/proc/meminfo") as fh:
                for line in fh:
                    k, _, v = line.partition(":")
                    info[k] = int(v.split()[0]) * 1024
            total, avail = info["MemTotal"], info["MemAvailable"]
        d["ram"] = {"used_gib": round((total - avail) / 2 ** 30, 1),
                    "total_gib": round(total / 2 ** 30, 1)}
    except Exception:
        pass
    try:
        du = shutil.disk_usage(path)
        d["disk"] = {"free_gib": round(du.free / 2 ** 30, 1), "total_gib": round(du.total / 2 ** 30, 1)}
    except Exception:
        pass
    return d


class Control:
    """status() / select() / cancel() / stop() / couple() -- the shell's whole contract."""

    def __init__(self, root: str, *, session: str = "collabosm", distro: str = "Ubuntu",
                 budget_cu: float = 200.0, idle_stop_min: int = 20,
                 max_session_h: float = 6.0, fake: bool = False,
                 state_dir: str | None = None, persist_ledger: bool = True,
                 external: str | None = None, external_key: str | None = None,
                 external_model: str | None = None):
        self.root = os.path.abspath(root)
        self.session = session
        self.distro = distro
        self.fake = fake
        self.fake_vm = (os.environ.get("COLLABOSM_FAKE_VM_URL") or "").rstrip("/") or None
        if not fake:
            self.fake_vm = None
        self.fake_vm_down = False
        # An endpoint that is already running -- a local llama-server, someone
        # else's box, anything OpenAI-compatible. Nothing here is billed and
        # nothing here is ever stopped automatically: we did not start it.
        self.external = external.rstrip("/") if external else None
        self.budget_cu = float(budget_cu)
        self.idle_stop_min = int(idle_stop_min)
        self.max_session_h = float(max_session_h)
        # A rehearsal must never write into the money ledger: without this, a
        # fake run that nobody stopped leaves an "open session" behind and the
        # rail charges CU for a VM that was never created.
        if state_dir:
            self.state_dir = state_dir
        elif fake:
            self.state_dir = os.path.join(self.root, ".tmp", "state-rehearsal")
        else:
            self.state_dir = os.path.join(os.path.expanduser("~"), ".collabosm")
        self.ledger_path = os.path.join(self.state_dir, "ledger.json")
        self.persist_ledger = persist_ledger

        self.lock = threading.RLock()
        self.proc = None
        self.job_recipe = None
        self.job_started = 0.0
        self.stage_started = 0.0
        self.last_activity = time.time()
        self.stop_wanted = False
        self.log = collections.deque(maxlen=40)
        self.started_here = False       # this process ran the job that holds the box
        self.charge_started = 0.0       # when this process took charge: job or adoption
        self.coupling = False
        self.no_adopt_until = 0.0
        self.server_last_request = 0.0  # from the server's own request log
        self.last_session_count = None
        # our assignment's endpoint, remembered from `colab sessions`: when the CLI
        # drops its local record the line reads `[?] <endpoint>`, and this is how
        # that line is still recognised as ours
        self.known_endpoint = None
        self.keepalive_busy = False
        self.sessions_failed = False
        self.absent_since = None        # first listing without our box, since the last with it
        self.downing = False            # down.sh is running: nothing new may start meanwhile
        # Bumped by every stop, job start and let-go: a coupling that began before one
        # of those must not write its (now stale) findings over what happened since.
        self.epoch = 0

        self.ledger = self._load_ledger()
        # An open session at startup was opened by a frontend that is gone: it
        # crashed or was closed with a box up, and that VM may still be billing.
        # If couple() then finds the box alive, it takes the entry over.
        self.stale = bool(self.ledger.get("open")) and not self.external
        # the endpoint that entry was for, so a `[?]` line for it is still ours
        ep = (self.ledger.get("open") or {}).get("endpoint")
        if ep and ENDPOINT_ID.match(ep):
            self.known_endpoint = ep
        self.state = {
            "stage": "idle",
            "progress": 0.0,
            "note": _note("idle.pick"),
            "foot": _note("foot.control"),
            "selected": None,
            "live_model": None,
            # live measurements, when something measures them; the reference
            # numbers live in `measured` and are never passed off as these
            "metrics": None,
            "recipes": RECIPES,
            "gpus": GPUS,
            "models": MODELS,
            "budget_cu": self.budget_cu,
            "cu_used": 0.0,
            "cu_left": self.budget_cu,
            "session_hours": 0.0,
            "idle_stop_min": self.idle_stop_min,
            "idle_left_s": None,
            "idle_policy": "frontend",   # frontend | server | off
            "max_session_h": self.max_session_h,
            "endpoint": dict(NO_ENDPOINT),
            "adopted": False,
            "link": dict(NO_LINK),
            "server": None,              # the last /v1/status
            "server_at": None,
            "machine": None,             # VRAM / RAM / disk of the box serving the model
            "machine_at": None,
            "machine_source": None,      # server | local
            "vm": None,                  # our session is live, its service is not up
            "billing": {"count": None, "sessions": [], "raw": "", "checked_at": None,
                        "ok": None},
            # the keep-alive: held while the box is in use, and when Colab last heard it
            "keepalive": dict(NO_HOLD),
            # the account's real balance and burn rate, from Colab (scripts/colab_ccu.py):
            # it counts every session on the account, which a local ledger cannot
            "balance": None,
            "cli": {"wsl": distro, "colab": "unknown", "root": _wsl_path(self.root),
                    "command": None, "checked_at": None},
            "confirm": None,
            "fake": fake,
            "outgoing_model": None,
        }
        self._refresh_money()
        if self.external:
            with self.lock:
                self.state.update(
                    stage="attached", progress=100.0,
                    live_model=external_model or "external",
                    note=_note("attached"),
                    endpoint={"base": self.external, "key": external_key, "key_pending": False},
                    foot=_note("foot.external", base=self.external))
        elif self.stale:
            op = self.ledger["open"]
            with self.lock:
                self.state["note"] = _note("ledger.stale", recipe=op.get("recipe"),
                                           start=op.get("start"))
        threading.Thread(target=self._watch, daemon=True).start()
        threading.Thread(target=self._link_loop, daemon=True).start()
        threading.Thread(target=self._probe_cli, daemon=True).start()

    # ---- money ---------------------------------------------------------- #

    def _load_ledger(self) -> dict:
        empty = {"closed_cu": 0.0, "entries": [], "open": None}
        if not self.persist_ledger or not os.path.exists(self.ledger_path):
            return empty
        try:
            with open(self.ledger_path) as fh:
                led = json.load(fh)
            led.setdefault("closed_cu", 0.0)
            led.setdefault("entries", [])
            led.setdefault("open", None)
            return led
        except Exception as exc:
            # Starting from zero silently would forget an open session that may still
            # be billing: keep the file, and say so in the log.
            kept = "%s.corrupt-%d" % (self.ledger_path, int(time.time()))
            try:
                os.replace(self.ledger_path, kept)
            except OSError:
                kept = self.ledger_path
            self._log("!! ledger.json could not be read (%r); kept as %s -- check `colab "
                      "sessions` for a box that may still be billing" % (exc, kept))
            return empty

    def _save_ledger(self) -> None:
        if not self.persist_ledger:
            return
        # under the lock: several threads save, and they share one temporary file
        with self.lock:
            try:
                os.makedirs(self.state_dir, exist_ok=True)
                tmp = self.ledger_path + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(self.ledger, fh, indent=1)
                os.replace(tmp, self.ledger_path)
            except Exception as exc:
                self._log("!! ledger write failed: %r" % exc)

    def _open_session(self, recipe: dict) -> None:
        with self.lock:
            if self.ledger.get("open") and not self.stale:
                return               # an adopted box: its entry already runs
            now = time.time()
            self.ledger["open"] = {"start": now, "seen": now, "recipe": recipe["id"],
                                   "cu_per_hour": recipe["cu_per_hour"]}
        self._save_ledger()

    def _billed_until(self, op: dict) -> float:
        """Now -- or, for a session no running frontend opened, the last heartbeat
        plus Colab's idle prune: the VM cannot have outlived that unattended, and a
        multi-day phantom would eat the whole month's budget."""
        now = time.time()
        if self.stale:
            return min(now, op.get("seen", op["start"]) + STALE_GRACE_S)
        return now

    def _close_session(self, reason: str) -> float:
        with self.lock:
            op = self.ledger.get("open")
            if not op:
                return 0.0
            end = self._billed_until(op)
            hours = max(0.0, end - op["start"]) / 3600.0
            cu = hours * op["cu_per_hour"]
            self.ledger["closed_cu"] = round(self.ledger.get("closed_cu", 0.0) + cu, 4)
            entry = {"start": op["start"], "end": end, "minutes": round(hours * 60, 1),
                     "cu": round(cu, 3), "recipe": op["recipe"],
                     "cu_per_hour": op["cu_per_hour"], "reason": reason}
            if op.get("adopted"):
                entry["adopted"] = True
                entry["start_from"] = op.get("start_from")
            self.ledger["entries"].append(entry)
            self.ledger["entries"] = self.ledger["entries"][-200:]
            self.ledger["open"] = None
            self.stale = False
        self._save_ledger()
        return cu

    def _refresh_money(self) -> None:
        op = self.ledger.get("open")
        used = float(self.ledger.get("closed_cu", 0.0))
        hours = 0.0
        if op:
            hours = max(0.0, self._billed_until(op) - op["start"]) / 3600.0
            used += hours * op["cu_per_hour"]
        with self.lock:
            self.state["cu_used"] = round(used, 3)
            self.state["cu_left"] = round(max(0.0, self.budget_cu - used), 2)
            self.state["session_hours"] = round(hours, 3)

    # ---- the contract ---------------------------------------------------- #

    def _idle_now(self, now: float):
        """Seconds idle, or None when this box gets no idle stop."""
        policy = self.state.get("idle_policy")
        if policy == "off":
            return None
        last = self.last_activity
        if policy == "server":
            last = max(last, self.server_last_request)
        return now - last

    def _in_use(self, now: float) -> bool:
        """A chat inside the idle-stop window: the only reason to hold the box.

        With a server log that is anyone's chat; without one (policy "off") it is
        chat through this frontend, the only traffic it can see -- a box nobody
        uses through us is then left to Colab rather than held for hours."""
        last = self.last_activity
        if self.state.get("idle_policy") == "server":
            last = max(last, self.server_last_request)
        return now - last < self.idle_stop_min * 60

    def status(self) -> dict:
        self._refresh_money()
        with self.lock:
            # the idle countdown is read here too, not only in the watcher, so a
            # freshly-ready box does not show "—" for up to one watch tick
            if self.state["stage"] == "ready" and self.ledger.get("open"):
                idle = self._idle_now(time.time())
                self.state["idle_left_s"] = (None if idle is None
                                             else int(max(0, self.idle_stop_min * 60 - idle)))
            st = json.loads(json.dumps(self.state))
            op = self.ledger.get("open")
            st["ledger_open"] = op is not None
            st["coupling"] = self.coupling
            st["downing"] = self.downing
            bal = self.state.get("balance") or {}
            # Colab's own rate when this box is the account's only assignment; the
            # recipe's figure otherwise (Colab's rate is for the whole account)
            rate = (bal.get("rate") if bal.get("rate") and bal.get("assignments") == 1
                    else (op or {}).get("cu_per_hour"))
            st["session_cu"] = round(self.state["session_hours"] * rate, 2) if op and rate else None
        # The VM key is ours to hold, not the browser's: the rail only needs to
        # know that there is one.
        st["endpoint"]["key"] = bool(st["endpoint"].get("key"))
        st["log_tail"] = list(self.log)[-12:]
        return st

    def select(self, recipe_id: str, confirm: bool = False) -> dict:
        recipe = next((r for r in RECIPES if r["id"] == recipe_id), None)
        if recipe is None:
            return {"ok": False, "code": "unknown_recipe"}
        with self.lock:
            stage = self.state["stage"]
            if stage == "attached":
                return {"ok": False, "code": "attached", "base": self.external}
            if stage in ("requesting", "uploading", "bootstrapping", "loading", "stopping") \
                    or self.coupling or self.downing:
                # downing: a failed run's box is still being stopped, and restore.py
                # could re-attach to it just before that stop lands
                return {"ok": False, "code": "busy", "stage": "stopping" if self.downing else stage}
            if stage == "ready":
                if self.state["live_model"] == recipe["model"] or self.state.get("adopted"):
                    return {"ok": True, "code": "already"}
                return {"ok": False, "code": "stop_first"}
            if self.ledger.get("open") and self.stale:
                # a session the last frontend never closed: stop (and account for)
                # it before a new one opens on top of it
                return {"ok": False, "code": "stale_ledger"}
            if not recipe["runnable"]:
                return {"ok": False, "code": "placeholder", "recipe": recipe["id"]}
            bal = self.state.get("balance") or {}
            left = bal["cu"] if bal.get("cu") is not None else self.state["cu_left"]
            need = round(recipe["cu_per_hour"] * ((recipe["eta_min"] + 5) / 60.0), 2)
            if left < max(3.0, need):
                return {"ok": False, "code": "budget", "cu_left": round(left, 1),
                        "cu_need": round(need, 1)}
            warn = {"recipe": recipe["id"], "card": recipe["card"], "model": recipe["model"],
                    "status": recipe["status"],
                    "rate_measured": bool((recipe["facts"].get("cu_per_hour") or {}).get("measured")),
                    "cu_per_hour": recipe["cu_per_hour"],
                    "usd_per_hour": round(recipe["cu_per_hour"] * 0.0999, 2),
                    "eta_min": recipe["eta_min"],
                    "cu_estimate": need,
                    "budget_cu": self.budget_cu,
                    "cu_left": round(left, 2),
                    "cu_left_after": round(left - need, 2),
                    "balance_real": bal.get("cu") is not None,
                    "idle_stop_min": self.idle_stop_min,
                    "max_session_h": self.max_session_h}
            if not confirm:
                self.state["confirm"] = warn
                return {"ok": True, "code": "confirm_required", "warning": warn}
            # taken before the lock is released: a second confirmed select (another
            # tab, a double click) must find the stage busy, not start a second VM
            self.state.update(stage="requesting", confirm=None)
        self._start_job(recipe)
        return {"ok": True, "code": "started", "recipe": recipe["id"]}

    def cancel(self) -> dict:
        """Forget a pending confirmation. Nothing was started, so nothing is billed.

        The shell used to clear only its own copy, and the next poll brought the
        card straight back with every recipe button still disabled.
        """
        with self.lock:
            self.state["confirm"] = None
        return {"ok": True, "code": "cancelled"}

    def stop(self, reason: str = "manual") -> dict:
        with self.lock:
            stage = self.state["stage"]
            if stage == "attached":
                self.state.update(stage="idle", progress=0.0, live_model=None, metrics=None,
                                  endpoint=dict(NO_ENDPOINT), link=dict(NO_LINK), server=None,
                                  machine=None, note=_note("detached"),
                                  foot=_note("foot.control"))
                return {"ok": True, "code": "detached"}
            if stage == "stopping":
                return {"ok": True, "code": "already_stopping"}
            open_entry = self.ledger.get("open") is not None
            ours_live = any(s.get("ours") for s in self.state["billing"].get("sessions") or [])
            if stage in ("idle", "stopped") and not (open_entry or ours_live):
                return {"ok": True, "code": "nothing_to_stop"}
            # `failed` gets here on purpose: _fail() already ran down.sh, and this
            # is the manual retry for when the billing probe still shows a VM.
            running_job = stage in ("requesting", "uploading", "bootstrapping", "loading")
            if self.stale and open_entry and stage in ("idle", "stopped"):
                reason = "stale"
            self.state["stage"] = "stopping"
            self.state["note"] = _note("stopping", reason=reason)
            self.stop_wanted = True
            self.epoch += 1          # a coupling still in flight must not revive this box
            proc = self.proc
        cu = self._close_session(reason)
        if running_job and proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        threading.Thread(target=self._do_stop, args=(reason, cu), daemon=True).start()
        return {"ok": True, "code": "stopping", "reason": reason, "cu": round(cu, 3)}

    def couple(self, why: str = "manual") -> dict:
        """Find the service on our live session and attach the rail to it.

        Also the rediscovery path: a ready rail whose tunnel stopped answering
        calls this to learn the new address (or that the VM is gone).
        """
        with self.lock:
            if self.coupling:
                return {"ok": True, "code": "coupling"}
            stage = self.state["stage"]
            if stage == "attached":
                return {"ok": False, "code": "attached", "base": self.external}
            if stage in ("requesting", "uploading", "bootstrapping", "stopping"):
                return {"ok": False, "code": "busy", "stage": stage}
            if stage == "loading" and why != "after_ready":
                return {"ok": False, "code": "busy", "stage": stage}
            self.coupling = True
            token = self.epoch
            before = self.state["note"]
            if stage in ("idle", "stopped", "failed") and why in ("manual", "startup"):
                self.state["note"] = _note("coupling")
        try:
            return self._couple(why, token)
        except Exception as exc:
            self._log("!! coupling failed: %r" % exc)
            return {"ok": False, "code": "error", "err": repr(exc)}
        finally:
            with self.lock:
                self.coupling = False
                # nothing found: put back what the rail said before (a stale-ledger
                # warning must survive a startup probe that finds no box)
                if (self.state.get("note") or {}).get("k") == "coupling":
                    self.state["note"] = before

    # ---- helpers the HTTP layer needs ------------------------------------ #

    def backend_base(self):
        with self.lock:
            stage = self.state["stage"]
            base = self.state["endpoint"].get("base")
        if self.fake and not self.fake_vm:
            # The rehearsal endpoint is a placeholder host, so keep talking to
            # --backend (the loopback stub) instead of trying to resolve it.
            return None
        if self.external and stage == "attached":
            return _root_of(self.external)
        return _root_of(base) if stage == "ready" and base else None

    def api_key(self):
        return self.state["endpoint"].get("key")

    def server_info(self):
        with self.lock:
            return self.state.get("server")

    def note_outgoing_model(self, model) -> None:
        with self.lock:
            self.state["outgoing_model"] = model
            self.last_activity = time.time()

    def note_metrics(self, metrics: dict) -> None:
        """One measured turn from the proxy: the server's timings when it sends
        them, the proxy's own first-token time always."""
        with self.lock:
            self.state["metrics"] = metrics

    def touch(self) -> None:
        """Any chat traffic counts as activity for the idle auto-stop."""
        self.last_activity = time.time()

    # ---- provisioning ---------------------------------------------------- #

    def _start_job(self, recipe: dict) -> None:
        with self.lock:
            self.epoch += 1
            self.job_recipe = recipe
            self.job_started = time.time()
            self.stage_started = time.time()
            self.stop_wanted = False
            self.started_here = True
            self.charge_started = time.time()
            self.fake_vm_down = False
            self.last_activity = time.time()
            self.state.update(stage="requesting", progress=0.0,
                              note=_note("stage.requesting"), adopted=False, vm=None,
                              selected=recipe["id"], live_model=None, metrics=None,
                              confirm=None, endpoint=dict(NO_ENDPOINT), link=dict(NO_LINK),
                              foot=_note("foot.billing", card=recipe["card"],
                                         cuph=recipe["cu_per_hour"]))
        self._open_session(recipe)
        self._balance_soon()
        threading.Thread(target=self._beat, daemon=True).start()
        threading.Thread(target=self._run, args=(recipe,), daemon=True).start()

    def _wsl_command(self, recipe: dict) -> list:
        # only the id: up.sh resolves the card, the model and every flag from the
        # same recipes.json, so the rail and a hand-run up.sh cannot disagree
        exports = {"SESSION": self.session, "COLAB": "$HOME/.local/bin/colab",
                   "RECIPE": recipe["id"]}
        env_str = " ".join("%s=%s" % (k, shlex.quote(v)) for k, v in exports.items())
        inner = ("cd %s && export %s && exec bash scripts/up.sh"
                 % (shlex.quote(_wsl_path(self.root)), env_str))
        return ["wsl.exe", "-d", self.distro, "--", "bash", "-lc", inner]

    def _spawn(self, recipe: dict):
        if self.fake:
            argv = [sys.executable, os.path.join(self.root, "frontend", "fake_provision.py")]
            return subprocess.Popen(argv, cwd=self.root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
        argv = self._wsl_command(recipe)
        env = dict(os.environ)
        env["WSL_UTF8"] = "1"        # wsl.exe otherwise prints UTF-16 for its own messages
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000      # CREATE_NO_WINDOW
        with self.lock:
            self.state["cli"]["command"] = " ".join(argv)
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1,
                                env=env, **kwargs)

    def _run(self, recipe: dict) -> None:
        try:
            proc = self._spawn(recipe)
        except Exception as exc:
            self._fail(_note("fail.spawn", err=repr(exc)), "could not launch: %r" % exc,
                       vm_possible=False)
            return
        self.proc = proc
        try:
            for raw in proc.stdout:
                self._absorb(_sanitize(raw))
            rc = proc.wait()
        except Exception as exc:
            self.proc = None
            try:
                proc.terminate()
            except Exception:
                pass
            self._fail(_note("fail.read", err=repr(exc)), "reading the job log failed: %r" % exc)
            return
        self.proc = None

        if self.stop_wanted:
            return
        if rc == 0 and self.state["stage"] == "ready":
            self._after_ready(recipe)
            return
        if rc == 0:
            self._fail(_note("fail.noready"), "up.sh exited 0 without READY")
            return
        # 2-5 never got a box and restore.py already stopped a 40 GB one (6); down.sh
        # is harmless for those and settles the question for all the rest.
        self._fail(_note("fail.exit", rc=rc), "up.sh exited %s" % rc)

    def _after_ready(self, recipe: dict) -> None:
        """Couple to what up.sh just brought up: the key and the published URL come
        from the VM's files, the rest over the tunnel -- the same path as adoption."""
        if self.fake and not self.fake_vm:
            with self.lock:
                self.state["endpoint"]["key"] = "sk-collabosm-rehearsal"
                self.state["endpoint"]["key_pending"] = False
                self.state["note"] = _note("ready")
            return
        with self.lock:
            self.state["endpoint"]["key_pending"] = True
        r = self.couple("after_ready")
        with self.lock:
            self.state["endpoint"]["key_pending"] = False
        if r.get("code") != "coupled":
            self._log("!! READY, but the rail could not couple (%s) -- chat will 503"
                      % r.get("code"))

    def _run_down(self) -> None:
        if self.fake:
            out = "[down] rehearsal: no VM was ever created"
            self.fake_vm_down = True
        else:
            # our session by name -- down.sh's default is "collabosm", and a stop that
            # names the wrong session "works" while ours keeps billing -- and our
            # endpoint, so down.sh can put back a record the CLI dropped before it stops
            ep = self.known_endpoint
            extra = "SESSION=%s " % shlex.quote(self.session)
            if ep and ENDPOINT_ID.match(ep):
                extra += "ENDPOINT=%s " % ep
            argv = ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                    "cd %s && %sCOLAB=$HOME/.local/bin/colab bash scripts/down.sh"
                    % (shlex.quote(_wsl_path(self.root)), extra)]
            out = self._wsl_run(argv, timeout=300)
        for line in out.splitlines()[-6:]:
            self._log(line)

    def _let_go(self) -> None:
        """Forget the box: whatever held it (a job, an adoption) no longer does."""
        self.epoch += 1
        self.started_here = False
        self.charge_started = 0.0
        self.no_adopt_until = time.time() + NO_ADOPT_AFTER_STOP_S
        # known_endpoint stays: until Colab stops listing it, a `[?]` line with it is
        # still our box (and still billing), not someone else's
        self.state.update(adopted=False, endpoint=dict(NO_ENDPOINT), link=dict(NO_LINK),
                          server=None, machine=None, machine_source=None, vm=None,
                          idle_policy="frontend", live_model=None, keepalive=dict(NO_HOLD))

    def _do_stop(self, reason: str, cu: float) -> None:
        self._run_down()
        with self.lock:
            self._let_go()
            self.state.update(stage="stopped", progress=0.0,
                              note=_note("stopped", reason=reason, cu=round(cu, 2)),
                              foot=_note("foot.stopped", reason=reason))
        self._refresh_money()
        self._probe_sessions()
        self._balance_soon()

    def _fail(self, note: dict, why: str, vm_possible: bool = True) -> None:
        cu = self._close_session("failed")
        note["a"]["autostop"] = "running" if vm_possible else "none"
        with self.lock:
            self._let_go()
            self.state.update(stage="failed", note=note, metrics=None,
                              foot=_note("foot.failed", cu=round(cu, 2)))
        self._log("!! " + why)
        self._refresh_money()
        if not vm_possible:
            return
        # up.sh stops nothing on its way out, so a failure after `assign` leaves a
        # VM billing behind a ledger entry that says it closed -- and the rail used
        # to refuse to stop a failed job at all.
        self._run_down_marked()
        with self.lock:
            if self.state["stage"] == "failed":
                self.state["note"]["a"]["autostop"] = "done"
        self._probe_sessions()

    def _vm_gone(self, token=None) -> None:
        """Our session vanished while coupled: Colab pruned it, or it was stopped
        somewhere else. Stop billing it here too -- and run the stop anyway: it is
        harmless for a box that is gone, and it settles one that only looked gone."""
        with self.lock:
            if token is not None and token != self.epoch:
                return                     # a stop or a new job got here first
        cu = self._close_session("vm_gone")
        with self.lock:
            self._let_go()
            self.state.update(stage="stopped", progress=0.0,
                              note=_note("vm.gone", cu=round(cu, 2)),
                              foot=_note("foot.stopped", reason="vm_gone"))
        self._log("[fe] our session is gone -- billing closed at %.2f CU" % cu)
        self._refresh_money()
        threading.Thread(target=self._run_down_marked, daemon=True).start()

    def _run_down_marked(self) -> None:
        """down.sh, with new starts refused while it runs: restore.py could re-attach
        to the very box this is stopping, and the stop would then kill the new job."""
        with self.lock:
            self.downing = True
        try:
            self._run_down()
        finally:
            with self.lock:
                self.downing = False

    def _wsl_run(self, argv: list, timeout: float = 120.0) -> str:
        try:
            env = dict(os.environ)
            env["WSL_UTF8"] = "1"
            kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
            res = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=timeout, env=env, **kwargs)
            return _sanitize((res.stdout or "") + (res.stderr or ""))
        except Exception as exc:
            return "!! %r" % exc

    # ---- coupling -------------------------------------------------------- #

    def _download(self, remote: str, timeout: float = 100.0):
        """One small file from the VM through the contents API -> its text, or None.

        Never `colab exec`: that goes through the kernel and was seen to hang for
        minutes. The work is in scripts/vm_fetch.sh, not an inline script: wsl.exe
        runs what follows `--` through the default shell first, so an inline
        script's variables were expanded, empty, before bash ever set them.
        """
        fetch = _wsl_path(self.root) + "/scripts/vm_fetch.sh"
        out = self._wsl_run(["wsl.exe", "-d", self.distro, "--", "bash", "-l", fetch,
                             self.session, remote, str(int(timeout) - 10)], timeout=timeout)
        m = re.search(r"__BEGIN__\n(.*?)\n?__END__", out, re.S)
        return m.group(1).strip() if m else None

    def _discover(self) -> dict:
        """Where the service is, what its key is, and what the VM says it is doing.

        serve.sh writes all of it into /content/endpoint.json, one download; a VM
        set up by an older serve.sh has /content/STATUS and api-key.txt instead.
        """
        if self.fake:
            if not self.fake_vm or self.fake_vm_down:
                return {}
            return {"url": self.fake_vm, "key": "sk-collabosm-rehearsal", "stage": "ready"}
        found = {}
        text = self._download("/content/endpoint.json")
        if text:
            try:
                ep = json.loads(text)
                found.update(url=ep.get("url"), key=ep.get("key"), stage="ready",
                             recipe=ep.get("recipe"))
            except ValueError:
                pass
        if not found.get("url") or not found.get("key"):
            status = self._download("/content/STATUS") or ""
            m = re.search(r"stage=(\S+)", status)
            found["stage"] = m.group(1) if m else (status.strip()[:60] or None)
            m = re.search(r"url=(\S+)", status)
            if m:
                found["url"] = m.group(1)
            if found.get("url") and not found.get("key"):
                found["key"] = (self._download("/content/api-key.txt") or "").strip() or None
        return found

    def _vm_stage_now(self):
        """What STATUS says right now (endpoint.json can outlive a restart)."""
        if self.fake:
            return "ready" if self.fake_vm and not self.fake_vm_down else None
        m = re.search(r"stage=(\S+)", self._download("/content/STATUS") or "")
        return m.group(1) if m else None

    def _recipe_for(self, ours: dict | None, server: dict | None, hint: str | None = None) -> dict:
        """Which recipe an adopted box runs: the server says so (RECIPE, reported in
        /v1/status and endpoint.json); an older server is judged by its card."""
        named = (server or {}).get("recipe") or hint
        found = next((r for r in RECIPES if r["id"] == named), None)
        if found:
            return found
        runnable = [r for r in RECIPES if r["runnable"]]
        total = ((((server or {}).get("machine") or {}).get("gpu")) or {}).get("total_mib")
        if total:
            fits = [r for r in runnable if r["vram_gb"] * 1024 * 0.9 <= total]
            if fits:
                return max(fits, key=lambda r: r["vram_gb"])
        endpoint = ((ours or {}).get("endpoint") or "").lower()
        hardware = ((ours or {}).get("hardware") or "").upper()
        if hardware == "A100" and "-hm-" not in endpoint and endpoint:
            return next((r for r in runnable if r["vram_gb"] == 40), runnable[0])
        return runnable[0]

    def _ensure_charge(self, ours: dict | None, server: dict | None, token=None):
        """A box we did not start has been billing since before we saw it: open
        its ledger entry from the VM's own uptime (at least the server's), not from
        the moment we noticed it -- but never from before the last entry closed,
        or the same hours are billed twice. None when a stop or a new job has
        happened since this coupling began (token): then nothing is written."""
        now = time.time()
        machine = (server or {}).get("machine") or {}
        # the VM's own uptime when the server reports it (capped at Colab's 24 h
        # runtime limit), else at least as long as the server has been up
        age = min(machine.get("uptime_s") or (server or {}).get("uptime_s") or 0, 24 * 3600)
        source = ("vm_uptime" if machine.get("uptime_s")
                  else "server_uptime" if age else "adoption")
        recipe = self._recipe_for(ours, server)
        with self.lock:
            if token is not None and token != self.epoch:
                return None
            op = self.ledger.get("open")
            # A stale entry is a previous VM's when this box is younger than it, or
            # when its last heartbeat is older than any unattended box can live
            # (with no uptime to go by, taking it over would bill days as "now").
            previous_vm = bool(op and self.stale and (
                (age and op["start"] < now - age - 600)
                or now - op.get("seen", op["start"]) > STALE_GRACE_S))
        if previous_vm:
            self._close_session("stale")       # a previous VM's entry; this one is new
        with self.lock:
            if token is not None and token != self.epoch:
                return None
            entries = self.ledger.get("entries") or []
            floor = max((e.get("end") or 0) for e in entries[-5:]) if entries else 0
            start = max(now - age, floor)
            op = self.ledger.get("open")
            if op:
                self.stale = False             # it is this box: keep its entry going
                op["seen"] = now
                if op.get("adopted") and op.get("start_from") in ("adoption", "server_uptime") \
                        and age and start < op["start"] and source != op.get("start_from"):
                    op["start"], op["start_from"] = start, source      # learned its age
            else:
                self.ledger["open"] = {"start": start, "seen": now, "recipe": recipe["id"],
                                       "cu_per_hour": recipe["cu_per_hour"], "adopted": True,
                                       "start_from": source if start == now - age else "last_entry",
                                       "endpoint": self.known_endpoint}
            if not self.charge_started:
                self.charge_started = now
        self._save_ledger()
        return recipe

    def _learn_endpoint(self, ep) -> None:
        """Our assignment's endpoint, remembered here and in the open ledger entry,
        so a `[?]` line for it is still ours -- after a frontend restart too."""
        if not ep or not ENDPOINT_ID.match(ep):
            return
        with self.lock:
            changed = self.known_endpoint != ep
            self.known_endpoint = ep
            op = self.ledger.get("open")
            if op is not None and op.get("endpoint") != ep:
                op["endpoint"] = ep
                changed = True
        if changed:
            self._save_ledger()

    def _couple(self, why: str, token=None) -> dict:
        sessions = self._probe_sessions()
        if sessions is None:
            # `colab sessions` itself failed (WSL, auth, network): that says nothing
            # about the VM, and a box that is serving must not be written off on it
            return {"ok": False, "code": "probe_failed"}
        ours = next((s for s in sessions if s.get("ours")), None)
        with self.lock:
            stage = self.state["stage"]
            gone = (self.absent_since is not None
                    and time.time() - self.absent_since >= GONE_AFTER_S)
            open_entry = self.ledger.get("open") is not None
        if ours is None:
            # once is not proof (the listing flakes); twice, a minute apart, is
            if gone and stage == "ready":
                self._vm_gone(token)
            elif gone and open_entry and stage in ("idle", "stopped", "failed"):
                # an entry opened for a box that never answered, or one the last
                # frontend left open: the box is gone, so is its billing
                with self.lock:
                    if token is not None and token != self.epoch:
                        return {"ok": False, "code": "not_live"}
                    reason = "stale" if self.stale else "vm_gone"
                cu = self._close_session(reason)
                self._log("[fe] Colab lists no box on %s: closed its ledger entry at %.2f CU"
                          % (self.session, cu))
                self._refresh_money()
            return {"ok": False, "code": "not_live"}
        if why in ("probe", "startup") and time.time() < self.no_adopt_until:
            return {"ok": False, "code": "cooling"}      # a box we just stopped
        if ours.get("name") != self.session:
            # the CLI dropped its record of our box: every `colab download -s` fails
            # until it is back, so put it back first (no ping -- coupling is not use)
            self._keep_alive(ping=False)
        found = self._discover()
        url, key = (found.get("url") or "").rstrip("/"), found.get("key")
        code, ms = 0, None
        if url and key:
            code, _, ms = _http_json(url + "/health", timeout=10)
        if code != 200:
            vm_stage = found.get("stage")
            if url and key and vm_stage == "ready":
                vm_stage = self._vm_stage_now() or vm_stage
            if url and key:
                # A fresh quick-tunnel hostname can fail its first lookup: keep what
                # the VM told us, so the link loop recovers by itself once it
                # resolves, instead of a READY rail that holds no key.
                with self.lock:
                    if self.state["stage"] == "ready" and (token is None or token == self.epoch):
                        self.state["endpoint"].update(base=url + "/v1", key=key, key_pending=False)
            self._note_vm(ours, vm_stage, reachable_url=None, token=token)
            return {"ok": False, "code": "unreachable" if url and key else "not_ready",
                    "stage": vm_stage}
        mcode, models, _ = _http_json(url + "/v1/models", key)
        ids = [m.get("id") for m in (models or {}).get("data") or []] if mcode == 200 else []
        scode, server, _ = _http_json(url + "/v1/status", key, timeout=15)
        server = server if scode == 200 and isinstance(server, dict) else None
        if server is not None and not server.get("recipe") and found.get("recipe"):
            server["recipe"] = found["recipe"]            # serve.sh wrote it down
        if not self._take_charge(ours, url, key, ids, server, ms, why, token):
            return {"ok": False, "code": "superseded"}
        return {"ok": True, "code": "coupled", "url": url}

    def _note_vm(self, ours: dict, vm_stage, reachable_url, token=None) -> None:
        """Our session is live but nothing answers: say what the VM is doing, and
        keep its cost on the rail -- it bills whether or not anything answers."""
        if self._ensure_charge(ours, None, token) is None:
            return                         # a stop or a new job got here first
        with self.lock:
            self.state["vm"] = {"session": ours["name"], "hardware": ours["hardware"],
                                "stage": vm_stage}
            if self.state["stage"] == "ready":
                self.state["link"]["ok"] = False
                self.state["note"] = _note("link.down", stage=vm_stage or "?")
            elif self.state["stage"] in ("idle", "stopped", "failed"):
                self.state["note"] = _note("vm.live", session=ours["name"],
                                           stage=vm_stage or "?")

    def _take_charge(self, ours, url, key, ids, server, ms, why, token=None) -> bool:
        recipe = self._ensure_charge(ours, server, token)
        if recipe is None:
            return False                   # a stop or a new job got here first
        now = time.time()
        with self.lock:
            if token is not None and token != self.epoch:
                return False
            old = self.state["endpoint"].get("base")
            adopted = not self.started_here
            moved = bool(old) and old != url + "/v1"
            self.state.update(
                stage="ready", progress=100.0, selected=recipe["id"],
                live_model=(ids[0] if ids else recipe["model"]), adopted=adopted,
                vm=None, confirm=None,
                endpoint={"base": url + "/v1", "key": key, "key_pending": False},
                link={"ok": True, "latency_ms": round(ms) if ms else None, "fails": 0,
                      "checked_at": now})
            if moved:
                self.state["note"] = _note("link.moved")
            elif adopted:
                self.state["note"] = _note("coupled", session=ours["name"])
            else:
                self.state["note"] = _note("ready")
            if adopted:
                self.state["foot"] = _note("foot.adopted", session=ours["name"],
                                           card=recipe["card"])
            if why != "rediscover":
                self.last_activity = max(self.last_activity, now)   # a fresh idle clock
        self._absorb_server(server, url)
        if why != "rediscover":
            self._balance_soon()
        if moved:
            self._log("[fe] tunnel moved: %s -> %s" % (_root_of(old), url))
        self._log("[fe] coupled to %s (%s)" % (url, "adopted" if adopted else why))
        return True

    def _absorb_server(self, server, root) -> None:
        """/v1/status -> the rail's server truth, the box's resources, idle policy."""
        now = time.time()
        machine = (server or {}).get("machine")
        local = None
        if not machine and root and _is_loopback(root):
            local = _local_machine(self.root)     # the endpoint runs on this PC
        with self.lock:
            if server is not None:
                self.state["server"], self.state["server_at"] = server, now
            if machine:
                self.state.update(machine=machine, machine_at=now, machine_source="server")
            elif local:
                self.state.update(machine=local, machine_at=now, machine_source="local")
            act = (server or {}).get("activity")
            if act is not None:
                self.state["idle_policy"] = "server"
                last = act.get("last_request_at") or 0
                if act.get("in_flight"):
                    last = now
                self.server_last_request = max(self.server_last_request, float(last))
            elif server is not None and self.state["stage"] == "ready":
                # no request log: an adopted box gets no idle stop, ours is judged
                # by this frontend's own traffic
                self.state["idle_policy"] = "off" if self.state.get("adopted") else "frontend"

    def _link_loop(self) -> None:
        """While an endpoint is live: /health every 20 s, /v1/status every 60 s.
        Plain HTTP through the tunnel -- no WSL on this path."""
        last_status = 0.0
        keyless = 0
        while True:
            time.sleep(LINK_EVERY_S)
            with self.lock:
                stage = self.state["stage"]
                base = self.state["endpoint"].get("base")
                key = self.state["endpoint"].get("key")
                busy = self.coupling
            if stage not in ("ready", "attached") or not base or busy:
                continue
            if self.fake and not self.fake_vm:
                continue                          # the rehearsal host is a placeholder
            root = _root_of(base)
            code, _, ms = _http_json(root + "/health", timeout=10)
            now = time.time()
            with self.lock:
                link = self.state["link"]
                was_down = link.get("ok") is False
                link["checked_at"] = now
                if code == 200:
                    link.update(ok=True, latency_ms=round(ms) if ms else None, fails=0)
                    if was_down and (self.state.get("note") or {}).get("k") == "link.down":
                        self.state["note"] = _note("ready")
                else:
                    link.update(ok=False, fails=int(link.get("fails") or 0) + 1)
                fails = link["fails"]
            if code == 200:
                if was_down:
                    self._log("[fe] %s answers again" % root)
                # healthy but keyless: every chat would get a 401 while the idle clock
                # and the keep-alive hold the box. Ask the VM for the key again.
                keyless = keyless + 1 if stage == "ready" and not key and not self.fake else 0
                if keyless == 1 or (keyless and keyless % 6 == 0):
                    self._log("[fe] %s answers, but the rail holds no key -- asking the VM" % root)
                    threading.Thread(target=self.couple, args=("rediscover",), daemon=True).start()
                if now - last_status >= STATUS_EVERY_S:
                    last_status = now
                    scode, server, _ = _http_json(root + "/v1/status", key, timeout=15)
                    if scode == 200 and isinstance(server, dict):
                        self._absorb_server(server, root)
                    elif scode in (404, 405):
                        self._absorb_server(None, root)   # not ours: local probe if loopback
                continue
            # ask the VM after three misses, then once a minute: a restarted serve.sh
            # is back with a new hostname a few minutes later, and the rail should
            # follow it without anyone pressing Reconnect
            if stage == "ready" and (fails == LINK_FAILS_TO_ASK_VM
                                     or (fails > LINK_FAILS_TO_ASK_VM and fails % 3 == 0)):
                with self.lock:
                    self.state["note"] = _note("link.down", stage="?")
                self._log("[fe] %s stopped answering (%d checks) -- asking the VM where it is"
                          % (root, fails))
                threading.Thread(target=self.couple, args=("rediscover",), daemon=True).start()

    # ---- log lines -> stages --------------------------------------------- #

    def _log(self, line: str) -> None:
        with self.lock:
            self.log.append(line.rstrip())

    def _set_stage(self, stage: str, note=None) -> None:
        with self.lock:
            cur = self.state["stage"]
            # a late log line must not revive a job that is stopping or over
            if cur in ("stopping", "stopped", "failed"):
                return
            if cur == "ready" and stage != "ready":
                return
            if (cur in STAGE_ORDER and stage in STAGE_ORDER
                    and STAGE_ORDER.index(stage) < STAGE_ORDER.index(cur)):
                return
            if cur != stage:
                self.stage_started = time.time()
            floor = STAGE_SPAN.get(stage, (100.0, 100.0))[0]
            self.state["stage"] = stage
            self.state["progress"] = max(self.state["progress"], floor)
            if note:
                self.state["note"] = note

    def _absorb(self, raw: str) -> None:
        line = raw.strip()
        if not line:
            return
        self._log(line)
        if "restoring/creating the " in line:
            return self._set_stage("requesting", _note("stage.requesting"))
        m = re.search(r"\[restore\] box: (.+)", line)
        if m:
            return self._set_stage("requesting", _note("box", box=m.group(1)))
        m = re.search(r"\[restore\] ready: session '([^']+)' -> (\S+)", line)
        if m:
            if m.group(1) == self.session:
                self._learn_endpoint(m.group(2))   # before any `colab sessions` has run
            return
        if "uploading the toolkit" in line:
            return self._set_stage("uploading", _note("stage.uploading"))
        if "bootstrapping (runtime" in line:
            return self._set_stage("bootstrapping", _note("stage.bootstrapping"))
        if "waiting up to" in line:
            return self._set_stage("bootstrapping", _note("waiting"))
        if re.search(r"\] READY$", line):
            with self.lock:
                self.state["progress"] = 99.0
                # a fresh idle clock: the idle stop counts from the job's start until
                # now, and a load slower than the idle window (a first pull, a slow
                # mirror) was stopped by the first watch tick after it came up
                self.last_activity = time.time()
                if self.job_recipe:
                    # the rail shows this; it is the model the recipe asked for
                    self.state["live_model"] = self.job_recipe["model"]
                    self.state["selected"] = self.job_recipe["id"]
            return self._set_stage("ready", _note("ready.fetching"))
        m = re.search(r"stage:\s+stage=(\w+)", line)
        if m:
            # serve.sh writes the published address into STATUS -- the quick
            # tunnel's hostname, or PUBLIC_URL for a named tunnel -- so it is read
            # here as well as from the trycloudflare pattern below.
            u = re.search(r"\burl=(https?://\S+)", line)
            if u:
                with self.lock:
                    self.state["endpoint"]["base"] = u.group(1).rstrip("/") + "/v1"
            rail = VM_STAGES.get(m.group(1))
            if rail:
                return self._set_stage(rail, _note("vm", vm=m.group(1)))
            return
        m = re.search(r"gpu_MiB:\s+([\d.]+),\s*([\d.]+)", line)
        if m:
            with self.lock:
                if self.state["stage"] not in ("stopping", "stopped", "failed"):
                    self.state["note"] = _note("vram", used=m.group(1), total=m.group(2))
            return
        m = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line)
        if m:
            with self.lock:
                self.state["endpoint"]["base"] = m.group(1) + "/v1"
            return
        if "!!" in line or "BOOTSTRAP_FAILED" in line:
            with self.lock:
                if self.state["stage"] not in ("stopping", "stopped", "failed"):
                    self.state["note"] = _note("raw", text=line)

    # ---- background loops ------------------------------------------------- #

    def _beat(self) -> None:
        """Time-based floor for the bar: up.sh sleeps 45 s between status lines."""
        while True:
            time.sleep(2.0)
            with self.lock:
                stage = self.state["stage"]
                if stage in ("ready", "failed", "stopped", "idle", "stopping"):
                    return
                floor, ceil = STAGE_SPAN.get(stage, (0.0, 0.0))
                elapsed = time.time() - self.stage_started
                grown = floor + (ceil - floor) * min(
                    1.0, elapsed / STAGE_SECS.get(stage, 120.0))
                self.state["progress"] = round(max(self.state["progress"], grown), 1)
                overdue = time.time() - self.job_started > 75 * 60
            if overdue:
                self._log("[fe] job ran past 75 min -- stopping")
                self.stop("job_timeout")
                return

    def _watch(self) -> None:
        last_probe = time.time()             # _probe_cli does the first one
        last_balance = time.time()
        last_hold = 0.0
        while True:
            time.sleep(5.0)
            self._refresh_money()
            now = time.time()
            if now - last_balance > BALANCE_EVERY_S:
                last_balance = now
                self._balance_soon()
            with self.lock:
                stage = self.state["stage"]
                op = self.ledger.get("open")
                idle = self._idle_now(now) if stage == "ready" and op else None
                self.state["idle_left_s"] = (None if idle is None
                                             else int(max(0, self.idle_stop_min * 60 - idle)))
                holding = stage == "ready" and bool(op) and self._in_use(now)
                self.state["keepalive"]["holding"] = holding
                hold_every = (KEEPALIVE_EVERY_S if self.state["keepalive"].get("at")
                              else KEEPALIVE_FIRST_S)
                in_charge_h = (now - self.charge_started) / 3600.0 if self.charge_started else 0.0
                # Heartbeat: if this process dies with a box up, the next one bills
                # that session up to here plus Colab's idle prune, not up to "now".
                if op and not self.stale and now - op.get("seen", 0) > 60:
                    op["seen"] = now
                    self._save_ledger()
                busy = self.coupling
            if stage == "ready" and op:
                if idle is not None and idle > self.idle_stop_min * 60:
                    self._log("[fe] idle for %d min -- stopping the VM" % int(idle / 60))
                    self.stop("idle")
                elif in_charge_h > self.max_session_h:
                    self._log("[fe] in charge for %.1f h -- stopping the VM" % in_charge_h)
                    self.stop("max_hours")
                elif holding and now - last_hold >= hold_every:
                    last_hold = now
                    threading.Thread(target=self._keep_alive, daemon=True).start()
            elif stage in ("idle", "stopped", "failed") and not busy \
                    and now - last_probe > SESSIONS_EVERY_S:
                last_probe = now
                threading.Thread(target=self.couple, args=("probe",), daemon=True).start()

    def _keep_alive(self, ping: bool = True) -> None:
        """Tell Colab the box is in use, and put back the CLI's record of it if the
        CLI dropped it (scripts/colab_keepalive.py). ping=False only does the
        second: coupling needs the record, and is not use."""
        with self.lock:
            if self.keepalive_busy:
                return
            self.keepalive_busy = True
            endpoint = self.known_endpoint
            stage = self.state["stage"]
        try:
            if self.fake:
                out = "KEEPALIVE ok %s" % (endpoint or "rehearsal-a100-hm-0")
            else:
                args = ([] if ping else ["--no-ping"]) + [self.session]
                if endpoint and ENDPOINT_ID.match(endpoint):
                    args.append(endpoint)
                out = self._wsl_run(["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                                     "$HOME/.local/share/uv/tools/google-colab-cli/bin/python "
                                     "%s/scripts/colab_keepalive.py %s"
                                     % (_wsl_path(self.root),
                                        " ".join(shlex.quote(a) for a in args))], timeout=90)
            # `KEEPALIVE ok <endpoint>[ reattached]` | `KEEPALIVE_GONE` | `KEEPALIVE_ERROR <why>`
            m = re.search(r"^KEEPALIVE( ok|_GONE|_ERROR)\b[ \t]*(.*)$", out, re.M)
            kind = m.group(1).strip() if m else "_ERROR"
            rest = (m.group(2) if m else (out.strip().splitlines() or ["no answer"])[-1]).strip()
            now = time.time()
            with self.lock:
                # a box that stopped while the call was out gets no hold painted on it
                current = self.state["stage"] == stage
                hold = self.state["keepalive"]
                before = (hold.get("ok"), hold.get("err"))
                if kind == "ok":
                    if ping and current:
                        hold.update(ok=True, err=None, at=now)
                elif current:
                    hold.update(ok=False, err="gone" if kind == "_GONE" else rest[:160])
                after = (hold.get("ok"), hold.get("err"))
            if kind == "ok" and rest.split():
                self._learn_endpoint(rest.split()[0])
            if kind == "ok" and rest.endswith(" reattached"):
                self._log("[fe] the CLI had dropped its record of %s; registered it again (%s)"
                          % (self.session, rest.split()[0]))
            if kind == "_GONE":
                self._log("[fe] Colab lists no assignment for %s -- the VM is gone" % self.session)
                if ping and stage == "ready" and current:
                    threading.Thread(target=self.couple, args=("rediscover",), daemon=True).start()
            elif kind == "_ERROR" and after != before:
                self._log("!! keep-alive failed: %s" % rest[:160])
            elif kind == "ok" and ping and current and before[0] is not True:
                self._log("[fe] keep-alive: Colab holds %s while it is in use" % self.session)
        finally:
            with self.lock:
                self.keepalive_busy = False

    def _probe_balance(self) -> None:
        """The account's real compute-unit balance and burn rate, from Colab itself."""
        if self.fake:
            return
        out = self._wsl_run(["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                             "$HOME/.local/share/uv/tools/google-colab-cli/bin/python "
                             "%s/scripts/colab_ccu.py" % _wsl_path(self.root)], timeout=90)
        m = re.search(r"^CCU (\{.*\})\s*$", out, re.M)
        if not m:
            self._log("!! balance probe failed: %s" % (out.strip().splitlines() or ["?"])[-1][:160])
            return
        data = json.loads(m.group(1))
        with self.lock:
            self.state["balance"] = {"cu": data.get("balance"), "rate": data.get("rate_per_hour"),
                                     "assignments": data.get("assignments"),
                                     "checked_at": time.time()}

    def _balance_soon(self) -> None:
        threading.Thread(target=self._probe_balance, daemon=True).start()

    def _probe_sessions(self):
        """What the server says is billing, parsed properly, with ours marked --
        or None when `colab sessions` did not answer, which is not "none"."""
        if self.fake:
            live = bool(self.fake_vm) and not self.fake_vm_down
            sessions = ([{"name": self.session, "endpoint": "rehearsal-a100-hm-0",
                          "hardware": "A100"}] if live else [])
            raw = ("rehearsal: a fake VM serving at %s" % self.fake_vm) if live else "rehearsal: no VM"
            answered = True
        else:
            out = self._wsl_run(["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                                 "$HOME/.local/bin/colab sessions"], timeout=120)
            sessions = _parse_sessions(out)
            raw = "\n".join(out.splitlines()[-6:])
            answered = bool(sessions) or "No active sessions found" in out
        if not answered:
            # keep the last answer on the rail: a count of 0 here would be a lie
            with self.lock:
                self.state["billing"].update(raw=raw, checked_at=time.time(), ok=False)
                first = not self.sessions_failed
                self.sessions_failed = True
            if first:
                self._log("!! `colab sessions` did not answer: %s"
                          % ((raw.strip().splitlines() or ["?"])[-1][:160]))
            return None
        now = time.time()
        named_ep = None
        with self.lock:
            self.sessions_failed = False
            for s in sessions:
                named = s.get("name") == self.session
                # `[?]`: the CLI lost its record of the box, not the box
                s["ours"] = named or (s.get("name") in (None, "?") and bool(s.get("endpoint"))
                                      and s.get("endpoint") == self.known_endpoint)
                if named and s.get("endpoint"):
                    named_ep = s["endpoint"]
            if any(s["ours"] for s in sessions):
                self.absent_since = None
            elif self.absent_since is None:
                self.absent_since = now
            elif now - self.absent_since >= GONE_AFTER_S:
                # missing from two listings a minute apart: gone, not a flake. Only
                # now is the endpoint forgotten -- one miss must leave a `[?]` line
                # for it recognisable as ours
                self.known_endpoint = None
            self.state["billing"] = {"count": len(sessions), "sessions": sessions, "raw": raw,
                                     "checked_at": now, "ok": True}
            changed = self.last_session_count != len(sessions)
            self.last_session_count = len(sessions)
        if named_ep:
            self._learn_endpoint(named_ep)
        if changed and sessions:
            self._log("[fe] server says %d assignment(s) are billing: %s"
                      % (len(sessions), ", ".join("%s (%s)" % (s.get("name"), s.get("hardware"))
                                                  for s in sessions)))
        return sessions

    def _probe_cli(self) -> None:
        if self.fake:
            with self.lock:
                self.state["cli"].update(colab="rehearsal", checked_at=time.time())
        else:
            argv = ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                    "test -x $HOME/.local/bin/colab && echo COLAB_OK || echo COLAB_MISSING"]
            out = self._wsl_run(argv, timeout=120)
            ok = "COLAB_OK" in out
            with self.lock:
                self.state["cli"].update(colab="ok" if ok else "missing", checked_at=time.time())
            if not ok:
                self._log("!! colab CLI not found in WSL %s (~/.local/bin/colab)" % self.distro)
                self._probe_sessions()
                return
        self._balance_soon()
        if self.external:
            self._probe_sessions()
            return
        # A box may already be up (started by hand, or before this process): take
        # it over instead of showing "no card" beside a VM that bills.
        self.couple("startup")
