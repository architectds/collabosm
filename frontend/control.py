#!/usr/bin/env python3
"""The control plane behind the shell's right rail -- the real one.

The shell asks four questions and nothing else:

    status()                     what should the rail draw
    select(recipe, confirm)      start paying for a card
    cancel()                     forget a confirmation that was never given
    stop(reason)                 stop paying for a card

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

Why an idle auto-stop, and a stop on failure
    A100 High-RAM is 7.52 CU/h and the plan is ~200 CU/month, so 26.6 h. An
    idle session left open overnight is a month of work. Nothing here starts a
    keep-alive: the job is the only thing that keeps the VM alive, and
    `idle_stop_min` (default 20) after the last chat request the VM is stopped
    and the reason is written into the ledger. A job that fails after `assign`
    is stopped the same way: up.sh stops nothing on its way out, and a failed
    run whose VM keeps billing is worse than no run at all.

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
    never written.
"""
from __future__ import annotations

import collections
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

# --------------------------------------------------------------------------- #
# recipes                                                                     #
# --------------------------------------------------------------------------- #

RECIPES = [
    {
        "id": "a100-80g/qwen38-fn",
        "card": "A100-80G High-RAM",
        "model": "Qwen3.8-Flash-Next",
        "quant": "EXL3 4.05bpw + MTP",
        "cu_per_hour": 7.52,
        "vram_gb": 80,
        "ram_gb": 167,
        "eta_min": 11,
        "verified": True,
        "env": {"CACHE_SIZE": 500224, "CACHE_QUANT": 4, "CPU_CACHE_GB": 32,
                "RECURRENT_CACHE_GB": 24, "NDT": 4, "GCS": 8192},
        # Data carries its own translations, so a new recipe needs no shell change.
        "note": {
            "en": "Measured in this repo: prefill 2,806 t/s (3,882 at gcs 8192), decode 97.4 t/s",
            "zh": "本仓库实测：prefill 2,806 t/s（gcs 8192 -> 3,882），decode 97.4 t/s",
            "ja": "本リポジトリ実測：prefill 2,806 t/s（gcs 8192 で 3,882）、decode 97.4 t/s",
        },
    },
    {
        "id": "a100-40g/qwen38-27b",
        "card": "A100-40G",
        "model": "Qwen3.8-27B",
        "quant": "EXL3 3.5bpw + MTP",
        "cu_per_hour": 5.37,
        "vram_gb": 40,
        "ram_gb": 83,
        "eta_min": 6,
        "verified": False,
        "env": {"CACHE_SIZE": 262144, "CACHE_QUANT": 4, "CPU_CACHE_GB": 0,
                "RECURRENT_CACHE_GB": 4, "NDT": 4, "GCS": 4096},
        "note": {
            "en": "No script yet: a 40 GB card cannot hold Flash-Next (~63.6 GiB must stay "
                  "in VRAM). This one is roadmap.",
            "zh": "配方还没有脚本：40 GB 卡装不下 Flash-Next（要 ~63.6 GiB 常驻显存），这一步是路线图",
            "ja": "スクリプト未整備：40 GB カードには Flash-Next が載りません（約 63.6 GiB を"
                  " VRAM に常駐させる必要）。ロードマップ項目です",
        },
    },
]

# Reference numbers, not this session's: the shell labels them as such.
MEASURED_REFERENCE = {"prefill": 2806.0, "prefill_gcs8192": 3882.0, "decode": 97.4,
                      "measured_on": "2026-09-25", "card": "A100-80G High-RAM"}

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

# A ledger session no frontend closed is billed up to the last heartbeat plus this:
# Colab idle-prunes an unattended VM after roughly 90 minutes (docs/RUNBOOK.md).
STALE_GRACE_S = 90 * 60

NO_ENDPOINT = {"base": None, "key": None, "key_pending": False}


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


class Control:
    """status() / select() / cancel() / stop() -- the whole contract the shell needs."""

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

        self.ledger = self._load_ledger()
        # An open session at startup was opened by a frontend that is gone: it
        # crashed or was closed with a box up, and that VM may still be billing.
        self.stale = bool(self.ledger.get("open")) and not self.external
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
            "budget_cu": self.budget_cu,
            "cu_used": 0.0,
            "cu_left": self.budget_cu,
            "session_hours": 0.0,
            "idle_stop_min": self.idle_stop_min,
            "idle_left_s": None,
            "max_session_h": self.max_session_h,
            "endpoint": dict(NO_ENDPOINT),
            "billing": {"count": None, "raw": "", "checked_at": None},
            "cli": {"wsl": distro, "colab": "unknown", "root": _wsl_path(self.root),
                    "command": None, "checked_at": None},
            "confirm": None,
            "measured": MEASURED_REFERENCE,
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
        threading.Thread(target=self._probe_cli, daemon=True).start()

    # ---- money ---------------------------------------------------------- #

    def _load_ledger(self) -> dict:
        if not self.persist_ledger:
            return {"closed_cu": 0.0, "entries": [], "open": None}
        try:
            with open(self.ledger_path) as fh:
                led = json.load(fh)
            led.setdefault("closed_cu", 0.0)
            led.setdefault("entries", [])
            led.setdefault("open", None)
            return led
        except Exception:
            return {"closed_cu": 0.0, "entries": [], "open": None}

    def _save_ledger(self) -> None:
        if not self.persist_ledger:
            return
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            tmp = self.ledger_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.ledger, fh, indent=1)
            os.replace(tmp, self.ledger_path)
        except Exception as exc:
            self._log("!! ledger write failed: %r" % exc)

    def _open_session(self, recipe: dict) -> None:
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
        op = self.ledger.get("open")
        if not op:
            return 0.0
        end = self._billed_until(op)
        hours = max(0.0, end - op["start"]) / 3600.0
        cu = hours * op["cu_per_hour"]
        self.ledger["closed_cu"] = round(self.ledger.get("closed_cu", 0.0) + cu, 4)
        self.ledger["entries"].append({"start": op["start"], "end": end,
                                       "minutes": round(hours * 60, 1), "cu": round(cu, 3),
                                       "recipe": op["recipe"],
                                       "cu_per_hour": op["cu_per_hour"], "reason": reason})
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

    def status(self) -> dict:
        self._refresh_money()
        with self.lock:
            # the idle countdown is read here too, not only in the watcher, so a
            # freshly-ready box does not show "—" for up to one watch tick
            if self.state["stage"] == "ready" and self.ledger.get("open"):
                idle = time.time() - self.last_activity
                self.state["idle_left_s"] = int(max(0, self.idle_stop_min * 60 - idle))
            st = json.loads(json.dumps(self.state))
            st["ledger_open"] = self.ledger.get("open") is not None
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
            if stage in ("requesting", "uploading", "bootstrapping", "loading", "stopping"):
                return {"ok": False, "code": "busy", "stage": stage}
            if stage == "ready":
                if self.state["live_model"] == recipe["model"]:
                    return {"ok": True, "code": "already"}
                return {"ok": False, "code": "stop_first"}
            if self.ledger.get("open"):
                # a session the last frontend never closed: stop (and account for)
                # it before a new one opens on top of it
                return {"ok": False, "code": "stale_ledger"}
            if not recipe["verified"]:
                return {"ok": False, "code": "unverified", "recipe": recipe["id"]}
            left = self.state["cu_left"]
            need = round(recipe["cu_per_hour"] * ((recipe["eta_min"] + 5) / 60.0), 2)
            if left < max(3.0, need):
                return {"ok": False, "code": "budget", "cu_left": round(left, 1),
                        "cu_need": round(need, 1)}
            warn = {"recipe": recipe["id"], "card": recipe["card"], "model": recipe["model"],
                    "cu_per_hour": recipe["cu_per_hour"],
                    "usd_per_hour": round(recipe["cu_per_hour"] * 0.0999, 2),
                    "eta_min": recipe["eta_min"],
                    "cu_estimate": need,
                    "budget_cu": self.budget_cu,
                    "cu_left": left,
                    "cu_left_after": round(left - need, 2),
                    "idle_stop_min": self.idle_stop_min,
                    "max_session_h": self.max_session_h}
            if not confirm:
                self.state["confirm"] = warn
                return {"ok": True, "code": "confirm_required", "warning": warn}
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
                                  endpoint=dict(NO_ENDPOINT), note=_note("detached"),
                                  foot=_note("foot.control"))
                return {"ok": True, "code": "detached"}
            if stage == "stopping":
                return {"ok": True, "code": "already_stopping"}
            stale = self.stale and self.ledger.get("open") is not None
            if stage in ("idle", "stopped") and not stale:
                return {"ok": True, "code": "nothing_to_stop"}
            # `failed` gets here on purpose: _fail() already ran down.sh, and this
            # is the manual retry for when the billing probe still shows a VM.
            running_job = stage in ("requesting", "uploading", "bootstrapping", "loading")
            if stale:
                reason = "stale"
            self.state["stage"] = "stopping"
            self.state["note"] = _note("stopping", reason=reason)
            self.stop_wanted = True
            proc = self.proc
        cu = self._close_session(reason)
        if running_job and proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        threading.Thread(target=self._do_stop, args=(reason, cu), daemon=True).start()
        return {"ok": True, "code": "stopping", "reason": reason, "cu": round(cu, 3)}

    # ---- helpers the HTTP layer needs ------------------------------------ #

    def backend_base(self):
        if self.fake:
            # The rehearsal endpoint is a placeholder host, so keep talking to
            # --backend (the loopback stub) instead of trying to resolve it.
            return None
        if self.external and self.state["stage"] == "attached":
            return _root_of(self.external)
        ep = self.state["endpoint"]
        base = ep.get("base")
        return _root_of(base) if self.state["stage"] == "ready" and base else None

    def api_key(self):
        return self.state["endpoint"].get("key")

    def note_outgoing_model(self, model) -> None:
        with self.lock:
            self.state["outgoing_model"] = model
            self.last_activity = time.time()

    def touch(self) -> None:
        """Any chat traffic counts as activity for the idle auto-stop."""
        self.last_activity = time.time()

    # ---- provisioning ---------------------------------------------------- #

    def _start_job(self, recipe: dict) -> None:
        with self.lock:
            self.job_recipe = recipe
            self.job_started = time.time()
            self.stage_started = time.time()
            self.stop_wanted = False
            self.last_activity = time.time()
            self.state.update(stage="requesting", progress=0.0,
                              note=_note("stage.requesting"),
                              selected=recipe["id"], live_model=None, metrics=None,
                              confirm=None, endpoint=dict(NO_ENDPOINT),
                              foot=_note("foot.billing", card=recipe["card"],
                                         cuph=recipe["cu_per_hour"]))
        self._open_session(recipe)
        threading.Thread(target=self._beat, daemon=True).start()
        threading.Thread(target=self._run, args=(recipe,), daemon=True).start()

    def _wsl_command(self, recipe: dict) -> list:
        exports = {"SESSION": self.session, "COLAB": "$HOME/.local/bin/colab"}
        exports.update({k: str(v) for k, v in recipe["env"].items()})
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
        """The tunnel is a quick tunnel: pull the key once so the proxy can inject it."""
        if self.fake:
            with self.lock:
                self.state["endpoint"]["key"] = "sk-collabosm-rehearsal"
                self.state["endpoint"]["key_pending"] = False
                self.state["note"] = _note("ready")
            return
        with self.lock:
            self.state["endpoint"]["key_pending"] = True
        argv = ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                "cd %s && $HOME/.local/bin/colab exec -s %s --timeout 60 -f scripts/api_key.py"
                % (shlex.quote(_wsl_path(self.root)), shlex.quote(self.session))]
        out = self._wsl_run(argv, timeout=180)
        key = None
        for line in out.splitlines():
            if line.startswith("APIKEY "):
                key = line[len("APIKEY "):].strip()
        with self.lock:
            self.state["endpoint"]["key"] = key
            self.state["endpoint"]["key_pending"] = False
            self.state["note"] = _note("ready")
            base = self.state["endpoint"]["base"]
        self._log("[fe] api key loaded" if key else
                  "!! could not read /content/api-key.txt -- chat will 503")
        if not base:
            self._log("!! READY but no published URL was seen -- chat will 503")

    def _run_down(self) -> None:
        if self.fake:
            out = "[down] rehearsal: no VM was ever created"
        else:
            argv = ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                    "cd %s && COLAB=$HOME/.local/bin/colab bash scripts/down.sh"
                    % shlex.quote(_wsl_path(self.root))]
            out = self._wsl_run(argv, timeout=300)
        for line in out.splitlines()[-6:]:
            self._log(line)

    def _do_stop(self, reason: str, cu: float) -> None:
        self._run_down()
        with self.lock:
            self.state.update(stage="stopped", progress=0.0, live_model=None,
                              endpoint=dict(NO_ENDPOINT),
                              note=_note("stopped", reason=reason, cu=round(cu, 2)),
                              foot=_note("foot.stopped", reason=reason))
        self._refresh_money()
        self._probe_sessions()

    def _fail(self, note: dict, why: str, vm_possible: bool = True) -> None:
        cu = self._close_session("failed")
        note["a"]["autostop"] = "running" if vm_possible else "none"
        with self.lock:
            self.state.update(stage="failed", note=note, live_model=None, metrics=None,
                              endpoint=dict(NO_ENDPOINT),
                              foot=_note("foot.failed", cu=round(cu, 2)))
        self._log("!! " + why)
        self._refresh_money()
        if not vm_possible:
            return
        # up.sh stops nothing on its way out, so a failure after `assign` leaves a
        # VM billing behind a ledger entry that says it closed -- and the rail used
        # to refuse to stop a failed job at all.
        self._run_down()
        with self.lock:
            if self.state["stage"] == "failed":
                self.state["note"]["a"]["autostop"] = "done"
        self._probe_sessions()

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
        if "restoring/creating the A100" in line:
            return self._set_stage("requesting", _note("stage.requesting"))
        m = re.search(r"\[restore\] box: (.+)", line)
        if m:
            return self._set_stage("requesting", _note("box", box=m.group(1)))
        if "uploading the toolkit" in line:
            return self._set_stage("uploading", _note("stage.uploading"))
        if "bootstrapping (runtime" in line:
            return self._set_stage("bootstrapping", _note("stage.bootstrapping"))
        if "waiting up to" in line:
            return self._set_stage("bootstrapping", _note("waiting"))
        if re.search(r"\] READY$", line):
            with self.lock:
                self.state["progress"] = 99.0
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
        last_probe = 0.0
        while True:
            time.sleep(5.0)
            self._refresh_money()
            with self.lock:
                stage = self.state["stage"]
                op = self.ledger.get("open")
                if stage == "ready" and op:
                    idle = time.time() - self.last_activity
                    self.state["idle_left_s"] = int(max(0, self.idle_stop_min * 60 - idle))
                else:
                    self.state["idle_left_s"] = None
                    idle = 0.0
                hours = self.state["session_hours"]
                # Heartbeat: if this process dies with a box up, the next one bills
                # that session up to here plus Colab's idle prune, not up to "now".
                if op and not self.stale and time.time() - op.get("seen", 0) > 60:
                    op["seen"] = time.time()
                    self._save_ledger()
            if stage == "ready":
                if idle > self.idle_stop_min * 60:
                    self._log("[fe] idle for %d min -- stopping the VM" % int(idle / 60))
                    self.stop("idle")
                elif hours > self.max_session_h:
                    self._log("[fe] session ran %g h -- stopping the VM" % hours)
                    self.stop("max_hours")
            elif stage in ("idle", "stopped", "failed") and time.time() - last_probe > 300:
                last_probe = time.time()
                threading.Thread(target=self._probe_sessions, daemon=True).start()

    def _probe_sessions(self) -> None:
        if self.fake:
            with self.lock:
                self.state["billing"] = {"count": 0, "raw": "rehearsal: no VM",
                                         "checked_at": time.time()}
            return
        argv = ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                "$HOME/.local/bin/colab sessions"]
        out = self._wsl_run(argv, timeout=120)
        count = len(re.findall(r"^\s*(?:A100|T4|L4|H100|G4)\b", out, re.M))
        with self.lock:
            self.state["billing"] = {"count": count, "raw": "\n".join(out.splitlines()[-6:]),
                                     "checked_at": time.time()}
        if count:
            self._log("[fe] server says %d assignment(s) are still billing" % count)

    def _probe_cli(self) -> None:
        if self.fake:
            with self.lock:
                self.state["cli"].update(colab="rehearsal", checked_at=time.time())
            self._probe_sessions()
            return
        argv = ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                "test -x $HOME/.local/bin/colab && echo COLAB_OK || echo COLAB_MISSING"]
        out = self._wsl_run(argv, timeout=120)
        ok = "COLAB_OK" in out
        with self.lock:
            self.state["cli"].update(colab="ok" if ok else "missing", checked_at=time.time())
        if not ok:
            self._log("!! colab CLI not found in WSL %s (~/.local/bin/colab)" % self.distro)
        self._probe_sessions()
