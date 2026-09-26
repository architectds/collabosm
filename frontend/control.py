#!/usr/bin/env python3
"""The control plane behind the shell's right rail -- the real one.

The shell asks three questions and nothing else:

    status()                     what should the rail draw
    select(recipe, confirm)      start paying for a card
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

Why an idle auto-stop
    A100 High-RAM is 7.52 CU/h and the plan is ~200 CU/month, so 26.6 h. An
    idle session left open overnight is a month of work. Nothing here starts a
    keep-alive: the job is the only thing that keeps the VM alive, and
    `idle_stop_min` (default 20) after the last chat request the VM is stopped
    and the reason is written into the ledger.

Adopt, never duplicate
    Provisioning goes through scripts/up.sh -> scripts/restore.py, which
    re-attaches to an existing assignment rather than creating a second one
    (that is what restore.py exists for: a pruned local session record once cost
    a duplicate VM).

Rehearsal
    `fake=True` swaps the WSL command for frontend/fake_provision.py, which
    emits the same log lines in ~24 s. That is how this file is exercised
    without a card: server.py --fake-provision.
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
        "note": "本仓库实测：prefill 2,806 t/s（gcs 8192 -> 3,882），decode 97.4 t/s",
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
        "note": "配方还没有脚本：40 GB 卡装不下 Flash-Next（要 ~63.6 GiB 常驻显存），这一步是路线图",
    },
]

MEASURED_REFERENCE = {"prefill": 2806.0, "prefill_gcs8192": 3882.0, "decode": 97.4,
                      "where": "本仓库 2026-09-25 在 A100-80G High-RAM 的实测，不是本次会话的数字"}

# --------------------------------------------------------------------------- #
# the provisioning stages, and how the rail should pace them                   #
# --------------------------------------------------------------------------- #

STAGE_LABEL = {"idle": "无卡无模型", "requesting": "申请中", "uploading": "上传工具包",
               "bootstrapping": "安装与下载", "loading": "写入显存", "ready": "已就绪",
               "stopping": "正在停机", "stopped": "已停机", "failed": "失败"}

STAGE_NOTE = {
    "requesting": "向 Colab 申请实例并校验形状；抽到 40 GB 会自动退回（约 0.13 CU）",
    "uploading": "上传 api_server / bootstrap / serve 到 /content",
    "bootstrapping": "安装 ExLlamaV3 运行时，并从 Hugging Face 拉 ~100 GiB 权重",
    "loading": "把 ~63.6 GiB 写进显存，建 KV 页表与 n-gram 表",
}

# (progress floor, progress ceiling) per stage, and the seconds it usually takes
STAGE_SPAN = {"requesting": (0.0, 8.0), "uploading": (8.0, 14.0),
              "bootstrapping": (14.0, 60.0), "loading": (60.0, 98.0)}
STAGE_SECS = {"requesting": 120.0, "uploading": 120.0,
              "bootstrapping": 330.0, "loading": 300.0}

VM_STAGE_NOTE = {"probing": "校验实例形状", "runtime": "安装 ExLlamaV3 运行时",
                 "weights": "从 Hugging Face 拉权重", "env": "写入启动环境",
                 "bootstrapped": "引导完成，准备起服务", "loading": "载入模型到显存",
                 "ready": "服务已健康"}

# stages only move forward, so a relayed `stage=probing` line cannot drag the
# rail back from 写入显存 to 安装与下载 while progress stays at 60%
STAGE_ORDER = ["idle", "requesting", "uploading", "bootstrapping", "loading", "ready"]

EXIT_NOTE = {
    1: "上传或引导失败，看下面几行日志",
    3: "服务端实例数已到上限，先停掉别的会话",
    4: "assign 请求被拒（账号可能没有 A100 资格）",
    5: "读不到 assignments，CLI 可能需要重新登录",
    6: "抽到 40 GB 标准卡，已自动退回并停机（约 0.13 CU）；再点一次重抽",
    7: "等待端点超时（默认 50 分钟）",
}


def _sanitize(text: str) -> str:
    return text.replace("\x00", "").replace("\r", "")


def _wsl_path(win_path: str) -> str:
    """E:\\models\\collabosm -> /mnt/e/models/collabosm (WSL sees the same files)."""
    p = os.path.abspath(win_path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    return "/mnt/%s/%s" % (m.group(1).lower(), m.group(2)) if m else p


class Control:
    """status() / select() / stop() -- the whole contract the shell needs."""

    def __init__(self, root: str, *, session: str = "collabosm", distro: str = "Ubuntu",
                 budget_cu: float = 200.0, idle_stop_min: int = 20,
                 max_session_h: float = 6.0, fake: bool = False,
                 state_dir: str | None = None):
        self.root = os.path.abspath(root)
        self.session = session
        self.distro = distro
        self.fake = fake
        self.budget_cu = float(budget_cu)
        self.idle_stop_min = int(idle_stop_min)
        self.max_session_h = float(max_session_h)
        self.state_dir = state_dir or os.path.join(os.path.expanduser("~"), ".collabosm")
        self.ledger_path = os.path.join(self.state_dir, "ledger.json")

        self.lock = threading.RLock()
        self.proc = None
        self.job_recipe = None
        self.job_started = 0.0
        self.stage_started = 0.0
        self.last_activity = time.time()
        self.stop_wanted = False
        self.log = collections.deque(maxlen=40)

        self.ledger = self._load_ledger()
        self.state = {
            "stage": "idle",
            "stage_label": STAGE_LABEL["idle"],
            "progress": 0.0,
            "progress_note": "还没选配方。选一个就开始计费，所以先把数字给你看。",
            "selected": None,
            "live_model": None,
            "metrics": None,
            "recipes": RECIPES,
            "footnote": "控制面：WSL -> colab CLI -> restore.py -> up.sh",
            "budget_cu": self.budget_cu,
            "cu_used": 0.0,
            "cu_left": self.budget_cu,
            "session_hours": 0.0,
            "idle_stop_min": self.idle_stop_min,
            "idle_left_s": None,
            "max_session_h": self.max_session_h,
            "endpoint": {"base": None, "key": None, "key_pending": False},
            "billing": {"count": None, "raw": "", "checked_at": None},
            "cli": {"wsl": distro, "colab": "unknown", "root": _wsl_path(self.root),
                    "command": None, "checked_at": None},
            "confirm": None,
            "measured": MEASURED_REFERENCE,
            "fake": fake,
            "outgoing_model": None,
        }
        self._refresh_money()
        threading.Thread(target=self._watch, daemon=True).start()
        threading.Thread(target=self._probe_cli, daemon=True).start()

    # ---- money ---------------------------------------------------------- #

    def _load_ledger(self) -> dict:
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
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            tmp = self.ledger_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.ledger, fh, indent=1)
            os.replace(tmp, self.ledger_path)
        except Exception as exc:
            self._log("!! ledger write failed: %r" % exc)

    def _open_session(self, recipe: dict) -> None:
        self.ledger["open"] = {"start": time.time(), "recipe": recipe["id"],
                               "cu_per_hour": recipe["cu_per_hour"]}
        self._save_ledger()

    def _close_session(self, reason: str) -> float:
        op = self.ledger.get("open")
        if not op:
            return 0.0
        hours = max(0.0, time.time() - op["start"]) / 3600.0
        cu = hours * op["cu_per_hour"]
        self.ledger["closed_cu"] = round(self.ledger.get("closed_cu", 0.0) + cu, 4)
        self.ledger["entries"].append({"start": op["start"], "end": time.time(),
                                       "minutes": round(hours * 60, 1), "cu": round(cu, 3),
                                       "recipe": op["recipe"],
                                       "cu_per_hour": op["cu_per_hour"], "reason": reason})
        self.ledger["entries"] = self.ledger["entries"][-200:]
        self.ledger["open"] = None
        self._save_ledger()
        return cu

    def _refresh_money(self) -> None:
        op = self.ledger.get("open")
        used = float(self.ledger.get("closed_cu", 0.0))
        hours = 0.0
        if op:
            hours = max(0.0, time.time() - op["start"]) / 3600.0
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
        st["log_tail"] = list(self.log)[-12:]
        return st

    def select(self, recipe_id: str, confirm: bool = False) -> dict:
        recipe = next((r for r in RECIPES if r["id"] == recipe_id), None)
        if recipe is None:
            return {"ok": False, "code": "unknown_recipe", "message": "没有这个配方"}
        with self.lock:
            stage = self.state["stage"]
            if stage in ("requesting", "uploading", "bootstrapping", "loading", "stopping"):
                return {"ok": False, "code": "busy",
                        "message": "正在忙：%s" % self.state["progress_note"]}
            if stage == "ready":
                if self.state["live_model"] == recipe["model"]:
                    return {"ok": True, "code": "already", "message": "这个配方已经就绪"}
                return {"ok": False, "code": "stop_first",
                        "message": "已经有一个实例在跑。先停机再换配方（换配方就是一次重新载入）"}
            if not recipe["verified"]:
                return {"ok": False, "code": "unverified", "message": recipe["note"]}
            left = self.state["cu_left"]
            need = round(recipe["cu_per_hour"] * ((recipe["eta_min"] + 5) / 60.0), 2)
            if left < max(3.0, need):
                return {"ok": False, "code": "budget",
                        "message": "本月 CU 只剩 %.1f，低于这次启动的估算 %.1f CU" % (left, need)}
            warn = {"recipe": recipe["id"], "card": recipe["card"], "model": recipe["model"],
                    "cu_per_hour": recipe["cu_per_hour"],
                    "usd_per_hour": round(recipe["cu_per_hour"] * 0.0999, 2),
                    "eta_min": recipe["eta_min"],
                    "cu_estimate": need,
                    "cu_left": left,
                    "cu_left_after": round(left - need, 2),
                    "idle_stop_min": self.idle_stop_min,
                    "max_session_h": self.max_session_h,
                    "headline": "点“开始”就开始计费：%s，%.2f CU/h，预计 %d 分钟"
                                % (recipe["card"], recipe["cu_per_hour"], recipe["eta_min"]),
                    "lines": [
                        "载入本身就值 ~%.1f CU（约 %d 分钟）；换一次配方就是一次载入。"
                        % (need, recipe["eta_min"]),
                        "空闲 %d 分钟自动停机，连续最长 %g 小时。" % (self.idle_stop_min,
                                                                    self.max_session_h),
                        "本月预算 %.0f CU，现在剩 %.1f；启动后预计剩 %.1f。"
                        % (self.budget_cu, left, left - need),
                    ]}
            if not confirm:
                self.state["confirm"] = warn
                return {"ok": True, "code": "confirm_required", "warning": warn}
        self._start_job(recipe)
        return {"ok": True, "code": "started", "recipe": recipe["id"]}

    def stop(self, reason: str = "manual") -> dict:
        with self.lock:
            stage = self.state["stage"]
            if stage in ("idle", "stopped", "failed"):
                return {"ok": True, "code": "nothing_to_stop",
                        "message": "现在没有在计费的实例"}
            running_job = stage in ("requesting", "uploading", "bootstrapping", "loading")
            self.state["stage"] = "stopping"
            self.state["stage_label"] = STAGE_LABEL["stopping"]
            self.state["progress_note"] = "正在停机（%s）" % reason
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
        ep = self.state["endpoint"]
        base = ep.get("base")
        return base if self.state["stage"] == "ready" and base else None

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
            self.state.update(stage="requesting", stage_label=STAGE_LABEL["requesting"],
                              progress=0.0, progress_note=STAGE_NOTE["requesting"],
                              selected=recipe["id"], live_model=None, metrics=None,
                              confirm=None,
                              endpoint={"base": None, "key": None, "key_pending": False},
                              footnote="计费已开始：%s · %.2f CU/h"
                                       % (recipe["card"], recipe["cu_per_hour"]))
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
            self._fail("启动不了命令：%r" % exc)
            return
        self.proc = proc
        try:
            for raw in proc.stdout:
                self._absorb(_sanitize(raw))
            rc = proc.wait()
        except Exception as exc:
            self.proc = None
            self._fail("读日志失败：%r" % exc)
            return
        self.proc = None

        if self.stop_wanted:
            return
        if rc == 0 and self.state["stage"] == "ready":
            self._after_ready(recipe)
            return
        if rc == 0:
            self._fail("up.sh 正常退出但没等到 READY（看日志）")
            return
        self._fail("up.sh 退出码 %s — %s" % (rc, EXIT_NOTE.get(rc, "看日志")))

    def _after_ready(self, recipe: dict) -> None:
        """The tunnel is a quick tunnel: pull the key once so the proxy can inject it."""
        if self.fake:
            with self.lock:
                self.state["endpoint"]["key"] = "sk-collabosm-rehearsal"
                self.state["endpoint"]["key_pending"] = False
                self.state["metrics"] = MEASURED_REFERENCE
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
            self.state["metrics"] = MEASURED_REFERENCE
        self._log("[fe] api key loaded" if key else
                  "!! could not read /content/api-key.txt -- chat will 503")

    def _do_stop(self, reason: str, cu: float) -> None:
        if self.fake:
            out = "[down] rehearsal: no VM was ever created"
        else:
            argv = ["wsl.exe", "-d", self.distro, "--", "bash", "-lc",
                    "cd %s && COLAB=$HOME/.local/bin/colab bash scripts/down.sh"
                    % shlex.quote(_wsl_path(self.root))]
            out = self._wsl_run(argv, timeout=300)
        for line in out.splitlines()[-6:]:
            self._log(line)
        with self.lock:
            self.state.update(stage="stopped", stage_label=STAGE_LABEL["stopped"],
                              progress=0.0, live_model=None,
                              endpoint={"base": None, "key": None, "key_pending": False},
                              progress_note="已停机（%s）· 这次约 %.2f CU" % (reason, cu),
                              footnote="停机原因：%s。控制面在，VM 不在。" % reason)
        self._refresh_money()

    def _fail(self, message: str) -> None:
        cu = self._close_session("failed")
        with self.lock:
            self.state.update(stage="failed", stage_label=STAGE_LABEL["failed"],
                              progress_note=message, live_model=None, metrics=None,
                              endpoint={"base": None, "key": None, "key_pending": False},
                              footnote="失败 · 这次约花了 %.2f CU" % cu)
        self._log("!! " + message)
        self._refresh_money()

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
            if self.state["stage"] in ("ready", "failed", "stopped") and stage != "ready":
                return
            cur, want = self.state["stage"], stage
            if (cur in STAGE_ORDER and want in STAGE_ORDER
                    and STAGE_ORDER.index(want) < STAGE_ORDER.index(cur)):
                return
            if STAGE_LABEL.get(self.state["stage"]) != STAGE_LABEL.get(stage):
                self.stage_started = time.time()
            floor = STAGE_SPAN.get(stage, (100.0, 100.0))[0]
            self.state["stage"] = stage
            self.state["stage_label"] = STAGE_LABEL.get(stage, stage)
            self.state["progress"] = max(self.state["progress"], floor)
            if note:
                self.state["progress_note"] = note

    def _absorb(self, raw: str) -> None:
        line = raw.strip()
        if not line:
            return
        self._log(line)
        if "restoring/creating the A100" in line:
            return self._set_stage("requesting", STAGE_NOTE["requesting"])
        m = re.search(r"\[restore\] box: (.+)", line)
        if m:
            return self._set_stage("requesting", "实例形状校验：%s" % m.group(1))
        if "uploading the toolkit" in line:
            return self._set_stage("uploading", STAGE_NOTE["uploading"])
        if "bootstrapping (runtime" in line:
            return self._set_stage("bootstrapping", STAGE_NOTE["bootstrapping"])
        if "waiting up to" in line:
            return self._set_stage("bootstrapping", "已在等待循环里，端点一健康就切换")
        if re.search(r"\] READY$", line):
            with self.lock:
                self.state["progress"] = 99.0
                self.state["progress_note"] = "服务已健康，正在取隧道地址与密钥"
                if self.job_recipe:
                    # the rail shows this; it is the model the recipe asked for
                    self.state["live_model"] = self.job_recipe["model"]
                    self.state["selected"] = self.job_recipe["id"]
            return self._set_stage("ready", "服务已健康")
        m = re.search(r"stage:\s+stage=(\w+)", line)
        if m:
            note = VM_STAGE_NOTE.get(m.group(1))
            if note:
                stage = "loading" if m.group(1) in ("loading", "ready") else "bootstrapping"
                return self._set_stage(stage, "VM：%s" % note)
        m = re.search(r"gpu_MiB:\s+([\d.]+),\s*([\d.]+)", line)
        if m:
            with self.lock:
                self.state["progress_note"] = "显存 %s / %s MiB" % (m.group(1), m.group(2))
            return
        m = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line)
        if m:
            with self.lock:
                self.state["endpoint"]["base"] = m.group(1) + "/v1"
            return
        if line.startswith("!!") or "BOOTSTRAP_FAILED" in line or "bootstrap reported failure" in line:
            with self.lock:
                self.state["progress_note"] = line

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
            if stage == "ready":
                if idle > self.idle_stop_min * 60:
                    self._log("[fe] idle for %d min -- stopping the VM" % int(idle / 60))
                    self.stop("idle")
                elif hours > self.max_session_h:
                    self._log("[fe] session ran %g h -- stopping the VM" % hours)
                    self.stop("max_hours")
            elif stage in ("idle", "stopped") and time.time() - last_probe > 300:
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
