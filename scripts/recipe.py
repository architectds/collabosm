#!/usr/bin/env python3
"""The recipe registry (recipes.json): which GPU runs which model, and how.

A recipe is a GPU from `gpus`, a model from `models`, and the launch settings for
the pair. Everything that starts a box reads it from here -- scripts/up.sh
(`RECIPE=<id> bash scripts/up.sh`) and the frontend's list of cards -- so a new
pairing is a new entry in recipes.json, not a new script.

Every number carries its evidence: `facts[<name>] = {"v": ..., "measured": bool,
"src": "..."}`, and a recipe's `status` says how far it has been taken:

    verified     run end to end on that card; its numbers are measured
    unmeasured   runnable launch path exists, but the full recipe is not end-to-end verified;
                 individual model-only facts may still be measured
    placeholder  listed so the choice is visible; refused until it is written

    python3 scripts/recipe.py list
    python3 scripts/recipe.py show  a100-80g/qwen38-fn
    python3 scripts/recipe.py env   a100-80g/qwen38-fn    # what up.sh evals
    python3 scripts/recipe.py vmenv a100-80g/qwen38-fn    # the file the VM sources
    python3 scripts/recipe.py check                       # validate recipes.json

`env` prints `[ -n "${VAR+x}" ] || VAR=value; export VAR`, so a variable already
set by the caller wins: `CACHE_SIZE=262144 bash scripts/up.sh` still overrides the
recipe for one run. `vmenv` then writes the effective values, overrides included.

Exit codes: 0 ok | 2 unknown recipe or a broken registry | 3 placeholder (not runnable)
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.environ.get("RECIPES_FILE") or os.path.join(HERE, os.pardir, "recipes.json")
STATUSES = ("verified", "unmeasured", "placeholder")
# what serves the model on the box: our ExLlamaV3 API, Strata, or Project Maya
ENGINES = ("exl3", "strata", "maya")
DEFAULT = "a100-40g/qwen38-27b"
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# What the VM keeps besides the recipe's own keys, when the caller sets them: they are
# read by bootstrap.sh, serve.sh and api_server.py, and a setting that stops at the
# laptop (`THINKING_DEFAULT=1 bash scripts/up.sh`) silently does nothing.
VM_EXTRA = ("RUNTIME", "PORT", "TUNNEL_TOKEN", "PUBLIC_URL", "THINKING_DEFAULT",
            "MAX_NEW_TOKENS", "IMAGE_URLS", "MAX_IMAGE_BYTES", "MAX_IMAGES",
            "HF_HUB_DISABLE_XET")


def load(path: str = REGISTRY) -> dict:
    with open(path, encoding="utf-8") as fh:
        reg = json.load(fh)
    problems = validate(reg)
    if problems:
        raise ValueError("%s: %s" % (os.path.basename(path), "; ".join(problems)))
    return reg


def validate(reg: dict) -> list:
    """Everything a launch depends on, checked before anything is billed."""
    out = []
    gpus = {g.get("id"): g for g in reg.get("gpus") or []}
    models = {m.get("id"): m for m in reg.get("models") or []}
    for g in gpus.values():
        for k in ("card", "accelerator", "shape", "vram_gb", "min_vram_gib", "cu_per_hour"):
            if g.get(k) in (None, ""):
                out.append("gpu %s: no %s" % (g.get("id"), k))
        if g.get("shape") not in ("hm", "st"):
            out.append("gpu %s: shape must be hm or st" % g.get("id"))
    dirs = {}
    for m in models.values():
        for k in ("name", "repo", "revision", "dir", "served_id"):
            if not m.get(k):
                out.append("model %s: no %s" % (m.get("id"), k))
        # the recipe menu shows "<name> · <short>": a few words, or it overflows the card
        if len(m.get("short") or "") > 24:
            out.append("model %s: short is over 24 characters" % m.get("id"))
        if m.get("engine", "exl3") not in ENGINES:
            out.append("model %s: engine must be one of %s" % (m.get("id"), "/".join(ENGINES)))
        if m.get("engine") == "strata":
            # Pin Strata source builds separately from the legacy prebuilt path. The
            # default prebuilt recipes still require both their release URL and SHA-256.
            s = m.get("strata") or {}
            if "build" in s and not isinstance(s["build"], bool):
                out.append("model %s: strata.build must be boolean" % m.get("id"))
            source_build = s.get("build", False) is True
            cuda_pattern = r"^(?:auto|1[23])$" if source_build else r"1[23]$"
            required = [("repo", r"https://github\.com/[\w.-]+/[\w.-]+$"),
                        ("commit", r"[0-9a-f]{40}$"), ("cuda", cuda_pattern),
                        ("quant", r"[A-Z0-9_]+$")]
            if source_build:
                if s.get("release") or s.get("sha256"):
                    out.append("model %s: source-built strata must not specify release or sha256" % m.get("id"))
            else:
                required += [("release", r"https://\S+/$"), ("sha256", r"[0-9a-f]{64}$")]
            for k, pat in required:
                if not re.match(pat, str(s.get(k, ""))):
                    out.append("model %s: strata.%s missing or malformed" % (m.get("id"), k))
        if m.get("engine") == "maya":
            maya = m.get("maya") or {}
            required = {
                "repo": r"https://github\.com/[\w.-]+/[\w.-]+\.git$",
                "commit": r"[0-9a-f]{40}$",
                "model": r"Maya-(?:S|M|L)$",
                "data_dir": r"/[^\s]+$",
                "model_dir": r"/[^\s]+$",
            }
            for key, pattern in required.items():
                if not re.fullmatch(pattern, str(maya.get(key, ""))):
                    out.append("model %s: maya.%s missing or malformed" % (m.get("id"), key))
            if maya.get("model_dir") != m.get("dir"):
                out.append("model %s: maya.model_dir must match dir" % m.get("id"))
            if not re.fullmatch(r"[0-9a-f]{40}", str(m.get("revision", ""))):
                out.append("model %s: Maya model revision must be a pinned 40-character commit" % m.get("id"))
            native_ctx = m.get("native_ctx")
            if isinstance(native_ctx, bool) or not isinstance(native_ctx, int) or native_ctx <= 0:
                out.append("model %s: native_ctx must be a positive integer" % m.get("id"))
            context = maya.get("context")
            if (isinstance(context, bool) or not isinstance(context, int) or context < 4096
                    or isinstance(native_ctx, bool) or not isinstance(native_ctx, int)
                    or native_ctx <= 0 or context > native_ctx):
                out.append("model %s: maya.context must be an integer from 4096 through native_ctx" % m.get("id"))
            reserve_mb = maya.get("reserve_mb")
            if isinstance(reserve_mb, bool) or not isinstance(reserve_mb, int) or reserve_mb <= 0:
                out.append("model %s: maya.reserve_mb must be a positive integer" % m.get("id"))
            for key in ("min_disk_gib", "min_vram_gib"):
                if not isinstance(maya.get(key), int) or maya[key] <= 0:
                    out.append("model %s: maya.%s must be a positive integer" % (m.get("id"), key))
            if not isinstance(maya.get("gpu"), int) or maya["gpu"] < 0:
                out.append("model %s: maya.gpu must be a non-negative integer index" % m.get("id"))
            if not isinstance(maya.get("vision"), bool) or maya.get("vision") is not False or m.get("vision") is not False:
                out.append("model %s: Maya currently requires text-only vision=false" % m.get("id"))
        # bootstrap.sh keeps one pack per directory (its .collabosm-revision marker)
        if m.get("dir") in dirs:
            out.append("models %s and %s share %s" % (dirs[m["dir"]], m.get("id"), m["dir"]))
        dirs.setdefault(m.get("dir"), m.get("id"))
    seen = set()
    for r in reg.get("recipes") or []:
        rid = r.get("id")
        if not rid or rid in seen:
            out.append("recipe id missing or repeated: %r" % rid)
        seen.add(rid)
        if r.get("gpu") not in gpus:
            out.append("%s: unknown gpu %r" % (rid, r.get("gpu")))
        if r.get("model") not in models:
            out.append("%s: unknown model %r" % (rid, r.get("model")))
        if r.get("status") not in STATUSES:
            out.append("%s: status must be one of %s" % (rid, "/".join(STATUSES)))
        for k, v in (r.get("env") or {}).items():
            if not ENV_NAME.match(k) or isinstance(v, bool) or not isinstance(v, (int, float, str)):
                out.append("%s: env %s must be an UPPER_CASE name with a scalar value" % (rid, k))
        for k, f in (r.get("facts") or {}).items():
            if not isinstance(f, dict) or "measured" not in f:
                out.append("%s: fact %s needs {v, measured, src}" % (rid, k))
    if not reg.get("recipes"):
        out.append("no recipes")
    return out


def _index(reg: dict):
    return ({g["id"]: g for g in reg["gpus"]}, {m["id"]: m for m in reg["models"]},
            {r["id"]: r for r in reg["recipes"]})


def find(reg: dict, rid: str) -> dict:
    recipes = _index(reg)[2]
    if rid not in recipes:
        raise KeyError("no recipe %r (have: %s)" % (rid, ", ".join(sorted(recipes))))
    return recipes[rid]


def launch_env(reg: dict, rid: str) -> dict:
    """Everything a box needs for this recipe, as environment variables: where the
    card comes from (restore.py), what to download (bootstrap.sh), how to load it
    (serve.sh -> api_server.py)."""
    gpus, models, _ = _index(reg)
    r = find(reg, rid)
    g, m = gpus[r["gpu"]], models[r["model"]]
    env = {
        "RECIPE": r["id"],
        "ACCELERATOR": g["accelerator"], "SHAPE": g["shape"], "MIN_VRAM_GIB": g["min_vram_gib"],
        "MODEL_REPO": m["repo"], "MODEL_REVISION": m["revision"], "MODEL_DIR": m["dir"],
        "MODEL_ID": m["served_id"],
        "MTP": 1 if m.get("mtp") else 0, "NGRAM": 1 if m.get("ngram") else 0,
        "ENGINE": m.get("engine", "exl3"),
    }
    if m.get("engine") == "strata":
        s = m["strata"]
        env.update({"STRATA_REPO": s["repo"], "STRATA_COMMIT": s["commit"],
                    "STRATA_PREBUILT_URL": s.get("release", ""), "STRATA_SHA256": s.get("sha256", ""),
                    "STRATA_CUDA": s["cuda"], "STRATA_MODEL": s["quant"],
                    "STRATA_BUILD": "1" if s.get("build", False) else "0",
                    "STRATA_DIR": s.get("dir", "/content/strata"),
                    # the image encoder on the card (a recipe's env can say cpu)
                    "STRATA_VISION": "gpu" if m.get("vision") else "none"})
    elif m.get("engine") == "maya":
        maya = m["maya"]
        env.update({
            "MAYA_REPO": maya["repo"], "MAYA_COMMIT": maya["commit"],
            "MAYA_MODEL": maya["model"], "MAYA_MODEL_REPO": m["repo"],
            "MAYA_MODEL_REVISION": m["revision"], "MAYA_CONTEXT": maya["context"],
            "MAYA_GPU": maya["gpu"], "MAYA_DATA_DIR": maya["data_dir"],
            "MAYA_MODEL_DIR": maya["model_dir"], "MAYA_MIN_DISK_GIB": maya["min_disk_gib"],
            "MAYA_MIN_VRAM_GIB": maya["min_vram_gib"],
            "MAYA_RESERVE_MB": maya["reserve_mb"],
            "MAYA_VISION": 1 if maya["vision"] else 0,
        })
    env.update(r.get("env") or {})
    return {k: _scalar(v) for k, v in env.items()}


def _scalar(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v)) if abs(v) >= 1 else str(v)
    return str(v)


def facts(reg: dict, rid: str) -> dict:
    """The recipe's evidence, with its card's (the CU rate is the card's fact)."""
    gpus, _, _ = _index(reg)
    r = find(reg, rid)
    out = dict(gpus[r["gpu"]].get("facts") or {})
    out.update(r.get("facts") or {})
    return out


def flat(reg: dict) -> list:
    """One dict per recipe, card and model folded in: what the frontend lists."""
    gpus, models, _ = _index(reg)
    out = []
    for r in reg["recipes"]:
        g, m = gpus[r["gpu"]], models[r["model"]]
        out.append({
            "id": r["id"], "status": r["status"],
            "verified": r["status"] == "verified", "runnable": r["status"] != "placeholder",
            "gpu": g["id"], "card": g["card"], "accelerator": g["accelerator"],
            "shape": g["shape"], "cu_per_hour": g["cu_per_hour"],
            "vram_gb": g["vram_gb"], "ram_gb": g.get("ram_gb"),
            "model_id": m["id"], "model": m["name"], "params": m.get("params"),
            "quant": m.get("quant"), "short": m.get("short") or (m.get("quant") or "").split(",")[0],
            "served_id": m["served_id"],
            "eta_min": r.get("eta_min"), "env": launch_env(reg, r["id"]),
            "facts": facts(reg, r["id"]), "why": r.get("why") or {},
        })
    return out


def _sh_env(env: dict) -> str:
    """Lines for bash `eval`: set only what the caller has not."""
    return "\n".join('[ -n "${%s+x}" ] || %s=%s; export %s' % (k, k, shlex.quote(v), k)
                     for k, v in env.items())


def main(argv: list) -> int:
    cmd = argv[0] if argv else "list"
    try:
        reg = load()
    except (OSError, ValueError) as exc:
        print("[recipe] !! %s" % exc, file=sys.stderr)
        return 2
    if cmd == "check":
        print("[recipe] ok: %d recipes, %d gpus, %d models"
              % (len(reg["recipes"]), len(reg["gpus"]), len(reg["models"])))
        return 0
    if cmd == "list":
        for r in flat(reg):
            print("%-24s %-11s %-18s %-20s %5.2f CU/h  ~%s min"
                  % (r["id"], r["status"], r["card"], r["model"], r["cu_per_hour"],
                     r["eta_min"] if r["eta_min"] is not None else "?"))
        return 0
    rid = argv[1] if len(argv) > 1 else DEFAULT
    try:
        r = find(reg, rid)
    except KeyError as exc:
        print("[recipe] !! %s" % exc.args[0], file=sys.stderr)
        return 2
    if cmd == "show":
        print(json.dumps(next(x for x in flat(reg) if x["id"] == rid), indent=1, ensure_ascii=False))
        return 0
    if cmd in ("env", "vmenv"):
        if r["status"] == "placeholder" and os.environ.get("FORCE_RECIPE") != "1":
            why = (r.get("why") or {}).get("en") or "it has no launch path yet"
            print("[recipe] !! %s is a placeholder: %s" % (rid, why), file=sys.stderr)
            return 3
        env = launch_env(reg, rid)
        if cmd == "env":
            print(_sh_env(env))
        else:
            # the effective values: up.sh evals `env` first, so a caller's override
            # is in os.environ by now and reaches the VM as well
            keys = list(env) + [k for k in VM_EXTRA if k not in env]
            for k in keys:
                v = os.environ.get(k, env.get(k))
                if v is not None:
                    print("export %s=%s" % (k, shlex.quote(str(v))))
        return 0
    print("[recipe] !! unknown command %r (list | show | env | vmenv | check)" % cmd,
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
