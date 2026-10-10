#!/usr/bin/env bash
# Project Maya adapter, sourced by bootstrap.sh / serve.sh on the Colab VM only.
set -uo pipefail
MAYA_DIR=${MAYA_DIR:-/content/project-maya}
MAYA_DATA_DIR=${MAYA_DATA_DIR:-/content/Maya-data}
MAYA_MODEL_DIR=${MAYA_MODEL_DIR:-$MAYA_DATA_DIR/models}
MAYA_REPO=${MAYA_REPO:-https://github.com/mw00/project-maya.git}
MAYA_COMMIT=${MAYA_COMMIT:-c189f7ac8114ca039e9643314d1fead014fc6ff5}
MAYA_MODEL=${MAYA_MODEL:-Maya-L}
MAYA_MODEL_REPO=${MAYA_MODEL_REPO:-peasantsmith/GLM-5.3-Flash-Maya-GGUF}
MAYA_MODEL_REVISION=${MAYA_MODEL_REVISION:-f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da}
MAYA_CONTEXT=${MAYA_CONTEXT:-131072}
MAYA_RESERVE_MB=${MAYA_RESERVE_MB:-2048}
MAYA_GPU=${MAYA_GPU:-0}
MAYA_MIN_DISK_GIB=${MAYA_MIN_DISK_GIB:-180}
MAYA_MIN_VRAM_GIB=${MAYA_MIN_VRAM_GIB:-70}
MAYA_PY=$MAYA_DIR/.venv/bin/python
MAYA_CONFIG=$MAYA_DIR/maya-maya-l.json

maya_fail() { say "!! Maya: $*"; return 1; }
maya_preflight() {
  local free mib total name started_shards
  [[ "$MAYA_GPU" =~ ^[0-9]+$ && "$MAYA_MIN_DISK_GIB" =~ ^[0-9]+$ && "$MAYA_MIN_VRAM_GIB" =~ ^[0-9]+$ && "$MAYA_CONTEXT" =~ ^[0-9]+$ && "$MAYA_RESERVE_MB" =~ ^[0-9]+$ ]] || { maya_fail "GPU, context, reserve, and minimum resource settings must be non-negative integers"; return 1; }
  (( MAYA_CONTEXT >= 4096 && MAYA_CONTEXT <= 1048576 && MAYA_RESERVE_MB > 0 )) || { maya_fail "context must be 4096..1048576 and expert reserve must be positive"; return 1; }
  [[ "$(uname -s)" == Linux ]] || { maya_fail "Project Maya requires Linux"; return 1; }
  command -v git >/dev/null && command -v nvidia-smi >/dev/null || { maya_fail "git and nvidia-smi are required"; return 1; }
  free=$(df -Pk "$MAYA_DATA_DIR" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')
  [[ "$free" =~ ^[0-9]+$ ]] || { maya_fail "cannot determine model disk free space"; return 1; }
  # The initial 156.3 GB download needs headroom. On a resumed or reattached
  # box those bytes already occupy disk; the upstream installer checks each
  # shard and the *remaining* space before continuing the download.
  started_shards=$(find "$MAYA_MODEL_DIR/Maya-L" -maxdepth 1 -type f -name 'GLM-5.3-Flash-Maya-L-IQ3_S-*.gguf' 2>/dev/null | wc -l)
  if (( started_shards == 0 && free < MAYA_MIN_DISK_GIB )); then
    maya_fail "need ${MAYA_MIN_DISK_GIB} GiB free for an initial Maya-L download; found ${free} GiB"
    return 1
  fi
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n "$((MAYA_GPU + 1))p" | tr -d '[:space:]')
  [[ "$mib" =~ ^[0-9]+$ ]] || { maya_fail "GPU $MAYA_GPU unavailable"; return 1; }
  total=$((mib / 1024))
  (( total >= MAYA_MIN_VRAM_GIB )) || { maya_fail "GPU $MAYA_GPU has ${total} GiB; need ${MAYA_MIN_VRAM_GIB} GiB"; return 1; }
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n "$((MAYA_GPU + 1))p")
  [[ "$name" == *A100* ]] || { maya_fail "selected GPU is not an A100: ${name:-unknown}"; return 1; }
}

maya_install() {
  status "stage=runtime"
  mkdir -p "$(dirname "$MAYA_DIR")" "$MAYA_DATA_DIR"
  maya_preflight || return 1
  if [[ "$(git -C "$MAYA_DIR" rev-parse HEAD 2>/dev/null || true)" != "$MAYA_COMMIT" ]]; then
    if [[ ! -d "$MAYA_DIR/.git" ]]; then git clone --no-checkout "$MAYA_REPO" "$MAYA_DIR" || { maya_fail "source clone failed"; return 1; }; fi
    git -C "$MAYA_DIR" fetch --depth 1 origin "$MAYA_COMMIT" && git -C "$MAYA_DIR" checkout --detach -f FETCH_HEAD || { maya_fail "pinned source checkout failed"; return 1; }
  fi
  [[ "$(git -C "$MAYA_DIR" rev-parse HEAD 2>/dev/null)" == "$MAYA_COMMIT" ]] || { maya_fail "source commit mismatch"; return 1; }
  # This exact source pins Maya-L to HF `main`. Replace only its catalog revision,
  # fail on source drift, and keep upstream per-shard SHA-256 checks active.
  python3 - "$MAYA_DIR/maya.py" "$MAYA_MODEL_REPO" "$MAYA_MODEL_REVISION" <<'PY' || { maya_fail "cannot pin Maya-L model revision in the catalog"; return 1; }
import pathlib, re, sys
p, repo, revision = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
s = p.read_text(encoding="utf-8")
pattern = r'("repo":\s*"' + re.escape(repo) + r'",\s*"revision":\s*)"([^"]+)"(,\s*"folder":\s*"Maya-L")'
matches = list(re.finditer(pattern, s))
if len(matches) != 1: raise SystemExit(f"expected exactly one Maya-L catalog revision, got {len(matches)}")
m = matches[0]
if m.group(2) not in ("main", revision): raise SystemExit(f"unexpected Maya-L source revision: {m.group(2)}")
if m.group(2) == "main":
    s = s[:m.start(2)] + revision + s[m.end(2):]
    p.write_text(s, encoding="utf-8")
PY
  if ! "$MAYA_PY" -m pip --version >/dev/null 2>&1; then
    rm -rf "$MAYA_DIR/.venv"
    python3 -m venv --without-pip "$MAYA_DIR/.venv" || { maya_fail "cannot create private Python environment"; return 1; }
    local getpip="$MAYA_DIR/.get-pip.py"
    curl -fsSL --retry 3 -o "$getpip" https://bootstrap.pypa.io/get-pip.py || { rm -f "$getpip"; maya_fail "cannot fetch pip bootstrap"; return 1; }
    echo "fb24e693bab954209a063d90953621412ccad4a500905a726286e038f508ddf6  $getpip" | sha256sum -c --status - || { rm -f "$getpip"; maya_fail "pip bootstrap checksum mismatch"; return 1; }
    "$MAYA_PY" "$getpip" --disable-pip-version-check -q || { rm -f "$getpip"; maya_fail "pip bootstrap failed"; return 1; }
    rm -f "$getpip"
  fi
  export MAYA_UPDATE_CHECK=0 MAYA_RESTART_ON_UPDATE=0
  status "stage=weights"
  ( cd "$MAYA_DIR" && "$MAYA_PY" maya.py --setup --model "$MAYA_MODEL" --gpu "$MAYA_GPU" --context "$MAYA_CONTEXT" --models-dir "$MAYA_MODEL_DIR" --env "STRATA_GLM_RESERVE_MB=$MAYA_RESERVE_MB" --download-model --no-vision --yes --no-start --port "$PORT" ) || { maya_fail "setup or model download failed"; return 1; }
  python3 - "$MAYA_CONFIG" "$MAYA_MODEL_DIR" "$MAYA_GPU" "$MAYA_CONTEXT" "$MAYA_RESERVE_MB" <<'PY' || { maya_fail "Maya config does not match the selected text-only recipe"; return 1; }
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); model_dir = pathlib.Path(sys.argv[2]); gpu = int(sys.argv[3]); ctx = str(sys.argv[4]); reserve = str(sys.argv[5])
cfg = json.loads(p.read_text())
args = cfg.get('args') or []
def arg(name):
    return args[args.index(name) + 1] if name in args[:-1] else None
assert cfg.get('model_name') == 'glm-5.3-flash'
assert (cfg.get('installer') or {}).get('quant') == 'Maya-L'
assert cfg.get('gpu') == [gpu]
assert arg('--max-context') == ctx
assert str(cfg.get('env', {}).get('STRATA_GLM_RESERVE_MB')) == reserve
assert pathlib.Path(arg('--glm-pack')).is_relative_to(model_dir / 'Maya-L')
assert not cfg.get('vision')
assert cfg.get('host', '127.0.0.1') in ('127.0.0.1', 'localhost')
PY
  say "Maya setup complete: $MAYA_MODEL, context $MAYA_CONTEXT"
}

maya_start() {
  [[ -n "${KEY:-}" ]] || { maya_fail "API key not initialized"; return 1; }
  [[ -f "$MAYA_CONFIG" ]] || { maya_fail "verified config missing"; return 1; }
  say "starting Maya API on loopback :$PORT (secret supplied through environment)"
  ( cd "$MAYA_DIR" && MAYA_UPDATE_CHECK=0 MAYA_RESTART_ON_UPDATE=0 STRATA_API_KEY="$KEY" exec nohup "$MAYA_PY" -u maya.py --config "$MAYA_CONFIG" --port "$PORT" --plain ) > "$LOG" 2>&1 &
  MAYA_PROC='maya.py'
}
