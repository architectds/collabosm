#!/usr/bin/env bash
# collabosm bootstrap (runs ON the Colab VM).
#
# Two things, in the order that fails cheapest:
#   1. runtime  - ExLlamaV3, from a prebuilt wheel (default), source, or a mirror URL
#   2. weights  - Qwen3.8-Flash-Next exl3, from Hugging Face at a PINNED revision
#
# Idempotent: existing, complete files are left alone, so a re-run after a VM
# hiccup costs seconds instead of 5 minutes.
#
# Everything prints BOOTSTRAP_OK or BOOTSTRAP_FAILED at the end; scripts/up.sh polls for it
# and chains serve.sh after a successful run (exit 0), so the model loads without a second trip.
set -uo pipefail

: "${RUNTIME:=wheel}"
: "${CACHE_SIZE:=262144}"
: "${CACHE_QUANT:=4}"
: "${CPU_CACHE_GB:=0}"
: "${RECURRENT_CACHE_GB:=4}"
: "${NDT:=4}"
: "${GCS:=4096}"
: "${PORT:=8090}"

MODEL_REPO=${MODEL_REPO:-turboderp/Qwen3.8-Flash-Next-exl3}
MODEL_REVISION=${MODEL_REVISION:-55a732e0c4c3d4614bc42b68493bb930d9b02c0a}
MODEL_DIR=${MODEL_DIR:-/content/exl3}
WHEEL_URL=${WHEEL_URL:-https://github.com/turboderp-org/exllamav3/releases/download/v1.5.1/exllamav3-1.5.1%2Bcu128.torch2.11.0-cp313-cp313-linux_x86_64.whl}
SRC_TAG=${SRC_TAG:-v1.5.1}
SRC_DIR=${SRC_DIR:-/content/exllamav3-src}

fail() { echo "BOOTSTRAP_FAILED: $*"; echo "$*" > /content/STATUS; exit 1; }
say()  { echo "[boot $(date -u +%H:%M:%S)] $*"; }
status() { echo "$*" > /content/STATUS; }

# ---------------------------------------------------------------- the image
say "image probe"
status "stage=probing"
python - <<'PY' || fail "image probe"
import json, sys
try:
    import torch
    print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
except Exception as e:
    raise SystemExit("torch missing: %r" % e)
PY

# ---------------------------------------------------------------- the runtime
status "stage=runtime"
if python -c "import exllamav3" 2>/dev/null; then
  say "exllamav3 already importable: $(python -c 'import exllamav3,os;print(getattr(exllamav3,"__version__","?"))' 2>/dev/null)"
else
  case "$RUNTIME" in
    wheel)
      say "installing the prebuilt wheel (no compile)"
      t0=$(date +%s)
      pip install --no-cache-dir "$WHEEL_URL" || fail "wheel install failed (ABI mismatch? probe python/torch/cuda and pick the matching release asset)"
      say "wheel installed in $(( $(date +%s) - t0 ))s"
      ;;
    source)
      say "building from source at $SRC_TAG (this costs metered minutes)"
      [ -d "$SRC_DIR/.git" ] || git clone --depth 1 --branch "$SRC_TAG" https://github.com/turboderp-org/exllamav3 "$SRC_DIR" || fail "clone"
      ( cd "$SRC_DIR" && pip install --no-cache-dir . ) || fail "source build failed"
      ;;
    *) fail "unknown RUNTIME=$RUNTIME (expected wheel|source)" ;;
  esac
  python -c "import exllamav3" || fail "exllamav3 still not importable after install"
fi
python -c "import exllamav3" || fail "exllamav3 import"

# ----------------------------------------------------------------- the weights
status "stage=weights"
if [ -f "$MODEL_DIR/config.json" ]; then
  n=$(ls "$MODEL_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l)
  say "weights already present: $n shards in $MODEL_DIR"
else
  say "downloading $MODEL_REPO @ $MODEL_REVISION -> $MODEL_DIR"
  say "  (~100 GiB. HF_HUB_DISABLE_XET=1 is the documented fallback when the Xet path stalls)"
  t0=$(date +%s)
  HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-0} python - <<PY || fail "model download"
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id=os.environ.get("MODEL_REPO", "turboderp/Qwen3.8-Flash-Next-exl3"),
    revision=os.environ.get("MODEL_REVISION", "55a732e0c4c3d4614bc42b68493bb930d9b02c0a"),
    local_dir=os.environ.get("MODEL_DIR", "/content/exl3"),
    max_workers=8,
)
print("snapshot at", p)
PY
  say "weights in $(( $(date +%s) - t0 ))s"
fi
[ -f "$MODEL_DIR/config.json" ] || fail "no config.json in $MODEL_DIR - the pack is incomplete"

# ------------------------------------------------------------------ the env
status "stage=env"
cat > /content/collabosm.env <<ENV
CACHE_SIZE=$CACHE_SIZE
CACHE_QUANT=$CACHE_QUANT
CPU_CACHE_GB=$CPU_CACHE_GB
RECURRENT_CACHE_GB=$RECURRENT_CACHE_GB
NDT=$NDT
GCS=$GCS
PORT=$PORT
MODEL_DIR=$MODEL_DIR
ENV
say "wrote /content/collabosm.env"
du -sh "$MODEL_DIR" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
echo "BOOTSTRAP_OK"
status "stage=bootstrapped"