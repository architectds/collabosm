#!/usr/bin/env bash
# collabosm down: stop the VM. On a metered plan this is the most important script here.
#
#   bash scripts/down.sh            # stop session $SESSION
#   SESSION=mybox bash scripts/down.sh
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COLAB=${COLAB:-$(command -v colab 2>/dev/null || echo "$HOME/.local/bin/colab")}
COLAB_PY=${COLAB_PY:-$(ls -d "$HOME"/.local/share/uv/tools/google-colab-cli/bin/python 2>/dev/null | head -1)}
SESSION=${SESSION:-collabosm}
CU_PER_HOUR=${CU_PER_HOUR:-6.77}     # A100 SXM4 High-RAM (80 GB / 167 GB): Colab's own rate for it

# The CLI drops its local record of a live box now and then (`colab sessions` then lists it
# as `[?] <endpoint>`), and `colab stop -s` cannot stop what it has no record of: the VM
# would keep billing behind a stop that "worked". Given the endpoint (ENDPOINT=..., which
# the frontend passes), put the record back first -- no ping, nothing assigned.
if [ -n "${ENDPOINT:-}" ] && [ -n "$COLAB_PY" ]; then
  "$COLAB_PY" "$SCRIPT_DIR/colab_keepalive.py" --no-ping "$SESSION" "$ENDPOINT" 2>&1 | sed 's/^/[down] /'
fi

echo "[down] stopping '$SESSION' ..."
out=$($COLAB stop -s "$SESSION" 2>&1) && echo "$out" || { echo "$out"; echo "[down] stop returned nonzero"; }

echo "[down] remaining server-side assignments:"
$COLAB sessions 2>&1 | sed 's/^/  /' || true

cat <<EOF

[down] If an assignment is still listed above, it is still billing.
       A100-80GB High-RAM costs about ${CU_PER_HOUR} CU/h (~\$0.68/h). 200 CU/month ≈ 29.5 h.
       Nothing here keeps a box alive: the frontend pings Colab only while a box is in use.
EOF