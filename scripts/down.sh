#!/usr/bin/env bash
# collabosm down: STOP THE VM. On a metered plan this is the most important command here.
# A shim: the steps live in scripts/provision.py (the same on every machine).
#
#   bash scripts/down.sh                  # = python scripts/provision.py down
#   SESSION=mybox bash scripts/down.sh
#   ENDPOINT=<endpoint> bash scripts/down.sh   # also puts back a record the CLI dropped
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
py=${COLAB_PY:-$(ls -d "$HOME"/.collabosm/colab-cli/bin/python \
                       "$HOME"/.local/share/uv/tools/google-colab-cli/bin/python 2>/dev/null | head -1)}
exec "${py:-python3}" "$here/provision.py" down "$@"
