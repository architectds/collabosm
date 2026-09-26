#!/usr/bin/env bash
# collabosm up -- a shim: the steps live in scripts/provision.py, which runs the same on
# every machine (Windows with or without WSL, macOS, Linux). This keeps the old command:
#
#   bash scripts/up.sh                                  # = python scripts/provision.py up
#   RECIPE=a100-40g/qwen38-27b bash scripts/up.sh       # another card + model
#   SESSION=mybox CACHE_SIZE=524288 bash scripts/up.sh  # override one setting, once
#
# It runs with the Colab CLI's interpreter ($COLAB_PY, else the one the Colab guide
# installed, else a `uv tool install`): restore.py and the keep-alive import the CLI.
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
py=${COLAB_PY:-$(ls -d "$HOME"/.collabosm/colab-cli/bin/python \
                       "$HOME"/.local/share/uv/tools/google-colab-cli/bin/python 2>/dev/null | head -1)}
exec "${py:-python3}" "$here/provision.py" up "$@"
