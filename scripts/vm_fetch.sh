#!/usr/bin/env bash
# Print one small file from a Colab session between __BEGIN__ / __END__ markers.
#
#   bash scripts/vm_fetch.sh <session> <remote path> [timeout seconds]
#
# The frontend couples to a running service by reading the VM's own files
# (/content/endpoint.json, STATUS, api-key.txt) through the contents API --
# `colab download`. `colab exec` goes through the kernel and was seen to hang for
# minutes. This lives in a file because wsl.exe hands whatever follows `--` to the
# default Linux shell first: an inline script is parsed twice, and a `$f` that the
# inner shell was meant to set is expanded -- to nothing -- by the outer one.
set -u
session=${1:?session name}
remote=${2:?remote path}
limit=${3:-90}
colab=${COLAB:-$HOME/.local/bin/colab}
f=$(mktemp)
timeout "$limit" "$colab" download -s "$session" "$remote" "$f" >/dev/null 2>&1
if [ -s "$f" ]; then
  echo __BEGIN__
  cat "$f"
  echo
  echo __END__
fi
rm -f "$f"
