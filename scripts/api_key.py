"""Runs ON the Colab VM: print the endpoint key so the local proxy can inject it.

serve.sh writes /content/api-key.txt once and the key is stable across restarts
(only the tunnel hostname changes), so this is read once per session, right after
the endpoint reports healthy.
"""
KEY_FILE = "/content/api-key.txt"

try:
    with open(KEY_FILE) as fh:
        print("APIKEY " + fh.read().strip())
except Exception as exc:                       # noqa: BLE001 - it is a probe
    print("APIKEY_ERROR %r" % exc)
