#!/usr/bin/env python3
"""Print this account's Colab compute-unit balance as one line: `CCU {json}`.

The CLI has no balance command, but Colab's own web UI reads it from
/tun/m/ccu-info with the same login the CLI already holds -- so this asks the
same question, read-only, through the CLI's authenticated session:

    {"currentBalance": 115.7, "consumptionRateHourly": 6.77, "assignmentsCount": 1, ...}

That is server truth: it counts every session on the account, including ones this
kit never saw, which a local ledger cannot. Run it with the CLI's own interpreter:

    ~/.local/share/uv/tools/google-colab-cli/bin/python scripts/colab_ccu.py
"""
import glob
import json
import os
import sys
from urllib.parse import urljoin

for cand in glob.glob(os.path.expanduser(
        "~/.local/share/uv/tools/google-colab-cli/lib/python3*/site-packages")):
    if cand not in sys.path:
        sys.path.insert(0, cand)

try:
    from colab_cli.common import state
    client = state.client
    r = client.session.request("GET", urljoin(client.colab_domain, "/tun/m/ccu-info"),
                               params={"authuser": "0"},
                               headers={"Accept": "application/json"}, timeout=30)
    body = r.text
    if body.startswith(")]}'"):                      # Google's XSSI prefix
        body = body.split("\n", 1)[1] if "\n" in body else body[4:]
    if not r.ok:
        raise RuntimeError("HTTP %s" % r.status_code)
    data = json.loads(body)
    print("CCU " + json.dumps({"balance": data.get("currentBalance"),
                               "rate_per_hour": data.get("consumptionRateHourly"),
                               "assignments": data.get("assignmentsCount")}))
except Exception as exc:                              # noqa: BLE001 - it is a probe
    print("CCU_ERROR %r" % exc)
