"""The Colab CLI's session record and keep-alive, whichever CLI version is installed.

google-colab-cli 0.7 keeps the runtime-proxy token's expiry (and the machine shape)
on the record, and refreshes the token itself when the expiry is near or missing:
it lists assignments first, and drops the record if that listing does not show the
box -- a listing that has been seen to leave out a live box for a few seconds. So a
record written without the expiry costs the next CLI command a listing, and with it
the chance of losing the record; `provision.py down` would then report "not found"
while the VM went on billing. Fields the installed CLI's SessionState does not have
are left out, so 0.6 is written the same record as before.

0.7 also dropped the keep-alive ping ("sessions stay alive as long as the kernel is
active"); scripts/heartbeat.py on the kernel is what holds a box either way.
"""
from datetime import datetime, timedelta, timezone


def _fields(SessionState) -> set:
    names = getattr(SessionState, "model_fields", None) or getattr(SessionState, "__fields__", None)
    return set(names or ())


def extra_fields(SessionState, assignment) -> dict:
    """What this CLI's record keeps beside name/token/url/endpoint, from a listed or
    freshly assigned box: the token's expiry and the machine shape (0.7), or nothing (0.6)."""
    have, out = _fields(SessionState), {}
    info = getattr(assignment, "runtime_proxy_info", None)
    if "token_expires_at" in have and info is not None:
        if hasattr(info, "expires_at"):
            out["token_expires_at"] = info.expires_at()
        elif getattr(info, "token_expires_in_seconds", None):
            out["token_expires_at"] = (datetime.now(timezone.utc)
                                       + timedelta(seconds=int(info.token_expires_in_seconds)))
    shape = getattr(getattr(assignment, "machine_shape", None), "name", None)
    if "machine_shape" in have and shape:
        out["machine_shape"] = shape
    return out


def keep_alive(client, endpoint) -> bool:
    """The CLI's keep-alive ping where the installed CLI still has one (0.6). False
    when it has none (0.7), which is not an error."""
    ping = getattr(client, "keep_alive_assignment", None)
    if ping is None:
        return False
    ping(endpoint)
    return True
