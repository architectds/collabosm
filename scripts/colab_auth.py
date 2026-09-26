#!/usr/bin/env python3
"""The Colab CLI's own sign-in, driven by the frontend instead of a terminal.

Run it with the Python that has google-colab-cli -- natively, or inside WSL -- and
it uses the CLI's own OAuth client, scopes and token file: a token made here is the
token the CLI uses, and a CLI that is already signed in is recognised here.

    python colab_auth.py status   # COLAB_AUTH {"token": none|ok|unverified|expired|invalid|scopes, ...}
    python colab_auth.py login    # AUTH_URL <url>, then AUTH_OK <email> | AUTH_ERROR <why>
    python colab_auth.py logout   # LOGOUT ok [revoke_failed] | LOGOUT_ERROR <why>

The CLI asks for a code pasted back from a Google page, because it has to work on
machines with no browser. This runs where the browser is, so it takes the standard
loopback redirect (http://localhost:<port>/) instead -- the client the CLI ships is
registered for exactly that -- and the user only clicks Allow. From WSL the same
works: WSL2 forwards localhost, so the Windows browser reaches the port.

Nothing here prints a token. `status` refreshes the access token in memory only; the
file on disk is written by `login` alone.
"""
from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

# The page the redirect lands on: all three of the rail's languages, because the
# browser tab does not know which one the rail is in.
DONE_PAGE = ("Colab is connected to collabosm. You can close this tab.\n\n"
             "Colab 已连接到 collabosm，可以关闭这个标签页。\n\n"
             "Colab と collabosm の接続が完了しました。"
             "このタブは閉じてかまいません。\n")

# What Colab itself refuses to work without (colab_cli.auth explains both): the
# session backend wants the email scope, the runtime service the colaboratory one.
NEEDED = ("https://www.googleapis.com/auth/colaboratory",
          "https://www.googleapis.com/auth/userinfo.email")

# Google's own endpoints -- never ones a token file names: a file that could steer
# where its refresh token is sent is a way to steal it. google-auth refreshes against
# its fixed endpoint for the same reason.
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"


def _cli_auth():
    from colab_cli import auth
    return auth


def version():
    try:
        return importlib.metadata.version("google-colab-cli")
    except Exception:
        return None


def client_config():
    """The OAuth client, found the way the CLI finds it: the user's own override
    (~/.colab-cli-oauth-config.json) first, then the one the package ships."""
    override = os.path.expanduser("~/.colab-cli-oauth-config.json")
    if os.path.exists(override):
        with open(override, encoding="utf-8") as fh:
            return json.load(fh)
    from importlib import resources
    return json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())


def tokeninfo(access_token):
    """-> {"email", "scope", ...} for an access token (the CLI's `whoami` asks the same)."""
    url = TOKENINFO_URL + "?" + urllib.parse.urlencode({"access_token": access_token})
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def email_of(id_token):
    """The email claim of an ID token, unverified: it came straight from Google's token
    endpoint over TLS, and it is only shown, never trusted for anything."""
    try:
        part = id_token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("email")
    except Exception:
        return None


def save(text, path):
    """Write the token file atomically, readable by this user only (the CLI reads it
    with Credentials.from_authorized_user_file, exactly as it wrote it)."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".token-", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if os.name != "nt":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _why(exc):
    text = "%s: %s" % (type(exc).__name__, exc)
    for known in ("access_denied", "invalid_grant", "invalid_scope", "redirect_uri_mismatch"):
        if known in text:
            return known
    if "Timed out" in text or "WSGITimeoutError" in text:
        return "timeout"
    return text.replace("\n", " ")[:200]


def cmd_status():
    auth = _cli_auth()
    path = auth.TOKEN_CONFIG_PATH
    out = {"version": version(), "token": "none", "account": None}
    if not os.path.exists(path):
        return out
    from google.auth import exceptions as gexc
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    try:
        # the scopes the file was granted, not the ones we would ask for: asking for
        # more at refresh time fails as if the token were dead
        creds = Credentials.from_authorized_user_file(path)
    except Exception as exc:
        out.update(token="invalid", error=type(exc).__name__)
        return out
    if not creds.refresh_token:
        out.update(token="invalid", error="no refresh token")
        return out
    try:
        creds.refresh(Request())
    except gexc.RefreshError as exc:
        out.update(token="expired", error=_why(exc))
        return out
    except Exception as exc:                # offline, DNS, a proxy: the token may be fine
        out.update(token="unverified", error=_why(exc))
        return out
    try:
        info = tokeninfo(creds.token)
    except Exception as exc:
        out.update(token="unverified", error=_why(exc))
        return out
    granted = set((info.get("scope") or "").split())
    missing = [s for s in NEEDED if s not in granted]
    out.update(account=info.get("email"))
    if missing:
        out.update(token="scopes", error=" ".join(s.rsplit("/", 1)[-1] for s in missing))
    else:
        out.update(token="ok")
    return out


def cmd_login(port, timeout):
    auth = _cli_auth()
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_config(client_config(), auth.PUBLIC_SCOPES)
    try:
        # the frontend opens the URL (it runs where the browser is; from WSL, this
        # process cannot); run_local_server prints it for us -- one line, flushed
        creds = flow.run_local_server(
            host="localhost", bind_addr="127.0.0.1", port=port, open_browser=False,
            authorization_prompt_message="AUTH_URL {url}", success_message=DONE_PAGE,
            timeout_seconds=timeout, prompt="consent", access_type="offline")
    except Exception as exc:
        print("AUTH_ERROR %s" % _why(exc))
        return 1
    if not creds or not creds.refresh_token:
        print("AUTH_ERROR no refresh token was issued")
        return 1
    try:
        save(creds.to_json(), auth.TOKEN_CONFIG_PATH)
    except Exception as exc:
        print("AUTH_ERROR could not save the token: %s" % _why(exc))
        return 1
    email = email_of(getattr(creds, "id_token", None) or "")
    if not email:
        try:
            email = tokeninfo(creds.token).get("email")
        except Exception:
            email = None
    print("AUTH_OK %s" % (email or "?"))
    return 0


def cmd_logout():
    auth = _cli_auth()
    path = auth.TOKEN_CONFIG_PATH
    if not os.path.exists(path):
        print("LOGOUT ok")
        return 0
    revoked = False
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        token = data.get("refresh_token") or data.get("token")
        if token:
            req = urllib.request.Request(
                REVOKE_URL, data=urllib.parse.urlencode({"token": token}).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                revoked = 200 <= resp.status < 300
    except Exception:
        revoked = False                     # still forget it here: that is what was asked
    try:
        os.remove(path)
    except OSError as exc:
        print("LOGOUT_ERROR %s" % _why(exc))
        return 1
    print("LOGOUT ok" + ("" if revoked else " revoke_failed"))
    return 0


def main(argv):
    try:
        sys.stdout.reconfigure(line_buffering=True)   # AUTH_URL must reach the frontend now
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    lg = sub.add_parser("login")
    lg.add_argument("--port", type=int, default=0, help="loopback port (0 = any free one)")
    lg.add_argument("--timeout", type=int, default=300, help="seconds to wait for the browser")
    sub.add_parser("logout")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "status":
            print("COLAB_AUTH " + json.dumps(cmd_status()))
            return 0
        if a.cmd == "login":
            return cmd_login(a.port, a.timeout)
        return cmd_logout()
    except ImportError as exc:
        why = "cannot import the Colab CLI: %s" % exc
        if a.cmd == "status":
            print("COLAB_AUTH " + json.dumps({"version": version(), "token": "error", "error": why}))
        else:
            print("%s %s" % ("AUTH_ERROR" if a.cmd == "login" else "LOGOUT_ERROR", why))
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
