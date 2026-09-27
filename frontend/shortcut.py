"""A desktop shortcut that starts collabosm and opens it: `python frontend/shortcut.py`.

frontend/server.py makes one on its first real start, and remembers that it did
(~/.collabosm/shortcut.json), so a shortcut the user deleted stays deleted; this
command makes it again. The shortcut runs this interpreter on frontend/server.py,
which opens http://127.0.0.1:3020 in the browser -- or, when collabosm is already
running, only opens the page. Its picture is the rail's mark (frontend/icon/).

- Windows: collabosm.lnk on the desktop. It opens a console window, minimized;
  closing that window stops collabosm.
- macOS: collabosm.app on the desktop. Quit it from the Dock. Its output goes to
  ~/.collabosm/server.log.
- Linux: collabosm.desktop in the applications menu and on the desktop. It runs
  in a terminal window; closing that window stops collabosm.
"""
import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SERVER = os.path.join(HERE, "server.py")
ICONS = os.path.join(HERE, "icon")
NAME = "collabosm"
MARKER = "shortcut.json"
BLURB = "collabosm: your Colab GPU, in the browser"


def desktop_dir() -> str:
    """Where the desktop is, which is not always ~/Desktop: OneDrive moves it on
    Windows, and XDG names it in the user's language on Linux."""
    if os.name == "nt":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                                  "[Environment]::GetFolderPath('Desktop')"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    elif sys.platform.startswith("linux") and shutil.which("xdg-user-dir"):
        try:
            out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True,
                                 timeout=10).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return os.path.join(os.path.expanduser("~"), "Desktop")


def python_exe() -> str:
    """This interpreter -- on Windows python.exe, not pythonw.exe, so that there is a
    window to close."""
    exe = sys.executable
    if os.name == "nt" and os.path.basename(exe).lower() == "pythonw.exe":
        console = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(console):
            exe = console
    return exe


def create_windows(desktop: str) -> str:
    """A .lnk, written by Windows' own WScript.Shell; every path goes in through the
    environment, so no quoting of it can go wrong."""
    path = os.path.join(desktop, NAME + ".lnk")
    env = dict(os.environ, CO_LNK=path, CO_PY=python_exe(), CO_ARGS='"%s"' % SERVER,
               CO_DIR=ROOT, CO_ICON=os.path.join(ICONS, "collabosm.ico") + ",0", CO_DESC=BLURB)
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:CO_LNK); "
              "$s.TargetPath = $env:CO_PY; $s.Arguments = $env:CO_ARGS; "
              "$s.WorkingDirectory = $env:CO_DIR; $s.IconLocation = $env:CO_ICON; "
              "$s.Description = $env:CO_DESC; $s.WindowStyle = 7; $s.Save()")
    p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       env=env, capture_output=True, text=True, timeout=60)
    if p.returncode != 0 or not os.path.exists(path):
        raise RuntimeError("could not write %s: %s"
                           % (path, (p.stderr or p.stdout or "no answer").strip()[:300]))
    return path


def _icns(resources: str):
    """collabosm.icns from the padded mark, with macOS's own sips and iconutil; None
    where they are missing (the app then shows the generic icon)."""
    if not (shutil.which("sips") and shutil.which("iconutil")):
        return None
    src = os.path.join(ICONS, "collabosm-mac.png")
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, NAME + ".iconset")
        os.makedirs(iconset)
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = str(size * scale)
                name = "icon_%dx%d%s.png" % (size, size, "@2x" if scale == 2 else "")
                subprocess.run(["sips", "-z", px, px, src, "--out", os.path.join(iconset, name)],
                               capture_output=True, check=True)
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o",
                        os.path.join(resources, NAME + ".icns")], capture_output=True, check=True)
    return NAME + ".icns"


def create_macos(desktop: str) -> str:
    """A minimal .app bundle: a shell script as its executable, the mark as its icon."""
    app = os.path.join(desktop, NAME + ".app")
    contents = os.path.join(app, "Contents")
    macos, resources = os.path.join(contents, "MacOS"), os.path.join(contents, "Resources")
    os.makedirs(macos, exist_ok=True)
    os.makedirs(resources, exist_ok=True)
    log = os.path.join(os.path.expanduser("~"), ".collabosm", "server.log")
    exe = os.path.join(macos, NAME)
    with open(exe, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#!/bin/bash\n"
                 'mkdir -p "$HOME/.collabosm"\n'
                 "cd %s || exit 1\n"
                 "exec %s %s >> %s 2>&1\n" % (shlex.quote(ROOT), shlex.quote(python_exe()),
                                              shlex.quote(SERVER), shlex.quote(log)))
    os.chmod(exe, 0o755)
    info = {"CFBundleName": NAME, "CFBundleDisplayName": NAME, "CFBundleExecutable": NAME,
            "CFBundleIdentifier": "io.collabosm.launcher", "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.0", "LSMinimumSystemVersion": "10.13"}
    try:
        icon = _icns(resources)
    except (OSError, subprocess.SubprocessError):
        icon = None
    if icon:
        info["CFBundleIconFile"] = icon
    with open(os.path.join(contents, "Info.plist"), "wb") as fh:
        plistlib.dump(info, fh)
    return app


def _exec_quote(arg: str) -> str:
    """One Exec= argument, quoted the way the Desktop Entry specification asks."""
    for ch in ("\\", '"', "`", "$"):
        arg = arg.replace(ch, "\\" + ch)
    return '"%s"' % arg


def create_linux(desktop: str) -> str:
    """A .desktop entry in the applications menu, and a copy on the desktop if there
    is one (marked trusted where GNOME's gio can, so it starts without a prompt)."""
    entry = ("[Desktop Entry]\nType=Application\nName=%s\nComment=%s\n"
             "Exec=%s %s\nPath=%s\nIcon=%s\nTerminal=true\nCategories=Development;Network;\n"
             % (NAME, BLURB, _exec_quote(python_exe()), _exec_quote(SERVER), ROOT,
                os.path.join(ICONS, "collabosm.png")))
    apps = os.path.join(os.environ.get("XDG_DATA_HOME")
                        or os.path.join(os.path.expanduser("~"), ".local", "share"), "applications")
    written = []
    for folder in (apps, desktop):
        if folder == desktop and not os.path.isdir(desktop):
            continue
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, NAME + ".desktop")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(entry)
        os.chmod(path, 0o755)
        written.append(path)
    if len(written) > 1 and shutil.which("gio"):
        subprocess.run(["gio", "set", written[-1], "metadata::trusted", "true"],
                       capture_output=True)
    return written[-1]


def create(desktop=None) -> str:
    """The shortcut for this system; returns where it is."""
    desktop = desktop or desktop_dir()
    if os.name == "nt":
        return create_windows(desktop)
    if sys.platform == "darwin":
        return create_macos(desktop)
    return create_linux(desktop)


def ensure_once(state_dir: str, log=print, desktop=None):
    """frontend/server.py's first real start: make the shortcut once, and remember that
    it was made, so that one the user deleted is not put back. Never raises."""
    marker = os.path.join(state_dir, MARKER)
    if os.path.exists(marker):
        return None
    try:
        path = create(desktop)
    except Exception as exc:                              # noqa: BLE001 - never stops the app
        log("[fe] shortcut   not made: %s (python frontend/shortcut.py tries again)" % exc)
        return None
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(marker, "w", encoding="utf-8") as fh:
            json.dump({"path": path, "at": int(time.time())}, fh)
    except OSError:
        pass
    log("[fe] shortcut   %s (made once; python frontend/shortcut.py makes it again)" % path)
    return path


if __name__ == "__main__":
    try:
        print("shortcut: %s" % create())
    except Exception as exc:                              # noqa: BLE001 - a command's answer
        print("could not make the shortcut: %s" % exc, file=sys.stderr)
        sys.exit(1)
