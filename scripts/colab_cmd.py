#!/usr/bin/env python3
"""The Colab CLI -- `colab ...` -- on any machine, Windows included.

google-colab-cli is pure Python and its backend is HTTP, but its command imports the
interactive console (`colab console`) at startup, and the console needs termios and
tty: Unix only. So on Windows `colab` dies before it reads an argument, although
nothing this kit runs -- new, sessions, exec, upload, download, stop -- goes near the
console. Where those two modules are missing, empty stand-ins let the rest load;
`colab console` is then the one command that cannot work. Elsewhere this is just
`colab`, the real modules untouched.

    python scripts/colab_cmd.py sessions        # = colab sessions
    python scripts/colab_cmd.py exec -s collabosm -f scripts/status.py

Run it with the interpreter that has google-colab-cli (the Colab guide's).
"""
import sys
import types


def main(argv):
    try:
        import termios  # noqa: F401
    except ImportError:
        # Windows. Its stdlib does have tty -- but tty is built on termios and dies
        # importing it (TCSAFLUSH), so both are stood in for, not just the missing one
        for name in ("termios", "tty"):
            sys.modules[name] = types.ModuleType(name)
    from colab_cli.cli import app
    sys.argv = ["colab"] + list(argv)
    return app()                    # click's standalone mode exits with the command's code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
