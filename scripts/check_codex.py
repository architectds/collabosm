"""Drive the real Codex CLI through manufactured model turns, and judge what it did.

The model's side is scripted (dev_stub --script): each scenario writes exactly the
bytes a model would -- a call in Qwen3-Coder XML, the older JSON body, a freeform
apply_patch, a namespaced multi-agent call, typed and verbatim arguments, two calls
in one turn, a call the token ceiling cuts off, a think block -- and they go through
the shipping Handler to Codex, run exactly as a user runs it. What is judged is what
Codex made of them: the commands it ran, the files its patches wrote, the history it
sent back, and the prompt the model would be shown next. No GPU, no tunnel.

    python scripts/check_codex.py                  # every scenario
    python scripts/check_codex.py patch typed      # some of them
    python scripts/check_codex.py --list
    python scripts/check_codex.py --keep           # keep each scenario's files

Needs `codex` on PATH (or --codex) and the pack's chat_template.jinja in --model-dir
(default $MODEL_DIR, else ./_dev_model); when it is not there it is fetched from the
default recipe's pinned revision. Codex runs with approvals off and no sandbox, in a
scratch directory, on the commands scripted here -- echo, and writing files there.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import recipe as registry  # noqa: E402

KEY_ENV = "COLLABOSM_TEST_KEY"
PRINT = threading.Lock()


# --- what the model writes --------------------------------------------------------

def call(name, **params):
    """One call the way this pack's template teaches it: Qwen3-Coder XML, each value
    raw text (a non-string as JSON), on its own lines."""
    body = "".join("<parameter=%s>\n%s\n</parameter>\n"
                   % (k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                   for k, v in params.items())
    return "<tool_call>\n<function=%s>\n%s</function>\n</tool_call>" % (name, body)


# A value full of what must not end it or be touched: tags (a </parameter> that text
# follows is still the value), quotes, a backtick, a $, backslashes, JSON, unicode.
VERBATIM = ("quotes \"double\" 'single' `backtick` $notavar \\back\\slash\n"
            "tags <b>bold</b> </function> <parameter=x> </parameter> kept\n"
            "json {\"a\": [1, 2]} & ampersand < less > greater\n"
            "unicode: 中文 ✓ é")
VERBATIM_CMD = "@'\n%s\n'@ | Set-Content -LiteralPath verbatim.txt -Encoding utf8" % VERBATIM

PATCH_ADD = ("*** Begin Patch\n*** Add File: notes.txt\n+first line\n"
             "+tags <b> & \"quotes\" and </parameter> kept\n+unicode ✓\n*** End Patch")
PATCH_UPDATE = ("*** Begin Patch\n*** Update File: notes.txt\n@@\n"
                " tags <b> & \"quotes\" and </parameter> kept\n-unicode ✓\n"
                "+unicode ✓ updated\n*** End Patch")
NOTES = "first line\ntags <b> & \"quotes\" and </parameter> kept\nunicode ✓ updated\n"

BIG = ["line %02d: if (a < b && c > d) { s = \"q\" + 'r'; } // é ✓" % i
       for i in range(1, 81)]
BIG[10] = "<tool_call> is only text in here"
BIG[20] = "</parameter> with text after it is still the value"
BIG[30] = "\tindented with a tab"
BIG[40] = "trailing spaces   "
BIG[50] = ""
BIG[60] = "*** End Patch is not the end when it is inside a line"
PATCH_BIG = ("*** Begin Patch\n*** Add File: big.txt\n"
             + "".join("+%s\n" % line for line in BIG) + "*** End Patch")

PLAN = [{"step": "write the file", "status": "completed"},
        {"step": "check it", "status": "in_progress"}]

EDGES = ("Compare 1 < 2 and 3 > 2; the tag <b> is literal; a stray <tool_c is text "
         "too; and it ends with <")

# A 1x1 red PNG, so the image scenario needs nothing but the stub's fake vision tower.
DOT_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c63f8cfc0000003010100c9fe92ef0000000049454e44ae426082")


# --- one run ----------------------------------------------------------------------

def _jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    return out


class Run:
    """What one scenario left behind: Codex's JSONL events, the request bodies it
    sent, the prompts the scripted model was shown, and its scratch directory."""

    def __init__(self, work, rc, stdout, stderr):
        self.work, self.rc, self.stderr = work, rc, stderr
        self.cwd = os.path.join(work, "cwd")
        self.events = []
        for line in stdout.splitlines():
            if line.startswith("{"):
                try:
                    self.events.append(json.loads(line))
                except ValueError:
                    pass
        self.items = [e["item"] for e in self.events if e.get("type") == "item.completed"]
        self.requests = _jsonl(os.path.join(work, "requests.jsonl"))
        self.prompts = _jsonl(os.path.join(work, "prompts.jsonl"))
        self.results = []

    def expect(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))

    def of(self, kind, every=False):
        src = ([e.get("item") or {} for e in self.events if e.get("type", "").startswith("item.")]
               if every else self.items)
        return [i for i in src if i.get("type") == kind]

    @property
    def commands(self):
        return self.of("command_execution")

    @property
    def said(self):
        return [i.get("text") or "" for i in self.of("agent_message")]

    @property
    def finished(self):
        return self.rc == 0 and any(e.get("type") == "turn.completed" for e in self.events)

    def history(self, k):
        """The input items of the k-th request (0-based): what Codex sent back."""
        if k >= len(self.requests):
            return []
        return [i for i in self.requests[k].get("input") or [] if isinstance(i, dict)]

    def calls(self, k):
        return [i for i in self.history(k) if i.get("type") in ("function_call", "custom_tool_call")]

    def outputs(self, k):
        return [i for i in self.history(k)
                if i.get("type") in ("function_call_output", "custom_tool_call_output")]

    def prompt(self, k):
        return self.prompts[k]["prompt"] if k < len(self.prompts) else ""

    def file(self, name):
        path = os.path.join(self.cwd, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8-sig", "replace").replace("\r\n", "\n")

    def tail(self):
        seen = ["%s%s" % (e.get("type"), ":%s" % (e.get("item") or {}).get("type") if "item" in e else "")
                for e in self.events]
        errs = [e.get("message") or (e.get("error") or {}).get("message")
                for e in self.events if e.get("type") in ("error", "turn.failed")]
        return "rc=%s events=%s errors=%s stderr=%s" % (
            self.rc, seen, errs, " | ".join(self.stderr.strip().splitlines()[-3:])[:400])


def args_of(item):
    try:
        return json.loads(item.get("arguments") or "null")
    except ValueError:
        return None


def ran(r, marker):
    return [c for c in r.commands if marker in (c.get("command") or "")]


def expect_ran(r, command):
    """Codex ran the command, and it printed what `echo X` (or a bare `X`) prints."""
    hit = ran(r, command)
    c = hit[0] if hit else {}
    printed = command[len("echo "):] if command.startswith("echo ") else command
    r.expect("Codex ran it: %s" % command,
             hit and c.get("exit_code") == 0 and printed in (c.get("aggregated_output") or ""),
             json.dumps(c)[:300] if hit else "commands: %s" % [x.get("command") for x in r.commands])


def expect_basics(r, requests, final):
    r.expect("the turn completed", r.finished, "" if r.finished else r.tail())
    r.expect("%d model request(s)" % requests, len(r.requests) == requests,
             "got %d" % len(r.requests))
    r.expect("final answer: %r" % final[:40], r.said and r.said[-1] == final,
             "said %r" % r.said)


def expect_reply(r, k, name, want_args=None, namespace=None):
    """Request k carries the call back -- named, namespaced and typed as Codex took
    it -- and its output under the same call_id."""
    calls = r.calls(k)
    c = calls[-1] if calls else {}
    got = args_of(c) if c.get("type") == "function_call" else c.get("input")
    ok = c.get("name") == name and c.get("namespace") == namespace
    if want_args is not None:
        ok = ok and got == want_args
    outs = {o.get("call_id") for o in r.outputs(k)}
    r.expect("history returns %s%s%s" % (namespace + "." if namespace else "", name,
                                         " with its arguments intact" if want_args is not None else ""),
             ok and c.get("call_id") in outs,
             "" if ok else "call=%s" % json.dumps(c, ensure_ascii=False)[:300])
    return c


# --- scenarios --------------------------------------------------------------------

def s_xml(r):
    expect_ran(r, "echo collabosm-xml-ok")
    expect_reply(r, 1, "shell_command", {"command": "echo collabosm-xml-ok"})
    r.expect("the prose before the call is its own message", "I will run it." in r.said, r.said)
    p = r.prompt(1)
    r.expect("next prompt replays the call as the model's own XML",
             call("shell_command", command="echo collabosm-xml-ok") in p)
    r.expect("  and the output as a <tool_response>",
             "<tool_response>" in p and "collabosm-xml-ok" in p.split("<tool_response>", 1)[-1])
    expect_basics(r, 2, "It printed collabosm-xml-ok.")


def s_typed(r):
    expect_ran(r, "echo typed-ok")
    c = expect_reply(r, 1, "shell_command",
                     {"command": "echo typed-ok", "timeout_ms": 20000, "login": False})
    a = args_of(c) or {}
    r.expect("  a number arrives as a number, a boolean as a boolean",
             type(a.get("timeout_ms")) is int and a.get("login") is False, a)
    c = expect_reply(r, 2, "shell_command", {"command": "1234", "timeout_ms": 5000})
    r.expect("  a string-only value that looks like a number stays a string",
             (args_of(c) or {}).get("command") == "1234")
    expect_ran(r, "1234")
    expect_basics(r, 3, "Typed arguments arrived typed.")


def s_verbatim(r):
    c = expect_reply(r, 1, "shell_command", {"command": VERBATIM_CMD})
    r.expect("  the multi-line value came through byte for byte",
             (args_of(c) or {}).get("command") == VERBATIM_CMD)
    got = r.file("verbatim.txt")
    r.expect("the command Codex ran wrote exactly that text",
             got is not None and got.rstrip("\n") == VERBATIM, repr(got)[:200])
    expect_basics(r, 2, "Written verbatim.")


def s_json_body(r):
    expect_ran(r, "echo collabosm-json-ok")
    expect_reply(r, 1, "shell_command", {"command": "echo collabosm-json-ok", "timeout_ms": 15000})
    expect_basics(r, 2, "The JSON form ran.")


def s_no_close(r):
    expect_ran(r, "echo collabosm-noclose-1")
    expect_ran(r, "echo collabosm-noclose-2")
    expect_reply(r, 1, "shell_command", {"command": "echo collabosm-noclose-1"})
    expect_reply(r, 2, "shell_command", {"command": "echo collabosm-noclose-2"})
    expect_basics(r, 3, "Both ran without their closing tags.")


def s_patch(r):
    c = expect_reply(r, 1, "apply_patch", PATCH_ADD)
    r.expect("  it went out as a freeform custom_tool_call", c.get("type") == "custom_tool_call", c.get("type"))
    expect_reply(r, 2, "apply_patch", PATCH_UPDATE)
    changes = [(ch.get("kind"), os.path.basename(ch.get("path") or ""))
               for i in r.of("file_change") if i.get("status") == "completed"
               for ch in i.get("changes") or []]
    r.expect("Codex applied both: add, then update", changes == [("add", "notes.txt"),
                                                                  ("update", "notes.txt")], changes)
    got = r.file("notes.txt")
    r.expect("the file holds exactly what the patches say", got == NOTES, repr(got)[:200])
    r.expect("next prompt replays the patch as the model wrote it",
             call("apply_patch", input=PATCH_ADD) in r.prompt(1))
    expect_basics(r, 3, "Patched twice.")


def s_big_patch(r):
    expect_reply(r, 1, "apply_patch", PATCH_BIG)
    got = r.file("big.txt")
    want = "\n".join(BIG) + "\n"
    r.expect("an 80-line patch streamed one character at a time lands intact", got == want,
             "" if got == want else "first difference at %s" % next(
                 (i for i, (a, b) in enumerate(zip(got or "", want)) if a != b), "the end"))
    expect_basics(r, 2, "The big file is in place.")


def s_plan(r):
    c = expect_reply(r, 1, "update_plan", {"explanation": "Two steps.", "plan": PLAN})
    r.expect("  the plan arrives as an array of objects, not a string",
             isinstance((args_of(c) or {}).get("plan"), list))
    todo = r.of("todo_list", every=True)
    last = todo[-1].get("items") if todo else []
    r.expect("Codex shows it as its to-do list",
             [(t.get("text"), t.get("completed")) for t in last or []]
             == [("write the file", True), ("check it", False)], last)
    out = (r.outputs(1) or [{}])[0].get("output")
    r.expect("  and answers the call", out == "Plan updated", out)
    expect_basics(r, 2, "Plan recorded.")


def s_namespace(r):
    expect_reply(r, 1, "close_agent", {"target": "agent-that-does-not-exist"},
                 namespace="multi_agent_v1")
    out = str((r.outputs(1) or [{}])[0].get("output"))
    r.expect("Codex routed it to the multi-agent tool (not 'unsupported call')",
             out and "unsupported" not in out.lower(), out[:160])
    expect_basics(r, 2, "There was no such agent.")


def s_serial(r):
    r.expect("Codex asked for one call at a time", r.requests and
             r.requests[0].get("parallel_tool_calls") is False,
             r.requests and r.requests[0].get("parallel_tool_calls"))
    expect_ran(r, "echo serial-one")
    r.expect("the second call was not handed over", not ran(r, "serial-two"),
             [c.get("command") for c in r.commands])
    r.expect("  and the history holds one call", len(r.calls(1)) == 1, len(r.calls(1)))
    expect_basics(r, 2, "Only the first one ran.")


def s_parallel(r):
    r.expect("Codex allowed parallel calls", r.requests and
             r.requests[0].get("parallel_tool_calls") is True,
             r.requests and r.requests[0].get("parallel_tool_calls"))
    expect_ran(r, "echo parallel-one")
    expect_ran(r, "echo parallel-two")
    calls, outs = r.calls(1), r.outputs(1)
    r.expect("history holds both calls and both outputs, ids paired",
             len(calls) == 2 and {c.get("call_id") for c in calls} == {o.get("call_id") for o in outs}
             and len({c.get("call_id") for c in calls}) == 2, (len(calls), len(outs)))
    r.expect("next prompt shows the model both results", r.prompt(1).count("<tool_response>") == 2,
             r.prompt(1).count("<tool_response>"))
    expect_basics(r, 2, "Both ran.")


def s_cut(r):
    r.expect("the half-written call never ran", not r.commands,
             [c.get("command") for c in r.commands])
    first = r.history(0)
    retry = r.history(1)
    r.expect("Codex treated the cut-off response as failed and asked again",
             len(r.requests) >= 2 and not r.calls(1)
             and [i for i in retry if i.get("type") != "message"] == []
             and len(retry) >= len(first), "requests=%d" % len(r.requests))
    noted = [e for e in r.events if "max_output_tokens" in json.dumps(e)]
    r.expect("  and said why (max_output_tokens)", noted, r.tail())
    expect_basics(r, 2, "Recovered after the cut.")


def s_think(r):
    r.expect("the template opened the think block", r.prompt(0).rstrip().endswith("<think>"))
    expect_ran(r, "echo collabosm-think-ok")
    rs = [i for i in r.history(1) if i.get("type") == "reasoning"]
    text = "".join(p.get("text", "") for i in rs for p in i.get("content") or [])
    r.expect("Codex sent the thought back as a reasoning item",
             text == "The user wants a check, so I will echo.", repr(text)[:120])
    r.expect("  and the model sees it again, before that step's call",
             "<think>\nThe user wants a check, so I will echo.\n</think>" in r.prompt(1))
    r.expect("no think tag leaked into what Codex showed",
             not any("think>" in s for s in r.said), r.said)
    expect_basics(r, 2, "It worked while thinking.")


def s_image(r):
    outs = r.outputs(1)
    parts = outs[0].get("output") if outs else None
    r.expect("view_image came back with the picture in its output",
             isinstance(parts, list) and any(p.get("type") == "input_image"
                                             and str(p.get("image_url", "")).startswith("data:image/")
                                             for p in parts), str(parts)[:160])
    p = r.prompt(1)
    r.expect("the model is shown it inside the tool result",
             r.prompts[1:2] and r.prompts[1].get("images") == 1
             and "<|image_pad|>" in p.split("<tool_response>", 1)[-1],
             r.prompts[1].get("images") if r.prompts[1:2] else "no second prompt")
    expect_basics(r, 2, "A small red square.")


def s_edges(r):
    r.expect("no call was made", not r.commands and not r.calls(0))
    expect_basics(r, 1, EDGES)


SCENARIOS = [
    dict(name="xml", about="one call in Qwen3-Coder XML, the whole turn in one fragment",
         chunk=4096, check=s_xml,
         turns=["I will run it.\n\n" + call("shell_command", command="echo collabosm-xml-ok"),
                "It printed collabosm-xml-ok."]),
    dict(name="typed", about="number and boolean values typed by the schema; a string that looks like a number",
         check=s_typed,
         turns=[call("shell_command", command="echo typed-ok", timeout_ms=20000, login=False),
                call("shell_command", command="1234", timeout_ms=5000),
                "Typed arguments arrived typed."]),
    dict(name="verbatim", about="a multi-line value with tags, quotes, backslashes and unicode",
         check=s_verbatim,
         turns=[call("shell_command", command=VERBATIM_CMD), "Written verbatim."]),
    dict(name="json-body", about="the older JSON body inside <tool_call>", check=s_json_body,
         turns=['<tool_call>\n{"name": "shell_command", "arguments": {"command": '
                '"echo collabosm-json-ok", "timeout_ms": 15000}}\n</tool_call>',
                "The JSON form ran."]),
    dict(name="no-close", about="the model stops before </tool_call>, then before </function>",
         check=s_no_close,
         turns=["Running.\n\n<tool_call>\n<function=shell_command>\n<parameter=command>\n"
                "echo collabosm-noclose-1\n</parameter>\n</function>",
                "<tool_call>\n<function=shell_command>\n<parameter=command>\n"
                "echo collabosm-noclose-2\n</parameter>",
                "Both ran without their closing tags."]),
    dict(name="patch", about="freeform apply_patch: add a file, then update it", check=s_patch,
         turns=["Adding the file.\n\n" + call("apply_patch", input=PATCH_ADD),
                call("apply_patch", input=PATCH_UPDATE), "Patched twice."]),
    dict(name="big-patch", about="an 80-line patch, streamed one character per fragment",
         chunk=1, check=s_big_patch,
         turns=[call("apply_patch", input=PATCH_BIG), "The big file is in place."]),
    dict(name="plan", about="update_plan: an array of objects as a parameter", check=s_plan,
         turns=[call("update_plan", explanation="Two steps.", plan=PLAN), "Plan recorded."]),
    dict(name="namespace", about="a tool from Codex's multi_agent_v1 namespace", check=s_namespace,
         turns=[call("close_agent", target="agent-that-does-not-exist"), "There was no such agent."]),
    dict(name="serial", about="two calls in one turn when the client asked for one at a time",
         check=s_serial,
         turns=["Two at once.\n\n" + call("shell_command", command="echo serial-one") + "\n"
                + call("shell_command", command="echo serial-two"), "Only the first one ran."]),
    dict(name="parallel", about="two calls in one turn, parallel calls allowed", parallel=True,
         check=s_parallel,
         turns=["Two at once.\n\n" + call("shell_command", command="echo parallel-one") + "\n"
                + call("shell_command", command="echo parallel-two"), "Both ran."]),
    dict(name="cut", about="the token ceiling lands inside a call's value", check=s_cut,
         turns=[{"text": "Writing it.\n\n<tool_call>\n<function=shell_command>\n"
                         "<parameter=command>\necho collabosm-half", "eos": "max_new_tokens"},
                "Recovered after the cut."]),
    dict(name="think", about="thinking on: a think block, a call, the thought replayed",
         env={"THINKING_DEFAULT": "1"}, check=s_think,
         turns=["The user wants a check, so I will echo.\n</think>\n\nChecking now.\n\n"
                + call("shell_command", command="echo collabosm-think-ok"),
                "The echo came back.\n</think>\n\nIt worked while thinking."]),
    dict(name="image", about="view_image: a picture inside a tool's output", stub_args=["--vision"],
         files={"dot.png": DOT_PNG}, check=s_image,
         turns=[call("view_image", path="dot.png"), "A small red square."]),
    dict(name="text-edges", about="an answer full of '<' with tools offered, and no call",
         check=s_edges, turns=[EDGES]),
]


# --- plumbing ---------------------------------------------------------------------

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def ensure_template(model_dir):
    """The pack's own chat template is what renders tools and replays history, so the
    run is only meaningful with it: fetch it from the pinned revision if missing."""
    path = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(path):
        return path
    env = registry.launch_env(registry.load(), registry.DEFAULT)
    url = "https://huggingface.co/%s/resolve/%s/chat_template.jinja" % (
        env["MODEL_REPO"], env["MODEL_REVISION"])
    print("fetching the pack's chat template: %s" % url, flush=True)
    os.makedirs(model_dir, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    with open(path + ".part", "wb") as fh:
        fh.write(data)
    os.replace(path + ".part", path)
    return path


def catalog(model_id, parallel):
    """Codex's model metadata for our model: freeform apply_patch, a shell tool, no
    reasoning levels of its own (thinking is the server's default, or not)."""
    return {"models": [{
        "slug": model_id, "display_name": model_id, "description": "collabosm (scripted)",
        "base_instructions": "You are Codex, a coding agent. Use the tools you are given.",
        "supported_reasoning_levels": [], "shell_type": "shell_command", "visibility": "list",
        "supported_in_api": True, "priority": 1, "availability_nux": None, "upgrade": None,
        "supports_reasoning_summaries": False, "support_verbosity": False,
        "default_verbosity": None, "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": parallel, "context_window": 262144,
        "experimental_supported_tools": []}]}


def run_scenario(sc, codex, model_dir, model_id, keep, timeout):
    work = tempfile.mkdtemp(prefix="collabosm-codex-%s-" % sc["name"])
    home, cwd = os.path.join(work, "home"), os.path.join(work, "cwd")
    os.makedirs(home)
    os.makedirs(cwd)
    for name, data in (sc.get("files") or {}).items():
        with open(os.path.join(cwd, name), "wb") as fh:
            fh.write(data)
    fwd = lambda p: p.replace("\\", "/")  # noqa: E731 -- TOML wants no backslashes
    with open(os.path.join(work, "turns.json"), "w", encoding="utf-8") as fh:
        json.dump(sc["turns"], fh, ensure_ascii=False)
    with open(os.path.join(work, "catalog.json"), "w", encoding="utf-8") as fh:
        json.dump(catalog(model_id, sc.get("parallel", False)), fh)
    port = free_port()
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write('model = "%s"\nmodel_provider = "collabosm"\napproval_policy = "never"\n'
                 'sandbox_mode = "danger-full-access"\nmodel_catalog_json = "%s"\n'
                 'show_raw_agent_reasoning = true\n\n'
                 '[model_providers.collabosm]\nname = "collabosm"\n'
                 'base_url = "http://127.0.0.1:%d/v1"\nwire_api = "responses"\n'
                 'env_key = "%s"\nstream_max_retries = 1\nrequest_max_retries = 0\n\n'
                 # no plugin marketplace: it is a git clone from GitHub on every start
                 '[features]\nplugins = false\n'
                 % (model_id, fwd(os.path.join(work, "catalog.json")), port, KEY_ENV))
    env = dict(os.environ, MODEL_DIR=model_dir, THINKING_DEFAULT="0",
               STUB_PROMPTS=os.path.join(work, "prompts.jsonl"),
               STUB_RECORD_REQUESTS=os.path.join(work, "requests.jsonl"),
               STUB_RECORD=os.path.join(work, "stub-record.json"), PYTHONIOENCODING="utf-8")
    env.update(sc.get("env") or {})
    log = open(os.path.join(work, "stub.log"), "w", encoding="utf-8")
    stub = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "dev_stub.py"), "--port", str(port),
         "--script", os.path.join(work, "turns.json"), "--chunk", str(sc.get("chunk", 3)),
         "--delay", "0"] + list(sc.get("stub_args") or []),
        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(150):
            try:
                socket.create_connection(("127.0.0.1", port), 0.3).close()
                break
            except OSError:
                if stub.poll() is not None:
                    break
                time.sleep(0.1)
        try:
            res = subprocess.run(
                [codex, "exec", "--json", "--skip-git-repo-check", sc.get("prompt", "go")],
                cwd=cwd, env=dict(os.environ, CODEX_HOME=home, **{KEY_ENV: "sk-scripted"}),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, stdin=subprocess.DEVNULL)
            rc, out, err = res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired as exc:
            rc, out, err = "timeout", exc.stdout or "", "codex did not finish in %ds" % timeout
            out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
    finally:
        stub.kill()
        stub.wait()
        log.close()
    for name, text in (("codex.jsonl", out), ("codex.err", err)):
        with open(os.path.join(work, name), "w", encoding="utf-8") as fh:
            fh.write(text or "")
    r = Run(work, rc, out, err)
    try:
        sc["check"](r)
    except Exception as exc:                  # a check that crashes is a failed check
        r.expect("checks ran", False, repr(exc))
    if not keep and all(ok for _, ok, _ in r.results):
        for _ in range(10):                   # Windows: a handle can outlive its process briefly
            shutil.rmtree(work, ignore_errors=True)
            if not os.path.exists(work):
                break
            time.sleep(0.5)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("only", nargs="*", help="scenario names (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--codex", default=os.environ.get("CODEX") or shutil.which("codex"))
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR")
                    or os.path.join(REPO, "_dev_model"))
    ap.add_argument("--jobs", type=int, default=4, help="scenarios run at once")
    ap.add_argument("--timeout", type=int, default=180, help="seconds per Codex run")
    ap.add_argument("--keep", action="store_true", help="keep every scenario's files")
    a = ap.parse_args()

    if a.list:
        for sc in SCENARIOS:
            print("%-11s %s" % (sc["name"], sc["about"]))
        return 0
    chosen = [sc for sc in SCENARIOS if not a.only or sc["name"] in a.only]
    unknown = set(a.only) - {sc["name"] for sc in SCENARIOS}
    if unknown or not chosen:
        print("unknown scenario(s): %s (see --list)" % ", ".join(sorted(unknown)))
        return 2
    if not a.codex:
        print("codex not found: put it on PATH or pass --codex")
        return 2
    a.model_dir = os.path.abspath(a.model_dir)   # the stub runs from the repo root
    ensure_template(a.model_dir)
    model_id = registry.launch_env(registry.load(), registry.DEFAULT)["MODEL_ID"]
    version = subprocess.run([a.codex, "--version"], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL).stdout.strip()
    print("%s, %d scenario(s), %d at a time\n" % (version, len(chosen), a.jobs), flush=True)

    failed, total = [], 0
    with concurrent.futures.ThreadPoolExecutor(max(1, a.jobs)) as pool:
        futs = {pool.submit(run_scenario, sc, a.codex, a.model_dir, model_id, a.keep,
                            a.timeout): sc for sc in chosen}
        for fut in concurrent.futures.as_completed(futs):
            sc, r = futs[fut], fut.result()
            with PRINT:
                print("== %s: %s" % (sc["name"], sc["about"]))
                for name, ok, detail in r.results:
                    total += 1
                    if not ok:
                        failed.append("%s: %s" % (sc["name"], name))
                    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                                          "" if ok or detail in ("", None) else "  -- %s" % (detail,)))
                if a.keep or not all(ok for _, ok, _ in r.results):
                    print("  files: %s" % r.work)
                print(flush=True)
    print("%d checks, %d failed" % (total, len(failed)))
    for f in failed:
        print("FAILED: " + f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
