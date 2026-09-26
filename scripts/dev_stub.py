"""Loopback harness: serve the real collabosm Handler with a stubbed engine.

The protocol is what keeps breaking, and the real server cannot answer anything
until ~78 GB of weights are resident. Iterating on the wire format through a
tunnel and a four-minute reload is therefore both slow and expensive. This
harness imports the *shipping* Handler, replaces only the engine seam, and
serves it on loopback so a real client (Codex, ModelDock, curl) drives the exact
code path that ships.

    python scripts/dev_stub.py --port 8099 --chunk 24 --delay 0.02

    # with a think block, to exercise the reasoning item lifecycle
    python scripts/dev_stub.py --port 8099 --think

    # a fixed list of model turns, played one per generation (scripts/check_codex.py)
    python scripts/dev_stub.py --port 8099 --script turns.json

Auth is disabled (COLLABOSM_NO_AUTH=1). Loopback only -- never expose this port.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import types
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ["COLLABOSM_NO_AUTH"] = "1"
os.environ.setdefault("MODEL_DIR", os.path.join(HERE, os.pardir, "_dev_model"))

# api_server imports exllamav3 at module scope. Stand in for it before import so
# the handler can be exercised on a machine with no CUDA and no weights.
_stub = types.ModuleType("exllamav3")


class _Generator:  # only the name has to exist
    pass


class _ModelInit:
    @staticmethod
    def add_args(*_a, **_k):
        return None

    @staticmethod
    def init(*_a, **_k):
        raise RuntimeError("dev_stub serves no model")


class _Job:
    pass


_stub.Generator = _Generator
_stub.model_init = _ModelInit
# Make it look like the real package layout so `from exllamav3.generator import
# Job` resolves exactly as it does on the VM.
_stub.__path__ = []
_gen = types.ModuleType("exllamav3.generator")
_gen.__path__ = []
_gen.Generator = _Generator
_gen.Job = _Job
sys.modules.setdefault("exllamav3", _stub)
sys.modules.setdefault("exllamav3.generator", _gen)

import api_server  # noqa: E402

ANSWER = (
    "stub answer: the collabosm API surface is up. This text is deliberately "
    "long enough that a correct incremental stream produces several delta "
    "events before the terminal event, while a broken one produces a single "
    "blob or no terminal event at all."
)
THOUGHTS = (
    "stub reasoning: no tokens were actually generated. The engine is faked so "
    "the protocol can be exercised without a GPU."
)


class _FakeVision:
    """Stands in for ExModel(component="vision") so the image path can be exercised
    end to end without a GPU: it hands back the same text_alias contract the real
    tower produces."""

    def __init__(self):
        self.calls = 0

    def get_image_embeddings(self, tokenizer=None, image=None, text_alias=None):
        if image is None:
            raise ValueError("fake vision got no image")
        self.calls += 1
        return types.SimpleNamespace(text_alias="<|image_pad|>", embeddings=None,
                                     tokenizer=None, image_size=getattr(image, "size", None))


RECORD = os.environ.get("STUB_RECORD",
                        os.path.join(tempfile.gettempdir(), "collabosm-stub-last.json"))


HISTORY = []


def _record(payload):
    """Keep a short history, not just the last call: a test has to be able to look
    at the request that carried an image even though a text-only call follows it."""
    HISTORY.append(payload)
    try:
        with open(RECORD, "w", encoding="utf-8") as f:
            json.dump(HISTORY[-10:], f)
    except Exception:
        pass


SAMPLE = {"string": "stub", "integer": 1, "number": 1.5, "boolean": True,
          "array": ["stub"], "object": {}}


# A command a real client may actually run: harmless, and its output recognisable.
SAMPLE_COMMAND = "echo collabosm-tool-ok"


def tool_call_for(prompt):
    """What this model writes when it calls a tool -- Qwen3-Coder XML -- aimed at the
    first tool the rendered prompt offers, every required parameter filled.

    A turn whose last message is a tool's result gets an answer instead, quoting
    that result: a real client (Codex) then sees one call, runs it, sends the output
    back and gets a final reply -- a conversation, not a loop of calls."""
    last_user = prompt.rfind("<|im_start|>user")
    if last_user >= 0 and "<tool_response>" in prompt[last_user:]:
        seen = prompt.rsplit("<tool_response>", 1)[1].split("</tool_response>", 1)[0].strip()
        return "The tool answered: %s" % (seen.splitlines() or ["(nothing)"])[0][:200]
    i = prompt.find("<tools>\n")
    if i < 0:
        return None
    first = prompt[i + len("<tools>\n"):].split("\n", 1)[0]
    try:
        tool = json.loads(first)
    except ValueError:
        return None
    fn = tool.get("function", tool)
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    lines = ["I will call %s." % fn.get("name"), "", "<tool_call>",
             "<function=%s>" % fn.get("name")]
    for name in params.get("required") or list(props)[:1]:
        kind = (props.get(name) or {}).get("type", "string")
        value = SAMPLE.get(kind if isinstance(kind, str) else "string", "stub")
        if name in ("command", "cmd") and kind == "string":
            value = SAMPLE_COMMAND
        lines += ["<parameter=%s>" % name,
                  value if isinstance(value, str) else json.dumps(value), "</parameter>"]
    return "\n".join(lines + ["</function>", "</tool_call>"])


def install_engine(answer, thoughts, chunk, delay, think, silent=False, vision=False,
                   tool_call=False, cut_call=False):
    plain = ("<think>%s</think>\n%s" % (thoughts, answer)) if think else answer

    def _engine_fragments(prompt, max_tokens, temperature=None, top_p=None,
                          stops=None, embeddings=None):
        """Same contract as the shipping seam: yield raw text fragments."""
        call = tool_call_for(prompt) if tool_call else None
        if call is not None and cut_call and "<tool_call>" in call:
            # the token ceiling lands inside the call's first value
            call = call[:call.index("<parameter=")] + "<parameter=command>\necho half-writ"
        text = plain if call is None else (
            ("<think>%s</think>\n%s" % (thoughts, call)) if think else call)
        _record({"prompt": prompt, "embeddings": len(embeddings or []),
                 "vision_calls": getattr(api_server.VISION, "calls", 0)})
        api_server.LAST.clear()
        api_server.LAST["prompt_tokens"] = 29
        if silent:
            # An engine that stops on the first token: nothing to stream. This is
            # the shape that produced empty answers on the live A100 pack, so the
            # harness has to be able to reproduce it.
            api_server.LAST["new_tokens"] = 1
            api_server.LAST["eos_reason"] = "stop_token"
            return
        for i in range(0, len(text), chunk):
            if delay:
                time.sleep(delay)
            yield text[i:i + chunk]
        api_server.LAST["new_tokens"] = len(text) // 4
        api_server.LAST["eos_reason"] = ("max_new_tokens" if call is not None and cut_call
                                         and "<tool_call>" in call else "stop_token")

    def _generate_blocking(prompt, max_tokens, temperature=None, top_p=None,
                           stops=None, embeddings=None):
        """The one-shot path the server falls back to (same signature as shipped)."""
        if delay:
            time.sleep(delay * 3)
        # A complete answer: the server now marks max_new_tokens as a truncation
        # (Responses status "incomplete"), which is not what this stub is standing in for.
        return {"text": plain, "prompt_tokens": 29, "cached_tokens": 0,
                "new_tokens": len(plain) // 4, "eos_reason": "stop_token"}

    api_server._engine_fragments = _engine_fragments
    api_server._generate_blocking = _generate_blocking
    api_server.STOP_IDS = []
    raw = os.environ.get("STUB_RECORD_REQUESTS")
    if raw:
        # the request bodies a client really sends (what a replayed history holds)
        orig = api_server.Handler._responses

        def _responses(self, req):
            try:
                with open(raw, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(req, ensure_ascii=False) + "\n")
            except Exception:
                pass
            return orig(self, req)

        api_server.Handler._responses = _responses
    if vision:
        api_server.VISION = _FakeVision()
        api_server.VISION_WANTED = True
        api_server.VISION_ERR = None


def install_script(path, chunk, delay):
    """Play a fixed list of model turns, one per generation, in order.

    Each turn is the exact text a model writes after the prompt -- prose, a think
    block's rest and its </think>, tool calls in whichever syntax -- plus why it
    stopped: {"text": ..., "eos": "stop_token" | "max_new_tokens"}, or just the text.
    A real client driving this sees those bytes go through the shipping parser, so a
    test knows precisely which syntax each turn exercises. A generation past the
    last turn answers "(script exhausted)", which a test can see.

    With STUB_PROMPTS set, every rendered prompt is appended there as JSON lines
    ({"turn", "prompt", "images"}): what the model would have been shown."""
    with open(path, encoding="utf-8") as fh:
        turns = [t if isinstance(t, dict) else {"text": t} for t in json.load(fh)]
    played = [0]
    prompts = os.environ.get("STUB_PROMPTS")

    def _engine_fragments(prompt, max_tokens, temperature=None, top_p=None,
                          stops=None, embeddings=None):
        n = played[0]
        played[0] += 1
        turn = turns[n] if n < len(turns) else {"text": "(script exhausted)"}
        text = turn["text"]
        if prompts:
            with open(prompts, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"turn": n, "prompt": prompt,
                                     "images": len(embeddings or [])}, ensure_ascii=False) + "\n")
        api_server.LAST.clear()
        api_server.LAST["prompt_tokens"] = len(prompt) // 4
        for i in range(0, len(text), chunk):
            if delay:
                time.sleep(delay)
            yield text[i:i + chunk]
        api_server.LAST["new_tokens"] = max(1, len(text) // 4)
        api_server.LAST["eos_reason"] = turn.get("eos") or "stop_token"

    api_server._engine_fragments = _engine_fragments
    api_server.STOP_IDS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--chunk", type=int, default=32,
                    help="characters per fragment (how visible the deltas are)")
    ap.add_argument("--delay", type=float, default=0.02,
                    help="seconds between fragments")
    ap.add_argument("--think", action="store_true",
                    help="emit a <think> block so the reasoning item is exercised")
    ap.add_argument("--silent", action="store_true",
                    help="engine yields nothing, to exercise the blocking fallback")
    ap.add_argument("--vision", action="store_true",
                    help="install a fake vision tower so image input can be tested")
    ap.add_argument("--tool-call", action="store_true",
                    help="when the request offers tools, answer with a call to the first one")
    ap.add_argument("--cut-call", action="store_true",
                    help="with --tool-call: the token ceiling lands inside the call")
    ap.add_argument("--script", metavar="JSON",
                    help="play these model turns in order, one per generation")
    a = ap.parse_args()

    install_engine(ANSWER, THOUGHTS, a.chunk, a.delay, a.think, a.silent, a.vision,
                   a.tool_call, a.cut_call)
    if a.script:
        install_script(a.script, a.chunk, a.delay)
    srv = ThreadingHTTPServer((a.host, a.port), api_server.Handler)
    print("[dev_stub] listening on http://%s:%d  (chunk=%d delay=%.3fs think=%s)"
          % (a.host, a.port, a.chunk, a.delay, a.think), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()