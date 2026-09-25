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


def install_engine(answer, thoughts, chunk, delay, think, silent=False, vision=False):
    text = ("<think>%s</think>\n%s" % (thoughts, answer)) if think else answer

    def _engine_fragments(prompt, max_tokens, temperature=None, top_p=None,
                          stops=None, embeddings=None):
        """Same contract as the shipping seam: yield raw text fragments."""
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
        api_server.LAST["eos_reason"] = "stop_token"

    def _generate_blocking(prompt, max_tokens, temperature=None, top_p=None,
                           stops=None):
        """The one-shot path the server falls back to."""
        if delay:
            time.sleep(delay * 3)
        return {"text": text, "prompt_tokens": 29, "cached_tokens": 0,
                "new_tokens": len(text) // 4, "eos_reason": "max_new_tokens"}

    api_server._engine_fragments = _engine_fragments
    api_server._generate_blocking = _generate_blocking
    api_server.STOP_IDS = []
    if vision:
        api_server.VISION = _FakeVision()
        api_server.VISION_WANTED = True
        api_server.VISION_ERR = None


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
    a = ap.parse_args()

    install_engine(ANSWER, THOUGHTS, a.chunk, a.delay, a.think, a.silent, a.vision)
    srv = ThreadingHTTPServer((a.host, a.port), api_server.Handler)
    print("[dev_stub] listening on http://%s:%d  (chunk=%d delay=%.3fs think=%s)"
          % (a.host, a.port, a.chunk, a.delay, a.think), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()