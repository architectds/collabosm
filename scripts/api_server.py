#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for ExLlamaV3 (collabosm).

Why this exists
---------------
exllamav3 v1.5.1 ships NO server: `examples/` contains chat.py / chat_console.py and
nothing that speaks HTTP, and TabbyAPI (the usual front end) does not expose the
parameters this setup depends on -- the pinned-RAM second-tier KV page cache
(`-ccs`), the recurrent checkpoint store (`-rcs`) and the generator chunk size
(`-gcs`). So this is a deliberately small server built on the library the rest of
the kit already uses.

Two details that are easy to get wrong and are load-bearing here:

1. ONE long-lived Generator. The PageTable and the CPU page cache are constructed
   inside Generator.__init__, so a Generator per request silently throws away prompt
   caching and the warm KV tier. This process builds exactly one and keeps it.
2. Requests are serialised. One Generator owns one cache of `cache_size` tokens, and
   `max_num_tokens` is the budget across concurrent jobs, so the lock is not a
   limitation of the server, it is the shape of the engine.

Surface: GET /health, GET /v1/models, POST /v1/chat/completions (stream and
non-stream). That is all on purpose -- this is how you point a client at the
measured-best configuration, not a production gateway.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("MODEL_DIR", "/content/exl3")
SRC = os.environ.get("EXL3_SRC", "/content/exllamav3-src")
if os.path.isdir(os.path.join(SRC, "examples")):
    sys.path.insert(0, os.path.join(SRC, "examples"))

from exllamav3 import Generator, model_init  # noqa: E402

try:
    from jinja2 import Template as JTemplate
except Exception:
    JTemplate = None

LOCK = threading.Lock()
GEN = None
TOK = None
ARGS = None
STOP_IDS = []

# --- auth ---------------------------------------------------------------------
# serve.sh creates /content/api-key.txt and then publishes the port through a
# quick tunnel. An open tunnel to an 80 GB GPU is not something to ship by
# accident, so the key is enforced whenever it exists. Set COLLABOSM_NO_AUTH=1
# for a loopback-only smoke test.
KEY_FILE = os.environ.get("KEY_FILE", "/content/api-key.txt")
NO_AUTH = os.environ.get("COLLABOSM_NO_AUTH", "0") == "1"
API_KEY = None
if not NO_AUTH and os.path.exists(KEY_FILE):
    try:
        API_KEY = open(KEY_FILE).read().strip() or None
    except Exception as exc:
        print("[api] could not read %s: %r" % (KEY_FILE, exc), flush=True)


def discover_stop_ids():
    """Stop tokens for this model.

    config.json in this pack declares eos_token_id = null; the real EOS list lives
    in the HF-style generation_config.json ([248046 <|im_end|>, 248044
    <|endoftext|>]), which exllamav3 does not read. Without stop conditions the only
    way generation ends is max_new_tokens, and the model will happily write the next
    turn's markup as ordinary text instead of stopping.
    """
    ids = set()
    path = os.path.join(MODEL_DIR, "generation_config.json")
    if os.path.exists(path):
        try:
            e = json.load(open(path)).get("eos_token_id")
            if isinstance(e, int):
                ids.add(int(e))
            elif isinstance(e, (list, tuple)):
                ids.update(int(x) for x in e)
        except Exception as exc:
            print("[api] generation_config.json unreadable: %r" % exc, flush=True)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            enc = TOK.encode(tok, encode_special_tokens=True)
            vals = enc.flatten().tolist() if hasattr(enc, "flatten") else list(enc)
            ids.update(int(v) for v in vals)
        except Exception:
            pass
    return sorted(ids)


def clean_completion(text):
    """Cut the trailing markup a non-stopping model writes as plain text."""
    for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|endofprompt|>"):
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
    return text.strip()


def build_engine():
    """Load once, with the flags the repository measured as best."""
    global GEN, TOK, ARGS, STOP_IDS
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_sampling_args=True,
                        add_draft_model_args=True,
                        default_cache_size=int(os.environ.get("CACHE_SIZE", 262144)),
                        default_autosplit_max_batch_size=1)
    for flags, kw in [
        (["-mode", "--mode"], dict(type=str, default="chatml")),
        (["-gcs", "--generator_chunk_size"], dict(type=int, default=4096)),
    ]:
        parser.add_argument(*flags, **kw)

    gcs = int(os.environ.get("GCS", 4096))
    ndt = int(os.environ.get("NDT", 4))
    ccs = float(os.environ.get("CPU_CACHE_GB", 0) or 0)
    rcs = float(os.environ.get("RECURRENT_CACHE_GB", 4) or 0)

    argv = ["serve", "-m", MODEL_DIR,
            "-cs", str(os.environ.get("CACHE_SIZE", 262144)),
            "-cq", str(os.environ.get("CACHE_QUANT", "4")),
            "-ngr",                       # n-gram/PLE table in host RAM: mandatory here
            "-mtp", "-ndt", str(ndt),
            "-gcs", str(gcs),
            "-mode", os.environ.get("CHAT_MODE", "chatml")]
    if ccs:
        argv += ["-ccs", str(ccs)]        # pinned-RAM second-tier KV page cache
    if rcs:
        argv += ["-rcs", str(rcs)]        # GDN checkpoint store (host RAM)
    sys.argv = argv
    ARGS = parser.parse_args()

    t0 = time.time()
    loaded = model_init.init(ARGS)
    model, config, cache, tokenizer = loaded[0], loaded[1], loaded[2], loaded[3]
    draft_model, draft_cache = (loaded[4], loaded[6]) if len(loaded) >= 7 else (None, None)
    print("[api] model loaded in %.1fs, cache=%d tokens, draft=%s"
          % (time.time() - t0, cache.max_num_tokens, draft_model is not None), flush=True)

    # ONE Generator for the process lifetime -- see the module docstring.
    GEN = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache,
                    max_chunk_size=gcs,
                    cpu_cache_size=int(ccs * 1024 ** 3),
                    recurrent_cache_size=int(rcs * 1024 ** 3))
    TOK = tokenizer
    STOP_IDS = discover_stop_ids()
    print("[api] stop ids: %s" % STOP_IDS, flush=True)
    print("[api] generator ready (cpu tier: %s)"
          % (getattr(GEN, "cpu_page_cache", None) is not None), flush=True)


def render_chat(messages):
    """Prefer the model's own chat template; fall back to ChatML."""
    tpl_path = os.path.join(MODEL_DIR, "chat_template.jinja")
    if JTemplate is not None and os.path.exists(tpl_path):
        try:
            src = open(tpl_path, encoding="utf-8").read()
            out = JTemplate(src).render(messages=messages, add_generation_prompt=True)
            if isinstance(out, str) and out.strip():
                return out, "jinja"
        except Exception as exc:
            print("[api] chat template failed (%r); falling back to ChatML" % exc, flush=True)
    parts = []
    for m in messages:
        parts.append("<|im_start|>%s\n%s<|im_end|>\n"
                     % (m.get("role", "user"), m.get("content", "") or ""))
    parts.append("<|im_start|>assistant\n")
    return "".join(parts), "chatml"


def generate(prompt, max_tokens, temperature=None, top_p=None):
    # end the turn at <|im_end|>/EOS instead of running to max_tokens
    kw = {"stop_conditions": list(STOP_IDS) + ["<|im_end|>"]}
    if temperature is not None:
        kw["temperature"] = float(temperature)
    if top_p is not None:
        kw["top_p"] = float(top_p)
    with LOCK:
        # generate() returns (completions, last_results). Reading "text" out of
        # last_results gives only the FINAL fragment of the completion -- that is
        # the bug that made this server answer " inputs" to a real question.
        # completion_only=True excludes the echoed prompt; the except path strips
        # it by hand for builds that do not accept the flag.
        try:
            try:
                comp, last = GEN.generate(prompt, max_new_tokens=max_tokens, add_bos=True,
                                          return_last_results=True,
                                          completion_only=True, **kw)
            except TypeError:
                try:
                    comp, last = GEN.generate(prompt, max_new_tokens=max_tokens, add_bos=True,
                                              return_last_results=True,
                                              completion_only=True)
                except TypeError:
                    comp, last = GEN.generate(prompt, max_new_tokens=max_tokens, add_bos=True,
                                              return_last_results=True)
                    if isinstance(comp, str) and comp.startswith(prompt):
                        comp = comp[len(prompt):]
        except Exception:
            raise
    if isinstance(comp, (list, tuple)):
        comp = comp[0] if comp else ""
    if not isinstance(last, dict):
        last = getattr(last, "__dict__", {}) or {}
    last = dict(last)
    last["text"] = clean_completion(comp or "")
    return last


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "collabosm"

    def log_message(self, fmt, *a):
        print("[api] " + (fmt % a), flush=True)

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
        self.send_header("Access-Control-Allow-Methods", "POST,GET,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorized(self):
        if NO_AUTH or not API_KEY:
            return True
        h = self.headers.get("Authorization", "") or ""
        return h == ("Bearer " + API_KEY) or h == API_KEY

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, "ok", "text/plain")
        if not self._authorized():
            return self._send(401, {"error": {"message": "missing or bad API key",
                                              "type": "invalid_request_error"}})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [
                {"id": "qwen3.8-flash-next-exl3", "object": "model",
                 "owned_by": "collabosm"}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            return self._send(404, {"error": "not found"})
        if not self._authorized():
            return self._send(401, {"error": {"message": "missing or bad API key",
                                              "type": "invalid_request_error"}})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            return self._send(400, {"error": {"message": "bad json: %r" % exc}})

        msgs = req.get("messages") or []
        prompt, how = render_chat(msgs)
        max_tokens = int(req.get("max_tokens") or 512)
        stream = bool(req.get("stream"))

        try:
            r = generate(prompt, max_tokens, req.get("temperature"), req.get("top_p"))
        except Exception as exc:
            traceback.print_exc()
            return self._send(500, {"error": {"message": repr(exc)}})

        text = r.get("text") or ""
        pt = r.get("prompt_tokens") or 0
        ct = r.get("cached_tokens") or 0
        nt = r.get("new_tokens") or 0
        hit = (100.0 * ct / pt) if pt else 0.0
        print("[api] template=%s prompt=%d cached=%d (%.1f%% hit) new=%d"
              % (how, pt, ct, hit, nt), flush=True)

        cid = "chatcmpl-%d" % int(time.time() * 1000)
        if not stream:
            return self._send(200, {
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": req.get("model", "qwen3.8-flash-next-exl3"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": pt, "completion_tokens": nt,
                          "total_tokens": pt + nt}})

        # Streaming: this build has no incremental callback here, so the whole
        # completion is delivered as one delta followed by [DONE]. Clients that
        # only need SSE framing are happy; there is no token-by-token typing.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        for chunk in (
            {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
             "model": req.get("model", "qwen3.8-flash-next-exl3"),
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                          "finish_reason": None}]},
            {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
             "model": req.get("model", "qwen3.8-flash-next-exl3"),
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ):
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8090)))
    a = ap.parse_args()
    build_engine()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("[api] listening on http://%s:%d" % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()