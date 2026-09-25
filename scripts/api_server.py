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
import base64
import io
import ipaddress
import json
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("MODEL_DIR", "/content/exl3")
SRC = os.environ.get("EXL3_SRC", "/content/exllamav3-src")
if os.path.isdir(os.path.join(SRC, "examples")):
    sys.path.insert(0, os.path.join(SRC, "examples"))

from exllamav3 import Generator, model_init  # noqa: E402

try:
    from exllamav3 import Model as ExModel  # noqa: E402
except Exception:  # pragma: no cover - older builds
    ExModel = None

try:
    from exllamav3.generator import Job  # noqa: E402
except Exception:  # pragma: no cover - older builds
    try:
        from exllamav3.generator.job import Job  # noqa: E402
    except Exception:
        Job = None

try:
    from jinja2 import Template as JTemplate
except Exception:
    JTemplate = None

LOCK = threading.Lock()
GEN = None
TOK = None
ARGS = None
STOP_IDS = []
SERVER_STARTED = 0

# The vision tower is a SEPARATE component: model_init.init() loads only "text" (or
# "mtp"), so a multimodal pack answers as text-only unless we load it ourselves.
# Off by default: this box already sits at 76.4/81.9 GiB, and the tower's VRAM cost
# has not been measured here yet. VISION=1 enables it.
VISION = None
VISION_ERR = None
VISION_WANTED = os.environ.get("VISION", "0") == "1"

# Image input is data: URLs ONLY by default. This server is published through a
# tunnel, so fetching a caller-supplied URL would be an SSRF primitive aimed at the
# VM's own metadata service -- and reading a caller-supplied path would be a local
# file read. IMAGE_URLS=1 opts into remote fetch, still restricted to public hosts.
IMAGE_URLS = os.environ.get("IMAGE_URLS", "0") == "1"
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 12 * 1024 * 1024))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", 8))

# What the server was launched with, for the read-only status contract.
LAUNCH = {}


class MMUnavailable(Exception):
    """A request carries an image this server cannot (or will not) embed."""

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


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def template_kwargs_from(req):
    """Thinking controls, as this pack's own template expects them.

    The template takes `enable_thinking` (false pre-closes the think block, so the
    model answers directly) and `reasoning_effort` (xhigh | medium | low). Default
    here is thinking OFF, which is what plain chat clients expect; turn it on with
    enable_thinking: true or a reasoning_effort value.
    """
    kw = {"enable_thinking": False}
    if isinstance(req.get("enable_thinking"), bool):
        kw["enable_thinking"] = req["enable_thinking"]
    ck = req.get("chat_template_kwargs")
    if isinstance(ck, dict):
        if isinstance(ck.get("enable_thinking"), bool):
            kw["enable_thinking"] = ck["enable_thinking"]
        if ck.get("reasoning_effort") in ("xhigh", "medium", "low"):
            kw["reasoning_effort"] = ck["reasoning_effort"]
    eff = req.get("reasoning_effort")
    if eff in ("xhigh", "medium", "low"):
        kw["enable_thinking"] = True
        kw["reasoning_effort"] = eff
    return kw


def split_thinking(text):
    """(reasoning, answer), reported the way Qwen/DeepSeek endpoints do.

    Split on the LAST </think>: when thinking is disabled the template pre-closes
    the think block, and the model can echo another closing marker, so the first
    one is not necessarily the real boundary. Any leftover marker is stripped from
    both halves.
    """
    t = text or ""
    i = t.rfind(THINK_CLOSE)
    if i >= 0:
        reasoning, answer = t[:i], t[i + len(THINK_CLOSE):]
    elif t.lstrip().startswith(THINK_OPEN):
        reasoning, answer = t, ""
    else:
        reasoning, answer = "", t
    for marker in (THINK_OPEN, THINK_CLOSE):
        reasoning = reasoning.replace(marker, "")
        answer = answer.replace(marker, "")
    return reasoning.strip(), answer.strip()


def assistant_delta(text, reasoning):
    d = {"role": "assistant"}
    if reasoning:
        d["reasoning_content"] = reasoning
    d["content"] = text
    return d


def assistant_message(text, reasoning):
    msg = {"role": "assistant", "content": text}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def build_engine():
    """Load once, with the flags the repository measured as best."""
    global GEN, TOK, ARGS, STOP_IDS, SERVER_STARTED
    SERVER_STARTED = int(time.time())
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
    LAUNCH.update({
        "cache_size": int(os.environ.get("CACHE_SIZE", 262144)),
        "cache_quant": os.environ.get("CACHE_QUANT", "4"),
        "generator_chunk_size": gcs,
        "num_draft_tokens": ndt,
        "cpu_cache_gb": ccs,
        "recurrent_cache_gb": rcs,
        "mode": os.environ.get("CHAT_MODE", "chatml"),
        "mtp": True,
    })

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

    global VISION, VISION_ERR
    prep = os.path.join(MODEL_DIR, "preprocessor_config.json")
    if not VISION_WANTED:
        print("[api] vision is OFF (set VISION=1 to load the tower). "
              "Image parts will be refused, never silently dropped.", flush=True)
    elif ExModel is None:
        VISION_ERR = "this exllamav3 build has no Model(component=...) API"
        print("[api] vision unavailable: %s" % VISION_ERR, flush=True)
    elif not os.path.exists(prep):
        VISION_ERR = ("no preprocessor_config.json in %s: this pack carries no "
                      "vision tower" % MODEL_DIR)
        print("[api] vision unavailable: %s" % VISION_ERR, flush=True)
    else:
        try:
            t_v = time.time()
            VISION = ExModel.from_config(config, component="vision")
            VISION.load(progressbar=False)
            print("[api] vision component loaded in %.1fs" % (time.time() - t_v),
                  flush=True)
        except Exception as exc:
            VISION = None
            VISION_ERR = repr(exc)
            print("[api] vision component FAILED to load: %s" % VISION_ERR, flush=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """One hop only: a redirect must not be able to bounce us to an internal host
    after the address check has already passed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _public_only(url):
    """Reject anything that is not an ordinary public address."""
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        raise MMUnavailable("image URL has no host")
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise MMUnavailable("cannot resolve image host %r: %r" % (host, exc))
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            raise MMUnavailable("unparseable address for %r" % host)
        if not ip.is_global:
            raise MMUnavailable(
                "refusing to fetch %s: %s is not a public address (private, "
                "loopback, link-local or metadata)" % (url, ip))
    return url


def _fetch_image(url):
    _public_only(url)
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "collabosm"})
    try:
        with opener.open(req, timeout=20) as resp:
            raw = resp.read(MAX_IMAGE_BYTES + 1)
    except Exception as exc:
        raise MMUnavailable("could not fetch image: %r" % exc)
    if len(raw) > MAX_IMAGE_BYTES:
        raise MMUnavailable("image exceeds MAX_IMAGE_BYTES (%d)" % MAX_IMAGE_BYTES)
    return raw


def _prepare_ref(ref):
    """Validate an image reference -> ("data", bytes) | ("url", url).

    Policy is checked here, deliberately BEFORE the capability check: a caller who
    sends a filesystem path should be told the reference is unsupported whether or
    not this server happens to have vision, and a remote URL must be refused even
    when no tower is loaded (otherwise a URL would be fetched and only then
    rejected). Only data: URLs are accepted unless IMAGE_URLS=1, and a caller can
    never make this server read its own disk or reach the cloud metadata service.
    """
    if isinstance(ref, dict):
        ref = ref.get("url") or ref.get("data") or ref.get("b64_json")
    if not isinstance(ref, str) or not ref.strip():
        raise MMUnavailable("image part carries no url/data")
    ref = ref.strip()
    if ref.startswith("data:"):
        _, _, payload = ref.partition(",")
        try:
            raw = base64.b64decode(payload, validate=False)
        except Exception as exc:
            raise MMUnavailable("image data URL is not valid base64: %r" % exc)
        if len(raw) > MAX_IMAGE_BYTES:
            raise MMUnavailable("image exceeds MAX_IMAGE_BYTES (%d)" % MAX_IMAGE_BYTES)
        return ("data", raw)
    if ref.startswith(("http://", "https://")):
        if not IMAGE_URLS:
            raise MMUnavailable(
                "remote image URLs are disabled (set IMAGE_URLS=1 to allow them). "
                "Send the image as a base64 data: URL instead.")
        return ("url", ref)
    raise MMUnavailable("unsupported image reference: expected a data: URL"
                        + (" or a public http(s) URL" if IMAGE_URLS else ""))


def _image_embedding(ref):
    """Embed one image -> (MMEmbedding, the prompt alias that stands in for it).

    The alias matters: it is inserted as text where the image belongs, and
    tokenizer.encode(..., embeddings=[...]) expands it into the placeholder span.
    """
    kind, payload = _prepare_ref(ref)
    if VISION is None:
        raise MMUnavailable(VISION_ERR or (
            "vision is not enabled on this server (VISION=1) -- refusing to answer "
            "as if the image were text"))
    try:
        from PIL import Image
    except Exception as exc:
        raise MMUnavailable("image input needs Pillow in the runtime: %r" % exc)
    raw = payload if kind == "data" else _fetch_image(payload)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    try:
        ie = VISION.get_image_embeddings(tokenizer=TOK, image=img)
    except TypeError:
        ie = VISION.get_image_embeddings(TOK, img)
    alias = getattr(ie, "text_alias", None)
    if not alias:
        raise MMUnavailable("the vision model returned no prompt alias for this image")
    return ie, alias


def flatten_parts(content, embs):
    """Fold OpenAI content parts into one string, inlining image aliases."""
    if not isinstance(content, list):
        return content
    out = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = (part.get("type") or "").lower()
        if ptype in ("image_url", "input_image", "image") or "image_url" in part:
            if len(embs) >= MAX_IMAGES:
                raise MMUnavailable("too many images in one request (MAX_IMAGES=%d)"
                                    % MAX_IMAGES)
            ie, alias = _image_embedding(part.get("image_url")
                                         or part.get("image") or part.get("url"))
            embs.append(ie)
            out.append(alias)
            continue
        out.append(part.get("text") or "")
    return "".join(out)


def extract_images(req, embs):
    """Rewrite image parts in place so they survive the text pipeline.

    Covers both dialects: chat `content` parts and Responses `input` items. Raises
    MMUnavailable rather than dropping an image, because answering as if the picture
    were text is exactly the bug this replaced.
    """
    for key in ("messages", "input"):
        items = req.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("content"), list):
                item["content"] = flatten_parts(item["content"], embs)


def render_chat(messages, template_kwargs=None):
    """Prefer the model's own chat template; fall back to ChatML."""
    tpl_path = os.path.join(MODEL_DIR, "chat_template.jinja")
    if JTemplate is not None and os.path.exists(tpl_path):
        try:
            src = open(tpl_path, encoding="utf-8").read()
            kw = {"messages": messages, "add_generation_prompt": True}
            if template_kwargs:
                kw.update(template_kwargs)
            out = JTemplate(src).render(**kw)
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


LAST = {}


def _engine_fragments(prompt, max_tokens, temperature=None, top_p=None, stops=None,
                      embeddings=None):
    """Yield decoded text fragments from exllamav3 as they are produced.

    Real streaming goes through enqueue() + iterate(): iterate() reports a
    "streaming" stage result for every decode step, so the HTTP layer can forward
    text while the job is still running. generate() cannot do that -- it
    accumulates the completion internally and returns only once the job is done,
    which is why this server used to go silent and then emit one blob.

    Every exllamav3-version-specific detail lives in here. Callers only ever see a
    fragment iterator, and the terminal metrics land in LAST.
    """
    stops_all = list(STOP_IDS) + ["<|im_end|>"] + list(stops or [])
    LAST.clear()
    if embeddings:
        # Special-token encoding is on for images: the alias IS a special token, and
        # only this call expands it into the span the embeddings line up with.
        try:
            input_ids = TOK.encode(prompt, encode_special_tokens=True, add_bos=True,
                                   embeddings=embeddings)
        except TypeError:
            input_ids = TOK.encode(prompt, add_bos=True, embeddings=embeddings)
    else:
        try:
            input_ids = TOK.encode(prompt, encode_special_tokens=False, add_bos=True)
        except TypeError:
            input_ids = TOK.encode(prompt, add_bos=True)
    n_prompt = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(input_ids)
    LAST["prompt_tokens"] = n_prompt
    # Mirror Generator.generate()'s own Job construction field for field. A Job
    # built with fewer fields behaves differently: it stopped after a single
    # token here, which is what an empty answer looks like from the client side.
    job = Job(input_ids=input_ids,
              max_new_tokens=max_tokens,
              min_new_tokens=0,
              stop_conditions=stops_all,
              sampler=None,
              filters=[],
              token_healing=False,
              decode_special_tokens=bool(embeddings),
              embeddings=list(embeddings or []),
              max_rq_tokens=None,
              stop_on_loop=None)
    serial = GEN.enqueue(job)
    while GEN.num_remaining_jobs():
        for r in GEN.iterate():
            if r.get("stage") == "error":
                # A contained per-job failure. Surface it rather than returning a
                # silently truncated completion.
                raise r["error"]
            if r.get("stage") != "streaming" or r.get("serial") != serial:
                continue
            frag = r.get("text") or ""
            if frag:
                yield frag
            if r.get("eos"):
                LAST["new_tokens"] = r.get("new_tokens")
                LAST["eos_reason"] = r.get("eos_reason")
                LAST["prompt_tokens"] = r.get("prompt_tokens") or n_prompt


def _generate_blocking(prompt, max_tokens, temperature=None, top_p=None, stops=None,
                       embeddings=None):
    """The old one-shot path, kept as the fallback for builds whose Job API differs."""
    kw = {"stop_conditions": list(STOP_IDS) + ["<|im_end|>"] + list(stops or [])}
    if embeddings:
        kw["embeddings"] = list(embeddings)
        kw["encode_special_tokens"] = True
        kw["decode_special_tokens"] = True
    if temperature is not None:
        kw["temperature"] = float(temperature)
    if top_p is not None:
        kw["top_p"] = float(top_p)
    with LOCK:
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


def collect(prompt, max_tokens, temperature=None, top_p=None, stops=None,
            on_delta=None, embeddings=None):
    """The one engine entry point, for streaming and non-streaming alike.

    on_delta(cumulative_clean_text) is called while the model is still decoding;
    pass None for a one-shot generation. Returns the same metrics dict the old
    generate() did, so non-streaming callers are unchanged.
    """
    buf = []
    try:
        with LOCK:
            for frag in _engine_fragments(prompt, max_tokens, temperature, top_p, stops,
                                          embeddings):
                buf.append(frag)
                if on_delta is not None:
                    on_delta(clean_completion("".join(buf)))
    except TypeError as exc:
        print("[api] enqueue path failed (%r); falling back to blocking generate()"
              % exc, flush=True)
        return _generate_blocking(prompt, max_tokens, temperature, top_p, stops,
                                  embeddings)
    out = dict(LAST)
    out["text"] = clean_completion("".join(buf))
    if not out["text"].strip():
        # Never answer empty. The incremental path can yield nothing if a build's
        # Job defaults differ, or if the model stops on token one; the blocking
        # path is the one that is known to work on this pack, so use it rather
        # than handing the client an empty completion.
        out["new_tokens"] = out.get("new_tokens")
        print("[api] incremental path produced no text (new=%s eos=%s); retrying "
              "with the blocking generator" % (out.get("new_tokens"),
                                               out.get("eos_reason")), flush=True)
        r = _generate_blocking(prompt, max_tokens, temperature, top_p, stops,
                               embeddings)
        print("[api] blocking fallback: new=%s eos=%s text=%r"
              % (r.get("new_tokens"), r.get("eos_reason"),
                 (r.get("text") or "")[:60]), flush=True)
        if on_delta is not None and (r.get("text") or "").strip():
            on_delta(r["text"])
        return r
    return out


def generate(prompt, max_tokens, temperature=None, top_p=None, stops=None,
             embeddings=None):
    """One-shot generation; kept because callers and diagnostics still use it."""
    return collect(prompt, max_tokens, temperature, top_p, stops,
                   embeddings=embeddings)


class _Delta:
    """Turn a cumulative completion into ordered reasoning/content deltas.

    Thinking is split on the LAST </think> exactly as the non-streaming path does,
    so a client that streams and a client that does not end up with the same
    answer. A fragment that would require retracting bytes already sent is dropped
    instead: an SSE stream cannot unsend.
    """

    def __init__(self):
        self.sent_reasoning = ""
        self.sent_content = ""
        self.content_started = False

    def feed(self, accumulated):
        reasoning, answer = split_thinking(accumulated)
        out = []
        if (not self.content_started
                and reasoning.startswith(self.sent_reasoning)
                and len(reasoning) > len(self.sent_reasoning)):
            out.append(("reasoning", reasoning[len(self.sent_reasoning):]))
            self.sent_reasoning = reasoning
        if answer.startswith(self.sent_content) and len(answer) > len(self.sent_content):
            if answer:
                self.content_started = True
            out.append(("content", answer[len(self.sent_content):]))
            self.sent_content = answer
        return out


def responses_messages(req):
    """Map a Responses-API request onto chat messages."""
    src = req.get("input")
    if src is None:
        src = req.get("messages")
    msgs = []
    if isinstance(src, str):
        msgs.append({"role": "user", "content": src})
    elif isinstance(src, list):
        for item in src:
            if isinstance(item, str):
                msgs.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role") or "user"
                c = item.get("content")
                if isinstance(c, list):
                    text = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                elif isinstance(c, str):
                    text = c
                else:
                    text = item.get("text") or ""
                msgs.append({"role": role, "content": text})
    if isinstance(req.get("instructions"), str) and req["instructions"].strip():
        msgs.insert(0, {"role": "system", "content": req["instructions"]})
    if not msgs:
        msgs = [{"role": "user", "content": "hello"}]
    return msgs


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
        if code >= 400:
            # Errors are terminal for the connection. Otherwise a client that was
            # told "no" reuses the socket, and any bytes we never read get parsed
            # as the next request line.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    # --- SSE framing -----------------------------------------------------------
    # A streamed body with no Content-Length and no Transfer-Encoding can only be
    # ended by closing the socket. Through a tunnel hop that is not reliable: the
    # origin closes, the edge keeps the client's read open, and the client waits
    # for an EOF that never comes ("stream disconnected before completion").
    # Explicit HTTP/1.1 chunked framing makes the end unambiguous at every hop.
    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self._chunked = True

    def _sse_write(self, text):
        data = text.encode("utf-8")
        if getattr(self, "_chunked", False):
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        else:
            self.wfile.write(data)

    def _sse_event(self, name, payload):
        self._sse_write("event: %s\ndata: %s\n\n" % (name, json.dumps(payload)))

    def _sse_end(self):
        if getattr(self, "_chunked", False):
            self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        self.close_connection = True

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

    def send_error(self, code, message=None, explain=None):
        """Never answer with BaseHTTPRequestHandler's HTML error page.

        The HTML 501 a client saw was produced by THIS process, not by a second
        listener: an unread request body poisoned the keep-alive socket, and the
        next "request line" was the previous JSON body, which the base class
        reported as an unsupported method. Two listeners were never involved.
        Every error path here answers JSON, so a protocol failure can never be
        mistaken for a missing model.
        """
        try:
            self._send(code, {"error": {"message": str(message or "request error"),
                                        "type": "invalid_request_error"}})
        except Exception:
            pass

    def _reject_method(self):
        self._read_body()
        return self._send(405, {"error": {"message": "method not allowed: %s" % self.command,
                                          "type": "invalid_request_error"}})

    def do_PUT(self):
        return self._reject_method()

    def do_PATCH(self):
        return self._reject_method()

    def do_DELETE(self):
        return self._reject_method()

    def do_HEAD(self):
        self._read_body()
        if self.path.startswith("/health"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "2")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        return self._send(405, {"error": {"message": "use GET",
                                          "type": "invalid_request_error"}})

    def _read_body(self):
        """Consume the whole request body, however it is framed.

        Not consuming it is what broke every client probe: with HTTP/1.1
        keep-alive the stranded bytes are parsed as the next request line, which
        surfaces as `Unsupported method ('{"model":...}POST')` and HTTP 501.
        """
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            out = b""
            while True:
                line = self.rfile.readline(65536).strip()
                if not line:
                    break
                try:
                    n = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if n == 0:
                    self.rfile.readline(65536)
                    break
                out += self.rfile.read(n)
                self.rfile.readline(65536)
            return out
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            n = 0
        return self.rfile.read(n) if n > 0 else b""

    def do_GET(self):
        self._read_body()
        if self.path.startswith("/health"):
            return self._send(200, "ok", "text/plain")
        if not self._authorized():
            return self._send(401, {"error": {"message": "missing or bad API key",
                                              "type": "invalid_request_error"}})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [
                {"id": "qwen3.8-flash-next-exl3", "object": "model",
                 "created": SERVER_STARTED, "owned_by": "collabosm"}]})
        if self.path.startswith("/v1/status"):
            # Read-only contract: what this server can actually do, so a client does
            # not have to guess (and so a UI can grey out what is unavailable).
            cache_tokens = getattr(getattr(GEN, "cache", None), "max_num_tokens", None)
            return self._send(200, {
                "service": "collabosm",
                "model": "qwen3.8-flash-next-exl3",
                "started": SERVER_STARTED,
                "uptime_s": int(time.time()) - SERVER_STARTED if SERVER_STARTED else None,
                "cache_max_tokens": cache_tokens,
                "dialects": ["/v1/chat/completions", "/v1/responses"],
                "vision": {
                    "enabled": VISION_WANTED,
                    "available": VISION is not None,
                    "error": VISION_ERR,
                },
                "image_input": {
                    "data_urls": True,
                    "remote_urls": IMAGE_URLS,
                    "max_bytes": MAX_IMAGE_BYTES,
                    "max_images_per_request": MAX_IMAGES,
                },
                "launch": LAUNCH,
            })
        if self.path in ("/", "/v1"):
            return self._send(200, {"service": "collabosm",
                                    "model": "qwen3.8-flash-next-exl3",
                                    "endpoints": ["/v1/models", "/v1/chat/completions",
                                                  "/v1/responses", "/health"]})
        return self._send(404, {"error": {"message": "not found: %s" % self.path}})

    def do_POST(self):
        # Read first, always: every early return below used to strand the body.
        raw = self._read_body()
        if not (self.path.startswith("/v1/chat/completions")
                or self.path.startswith("/v1/responses")):
            return self._send(404, {"error": {"message": "not found: %s" % self.path}})
        if not self._authorized():
            return self._send(401, {"error": {"message": "missing or bad API key",
                                              "type": "invalid_request_error"}})
        try:
            req = json.loads(raw or b"{}")
        except Exception as exc:
            return self._send(400, {"error": {"message": "bad json: %r" % exc}})
        if self.path.startswith("/v1/responses"):
            return self._responses(req)
        return self._completions(req)

    def _completions(self, req):
        msgs = req.get("messages") or []
        embs = []
        try:
            # Before anything reads the content: image parts become prompt aliases
            # plus the embeddings that stand behind them, or the request is refused.
            extract_images(req, embs)
        except MMUnavailable as exc:
            return self._send(400, {"error": {"message": str(exc),
                                              "type": "vision_unavailable"}})
        if not isinstance(msgs, list) or not msgs:
            return self._send(400, {"error": {"message": "messages must be a non-empty array",
                                              "type": "invalid_request_error",
                                              "param": "messages"}})
        prompt, how = render_chat(msgs, template_kwargs_from(req))
        # OpenAI renamed max_tokens -> max_completion_tokens; accept both.
        max_tokens = int(req.get("max_completion_tokens")
                         or req.get("max_tokens") or 512)
        stream = bool(req.get("stream"))
        stop = req.get("stop")
        stops = [stop] if isinstance(stop, str) else list(stop or [])

        cid = "chatcmpl-%d" % int(time.time() * 1000)
        model = req.get("model", "qwen3.8-flash-next-exl3")

        def finish_reason(r):
            return "length" if (r.get("eos_reason") or "") == "max_new_tokens" else "stop"

        def report(r):
            pt_ = r.get("prompt_tokens") or 0
            ct_ = r.get("cached_tokens") or 0
            nt_ = r.get("new_tokens") or 0
            hit = (100.0 * ct_ / pt_) if pt_ else 0.0
            print("[api] template=%s prompt=%d cached=%d (%.1f%% hit) new=%d finish=%s"
                  % (how, pt_, ct_, hit, nt_, finish_reason(r)), flush=True)

        def usage_for(r):
            pt_ = r.get("prompt_tokens") or 0
            ct_ = r.get("cached_tokens") or 0
            nt_ = r.get("new_tokens") or 0
            return {"prompt_tokens": pt_, "completion_tokens": nt_,
                    "total_tokens": pt_ + nt_,
                    "prompt_tokens_details": {"cached_tokens": ct_}}

        def chunk(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        if not stream:
            try:
                r = collect(prompt, max_tokens, req.get("temperature"),
                            req.get("top_p"), stops, embeddings=embs)
            except Exception as exc:
                traceback.print_exc()
                return self._send(500, {"error": {"message": repr(exc)}})
            reasoning, text = split_thinking(r.get("text") or "")
            report(r)
            return self._send(200, {
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "finish_reason": finish_reason(r),
                             "message": assistant_message(text, reasoning)}],
                "usage": usage_for(r)})

        # Genuinely incremental: text is forwarded while the engine decodes it,
        # rather than being chunked up after the completion is already finished.
        self._sse_start()
        delta = _Delta()
        started = [False]

        def emit(deltas):
            for kind, piece in deltas:
                if not started[0]:
                    self._sse_write("data: " + json.dumps(
                        chunk({"role": "assistant", "content": ""})) + "\n\n")
                    started[0] = True
                if kind == "reasoning":
                    self._sse_write("data: " + json.dumps(
                        chunk({"reasoning_content": piece})) + "\n\n")
                else:
                    self._sse_write("data: " + json.dumps(
                        chunk({"content": piece})) + "\n\n")

        try:
            r = collect(prompt, max_tokens, req.get("temperature"), req.get("top_p"),
                        stops, on_delta=lambda acc: emit(delta.feed(acc)),
                        embeddings=embs)
        except Exception as exc:
            traceback.print_exc()
            self._sse_write("data: " + json.dumps(
                {"error": {"message": repr(exc)}}) + "\n\n")
            self._sse_write("data: [DONE]\n\n")
            return self._sse_end()
        emit(delta.feed(r.get("text") or ""))
        report(r)
        self._sse_write("data: " + json.dumps(chunk({}, finish_reason(r))) + "\n\n")
        so = req.get("stream_options")
        if isinstance(so, dict) and so.get("include_usage"):
            self._sse_write("data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model, "choices": [],
                "usage": usage_for(r)}) + "\n\n")
        self._sse_write("data: [DONE]\n\n")
        self._sse_end()

    # ------------------------------------------------------------- Responses API
    def _responses(self, req):
        """Minimal OpenAI Responses surface.

        Clients probe both this and /v1/chat/completions; answering only one of
        them is what produced "supports neither Responses nor Chat Completions".

        Codex's Responses parser binds every delta to an item it was told about
        and will not accept a stream that ends without a terminal event, so the
        whole lifecycle is emitted: created -> in_progress -> [reasoning item] ->
        message item -> deltas -> done -> completed. Every event carries
        sequence_number, and the ids in the terminal event are the same ids the
        stream announced.
        """
        embs = []
        try:
            extract_images(req, embs)
        except MMUnavailable as exc:
            return self._send(400, {"error": {"message": str(exc),
                                              "type": "vision_unavailable"}})
        msgs = responses_messages(req)
        prompt, how = render_chat(msgs, template_kwargs_from(req))
        max_tokens = int(req.get("max_output_tokens") or req.get("max_tokens") or 512)
        stream = bool(req.get("stream"))
        now = int(time.time())
        model = req.get("model", "qwen3.8-flash-next-exl3")
        base = now * 1000
        rid = "resp_%d" % base
        mid = "msg_%d" % (base + 1)
        rsn_id = "rs_%d" % (base + 2)

        def payload_for(r, text, reasoning):
            pt = r.get("prompt_tokens") or 0
            nt = r.get("new_tokens") or 0
            output = []
            if reasoning:
                output.append({"id": rsn_id, "type": "reasoning", "status": "completed",
                               "summary": [{"type": "summary_text", "text": reasoning}]})
            output.append({"id": mid, "type": "message", "role": "assistant",
                           "status": "completed",
                           "content": [{"type": "output_text", "text": text,
                                        "annotations": []}]})
            return {
                "id": rid,
                "object": "response",
                "created_at": now,
                "status": "completed",
                "model": model,
                "output": output,
                "output_text": text,
                "parallel_tool_calls": True,
                "tool_calls": [],
                "reasoning": ({"summary": [{"type": "summary_text", "text": reasoning}]}
                              if reasoning else None),
                "usage": {"input_tokens": pt, "output_tokens": nt,
                          "total_tokens": pt + nt,
                          "input_tokens_details": {"cached_tokens": r.get("cached_tokens") or 0},
                          "output_tokens_details": {"reasoning_tokens": 0}},
                "incomplete_details": None,
                "error": None,
            }

        if not stream:
            try:
                r = collect(prompt, max_tokens, req.get("temperature"), req.get("top_p"),
                            embeddings=embs)
            except Exception as exc:
                traceback.print_exc()
                return self._send(500, {"error": {"message": repr(exc)}})
            reasoning, text = split_thinking(r.get("text") or "")
            print("[api] /v1/responses template=%s prompt=%s new=%s eos=%s text=%r"
                  % (how, r.get("prompt_tokens"), r.get("new_tokens"),
                     r.get("eos_reason"), text[:60]), flush=True)
            return self._send(200, payload_for(r, text, reasoning))

        seq = [0]

        def ev(name, obj):
            body = dict(obj)
            body["type"] = name
            body["sequence_number"] = seq[0]
            seq[0] += 1
            self._sse_event(name, body)

        self._sse_start()
        rmeta = {"id": rid, "object": "response", "created_at": now, "model": model,
                 "status": "in_progress", "output": []}
        ev("response.created", {"response": rmeta})
        ev("response.in_progress", {"response": rmeta})

        delta = _Delta()
        state = {"open": None, "idx": 0}

        def reasoning_item(status, text):
            return {"id": rsn_id, "type": "reasoning", "status": status,
                    "summary": ([{"type": "summary_text", "text": text}] if text else [])}

        def message_item(status, text):
            return {"id": mid, "type": "message", "role": "assistant", "status": status,
                    "content": ([{"type": "output_text", "text": text,
                                  "annotations": []}] if text else [])}

        def close_reasoning():
            if state["open"] != "reasoning":
                return
            ev("response.reasoning_text.done",
               {"item_id": rsn_id, "output_index": state["idx"], "content_index": 0,
                "text": delta.sent_reasoning})
            ev("response.output_item.done",
               {"output_index": state["idx"],
                "item": reasoning_item("completed", delta.sent_reasoning)})
            state["open"] = None
            state["idx"] += 1

        def open_message():
            if state["open"] == "message":
                return
            ev("response.output_item.added",
               {"output_index": state["idx"], "item": message_item("in_progress", "")})
            ev("response.content_part.added",
               {"item_id": mid, "output_index": state["idx"], "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}})
            state["open"] = "message"

        def close_message(text):
            if state["open"] != "message":
                return
            ev("response.output_text.done",
               {"item_id": mid, "output_index": state["idx"], "content_index": 0,
                "text": text})
            ev("response.content_part.done",
               {"item_id": mid, "output_index": state["idx"], "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []}})
            ev("response.output_item.done",
               {"output_index": state["idx"], "item": message_item("completed", text)})
            state["open"] = None

        def emit(deltas):
            for kind, piece in deltas:
                if kind == "reasoning":
                    if state["open"] is None:
                        ev("response.output_item.added",
                           {"output_index": state["idx"],
                            "item": reasoning_item("in_progress", "")})
                        state["open"] = "reasoning"
                    if state["open"] == "reasoning":
                        ev("response.reasoning_text.delta",
                           {"item_id": rsn_id, "output_index": state["idx"],
                            "content_index": 0, "delta": piece})
                else:
                    close_reasoning()
                    open_message()
                    ev("response.output_text.delta",
                       {"item_id": mid, "output_index": state["idx"],
                        "content_index": 0, "delta": piece})

        try:
            r = collect(prompt, max_tokens, req.get("temperature"), req.get("top_p"),
                        on_delta=lambda acc: emit(delta.feed(acc)), embeddings=embs)
        except Exception as exc:
            traceback.print_exc()
            ev("response.failed", {"response": {
                "id": rid, "object": "response", "created_at": now, "model": model,
                "status": "failed",
                "error": {"code": "server_error", "message": repr(exc)}}})
            return self._sse_end()

        final = r.get("text") or ""
        reasoning, text = split_thinking(final)
        emit(delta.feed(final))
        close_reasoning()
        open_message()
        close_message(text)
        print("[api] /v1/responses template=%s prompt=%s new=%s"
              % (how, r.get("prompt_tokens"), r.get("new_tokens")), flush=True)
        ev("response.completed", {"response": payload_for(r, text, reasoning)})
        self._sse_end()


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
