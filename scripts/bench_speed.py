"""Speed of the running box, from its own timings: `python scripts/bench_speed.py`.

Through the frontend (http://127.0.0.1:3020/v1 by default), so it measures what a client
gets; the numbers are the server's `timings`, so the tunnel's round trip is not in them.

- cold prefill at a few lengths: pairs, each prompt with a unique nonce so nothing is
  cached; the second of a pair is the figure to trust (the first may pay a kernel autotune)
- decode at short, medium and long context: 512 tokens of an essay, the long prompts
  reused from the prefill runs so they are cached
- an agent-like turn: the longest prompt cached plus a few hundred new tokens -- the
  time before the first token there is what an agent waits on every turn

It sends real requests to a box that bills by the hour; the whole run takes a few
minutes, most of it the longest prefills. Nothing is kept on the box but its cache.
"""
import argparse
import json
import random
import time
import urllib.request

WORDS = ("river mountain quiet engine window harvest silver lantern orbit canyon meadow signal "
         "copper garden thunder pocket marble ribbon cedar planet velvet harbor needle falcon "
         "summit crystal bridge compass ember glacier timber violet anchor saddle wander "
         "orchard beacon pepper linen quartz raven willow tundra marsh cobalt fable prism "
         "cinder delta echo fjord gravel hollow island jasmine kettle lagoon mosaic nectar").split()
ESSAY = ("\n\nIgnore the text above. Write a long, detailed essay (at least 800 words) about the "
         "history of bridge engineering, from stone arches to modern cable-stayed bridges.")


def filler(n_words, seed):
    rnd = random.Random(seed)
    out, sent = [], []
    for _ in range(n_words):
        sent.append(rnd.choice(WORDS))
        if len(sent) >= rnd.randint(8, 16):
            out.append(" ".join(sent).capitalize() + ".")
            sent = []
    if sent:
        out.append(" ".join(sent).capitalize() + ".")
    return " ".join(out)


def ask(base, key, content, max_tokens, timeout=900):
    body = {"model": "x", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_wall"] = time.time() - t0
    return out


def row(label, out):
    t, u = out.get("timings") or {}, out.get("usage") or {}
    print("%-36s prompt %7s  cached %7s  new %6s  prefill %8s t/s (%6.1f s)  out %4s  decode %6s t/s"
          % (label, u.get("prompt_tokens"), t.get("cache_n"), t.get("prompt_n"),
             t.get("prompt_per_second"), (t.get("prompt_ms") or 0) / 1000.0,
             t.get("predicted_n"), t.get("predicted_per_second")), flush=True)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:3020/v1")
    ap.add_argument("--key", default="sk-local", help="the frontend replaces it with the box's own")
    ap.add_argument("--lengths", default="4096,32768,120000",
                    help="cold prefill lengths in tokens, comma-separated")
    ap.add_argument("--decode", type=int, default=512, help="tokens per decode run")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]

    cal = ask(a.base, a.key, "n%d " % time.time_ns() + filler(2000, 1), 1)
    tpw = (cal["usage"]["prompt_tokens"] - 20) / 2000.0
    print("calibration: %.3f tokens per filler word" % tpw, flush=True)

    kept = {}
    for target in lengths:
        for run in (0, 1):
            text = ("Nonce %d-%d-%d. " % (time.time_ns(), target, run)
                    + filler(int(target / tpw), target * 10 + run))
            row("cold prefill %7d (r%d)" % (target, run), ask(a.base, a.key, text, 1))
            kept[target] = text                  # the second run's prompt stays cached
    print(flush=True)
    row("decode @ short context", ask(a.base, a.key, "Nonce %d." % time.time_ns() + ESSAY, a.decode))
    for target in lengths[1:]:
        row("decode @ %d context" % target, ask(a.base, a.key, kept[target] + ESSAY, a.decode))
    print(flush=True)
    longest = kept[lengths[-1]]
    row("agent turn: %dK cached + ~400 new" % (lengths[-1] // 1000),
        ask(a.base, a.key, longest + ESSAY + " " + filler(int(400 / tpw), 777) + " Answer briefly.", 64))


if __name__ == "__main__":
    main()
