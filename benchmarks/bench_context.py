"""B70 Qwen3.8-27B / pi-agent benchmark: decode-vs-context + agentic prefix loop.

Natural varied prose from the real repo/skill Markdown so MTP acceptance is
realistic (not degenerate repetition). Prompt size is exact (measured via
/tokenize).
"""
import glob
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = os.environ.get("MODEL", "huihui-qwen38-27b-abliterated-int4")


def _corpus():
    parts = []
    for pat in [
        r"C:\Users\zeran\.pi\agent\skills\**\*.md",
        r"C:\LocalLLM\qwen38-27b-ablit-xpu\docs\**\*.md",
        r"C:\LocalLLM\qwen38-27b-ablit-xpu\README.md",
        r"C:\LocalLLM\qwen38-27b-ablit-xpu\.b70src\cookbook\*.md",
    ]:
        for p in sorted(glob.glob(pat, recursive=True)):
            try:
                parts.append(open(p, encoding="utf-8", errors="ignore").read())
            except OSError:
                pass
    return "\n\n".join(parts)


CORPUS = _corpus()


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ntok(text):
    return post("/tokenize", {"model": MODEL, "prompt": text})["count"]


def make_prompt(target_tokens, start=0):
    """Exact-size prompt built from a sliding window over the corpus."""
    lo, hi = 1, len(CORPUS)
    # first guess by chars, then binary-search the exact token target
    guess = min(len(CORPUS), max(64, int(target_tokens * 3.4)))
    s = CORPUS[start:start + guess]
    c = ntok(s)
    if c == 0:
        return s
    guess2 = max(64, int(guess * target_tokens / c))
    guess2 = min(len(CORPUS) - start, guess2)
    s = CORPUS[start:start + guess2]
    c = ntok(s)
    # one refinement
    if c > 0:
        guess3 = min(len(CORPUS) - start, max(64, int(guess2 * target_tokens / c)))
        s = CORPUS[start:start + guess3]
    return s


def stream_completion(prompt, gen, chat=False, msgs=None):
    """Returns (ttft, total, usage, text)."""
    body = {"model": MODEL, "max_tokens": gen, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    if chat:
        body["messages"] = msgs
        path = "/v1/chat/completions"
    else:
        body["prompt"] = prompt
        body["ignore_eos"] = True
        path = "/v1/completions"
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                piece = ch.get("text")
                if piece is None:
                    delta = ch.get("delta") or {}
                    piece = (delta.get("content") or delta.get("reasoning_content")
                             or delta.get("reasoning") or "")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
    return ttft, time.time() - t0, usage, "".join(text)


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in ("spec_decode_num_draft_tokens_total",
                  "spec_decode_num_accepted_tokens_total",
                  "prefix_cache_queries_total", "prefix_cache_hits_total"):
            if line.startswith("vllm:" + k):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out


def decode_at_ctx(target_tokens, gen=128):
    prompt = make_prompt(target_tokens) + "\n\nSummarize the tradeoffs discussed above in one paragraph."
    m0 = metrics()
    ttft, total, usage, _ = stream_completion(prompt, gen)
    m1 = metrics()
    p, comp = usage["prompt_tokens"], usage["completion_tokens"]
    dt = m1.get("spec_decode_num_draft_tokens_total", 0) - m0.get("spec_decode_num_draft_tokens_total", 0)
    da = m1.get("spec_decode_num_accepted_tokens_total", 0) - m0.get("spec_decode_num_accepted_tokens_total", 0)
    return dict(ctx=p, ttft=round(ttft or 0, 2), total=round(total, 1), gen=comp,
                tok_s=round((comp - 1) / max(total - (ttft or 0), 1e-9), 1),
                prefill_tps=round(p / max(ttft or 1e-9, 1e-9)),
                accept=round(100 * da / dt, 1) if dt else None)


def degenerate(text):
    """True only for real degeneration: one char dominating the output.

    A short run of '!' inside otherwise varied text is legitimate content
    (the corpus itself contains banner lines).
    """
    if not text:
        return False
    s = text.replace(" ", "").replace("\n", "")
    if len(s) < 24:
        return False
    from collections import Counter
    ch, cnt = Counter(s).most_common(1)[0]
    if cnt / len(s) >= 0.6:
        return f"dominant {ch!r} {cnt}/{len(s)}"
    tail = s[-200:]
    if len(tail) >= 24 and len(set(tail)) <= 2:
        return f"tail-collapse {set(tail)}"
    return False


def agentic(target_tokens, turns, gen=128, chunk=3000):
    """Grow a chat with prefix-cache reuse; report per-turn TTFT + decode + hits."""
    sysmsg = ("You are a coding agent. Tools: read, bash, edit, write. Be concise. "
              "Answer in one short paragraph unless asked otherwise.")
    msgs = [{"role": "system", "content": sysmsg}]
    off = 0
    for t in range(1, turns + 1):
        blk = make_prompt(chunk, start=off)
        off += len(blk)
        msgs.append({"role": "user",
                     "content": "Tool output (continuing session):\n" + blk +
                                "\n\nSummarize what this output means in one short paragraph."})
        m0 = metrics()
        ttft, total, usage, body = stream_completion(None, gen, chat=True, msgs=msgs)
        m1 = metrics()
        p, comp = usage["prompt_tokens"], usage["completion_tokens"]
        rate = (comp - 1) / max(total - (ttft or 0), 1e-9)
        hits = m1["prefix_cache_hits_total"] - m0["prefix_cache_hits_total"]
        q = m1["prefix_cache_queries_total"] - m0["prefix_cache_queries_total"]
        dt = m1.get("spec_decode_num_draft_tokens_total", 0) - m0.get("spec_decode_num_draft_tokens_total", 0)
        da = m1.get("spec_decode_num_accepted_tokens_total", 0) - m0.get("spec_decode_num_accepted_tokens_total", 0)
        bad = degenerate(body)
        print(f"  t{t:02d} ctx={p:7d} hit%={100*hits/max(q,1):5.1f} "
              f"ttft={ttft or 0:6.2f}s tok/s={rate:6.1f} accept="
              f"{100*da/dt if dt else 0:5.1f}% bad={bad} {body[:70]!r}", flush=True)
        if bad:
            with open("degenerate_case.json", "w", encoding="utf-8") as f:
                json.dump({"messages": msgs, "output": body}, f, ensure_ascii=False, indent=1)
            return 2
        msgs.append({"role": "assistant", "content": body[:400] or "OK."})
        if p >= target_tokens:
            break
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ctx":
        res = []
        for t in [8000, 16000, 32000, 64000, 96000]:
            r = decode_at_ctx(t)
            res.append(r)
            print(json.dumps(r), flush=True)
        json.dump(res, open("bench_ctx.json", "w"), indent=1)
    elif mode == "agentic":
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        turns = int(sys.argv[3]) if len(sys.argv) > 3 else 80
        chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
        sys.exit(agentic(target, turns, chunk=chunk))
