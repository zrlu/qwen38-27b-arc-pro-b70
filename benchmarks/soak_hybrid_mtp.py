"""Soak test: grows a cached agentic conversation, checking every turn for
(a) degenerate output and (b) silent state corruption via a buried needle.

A wrong GDN/KV state can either flatline to '!' (NaN) or silently return
content belonging to another position. The needle catches the silent case.

Usage: python soak.tmp.py <target_tokens> <turns> <chunk_tokens>
"""
import glob
import json
import os
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = os.environ.get("MODEL", "huihui-qwen38-27b-abliterated-int4")


def _corpus():
    parts = []
    for pat in [r"C:\Users\zeran\.pi\agent\skills\**\*.md",
                r"C:\LocalLLM\qwen38-27b-ablit-xpu\docs\**\*.md",
                r"C:\LocalLLM\qwen38-27b-ablit-xpu\.b70src\cookbook\*.md"]:
        for p in sorted(glob.glob(pat, recursive=True)):
            try:
                parts.append(open(p, encoding="utf-8", errors="ignore").read())
            except OSError:
                pass
    return "\n\n".join(parts)


C = _corpus()


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


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


def chat(msgs, gen):
    b = {"model": MODEL, "messages": msgs, "max_tokens": gen,
         "temperature": 0.0, "stream": True,
         "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(b).encode(),
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
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or delta.get("reasoning_content")
                         or delta.get("reasoning") or "")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
    return ttft, time.time() - t0, usage, "".join(text)


def degenerate(text):
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


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

    code = "ZQX-%04d" % random.randint(1000, 9999)
    sysmsg = ("You are a coding agent. Tools: read, bash, edit, write. Be concise.\n"
              f"The session secret code is {code}. Remember it for the whole session.")
    msgs = [{"role": "system", "content": sysmsg}]
    off = 0
    fails = 0
    for t in range(1, turns + 1):
        blk = C[off:off + chunk * 3]
        off += len(blk)
        msgs.append({"role": "user",
                     "content": "Tool output (continuing session):\n" + blk +
                                "\n\nSummarize this tool output in one short paragraph."})
        m0 = metrics()
        ttft, total, usage, body = chat(msgs, 128)
        m1 = metrics()
        p, comp = usage["prompt_tokens"], usage["completion_tokens"]
        rate = (comp - 1) / max(total - (ttft or 0), 1e-9)
        hits = m1["prefix_cache_hits_total"] - m0["prefix_cache_hits_total"]
        q = m1["prefix_cache_queries_total"] - m0["prefix_cache_queries_total"]
        bad = degenerate(body)

        # needle probe: a short full-context request
        nt, ntt, nu, nbody = chat(msgs + [
            {"role": "assistant", "content": body[:200] or "OK."},
            {"role": "user", "content": "What is the session secret code? Reply with only the code."}],
            300)
        needle_ok = code in nbody
        if bad or not needle_ok:
            fails += 1
        print(f"  t{t:02d} ctx={p:7d} hit%={100*hits/max(q,1):5.1f} ttft={ttft or 0:6.2f}s "
              f"tok/s={rate:6.1f} needle={'OK' if needle_ok else 'FAIL'} bad={bad} "
              f"{body[:55]!r}", flush=True)
        if bad or not needle_ok:
            with open(f"soak_fail_t{t}.json", "w", encoding="utf-8") as f:
                json.dump({"messages": msgs, "output": body, "needle_out": nbody,
                           "code": code}, f, ensure_ascii=False, indent=1)
            print(f"  !! FAILURE at turn {t} (needle_out={nbody[:120]!r})", flush=True)
            return 2
        msgs.append({"role": "assistant", "content": body[:400] or "OK."})
        if p >= target:
            break
    print(f"SOAK CLEAN: reached {p} tokens, {turns} turns, code={code}, fails={fails}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
