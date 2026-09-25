"""Overnight session-degradation probe.

Emulates the real workload (a growing multi-turn session) and, every interval,
records the two numbers that must be told apart:

  ttft_s        prefill cost of the step. With a working prefix cache only the
                NEW chunk is prefilled, so ttft stays ~flat as ctx grows.
                Without reuse it grows linearly with ctx.
  ctx_over_ttft = ctx / ttft. Rises with context when caching works; flat when
                the whole context is re-prefilled every turn.
  step_ms       decode-only step time = (total - ttft) / steps, where
                steps = generation_tokens_delta / tokens_per_step and
                tokens_per_step = 1 + num_spec * accepted/drafted.

A session grows by --chunk tokens per sample and resets past --max-ctx, so the
CSV contains several complete cycles and the drift is visible within a cycle.

Usage:
  python benchmarks/watch_session_degradation.py --interval 300 --chunk 2000
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
_CORPUS = []


def corpus():
    if not _CORPUS:
        parts = []
        for pat in [r"C:\Users\zeran\.pi\agent\skills\**\*.md",
                    r"C:\LocalLLM\qwen38-27b-ablit-xpu\.b70src\cookbook\*.md",
                    r"C:\LocalLLM\qwen38-27b-ablit-xpu\README.md"]:
            for p in sorted(glob.glob(pat, recursive=True)):
                try:
                    parts.append(open(p, encoding="utf-8", errors="ignore").read())
                except OSError:
                    pass
        _CORPUS.append("\n\n".join(parts))
    return _CORPUS[0]


def ntok(model, text):
    req = urllib.request.Request(
        BASE + "/tokenize",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["count"]


def chunk_text(model, target_tokens, offset):
    """A distinct ~target_tokens block, so each turn adds new content."""
    c = corpus()
    n = len(c)
    start = offset % max(1, n - 1000)
    guess = max(500, int(target_tokens * 3.4))
    s = c[start:start + guess]
    for _ in range(6):
        cnt = ntok(model, s)
        if abs(cnt - target_tokens) < 300:
            break
        s = c[start:start + max(500, int(len(s) * target_tokens / max(cnt, 1)))]
    return s


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    keys = {
        "vllm:spec_decode_num_draft_tokens_total": "drafted",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted",
        "vllm:generation_tokens_total": "gen",
        "vllm:prefix_cache_hits_total": "hits",
        "vllm:prefix_cache_queries_total": "queries",
        "vllm:kv_cache_usage_perc": "kv",
    }
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k, name in keys.items():
            if line.startswith(k + "{"):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
                break
    return out


def gpu_stats():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "& xpu-smi stats -d 0"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return "", "", ""
    p = m = r = ""
    for line in out.splitlines():
        if "GPU Power (W)" in line and "avg:" in line:
            p = line.split("avg:")[1].split(",")[0].strip()
        elif "GPU Frequency (MHz)" in line and "avg:" in line:
            m = line.split("avg:")[1].split(",")[0].strip()
        elif "GPU Memory Read (kB/s)" in line and "avg:" in line:
            try:
                r = round(float(line.split("avg:")[1].split(",")[0].strip()) / 1e6, 1)
            except ValueError:
                pass
    return p, m, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--max-ctx", type=int, default=120000)
    ap.add_argument("--spec", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--csv", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "session-degradation.csv"))
    ap.add_argument("--gpu-every", type=int, default=4)
    a = ap.parse_args()

    model = a.model or json.loads(
        urllib.request.urlopen(BASE + "/v1/models", timeout=30).read())["data"][0]["id"]

    path = os.path.abspath(a.csv)
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp", "turn", "ctx", "new_tokens", "ttft_s",
                    "ctx_over_ttft", "thr_tok_s", "step_ms", "accept_pct",
                    "hit_pct", "kv_pct", "power_w", "gpu_mhz", "mem_read_gbs"])
        f.flush()
    print(f"session probe: model={model} chunk={a.chunk} gen={a.gen} "
          f"interval={a.interval}s max_ctx={a.max_ctx} -> {path}", flush=True)

    prompt = ""
    offset = 0
    turn = 0
    n = 0
    while True:
        try:
            if not prompt or ntok(model, prompt) > a.max_ctx:
                offset = (offset + 977) % 100000
                prompt = ""
                turn = 0
                print(f"  -- new session (offset={offset})", flush=True)
            blk = chunk_text(model, a.chunk, offset + turn * 7919)
            prompt = (prompt + "\n\n" + blk +
                      "\n\nSummarize the tradeoffs discussed above in one short paragraph.")
            turn += 1
            n += 1

            ctx_tokens = ntok(model, prompt)
            m0 = metrics()
            body = {"model": model, "prompt": prompt, "max_tokens": a.gen,
                    "temperature": 0.0, "ignore_eos": True, "stream": True,
                    "stream_options": {"include_usage": True}}
            req = urllib.request.Request(
                BASE + "/v1/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            t0 = time.time()
            ttft = None
            usage = None
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
                        if ch.get("text") and ttft is None:
                            ttft = time.time() - t0
            total = time.time() - t0
            m1 = metrics()

            drafted = m1.get("drafted", 0) - m0.get("drafted", 0)
            accepted = m1.get("accepted", 0) - m0.get("accepted", 0)
            gen = m1.get("gen", 0) - m0.get("gen", 0)
            q = m1.get("queries", 0) - m0.get("queries", 0)
            h = m1.get("hits", 0) - m0.get("hits", 0)
            spec = accepted / drafted if drafted else 0.0
            tps = 1.0 + a.spec * spec
            steps = gen / tps if tps > 0 else 0.0
            decode_s = max(total - (ttft or 0.0), 1e-9)
            step_ms = round(1000.0 * decode_s / steps, 1) if steps > 0 else ""
            ctx = usage["prompt_tokens"] if usage else ctx_tokens
            thr = round(gen / decode_s, 1)
            pw = gm = mr = ""
            if n % a.gpu_every == 1:
                pw, gm, mr = gpu_stats()

            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([stamp, turn, ctx, ctx_tokens, round(ttft or 0, 2),
                        round(ctx / (ttft or 1e-9)), thr, step_ms,
                        round(100 * spec, 1),
                        round(100.0 * h / q, 1) if q else "",
                        round(m1.get("kv", 0), 1), pw, gm, mr])
            f.flush()
            print(f"  {stamp} turn={turn} ctx={ctx} ttft={round(ttft or 0,2)}s "
                  f"ctx/ttft={round(ctx/(ttft or 1e-9))} thr={thr} step_ms={step_ms} "
                  f"accept={round(100*spec,1)}% hit={round(100.0*h/q,1) if q else '-'}% "
                  f"kv={round(m1.get('kv',0),1)} W={pw} MHz={gm}", flush=True)
        except Exception as e:
            print(f"  sample failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
