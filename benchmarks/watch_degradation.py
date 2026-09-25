"""Degradation monitor: fixed probe every N seconds, writes a CSV.

Why a probe: the symptom is step time growing from ~60 ms to ~270 ms over tens
of hours while acceptance stays flat. Reading vLLM's periodic log line only
works while someone is using the server, so this script sends its own fixed
request instead and computes the step time from the spec-decode counters.

tokens/step = 1 + num_spec * accepted/drafted   (draft counters from /metrics)
steps       = generation_tokens_delta / tokens_per_step
step_ms     = elapsed_ms / steps

Every GPU_EVERY samples it also records xpu-smi power / clock / memory read.

Usage: python benchmarks/watch_degradation.py [--interval 300] [--ctx 2048]
                                             [--gen 200] [--csv path]
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


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    keys = {
        "vllm:spec_decode_num_draft_tokens_total": "drafted",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted",
        "vllm:generation_tokens_total": "gen",
        "vllm:num_requests_running": "running",
        "vllm:prefix_cache_hits_total": "hits",
        "vllm:prefix_cache_queries_total": "queries",
    }
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k, name in keys.items():
            if line.startswith(k + "{"):
                out[name] = float(line.rsplit(" ", 1)[1])
                break
    return out


_CORPUS = []


def corpus():
    if not _CORPUS:
        parts = []
        for pat in [r"C:\Users\zeran\.pi\agent\skills\**\*.md",
                    r"C:\LocalLLM\qwen38-27b-ablit-xpu\.b70src\cookbook\*.md"]:
            for p in sorted(glob.glob(pat, recursive=True)):
                try:
                    parts.append(open(p, encoding="utf-8", errors="ignore").read())
                except OSError:
                    pass
        _CORPUS.append("\n\n".join(parts))
    return _CORPUS[0]


def make_prompt(model, target_tokens):
    c = corpus()
    s = c[: int(len(c) * 0.08)]
    for _ in range(8):
        cnt = json.loads(urllib.request.urlopen(
            urllib.request.Request(BASE + "/tokenize",
                                   data=json.dumps({"model": model, "prompt": s}).encode(),
                                   headers={"Content-Type": "application/json"}),
            timeout=120).read())["count"]
        if abs(cnt - target_tokens) < 800:
            break
        s = c[: max(1000, int(len(s) * target_tokens / max(cnt, 1)))]
    return s + "\n\nSummarize the tradeoffs above in one short paragraph."


def gpu_stats():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "& xpu-smi stats -d 0"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return "", "", ""
    p = m = r = ""
    for line in out.splitlines():
        if "GPU Power (W)" in line and "avg:" in line:
            p = line.split("avg:")[1].split(",")[0].strip()
        if "GPU Frequency (MHz)" in line and "avg:" in line:
            m = line.split("avg:")[1].split(",")[0].strip()
        if "GPU Memory Read (kB/s)" in line and "avg:" in line:
            try:
                r = round(float(line.split("avg:")[1].split(",")[0].strip()) / 1e6, 1)
            except ValueError:
                pass
    return p, m, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--model", default=None)
    ap.add_argument("--csv", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "degradation-probe.csv"))
    ap.add_argument("--gpu-every", type=int, default=3)
    a = ap.parse_args()

    model = a.model
    if not model:
        d = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=30).read())
        model = d["data"][0]["id"]

    csv_path = os.path.abspath(a.csv)
    new = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp", "ctx", "gen", "thr_tok_s", "accept_pct",
                    "tokens_per_step", "step_ms", "ttft_s", "kv_pct", "hit_pct",
                    "power_w", "gpu_mhz", "mem_read_gbs"])
        f.flush()
    print(f"monitor: model={model} ctx={a.ctx} gen={a.gen} interval={a.interval}s -> {csv_path}")
    print("(first request warms up and is discarded)")

    prompt = None
    first = True
    n = 0
    while True:
        n += 1
        try:
            if prompt is None:
                prompt = make_prompt(model, a.ctx)
            m0 = metrics()
            body = {"model": model, "prompt": prompt, "max_tokens": a.gen,
                    "temperature": 0.0, "ignore_eos": True, "stream": True,
                    "stream_options": {"include_usage": True}}
            req = urllib.request.Request(BASE + "/v1/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            ttft = None
            usage = None
            with urllib.request.urlopen(req, timeout=1800) as r:
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
            elapsed = time.time() - t0
            m1 = metrics()

            drafted = m1["drafted"] - m0["drafted"]
            accepted = m1["accepted"] - m0["accepted"]
            gen = m1["gen"] - m0["gen"]
            spec = accepted / drafted if drafted else 0.0
            tps = 1.0 + 3.0 * spec          # MTP3
            steps = gen / tps if tps > 0 else 0.0
            # decode time only: the step time is what degrades, and elapsed includes prefill
            decode_s = max(elapsed - (ttft or 0.0), 1e-9)
            step_ms = round(1000.0 * decode_s / steps, 1) if steps > 0 else ""
            thr = round(gen / decode_s, 1)
            q = m1["queries"] - m0["queries"]
            hits = m1["hits"] - m0["hits"]
            hit_pct = round(100.0 * hits / q, 1) if q else ""
            kv = ""
            try:
                txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
                for line in txt.splitlines():
                    if line.startswith("vllm:kv_cache_usage_perc"):
                        kv = round(100.0 * float(line.rsplit(" ", 1)[1]), 1)
                        break
            except Exception:
                pass
            power = gpu = mem = ""
            if n % a.gpu_every == 1:
                power, gpu, mem = gpu_stats()

            if first:
                first = False
                print(f"  warmup: ctx={usage['prompt_tokens']} step_ms={step_ms} (discarded)")
            else:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                w.writerow([stamp, usage["prompt_tokens"], usage["completion_tokens"],
                            thr, round(100 * spec, 1), round(tps, 2), step_ms,
                            round(ttft or 0, 2), kv, hit_pct, power, gpu, mem])
                f.flush()
                print(f"  {stamp} thr={thr} accept={round(100*spec,1)}% "
                      f"step_ms={step_ms} ttft={round(ttft or 0,2)}s kv={kv} "
                      f"hit={hit_pct} W={power} MHz={gpu} memGB/s={mem}", flush=True)
        except Exception as e:
            print(f"  sample failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
