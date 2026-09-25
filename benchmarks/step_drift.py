"""Within-request step-time drift.

Answers: does per-token latency grow with the number of steps already executed,
inside a single request? A rise across one long generation means the cost is
per-step (a leak in the step path), which reproduces in minutes instead of
needing an hours-long server uptime.

Usage: python benchmarks/step_drift.py [tokens] [--ctx-chars N]
"""
import glob
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def main():
    tokens = 4000
    chars = 7000
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        tokens = int(args[0])
    for a in sys.argv[1:]:
        if a.startswith("--ctx-chars"):
            chars = int(a.split("=", 1)[1])

    model = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=30).read())
    model = model["data"][0]["id"]
    corpus = "\n\n".join(
        open(p, encoding="utf-8", errors="ignore").read()
        for p in sorted(glob.glob(r"C:\Users\zeran\.pi\agent\skills\**\*.md",
                                  recursive=True)))
    prompt = (corpus[:chars] +
              "\n\nWrite a very long, detailed, continuous essay about the history "
              "of computing. Do not stop early.")

    body = {"model": model, "prompt": prompt, "max_tokens": tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps = []
    ttft = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
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
            for ch in d.get("choices") or []:
                if ch.get("text"):
                    if ttft is None:
                        ttft = time.time() - t0
                    stamps.append(time.time())

    total = time.time() - t0
    deltas = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    if not deltas:
        print("no deltas")
        return
    k = max(1, len(deltas) // 10)
    head = statistics.median(deltas[:k])
    mid = statistics.median(deltas[len(deltas) // 2 - k // 2:
                                   len(deltas) // 2 + k // 2 + 1])
    tail = statistics.median(deltas[-k:])
    print(f"chunks={len(stamps)} ttft={ttft:.2f}s wall={total:.1f}s "
          f"tok/s={len(stamps) / max(total - ttft, 1e-9):.1f}")
    print(f"chunk-interval median: head={head * 1000:.1f}ms mid={mid * 1000:.1f}ms "
          f"tail={tail * 1000:.1f}ms  drift={100 * (tail / head - 1):+.0f}%")
    # quintiles, to show monotonic drift rather than noise
    q = len(deltas) // 5
    print("  quintile medians (ms): " +
          " ".join(f"{statistics.median(deltas[i * q:(i + 1) * q]) * 1000:.1f}"
                   for i in range(5)))


if __name__ == "__main__":
    main()
