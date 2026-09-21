"""Compare two bench_context.py 'ctx' runs (driver A/B).

Usage:  python compare.py <before.json> <after.json>

Reads the JSON arrays written by `bench_context.py ctx` and prints a per-depth
table plus the mean delta. Only the numbers the harness actually measures are
compared; acceptance is shown because a driver change that shifts acceptance
would move tok/s without moving the kernel at all.
"""
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {r["ctx"]: r for r in rows}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])

    print(f"before: {sys.argv[1]}")
    print(f"after : {sys.argv[2]}\n")
    print("| ctx | tok/s before | tok/s after | delta | ttft before | ttft after "
          "| prefill before | prefill after | accept before | accept after |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    d_tok, d_pre = [], []
    for ctx in sorted(set(a) & set(b)):
        ra, rb = a[ctx], b[ctx]
        dt = rb["tok_s"] - ra["tok_s"]
        d_tok.append(dt / ra["tok_s"] * 100 if ra["tok_s"] else 0)
        if ra.get("prefill_tps") and rb.get("prefill_tps"):
            d_pre.append((rb["prefill_tps"] - ra["prefill_tps"]) / ra["prefill_tps"] * 100)
        print(f"| {ctx} | {ra['tok_s']} | {rb['tok_s']} | {dt:+.1f} | "
              f"{ra['ttft']} | {rb['ttft']} | {ra.get('prefill_tps')} | {rb.get('prefill_tps')} | "
              f"{ra.get('accept')} | {rb.get('accept')} |")

    if d_tok:
        print(f"\nmean decode delta: {sum(d_tok)/len(d_tok):+.1f}%")
    if d_pre:
        print(f"mean prefill delta: {sum(d_pre)/len(d_pre):+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
