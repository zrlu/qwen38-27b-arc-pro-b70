"""bench_suite.py — run the SAME benchmark suite against any served model.

Performance only (card g128, prose decode, C4 wave, long-context curve,
prefill).

Usage:  python bench_suite.py --label <name> --out <dir> [--card 3]
Outputs: <out>/<label>.json + <out>/<label>.md
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request

BASE = "http://localhost:8000"

def post(body, timeout=900):
    req = urllib.request.Request(f"{BASE}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    o = json.load(urllib.request.urlopen(req, timeout=timeout))
    return o, time.time() - t0

PROSE = ("Explain the difference between speculative decoding and conventional "
         "autoregressive decoding in language models. Focus on speed and accuracy tradeoffs.")
REAL_CN = ("这是会话历史中的一段工具输出，包含运行日志、路径和代码片段。代理正在处理一个长期任务，"
           "需要保持对上下文的完整理解。请忽略输出中的噪音，专注于与问题相关的部分。")

def decode(n, prompt=PROSE):
    o, dt = post({"model": "qwen38", "prompt": prompt, "max_tokens": n,
                  "temperature": 0.0, "ignore_eos": True})
    return o["usage"]["completion_tokens"] / dt if dt > 0 else 0

def wave(n, gen):
    def run(i):
        p = PROSE + f" Topic {i}."
        body = {"model": "qwen38", "prompt": p, "max_tokens": gen, "temperature": 0.0, "ignore_eos": True}
        req = urllib.request.Request(f"{BASE}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time(); o = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t0
        return o["usage"]["completion_tokens"] / dt if dt > 0 else 0
    res = [0.0] * n
    ts = []
    for i in range(n):
        t = threading.Thread(target=lambda i=i: res.__setitem__(i, run(i))); t.start(); ts.append(t)
    for t in ts: t.join()
    return sorted(res)

def curve(L):
    n = max(3, L // 28)
    prompt = "\n".join(f"工具输出[{i}]：{REAL_CN}" for i in range(n)) + "\n\n请总结以上上下文并给出下一步建议。"
    def one():
        try:
            o, dt = post({"model": "qwen38", "prompt": prompt, "max_tokens": 128,
                          "temperature": 0.0, "ignore_eos": True})
            return o["usage"]["completion_tokens"] / dt if o and dt > 0 else None
        except Exception:
            return None
    one()  # JIT warm
    return one()

def prefill(L):
    n = max(3, L // 28)
    prompt = "\n".join(f"工具输出[{i}]：{REAL_CN}" for i in range(n))
    o, dt = post({"model": "qwen38", "prompt": prompt, "max_tokens": 8, "temperature": 0.0}, timeout=300)
    p = o["usage"]["prompt_tokens"]
    return round(p / dt, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench-results"))
    ap.add_argument("--card", type=int, default=3, help="cookbook card runs (3)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    r = {"label": args.label, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    r["decode_prose_128"] = round(decode(128), 1)
    r["decode_prose_256"] = round(decode(256), 1)
    wave4 = wave(4, 256)
    r["wave4_median"] = round(wave4[2], 1)
    r["wave4_sum"] = round(sum(wave4), 1)
    r["curve"] = {str(L): (round(curve(L), 2) if curve(L) is not None else "rejected>100k")
                  for L in (16384, 32768, 48000)}
    r["prefill"] = {str(L): prefill(L) for L in (512, 2048, 8192)}

    if args.card:
        cook = os.environ.get("BENCH_COOKBOOK_HARNESS", r"Test-CookbookDecode.ps1")
        if os.path.exists(cook):
            cp = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", cook,
                                 "-PromptTokens", "512", "-GenerateTokens", "128", "-Runs", str(args.card)],
                                capture_output=True, text=True, timeout=600)
            med = re.search(r"Median:\s*([0-9.]+)", cp.stdout)
            acc = re.search(r"acceptance:\s*([0-9.]+)%", cp.stdout)
            r["card_p512_g128_median"] = float(med.group(1)) if med else None
            r["card_acceptance"] = float(acc.group(1)) if acc else None

    with open(os.path.join(args.out, f"{args.label}.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    md = (f"# {args.label} benchmark\n\n| metric | value |\n|---|---:|\n"
          + "".join(f"| {k} | {v} |\n" for k, v in r.items() if not isinstance(v, (dict, list))))
    with open(os.path.join(args.out, f"{args.label}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(r, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    sys.exit(main())