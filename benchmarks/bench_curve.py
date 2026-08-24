import json, time, urllib.request

CN = ("这是会话历史中的一段工具输出，包含运行日志、路径和代码片段。"
      "代理正在处理一个长期任务，需要保持对上下文的完整理解。"
      "请忽略输出中的噪音，专注于与问题相关的部分。")

def run(body, timeout=900):
    req = urllib.request.Request("http://localhost:8000/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    o = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    return o["usage"], dt

out = {"decode": [], "prefill": []}
for L in (16384, 32768, 65536):
    n = max(3, L // 28)
    prompt = "\n".join(f"工具输出[{i}]：{CN}" for i in range(n)) + "\n\n请总结以上上下文并给出下一步建议。"
    rates = []
    for rep in range(2):          # 1st = JIT/warm, report 2nd
        u, dt = run({"model": "qwen38", "prompt": prompt, "max_tokens": 128,
                     "temperature": 0.0, "ignore_eos": True})
        p, comp = u["prompt_tokens"], u["completion_tokens"]
        rates.append(round(comp / dt, 2))
        print("decode L=%d rep=%d ctx=%d wall=%.2fs tps=%.2f" % (L, rep + 1, p, dt, comp / dt), flush=True)
    out["decode"].append({"target_len": L, "ctx": p, "warm_tps": rates[-1], "first_tps": rates[0]})

for L in (512, 2048, 8192):
    n = max(3, L // 28)
    prompt = "\n".join(f"工具输出[{i}]：{CN}" for i in range(n))
    u, dt = run({"model": "qwen38", "prompt": prompt, "max_tokens": 8,
                 "temperature": 0.0}, timeout=300)
    p = u["prompt_tokens"]
    out["prefill"].append({"prompt": p, "ttft": round(dt, 3), "input_rate": round(p / dt, 1)})
    print("prefill p=%d ttft=%.3fs rate=%.1f" % (p, dt, p / dt), flush=True)

with open("benchmark_curve.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("DONE")