"""Greedy determinism A/B: run a fixed prompt set and print a SHA per prompt.

Used to gate the draft-INT4 overlay: because the target still verifies with its
own fp16 LM head, the emitted greedy sequence must be unchanged. Compare the
SHAs between B70_DRAFT_LMHEAD_INT4=0 and =1.

Usage: python benchmarks/quality_ab.py [label]
Writes quality_ab_<label>.json
"""
import hashlib
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "huihui-qwen38-27b-abliterated-int4"

PROMPTS = {
    "reason": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
              "the ball. How much does the ball cost? Show the reasoning, then give "
              "the final answer on its own line.",
    "code": "Write a Python function `sum_of_squares(n)` that returns the sum of the "
            "squares of the first n positive integers. Include a short docstring.",
    "json": "Return a JSON object with keys name, age, city for a fictional person. "
            "Output only the JSON.",
    "zh": "用三句话解释什么是投机解码（speculative decoding），并说明它为什么能加速。",
    "list": "List the first 12 prime numbers, separated by commas.",
    "long": "The quick brown fox jumps over the lazy dog. " * 40 +
            "\n\nHow many times does the word 'fox' appear above? Answer with a number.",
    "instruct": "Summarize the following in exactly two sentences: vLLM serves LLMs with "
                "paged attention and continuous batching. Speculative decoding adds a "
                "draft model whose proposals the target verifies in one forward pass.",
}


def run(prompt, max_tokens=200):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "top_p": 1.0, "top_k": 0, "ignore_eos": True}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d["choices"][0].get("text") or "", d["usage"]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    out = {}
    for name, prompt in PROMPTS.items():
        text, usage = run(prompt)
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        out[name] = {"sha": sha, "len": len(text), "prompt_tokens": usage["prompt_tokens"],
                     "text": text}
        print(f"  {name:9s} sha={sha} len={len(text):4d} {text[:60]!r}", flush=True)
    with open(f"quality_ab_{label}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved quality_ab_{label}.json")


if __name__ == "__main__":
    main()
