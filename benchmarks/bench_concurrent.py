import json, sys, time, urllib.request, threading

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
G = int(sys.argv[2]) if len(sys.argv) > 2 else 256
WORDS = ['amber','binary','cedar','delta','ember','frost','granite','harbor',
         'indigo','jungle','kernel','lantern','matrix','nebula','orbit','prairie',
         'quartz','river','signal','timber','ultra','vector','willow','xenon','yellow','zenith']

def prompt(seed):
    return (' '.join(WORDS[(i * 11 + seed * 7) % len(WORDS)] for i in range(60))
            + f" Write a concise function about topic {seed} in detail.")

def run(i):
    body = {'model': 'qwen38', 'prompt': prompt(i), 'max_tokens': G,
            'temperature': 0.0, 'ignore_eos': True}
    req = urllib.request.Request('http://localhost:8000/v1/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    o = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t0
    u = o['usage']
    return u['completion_tokens'], dt, u['prompt_tokens']

def counters():
    txt = urllib.request.urlopen('http://localhost:8000/metrics', timeout=20).read().decode()
    def g(key):
        for l in txt.splitlines():
            if l.startswith(key) and 'counter' not in l:
                return float(l.split()[-1])
        return 0.0
    return (g('vllm:spec_decode_num_draft_tokens_total'),
            g('vllm:spec_decode_num_accepted_tokens_total'))

d0, a0 = counters()
res = [None] * N
def worker(i):
    res[i] = run(i)
threads = []
t0w = time.time()
for i in range(N):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
wave = time.time() - t0w
d1, a1 = counters()
rates = []
for nt, dt, pt in res:
    r = (nt - 1) / max(dt - 1e-6, 1e-9)
    rates.append(r)
    print(f"  req: prompt={pt} comp={nt} wall={dt:.2f}s post-first={r:.2f} tok/s")
rates.sort()
n = len(rates)
median = rates[n // 2]
acc = (a1 - a0) / (d1 - d0) * 100 if (d1 - d0) > 0 else 0.0
print(f"N={N}: median-per-req={median:.2f} SUM-streams={sum(rates):.2f} "
      f"wave-wall={wave:.2f}s aggregate_wall_tps={sum(nt for nt,_,_ in res)/wave:.2f} accept={acc:.1f}%")