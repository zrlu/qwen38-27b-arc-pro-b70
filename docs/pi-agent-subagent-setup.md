# pi-agent + subagent setup (qwen38 on the B70 server)

Client-side wiring for [`pi`](https://pi.dev) (+ its official subagent
extension) against `http://localhost:8000/v1`, and the Windows UTF-8 console
fix that stops Chinese mojibake ("锛?").

## 1. UTF-8 console (fixes GBK(936) mojibake)

- PowerShell profile: set `[Console]::Input/OutputEncoding` UTF-8, `$OutputEncoding`
  UTF-8, `chcp 65001`, `PYTHONUTF8=1`.
- cmd AutoRun (HKCU `Software\Microsoft\Command Processor`): `chcp 65001 >nul` —
  makes tool-spawned cmd UTF-8.
- Reopen terminals after the change (old consoles keep 936).
- Never run two `pi` instances on the same profile at once (a second `pi -p`
  blocks while one interactive pi is active).

## 2. `~/.pi/agent/models.json` — local providers

This machine registers two local providers: `local-xpu` (this deployment —
INT4 + MTP on the Arc Pro B70, port 8000) and `local-cuda` (the sibling
NVFP4 quant on an RTX 5090, port 8100 — drop it on machines without the
CUDA variant). The full file ships in the repo:
copy `pi-agent/models.json` into `~/.pi/agent/`.

The B70 entry (the repo copy additionally contains the `local-cuda`
sibling):

```json
{
  "providers": {
    "local-xpu": {
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "huihui-qwen38-27b-abliterated-int4",
          "name": "Huihui Qwen3.8-27B Abliterated INT4+MTP Arc Pro B70 (:8000)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 100000,
          "maxTokens": 32768,
          "thinkingLevelMap": {"minimal": null, "low": "low", "medium": "medium", "high": null, "xhigh": "xhigh", "max": null},
          "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": true}
        }
      ]
    }
  }
}
```

- `id` must match what the server actually serves — verified via
  `/v1/models`: `huihui-qwen38-27b-abliterated-int4` (the old short
  `qwen38` alias is gone; anything still referencing it needs this full id).
- `maxTokens: 32768` stays within the headroom `reserveTokens: 24576`
  leaves under the 100k cap, so prompt+output never overflows at
  compaction time.
- `thinkingLevelMap`: `low`/`medium`/`xhigh` map 1:1 to the server's
  `reasoning_effort`; `minimal`/`high`/`max` map to `null` — no effort
  sent, the server template default applies.

## 3. `~/.pi/agent/settings.json`

Also ships in the repo (`pi-agent/settings.json`) — copy into
`~/.pi/agent/`. The operative parts:

```json
{
  "defaultProvider": "local-xpu",
  "defaultModel": "huihui-qwen38-27b-abliterated-int4",
  "defaultThinkingLevel": "medium",
  "compaction": { "reserveTokens": 24576, "keepRecentTokens": 20000 }
}
```

Compaction triggers ~75k (earlier than 84k), leaving output room.
(`theme` / `hideThinkingBlock` / `lastChangelogVersion` in the repo copy
are pi bookkeeping — per-machine, harmless either way.)

## 4. Truly disable thinking

`pi --thinking off` sends no `reasoning_effort`; the server default
`chat_template_kwargs.enable_thinking=false` applies → **no thinking tokens**,
not merely hidden (verified `usage.reasoning=0`). In-session: `/thinking off`.

## 5. Official subagent extension

Copy from the pi npm package `examples/extensions/subagent/`:

```
~/.pi/agent/extensions/subagent/index.ts
~/.pi/agent/extensions/subagent/agents.ts
~/.pi/agent/agents/*.md        (scout, planner, reviewer, worker)
~/.pi/agent/prompts/*.md       (implement, scout-and-plan, implement-and-review)
```

Then **remove the `model: claude-…` line** from each agent frontmatter so
subagents inherit the active model (`local-xpu` / the B70 qwen38 model).
Usage in pi:

```
Use scout to find all authentication code
Run 3 scouts in parallel: models, providers, tests
/implement add input validation to the API
```

Modes: single / parallel (≤8 tasks, 4 concurrent) / chain (`{previous}`);
each subagent is an isolated `pi --mode json -p` child; ≤50 KB results return.

## 6. Server-side knobs relevant to agents

`MTP_TOKENS=3 DRAFT_INT4=0 TOOL_CALL_PARSER=qwen3_xml
PREFIX_CACHE=1 MAX_NUM_SEQS=1 B70_THINKING_BUDGET=4096`,
KV 4.3 GiB (~100k token pool), `MAX_MODEL_LEN=100000`.

Sampling defaults (`temperature 1.0, top_k 20, top_p 0.95`) come from the
model's own `generation_config.json` — vLLM applies them automatically
(`--generation-config auto`).

## 7. Vision / image limit (max ONE image per message)

The server limits images per request via `MM_IMAGES` (default 1; this
deployment runs 16 as tolerance). pi has no per-request image-count field, and
it RE-SENDS the full conversation every request — historical images accumulate
and can trip the limit (HTTP 400, which also kills the session). Two defenses:

- Install the trimming extension that ships in this repository
  (`pi-agent/extensions/trim-images/index.ts` → copy the directory into
  `~/.pi/agent/extensions/`): before every LLM call it removes image parts
  from all messages except the most recent image-bearing one, so history
  never accumulates images.
- Additionally instruct the agent in the system prompt:

```
pi --provider local-xpu --model huihui-qwen38-27b-abliterated-int4 \
   --append-system-prompt "Rule: attach at most ONE image per message (server limit). Never duplicate an image in the same message."
```

To raise the limit instead: run the container with `-e MM_IMAGES=16` (this deployment; the repo's `trim-images` extension already keeps per-request images low).
