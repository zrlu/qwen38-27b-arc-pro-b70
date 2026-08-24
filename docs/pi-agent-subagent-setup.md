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

## 2. `~/.pi/agent/models.json` — qwen-local provider

```json
{
  "providers": {
    "qwen-local": {
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "qwen38",
          "name": "Qwen3.8-27B Abliterated (B70 :8000)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 100000,
          "maxTokens": 16000,
          "thinkingLevelMap": {"minimal":"low","low":"low","medium":"medium","high":"medium","xhigh":"xhigh","max":"xhigh"},
          "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": true}
        }
      ]
    }
  }
}
```

`maxTokens: 16000` (not 32000) so prompt+output never exceeds the server's 100k
cap at compaction time.

## 3. `~/.pi/agent/settings.json`

```json
{ "defaultProvider": "qwen-local", "defaultModel": "qwen38",
  "defaultThinkingLevel": "high", "reserveTokens": 24576, "keepRecentTokens": 20000 }
```

Compaction triggers ~75k (earlier than 84k), leaving output room.

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
subagents inherit the active model (qwen38). Usage in pi:

```
Use scout to find all authentication code
Run 3 scouts in parallel: models, providers, tests
/implement add input validation to the API
```

Modes: single / parallel (≤8 tasks, 4 concurrent) / chain (`{previous}`);
each subagent is an isolated `pi --mode json -p` child; ≤50 KB results return.

## 6. Server-side knobs relevant to agents

`MTP_TOKENS=4 DRAFT_INT4=1 B70_MTP_BF16_DRAFT=1 TOOL_CALL_PARSER=qwen3_xml
PREFIX_CACHE=1 MAX_NUM_SEQS=4 OVERRIDE_RP=1.05 OVERRIDE_PP=0.5`
(long-context "!"-degeneration guard), KV 8.8 GiB (~206k token pool).
## 7. Vision / image limit (max ONE image per message)

The server limits images per request via `MM_IMAGES` (default 1; this
deployment runs 16 as tolerance). pi has no per-request image-count field, and
it RE-SENDS the full conversation every request — historical images accumulate
and can trip the limit (HTTP 400, which also kills the session). Two defenses:

- Ship a trimming extension (`~/.pi/agent/extensions/trim-images/index.ts`):
  before every LLM call it removes image parts from all messages except the
  most recent image-bearing one, so history never accumulates images.
- Additionally instruct the agent in the system prompt:

```
pi --provider qwen-local --model qwen38 \
   --append-system-prompt "Rule: attach at most ONE image per message (server limit). Never duplicate an image in the same message."
```

To raise the limit instead: run the container with `-e MM_IMAGES=16` (this deployment; the included `trim-images` extension already keeps per-request images low).
