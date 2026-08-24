#!/usr/bin/env bash
set -eu

# Disable CUDA detection to avoid platform conflicts with XPU
export CUDA_VISIBLE_DEVICES=""
export VLLM_ALLOW_RUNTIME_PLUGIN_REGISTER=1

MODEL_PATH="${MODEL_PATH:-/model}"
MODEL_NAME="${MODEL_NAME:-qwen38}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MTP_TOKENS="${MTP_TOKENS:-2}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
DRAFT_INT4="${DRAFT_INT4:-1}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
ENABLE_TOOLS="${ENABLE_TOOLS:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
MM_IMAGES="${MM_IMAGES:-0}"

MODEL_BOOTSTRAP="${MODEL_BOOTSTRAP:-0}"
HF_REPO="${HF_REPO:-zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70}"
if [ "$MODEL_BOOTSTRAP" = "1" ] && [ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
  echo "[start] /model is empty -> downloading ${HF_REPO} (first run only)"
  python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_REPO}', local_dir='${MODEL_PATH}')"
fi

python /opt/qwen38/diagnose.py
python /opt/qwen38/patch_mtp_nightly.py
python /opt/qwen38/patch_mtp_boundary.py
python /opt/qwen38/patch_gdn_mixed_split_v5.py
python /opt/qwen38/patch_draft_lmhead_int4.py
python /opt/qwen38/patch_draft_mtp_int4.py
python /opt/qwen38/patch_xpu_single_gpu_warmup.py

if (( DRAFT_INT4 > 0 )); then
  export B70_DRAFT_LMHEAD_INT4=1
  export B70_DRAFT_MTP_INT4=1
  echo "[start] draft-INT4 S+M1 overlay ENABLED"
else
  echo "[start] draft-INT4 overlay OFF (BF16 draft)"
fi

REASONING_PARSER="${REASONING_PARSER:-qwen3}"

args=(
  --quantization gptq \
  --dtype float16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --port 8000 \
  --host 0.0.0.0 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  $(if [ "$PREFIX_CACHE" = "1" ]; then echo "--enable-prefix-caching"; else echo "--no-enable-prefix-caching"; fi) \
  --served-model-name "$MODEL_NAME" \
  --generation-config auto \
  $(if [ -n "$REASONING_PARSER" ]; then echo "--reasoning-parser $REASONING_PARSER"; fi) \
  --default-chat-template-kwargs '{"enable_thinking": false}'
)

if [ "${MM_IMAGES}" != "0" ]; then
  args+=(--limit-mm-per-prompt "{\"image\": ${MM_IMAGES}}")
else
  args+=(--language-model-only)
fi

if (( KV_CACHE_MEMORY_BYTES > 0 )); then
  args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi

if (( ENABLE_TOOLS > 0 )); then
  args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if (( MTP_TOKENS > 0 )); then
  args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

gc=""
if [ -n "${OVERRIDE_GENERATION_CONFIG:-}" ]; then
  gc="${OVERRIDE_GENERATION_CONFIG}"
elif [ -n "${OVERRIDE_RP:-}${OVERRIDE_PP:-}" ]; then
  gc="{"
  first=1
  if [ -n "${OVERRIDE_RP:-}" ]; then gc="${gc}\"repetition_penalty\":${OVERRIDE_RP}"; first=0; fi
  if [ -n "${OVERRIDE_PP:-}" ]; then [ $first -eq 0 ] && gc="${gc},"; gc="${gc}\"presence_penalty\":${OVERRIDE_PP}"; fi
  gc="${gc}}"
fi
if [ -n "$gc" ]; then
  echo "[start] override-generation-config: ${gc}"
  args+=(--override-generation-config "${gc}")
fi

exec vllm serve "$MODEL_PATH" "${args[@]}"
