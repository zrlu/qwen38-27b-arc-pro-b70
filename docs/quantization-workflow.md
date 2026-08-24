# Quantization workflow — Huihui-Qwen3.8-27B-abliterated → GPTQ-INT4 (B70)

Reproduces `abliterated-gptq-v2` (published on HF as
`Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70`).

## Recipe

```bash
# 1) Source (BF16, ~51.8 GB): huihui-ai/Huihui-Qwen3.8-27B-abliterated
#    (updated abliteration: layers 18-51 only; MTP + vision untouched)

# 2) Quantize on a CUDA GPU (RTX 5090) with gptqmodel==7.3.2:
docker run --rm --gpus all -v src:/model-in -v out:/model-out \
  -e MODEL_IN=/model-in -e MODEL_OUT=/model-out \
  vllm/vllm-openai:latest \
  bash -lc "pip install -q gptqmodel==7.3.2 datasets && python /quant.py"
```

`quant.py` (identical contract to `quantize_ablit.py`):

```python
from gptqmodel import GPTQModel, QuantizeConfig
qc = QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True,
                    dynamic={"-:.*mtp.*": {}},   # MTP heads stay BF16
                    lm_head=False)
m = GPTQModel.load(MODEL_IN, qc, device_map="cuda", trust_remote_code=True)
m.quantize(calibration=calib)      # wikitext-2 first 32 (fallback text ok)
m.save_quantized(MODEL_OUT)
```

## Validate before publishing

- `config.json`: `quantization_config` = gptq, 4-bit, sym, g128, desc_act=false,
  `dynamic -:.*mtp.*`, lm_head=false; arch `Qwen3_5ForConditionalGeneration`,
  `image_token_id 248056`.
- Shards: 5 × safetensors ≈ 18.2 GB total (−64.8% from 51.75 GB).
- Counts: 400 quantized INT4 tensors + 15 preserved BF16 `mtp.*` tensors.
- Serve + sanity-check: `/health`, one decode, tool-call parse.

## Pitfalls

- **Windows quoting**: `-e KEY='{"a":1}'` loses quotes through PowerShell→docker;
  pass numbers (`OVERRIDE_RP=1.05`) and let start.sh build the JSON.
- Use the **public champion base** digest for reproducibility
  (`vllm/vllm-openai-xpu@sha256:f01e24f6…`).
- Publishing derivative quants: follow `LICENSING_SKILL.md` — Qwen base ⇒
  `license: apache-2.0`, `base_model:` points at huihui-ai repo, README credit,
  `.gitattributes` LFS for safetensors.