# Qwen3.8-27B x Intel Arc Pro B70 - one-click Docker, pi-agent setup, benchmarks

Abliterated **Qwen3.8-27B** -> **GPTQ-INT4 (sym G128, MTP-BF16)**, tuned and
published for a **single Intel Arc Pro B70** (Xe2, 32 GB class). Reference
stack: vLLM XPU `0.27.2rc1.dev77+gac7509e2b`, kernels `0.1.12.3`, MTP4 +
draft-INT4 overlay, prefix caching, `qwen3_xml` tool-call parser.

| Artifact | Link |
|---|---|
| Model (HF, huihui) | [zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70](https://huggingface.co/zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70) |
| Image (Docker Hub) | `docker pull zrlu/qwen38-27b-arc-pro-b70:2026.08.24` |
| Upstream reference | [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) |

## How to run (Windows + Docker Desktop/WSL2)

```powershell
./start-qwen38-ablit-xpu.ps1
```

First start auto-downloads the HF model (~18 GB) into `/model`, then serves in
~4-5 min

Native Linux: replace `--device /dev/dxg` with `--device /dev/dri` +
`--group-add $(stat -c '%g' /dev/dri/render*)`, drop the wsl-lib mounts.

Tuned defaults baked in: MTP4, KV 8.8 GiB (~206k token pool),
`MAX_NUM_SEQS=1`, prefix cache ON, `repetition_penalty=1.05` +
`presence_penalty=0.5`, `qwen3_xml` parser.

## License / credits

Apache-2.0 (inherited; quantization only). Models derived from
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated),
[OBLITERATUS/…](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) and
[Jiunsong/SuperQwen…](https://huggingface.co/Jiunsong/SuperQwen3.8-27b-abliterated)
(Qwen/Qwen3.8-27B lineage); tuning methodology from the SergiioB B70 cookbook.
Benchmarks are Windows WSL2, self-reported - not comparable cell-for-cell with
native-Linux cookbook numbers.
