#!/usr/bin/env python3
"""
B70 Phase M1: MTP layer (draft) quantized to INT4 g128 sym (GPTQ format).

v2: Fixed int4_gemm_w4a16 call signature:
    - Added .t() to qweight (matches official xpu.py implementation)
    - Added group_idx parameter (explicitly passed as None)
    - Skip fc layer (dimension mismatch in this model variant)

Environment gate: B70_DRAFT_MTP_INT4=1 (default off)
"""

from __future__ import annotations

import os
import sys

MARKER = "B70_DRAFT_MTP_INT4_V2"
HELPER_MODULE = "b70_draft_mtp_int4_v2.py"

HELPER_SOURCE = '''\
"""B70 Phase M1 runtime helper: draft MTP layer INT4 g128 sym.

v2: Fixed int4_gemm_w4a16 call signature with .t() and group_idx.
    Skip fc layer (dimension mismatch in this model variant).
"""

from __future__ import annotations

import os
import torch


def quantize_to_int4(weight: torch.Tensor, group_size: int = 128):
    """Quantize weight [N, K] to GPTQ INT4 g128 sym (int4_gemm_w4a16 format)."""
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    chunk = 4096
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    parts, scale_parts = [], []
    for i in range(0, N, chunk):
        wc = weight[i:i + chunk].float()
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(
            wc.shape[0], K // 8
        )
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)
    scales_contig = torch.cat(scale_parts, dim=0)
    qweight = qweight_contig.t()
    scales = scales_contig.t().contiguous()
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def _collect_linears(predictor) -> list[tuple[str, torch.nn.Module]]:
    """Collect 4 MTP linears for quantization (fc skipped)."""
    found: list[tuple[str, torch.nn.Module]] = []
    for li, layer in enumerate(predictor.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            found.append((f"layers.{li}.self_attn.qkv_proj", attn.qkv_proj))
            found.append((f"layers.{li}.self_attn.o_proj", attn.o_proj))
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            gate_up = getattr(mlp, "gate_up_proj", None)
            down = getattr(mlp, "down_proj", None)
            if gate_up is not None:
                found.append((f"layers.{li}.mlp.gate_up_proj", gate_up))
            if down is not None:
                found.append((f"layers.{li}.mlp.down_proj", down))
    return found


class _B70MTPInt4LinearMethod:
    """Duck-typed quant_method using int4_gemm_w4a16."""

    def __init__(self, qweight, scales, qzeros, group_size, out_features):
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.group_size = group_size
        self.out_features = out_features

    def create_weights(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, *args, **kwargs):
        pass

    def apply(self, layer, x, bias):
        flat = x.reshape(-1, x.shape[-1])
        if flat.dtype != torch.float16:
            flat = flat.to(torch.float16)
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            flat,
            self.qweight.t(),
            None,
            self.scales,
            self.qzeros,
            self.group_size,
            None,
        )
        return out.reshape(*x.shape[:-1], self.out_features)


@torch.no_grad()
def build_draft_mtp_int4(model) -> None:
    if os.environ.get("B70_DRAFT_MTP_INT4") != "1":
        return
    if getattr(model, "_b70_mtp_int4_built", False):
        return

    predictor = getattr(model, "model", None)
    if predictor is None or not hasattr(predictor, "layers"):
        print("[B70] draft MTP INT4: no Qwen3_5MultiTokenPredictor; "
              "MTP stays in BF16", flush=True)
        return

    linears = _collect_linears(predictor)
    if not linears:
        print("[B70] draft MTP INT4: no linears found; skip", flush=True)
        return

    print(f"[B70] draft MTP INT4: quantizing {len(linears)} linears "
          f"(fc skipped) -> INT4 g128 sym", flush=True)

    total_fp16, total_int4 = 0, 0
    for name, lin in linears:
        w = getattr(lin, "weight", None)
        if w is None:
            continue

        out_features = w.shape[0]
        qweight, scales, qzeros, gs = quantize_to_int4(w.detach())
        lin._b70_mtp_int4 = _B70MTPInt4LinearMethod(
            qweight, scales, qzeros, gs, out_features
        )
        lin.quant_method = lin._b70_mtp_int4

        fp16_bytes = w.numel() * w.element_size()
        int4_bytes = qweight.numel() * qweight.element_size() + (
            scales.numel() * scales.element_size()
        )
        total_fp16 += fp16_bytes
        total_int4 += int4_bytes

        print(f"[B70] draft MTP INT4: {name} {tuple(w.shape)} "
              f"{fp16_bytes/1e6:.0f}MB -> {int4_bytes/1e6:.0f}MB", flush=True)

    model._b70_mtp_int4_built = True
    print(f"[B70] draft MTP INT4: ready. {total_fp16/1e9:.2f}GB BF16 -> "
          f"{total_int4/1e9:.2f}GB INT4 (saves "
          f"{(total_fp16 - total_int4)/1e6:.0f}MB)", flush=True)
'''

QWMTP_FORWARD_OLD = (
    "        loader = AutoWeightsLoader(self)\n"
    "        return loader.load_weights(remap_weight_names(weights))\n"
)
QWMTP_FORWARD_NEW = (
    "        loader = AutoWeightsLoader(self)\n"
    "        result = loader.load_weights(remap_weight_names(weights))\n"
    "        if os.environ.get(\"B70_DRAFT_MTP_INT4\") == \"1\":\n"
    "            from vllm.model_executor.models.b70_draft_mtp_int4_v2 import (\n"
    "                build_draft_mtp_int4,\n"
    "            )\n"
    "\n"
    "            build_draft_mtp_int4(self)\n"
    "        return result\n"
)


def _write_helper(vllm_dir: str) -> str:
    models_dir = os.path.join(vllm_dir, "model_executor", "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, HELPER_MODULE)
    if os.path.exists(path):
        existing = open(path).read()
        if existing == HELPER_SOURCE:
            print(f"helper already present {path}")
            return path
    with open(path, "w") as f:
        f.write(HELPER_SOURCE)
    print(f"helper written {path}")
    return path


def _patch_qwen3_5_mtp(vllm_dir: str) -> None:
    path = os.path.join(vllm_dir, "model_executor", "models", "qwen3_5_mtp.py")
    text = open(path).read()
    if MARKER in text:
        print(f"already patched {path}")
        return
    if QWMTP_FORWARD_OLD not in text:
        sys.exit(f"anchor not found in {path}: Qwen3_5MTP.forward")
    text = text.replace(QWMTP_FORWARD_OLD, QWMTP_FORWARD_NEW, 1)
    if "\nimport os\n" not in text and not text.startswith("import os\n"):
        text = text.replace("import torch\n", "import os\nimport torch\n", 1)
        if "\nimport os\n" not in text and not text.startswith("import os\n"):
            sys.exit(f"could not inject import os in {path}")
    compile(text, path, "exec")
    open(path, "w").write(text)
    print(f"patched {path}")


def main() -> None:
    import vllm
    vllm_dir = os.path.dirname(vllm.__file__)
    _write_helper(vllm_dir)
    _patch_qwen3_5_mtp(vllm_dir)


if __name__ == "__main__":
    main()