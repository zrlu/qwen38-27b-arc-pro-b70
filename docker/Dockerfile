# Qwen3.8-27B (huihui abliterated) GPTQ-INT4 MTP-BF16 — single Intel Arc Pro B70
# One-click serving image (Windows Docker Desktop + WSL2 or native Linux).
#
# Two reproducible builds come out of this file:
#
#   0.28.0 (current default — the shipped image):
#     docker build -t zrlu/qwen38-27b-arc-pro-b70:0.28.0-apcfix docker
#
#   0.29.0 (experimental, DOES NOT RUN on WSL2 — see the README section
#   "vLLM 0.29.0 is blocked on WSL2"; its oneDNN W4A16 GEMM needs an OpenCL
#   compiler that the WSL driver does not provide):
#     docker build -t zrlu/qwen38-27b-arc-pro-b70:0.29.0-apcfix docker \
#       --build-arg BASE_IMAGE=vllm/vllm-openai-xpu:v0.29.0
#
# BASE_IMAGE is pinned to an immutable tag/digest; do not point it at `latest`.
ARG BASE_IMAGE=zrlu/qwen38-27b-arc-pro-b70:snapshot-0.28.0-apc-broken
FROM ${BASE_IMAGE}

# 0.1.12.3 for the 0.28.0 snapshot. Leave KERNELS_VERSION empty when the base
# already ships the right wheel (the official v0.29.0 XPU image ships 0.1.14.1).
ARG KERNELS_VERSION=0.1.12.3
RUN if [ -n "$KERNELS_VERSION" ]; then \
      pip install --no-cache-dir "vllm-xpu-kernels==${KERNELS_VERSION}"; \
    fi && \
    python -c "import importlib.metadata as m; print('vllm-xpu-kernels', m.version('vllm-xpu-kernels'))" && \
    python -c "import importlib.metadata as m; print('vllm', m.version('vllm'))"

# Create stub files to prevent NVIDIA library lookups on WSL2/Arc systems
RUN mkdir -p /usr/lib/wsl/lib && \
    touch /usr/lib/wsl/lib/libnvidia-ml.so.1 && \
    touch /usr/lib/wsl/lib/libcuda.so.1

# B70 runtime patches + start script (see start.sh for env knobs)
COPY opt-qwen38/ /opt/qwen38/

# Disable CUDA to avoid platform conflicts
ENV CUDA_VISIBLE_DEVICES="" VLLM_ALLOW_RUNTIME_PLUGIN_REGISTER=1

# WSL2/Arc production defaults. The launcher (start-*.ps1) passes every knob
# explicitly too, with B70_* env overrides; these are the fallbacks.
ENV VLLM_TARGET_DEVICE=xpu \
    VLLM_TRITON_USE_TD=1 \
    ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
    ZE_AFFINITY_MASK=0 \
    VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    CCL_ATL_TRANSPORT=ofi \
    CCL_ENABLE_SYCL_KERNELS=0 \
    MTP_TOKENS=3 \
    DRAFT_INT4=0 \
    MAX_NUM_SEQS=1 \
    MAX_MODEL_LEN=200000 \
    KV_CACHE_MEMORY_BYTES=8053063680 \
    KV_CACHE_DTYPE=fp8 \
    GPU_MEMORY_UTILIZATION=0.88 \
    MAX_NUM_BATCHED_TOKENS=8192 \
    PREFIX_CACHE=1 \
    ENFORCE_EAGER=0 \
    MM_IMAGES=16 \
    TOOL_CALL_PARSER=qwen3_xml \
    B70_FIX_ACCEPT_SYNC=1 \
    B70_FIX_BACKWARD_COPY=1 \
    B70_FIX_EAGLE_DROP=0 \
    MODEL_BOOTSTRAP=1 \
    HF_REPO=zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70

EXPOSE 8000
LABEL version="0.28.0-apcfix"
ENTRYPOINT ["bash", "/opt/qwen38/start.sh"]
