# Qwen3.8-27B (huihui abliterated) GPTQ-INT4 MTP-BF16 — single Intel Arc Pro B70
# One-click serving image (Windows Docker Desktop + WSL2 or native Linux).
# Based on the in-use image (which already has the original base + kernels).
FROM zrlu/qwen38-27b-arc-pro-b70:latest
RUN pip install --no-cache-dir vllm-xpu-kernels==0.1.12.3 && \
    python -c "import vllm_xpu_kernels, importlib.metadata as m; print('vllm-xpu-kernels', m.version('vllm-xpu-kernels'))"

# Create stub files to prevent NVIDIA library lookups on WSL2/Arc systems
RUN mkdir -p /usr/lib/wsl/lib && \
    touch /usr/lib/wsl/lib/libnvidia-ml.so.1 && \
    touch /usr/lib/wsl/lib/libcuda.so.1

# B70 runtime patches + start script (see start.sh for env knobs)
COPY opt-qwen38/ /opt/qwen38/

# Disable CUDA to avoid platform conflicts
ENV CUDA_VISIBLE_DEVICES="" VLLM_ALLOW_RUNTIME_PLUGIN_REGISTER=1

# WSL2/Arc production defaults
ENV B70_THINKING_BUDGET=4096 \
    CUDA_VISIBLE_DEVICES="" \
    VLLM_ALLOW_RUNTIME_PLUGIN_REGISTER=1 \
    VLLM_TARGET_DEVICE=xpu \
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
    MAX_MODEL_LEN=100000 \
    KV_CACHE_MEMORY_BYTES=4617089843 \
    KV_CACHE_DTYPE=fp8 \
    GPU_MEMORY_UTILIZATION=0.88 \
    MAX_NUM_BATCHED_TOKENS=8192 \
    PREFIX_CACHE=1 \
    ENFORCE_EAGER=0 \
    MM_IMAGES=16 \
    TOOL_CALL_PARSER=qwen3_xml \
    MODEL_BOOTSTRAP=1 \
    HF_REPO=zrlu/Huihui-Qwen3.8-27B-abliterated-GPTQ-Int4-sym-G128-MTP-BF16-B70

EXPOSE 8000
LABEL version="v14"
ENTRYPOINT ["bash", "/opt/qwen38/start.sh"]
