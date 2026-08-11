# syntax=docker/dockerfile:1.6
FROM jfrog.fkinternal.com/fk-base-images/python:3.12.0-debian12.10

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    NVTE_FRAMEWORK=pytorch

# ---- OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates gnupg \
      libopenmpi-dev openmpi-bin \
      fuse pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_TRUSTED_HOST="10.24.14.195"
ENV PIP_INDEX_URL="http://10.24.14.195/artifactory/api/pypi/python_virtual/simple"

RUN pip install --no-cache-dir uv

WORKDIR /workspace

COPY nvidia_cuda_nvcc-13.0.88-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
     nvidia_cuda_cccl-13.0.85-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
     /workspace/
RUN pip install --no-cache-dir \
      "nvidia-nvvm==13.0.*" \
      "nvidia-cuda-crt==13.0.*" \
      "nvidia-cuda-runtime==13.0.*" \
      /workspace/nvidia_cuda_nvcc-13.0.88-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
      /workspace/nvidia_cuda_cccl-13.0.85-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
    && rm /workspace/nvidia_cuda_nvcc-*.whl /workspace/nvidia_cuda_cccl-*.whl


RUN set -eux; \
    SITE_PKG="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    D="$SITE_PKG/nvidia/cu13"; \
    mkdir -p /usr/local/cuda; \
    ln -sf "$D"/bin     /usr/local/cuda/bin; \
    ln -sf "$D"/include /usr/local/cuda/include; \
    ln -sf "$D"/lib     /usr/local/cuda/lib64; \
    for f in "$D"/lib/*.so.*; do \
      b="${f%%.so.*}.so"; \
      [ -e "$b" ] || ln -s "$(basename "$f")" "$b"; \
    done; \
    which nvcc; which ptxas; \
    echo "--- nvcc ---"; /usr/local/cuda/bin/nvcc --version; \
    echo "--- ptxas ---"; /usr/local/cuda/bin/ptxas --version; \
    echo "--- compiler package versions (all must be 13.0.x) ---"; \
    pip list 2>/dev/null | grep -Ei 'nvidia-(nvvm|cuda-crt|cuda-nvcc|cuda-cccl|cuda-runtime)'; \
    test -f /usr/local/cuda/include/nv/target; \
    test -e /usr/local/cuda/lib64/libcudart.so; \
    echo "--- smoke compile + link for sm_90 ---"; \
    printf '#include <cuda_runtime.h>\n__global__ void k(){}\nint main(){k<<<1,1>>>();return 0;}\n' > /tmp/t.cu; \
    nvcc -arch=sm_90 /tmp/t.cu -o /tmp/t.bin -L/usr/local/cuda/lib64 -lcudart; \
    rm /tmp/t.cu /tmp/t.bin

RUN pip install --no-cache-dir torch torchvision
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"


RUN set -eux; \
    SITE_PKG="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    for d in "$SITE_PKG"/nvidia/*/; do \
      case "$d" in */cu13/) continue;; esac; \
      [ -d "$d/include" ] && ln -sf "$d"include/* /usr/local/cuda/include/ || true; \
      [ -d "$d/lib" ]     && ln -sf "$d"lib/*.so* /usr/local/cuda/lib64/  || true; \
    done; \
    for f in /usr/local/cuda/lib64/*.so.*; do \
      b="${f%%.so.*}.so"; \
      [ -e "$b" ] || ln -s "$(basename "$f")" "$b"; \
    done; \
    test -e /usr/local/cuda/include/cudnn.h; \
    test -e /usr/local/cuda/include/nccl.h; \
    test -e /usr/local/cuda/lib64/libcudnn.so; \
    test -e /usr/local/cuda/lib64/libnccl.so


RUN pip install --no-cache-dir packaging ninja
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    TORCH_CUDA_ARCH_LIST="9.0"

RUN MAX_JOBS=4 pip wheel --no-cache-dir --no-build-isolation --no-deps \
      -w /workspace/wheels causal-conv1d \
    && pip install --no-cache-dir /workspace/wheels/causal_conv1d-*.whl
RUN MAX_JOBS=4 pip wheel --no-cache-dir --no-build-isolation --no-deps \
      -w /workspace/wheels mamba-ssm \
    && pip install --no-cache-dir /workspace/wheels/mamba_ssm-*.whl


ENV CUDNN_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn
RUN MAX_JOBS=4 pip wheel --no-cache-dir --no-build-isolation --no-deps \
      -w /workspace/wheels transformer_engine_torch "transformer-engine[pytorch]" \
    && pip install --no-cache-dir /workspace/wheels/transformer_engine*.whl

RUN MAX_JOBS=2 NVCC_THREADS=2 pip wheel --no-cache-dir --no-build-isolation --no-deps \
      -w /workspace/wheels flash-attn \
    && pip install --no-cache-dir /workspace/wheels/flash_attn-*.whl

COPY megatron-bridge.tar.gz /workspace/
RUN tar xzf megatron-bridge.tar.gz && rm megatron-bridge.tar.gz
WORKDIR /workspace/megatron-bridge
RUN pip install --no-cache-dir --no-build-isolation -e .

# ---- gcsfuse (optional — GKE CSI driver handles mounting in-cluster) ----
# NOTE: needs egress to packages.cloud.google.com. If the build VM can't
# reach it, delete this block — the GKE CSI driver makes it unnecessary
# in-cluster anyway.
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-bookworm main" \
      | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y --no-install-recommends gcsfuse \
    && rm -rf /var/lib/apt/lists/*

# ---- hand /workspace back to the app user, restore original user ----
RUN chown -R app:app /workspace

WORKDIR /workspace/megatron-bridge
USER app
