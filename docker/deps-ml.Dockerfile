# IOWarp ML Inference Dependencies Container
# Extends deps-nvidia with llama.cpp, vLLM, and SGLang for LLM inference experiments.
#
# Provides:
#   - llama.cpp C++ library + server (CUDA-enabled, system-installed)
#   - llama-cpp-python (Python bindings for llama.cpp, CUDA-enabled)
#   - vLLM (packaged release for experimentation; submodule at external/vllm for dev)
#   - SGLang (packaged release for experimentation; submodule at external/sglang for dev)
#   - PyTorch with CUDA 12.6 support
#
# Usage:
#   docker build -t iowarp/deps-ml:latest -f docker/deps-ml.Dockerfile .
#
FROM iowarp/deps-nvidia:latest
LABEL maintainer="llogan@hawk.iit.edu"
LABEL description="IOWarp ML inference dependencies (llama.cpp + vLLM + SGLang)"

ARG DEBIAN_FRONTEND=noninteractive

USER root

#------------------------------------------------------------
# System dependencies for llama.cpp C++ build
#------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------
# Build and install llama.cpp C++ library + server (system-wide)
# This provides libllama.so, headers, and llama-server binary
# for the C++ integration work (iowarp-llm/).
# The workspace submodule at external/llama.cpp is used for
# modifying llama.cpp itself; this install is for stable baseline use.
#------------------------------------------------------------
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp-build \
    && cmake -B /tmp/llama.cpp-build/build -S /tmp/llama.cpp-build \
       -DGGML_CUDA=ON \
       -DBUILD_SHARED_LIBS=ON \
       -DLLAMA_BUILD_SERVER=ON \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/llama.cpp-build/build --config Release -j$(nproc) \
    && cmake --install /tmp/llama.cpp-build/build \
    && ldconfig \
    && rm -rf /tmp/llama.cpp-build

#------------------------------------------------------------
# Python ML stack (installed as iowarp user into conda base)
#------------------------------------------------------------
USER iowarp

# PyTorch with CUDA 12.6 support
RUN /home/iowarp/miniconda3/bin/pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# llama-cpp-python: Python bindings for llama.cpp, CUDA-enabled
# GGML_CUDA=ON matches the flag used for the C++ build above
RUN CMAKE_ARGS="-DGGML_CUDA=ON" \
    /home/iowarp/miniconda3/bin/pip install --no-cache-dir \
    llama-cpp-python

# vLLM packaged release (for quick experimentation and benchmarking baseline)
# The git submodule at external/vllm is for source-level development
RUN /home/iowarp/miniconda3/bin/pip install --no-cache-dir vllm

# SGLang packaged release (for quick experimentation)
# The git submodule at external/sglang is for iowarp-specific modifications
RUN /home/iowarp/miniconda3/bin/pip install --no-cache-dir "sglang[all]"

# Useful ML/inference utilities
RUN /home/iowarp/miniconda3/bin/pip install --no-cache-dir \
    transformers \
    huggingface_hub \
    accelerate \
    datasets \
    sentencepiece \
    numpy \
    pandas

#------------------------------------------------------------
# Update bashrc with ML environment info
#------------------------------------------------------------
RUN echo '' >> /home/iowarp/.bashrc \
    && echo '# ML Inference environment' >> /home/iowarp/.bashrc \
    && echo 'export LLAMA_CPP_HOME=/usr/local' >> /home/iowarp/.bashrc \
    && echo '# Run: llama-server -m <model.gguf> --n-gpu-layers 99 --port 8080' >> /home/iowarp/.bashrc

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
