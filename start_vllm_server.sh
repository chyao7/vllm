#!/bin/bash

# Conda ICU/sqlite3 需要 CXXABI_1.3.15；系统 /usr/lib/x86_64-linux-gnu 的 libstdc++ 过旧
CONDA_ENV="${CONDA_PREFIX:-/home/chyao/softwares/miniconda3/envs/vllm}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/wsl/lib"

# vLLM 扩展若链接 libcudart.so.13 而 PyTorch 用 libcudart.so.12，会在 profile_run 触发 free(): invalid pointer
_VLLM_C_SO="$("${CONDA_ENV}/bin/python" -c "import glob, vllm, os; print(glob.glob(os.path.join(os.path.dirname(vllm.__file__), '_C*.so'))[0])" 2>/dev/null)" || true
if [[ -n "${_VLLM_C_SO}" ]] && ldd "${_VLLM_C_SO}" 2>/dev/null | grep -q 'libcudart.so.13'; then
  _TORCH_CUDA_SO="$("${CONDA_ENV}/bin/python" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib/libtorch_cuda.so'))" 2>/dev/null)" || true
  if [[ -n "${_TORCH_CUDA_SO}" ]] && ldd "${_TORCH_CUDA_SO}" 2>/dev/null | grep -q 'libcudart.so.12'; then
    echo "ERROR: CUDA 运行时版本不一致：vLLM 扩展需要 libcudart.so.13，PyTorch 使用 libcudart.so.12。"
    echo "请对齐后重装：升级 torch 到 cu130 并重新 pip install -e .，或用 CUDA12 nvcc 重编 vLLM。"
    exit 1
  fi
fi

vllm serve /home/chyao/projects/models/qwen3-2.7b \
    --served-model-name qwen3-2.7b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --max-model-len 3276 \
    --max-num-batched-tokens 1338 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --skip-mm-profiling

echo "vLLM server stopped."


# cd /home/chyao/projects/vllm
# # 自动找 _C 用的 ninja 目录（排除 .deps 里的 subbuild）
# BUILD_DIR=$(find build/temp.* -maxdepth 1 -name build.ninja 2>/dev/null | head -1)
# BUILD_DIR=$(dirname "$BUILD_DIR")
# echo "BUILD_DIR=$BUILD_DIR"
# # 只重编并安装 _C
# cmake --build "$BUILD_DIR" --target _C -j8
# t