#!/usr/bin/env bash
# Build only vllm._C (includes minimax_m2 fused QKV), skipping flash-attn / _moe_C / etc.
#
# Usage:
#   bash scripts/build_minimax_m2_ops.sh
#
# Environment:
#   PYTHON                Python executable (default: .venv/bin/python)
#   MAX_JOBS              Parallel compile jobs (default: nproc)
#   NVCC_THREADS          NVCC threads per job (default: 1)
#   TORCH_CUDA_ARCH_LIST  GPU arch list, e.g. "9.0a" for H100 (required for CUTLASS TMA)
#   VLLM_CUTLASS_SRC_DIR  Local CUTLASS checkout (default: .deps/cutlass-src)
#   CMAKE_BUILD_TYPE      Default: RelWithDebInfo
#
# If pip install -e . fails with "Failed to clone cutlass", use this script instead:
#   export VLLM_CUTLASS_SRC_DIR=/path/to/cutlass   # v4.2.1
#   bash scripts/build_minimax_m2_ops.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"

CUTLASS_DIR="${VLLM_CUTLASS_SRC_DIR:-${ROOT}/.deps/cutlass-src}"
if [[ -f "${CUTLASS_DIR}/include/cutlass/cutlass.h" ]]; then
  export VLLM_CUTLASS_SRC_DIR="${CUTLASS_DIR}"
  echo "==> Using local CUTLASS: ${VLLM_CUTLASS_SRC_DIR}"
else
  echo "ERROR: CUTLASS v4.2.1 not found at ${CUTLASS_DIR}" >&2
  echo "GitHub HTTPS is often unstable here. Clone manually, then retry:" >&2
  echo "  git clone --depth 1 --branch v4.2.1 https://github.com/nvidia/cutlass.git ${CUTLASS_DIR}" >&2
  exit 1
fi

PYTAG="$("${PYTHON}" -c 'import sys; print(f"cpython-{sys.version_info.major}{sys.version_info.minor}")')"
BUILD_TEMP="${ROOT}/build/temp.linux-$(uname -m)-${PYTAG}"
ARCH_MARKER="${BUILD_TEMP}/.torch_cuda_arch_list"
PYTHON_PATH="$("${PYTHON}" -c 'import sys; print(":".join(sys.path))')"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"

echo "==> Python: ${PYTHON}"
echo "==> Build dir: ${BUILD_TEMP}"
echo "==> MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS}, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

if [[ -f "${BUILD_TEMP}/build.ninja" ]]; then
  if [[ ! -f "${ARCH_MARKER}" ]] || [[ "$(cat "${ARCH_MARKER}")" != "${TORCH_CUDA_ARCH_LIST}" ]]; then
    echo "==> TORCH_CUDA_ARCH_LIST changed (${TORCH_CUDA_ARCH_LIST}); reconfiguring CMake ..."
    rm -rf "${BUILD_TEMP}"
  fi
fi

if [[ ! -f "${BUILD_TEMP}/build.ninja" ]]; then
  echo "==> Configuring CMake (first time only) ..."
  mkdir -p "${BUILD_TEMP}"
  cmake "${ROOT}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="${PYTHON}" \
    -DVLLM_PYTHON_PATH="${PYTHON_PATH}" \
    -DFETCHCONTENT_BASE_DIR="${ROOT}/.deps" \
    -DNVCC_THREADS="${NVCC_THREADS}" \
    -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
    -DCMAKE_JOB_POOLS:STRING="compile=${MAX_JOBS}" \
    -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
    -B "${BUILD_TEMP}"
  echo "${TORCH_CUDA_ARCH_LIST}" > "${ARCH_MARKER}"
fi

echo "==> Building target _C ..."
cmake --build "${BUILD_TEMP}" --target _C -j"${MAX_JOBS}"

echo "==> Installing component _C into ${ROOT}/vllm/ ..."
cmake --install "${BUILD_TEMP}" --prefix "${ROOT}" --component _C

echo "==> Done. Installed:"
ls -la "${ROOT}"/vllm/_C*.so
