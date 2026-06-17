#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch 8-GPU tensor-parallel benchmark for MiniMax-M2 fused QKV.
#
# Usage (from repo root):
#   bash benchmarks/kernels/run_minimax_m2_fused_qkv_8gpu.sh
#
# Optional env overrides:
#   NPROC=8
#   NUM_TOKENS="1 64 256 1024 4096"
#   WARMUP=10 ITERS=100
#   EXTRA_ARGS="--compare-eager --include-gemm-only"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

NPROC="${NPROC:-8}"
NUM_TOKENS="${NUM_TOKENS:-1 4 16 64 256 1024 4096}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "Repo:      $ROOT"
echo "Python:    $PYTHON"
echo "GPUs:      $NPROC"
echo "Tokens:    $NUM_TOKENS"
echo "Warmup:    $WARMUP  Iters: $ITERS"
echo

# Use the same Python as vLLM (.venv) — plain `torchrun` may pick /usr/bin/python3.
exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  benchmarks/kernels/benchmark_minimax_m2_fused_qkv.py \
  --profile-stages \
  --compare-eager \
  --compare-staged \
  --num-tokens $NUM_TOKENS \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  $EXTRA_ARGS
