#!/bin/bash
# EvalScope 性能压测（改「配置」区即可，需先 bash start_vllm_server.sh）
#
# pip install 'evalscope[perf]' -U

set -euo pipefail

# ============ 配置 ============
# 与 start_vllm_server.sh 保持一致
MODEL_PATH="/home/chyao/projects/models/qwen3-2.7b"
SERVED_MODEL_NAME="qwen3-2.7b"
API_BASE_URL="http://127.0.0.1:8000"

# 请求地址（random 用 chat；speed_benchmark 改为 ${API_BASE_URL}/v1/completions）
CHAT_COMPLETIONS_URL="${API_BASE_URL}/v1/chat/completions"

# 压测参数
PARALLEL="4"
NUMBER="8"
DATASET="random"
MIN_PROMPT_LENGTH=128
MAX_PROMPT_LENGTH=128
MIN_TOKENS=1280
MAX_TOKENS=1280

# ============ 运行 ============
echo "model:     ${SERVED_MODEL_NAME}"
echo "url:       ${CHAT_COMPLETIONS_URL}"
echo "tokenizer: ${MODEL_PATH}"
echo ""

evalscope perf \
  --parallel ${PARALLEL} \
  --number ${NUMBER} \
  --model "${SERVED_MODEL_NAME}" \
  --url "${CHAT_COMPLETIONS_URL}" \
  --api openai \
  --dataset "${DATASET}" \
  --min-prompt-length "${MIN_PROMPT_LENGTH}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --min-tokens "${MIN_TOKENS}" \
  --max-tokens "${MAX_TOKENS}" \
  --tokenizer-path "${MODEL_PATH}" \
  --extra-args '{"ignore_eos": true}' \
  "$@"
