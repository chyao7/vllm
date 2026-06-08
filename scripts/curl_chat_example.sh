#!/bin/bash
# vLLM OpenAI 兼容 API 的 curl 示例（需先启动 start_vllm_server.sh）
#
# 用法:
#   bash scripts/curl_chat_example.sh
#   bash scripts/curl_chat_example.sh health
#   bash scripts/curl_chat_example.sh stream

set -euo pipefail

# ============ 配置（与 start_vllm_server.sh 保持一致）============
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-2.7b}"

MODE="${1:-chat}"

case "${MODE}" in
  health)
    echo "==> GET /health"
    curl -sS "${API_BASE_URL}/health"
    echo
    ;;
  models)
    echo "==> GET /v1/models"
    curl -sS "${API_BASE_URL}/v1/models" | python3 -m json.tool
    ;;
  stream)
    echo "==> POST /v1/chat/completions (stream)"
    curl -sS -N "${API_BASE_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$(cat <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "user", "content": "用一句话介绍你自己。"}
  ],
  "max_tokens": 64,
  "temperature": 0.7,
  "stream": true
}
EOF
)"
    echo
    ;;
  chat | *)
    echo "==> POST /v1/chat/completions"
    curl -sS "${API_BASE_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$(cat <<EOF
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "1+1等于几？只回答数字。"}
  ],
  "max_tokens": 32,
  "temperature": 0.0
}
EOF
)" | python3 -m json.tool
    ;;
esac
