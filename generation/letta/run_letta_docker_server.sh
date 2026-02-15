#!/usr/bin/env bash
set -euo pipefail

# One-click Letta Docker server launcher (OpenAI only).
#
# Required:
#   OPENAI_API_KEY
#
# Optional:
#   LETTA_IMAGE=letta/letta:latest
#   LETTA_CONTAINER_NAME=letta-server
#   LETTA_PORT=8283
#   LETTA_PERSIST_DIR=$HOME/.letta/.persist/pgdata
#   LETTA_SECURE=false
#   LETTA_SERVER_PASSWORD=<password>  # used only when LETTA_SECURE=true

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker not found. Please install Docker first."
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is not set."
  echo "Example:"
  echo "  export OPENAI_API_KEY='sk-xxxx'"
  exit 1
fi

LETTA_IMAGE="${LETTA_IMAGE:-letta/letta:latest}"
LETTA_CONTAINER_NAME="${LETTA_CONTAINER_NAME:-letta-server}"
LETTA_PORT="${LETTA_PORT:-8283}"
LETTA_PERSIST_DIR="${LETTA_PERSIST_DIR:-$HOME/.letta/.persist/pgdata}"
LETTA_SECURE="${LETTA_SECURE:-false}"

mkdir -p "$LETTA_PERSIST_DIR"

if docker ps -a --format '{{.Names}}' | grep -Eq "^${LETTA_CONTAINER_NAME}$"; then
  echo "[INFO] Removing existing container: ${LETTA_CONTAINER_NAME}"
  docker rm -f "$LETTA_CONTAINER_NAME" >/dev/null
fi

run_cmd=(
  docker run -d
  --name "$LETTA_CONTAINER_NAME"
  -v "${LETTA_PERSIST_DIR}:/var/lib/postgresql/data"
  -p "${LETTA_PORT}:8283"
  -e "OPENAI_API_KEY=${OPENAI_API_KEY}"
)

if [[ "$LETTA_SECURE" == "true" ]]; then
  run_cmd+=(-e "SECURE=true")
  if [[ -n "${LETTA_SERVER_PASSWORD:-}" ]]; then
    run_cmd+=(-e "LETTA_SERVER_PASSWORD=${LETTA_SERVER_PASSWORD}")
  fi
fi

run_cmd+=("$LETTA_IMAGE")

echo "[INFO] Starting Letta server container..."
"${run_cmd[@]}"

echo "[OK] Letta server is running."
echo "  Container : ${LETTA_CONTAINER_NAME}"
echo "  Base URL  : http://localhost:${LETTA_PORT}"
if [[ "$LETTA_SECURE" == "true" ]]; then
  echo "  Auth      : enabled (use LETTA_SERVER_PASSWORD as api_key)"
else
  echo "  Auth      : disabled"
fi
echo
echo "Health check:"
echo "  curl http://localhost:${LETTA_PORT}/v1/health"

