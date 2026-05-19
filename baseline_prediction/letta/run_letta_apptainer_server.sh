#!/usr/bin/env bash
set -euo pipefail

# MSI-friendly Letta server launcher via Apptainer.
#
# Typical use:
#   export AZURE_API_KEY="..."
#   export AZURE_BASE_URL="https://...openai.azure.com"
#   export AZURE_API_VERSION="2024-10-21"
#   bash baseline_prediction/letta/run_letta_apptainer_server.sh
#
# Optional:
#   LETTA_APPTAINER_SOURCE=docker://letta/letta:latest
#   LETTA_APPTAINER_IMAGE=$HOME/.letta/apptainer/letta_latest.sif
#   LETTA_PORT=8283
#   LETTA_PERSIST_DIR=$HOME/.letta/.persist/pgdata
#   LETTA_LOG_DIR=$HOME/.letta/logs
#   LETTA_BACKGROUND=true
#   LETTA_SECURE=false
#   LETTA_SERVER_PASSWORD=...

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[ERROR] apptainer not found. This launcher is intended for MSI-style environments."
  exit 1
fi

LETTA_APPTAINER_SOURCE="${LETTA_APPTAINER_SOURCE:-docker://letta/letta:latest}"
default_storage_root="$HOME/.letta"
LETTA_STORAGE_ROOT="${LETTA_STORAGE_ROOT:-$default_storage_root}"
LETTA_APPTAINER_IMAGE="${LETTA_APPTAINER_IMAGE:-$LETTA_STORAGE_ROOT/apptainer/letta_latest.sif}"
LETTA_PORT="${LETTA_PORT:-8283}"
LETTA_PERSIST_DIR="${LETTA_PERSIST_DIR:-$LETTA_STORAGE_ROOT/.persist/pgdata}"
LETTA_RUNTIME_DIR="${LETTA_RUNTIME_DIR:-$LETTA_STORAGE_ROOT/.persist/run_postgresql}"
LETTA_LOG_DIR="${LETTA_LOG_DIR:-$LETTA_STORAGE_ROOT/logs}"
LETTA_BACKGROUND="${LETTA_BACKGROUND:-true}"
LETTA_SECURE="${LETTA_SECURE:-false}"

mkdir -p "$(dirname "$LETTA_APPTAINER_IMAGE")" "$LETTA_PERSIST_DIR" "$LETTA_RUNTIME_DIR" "$LETTA_LOG_DIR"

if [[ ! -f "$LETTA_APPTAINER_IMAGE" ]]; then
  echo "[INFO] Pulling Letta image into Apptainer SIF:"
  echo "       source: $LETTA_APPTAINER_SOURCE"
  echo "       target: $LETTA_APPTAINER_IMAGE"
  apptainer pull "$LETTA_APPTAINER_IMAGE" "$LETTA_APPTAINER_SOURCE"
fi

env_args=()
for name in \
  OPENAI_API_KEY \
  AZURE_API_KEY \
  AZURE_BASE_URL \
  AZURE_API_VERSION \
  HOST \
  PORT \
  LETTA_PORT \
  LETTA_SERVER_PASSWORD; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    env_args+=(--env "${name}=${value}")
  fi
done

if [[ "$LETTA_SECURE" == "true" ]]; then
  env_args+=(--env "SECURE=true")
fi

env_args+=(--env "HOST=127.0.0.1")
env_args+=(--env "PORT=${LETTA_PORT}")
env_args+=(--env "ALEMBIC_CONFIG=/app/alembic.ini")

run_cmd=(
  apptainer exec
  --pwd /tmp
  --bind "${LETTA_PERSIST_DIR}:/var/lib/postgresql/data"
  --bind "${LETTA_RUNTIME_DIR}:/var/run/postgresql"
  "${env_args[@]}"
  "$LETTA_APPTAINER_IMAGE"
  bash
  -lc
  'unset LETTA_PG_URI LETTA_REDIS_HOST; export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt; export PATH=/app/.venv/bin:/usr/lib/postgresql/15/bin:$PATH; ln -sfn /app/alembic /tmp/alembic && exec /app/letta/server/startup.sh'
)

echo "[INFO] Letta Apptainer image: $LETTA_APPTAINER_IMAGE"
echo "[INFO] Persist dir          : $LETTA_PERSIST_DIR"
echo "[INFO] Runtime dir          : $LETTA_RUNTIME_DIR"
echo "[INFO] Expected base URL    : http://127.0.0.1:${LETTA_PORT}"

if [[ "$LETTA_BACKGROUND" == "true" ]]; then
  log_path="${LETTA_LOG_DIR}/letta_apptainer_server.log"
  pid_path="${LETTA_LOG_DIR}/letta_apptainer_server.pid"
  echo "[INFO] Starting Letta server in background..."
  nohup "${run_cmd[@]}" >"$log_path" 2>&1 &
  echo $! >"$pid_path"
  echo "[OK] Launched Letta server."
  echo "  PID file  : $pid_path"
  echo "  Log file  : $log_path"
else
  echo "[INFO] Starting Letta server in foreground..."
  exec "${run_cmd[@]}"
fi

echo
echo "Next step in the same shell/session:"
echo "  export LETTA_BASE_URL=http://127.0.0.1:${LETTA_PORT}"
if [[ "$LETTA_SECURE" == "true" ]]; then
  echo "  export LETTA_API_KEY=\"\$LETTA_SERVER_PASSWORD\""
else
  echo "  unset LETTA_API_KEY  # optional if local server has no auth"
fi
