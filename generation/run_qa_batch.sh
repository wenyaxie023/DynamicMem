#!/usr/bin/env bash
set -euo pipefail

# Batch wrapper around generation.run_qa.
# Edit CONFIG + USERS below, or pass --config/--defaults/--dry-run and extra args.

CONFIG="configs/baselines/oracle_qa.yaml"
DEFAULTS="configs/qa.default.yaml"
USERS=(
  "001_user_001"
  "002_user_002"
  "003_user_003"
  "004_user_004"
  "005_user_005"
  "006_user_006"
  "007_user_007"
  "008_user_008"
  "009_user_009"
  "010_user_010"
)

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --defaults)
      DEFAULTS="$2"
      shift 2
      ;;
    --users)
      IFS=',' read -r -a USERS <<< "$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "=========================================="
echo "Starting QA Batch"
echo "Config: ${CONFIG}"
echo "Users: ${USERS[*]}"
echo "=========================================="

for user in "${USERS[@]}"; do
  echo "[QA] user=${user}"
  python3 -m generation.run_qa \
    --config "$CONFIG" \
    --defaults "$DEFAULTS" \
    --user-id "$user" \
    "${EXTRA_ARGS[@]}"
done

echo "=========================================="
echo "QA Batch Complete"
echo "=========================================="
