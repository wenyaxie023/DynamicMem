#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_export_sa_tasks.sh"
bash "$SCRIPT_DIR/run_eval_sa_api.sh"