#!/usr/bin/env bash
# Run all cli tests: health, list-db, then scenario CLI runs with logging.
# Run from anywhere; uses repo root for poetry.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_FILE="${REPO_ROOT}/phase9_gate.log"

{
  echo "=== Phase 9 Gate � $(date) ==="
  echo ""
  echo "--- --health ---"
  poetry run healthcare-cli --health
  echo ""
  echo "--- --list-db ---"
  poetry run healthcare-cli --list-db
  echo ""
  echo "--- hypertension --strict ---"
  poetry run healthcare-cli --scenario hypertension --strict
  echo ""
  echo "--- diabetes --strict --show-trace ---"
  poetry run healthcare-cli --scenario diabetes --strict --show-trace
  echo ""
  echo "--- ckd --strict ---"
  poetry run healthcare-cli --scenario ckd --strict
  echo ""
  echo "--- all --strict ---"
  poetry run healthcare-cli --scenario all --strict
  echo ""
} 2>&1 | tee "${LOG_FILE}"
