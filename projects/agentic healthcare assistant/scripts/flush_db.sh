#!/usr/bin/env bash
# Remove local SQLite DB and FAISS index (paths match default config.yaml).
# Run from anywhere; resolves paths relative to the repository root.
#
# macOS / Linux / Git Bash on Windows: run ./scripts/flush_db.sh
# Native Windows (PowerShell):         run pwsh -File scripts/flush_db.ps1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_FILE="${REPO_ROOT}/data/healthcare.db"
FAISS_DIR="${REPO_ROOT}/data/faiss"

log() {
  printf '[flush-db] %s\n' "$*"
}

echo ""
log "This will permanently delete:"
log "  - SQLite: ${DB_FILE}"
log "  - FAISS:  ${FAISS_DIR}/"
echo ""
read -r -p "Type YES to delete all local patient data and vector index (anything else aborts): " reply
if [[ "${reply}" != "YES" ]]; then
  log "Aborted — no files were removed."
  exit 1
fi

echo ""
removed_any=false

if [[ -f "${DB_FILE}" ]]; then
  rm -f "${DB_FILE}"
  log "Removed SQLite database file: ${DB_FILE}"
  removed_any=true
else
  log "SQLite file not present (skipped): ${DB_FILE}"
fi

if [[ -d "${FAISS_DIR}" ]]; then
  rm -rf "${FAISS_DIR}"
  log "Removed FAISS directory: ${FAISS_DIR}"
  removed_any=true
else
  log "FAISS directory not present (skipped): ${FAISS_DIR}"
fi

echo ""
if [[ "${removed_any}" == true ]]; then
  log "Flush complete. Re-seed with: poetry run healthcare-cli --init-db"
  log "If Streamlit was running, restart it so cached resources reload."
else
  log "Nothing to remove — data paths were already empty."
fi
