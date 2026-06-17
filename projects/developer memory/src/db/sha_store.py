"""Thread-safe JSON file store for last-synced commit SHAs.

Spec §6A — stores commit SHA per (repo_url, branch) key so the SHA freshness gate
in delta_extractor can detect unchanged repos without cloning.

Storage: {CHROMA_DATA_PATH}/sha_store.json — pure stdlib, no ChromaDB dependency.
Thread-safe via threading.Lock so concurrent sync jobs in the same process are safe.
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path


class SHAStore:
    """Thread-safe JSON file store for last-synced commit SHAs and skills generation SHAs.

    Reads/writes {CHROMA_DATA_PATH}/sha_store.json.
    No ChromaDB dependency — stdlib only.

    Schema (per key):
        {
            "https://github.com/org/repo:main": {
                "sha": "abc123def456...",           # last synced commit SHA
                "synced_at": "2026-05-05T...",      # ISO timestamp of last sync
                "skills_sha": "abc123def456...",    # commit SHA when skills were generated
                "skills_path": "/app/PROJECT_SKILLS.md",  # path of generated file
                "skills_generated_at": "2026-05-13T..."   # ISO timestamp of skills generation
            }
        }

    All write operations use get-merge-update: existing fields are preserved when new
    fields are written. This ensures set_last_sha does not wipe skills_sha / skills_path,
    and set_skills_sha does not wipe sha / synced_at.
    """

    def __init__(self, data_path: str | None = None) -> None:
        base = data_path or os.environ.get("CHROMA_DATA_PATH", "chroma_data")
        self._path = Path(base) / "sha_store.json"
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_url(repo_url: str) -> str:
        """Strip 'www.' prefix from hostname so github.com and www.github.com resolve identically."""
        if "://www." in repo_url:
            return repo_url.replace("://www.", "://", 1)
        return repo_url

    def _key(self, repo_url: str, branch: str) -> str:
        """Format the store key as 'normalized_repo_url:branch'."""
        return f"{self._normalize_url(repo_url)}:{branch}"

    def _read_data(self) -> dict:
        """Read sha_store.json under the lock, returning empty dict on any error.

        Must be called with self._lock already held.
        """
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _write_data(self, data: dict) -> None:
        """Write data dict to sha_store.json under the lock.

        Must be called with self._lock already held.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_last_sha(self, repo_url: str, branch: str) -> str | None:
        """Return last-synced SHA or None on miss/read error.

        Args:
            repo_url: Git repository URL.
            branch: Branch name.

        Returns:
            SHA hex string if previously synced and stored; None on cache miss or
            any read/parse error (safe — caller falls through to full clone).
        """
        try:
            with self._lock:
                data = self._read_data()
            return data.get(self._key(repo_url, branch), {}).get("sha")
        except Exception:  # noqa: BLE001
            return None

    def set_last_sha(self, repo_url: str, branch: str, sha: str) -> None:
        """Upsert last-synced SHA using get-merge-update. Creates the JSON file on first write.

        Uses get-merge-update so existing skills_sha / skills_path / skills_generated_at
        fields are preserved — a new sync does not wipe a prior skills generation record.

        Args:
            repo_url: Git repository URL.
            branch: Branch name.
            sha: Commit SHA to store.
        """
        key = self._key(repo_url, branch)
        with self._lock:
            data = self._read_data()
            existing = data.get(key, {})
            # Merge: update only sha + synced_at; preserve all other fields
            existing["sha"] = sha
            existing["synced_at"] = datetime.now(UTC).isoformat()
            data[key] = existing
            self._write_data(data)

    def get_skills_entry(self, repo_url: str, branch: str) -> dict[str, str]:
        """Return the skills cache entry for (repo_url, branch), or empty dict on miss.

        Returns the full skills sub-dict so the caller can read both 'skills_sha' and
        'skills_path' in a single lock acquisition — avoids a TOCTOU window from two
        separate lookups.

        Args:
            repo_url: Git repository URL.
            branch: Branch name.

        Returns:
            Dict with keys 'skills_sha', 'skills_path', 'skills_generated_at' when
            present. Empty dict on miss, file-absent, parse error, or missing skills fields.
        """
        try:
            with self._lock:
                data = self._read_data()
            record = data.get(self._key(repo_url, branch), {})
            # Return only the skills fields — empty dict if any are absent (cache miss)
            if "skills_sha" not in record:
                return {}
            return {
                "skills_sha": record.get("skills_sha", ""),
                "skills_path": record.get("skills_path", ""),
                "skills_generated_at": record.get("skills_generated_at", ""),
            }
        except Exception:  # noqa: BLE001
            return {}

    def set_skills_sha(
        self,
        repo_url: str,
        branch: str,
        sha: str,
        skills_path: str,
    ) -> None:
        """Record that PROJECT_SKILLS.md was successfully generated at the given SHA.

        Uses get-merge-update pattern — reads current record, merges new skills fields,
        writes back. Preserves existing 'sha' and 'synced_at' fields.

        Args:
            repo_url: Git repository URL.
            branch: Branch name.
            sha: Commit SHA at which skills were generated.
            skills_path: Absolute path where PROJECT_SKILLS.md was written.
        """
        key = self._key(repo_url, branch)
        with self._lock:
            data = self._read_data()
            existing = data.get(key, {})
            # Merge: update only skills fields; preserve sha + synced_at
            existing["skills_sha"] = sha
            existing["skills_path"] = skills_path
            existing["skills_generated_at"] = datetime.now(UTC).isoformat()
            data[key] = existing
            self._write_data(data)
