"""Tests for SHAStore (src/db/sha_store.py).

Uses tmp_path fixture — no live ChromaDB, no env side effects.

Cases:
  - Miss (no file) → None
  - Hit → correct SHA returned
  - First write creates file
  - Second write updates without corrupting
  - Different (repo, branch) keys are isolated
  - Corrupted JSON → get_last_sha returns None (safe)
  - Concurrent threads do not corrupt the file
  - set_last_sha preserves skills_sha / skills_path (get-merge-update)
  - get_skills_entry / set_skills_sha round-trip
  - www normalization applies to skills methods
"""

import json
import threading
from pathlib import Path

from src.db.sha_store import SHAStore


class TestSHAStoreMiss:
    def test_miss_returns_none_when_file_absent(self, tmp_path: Path) -> None:
        """No sha_store.json yet → get_last_sha returns None."""
        store = SHAStore(data_path=str(tmp_path))
        result = store.get_last_sha("https://github.com/user/repo", "main")
        assert result is None

    def test_miss_for_unknown_key(self, tmp_path: Path) -> None:
        """Key not in file → None."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_last_sha("https://github.com/user/repo", "main", "abc123")
        result = store.get_last_sha("https://github.com/other/repo", "main")
        assert result is None


class TestSHAStoreHit:
    def test_hit_returns_correct_sha(self, tmp_path: Path) -> None:
        """After set_last_sha, get_last_sha returns the same SHA."""
        store = SHAStore(data_path=str(tmp_path))
        sha = "deadbeef" * 8
        store.set_last_sha("https://github.com/user/repo", "main", sha)
        result = store.get_last_sha("https://github.com/user/repo", "main")
        assert result == sha

    def test_hit_correct_sha_for_branch(self, tmp_path: Path) -> None:
        """SHA is keyed per branch — different branches return different SHAs."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_last_sha("https://github.com/user/repo", "main", "sha_main")
        store.set_last_sha("https://github.com/user/repo", "feature", "sha_feature")
        assert store.get_last_sha("https://github.com/user/repo", "main") == "sha_main"
        assert store.get_last_sha("https://github.com/user/repo", "feature") == "sha_feature"


class TestSHAStoreFileCreation:
    def test_first_write_creates_file(self, tmp_path: Path) -> None:
        """set_last_sha creates sha_store.json if it does not exist."""
        store = SHAStore(data_path=str(tmp_path))
        json_path = tmp_path / "sha_store.json"
        assert not json_path.exists()

        store.set_last_sha("https://github.com/user/repo", "main", "abc123")

        assert json_path.exists()

    def test_first_write_valid_json(self, tmp_path: Path) -> None:
        """Written file contains valid JSON with expected structure."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_last_sha("https://github.com/user/repo", "main", "abc123")

        data = json.loads((tmp_path / "sha_store.json").read_text())
        key = "https://github.com/user/repo:main"
        assert key in data
        assert data[key]["sha"] == "abc123"
        assert "synced_at" in data[key]

    def test_second_write_updates_sha(self, tmp_path: Path) -> None:
        """Second set_last_sha for same key updates without corrupting other keys."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_last_sha("https://github.com/user/repo", "main", "sha_v1")
        store.set_last_sha("https://github.com/user/other", "main", "other_sha")
        store.set_last_sha("https://github.com/user/repo", "main", "sha_v2")

        assert store.get_last_sha("https://github.com/user/repo", "main") == "sha_v2"
        assert store.get_last_sha("https://github.com/user/other", "main") == "other_sha"


class TestSHAStoreCorruption:
    def test_corrupted_json_get_returns_none(self, tmp_path: Path) -> None:
        """Corrupted sha_store.json → get_last_sha returns None (safe, no raise)."""
        json_path = tmp_path / "sha_store.json"
        json_path.write_text("{{not valid json!", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        result = store.get_last_sha("https://github.com/user/repo", "main")
        assert result is None

    def test_corrupted_json_set_recovers(self, tmp_path: Path) -> None:
        """Corrupted sha_store.json → set_last_sha starts fresh without raising."""
        json_path = tmp_path / "sha_store.json"
        json_path.write_text("not json", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_last_sha("https://github.com/user/repo", "main", "abc123")
        assert store.get_last_sha("https://github.com/user/repo", "main") == "abc123"


class TestSHAStoreSetLastShaPreservesSkillsFields:
    def test_set_last_sha_preserves_skills_fields(self, tmp_path: Path) -> None:
        """set_last_sha after set_skills_sha keeps skills_sha / skills_path in JSON.

        Validates the get-merge-update migration: a new sync must not wipe the
        skills cache entry written by a prior generate_skills_pkg call.
        """
        store = SHAStore(data_path=str(tmp_path))
        url = "https://github.com/user/repo"
        store.set_last_sha(url, "main", "sync_sha_v1")
        store.set_skills_sha(url, "main", "sync_sha_v1", "/app/PROJECT_SKILLS.md")

        # Simulate a new sync completing — must not wipe skills fields
        store.set_last_sha(url, "main", "sync_sha_v2")

        data = json.loads((tmp_path / "sha_store.json").read_text())
        record = data["https://github.com/user/repo:main"]
        assert record["sha"] == "sync_sha_v2"
        assert record["skills_sha"] == "sync_sha_v1"
        assert record["skills_path"] == "/app/PROJECT_SKILLS.md"
        assert "skills_generated_at" in record


class TestGetSkillsEntry:
    def test_miss_no_file(self, tmp_path: Path) -> None:
        """Returns {} when sha_store.json is absent."""
        store = SHAStore(data_path=str(tmp_path))
        assert store.get_skills_entry("https://github.com/user/repo", "main") == {}

    def test_miss_no_skills_field(self, tmp_path: Path) -> None:
        """Returns {} when key exists but skills_sha field is absent (backward compat)."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_last_sha("https://github.com/user/repo", "main", "abc")
        assert store.get_skills_entry("https://github.com/user/repo", "main") == {}

    def test_hit_returns_dict(self, tmp_path: Path) -> None:
        """Returns dict with skills_sha, skills_path, skills_generated_at after set_skills_sha."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha("https://github.com/user/repo", "main", "abc123", "/app/s.md")
        entry = store.get_skills_entry("https://github.com/user/repo", "main")
        assert entry["skills_sha"] == "abc123"
        assert entry["skills_path"] == "/app/s.md"
        assert "skills_generated_at" in entry

    def test_sha_matches(self, tmp_path: Path) -> None:
        """entry['skills_sha'] equals the SHA passed to set_skills_sha."""
        store = SHAStore(data_path=str(tmp_path))
        sha = "deadbeef" * 10
        store.set_skills_sha("https://github.com/user/repo", "main", sha, "/tmp/s.md")
        assert store.get_skills_entry("https://github.com/user/repo", "main")["skills_sha"] == sha

    def test_corrupted_json_returns_empty(self, tmp_path: Path) -> None:
        """Returns {} on corrupted file (safe)."""
        (tmp_path / "sha_store.json").write_text("{bad", encoding="utf-8")
        store = SHAStore(data_path=str(tmp_path))
        assert store.get_skills_entry("https://github.com/user/repo", "main") == {}

    def test_www_normalization_applies(self, tmp_path: Path) -> None:
        """www.github.com and github.com resolve to the same key."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha("https://github.com/user/repo", "main", "sha1", "/tmp/s.md")
        entry = store.get_skills_entry("https://www.github.com/user/repo", "main")
        assert entry.get("skills_sha") == "sha1"


class TestSetSkillsSha:
    def test_adds_skills_fields(self, tmp_path: Path) -> None:
        """skills_sha, skills_path, skills_generated_at are present in JSON after write."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha("https://github.com/user/repo", "main", "sha1", "/a/b.md")
        data = json.loads((tmp_path / "sha_store.json").read_text())
        record = data["https://github.com/user/repo:main"]
        assert record["skills_sha"] == "sha1"
        assert record["skills_path"] == "/a/b.md"
        assert "skills_generated_at" in record

    def test_preserves_sync_sha(self, tmp_path: Path) -> None:
        """set_skills_sha does not overwrite existing sha or synced_at."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_last_sha("https://github.com/user/repo", "main", "sync_sha")
        store.set_skills_sha("https://github.com/user/repo", "main", "skills_sha", "/p.md")
        assert store.get_last_sha("https://github.com/user/repo", "main") == "sync_sha"

    def test_updates_existing_skills_fields(self, tmp_path: Path) -> None:
        """Second set_skills_sha call overwrites skills_sha/skills_path but preserves sha."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_last_sha("https://github.com/user/repo", "main", "sync_sha")
        store.set_skills_sha("https://github.com/user/repo", "main", "sha_v1", "/old.md")
        store.set_skills_sha("https://github.com/user/repo", "main", "sha_v2", "/new.md")

        entry = store.get_skills_entry("https://github.com/user/repo", "main")
        assert entry["skills_sha"] == "sha_v2"
        assert entry["skills_path"] == "/new.md"
        assert store.get_last_sha("https://github.com/user/repo", "main") == "sync_sha"

    def test_creates_file_on_first_write(self, tmp_path: Path) -> None:
        """set_skills_sha creates sha_store.json when it does not exist."""
        store = SHAStore(data_path=str(tmp_path))
        assert not (tmp_path / "sha_store.json").exists()
        store.set_skills_sha("https://github.com/user/repo", "main", "sha1", "/p.md")
        assert (tmp_path / "sha_store.json").exists()


class TestSHAStoreConcurrency:
    def test_concurrent_writes_do_not_corrupt(self, tmp_path: Path) -> None:
        """Multiple threads writing different keys do not corrupt each other."""
        store = SHAStore(data_path=str(tmp_path))
        errors: list[Exception] = []

        def writer(idx: int) -> None:
            try:
                url = f"https://github.com/user/repo{idx}"
                store.set_last_sha(url, "main", f"sha{idx:04d}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

        # Verify all writes are readable
        for i in range(20):
            url = f"https://github.com/user/repo{i}"
            result = store.get_last_sha(url, "main")
            assert result == f"sha{i:04d}", f"Mismatch for repo{i}: got {result}"
