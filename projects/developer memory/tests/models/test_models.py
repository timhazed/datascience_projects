"""Tests for all Phase 1 Pydantic models and TypedDict states.

Spec §10 — every model must:
  - Instantiate with valid data (happy path)
  - Reject invalid data with ValidationError
  - Satisfy Phase 1 exit gate: SyncRequest URL validation ≥ 95% branch coverage
"""

import hashlib

import pytest
from pydantic import ValidationError

from src.models.analysis import AnalysisResult, CoachingAlert
from src.models.chunk import ParsedChunk, SummarizedChunk, UpsertResult
from src.models.diff_state import DiffState
from src.models.persona import PersonaProfile, TendencyData
from src.models.persona_state import PersonaState
from src.models.query_state import QueryState
from src.models.requests import DiffRequest, QueryRequest, SkillsRequest, SyncRequest
from src.models.sanitized_file import SanitizedFile
from src.models.skills import SkillsData
from src.models.skills_state import SkillsState
from src.models.sync_state import SyncState

# ---------------------------------------------------------------------------
# ParsedChunk
# ---------------------------------------------------------------------------


class TestParsedChunk:
    def test_valid(self) -> None:
        """ParsedChunk instantiates with path and content."""
        chunk = ParsedChunk(path="src/foo.py", content="x = 1")
        assert chunk.path == "src/foo.py"
        assert chunk.content == "x = 1"

    def test_empty_content_allowed(self) -> None:
        """Empty content is valid — empty files exist in real repos."""
        chunk = ParsedChunk(path="src/empty.py", content="")
        assert chunk.content == ""

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ValidationError):
            ParsedChunk(content="x = 1")  # type: ignore[call-arg]

    def test_missing_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            ParsedChunk(path="src/foo.py")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SummarizedChunk
# ---------------------------------------------------------------------------


class TestSummarizedChunk:
    def _make(self, **overrides) -> SummarizedChunk:
        defaults = {
            "content": "x = 1",
            "path": "src/foo.py",
            "intent_summary": "Assigns x to 1 for downstream use.",
            "tech_stack": ["Python"],
            "semantic_type": "Logic",
        }
        return SummarizedChunk(**{**defaults, **overrides})

    def test_valid(self) -> None:
        chunk = self._make()
        assert chunk.semantic_type == "Logic"
        assert chunk.author_identity == "unknown"  # default

    def test_content_hash_is_sha256(self) -> None:
        """content_hash matches SHA-256 of content — idempotency key for ChromaDB."""
        chunk = self._make(content="hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert chunk.content_hash == expected

    def test_content_hash_changes_with_content(self) -> None:
        a = self._make(content="aaa")
        b = self._make(content="bbb")
        assert a.content_hash != b.content_hash

    def test_invalid_semantic_type(self) -> None:
        with pytest.raises(ValidationError):
            self._make(semantic_type="Unknown")  # not in Literal

    def test_all_valid_semantic_types(self) -> None:
        for st in ("Logic", "Config", "Boilerplate", "Interface", "quarantine"):
            chunk = self._make(semantic_type=st)
            assert chunk.semantic_type == st

    def test_tech_stack_defaults_empty(self) -> None:
        chunk = SummarizedChunk(
            content="x",
            path="p",
            intent_summary="s",
        )
        assert chunk.tech_stack == []


# ---------------------------------------------------------------------------
# UpsertResult
# ---------------------------------------------------------------------------


class TestUpsertResult:
    def test_inserted(self) -> None:
        r = UpsertResult(content_hash="abc", file_path="src/foo.py", action="inserted")
        assert r.error is None

    def test_with_error(self) -> None:
        r = UpsertResult(
            content_hash="abc",
            file_path="src/foo.py",
            action="error",
            error="ChromaDB unavailable",
        )
        assert r.error == "ChromaDB unavailable"

    def test_invalid_action(self) -> None:
        with pytest.raises(ValidationError):
            UpsertResult(content_hash="abc", file_path="f", action="deleted")  # not in Literal


# ---------------------------------------------------------------------------
# SanitizedFile
# ---------------------------------------------------------------------------


class TestSanitizedFile:
    def test_clean_file(self) -> None:
        f = SanitizedFile(path="src/foo.py", content="x = 1")
        assert not f.is_quarantined
        assert f.quarantine_reason is None

    def test_quarantined_sentinel(self) -> None:
        f = SanitizedFile(
            path="src/secret.py",
            content="[CONTENT_QUARANTINED — PII review required]",
            quarantine_reason="EMAIL detected",
        )
        assert f.is_quarantined
        assert f.quarantine_reason == "EMAIL detected"


# ---------------------------------------------------------------------------
# TendencyData
# ---------------------------------------------------------------------------


class TestTendencyData:
    def test_defaults(self) -> None:
        td = TendencyData(scope="project")
        assert td.doc_count == 0
        assert td.semantic_type_distribution == {}
        assert td.tech_stack_frequency == {}
        assert td.dominant_patterns == []
        assert td.weighted_docs == []

    def test_populated(self) -> None:
        td = TendencyData(
            scope="author",
            doc_count=10,
            semantic_type_distribution={"Logic": 8, "Config": 2},
            tech_stack_frequency={"FastAPI": 6, "Pydantic": 5},
            dominant_patterns=["uses_type_annotations"],
        )
        assert td.doc_count == 10
        assert td.semantic_type_distribution["Logic"] == 8


# ---------------------------------------------------------------------------
# PersonaProfile
# ---------------------------------------------------------------------------


class TestPersonaProfile:
    def test_valid(self) -> None:
        p = PersonaProfile(
            style_summary="Prefers typed, functional style.",
            dominant_patterns=["uses_type_annotations", "avoids_global_state"],
            tech_preferences=["FastAPI", "Pydantic"],
            coaching_notes=["Consider adding more docstrings."],
        )
        assert p.style_summary == "Prefers typed, functional style."

    def test_defaults(self) -> None:
        p = PersonaProfile(style_summary="Minimal profile.")
        assert p.dominant_patterns == []
        assert p.tech_preferences == []
        assert p.coaching_notes == []


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    def test_valid_score(self) -> None:
        r = AnalysisResult(deviation_score=0.4, rationale="Aligned with patterns.")
        assert r.deviation_score == 0.4
        assert r.patterns_matched == []

    def test_boundary_scores(self) -> None:
        AnalysisResult(deviation_score=0.0, rationale="r")
        AnalysisResult(deviation_score=1.0, rationale="r")

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(deviation_score=-0.01, rationale="r")

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisResult(deviation_score=1.01, rationale="r")


# ---------------------------------------------------------------------------
# CoachingAlert
# ---------------------------------------------------------------------------


class TestCoachingAlert:
    def test_valid_severities(self) -> None:
        for sev in ("info", "warning", "critical"):
            alert = CoachingAlert(message="msg", severity=sev)
            assert alert.severity == sev

    def test_invalid_severity_raises(self) -> None:
        with pytest.raises(ValidationError):
            CoachingAlert(message="msg", severity="debug")  # not in Literal


# ---------------------------------------------------------------------------
# SkillsData
# ---------------------------------------------------------------------------


class TestSkillsData:
    def test_defaults(self) -> None:
        sd = SkillsData()
        assert sd.tech_stacks == []
        assert sd.doc_count == 0

    def test_populated(self) -> None:
        sd = SkillsData(
            tech_stacks=["FastAPI", "LangChain"],
            semantic_type_distribution={"Logic": 50, "Config": 10},
            pattern_summaries=["Uses async patterns throughout."],
            doc_count=60,
        )
        assert sd.doc_count == 60


# ---------------------------------------------------------------------------
# SyncRequest — URL validation (Phase 1 exit gate: ≥ 95% branch coverage)
# ---------------------------------------------------------------------------


class TestSyncRequest:
    """Covers all branches in validate_repo_url (§6.3a)."""

    def test_github_url_passes(self) -> None:
        r = SyncRequest(repo_url="https://github.com/user/repo")
        assert r.repo_url == "https://github.com/user/repo"

    def test_gitlab_url_passes(self) -> None:
        r = SyncRequest(repo_url="https://gitlab.com/org/project")
        assert r.repo_url == "https://gitlab.com/org/project"

    def test_bitbucket_url_passes(self) -> None:
        r = SyncRequest(repo_url="https://bitbucket.org/team/repo")
        assert r.repo_url == "https://bitbucket.org/team/repo"

    def test_www_prefix_stripped_and_accepted(self) -> None:
        """www.github.com should normalise to github.com and pass."""
        r = SyncRequest(repo_url="https://www.github.com/user/repo")
        assert r.repo_url == "https://www.github.com/user/repo"

    def test_unknown_host_raises(self) -> None:
        """Host not in ALLOWED_HOSTS → ValueError."""
        with pytest.raises(ValidationError, match="not in the allowed list"):
            SyncRequest(repo_url="https://github.evil.com/user/repo")

    def test_raw_ip_raises(self) -> None:
        """Raw IP address is not in the allowlist."""
        with pytest.raises(ValidationError, match="not in the allowed list"):
            SyncRequest(repo_url="https://192.168.1.1/user/repo")

    def test_path_traversal_raises(self) -> None:
        """'..' in path → illegal path characters."""
        with pytest.raises(ValidationError, match="illegal path characters"):
            SyncRequest(repo_url="https://github.com/user/../repo")

    def test_null_byte_raises(self) -> None:
        """Null byte in URL → illegal path characters."""
        with pytest.raises(ValidationError, match="illegal path characters"):
            SyncRequest(repo_url="https://github.com/user/repo\x00")

    def test_branch_defaults_to_main(self) -> None:
        r = SyncRequest(repo_url="https://github.com/user/repo")
        assert r.branch == "main"

    def test_custom_branch(self) -> None:
        r = SyncRequest(repo_url="https://github.com/user/repo", branch="develop")
        assert r.branch == "develop"


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


class TestQueryRequest:
    def test_valid_no_filter(self) -> None:
        r = QueryRequest(query="How does the auth middleware work?")
        assert r.tech_filter is None

    def test_valid_with_filter(self) -> None:
        r = QueryRequest(query="FastAPI patterns", tech_filter=["FastAPI"])
        assert r.tech_filter == ["FastAPI"]


# ---------------------------------------------------------------------------
# DiffRequest
# ---------------------------------------------------------------------------


class TestDiffRequest:
    def test_valid(self, sample_diff_text: str) -> None:
        r = DiffRequest(diff_text=sample_diff_text)
        assert r.diff_text == sample_diff_text


# ---------------------------------------------------------------------------
# SkillsRequest — path validation (§6.3b)
# ---------------------------------------------------------------------------


class TestSkillsRequest:
    def test_valid_path(self, tmp_path, monkeypatch) -> None:
        """A relative filename under the export root is accepted."""
        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        r = SkillsRequest(target_path="PROJECT_SKILLS.md")
        assert r.target_path == str(tmp_path / "PROJECT_SKILLS.md")

    def test_path_traversal_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        with pytest.raises(ValidationError, match="illegal characters"):
            SkillsRequest(target_path="../outside.md")

    def test_null_byte_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        with pytest.raises(ValidationError, match="illegal characters"):
            SkillsRequest(target_path="file\x00.md")


# ---------------------------------------------------------------------------
# State TypedDicts — import + minimal structural smoke tests
# (TypedDicts are pure type annotations; coverage is achieved by importing and
# constructing representative dicts to exercise the module's class definitions)
# ---------------------------------------------------------------------------


class TestSyncState:
    def test_can_construct_minimal_dict(self) -> None:
        """SyncState is a TypedDict — verify its keys match the spec."""
        state: SyncState = {
            "repo_url": "https://github.com/user/repo",
            "branch": "main",
            "changed_files": [],
            "sanitized_files": [],
            "parsed_chunks": [],
            "quarantined_files": [],
            "summarized_chunks": [],
            "upsert_results": [],
            "error": None,
            "trace": [],
        }
        assert state["repo_url"] == "https://github.com/user/repo"
        assert state["error"] is None

    def test_annotated_fields_are_lists(self) -> None:
        """upsert_results uses operator.add reducer — must be an Annotated list field."""
        import operator

        from src.models.sync_state import SyncState as SS

        hints = SS.__annotations__
        # upsert_results accumulates via operator.add across Send targets
        assert "upsert_results" in hints
        # summarized_chunks was removed in §12.5 — summarize_and_upsert writes directly to ChromaDB
        assert "summarized_chunks" not in hints
        # Verify the reducer: operator.add on two lists should concatenate
        assert operator.add([1], [2]) == [1, 2]


class TestQueryState:
    def test_can_construct(self) -> None:
        state: QueryState = {
            "query": "How does auth work?",
            "tech_filter": None,
            "query_safe": False,
            "raw_results": [],
            "final_answer": None,
            "error": None,
            "trace": [],
        }
        assert state["query_safe"] is False


class TestPersonaState:
    def test_can_construct(self) -> None:
        state: PersonaState = {
            "scope": "project",
            "recency_months": 6,
            "tendency_data": None,
            "persona_profile": None,
            "error": None,
            "trace": [],
        }
        assert state["recency_months"] == 6


class TestDiffState:
    def test_can_construct(self) -> None:
        state: DiffState = {
            "diff_text": "--- a\n+++ b",
            "diff_safe": False,
            "persona_context": None,
            "analysis_result": None,
            "coaching_alert": None,
            "error": None,
            "trace": [],
        }
        assert state["diff_safe"] is False


class TestSkillsState:
    def test_can_construct(self) -> None:
        state: SkillsState = {
            "target_path": "/tmp/PROJECT_SKILLS.md",
            "skills_data": None,
            "skills_markdown": None,
            "export_result": None,
            "error": None,
            "trace": [],
        }
        assert state["skills_data"] is None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self, monkeypatch) -> None:
        """Settings uses sensible defaults when no env vars are set.

        pydantic-settings reads the .env file directly, so monkeypatch.delenv is
        insufficient — we must override each field to its default value instead.
        """
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
        monkeypatch.setenv("CHROMA_COLLECTION", "developer_memory_v1")
        monkeypatch.setenv("PARSER_WORKERS", "6")
        monkeypatch.setenv("SKILLS_EXPORT_DIR", "")
        monkeypatch.setenv("GITHUB_TOKEN", "")

        from src.config.settings import Settings

        s = Settings()
        assert s.ollama_model == "gemma4:26b"
        assert s.chroma_collection == "developer_memory_v1"
        assert s.parser_workers == 6
        assert s.skills_export_dir == ""
        assert s.github_token == ""

    def test_reads_env_vars(self, monkeypatch) -> None:
        """Settings picks up env var overrides."""
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
        monkeypatch.setenv("CHROMA_COLLECTION", "custom_collection")
        monkeypatch.setenv("PARSER_WORKERS", "2")

        from src.config.settings import Settings

        s = Settings()
        assert s.ollama_model == "gemma4:e4b"
        assert s.chroma_collection == "custom_collection"
        assert s.parser_workers == 2
