"""file_exporter LangGraph node — Spec §2.5, §6.3b, §7 (FR-SKILLS-01).

Writes PROJECT_SKILLS.md to the validated target_path. The path was already validated
by SkillsRequest.validate_path (§6.3b path guard) before pipeline invocation — this
node trusts the validated path and focuses on the write operation.

Write errors (permissions, disk full) are non-fatal: error is surfaced to the user,
export_result is None. No stack traces or filesystem internals are exposed.

Optional sha_store injection: when provided, calls set_skills_sha after a successful
write so future skills_cache_guard invocations can short-circuit synthesis.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from src.db.sha_store import SHAStore

logger = logging.getLogger(__name__)

_OUTPUT_FILENAME = "PROJECT_SKILLS.md"


def make_file_exporter_node(sha_store: SHAStore | None = None) -> Callable[[dict], dict]:
    """Return a file_exporter node with optional SHAStore injection.

    Args:
        sha_store: When provided, records the skills SHA after a successful write so
            future calls to skills_cache_guard can skip re-synthesis.

    Returns:
        LangGraph node function that reads state["skills_markdown"] and state["target_path"]
        and writes PROJECT_SKILLS.md to target_path. Returns export_result dict on success.
    """

    def file_exporter(state: dict) -> dict:
        """Write skills_markdown to target_path/PROJECT_SKILLS.md.

        Creates the target directory if it does not exist. Writes UTF-8 encoded Markdown.
        Calls sha_store.set_skills_sha after a successful write when sha_store is injected.
        Returns export_result with path and bytes_written on success, None on failure.
        """
        skills_markdown: str | None = state.get("skills_markdown")
        target_path: str = state.get("target_path", "")
        trace: list[str] = list(state.get("trace", []))

        if not skills_markdown:
            return {
                "export_result": None,
                "error": "No skills Markdown to export.",
                "trace": trace + ["file_exporter: skipped (no markdown)"],
            }

        # Resolve write destination — target_path may be a directory or a full file path
        resolved = Path(target_path)
        if resolved.suffix.lower() != ".md":
            # target_path is a directory; append the standard filename
            resolved = resolved / _OUTPUT_FILENAME

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            encoded = skills_markdown.encode("utf-8")
            resolved.write_bytes(encoded)
        except Exception as exc:
            logger.error(
                "file_exporter failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "export_result": None,
                "error": "Cannot write to target path.",
                "trace": trace + ["file_exporter: error"],
            }

        bytes_written = len(encoded)  # reuse bytes already computed for write()
        logger.info("file_exporter: wrote %d bytes to %s", bytes_written, resolved)

        # Record the skills SHA so future guard invocations can skip synthesis
        if sha_store is not None:
            repo_url: str = state.get("repo_url", "")
            branch: str = state.get("branch", "")
            synced_sha: str = state.get("synced_sha", "")
            if repo_url and branch and synced_sha:
                try:
                    sha_store.set_skills_sha(repo_url, branch, synced_sha, str(resolved))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "file_exporter: set_skills_sha failed [%s]: %s",
                        type(exc).__name__,
                        str(exc)[:200],
                    )

        return {
            "export_result": {"path": str(resolved), "bytes_written": bytes_written},
            "trace": trace + [f"file_exporter: wrote {bytes_written}B to {resolved}"],
        }

    return file_exporter
