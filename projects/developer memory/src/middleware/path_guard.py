"""Write-path traversal guard for the skills export pipeline.

Spec §6.3b — generate_skills_pkg accepts a caller-supplied target_path. Without
validation a compromised MCP client could write to arbitrary filesystem locations.
"""

import os
from pathlib import Path


def validate_target_path(path: str) -> str:
    """Validate a write target path before passing to file_exporter.

    Rules:
      - Must not contain '..' (path traversal)
      - Must not contain null bytes (encoding attack)
      - Must resolve to a location under SKILLS_EXPORT_DIR (env var, default: cwd)

    Uses Path.is_relative_to() (Python 3.9+) for correct path-component-aware containment
    checking. str.startswith() is insufficient — /tmp/export_evil starts with /tmp/export
    as a string but is NOT under that directory.

    Args:
        path: The raw target path provided by the MCP client.

    Returns:
        The fully-resolved absolute path string if valid.

    Raises:
        ValueError: With a safe, non-leaking message on any violation.
    """
    if ".." in path or "\x00" in path:
        raise ValueError("target_path contains illegal characters")

    export_root = Path(os.environ.get("SKILLS_EXPORT_DIR", str(Path.cwd()))).resolve()
    resolved = (export_root / path).resolve()

    # is_relative_to() is path-component-aware: /tmp/export_evil is NOT relative to /tmp/export
    if not resolved.is_relative_to(export_root):
        raise ValueError("target_path resolves outside the allowed export directory")

    return str(resolved)
