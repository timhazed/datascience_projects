import contextlib
from pathlib import Path
from typing import Literal

# Skill types that can be loaded
SkillType = Literal["kinesiologist", "recovery", "gatekeeper", "intake"]

# Base path for skills directory (relative to project root)
SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"


class SkillLoadError(Exception):
    """Raised when a skill file cannot be loaded."""

    pass


class SkillLoader:
    """
    Loads SKILL.md files for agents at runtime.

    Skills are cached after first load to avoid repeated file I/O.
    """

    def __init__(self, skills_dir: Path | None = None):
        """
        Initialize the skill loader.

        Args:
            skills_dir: Optional custom path to skills directory.
                       Defaults to project's skills/ directory.
        """
        self.skills_dir = skills_dir or SKILLS_DIR
        self._instance_cache: dict[str, str] = {}

    def load(self, skill_type: SkillType) -> str:
        """
        Load a SKILL.md file by skill type.

        Args:
            skill_type: The type of skill to load.

        Returns:
            The contents of the SKILL.md file.

        Raises:
            SkillLoadError: If the skill file cannot be found or read.
        """
        # Check instance cache first
        if skill_type in self._instance_cache:
            return self._instance_cache[skill_type]

        skill_path = self.skills_dir / skill_type / "SKILL.md"

        if not skill_path.exists():
            raise SkillLoadError(
                f"Skill file not found: {skill_path}. "
                f"Ensure skills/{skill_type}/SKILL.md exists."
            )

        try:
            content = skill_path.read_text(encoding="utf-8")
            self._instance_cache[skill_type] = content
            return content
        except Exception as e:
            raise SkillLoadError(f"Failed to read skill file {skill_path}: {e}") from e

    def load_all(self) -> dict[SkillType, str]:
        """
        Load all available SKILL.md files.

        Returns:
            Dictionary mapping skill type to skill content.
        """
        skills: dict[SkillType, str] = {}
        skill_types: list[SkillType] = ["kinesiologist", "recovery", "gatekeeper", "intake"]

        for skill_type in skill_types:
            with contextlib.suppress(SkillLoadError):
                skills[skill_type] = self.load(skill_type)

        return skills

    def clear_cache(self) -> None:
        """Clear the skill cache to force reload on next access."""
        self._instance_cache.clear()

    def cache_info(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "size": len(self._instance_cache),
            "keys": list(self._instance_cache.keys()),
        }


# Module-level loader instance for convenience
_default_loader = SkillLoader()


def load_skill(skill_type: SkillType) -> str:
    """
    Convenience function to load a skill using the default loader.

    Args:
        skill_type: The type of skill to load.

    Returns:
        The contents of the SKILL.md file.
    """
    return _default_loader.load(skill_type)


