import pytest

from src.skills.loader import SkillLoader, SkillLoadError, load_skill


class TestSkillLoader:
    """Tests for SkillLoader class."""

    def test_load_kinesiologist_skill(self):
        """Test loading kinesiologist SKILL.md."""
        loader = SkillLoader()
        content = loader.load("kinesiologist")

        assert "Kinesiologist Specialist" in content
        assert "Version:" in content
        assert "Persona:" in content
        assert "Volume" in content

    def test_load_recovery_skill(self):
        """Test loading recovery SKILL.md."""
        loader = SkillLoader()
        content = loader.load("recovery")

        assert "Recovery" in content
        assert "Mobility" in content
        assert "Sequence" in content

    def test_load_gatekeeper_skill(self):
        """Test loading gatekeeper SKILL.md."""
        loader = SkillLoader()
        content = loader.load("gatekeeper")

        assert "Red Flag" in content
        assert "Triage" in content
        assert "Neurological" in content

    def test_load_intake_skill(self):
        """Test loading intake SKILL.md."""
        loader = SkillLoader()
        content = loader.load("intake")

        assert "Intake" in content
        assert "Operational Logic" in content

    def test_skill_caching(self):
        """Test that skills are cached after first load."""
        loader = SkillLoader()

        # First load
        content1 = loader.load("kinesiologist")
        assert loader.cache_info()["size"] == 1

        # Second load should return cached version
        content2 = loader.load("kinesiologist")

        assert content1 == content2
        # Cache size should still be 1 (same skill)
        assert loader.cache_info()["size"] == 1

    def test_clear_cache(self):
        """Test cache clearing."""
        loader = SkillLoader()

        # Load to populate cache
        loader.load("kinesiologist")
        assert loader.cache_info()["size"] == 1

        # Clear cache
        loader.clear_cache()
        assert loader.cache_info()["size"] == 0

        # Load again
        loader.load("kinesiologist")
        assert loader.cache_info()["size"] == 1

    def test_load_all_skills(self):
        """Test loading all available skills."""
        loader = SkillLoader()
        skills = loader.load_all()

        assert "kinesiologist" in skills
        assert "recovery" in skills
        assert "gatekeeper" in skills
        assert "intake" in skills

    def test_invalid_skill_raises_error(self):
        """Test that loading non-existent skill raises error."""
        loader = SkillLoader()

        with pytest.raises(SkillLoadError) as exc_info:
            loader.load("nonexistent")  # type: ignore

        assert "not found" in str(exc_info.value)


class TestLoadSkillFunction:
    """Tests for the load_skill convenience function."""

    def test_load_skill_returns_content(self):
        """Test that load_skill returns skill content."""
        content = load_skill("kinesiologist")

        assert "Kinesiologist" in content
        assert len(content) > 100

    def test_load_skill_all_types(self):
        """Test loading all skill types via convenience function."""
        skill_types = ["kinesiologist", "recovery", "gatekeeper", "intake"]

        for skill_type in skill_types:
            content = load_skill(skill_type)  # type: ignore
            assert len(content) > 0
            # Each skill has Version or (gatekeeper) Red Flag as structure marker
            assert (
                "Version:" in content
                or "version" in content.lower()
                or "Red Flag" in content
            )


class TestSkillContentIntegrity:
    """Tests to verify SKILL.md content matches specification."""

    def test_kinesiologist_has_required_sections(self):
        """Test kinesiologist SKILL.md has all required sections."""
        content = load_skill("kinesiologist")

        assert "Version:" in content
        assert "Persona:" in content
        assert "Operational Logic" in content
        assert "SAID" in content or "Volume" in content

    def test_recovery_has_required_sections(self):
        """Test recovery SKILL.md has all required sections."""
        content = load_skill("recovery")

        assert "Version:" in content
        assert "Persona:" in content
        assert "Operational Logic" in content
        assert "Sequence" in content or "SMR" in content

    def test_gatekeeper_has_required_sections(self):
        """Test gatekeeper SKILL.md has all required sections."""
        content = load_skill("gatekeeper")

        assert "Red Flag" in content
        assert "Triage" in content
        assert "Neurological" in content
        assert "Cardiovascular" in content

    def test_gatekeeper_has_red_flag_categories(self):
        """Test gatekeeper SKILL.md has red flag categories."""
        content = load_skill("gatekeeper")

        assert "Neurological" in content
        assert "Cardiovascular" in content
