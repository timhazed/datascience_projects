"""Tests for src/utils/constants.py — ensures the module has no imports and correct values."""



class TestConstants:
    def test_max_history_turns_is_positive_int(self):
        from src.utils.constants import MAX_HISTORY_TURNS
        assert isinstance(MAX_HISTORY_TURNS, int)
        assert MAX_HISTORY_TURNS > 0

    def test_constants_module_has_no_project_imports(self):
        """constants.py must be a zero-import file to prevent circular dependencies."""
        import inspect

        import src.utils.constants as mod
        source = inspect.getsource(mod)
        # Should not contain any 'import src.' or 'from src.' statements
        assert "import src." not in source
        assert "from src." not in source
