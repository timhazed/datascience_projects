"""Tests for src/utils/provider_builder.py — LLMProviderBuilder ABC."""

import pytest

from src.utils.provider_builder import LLMProviderBuilder


class TestLLMProviderBuilder:
    def test_cannot_instantiate_abc_directly(self):
        """LLMProviderBuilder is abstract and must not be instantiatable directly."""
        with pytest.raises(TypeError):
            LLMProviderBuilder()  # type: ignore[abstract]

    def test_concrete_subclass_without_build_is_abstract(self):
        """A subclass that doesn't implement build() is also abstract."""
        class IncompleteBuilder(LLMProviderBuilder):
            pass

        with pytest.raises(TypeError):
            IncompleteBuilder()  # type: ignore[abstract]

    def test_concrete_subclass_with_build_instantiates(self):
        """A subclass that implements build() can be instantiated."""
        class ConcreteBuilder(LLMProviderBuilder):
            def build(self, config):
                return None

        builder = ConcreteBuilder()
        assert builder is not None
