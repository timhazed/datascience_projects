"""Tests for src/data/enums.py."""


from src.data.enums import AgentName, LLMProvider, NewsCategory


class TestNewsCategory:
    def test_values_are_strings(self):
        assert NewsCategory.BUSINESS == "business"
        assert NewsCategory.SPORTS == "sports"
        assert NewsCategory.GENERAL == "general"
        assert NewsCategory.WEB == "web"

    def test_all_four_categories_exist(self):
        assert len(NewsCategory) == 4

    def test_str_enum_behaves_as_string(self):
        assert isinstance(NewsCategory.BUSINESS, str)


class TestLLMProvider:
    def test_values(self):
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.GROQ == "groq"

    def test_both_providers_exist(self):
        assert len(LLMProvider) == 2


class TestAgentName:
    def test_values(self):
        assert AgentName.BUSINESS == "business_agent"
        assert AgentName.SPORTS == "sports_agent"
        assert AgentName.GENERAL == "general_agent"
        assert AgentName.WEB_SEARCH == "web_search_agent"

    def test_all_four_agents_exist(self):
        assert len(AgentName) == 4

    def test_str_enum_behaves_as_string(self):
        assert isinstance(AgentName.BUSINESS, str)
