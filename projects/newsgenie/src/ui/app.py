"""NewsGenie Streamlit application."""

import uuid

import streamlit as st

from src.data.enums import AgentName, NewsCategory
from src.data.response import SupervisorResponse
from src.data.state import AgentState
from src.data.user_query import UserQuery
from src.graph.builder import build_graph
from src.ui.article_card import render_article_card
from src.utils.config import settings
from src.utils.llm_factory import LLMFactory

# Human-readable backend labels for source badge (#5).
BACKEND_LABEL: dict[AgentName, str] = {
    AgentName.BUSINESS: "NewsAPI · Business",
    AgentName.SPORTS: "Guardian · Sports",
    AgentName.GENERAL: "NewsAPI · World",
    AgentName.WEB_SEARCH: "Web Search",
}

# Fixed category chip definitions.
# Headlines group: backed by wire news APIs.
HEADLINE_CHIPS: list[tuple[str, NewsCategory]] = [
    ("Business", NewsCategory.BUSINESS),
    ("World", NewsCategory.GENERAL),
    ("Sports", NewsCategory.SPORTS),
]
# Topics group: satisfied via web search only.
TOPIC_CHIPS: list[tuple[str, NewsCategory]] = [
    ("Cooking", NewsCategory.WEB),
    ("Fashion", NewsCategory.WEB),
]

# Stable label map used by _compose_query — avoids dict-reversal and enum str() edge cases.
HEADLINE_CHIP_LABELS: dict[NewsCategory, str] = {
    NewsCategory.BUSINESS: "Business",
    NewsCategory.GENERAL: "World",
    NewsCategory.SPORTS: "Sports",
}


@st.cache_resource
def get_graph():
    """Build and cache the compiled LangGraph graph for the app lifetime."""
    return build_graph()


def _init_session() -> None:
    """Initialise session state on first load."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_chip" not in st.session_state:
        st.session_state.selected_chip = None


def _build_history(messages: list[dict]) -> list[dict]:
    """Convert st.session_state.messages to AgentState conversation_history format.

    User turns produce {"role": "user", "content": message["content"]}.
    Assistant turns produce {"role": "assistant", "content": "[Covered: <queries>]"}
    where queries are the sub-query strings from the SupervisorResponse sections.
    """
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append({"role": "user", "content": msg["content"]})
        else:
            response: SupervisorResponse = msg["response"]
            covered = ", ".join(s.query for s in response.sections)
            history.append({"role": "assistant", "content": f"[Covered: {covered}]"})
    return history


def _render_advanced() -> None:
    """Render display-only active config in a collapsed expander on the main page."""
    with st.expander("Advanced", expanded=False):
        st.caption("Active config (set via .env)")
        providers = [p.value for p in LLMFactory.supported_providers()]
        st.selectbox(
            "LLM Provider",
            options=providers,
            index=providers.index(settings.llm_provider.value)
            if settings.llm_provider.value in providers
            else 0,
            disabled=True,
            key="sidebar_provider",
        )
        st.text_input(
            "Model",
            value=settings.supervisor_model,
            disabled=True,
            key="sidebar_model",
        )


def _render_category_chips() -> None:
    """Render grouped category chip selectors."""
    st.markdown("#### Headlines")
    st.caption("From our news feeds")
    cols = st.columns(len(HEADLINE_CHIPS))
    for col, (label, category) in zip(cols, HEADLINE_CHIPS, strict=False):
        with col:
            selected = st.session_state.selected_chip == category
            if label == "World":
                help_text = "Major global and political headlines."  # #3
            else:
                help_text = None
            if st.button(
                f"{'✓ ' if selected else ''}{label}",
                key=f"chip_{label}",
                use_container_width=True,
                help=help_text,
            ):
                st.session_state.selected_chip = None if selected else category

    st.markdown("#### Topics")
    st.caption("From web search")
    cols = st.columns(len(TOPIC_CHIPS))
    for col, (label, _category) in zip(cols, TOPIC_CHIPS, strict=False):
        with col:
            selected = st.session_state.selected_chip == label
            if st.button(
                f"{'✓ ' if selected else ''}{label}",
                key=f"chip_{label}",
                use_container_width=True,
            ):
                st.session_state.selected_chip = None if selected else label


def _compose_query(user_input: str) -> tuple[str, list[NewsCategory]]:
    """
    Compose UserQuery.text and categories from chip selection + chat input.

    Rules:
    - Chip only → "Show me the latest news on {label}"
    - Chip + text → "{text} ({label})"
    - Text only → text as-is; no category restriction
    - Topic chips → web search; NewsCategory.WEB passed as category
    """
    chip = st.session_state.selected_chip
    categories: list[NewsCategory] = []

    if chip is None:
        return user_input, categories

    # Topic chip (stored as plain string label, not NewsCategory enum)
    # NewsCategory is (str, Enum) so must check for NewsCategory first.
    if not isinstance(chip, NewsCategory):
        label = chip
        categories = [NewsCategory.WEB]
        if user_input:
            text = f"{user_input} ({label})"
        else:
            text = f"Show me the latest news on {label}"
        return text, categories

    # Headline chip (stored as NewsCategory enum)
    categories = [chip]
    label = HEADLINE_CHIP_LABELS.get(chip, chip.value.title())
    if user_input:
        text = f"{user_input} ({label})"
    else:
        text = f"Show me the latest news on {label}"
    return text, categories


def _render_response(response: SupervisorResponse) -> None:
    """Render a SupervisorResponse inside a chat assistant bubble."""
    with st.chat_message("assistant"):
        if response.fallback_used:
            st.warning("One or more news sources returned no results.")

        for i, section in enumerate(response.sections):
            if i > 0:
                st.divider()

            st.markdown(f"**{section.query}**")
            # Source badge — backend name + article count (#5)
            badge = BACKEND_LABEL.get(section.agent, section.agent.value)
            st.caption(f"{badge} · {len(section.articles)} article{'s' if len(section.articles) != 1 else ''}")

            if not section.articles:
                st.caption("No results found for this query.")
                continue

            with st.expander(f"Sources ({len(section.articles)})", expanded=False):
                for article in section.articles:
                    render_article_card(article)


def main() -> None:
    st.set_page_config(
        page_title="NewsGenie",
        page_icon="📰",
        layout="wide",
    )

    _init_session()

    st.title("NewsGenie")
    st.caption("Real-time news across Business, World, and Sports — plus open web search.")
    _render_advanced()

    # Replay conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # Re-render stored response object
                _render_response(message["response"])

    _render_category_chips()

    get_news_clicked = False
    if st.session_state.selected_chip is not None:
        get_news_clicked = st.button(
            "Get News",
            key="get_news_btn",
            use_container_width=False,
        )

    user_input = st.chat_input("Ask about any news topic…", max_chars=1000) or ""

    chip = st.session_state.selected_chip

    if user_input or get_news_clicked or (chip is not None and not user_input):
        composed_text, categories = _compose_query(user_input)

        try:
            query = UserQuery(text=composed_text, session_id=st.session_state.session_id)
        except Exception as exc:
            st.error(f"Invalid query: {exc}")
            return

        # Show user turn
        with st.chat_message("user"):
            st.markdown(composed_text)
        st.session_state.messages.append({"role": "user", "content": composed_text})

        # Run graph
        state = AgentState(
            query=query,
            conversation_history=_build_history(st.session_state.messages),
        )
        with st.spinner("Fetching news…"):
            try:
                output = get_graph().invoke(state)
            except Exception as exc:
                st.error(f"I encountered an issue processing your request. {exc}")
                return

        response: SupervisorResponse = output["final_response"]
        _render_response(response)
        st.session_state.messages.append({"role": "assistant", "response": response})

        # Clear chip after submit
        st.session_state.selected_chip = None


if __name__ == "__main__":
    main()
