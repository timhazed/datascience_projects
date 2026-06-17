"""Developer Memory UI — Query Memory page renderer.

Owns the Query Memory page, which performs semantic search via the query_memory MCP tool.
"""

import streamlit as st

from ui.http_client import _call_tool, _mcp_error


def page() -> None:
    """Render the Query Memory page.

    Presents a search form and calls the query_memory MCP tool with the user's
    query and optional tech-stack filter. Results are rendered as markdown.
    """
    st.header("Query Memory")
    st.markdown("Ask a natural-language question about your indexed codebase.")

    with st.form("query_form"):
        query = st.text_area(
            "Query",
            placeholder="Where have I used dependency injection?",
            height=80,
        )
        tech_filter_raw = st.text_input(
            "Tech Filter (optional)",
            placeholder="FastAPI, Pydantic  (comma-separated)",
            help="Narrow results to specific tech stack labels.",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        if not query.strip():
            st.error("Query cannot be empty.")
            return

        # Parse optional tech filter — empty list treated as None (no filter)
        tech_filter = (
            [t.strip() for t in tech_filter_raw.split(",") if t.strip()]
            if tech_filter_raw.strip()
            else None
        )

        try:
            with st.spinner("Searching…"):
                answer = _call_tool("query_memory", query=query.strip(), tech_filter=tech_filter)
        except Exception as exc:  # noqa: BLE001
            _mcp_error(exc)
            return

        st.markdown("### Result")
        st.markdown(str(answer))
