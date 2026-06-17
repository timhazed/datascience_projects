"""Developer Memory UI — Developer Persona page renderer.

Owns the Developer Persona page, which synthesizes a developer profile via the
get_dev_persona MCP tool.
"""

import streamlit as st

from ui.http_client import _call_tool, _mcp_error


def page() -> None:
    """Render the Developer Persona page.

    Calls the get_dev_persona MCP tool with user-selected scope and recency window,
    then displays the resulting persona profile as a structured breakdown.
    """
    st.header("Developer Persona")
    st.markdown("Synthesize a stylistic profile from the indexed codebase.")

    col1, col2 = st.columns(2)
    with col1:
        scope = st.selectbox("Scope", ["project", "file", "author"], index=0)
    with col2:
        recency_months = st.slider("Recency window (months)", min_value=1, max_value=24, value=6)

    if st.button("Generate Persona", type="primary"):
        try:
            with st.spinner("Synthesizing persona…"):
                result = _call_tool("get_dev_persona", scope=scope, recency_months=recency_months)
        except Exception as exc:  # noqa: BLE001
            _mcp_error(exc)
            return

        if not isinstance(result, dict):
            st.error("Unexpected response from MCP server.")
            return

        if result.get("error"):
            st.warning(f"Could not generate persona: {result['error']}")
            return

        profile = result.get("persona_profile")
        if profile is None:
            st.info("No persona data available — sync a repository first.")
            return

        st.markdown("### Persona Profile")
        # PersonaProfile may arrive as a dict (JSON-decoded from MCP response)
        profile_data = profile if isinstance(profile, dict) else {"profile": str(profile)}
        for key, value in profile_data.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

        with st.expander("Execution trace"):
            for entry in result.get("trace", []):
                st.text(entry)
