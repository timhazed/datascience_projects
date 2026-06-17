"""Developer Memory UI — Analyze Diff page renderer.

Owns the Analyze Diff page, which sends a unified diff to the analyze_diff MCP tool
and surfaces any coaching alerts from the developer persona comparison.
"""

import streamlit as st

from ui.http_client import _call_tool, _mcp_error


def page() -> None:
    """Render the Analyze Diff page.

    Accepts a unified diff as text input, calls the analyze_diff MCP tool, and
    renders coaching feedback. An alert is surfaced when deviation score exceeds 0.65.
    """
    st.header("Analyze Diff")
    st.markdown(
        "Paste a unified diff to receive coaching feedback relative to the developer persona. "
        "An alert is raised when deviation score exceeds **0.65**."
    )

    with st.form("diff_form"):
        diff_text = st.text_area(
            "Unified Diff",
            placeholder="--- a/src/main.py\n+++ b/src/main.py\n@@ -1,3 +1,4 @@\n...",
            height=200,
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        if not diff_text.strip():
            st.error("Diff text cannot be empty.")
            return

        try:
            with st.spinner("Analyzing diff…"):
                result = _call_tool("analyze_diff", diff_text=diff_text.strip())
        except Exception as exc:  # noqa: BLE001
            _mcp_error(exc)
            return

        if not isinstance(result, dict):
            st.error("Unexpected response from MCP server.")
            return

        if result.get("error"):
            st.error(f"Analysis failed: {result['error']}")

        # Coaching alert — surfaced prominently when deviation > 0.65
        alert = result.get("coaching_alert")
        if alert is not None:
            alert_data = alert if isinstance(alert, dict) else {}
            alert_msg = alert_data.get("message", str(alert))
            severity = alert_data.get("severity", "info")
            _severity_widget = {"info": st.info, "warning": st.warning, "critical": st.error}
            _severity_widget.get(severity, st.info)(f"Coaching Alert ({severity}): {alert_msg}")
        else:
            st.success("No deviation detected — code aligns with historical patterns.")

        # Analysis detail
        analysis = result.get("analysis_result")
        if analysis is not None:
            st.markdown("### Analysis Detail")
            analysis_data = analysis if isinstance(analysis, dict) else {}
            for key, value in analysis_data.items():
                st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

        with st.expander("Execution trace"):
            for entry in result.get("trace", []):
                st.text(entry)
