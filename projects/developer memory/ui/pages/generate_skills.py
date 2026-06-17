"""Developer Memory UI — Generate Skills Package page renderer.

Owns the Generate Skills Package page, which exports PROJECT_SKILLS.md via the
generate_skills_pkg MCP tool and offers the result as a download.
"""

import streamlit as st

from ui.http_client import _call_tool, _mcp_error


def page() -> None:
    """Render the Generate Skills Package page.

    Calls the generate_skills_pkg MCP tool with a user-supplied output path and
    surfaces the export result with a download button for the generated markdown.
    """
    st.header("Generate Skills Package")
    st.markdown(
        "Export a **PROJECT_SKILLS.md** summarizing this repository's Professional DNA. "
        "The file is written to the target directory on the MCP server and made available for download."
    )

    with st.form("skills_form"):
        target_dir = st.text_input(
            "Output Directory",
            value="PROJECT_SKILLS.md",
            help="Path relative to SKILLS_EXPORT_DIR on the MCP server.",
        )
        submitted = st.form_submit_button("Generate & Export", type="primary")

    if submitted:
        try:
            with st.spinner("Generating skills package…"):
                result = _call_tool("generate_skills_pkg", target_path=target_dir)
        except Exception as exc:  # noqa: BLE001
            _mcp_error(exc)
            return

        if not isinstance(result, dict):
            st.error("Unexpected response from MCP server.")
            return

        if result.get("error"):
            st.error(f"Export failed: {result['error']}")
            return

        export = result.get("export_result")
        if export and isinstance(export, dict):
            written_path = export.get("path", "")
            bytes_written = export.get("bytes_written", 0)
            st.success(f"Exported {bytes_written:,} bytes → `{written_path}`")

            # Offer markdown content for download if returned in the result
            markdown_content = result.get("skills_markdown", "")
            if markdown_content:
                st.download_button(
                    label="⬇ Download PROJECT_SKILLS.md",
                    data=markdown_content,
                    file_name="PROJECT_SKILLS.md",
                    mime="text/markdown",
                )
        else:
            st.info("No skills data available — sync a repository first.")

        with st.expander("Execution trace"):
            for entry in result.get("trace", []):
                st.text(entry)
