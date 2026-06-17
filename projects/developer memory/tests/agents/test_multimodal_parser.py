"""Tests for make_multimodal_parser_node and parser helpers.

Phase 4 exit gate — no filesystem access needed; content passed directly via SanitizedFile.

Cases:
  - .py file → python_parser called; chunks produced
  - .md file → markdown_parser called; chunks produced
  - unknown extension (.ts, .yaml) → plain text chunks produced
  - empty content → 0 chunks (no crash)
  - multi-file state → chunks from all files accumulated (SINGLE-WRITER)
  - empty sanitized_files → parsed_chunks is []
  - large file (>CHUNK_SIZE) → multiple chunks produced; all chunks ≤ CHUNK_SIZE
  - chunk content: no leading/trailing whitespace
  - trace list gains one entry per file
"""


from src.agents.multimodal_parser import make_multimodal_parser_node
from src.models.sanitized_file import SanitizedFile
from src.models.trace_event import TraceEventType
from src.utils.parsers.python_parser import CHUNK_SIZE as PY_CHUNK_SIZE


def _state(sanitized_files: list[SanitizedFile]) -> dict:
    return {
        "sanitized_files": sanitized_files,
        "quarantined_files": [],
        "parsed_chunks": [],
        "summarized_chunks": [],
        "upsert_results": [],
        "error": None,
        "trace": [],
    }


def _sf(path: str, content: str) -> SanitizedFile:
    return SanitizedFile(path=path, content=content)


class TestMultimodalParserNode:
    def test_python_file_produces_chunks(self) -> None:
        """.py file → at least 1 ParsedChunk with correct path."""
        node = make_multimodal_parser_node()
        result = node(_state([_sf("src/main.py", "def hello():\n    return 'world'\n")]))

        assert len(result["parsed_chunks"]) >= 1
        assert all(c.path == "src/main.py" for c in result["parsed_chunks"])

    def test_markdown_file_produces_chunks(self) -> None:
        """.md file → at least 1 ParsedChunk."""
        node = make_multimodal_parser_node()
        result = node(_state([_sf("README.md", "# Title\n\nSome content here.\n")]))

        assert len(result["parsed_chunks"]) >= 1
        assert result["parsed_chunks"][0].path == "README.md"

    def test_unknown_extension_treated_as_plain_text(self) -> None:
        """.ts file → plain text chunking; at least 1 chunk produced."""
        node = make_multimodal_parser_node()
        result = node(_state([_sf("src/app.ts", "const x: number = 1;\nexport default x;\n")]))

        assert len(result["parsed_chunks"]) >= 1
        assert result["parsed_chunks"][0].path == "src/app.ts"

    def test_empty_content_produces_no_chunks(self) -> None:
        """File with blank content → parsed_chunks is [] (no crash)."""
        node = make_multimodal_parser_node()
        result = node(_state([_sf("src/empty.py", "   \n  ")]))

        assert result["parsed_chunks"] == []

    def test_multi_file_chunks_accumulated(self) -> None:
        """Chunks from all sanitized_files are accumulated in parsed_chunks (SINGLE-WRITER)."""
        node = make_multimodal_parser_node()
        files = [
            _sf("src/a.py", "def a(): pass"),
            _sf("src/b.md", "# Section\ncontent"),
            _sf("src/c.ts", "export const c = 3;"),
        ]
        result = node(_state(files))

        assert len(result["parsed_chunks"]) >= 3

    def test_empty_sanitized_files_returns_empty_parsed_chunks(self) -> None:
        """No sanitized files → parsed_chunks is []."""
        node = make_multimodal_parser_node()
        result = node(_state([]))

        assert result["parsed_chunks"] == []

    def test_large_file_split_into_multiple_chunks(self) -> None:
        """File content > CHUNK_SIZE → multiple chunks; each ≤ CHUNK_SIZE."""
        node = make_multimodal_parser_node()
        large_content = "x = 1\n" * 400  # ~2400 chars > 1500
        result = node(_state([_sf("src/large.py", large_content)]))

        assert len(result["parsed_chunks"]) >= 2
        for chunk in result["parsed_chunks"]:
            assert len(chunk.content) <= PY_CHUNK_SIZE, (
                f"chunk too large: {len(chunk.content)} chars"
            )

    def test_chunk_content_has_no_extra_whitespace(self) -> None:
        """Chunk content must not have leading or trailing whitespace."""
        node = make_multimodal_parser_node()
        result = node(_state([_sf("src/x.py", "  def f():\n    pass\n  ")]))

        for chunk in result["parsed_chunks"]:
            assert chunk.content == chunk.content.strip(), (
                f"chunk has extraneous whitespace: {chunk.content!r}"
            )

    def test_pdf_file_routed_to_pdf_parser(self) -> None:
        """.pdf file with pre-extracted text → ParsedChunk produced via parse_pdf_file."""
        node = make_multimodal_parser_node()
        result = node(_state([_sf("reports/q1.pdf", "Revenue increased 15% year over year.")]))

        assert len(result["parsed_chunks"]) == 1
        assert result["parsed_chunks"][0].path == "reports/q1.pdf"

    def test_trace_gains_one_entry_per_file(self) -> None:
        """Trace list gains one entry per sanitized file processed."""
        node = make_multimodal_parser_node()
        files = [_sf(f"src/{i}.py", f"x = {i}") for i in range(4)]
        result = node(_state(files))

        assert len(result["trace"]) == 4


class TestPythonParser:
    """Focused tests on python_parser internals via the node."""

    def test_definition_boundaries_produce_separate_chunks(self) -> None:
        """Top-level class/def boundaries produce separate chunks when content is large."""
        from src.utils.parsers.python_parser import parse_python_file

        big_def = "def big_function():\n" + ("    x = 1\n" * 300)  # ~1800 chars per def
        content = big_def + big_def  # two identical big functions

        chunks = parse_python_file("src/x.py", content)
        assert len(chunks) >= 2

    def test_first_commit_file_with_no_parents_chunked(self) -> None:
        """A typical small Python file produces exactly 1 chunk."""
        from src.utils.parsers.python_parser import parse_python_file

        content = "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"
        chunks = parse_python_file("src/greet.py", content)

        assert len(chunks) == 1
        assert chunks[0].path == "src/greet.py"
        assert "greet" in chunks[0].content


class TestMarkdownParser:
    """Focused tests on markdown_parser internals."""

    def test_headers_create_section_boundaries(self) -> None:
        """H1/H2 headers create section splits for large content."""
        from src.utils.parsers.markdown_parser import parse_markdown_file

        section = "word " * 400  # ~2000 chars per section
        content = f"# Section 1\n{section}\n\n# Section 2\n{section}"
        chunks = parse_markdown_file("README.md", content)

        assert len(chunks) >= 2

    def test_small_document_single_chunk(self) -> None:
        """A short document fits in one chunk."""
        from src.utils.parsers.markdown_parser import parse_markdown_file

        content = "# Title\n\nShort description.\n"
        chunks = parse_markdown_file("doc.md", content)

        assert len(chunks) == 1


class TestMultimodalParserTraceEvents:
    """Assert trace_events are emitted on every code path."""

    def test_empty_files_emits_ok_with_zero_count(self) -> None:
        """No sanitized files → TraceEventType.OK with count=0."""
        node = make_multimodal_parser_node()
        result = node(_state([]))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].node == "multimodal_parser"
        assert events[0].event == TraceEventType.OK
        assert events[0].count == 0

    def test_one_file_emits_ok_with_chunk_count(self) -> None:
        """One sanitized file → TraceEventType.OK with count matching parsed chunks."""
        node = make_multimodal_parser_node()
        sf = SanitizedFile(path="src/main.py", content="def foo(): pass\n")
        result = node(_state([sf]))
        events = result["trace_events"]
        assert events[0].event == TraceEventType.OK
        assert events[0].count == len(result["parsed_chunks"])
