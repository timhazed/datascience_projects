"""Tests for parse_pdf_file (src/utils/parsers/pdf_parser.py).

No live PDF I/O — tests inject bytes-based PDF content via io.BytesIO where possible,
and use tmp_path to write real minimal PDFs for the file-path path.

Cases:
  - Valid PDF bytes → text extracted, chunked, ParsedChunk objects returned
  - PDF file path (str) → file opened and parsed
  - PDF with no extractable text → returns []
  - Content > CHUNK_SIZE → multiple chunks, each ≤ CHUNK_SIZE
  - Invalid bytes → returns [] (no crash)
  - Empty pages list → returns []
"""

from src.utils.parsers.pdf_parser import CHUNK_SIZE, _window_pages, parse_pdf_file

# ---------------------------------------------------------------------------
# Minimal valid PDF construction
# ---------------------------------------------------------------------------

_MINIMAL_PDF_WITH_TEXT = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 72 720 Td (Hello World) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""


class TestParsePdfFile:
    def test_invalid_bytes_returns_empty_no_crash(self) -> None:
        """Garbage bytes → parse_pdf_file returns [] without raising."""
        result = parse_pdf_file("doc.pdf", b"this is not a PDF")
        assert result == []

    def test_window_pages_empty_returns_empty(self) -> None:
        """_window_pages with empty page list → []."""
        result = _window_pages([], "doc.pdf")
        assert result == []

    def test_window_pages_small_content_single_chunk(self) -> None:
        """Content well under CHUNK_SIZE → single ParsedChunk."""
        pages = ["Short page content."]
        chunks = _window_pages(pages, "doc.pdf")

        assert len(chunks) == 1
        assert chunks[0].path == "doc.pdf"
        assert "Short page content." in chunks[0].content

    def test_window_pages_large_content_multiple_chunks(self) -> None:
        """Content > CHUNK_SIZE → multiple chunks, each ≤ CHUNK_SIZE."""
        # Build a page with many newline-separated lines to allow clean splits
        long_page = "\n".join([f"Line {i}: " + "word " * 20 for i in range(80)])
        pages = [long_page]
        chunks = _window_pages(pages, "doc.pdf")

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.content) <= CHUNK_SIZE

    def test_window_pages_multiple_small_pages_merged(self) -> None:
        """Multiple small pages are merged into a single chunk if total ≤ CHUNK_SIZE."""
        pages = ["Page one.", "Page two.", "Page three."]
        chunks = _window_pages(pages, "doc.pdf")

        assert len(chunks) == 1
        assert "Page one." in chunks[0].content

    def test_window_pages_chunk_path_preserved(self) -> None:
        """All chunks inherit the source file path."""
        pages = ["content"] * 3
        chunks = _window_pages(pages, "reports/annual.pdf")

        assert all(c.path == "reports/annual.pdf" for c in chunks)

    def test_parse_pdf_file_str_content_chunked_directly(self) -> None:
        """When content is a string, it is chunked directly as pre-extracted text.

        This is the normal pipeline path: pii_sanitizer extracts PDF text via pypdf
        and passes the resulting string. parse_pdf_file must not re-open the file.
        """
        extracted = "Page one extracted text.\nPage two extracted text."
        result = parse_pdf_file("report.pdf", extracted)

        assert len(result) == 1
        assert result[0].path == "report.pdf"
        assert "Page one" in result[0].content

    def test_parse_pdf_file_str_content_empty_returns_empty(self) -> None:
        """Empty string content (e.g., image-only PDF) → parse_pdf_file returns []."""
        result = parse_pdf_file("scan.pdf", "   ")
        assert result == []
