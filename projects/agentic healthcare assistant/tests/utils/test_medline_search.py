"""Tests for medline_search — urllib.request.urlopen mocked; no live NCBI calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.utils.medline_search import _SOURCE_DOMAIN, search_medline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ESEARCH_ONE_PMID = json.dumps({"esearchresult": {"idlist": ["12345"]}}).encode()
_ESEARCH_ZERO_PMIDS = json.dumps({"esearchresult": {"idlist": []}}).encode()

_EFETCH_XML_WITH_ABSTRACT = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>CKD management guidelines</ArticleTitle>
        <Abstract>
          <AbstractText>ACE inhibitors reduce proteinuria in CKD patients.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

_EFETCH_XML_NO_ABSTRACT = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999</PMID>
      <Article>
        <ArticleTitle>A review with no abstract</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

_EFETCH_XML_STRUCTURED = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111</PMID>
      <Article>
        <ArticleTitle>Hypertension treatment</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Hypertension affects millions.</AbstractText>
          <AbstractText Label="METHODS">RCT of telmisartan vs placebo.</AbstractText>
          <AbstractText Label="RESULTS">Telmisartan reduced BP by 8mmHg.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _mock_urlopen(esearch_bytes: bytes, efetch_bytes: bytes):
    """Return a context manager mock that yields esearch then efetch responses."""
    calls = [esearch_bytes, efetch_bytes]
    call_iter = iter(calls)

    def _side_effect(url, timeout):
        data = next(call_iter)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = data
        return mock_resp

    return _side_effect


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_returns_search_result_items() -> None:
    """Happy path: returns SearchResultItem list with correct fields."""
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(
        _ESEARCH_ONE_PMID, _EFETCH_XML_WITH_ABSTRACT
    )):
        results = search_medline("CKD treatment", max_results=1)

    assert len(results) == 1
    assert results[0].source_domain == _SOURCE_DOMAIN
    assert results[0].url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


def test_snippet_contains_title_and_abstract() -> None:
    """Snippet is 'Title — Abstract' when abstract is present."""
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(
        _ESEARCH_ONE_PMID, _EFETCH_XML_WITH_ABSTRACT
    )):
        results = search_medline("CKD treatment")

    assert "CKD management guidelines" in results[0].snippet
    assert "ACE inhibitors" in results[0].snippet


def test_title_is_extracted_from_snippet() -> None:
    """Title field is the part before ' — ' in the snippet."""
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(
        _ESEARCH_ONE_PMID, _EFETCH_XML_WITH_ABSTRACT
    )):
        results = search_medline("CKD")

    assert results[0].title == "CKD management guidelines"


def test_structured_abstract_concatenated() -> None:
    """Multiple AbstractText elements (structured abstract) are joined."""
    esearch = json.dumps({"esearchresult": {"idlist": ["11111"]}}).encode()
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(
        esearch, _EFETCH_XML_STRUCTURED
    )):
        results = search_medline("hypertension")

    assert len(results) == 1
    snippet = results[0].snippet
    # All three label sections should be present
    assert "Hypertension affects millions" in snippet
    assert "telmisartan vs placebo" in snippet
    assert "reduced BP" in snippet


def test_no_abstract_uses_title_only() -> None:
    """When efetch has no AbstractText, snippet is just the title."""
    esearch = json.dumps({"esearchresult": {"idlist": ["99999"]}}).encode()
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(
        esearch, _EFETCH_XML_NO_ABSTRACT
    )):
        results = search_medline("review")

    assert len(results) == 1
    assert results[0].snippet == "A review with no abstract"
    assert results[0].title == "A review with no abstract"


# ---------------------------------------------------------------------------
# Zero-results paths
# ---------------------------------------------------------------------------


def test_zero_pmids_returns_empty_list() -> None:
    """esearch returning 0 PMIDs → empty list, no efetch call made."""
    call_count = 0

    def _side_effect(url, timeout):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = _ESEARCH_ZERO_PMIDS
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        results = search_medline("obscure query")

    assert results == []
    assert call_count == 1  # only esearch called, not efetch


# ---------------------------------------------------------------------------
# Error / degradation paths
# ---------------------------------------------------------------------------


def test_esearch_network_error_returns_empty_list() -> None:
    """Network error on esearch → empty list, no exception propagated."""
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        results = search_medline("CKD")

    assert results == []


def test_efetch_network_error_returns_empty_list() -> None:
    """Network error on efetch (after successful esearch) → empty list."""
    call_count = 0

    def _side_effect(url, timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # esearch succeeds
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = _ESEARCH_ONE_PMID
            return mock_resp
        # efetch fails
        raise OSError("connection reset")

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        results = search_medline("CKD")

    assert results == []
