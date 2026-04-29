"""medline_search — NCBI PubMed E-utilities search for clinical literature.

Uses the NCBI esearch endpoint to retrieve PMIDs then efetch (XML) to pull
title + abstract. No API key required. Per Experiment 2 findings (rec. 1):
abstracts are fetched via efetch rather than titles-only via esummary —
title-only search was low-signal for 35% of clinical queries.

Returns an empty list on any error so callers always degrade gracefully.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from src.models.search_result_item import SearchResultItem

logger = logging.getLogger(__name__)

_NCBI_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_NCBI_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_NCBI_TIMEOUT = 10  # seconds
_SOURCE_DOMAIN = "pubmed.ncbi.nlm.nih.gov"


def _fetch_pmids(query: str, max_results: int) -> list[str]:
    """Retrieve PubMed IDs matching query via NCBI esearch.

    Args:
        query: Clinical search query (Title/Abstract field search).
        max_results: Maximum number of PMIDs to return.

    Returns:
        List of PMID strings, empty if no matches or on error.
    """
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query + " [Title/Abstract]",
        "retmode": "json",
        "retmax": max_results,
    })
    with urllib.request.urlopen(f"{_NCBI_SEARCH_URL}?{params}", timeout=_NCBI_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data.get("esearchresult", {}).get("idlist", [])


def _fetch_abstracts(pmids: list[str]) -> dict[str, str]:
    """Fetch title + abstract text for each PMID via NCBI efetch (XML).

    Structured abstracts (multiple AbstractText elements) are concatenated
    in document order. When no abstract is present, only the title is returned.

    Args:
        pmids: List of PubMed IDs to fetch.

    Returns:
        Dict mapping PMID → "Title — Abstract" snippet string.
    """
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    })
    with urllib.request.urlopen(f"{_NCBI_FETCH_URL}?{params}", timeout=_NCBI_TIMEOUT) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    abstracts: dict[str, str] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text

        title_el = article.find(".//ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""

        # Structured abstracts have multiple AbstractText elements — concatenate all.
        abstract_parts = [el.text or "" for el in article.findall(".//AbstractText")]
        abstract = " ".join(p for p in abstract_parts if p).strip()

        abstracts[pmid] = f"{title} — {abstract}" if abstract else title

    return abstracts


def search_medline(query: str, max_results: int = 3) -> list[SearchResultItem]:
    """Search NCBI PubMed and return results with title + abstract snippets.

    Implements Experiment 2, finding recommendation 1: fetches abstracts via
    efetch rather than title-only via esummary. Returns an empty list (not an
    exception) on any NCBI API failure — callers must handle empty results.

    Args:
        query: Clinical search query.
        max_results: Maximum number of PubMed articles to retrieve. Default 3.

    Returns:
        List of SearchResultItem with source_domain="pubmed.ncbi.nlm.nih.gov".
        Empty list when 0 PMIDs found or on any network/parse error.
    """
    try:
        pmids = _fetch_pmids(query, max_results)
    except Exception as exc:  # noqa: BLE001
        logger.error("[medline_search] esearch failed: %s — %s", type(exc).__name__, exc)
        return []

    if not pmids:
        logger.info("[medline_search] 0 PMIDs for query %r", query)
        return []

    try:
        abstracts = _fetch_abstracts(pmids)
    except Exception as exc:  # noqa: BLE001
        logger.error("[medline_search] efetch failed: %s — %s", type(exc).__name__, exc)
        return []

    items: list[SearchResultItem] = []
    for pmid in pmids:
        snippet = abstracts.get(pmid, f"PMID {pmid}")
        # Title is the part before " — "; fall back to snippet when no separator
        title = snippet.split(" — ")[0] if " — " in snippet else snippet
        items.append(
            SearchResultItem(
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                snippet=snippet,
                source_domain=_SOURCE_DOMAIN,
            )
        )

    logger.info("[medline_search] %d results for query %r", len(items), query)
    return items
