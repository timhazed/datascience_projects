"""Converts raw API response dicts into NormalizedArticle objects.

Three normalizer functions — one per active news API provider:
  normalize_newsapi   — for NewsAPI /v2/everything responses (Business + General/World)
  normalize_guardian  — for Guardian Open Platform responses (Sports)
  normalize_espn      — for ESPN public scoreboard API responses (Sports scores)

Web search normalization is handled inline in src/graph/web_search_node.py since
it requires access to the active WebSearchProvider setting at call time.
"""

from src.data.enums import NewsCategory
from src.data.normalized_article import NormalizedArticle


def normalize_newsapi(raw: dict, category: NewsCategory, index: int) -> NormalizedArticle:
    """Convert a single NewsAPI article dict to a NormalizedArticle.

    Args:
        raw: One element from NewsAPI response["articles"].
        category: NewsCategory to assign — BUSINESS or GENERAL depending on the calling agent.
        index: Position in the result set, used to build a stable article_id.
    """
    title = raw.get("title") or ""
    summary = (raw.get("description") or title)[:500]  # null description → title fallback
    source = raw.get("source") or {}
    return NormalizedArticle(
        article_id=f"newsapi_{index}",
        title=title,
        summary=summary,
        url=raw.get("url") or "",
        source_name=source.get("name") or "NewsAPI",
        published_at=raw.get("publishedAt") or "1970-01-01T00:00:00Z",
        category=category,
        provider="newsapi",
        image_url=raw.get("urlToImage"),
    )


def normalize_espn(event: dict, index: int) -> NormalizedArticle:
    """Convert a single ESPN scoreboard event dict to a NormalizedArticle.

    All field access is defensive — the ESPN API is undocumented and field presence
    varies by game state (STATUS_FINAL, STATUS_SCHEDULED, STATUS_IN_PROGRESS).

    Title / summary rules:
      STATUS_FINAL + headlines present → use shortLinkText / description from headlines[0]
      Otherwise → title = event shortName; summary = score string from competitors

    Args:
        event: One element from ESPN scoreboard response["events"].
        index: Position in the result set, used to build a stable article_id.
    """
    competitions = event.get("competitions") or []
    competition = competitions[0] if competitions else {}
    status_name = (competition.get("status") or {}).get("type", {}).get("name", "")
    headlines = competition.get("headlines") or []
    short_name = event.get("shortName") or ""

    if status_name == "STATUS_FINAL" and headlines:
        title = headlines[0].get("shortLinkText") or short_name
        summary = headlines[0].get("description") or short_name
    else:
        title = short_name
        competitors = competition.get("competitors") or []
        if len(competitors) >= 2:
            t1 = (competitors[0].get("team") or {}).get("displayName", "")
            s1 = competitors[0].get("score", "")
            t2 = (competitors[1].get("team") or {}).get("displayName", "")
            s2 = competitors[1].get("score", "")
            summary = f"{t1} {s1} - {t2} {s2}".strip(" -").strip()
        else:
            summary = short_name

    links = event.get("links") or []
    url = links[0].get("href", "") if links else ""

    competitors = competition.get("competitors") or []
    image_url = None
    if competitors:
        image_url = (competitors[0].get("team") or {}).get("logo")

    return NormalizedArticle(
        article_id=f"espn_{index}",
        title=title,
        summary=summary[:500],
        url=url,
        source_name="ESPN",
        published_at=event.get("date") or "1970-01-01T00:00:00Z",
        category=NewsCategory.SPORTS,
        provider="espn",
        image_url=image_url,
    )


def normalize_guardian(raw: dict, category: NewsCategory, index: int) -> NormalizedArticle:
    """Convert a single Guardian API result dict to a NormalizedArticle.

    Args:
        raw: One element from Guardian response["response"]["results"].
        category: NewsCategory to assign — always SPORTS in v1.0.
        index: Position in the result set, used to build a stable article_id.
    """
    fields = raw.get("fields") or {}
    # Prefer the richer headline from show-fields; fall back to webTitle.
    title = fields.get("headline") or raw.get("webTitle") or ""
    # null trailText → title fallback
    summary = (fields.get("trailText") or title)[:500]
    return NormalizedArticle(
        article_id=f"guardian_{index}",
        title=title,
        summary=summary,
        url=raw.get("webUrl") or "",
        source_name=raw.get("sectionName") or "The Guardian",
        published_at=raw.get("webPublicationDate") or "1970-01-01T00:00:00Z",
        category=category,
        provider="guardian",
        image_url=fields.get("thumbnail"),
    )
