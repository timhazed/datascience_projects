"""HTTP client for the ESPN public scoreboard API.

Used by SportsNewsAgent to fetch today's scores alongside Guardian editorial results.

This wraps the undocumented ESPN site API:
  https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard

No API key required. Use low request frequency — no published rate limit or SLA.
"""

import logging
from datetime import UTC, datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# Keyword → (sport_path, league_slug) mapping.
# Keys are lowercase tokens checked against the query.
_LEAGUE_KEYWORDS: dict[str, tuple[str, str]] = {
    # NFL
    "nfl": ("football", "nfl"),
    "football": ("football", "nfl"),
    "chiefs": ("football", "nfl"),
    "patriots": ("football", "nfl"),
    "49ers": ("football", "nfl"),
    "cowboys": ("football", "nfl"),
    "packers": ("football", "nfl"),
    "superbowl": ("football", "nfl"),
    "super bowl": ("football", "nfl"),
    # NBA
    "nba": ("basketball", "nba"),
    "basketball": ("basketball", "nba"),
    "lakers": ("basketball", "nba"),
    "celtics": ("basketball", "nba"),
    "warriors": ("basketball", "nba"),
    "knicks": ("basketball", "nba"),
    "bulls": ("basketball", "nba"),
    "heat": ("basketball", "nba"),
    # MLB
    "mlb": ("baseball", "mlb"),
    "baseball": ("baseball", "mlb"),
    "yankees": ("baseball", "mlb"),
    "red sox": ("baseball", "mlb"),
    "dodgers": ("baseball", "mlb"),
    "cubs": ("baseball", "mlb"),
    "mets": ("baseball", "mlb"),
    # NHL
    "nhl": ("hockey", "nhl"),
    "hockey": ("hockey", "nhl"),
    "bruins": ("hockey", "nhl"),
    "maple leafs": ("hockey", "nhl"),
    "penguins": ("hockey", "nhl"),
    "blackhawks": ("hockey", "nhl"),
    "rangers": ("hockey", "nhl"),
}

# All four supported leagues — used when no keyword matches.
_ALL_LEAGUES: list[tuple[str, str]] = [
    ("football", "nfl"),
    ("basketball", "nba"),
    ("baseball", "mlb"),
    ("hockey", "nhl"),
]


def _leagues_for_query(query: str) -> list[tuple[str, str]]:
    """Map a free-text query to the ESPN (sport_path, league_slug) pairs to fetch.

    Checks query tokens case-insensitively against known league keywords and team
    names. Deduplicates results so each league appears at most once.
    If no keywords match, returns all four supported leagues — safe default for
    a generic "Sports" chip selection.
    """
    lower = query.lower()
    seen: set[tuple[str, str]] = set()
    matched: list[tuple[str, str]] = []
    for keyword, league in _LEAGUE_KEYWORDS.items():
        if keyword in lower and league not in seen:
            seen.add(league)
            matched.append(league)
    return matched if matched else list(_ALL_LEAGUES)


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _fetch_league(
    sport: str,
    league: str,
    date: str,
    limit: int,
    client: httpx.Client,
) -> list[dict]:
    """Fetch scoreboard events for one league. Returns raw event dicts."""
    url = f"{_BASE_URL}/{sport}/{league}/scoreboard"
    params = {"dates": date, "limit": limit}
    response = client.get(url, params=params)
    if response.is_error:
        logger.warning("espn_client: HTTP %d for %s/%s", response.status_code, sport, league)
        return []
    data = response.json()
    return data.get("events", [])


def fetch_scores(query: str, date: str | None = None, limit: int = 5) -> list[dict]:
    """Fetch today's scoreboard events for leagues matching the query.

    Args:
        query: Free-text sports query used to select relevant league(s).
        date: YYYYMMDD string. Defaults to today (UTC).
        limit: Max events per league request. Caller passes settings.max_articles_per_agent.

    Returns:
        List of raw ESPN event dicts, deduplicated by event id across leagues.
        Returns [] on any HTTP or network error — never raises to the caller.
    """
    if date is None:
        date = datetime.now(UTC).strftime("%Y%m%d")

    leagues = _leagues_for_query(query)
    seen_ids: set[str] = set()
    events: list[dict] = []

    with httpx.Client(timeout=_TIMEOUT) as client:
        for sport, league in leagues:
            try:
                raw = _fetch_league(sport, league, date, limit, client)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as exc:
                logger.warning("espn_client: failed fetching %s/%s — %s", sport, league, exc)
                continue
            for event in raw:
                event_id = event.get("id", "")
                if event_id and event_id not in seen_ids:
                    seen_ids.add(event_id)
                    events.append(event)

    return events
