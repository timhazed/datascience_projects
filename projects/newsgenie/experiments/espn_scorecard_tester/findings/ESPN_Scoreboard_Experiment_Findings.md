# ESPN Scoreboard API — Experiment Findings

This report summarizes what we learned by exercising ESPN’s public **`site.api.espn.com`** scoreboard JSON for **NFL, NBA, MLB, and NHL** using the `espn_scorecard_tester` harness (`experiments/espn_scorecard_tester/`). 

---

## What we tested

- **Endpoint pattern:**  
  `https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard`
- **Inputs (JSONL):** cases in `data/scoreboard_cases.jsonl` — default scoreboard window per league, `limit` capping, and a **single-day** `dates=YYYYMMDD` query (historical / archive-style use).
- **Outputs:** per-case **Stats** and **Results** JSON under `data/` (filename includes league and case id, plus a shared run timestamp).
- **Vetting:** optional expectations on top-level keys (`events`, `leagues`), event counts, competition `status.type.name` values (e.g. `STATUS_FINAL`, `STATUS_SCHEDULED`, `STATUS_IN_PROGRESS`), and optional substring checks.

Representative successful runs showed **HTTP 200**, JSON bodies with **`events`** and **`leagues`**, and per-game data under **`events[].competitions[]`** (teams, scores, status). Single-request latencies were often in the **tens to low hundreds of milliseconds** on a research run (exact numbers vary by network and ESPN load).

---

## Strengths

1. **No API key** — Simple `GET` with a normal `User-Agent`; useful for prototypes and low-volume augmentation without credential management.
2. **Structured, score-centric payload** — Rich enough to drive “who played, what was the score, what state is the game in?” without HTML scraping: event ids, short names, competitors, and **`status.type.name`** for lifecycle (final, scheduled, in progress, postponed, etc., depending on what ESPN returns for that window).
3. **Query knobs that behave as expected for testing** — **`limit`** to cap how many events come back when many games exist; **`dates`** to target a calendar day or range (format as used by ESPN’s API), which supports “scores on a specific day” style queries.
4. **Latency generally suitable for interactive UX** — Cold-ish requests often complete in **under a few hundred milliseconds** in practice, which is acceptable for a secondary fetch behind a sports node (often after or alongside Guardian).
5. **US major leagues fit** — NFL / NBA / MLB / NHL align with the product gap (US scores vs Guardian’s UK/editorial strength).

---

## Weaknesses

1. **Undocumented and unofficial** — The contract is whatever ESPN serves today. Field names, nesting, and enums can change without notice; client code should treat parsing as defensive and centralize normalization.
2. **Not a news feed** — Scoreboards answer **results and schedule-shaped facts**, not headlines or analysis. They do **not** replace Guardian for “sports news” articles; they **complement** it when the goal is scores or game state.
3. **ESPN editorial and coverage bias** — Data reflects ESPN’s world model (leagues, naming, logos). International or niche competitions outside the tested path are out of scope unless mapped explicitly.
4. **Empty or sparse `events`** — Off-season windows, wrong `dates`, or narrow filters can yield **zero events** while still returning HTTP 200. Product logic must treat “no games” as valid, not as failure.
5. **Status vocabulary** — `status.type.name` is a fixed-ish string set but not formally guaranteed; vetting should allow the union of values you observe in production (final, scheduled, live, delayed, postponed, etc.).

---

## Limitations and risks

| Area | Notes |
|------|--------|
| **Terms of use / stability** | No published SLA; use **low frequency** and avoid aggressive polling. IP throttling or blocking is a real operational risk. |
| **Rate limiting** | Not formally documented; the harness uses a **delay between requests** to reduce abuse risk. Production should cache and deduplicate. |
| **Scope of this experiment** | Only four leagues and a small JSONL matrix. Other sports/leagues would need their own `{sport}/{league}` path validation. |
| **Historical vs “current”** | `dates` behavior for old days depends on ESPN retention; very old dates may return empty `events` even when games existed. |
| **Compliance** | Treat as integration research; legal/compliance review may be required before high-volume or commercial use. |

---

## Conclusion for News Genie

The ESPN scoreboard endpoint is a **strong technical fit as an optional, secondary source for US league scores and game status**, with **low integration friction** (no key) and **reasonable latency**. It is **weak as a substitute for sports journalism** and **fragile as a long-term contract** unless wrapped behind a dedicated client, normalization layer, and graceful degradation when the API shape or availability changes.

**Recommended approach:** **Guardian first for sports news**, **ESPN scoreboard when scores/game state matter** or when Guardian returns nothing relevant for a US-centric query, with caching and conservative call rates.

---

## Artifacts

- Harness: `experiments/espn_scorecard_tester/espn_scorecard_tester.py`
- Cases: `experiments/espn_scorecard_tester/data/scoreboard_cases.jsonl`
- Per-run outputs: `experiments/espn_scorecard_tester/data/ESPN_Scoreboard_{Stats,Results}_<league>_<case_id>_<timestamp>.json`

Regenerate results after code or case changes:

```bash
poetry run espn-scorecard-tester
```
