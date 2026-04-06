# Experiments — Lessons Learned

This note coalesces **what was run**, **what we measured**, and **how findings drove architecture** for NewsGenie. Canonical technical detail remains in **`docs/ARCHITECTURE.md`**; harness code lives under **`experiments/`**.

**PDF artifacts:** `Intent_Validation_Findings.pdf` and `News_API_Validation_Findings.pdf` are referenced from the architecture spec and may live only in local or release bundles. Where they are absent from a checkout, the summaries below match the **`docs/ARCHITECTURE.md` §1 empirical baseline** and the Markdown companion paths noted there.

---

## 1. Why experiments exist

The product pairs a **single supervisor LLM** (routing only) with **parallel HTTP-backed agents** (news APIs, web search, ESPN scoreboard). Experiments validate:

- **Routing quality** — intent decomposition vs ground truth (mini benchmark; not the full ≥90% / 50-query gate).
- **Wire latency & provider fit** — which feeds belong in which lane before coding assumptions into LangGraph.
- **Undocumented integrations** — ESPN’s public scoreboard JSON: shape, latency, and operational risk.

Together, these justified **feed selection**, **parallel fan-out**, **sports = Guardian + ESPN**, and **deprecation of slow or rate-limited alternatives**.

---

## 2. Traceability (architecture ↔ experiments)

| Topic | Primary artifacts | When to re-run |
|--------|-------------------|----------------|
| Supervisor **model** (speed vs accuracy) | `experiments/intent_tester/findings/Intent_Validation_Findings.pdf`, `data/*.stats.json` | Material changes to `SUPERVISOR_PROMPT` or routing rules |
| **Agent / feed** backends | `experiments/news_api_tester/findings/News_API_Validation_Findings.pdf` (and `.md` if maintained) | Provider or `WEB_SEARCH_PROVIDER` changes |
| **ESPN scoreboard** | `experiments/espn_scorecard_tester/findings/ESPN_Scoreboard_Experiment_Findings.md`, `data/ESPN_Scoreboard_*` | Case matrix, URL builder, or normalization changes |

---

## 3. Intent validation (`experiments/intent_tester/`)

### What was run

- **Harness:** `experiments/intent_tester/intent_tester.py` (aligned with the supervisor prompt template; `categories_hint="all"` for unfiltered routing).
- **Dataset:** `data/Intent_Validation_Mini.jsonl` — **10 prompts** covering Business / Sports / General / Web Search decomposition.
- **Runs (2026-03-31):** four model configurations with machine-readable outputs in `data/Intent_Validation_Mini_RUN_<provider>_<model>_*.stats.json` and matching `.jsonl`.

### Findings (intent-set match on 10 prompts)

| Provider / model | Exact intent-set match | Elapsed (s) |
|------------------|------------------------|-------------|
| OpenAI `gpt-4o-mini` | 90% (9/10) | ~27.5 |
| OpenAI `gpt-3.5-turbo-0125` | 90% (9/10) | ~7.6 |
| Groq `gpt-oss-120b` | 90% (9/10) | ~7.8 |
| Groq `gpt-oss-20b` | 80% (8/10) | ~3.3 |

**Per-intent pattern (top models):** Sports News and Web Search reached **100%** precision/recall on the mini set; **Business vs General** drove the misses.

**Consensus error:** All four models misclassified the same case — **“Global Chip Shortage”** — as **General News** where ground truth labeled **Business News**. That points to **taxonomy ambiguity** (macro supply-chain vs “general” world news), not a single flaky model.

### Lessons learned

1. **Mini benchmarks are directional.** They do **not** replace the production criterion (≥90% on a **50-query** eval in `ARCHITECTURE.md`).
2. **Speed vs reliability:** Fast Groq `gpt-oss-20b` traded accuracy on the mini set; `gpt-3.5-turbo-0125` and `gpt-oss-120b` sat in a middle band on both axes.
3. **Prompt / eval alignment:** Persistent disagreement on one prompt suggests **few-shot examples**, **clearer Business vs General definitions**, or **ground-truth review** — not only model swaps.
4. **Architecture implication:** Routing stays a **plain-text, regex-parsed** contract (`INTENT: … | QUERY: …`); experiments validated that multiple providers can sit behind the same pattern, but **quality must be re-checked** when the default model changes (`LLM_PROVIDER` / `SUPERVISOR_MODEL`).

---

## 4. News API & web search (`experiments/news_api_tester/`)

### What was run (per architecture summary)

Latency and behavior were exercised for **MarketAux**, **Guardian**, **NewsAPI**, **SerpAPI**, **Serper** (LangChain wrappers), with calls run **sequentially** in the harness.

### Findings (summarized from `docs/ARCHITECTURE.md` and the News API validation report)

| Observation | Implication |
|-------------|-------------|
| **MarketAux** was the **long pole** (multi-second means; peaks around **~16 s** on a single call) | **Rejected** for the Business lane; not implemented as the Business feed. |
| **NewsAPI** and **Guardian** were **low-latency** in harness runs | **Coalesce Business and World/General on NewsAPI** (`GET /v2/everything`); **Guardian** for **sports articles**. |
| Web search means varied (**SerpAPI** vs **Serper**) | Provider choice is an **env-driven** integration concern; re-benchmark when switching `WEB_SEARCH_PROVIDER`. |
| Harness is **sequential**; production graph uses **parallel fan-out** | User-visible latency tracks **max(branch)**, not the sum of sequential benchmark times. |
| **Alpha Vantage** (News & Sentiments) | **Not adopted** — strict **request-rate limits** for interactive multi-call use. |

### Lessons learned

1. **Pick feeds by latency envelope + keys + semantics**, not headline feature lists alone.
2. **Parallelism changes the story:** sequential API benchmarks **overstate** total wait when branches run concurrently in LangGraph.
3. **Architecture revision:** Business and General/World share **one wire client** (`newsapi_client.py`); Sports uses **Guardian + ESPN** in parallel; **topic verticals** outside those lanes use **web search** — matching requirements.

---

## 5. ESPN scoreboard (`experiments/espn_scorecard_tester/`)

### What was run

- **Endpoint:** `https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard`
- **Cases:** `data/scoreboard_cases.jsonl` — default window, `limit`, and `dates=YYYYMMDD` variants.
- **Outputs:** paired **Stats** and **Results** JSON per run (filename encodes league, case id, timestamp).

### Representative measurement (MLB default window)

From `experiments/espn_scorecard_tester/data/ESPN_Scoreboard_Stats_mlb_mlb_scoreboard_default_20260401_150256.json` and the corresponding **Results** file:

- **HTTP:** 200, **`success: true`**, **`vet_passed: true`**
- **Latency:** ~**71 ms** for that call (single-sample; network-dependent)
- **Payload:** `leagues`, `events`, competitions with teams, scores, and `status.type.name` (e.g. `STATUS_FINAL` for a snapshot run)
- **Vetting:** optional checks on top-level keys, event counts, and status vocabulary

Full narrative (strengths, weaknesses, compliance posture) is in **`experiments/espn_scorecard_tester/findings/ESPN_Scoreboard_Experiment_Findings.md`**.

### Lessons learned

1. **No API key** and **structured score-centric JSON** make ESPN a **practical secondary source** for US major-league **scores and game state**.
2. **Undocumented contract** — defensive parsing, centralized normalization, and graceful degradation when shape or availability changes.
3. **Not a news feed:** ESPN **complements** Guardian for sports; it does **not** replace editorial sports news.
4. **Operational risk:** no SLA; use **caching**, **deduplication**, and **conservative** call rates; empty `events` with HTTP 200 is valid (off-season, filters, retention limits).
5. **Architecture alignment:** **Guardian first for sports news**; **ESPN in parallel** when scores/state matter; product routing also sends **live score-style questions** toward **Web Search** when Guardian is a poor fit — see **`docs/ARCHITECTURE.md`** routing notes.

---

## 6. Cross-cutting themes (refactors & revisions)

1. **Supervisor often dominates end-to-end time** — API branches may be sub-second, but **`T_round_trip ≈ T_supervisor + max(branches) + T_assemble`**; tuning `SUPERVISOR_MODEL` matters as much as feed choice.
2. **Single normalization model** — heterogeneous providers map to **`NormalizedArticle`**; experiments justified **which** providers to wire, not ad-hoc per-demo clients.
3. **Evidence-driven deprecation** — MarketAux and Alpha Vantage fell out of the design for **latency / rate-limit** reasons documented in validation work.
4. **Re-validation discipline** — After **prompt**, **API**, or **search provider** changes, re-run the relevant harnesses and update PDFs or pinned stats so architecture and README stay honest.

---

## 7. Artifact index (quick reference)

| Experiment | Key outputs |
|------------|-------------|
| Intent | `experiments/intent_tester/data/*.stats.json`, `findings/Intent_Validation_Findings.pdf` |
| News / search | `experiments/news_api_tester/findings/News_API_Validation_Findings.pdf` |
| ESPN | `experiments/espn_scorecard_tester/findings/ESPN_Scoreboard_Experiment_Findings.md`, `data/ESPN_Scoreboard_{Stats,Results}_*.json` |

---

*Last aligned with **`docs/ARCHITECTURE.md`** §1 (Goal & Scope, empirical baseline, latency model) and experiment outputs available in-repo as of document creation.*
