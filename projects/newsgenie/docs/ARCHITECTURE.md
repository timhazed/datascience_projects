# NewsGenie — Architecure v1.0

---

## 1. Goal & Scope

### Problem Statement

Users need a single conversational interface to query real-time news across **Business**, **World** (general / international headlines), and **Sports**, plus **web search** for topic verticals outside those wired feeds, without switching between multiple platforms or APIs. The system must intelligently route queries, fan out to specialized agents, normalize heterogeneous API responses, and **present** curated article results (and web search hits) in a single structured response. Product language and routing rules are aligned with **requirements**.

### In Scope

- Multi-agent LangGraph supervisor with linear fan-out pattern
- Three domain news agents backed by two wire APIs plus ESPN Scoreboard: **Business** and **General / World** headlines share **NewsAPI** `GET /v2/everything`; **Sports** uses **The Guardian** Open Platform plus the **ESPN Scoreboard** (undocumented `site.api.espn.com` API; no key required; US league scores fetched in parallel with Guardian) — per **requirements** and **`experiments/news_api_tester/findings/News_API_Validation_Findings.pdf`**; see **`experiments/espn_scorecard_tester/findings/ESPN_Scoreboard_Experiment_Findings.md`**
- LLM provider factory with OpenAI and Groq support (v1.0); plain text LLM output makes both providers interchangeable via env var
- Supervisor routing with multi-intent decomposition into up to four destination types (Business / Sports / General news APIs, plus Web Search)
- Normalized article schema across all news providers and web search results
- **Programmatic UI response construction:** the only LLM in v1.0 is the **supervisor** (routing). The `assemble` node builds `SupervisorResponse` from fetched articles — **no** LLM-generated prose summaries in v1.0 (see **Future Features**).
- CLI interface (Phase 4) and Streamlit UI (Phase 5)
- Conversation context/session management with bounded history
- Fallback handling for API failures and missing keys

### Out of Scope

- User authentication or multi-tenancy
- News article storage or persistence (no database)
- Subscription/notification features
- Custom fine-tuned models
- Non-English news queries (v1)
- Real-time streaming tokens to UI (v1)
- Model training or fine-tuning (v1 uses prompt-based routing only)
- Sub-second or contractual **SLA** on end-to-end latency (v1.0 uses **design targets** instead — see **Latency model & “instant” responses** below)

### Success Criteria

| Criterion | Measure |
|---|---|
| Intent classification accuracy | ≥ 90% on 50-query eval set |
| Fan-out agent response time | ≤ 4s p95 (parallel execution) |
| API normalization coverage | 100% of fields mapped for all 3 providers |
| Test coverage | ≥ 80% overall; ≥ 90% on `graph/supervisor_node.py`, `graph/assemble_node.py`, `utils/normalizer.py` |
| LLM provider swap (OpenAI ↔ Groq) | No application code edits — `LLM_PROVIDER` (and matching API key) via env only; **routing quality** is not guaranteed to stay ≥ 90% until the chosen default models are re-validated on the 50-query eval |
| Multi-intent accuracy | Correct section count for 20-query multi-part eval set |

### Empirical baseline (optional research record)

The **production** contract is `SUPERVISOR_PROMPT` in §3 (`src/graph/supervisor_node.py`). Research harnesses under `experiments/` may use a similar prompt or HTTP clients for benchmarking; they are not required to match every production revision.

#### Intent validation — `experiments/intent_tester/`

| Artifact | Role |
|----------|------|
| `experiments/intent_tester/findings/Intent_Validation_Findings.pdf` | Comparative supervisor routing across **four** model runs (2026-03-31): Business / Sports / General / Web Search decomposition. |
| `experiments/intent_tester/data/*.stats.json` | Machine-readable stats per run. |

**Findings (summary):** On `data/Intent_Validation_Mini.jsonl`, **gpt-4o-mini**, **gpt-3.5-turbo-0125**, and Groq **gpt-oss-120b** achieved **90%** intent-set match; Groq **gpt-oss-20b** achieved **80%**. Elapsed times ranged ~**3.3 s–27.5 s** per 10-prompt run; **gpt-oss-20b** was fastest; **gpt-4o-mini** slowest in that harness. **Consensus error:** all models mislabeled the **“Global Chip Shortage”** case (ground truth **Business News** vs predicted **General News**) — see report for prompt/ground-truth alignment. **Sports** and **Web Search** reached **100%** precision/recall on top models. **Recommendations in the PDF:** trade **speed** (e.g. **gpt-oss-20b**) vs **reliability** (**gpt-3.5-turbo**); consider few-shot or taxonomy tweaks for supply-chain vs general news. **Does not replace** the ≥90% / 50-query production gate. Re-run after material prompt changes.

#### News API & web search latency — `experiments/news_api_tester/`

| Artifact | Role |
|----------|------|
| `experiments/news_api_tester/findings/News_API_Validation_Findings.pdf` | Latency benchmark: MarketAux, Guardian, NewsAPI, SerpAPI, Serper (LangChain wrappers). |
| `experiments/news_api_tester/findings/News_API_Validation_Findings.md` | Maintained Markdown companion; regenerate the PDF from it when needed. |

**Findings (summary):** **MarketAux** was the **long pole** (multi-second means; peaks ~**16 s** on a single call). **NewsAPI** and **Guardian** were **low-latency**; web search means varied by provider (**SerpAPI** vs **Serper** in recorded runs). The harness runs calls **sequentially**; **parallel fan-out** in LangGraph limits end-user wait to **max(branch)**, not the sum of branches. **Architecture takeaway aligned with requirements:** **coalesce Business and World on NewsAPI** (`/v2/everything`); **Guardian** for **Sports**; **web search** for topic verticals outside those lanes. Alpha Vantage (News & Sentiments) was **not** adopted here due to strict **request-rate limits** on interactive multi-call runs.

#### Traceability

| Topic | Intent report | News API report |
|-------|----------------|------------------|
| Supervisor **model** choice | Speed vs accuracy (see PDF §4) | — |
| **Agent / feed** backends | — | NewsAPI + Guardian + web search; MarketAux deprecated for Business |
| Re-validation | After `SUPERVISOR_PROMPT` changes | After API or `WEB_SEARCH_PROVIDER` changes |

### Latency model & “instant” responses (requirements Result 1)

Requirements refer to **“instant”** responses; v1.0 does **not** stream tokens and cannot guarantee sub-second delivery. Here is how that language maps to engineering:

**Meaning of “instant” for v1.0:** users get a **complete** assistant turn within a **bounded, reasonable** wait under normal conditions, with **visible progress** in the UI during execution (see §6 — e.g. spinner while the graph runs).

**End-to-end path:** `T_round_trip ≈ T_supervisor + T_parallel_fan_out + T_assemble`. Fan-out branches run in parallel, so agent time is **max(branch)**, not sum.

| Stage | Design target (p95, nominal conditions) | Notes |
|-------|-------------------------------------------|--------|
| `supervisor_node` (LLM) | ≤ **8 s** | Dominates when using larger/slower models; tune `SUPERVISOR_MODEL` or provider |
| Parallel agent nodes (news APIs + web search) | ≤ **4 s** | Same row as **Success Criteria** — slowest branch wins; validate against **`News_API_Validation_Findings.pdf`** (NewsAPI/Guardian typically sub-second mean in harness; supervisor often dominates end-to-end) |
| `assemble` | negligible | Deterministic |
| **Round trip (design envelope)** | ≤ **15 s p95** | Not a contractual SLA — cold start, retries, rate limits, or user network can exceed |

These targets are for **planning and acceptance testing**, not legal guarantees. Instrument `supervisor_node` and agent nodes in Hardening to validate p95 against production-like runs.

### Fallbacks & optimization narrative (requirements Result 4)

Stakeholders asked for a **detailed explanation** of fallback behavior and optimization. **Canonical depth** lives in **`README.md`** (Phase 6): when each fallback triggers, what the user sees, env knobs, and tuning trade-offs.

This specification is the **technical source of truth**:

| Theme | Where defined | README must explain (non-code audience) |
|-------|----------------|----------------------------------------|
| **Fallbacks** | §5 Safety & Guardrails, §3 routing fallback, category filter → web search | Missing keys, empty agents, regex parse failure, LLM errors, `fallback_used` |
| **Optimization** | §3 parallel fan-out, §3 retries, §4 LLM single-call design, `max_articles_per_agent`, category filter | Why parallel fan-out, retries on transient errors, bounded history, model/provider selection |

**Traceability:** Phase 6 README updates should cross-check tables in §5 and the latency table above so marketing language (“instant,” “reliable”) stays aligned with documented behavior.

---

## 2. Pydantic Data Models

Models are split one-class-per-file under `src/data/`. All re-exported from `src/data/__init__.py`.
Enums are grouped in a single `enums.py` (standard practice; enums carry no OOP state).
Models that are tight structural wrappers share a file only when they have exactly one shared reason to change (documented per case).

### `src/data/enums.py`

```python
from enum import Enum


class NewsCategory(str, Enum):
    """Category assigned to a normalized article. Used for filtering and display."""
    BUSINESS = "business"
    SPORTS = "sports"
    GENERAL = "general"
    WEB = "web"       # Web search results


class LLMProvider(str, Enum):
    """
    Registered LLM providers. Adding a new provider requires only:
    1. Add the enum value here.
    2. Implement LLMProviderBuilder (src/utils/provider_builder.py).
    3. Call LLMFactory.register() once at app startup.
    No other code changes required.
    """
    OPENAI = "openai"
    GROQ = "groq"   # Supported in v1.0 — plain text LLM output requires no JSON schema mode


class WebSearchProvider(str, Enum):
    """
    Registered web search providers. Exactly one is active per run, selected by
    WEB_SEARCH_PROVIDER in .env. Only that provider's API key is required.

    SERPAPI  — LangChain SerpAPIWrapper (google-search-results SDK)
    SERPER   — LangChain GoogleSerperAPIWrapper (serper.dev API)

    Both wrappers expose wrapper.results(query) → dict. Key differences:
      SerpAPI : results dict key = "organic_results"; item fields: title, snippet, link, source, date
      Serper  : results dict key = "organic";         item fields: title, snippet, link, source
    """
    SERPAPI = "serpapi"
    SERPER = "serper"


class AgentName(str, Enum):
    """
    The four routing destinations produced by the supervisor node.
    Maps directly to the agent_map in supervisor_node.

    business_agent, sports_agent, general_agent: custom domain agents (src/agents/).
    web_search_agent: plain LangGraph node — invokes the configured web search provider
    (SerpAPI or Serper) via LangChain community wrappers. Not a custom agent class.
    """
    BUSINESS = "business_agent"
    SPORTS = "sports_agent"
    GENERAL = "general_agent"
    WEB_SEARCH = "web_search_agent"
```

### `src/data/llm_config.py`

```python
from pydantic import BaseModel, Field
from src.data.enums import LLMProvider


class LLMConfig(BaseModel):
    """Configuration for a single LLM instantiation. One config = one cache entry in LLMFactory."""

    provider: LLMProvider = Field(description="LLM provider to use")
    model_name: str = Field(description="Model identifier string from provider")
    temperature: float = Field(description="Sampling temperature — must be set explicitly. Use 0.0 for routing/classification tasks; higher values only for creative generation.")
    max_tokens: int = Field(description="Token budget for this specific use case — calculated, not arbitrary")
```

### `src/data/user_query.py`

```python
from pydantic import BaseModel, Field, field_validator
from src.data.enums import NewsCategory


class UserQuery(BaseModel):
    """Validated user input. Never passed directly to an LLM — used to construct prompts."""

    text: str = Field(min_length=1, max_length=1000, description="Raw user input text")
    categories: list[NewsCategory] = Field(
        default_factory=list,
        description="Categories selected in the UI. **Empty list = no restriction** (all routing destinations "
                    "allowed; no post-filter). Non-empty = keep only plan steps in those categories — see §3 **User category filter**."
    )
    session_id: str = Field(description="Unique session identifier for conversation context")

    @field_validator("text")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        """Strip whitespace; reject blank strings and special-character-only input."""
        v = v.strip()
        if not any(c.isalnum() for c in v):
            raise ValueError("Query must contain at least one alphanumeric character")
        return v
```

### `src/data/plan.py`

The supervisor node produces a routing plan by parsing plain text LLM output with regex — no JSON schema or Pydantic structured output is used. The plan is a simple list of `PlanStep` objects.

```python
from pydantic import BaseModel, Field
from src.data.enums import AgentName


class PlanStep(BaseModel):
    """One routing instruction produced by the supervisor node for one identified intent."""

    agent: AgentName = Field(description="Which agent handles this step")
    query: str = Field(description="Extracted sub-query for this agent, verbatim from LLM output")


class RoutingPlan(BaseModel):
    """
    The full routing plan for a user message. Produced by supervisor_node via regex parsing.
    Up to 3 steps (matching the supervisor prompt's max-3-intent rule).
    """

    steps: list[PlanStep] = Field(
        min_length=1,
        max_length=3,
        description="Ordered list of agent dispatch instructions. Fan-out executes all steps in parallel."
    )
    original_query: str = Field(description="The original unmodified user query text")
```

### `src/data/raw_articles.py`

The two raw article models share this file: both are pre-normalization provider DTOs with no shared behavior. They change only when a provider's API contract changes, and are always imported together by `normalizer.py`.

> MarketAux was evaluated and rejected as the Business lane feed (multi-second latency, see `experiments/news_api_tester/findings/News_API_Validation_Findings.pdf`). It is not implemented. Business and General/World both use `NewsAPIRawArticle` via `newsapi_client.py`.

```python
from typing import Optional
from pydantic import BaseModel, Field


class GuardianRawArticle(BaseModel):
    """Raw response shape from Guardian Open Platform — pre-normalization."""

    id: str
    webTitle: str
    webUrl: str
    webPublicationDate: str
    sectionName: str
    fields: Optional[dict] = None


class NewsAPIRawArticle(BaseModel):
    """Raw response shape from NewsAPI — pre-normalization."""

    source: dict
    author: Optional[str] = None
    title: str
    description: Optional[str] = None
    url: str
    urlToImage: Optional[str] = None
    publishedAt: str
    content: Optional[str] = None
```

### `src/data/normalized_article.py`

```python
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from src.data.enums import NewsCategory


class NormalizedArticle(BaseModel):
    """Canonical article schema across all three providers. Output of every normalizer function."""

    article_id: str = Field(description="Unique identifier, provider-prefixed (e.g. newsapi_uuid)")
    title: str = Field(description="Article headline")
    summary: str = Field(description="Article summary or snippet, max 500 chars")
    url: str = Field(description="Canonical URL to full article")
    source_name: str = Field(description="Name of the publication or feed")
    published_at: datetime = Field(description="Publication timestamp in UTC")
    category: NewsCategory = Field(description="News category this article belongs to")
    provider: str = Field(
        description="Source id: newsapi | guardian | espn | serpapi | serper"
    )
    image_url: Optional[str] = Field(default=None, description="Thumbnail image URL if available")
    sentiment_score: Optional[float] = Field(
        default=None, description="Sentiment score -1.0 to 1.0 if provider supplies it"
    )
    tags: list[str] = Field(default_factory=list, description="Topic tags or entities")

    @field_validator("published_at", mode="before")
    @classmethod
    def parse_published_at(cls, v: Any) -> datetime:
        """Coerce any date string to datetime.

        Handles ISO 8601 (NewsAPI, Guardian), human-readable strings from SerpAPI
        (e.g. 'Mar 31, 2026', '3 hours ago'), and datetime objects.
        Falls back to Unix epoch (UTC) for unparseable values so that validation
        never raises on a missing or malformed date.
        """
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            from dateutil import parser as dtparser
            try:
                return dtparser.parse(v)
            except (ValueError, OverflowError):
                return datetime.fromtimestamp(0, tz=timezone.utc)
        return datetime.fromtimestamp(0, tz=timezone.utc)
```

### `src/data/agent_result.py`

```python
from typing import Optional
from pydantic import BaseModel, Field
from src.data.enums import AgentName, NewsCategory
from src.data.normalized_article import NormalizedArticle


class NewsAgentResult(BaseModel):
    """Output of a single agent execution after fetching and normalizing articles."""

    agent: AgentName = Field(description="Which agent produced this result — used to match back to the plan step")
    category: NewsCategory = Field(description="Category this agent owns")
    articles: list[NormalizedArticle] = Field(description="Normalized articles returned")
    query_used: str = Field(description="Actual query string sent to the API or search tool")
    success: bool = Field(description="Whether the API/tool call succeeded")
    error_message: Optional[str] = Field(default=None, description="Error detail if success=False")
    latency_ms: int = Field(description="API round-trip time in milliseconds")
```

### `src/data/response.py`

`IntentSection` and `SupervisorResponse` share this file: `SupervisorResponse` is a direct structural wrapper around `IntentSection`. Same single reason to change.

Articles are presented directly — no LLM prose synthesis. One section per plan step; web search results appear as articles with `provider="serpapi"`.

```python
from pydantic import BaseModel, Field
from src.data.enums import AgentName, NewsCategory
from src.data.normalized_article import NormalizedArticle


class IntentSection(BaseModel):
    """
    One section of the final response, corresponding to one PlanStep.
    Constructed programmatically from agent_results articles for that agent.
    Multi-step plans produce multiple IntentSections in SupervisorResponse.
    """

    agent: AgentName = Field(description="Which agent produced this section's articles")
    query: str = Field(description="The sub-query this section answers, from the routing plan")
    articles: list[NormalizedArticle] = Field(
        default_factory=list,
        description="Normalized articles returned by this agent. Empty if agent returned no results."
    )


class SupervisorResponse(BaseModel):
    """Final structured response returned to the UI and CLI."""

    session_id: str
    sections: list[IntentSection] = Field(
        min_length=1,
        description="One section per plan step. Single-intent prompts have one section."
    )
    sources_used: list[str] = Field(description="Deduplicated provider names across all sections")
    fallback_used: bool = Field(description="True if any agent returned empty results")
```

### `src/utils/constants.py`

Zero-import file. Defines cross-layer constants that both `src/data/` and `src/utils/config.py` need, without creating a circular dependency.

```python
# src/utils/constants.py
# Zero imports — safe to import from any layer without circular dependency risk.

# Maximum conversation turns retained in context window.
# One turn = one user message + one assistant message (2 messages total).
MAX_HISTORY_TURNS: int = 10
```

---

### `src/data/state.py`

```python
from __future__ import annotations

import operator
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from src.data.agent_result import NewsAgentResult
from src.data.plan import RoutingPlan
from src.data.response import SupervisorResponse
from src.data.user_query import UserQuery
from src.utils.constants import MAX_HISTORY_TURNS


class AgentState(BaseModel):
    """
    Mutable LangGraph graph state. Passed between all nodes.

    LangGraph note: nodes return dict patches (not mutated AgentState instances).
    The graph is constructed as StateGraph(AgentState) — LangGraph 0.3+ handles
    Pydantic v2 models as state natively.

    agent_results uses Annotated[..., operator.add] so LangGraph's parallel
    fan-out dispatch accumulates results across branches rather than overwriting.
    """

    query: UserQuery
    llm_config: Optional[LLMConfig] = None       # Session override; None → SUPERVISOR_LLM_CONFIG default
    plan: Optional[RoutingPlan] = None           # Produced by supervisor_node via regex parse
    agent_results: Annotated[list[NewsAgentResult], operator.add] = Field(default_factory=list)
    final_response: Optional[SupervisorResponse] = None
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="List of {'role': str, 'content': str} dicts. "
                    "Capped at MAX_HISTORY_TURNS turns (20 messages). Oldest turns dropped FIFO."
    )
    error: Optional[str] = None
```

### `src/data/__init__.py`

```python
from src.data.enums import AgentName, LLMProvider, NewsCategory, WebSearchProvider
from src.data.llm_config import LLMConfig
from src.data.user_query import UserQuery
from src.data.plan import PlanStep, RoutingPlan
from src.data.raw_articles import GuardianRawArticle, NewsAPIRawArticle
from src.data.normalized_article import NormalizedArticle
from src.data.agent_result import NewsAgentResult
from src.data.response import IntentSection, SupervisorResponse
from src.data.state import AgentState

__all__ = [
    "AgentName", "LLMProvider", "NewsCategory", "WebSearchProvider",
    "LLMConfig",
    "UserQuery",
    "PlanStep", "RoutingPlan",
    "GuardianRawArticle", "NewsAPIRawArticle",
    "NormalizedArticle",
    "NewsAgentResult",
    "IntentSection", "SupervisorResponse",
    "AgentState",
]
```

---

## 3. LangGraph Graph Topology

### Linear Fan-Out Architecture

![New Genie Architecture](../images/news_genie_architecture.png)

### Node Responsibilities

| Node | Input | Output field written | LLM Call |
|---|---|---|---|
| `supervisor_node` | `AgentState.query` | `state.plan: RoutingPlan` | Yes — plain text; regex-parsed into `RoutingPlan` |
| `business_node` | `AgentState.plan` step | appends to `state.agent_results` | No — **NewsAPI** `everything` + normalize (same wire API as `general_node`; see News API validation report) |
| `sports_node` | `AgentState.plan` step | appends to `state.agent_results` | No — Guardian API + ESPN Scoreboard (always parallel) + normalize |
| `general_node` | `AgentState.plan` step | appends to `state.agent_results` | No — NewsAPI + normalize |
| `web_search_node` | `AgentState.plan` step | appends to `state.agent_results` | No — SerpAPI tool (LangGraph built-in) |
| `assemble` | `AgentState` with all results | `state.final_response: SupervisorResponse` | No — purely programmatic |

**Web search is a tool, not a fourth “news agent” class.** `business_node`, `sports_node`, and `general_node` are thin wrappers around **HTTP clients** (`httpx`) plus **normalization** to `NormalizedArticle`. `web_search_node` is different: it wires a **LangChain community search utility** (e.g. `SerpAPIWrapper` / provider selected via `WEB_SEARCH_PROVIDER`) as a **LangGraph tool invocation**—there is **no** LLM inside that node and **no** bespoke agent class beyond binding the tool and mapping its output into the same `NormalizedArticle` shape. The supervisor still labels the intent **“Web Search”** in plain text; the runtime mechanism is **tool call**, not a chat-style agent loop.

### Supervisor Node — Plain Text + Regex

No JSON schema. The LLM returns formatted plain text; regex extracts intent/query pairs.

**Regex limitations:** The pattern is one line per intent: `INTENT: … | QUERY: …`. Sub-queries that contain `|` or newlines inside the query text can truncate or fail to parse — treat as rare; if validation fails, fall back to Web Search on the full user message. Prefer keeping sub-queries short and without pipe characters in prompts (documented in test edge cases).

```python
SUPERVISOR_PROMPT = """
You are a News Facilitator. Your task is to decompose a user query into specific intents for specialized agents: [Business News, Sports News, General News, Web Search].

USER-SELECTED CATEGORIES (from UI): {categories_hint}
If this is not "all", prefer routing into these domains when the query supports it.

PRIOR CONVERSATION:
{history}

INTENT DEFINITIONS:
1. Business News: Use for EVERYTHING involving money, company earnings, stock prices, crypto, inflation, the Fed, exchange rates, and semiconductor industry news.
2. Sports News: Use ONLY for sports-related news, player trades, injury updates, or tournament news (e.g., "NBA trade deadline").
3. General News: Use for global headlines, politics, international conflict, and major non-financial/non-sporting events.
4. Web Search: Use for real-time sports scores (e.g., "did the Red Sox win"), game results, weather, non-news facts (e.g., "how to bake bread"), or local niche data.

RULES:
1. Identify up to 3 distinct intents. Do not repeat the same intent/query pair.
2. For each intent, extract the specific sub-query relevant ONLY to that agent.
3. If a query mentions a specific "price," "rate," or "ticker," it MUST be Business News.
4. If a query asks "who won," "what was the score," or refers to a specific game result from "yesterday" or "last night," it MUST be Web Search.
5. If a query is about general sports topics (trades, news, injuries) WITHOUT asking for a score, use Sports News.
6. If unsure or information is missing, DO NOT GUESS. Default to:
   INTENT: Web Search | QUERY: [Full user sub-query]

7. Respond ONLY in the following format:
INTENT: [Intent Name] | QUERY: [Extracted Sub-query]

USER PROMPT: {prompt}
"""
```

**`{history}` slot:** filled by `_format_history(trimmed)` before the LLM call. Produces `"ROLE: content\n..."` lines from the trimmed conversation history, or `"No prior conversation."` when empty. Allows the supervisor to resolve pronouns and follow-up references (e.g. "tell me more about the second story") without the user re-stating context.

```python
def _format_history(history: list[dict]) -> str:
    """Convert trimmed conversation history to a plain-text block for the supervisor prompt."""
    if not history:
        return "No prior conversation."
    return "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history)
```

**Routing change from initial spec:** real-time scores and game results (e.g. "did the Bruins win last night?") are routed to **Web Search** rather than Sports News. The Guardian API does not carry live scores reliably; web search returns up-to-date results for these queries. General sports news (trades, injuries, tournament previews) continues to use Sports News → Guardian.

**Experiments:** `experiments/intent_tester/intent_tester.py` uses this same template with `categories_hint="all"` so JSONL benchmarks match **unfiltered** routing (equivalent to no Streamlit category restriction).

```python
# Maps LLM intent labels to AgentName enum members (never use raw strings for PlanStep.agent).
INTENT_LABEL_TO_AGENT: dict[str, AgentName] = {
    "Business News": AgentName.BUSINESS,
    "Sports News": AgentName.SPORTS,
    "General News": AgentName.GENERAL,
    "Web Search": AgentName.WEB_SEARCH,
}


def _categories_hint(categories: list[NewsCategory]) -> str:
    if not categories:
        return "all"  # human-readable hint for the LLM; not an enum value
    return ", ".join(c.value for c in categories)


def _parse_supervisor_steps(raw: str) -> list[PlanStep]:
    pattern = r"INTENT:\s*(.*?)\s*\|\s*QUERY:\s*(.*)"
    matches = re.findall(pattern, raw, flags=re.MULTILINE)
    return [
        PlanStep(
            agent=INTENT_LABEL_TO_AGENT.get(i.strip(), AgentName.WEB_SEARCH),
            query=q.strip(),
        )
        for i, q in matches
    ]


def filter_plan_to_user_categories(plan: RoutingPlan, categories: list[NewsCategory]) -> RoutingPlan:
    """
    After the LLM plan is parsed, drop any step whose agent is not in the user's allowed categories.
    If the list is **empty**, the user did not restrict categories — the plan is unchanged.
    If every step is filtered out, fall back to a single Web Search step on the original query.
    """
    if not categories:
        return plan
    allowed = set(categories)
    agent_to_category = {
        AgentName.BUSINESS: NewsCategory.BUSINESS,
        AgentName.SPORTS: NewsCategory.SPORTS,
        AgentName.GENERAL: NewsCategory.GENERAL,
        AgentName.WEB_SEARCH: NewsCategory.WEB,
    }
    filtered = [s for s in plan.steps if agent_to_category[s.agent] in allowed]
    if not filtered:
        return RoutingPlan(
            steps=[PlanStep(agent=AgentName.WEB_SEARCH, query=plan.original_query)],
            original_query=plan.original_query,
        )
    return RoutingPlan(steps=filtered, original_query=plan.original_query)


def supervisor_node(state: AgentState) -> dict:
    """
    Invoke the LLM with a plain text prompt; parse output with regex into a RoutingPlan.
    No structured output — provider-agnostic by design.

    LLM is fetched via LLMFactory.get() which caches the instance by config key —
    effectively a singleton per config, built on first call rather than at import time
    to allow test injection without side effects during module collection.
    state.llm_config overrides SUPERVISOR_LLM_CONFIG when set (e.g. from Streamlit sidebar).

    Conversation history is trimmed to MAX_HISTORY_TURNS before injection so the
    context window never grows unboundedly across long sessions.
    """
    llm = LLMFactory.get(state.llm_config or SUPERVISOR_LLM_CONFIG)
    trimmed = trim_history(state.conversation_history)
    raw_output = llm.invoke(
        SUPERVISOR_PROMPT.format(
            categories_hint=_categories_hint(state.query.categories),
            history=_format_history(trimmed),
            prompt=state.query.text,
        )
    )
    steps = _parse_supervisor_steps(raw_output.content)
    base = RoutingPlan(
        steps=steps or [PlanStep(agent=AgentName.WEB_SEARCH, query=state.query.text)],
        original_query=state.query.text,
    )
    plan = filter_plan_to_user_categories(base, state.query.categories)
    return {"plan": plan}
```

### Conditional Edge — Fan-Out from Plan

```python
def route_from_plan(state: AgentState) -> list[str]:
    """
    Fan out to all nodes identified in the routing plan.
    Returns a list of node names — LangGraph dispatches them in parallel.
    Registered as: graph.add_conditional_edges("supervisor_node", route_from_plan)
    Guard: if plan is absent (supervisor raised before writing state), fall back to web_search.
    """
    if state.plan is None:
        return [AgentName.WEB_SEARCH.value]
    return [step.agent.value for step in state.plan.steps]
```

### Assemble Node — Purely Programmatic

```python
def assemble(state: AgentState) -> dict:
    """
    Build SupervisorResponse from agent_results — no LLM call.
    One IntentSection per plan step; articles matched by agent name.
    """
    sections = []
    for step in state.plan.steps:
        step_articles = [
            a for r in state.agent_results
            if r.agent == step.agent
            for a in r.articles
        ]
        sections.append(IntentSection(agent=step.agent, query=step.query, articles=step_articles))

    providers = sorted({a.provider for s in sections for a in s.articles})
    fallback_used = any(not s.articles for s in sections)
    response = SupervisorResponse(
        session_id=state.query.session_id,
        sections=sections,
        sources_used=providers,
        fallback_used=fallback_used,
    )
    return {"final_response": response}
```

### Domain Agent Nodes — `src/graph/agent_nodes.py`

`src/agents/` contains the fetch-and-normalize domain logic (one class per file, abstract `fetch()` interface). `src/graph/agent_nodes.py` contains the three thin LangGraph node functions that instantiate those agents and bridge them into graph state.

**Call pattern:** each node function instantiates its agent (lightweight — agents hold no state), calls `fetch()`, and returns a dict patch to `agent_results`. Agent instances are per-call objects (stateless HTTP wrappers); the HTTP clients they use are module-level singletons in `src/api/`.

```python
# src/graph/agent_nodes.py
# Three graph node functions — one per domain agent.
# Each instantiates its agent, calls fetch(), and returns a state patch.
# HTTP clients (newsapi_client, guardian_client) are module-level singletons in src/api/.

from src.agents.business_agent import BusinessNewsAgent
from src.agents.sports_agent import SportsNewsAgent
from src.agents.general_news_agent import GeneralNewsAgent
from src.data import AgentName, AgentState, NewsAgentResult, NewsCategory
from src.utils.config import settings


def business_node(state: AgentState) -> dict:
    """Fetch and normalize Business News articles for the plan step assigned to this agent."""
    step = next((s for s in state.plan.steps if s.agent == AgentName.BUSINESS), None)
    if step is None:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.BUSINESS, category=NewsCategory.BUSINESS,
            articles=[], query_used="", success=False,
            error_message="Step not found in plan", latency_ms=0,
        )]}
    result = BusinessNewsAgent(settings=settings).fetch(step.query)
    return {"agent_results": [result]}


def sports_node(state: AgentState) -> dict:
    """Fetch and normalize Sports News articles for the plan step assigned to this agent."""
    step = next((s for s in state.plan.steps if s.agent == AgentName.SPORTS), None)
    if step is None:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.SPORTS, category=NewsCategory.SPORTS,
            articles=[], query_used="", success=False,
            error_message="Step not found in plan", latency_ms=0,
        )]}
    result = SportsNewsAgent(settings=settings).fetch(step.query)
    return {"agent_results": [result]}


def general_node(state: AgentState) -> dict:
    """Fetch and normalize General/World News articles for the plan step assigned to this agent."""
    step = next((s for s in state.plan.steps if s.agent == AgentName.GENERAL), None)
    if step is None:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.GENERAL, category=NewsCategory.GENERAL,
            articles=[], query_used="", success=False,
            error_message="Step not found in plan", latency_ms=0,
        )]}
    result = GeneralNewsAgent(settings=settings).fetch(step.query)
    return {"agent_results": [result]}
```

### Web Search Node — `src/graph/web_search_node.py`

`web_search_node` is a plain LangGraph node function — not a chat agent loop. It selects the active web search provider from `settings.web_search_provider`, constructs the appropriate LangChain wrapper, and maps results into `NormalizedArticle` objects. No LLM is involved.

```python
# src/graph/web_search_node.py
import time
from langchain_community.utilities import GoogleSerperAPIWrapper, SerpAPIWrapper
from src.data import AgentState, AgentName, NewsAgentResult, NormalizedArticle, NewsCategory
from src.data.enums import WebSearchProvider
from src.utils.config import settings


def web_search_node(state: AgentState) -> dict:
    """
    Execute a web search for the Web Search plan step using the configured provider.

    Provider selection (from settings.web_search_provider):
      SERPAPI → SerpAPIWrapper(serpapi_api_key=...), results key "organic_results"
      SERPER  → GoogleSerperAPIWrapper(serper_api_key=...), results key "organic"

    Returns NewsAgentResult(success=False) if the key is absent or any call fails.

    Guard order (single-exit pattern):
      1. Resolve provider + key; missing key → early return (top-of-function guard)
      2. Missing plan step → early return (defensive guard)
      3. Web search call inside try/except → single return at bottom
    """
    # Guard 1 — resolve provider and API key; return immediately if key absent.
    web_provider = settings.web_search_provider
    api_key = (
        settings.serpapi_api_key
        if web_provider == WebSearchProvider.SERPAPI
        else settings.serperdev_api_key
    )
    if not api_key:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
            articles=[], query_used="", success=False,
            error_message=f"{web_provider.value.upper()}_API_KEY not configured", latency_ms=0,
        )]}

    # Guard 2 — step not in plan (defensive; LangGraph guarantees routing, but guard regardless).
    step = next((s for s in state.plan.steps if s.agent == AgentName.WEB_SEARCH), None)
    if step is None:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
            articles=[], query_used="", success=False,
            error_message="Step not found in plan", latency_ms=0,
        )]}

    # Construct wrapper and determine results key for the active provider.
    if web_provider == WebSearchProvider.SERPAPI:
        wrapper = SerpAPIWrapper(serpapi_api_key=api_key)
        results_key = "organic_results"
    else:
        wrapper = GoogleSerperAPIWrapper(serper_api_key=api_key)
        results_key = "organic"
    provider_label = web_provider.value

    start = time.monotonic()
    try:
        raw = wrapper.results(step.query)
    except ValueError as exc:
        # SerpAPI only: _process_response raises ValueError when response contains an "error" key.
        latency_ms = int((time.monotonic() - start) * 1000)
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
            articles=[], query_used=step.query, success=False,
            error_message=str(exc), latency_ms=latency_ms,
        )]}
    except Exception as exc:
        # Network failure, timeout, or unexpected error from either provider.
        latency_ms = int((time.monotonic() - start) * 1000)
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
            articles=[], query_used=step.query, success=False,
            error_message=repr(exc), latency_ms=latency_ms,
        )]}

    latency_ms = int((time.monotonic() - start) * 1000)

    # Serper guard — HTTP 200 with error in JSON body (e.g. invalid key returns
    # {"message": "Invalid API key"} at 200). SerpAPI raises ValueError instead.
    if web_provider == WebSearchProvider.SERPER:
        for key in ("message", "error", "errors"):
            val = raw.get(key) if isinstance(raw, dict) else None
            err_text = (
                val if isinstance(val, str) and val.strip()
                else val[0] if isinstance(val, list) and val and isinstance(val[0], str)
                else None
            )
            if err_text:
                return {"agent_results": [NewsAgentResult(
                    agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
                    articles=[], query_used=step.query, success=False,
                    error_message=f"Serper API error: {err_text}", latency_ms=latency_ms,
                )]}

    articles = [
        NormalizedArticle(
            article_id=f"{provider_label}_{i}",
            title=r.get("title", ""),
            summary=r.get("snippet", r.get("title", "")),
            url=r.get("link", ""),
            source_name=r.get("source", "Web"),
            # published_at: NormalizedArticle.parse_published_at handles arbitrary date strings.
            # Serper does not return a date field; epoch fallback is intentional.
            published_at=r.get("date", "1970-01-01T00:00:00Z"),
            category=NewsCategory.WEB,
            provider=provider_label,
        )
        for i, r in enumerate(raw.get(results_key, [])[:settings.max_articles_per_agent])
    ]
    return {"agent_results": [NewsAgentResult(
        agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
        articles=articles, query_used=step.query,
        success=True, latency_ms=latency_ms,
    )]}
```

> **Provider wrapper usage** (confirmed against `experiments/news_api_tester/news_api_tester.py`):
> - **SerpAPI**: `SerpAPIWrapper(serpapi_api_key=k)`, `wrapper.results(query)` → dict with `"organic_results"` list. `ValueError` raised by `_process_response` when the response dict contains an `"error"` key.
> - **Serper**: `GoogleSerperAPIWrapper(serper_api_key=k)`, `wrapper.results(query)` → dict with `"organic"` list. Does **not** raise `ValueError` on API errors — instead returns HTTP 200 with `{"message": "..."}` / `{"error": "..."}` / `{"errors": [...]}` in the body. The node checks for these keys after the call. Key read from `SERPERDEV_API_KEY` (mapped to `settings.serperdev_api_key`); also accepted as `SERPER_API_KEY` in the env.

### Graph Builder — `src/graph/builder.py`

```python
# src/graph/builder.py
from langgraph.graph import StateGraph, END
from src.data import AgentState, AgentName
from src.graph.supervisor_node import supervisor_node
from src.graph.agent_nodes import business_node, sports_node, general_node
from src.graph.web_search_node import web_search_node
from src.graph.assemble_node import assemble
from src.graph.edges import route_from_plan


def build_graph():
    """
    Assemble and compile the NewsGenie LangGraph graph.
    Returns a compiled graph ready for .invoke() calls.

    Topology:
      supervisor_node → [conditional fan-out] → business_node | sports_node |
                                                  general_node | web_search_node
      all branches → assemble → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("business_agent", business_node)
    graph.add_node("sports_agent", sports_node)
    graph.add_node("general_agent", general_node)
    graph.add_node("web_search_agent", web_search_node)
    graph.add_node("assemble", assemble)

    graph.set_entry_point("supervisor_node")

    # route_from_plan returns list[str] of node names directly.
    # No path_map needed: list fan-out uses the strings as node names without a dict lookup.
    graph.add_conditional_edges("supervisor_node", route_from_plan)

    for agent_node in [AgentName.BUSINESS.value, AgentName.SPORTS.value,
                       AgentName.GENERAL.value, AgentName.WEB_SEARCH.value]:
        graph.add_edge(agent_node, "assemble")

    graph.add_edge("assemble", END)

    return graph.compile()
```

> **Fan-in pattern:** all four agent nodes write to `agent_results` via the `operator.add` reducer. LangGraph accumulates writes from all parallel branches before advancing to `assemble`. The `add_edge(agent_node, "assemble")` wires each branch to converge there.

### Retry Strategy

Every supervisor LLM invocation and every external API/tool call is wrapped with a `tenacity` retry decorator. Transient errors (rate limits, timeouts, connection errors) are retried with exponential backoff; logic errors (`ValidationError`, `ValueError`) are not retried. Detailed per-call retry tuning is deferred to the Hardening phase (see Future Features).

### Conversation History Trim Policy

```python
def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Retain the most recent max_turns complete turns (user + assistant pairs).
    One turn = 2 messages. Older turns dropped FIFO.
    Returns a new list — does not mutate state in-place.
    """
    max_messages = max_turns * 2
    return history[-max_messages:] if len(history) > max_messages else history
```

Applied by every node that injects `conversation_history` into a prompt, before the inject.

---

## 4. LLM Provider Config

### One LLM Call, Fully Provider-Agnostic

There is exactly **one LLM call** in the system: the `supervisor_node` that classifies and decomposes the user's query. The output is plain text parsed with regex — no JSON schema, no structured output mode.

This means **any provider that supports basic chat completion works without modification**. OpenAI and Groq are both supported in v1.0.

### Provider Factory — Three Files (One Class Per File)

#### `src/utils/provider_builder.py`

```python
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from src.data import LLMConfig


class LLMProviderBuilder(ABC):
    """
    Abstract builder contract for LLM provider construction.
    One concrete subclass per supported LLM provider.
    Implement this class and register with LLMFactory.register() to add a new provider.
    """

    @abstractmethod
    def build(self, config: LLMConfig) -> BaseChatModel:
        """Construct and return a fully configured LLM instance for this provider."""
```

#### `src/utils/openai_builder.py`

```python
from langchain_core.language_models import BaseChatModel
from src.data import LLMConfig
from src.utils.provider_builder import LLMProviderBuilder


class OpenAIBuilder(LLMProviderBuilder):
    """Builds ChatOpenAI instances. Registered by default at app startup."""

    def build(self, config: LLMConfig) -> BaseChatModel:
        """Construct a ChatOpenAI instance from the provided config."""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
```

#### `src/utils/llm_factory.py`

```python
from langchain_core.language_models import BaseChatModel
from src.data import LLMConfig, LLMProvider
from src.utils.provider_builder import LLMProviderBuilder
from src.utils.openai_builder import OpenAIBuilder


class LLMFactory:
    """
    Singleton-per-config LLM constructor. Never instantiated per-request.

    Cache key includes provider, model_name, temperature, AND max_tokens so that
    configs with different budgets or temperatures never share an instance even
    when using the same underlying model.

    Extending to a new provider: implement LLMProviderBuilder, then call
    LLMFactory.register(LLMProvider.NEW, NewBuilder()) once at app startup.
    No changes to this class required.
    """

    _instances: dict[str, BaseChatModel] = {}
    _builders: dict[LLMProvider, LLMProviderBuilder] = {
        LLMProvider.OPENAI: OpenAIBuilder(),
    }

    @classmethod
    def register(cls, provider: LLMProvider, builder: LLMProviderBuilder) -> None:
        """Register a builder for a new provider. Call once at app startup."""
        cls._builders[provider] = builder

    @classmethod
    def reset(cls) -> None:
        """Clear all cached instances and reset builders to the built-in defaults.
        Intended for use in tests only — call in a conftest autouse fixture to prevent
        instance bleed across test cases.
        """
        cls._instances.clear()
        cls._builders = {LLMProvider.OPENAI: OpenAIBuilder()}

    @classmethod
    def supported_providers(cls) -> list[LLMProvider]:
        """Return all currently registered providers. Used by UI to populate provider selectbox."""
        return list(cls._builders.keys())

    @classmethod
    def get(cls, config: LLMConfig) -> BaseChatModel:
        """Return a cached LLM instance for this exact config. Build once, reuse forever."""
        key = f"{config.provider}:{config.model_name}:{config.temperature}:{config.max_tokens}"
        if key not in cls._instances:
            if config.provider not in cls._builders:
                raise ValueError(
                    f"No builder registered for provider '{config.provider}'. "
                    f"Call LLMFactory.register() at startup to add support."
                )
            cls._instances[key] = cls._builders[config.provider].build(config)
        return cls._instances[key]
```

#### `src/utils/groq_builder.py`

```python
from langchain_core.language_models import BaseChatModel
from src.data import LLMConfig
from src.utils.provider_builder import LLMProviderBuilder


class GroqBuilder(LLMProviderBuilder):
    """Builds ChatGroq instances. Supported in v1.0 — plain text output requires no JSON schema mode."""

    def build(self, config: LLMConfig) -> BaseChatModel:
        """Construct a ChatGroq instance from the provided config."""
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
```

Both `OpenAIBuilder` and `GroqBuilder` are registered at app startup. Provider is selected via the `LLM_PROVIDER` env var.

### LLM Usage — One Config Only

There is exactly **one LLM config** in v1.0:

| Config | Purpose | Node | Structured output? |
|---|---|---|---|
| `SUPERVISOR_LLM_CONFIG` | Decomposes user query into a routing plan | `supervisor_node` | **No** — plain text; regex-parsed. Works with any provider. |

All other nodes (domain agents, web search, assemble) make no LLM calls.

### Settings Class

```python
# src/utils/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.data import LLMConfig, LLMProvider
from src.utils.constants import MAX_HISTORY_TURNS  # imported, not redefined


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    All API key fields have empty-string defaults — a missing key disables
    that agent/provider gracefully rather than crashing at startup.
    LLM_PROVIDER must be set to a registered provider or startup fails.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: LLMProvider = LLMProvider.OPENAI
    supervisor_model: str = "gpt-4o-mini"   # Used only by supervisor_node

    # LLM API keys — one per supported provider; only the active provider's key is required
    openai_api_key: str = ""
    groq_api_key: str = ""

    # News API keys — empty string disables that agent gracefully
    # Business + General/World share NEWSAPI_API_KEY (see News_API_Validation_Findings.pdf)
    guardian_api_key: str = ""
    newsapi_api_key: str = ""

    # Web search — one provider per run; only the active provider's key is required
    web_search_provider: WebSearchProvider = WebSearchProvider.SERPAPI
    serpapi_api_key: str = ""      # Required when web_search_provider=serpapi
    serperdev_api_key: str = ""    # Required when web_search_provider=serper

    max_articles_per_agent: int = 5
    max_history_turns: int = MAX_HISTORY_TURNS
    log_level: str = "INFO"


settings = Settings()  # Module-level singleton — loaded once at import time

# One LLM config — supervisor_node only
SUPERVISOR_LLM_CONFIG = LLMConfig(
    provider=settings.llm_provider,
    model_name=settings.supervisor_model,
    temperature=0.0,   # Deterministic routing — same query must produce the same plan
    max_tokens=300,    # Max output: 3 lines of "INTENT: X | QUERY: Y" ≈ 60 tokens each
)
```


### Temperature

`SUPERVISOR_LLM_CONFIG` uses `temperature=0.0` — routing must be deterministic. The same query should always produce the same plan.

Token budget detail is deferred to the Hardening phase (see Future Features).

---

## 5. Safety & Guardrails

### Input Validation

Handled by `UserQuery` at the system boundary — before any LLM call:
- Empty string after strip → rejected
- Exceeding 1000 characters → rejected by `max_length=1000`
- Special-character-only input → rejected by `sanitize_text` validator

### Routing Fallback

If the supervisor LLM returns output that doesn't match the `INTENT: X | QUERY: Y` pattern (zero regex matches), the system defaults to a single Web Search step using the full user query. This ensures the user always receives a response.

### Error Surface — What Users See

| Failure | User Message |
|---|---|
| API key missing | "The [Business/Sports/General] news service is not configured. Results from other sources are shown." |
| API rate limit | "News data is temporarily unavailable. Please try again in a few seconds." |
| All agents return empty | "No results found for your query. Try rephrasing or broadening the search." |
| LLM provider error | "I encountered an issue processing your request. Please try again." |
| Invalid query | "Please enter a valid query (1–1000 characters)." |
| Unregistered provider | "The selected LLM provider is not available. Please choose a supported provider." |

All errors logged at `ERROR` level with `session_id` and traceback. Users never see stack traces.

---

## 6. Streamlit UI Component Map

**Layout:** Streamlit’s layout is **desktop-first**. True responsive breakpoints (e.g. mobile) are limited by the framework; v1.0 targets a usable wide layout with `use_container_width` where applicable rather than a separate mobile design.

### Requirements alignment with requirements

The UI implements **fixed, hardcoded category controls** (not user-defined dynamic categories). Product rules:

- **Three wire-news lanes:** **Business**, **World**, **Sports** — map to the same routing/supervisor intents as **Business News**, **General News** (World / general headlines), and **Sports News** in §3. UI copy may say **“World”** while the supervisor still emits **“General News”** until labels are renamed end-to-end.
- **Topic verticals** (examples: **Cooking**, **Fashion** — exact list is product TBD) **do not** call NewsAPI/Guardian; they are satisfied via **web search** using the same prompt pattern: *“Show me the latest news on {X}.”*
- **Anything not offered as a chip** is entered in **`st.chat_input`** as a normal user message through the same graph.

**Tying UI → graph:** A chip selection (alone or with optional chat context) produces `UserQuery.text` (and optionally `UserQuery.categories` for the supervisor `categories_hint` and `filter_plan_to_user_categories`). Phase 5 implements the exact widget (e.g. `st.segmented_control`, `st.pills`, or `st.radio` + submit) — the table below remains the logical contract.

### Input Controls

| Widget | Variable | Maps To | Constraints |
|---|---|---|---|
| `st.chat_input` | `user_input` | `UserQuery.text` | max_chars=1000; carries typed prompts; may be empty if the UI submits only a composed chip prompt (implementation choice) |
| Category chips / multiselect (hardcoded options) | `categories` | `UserQuery.categories` | default `[]` (no restriction); **options = fixed set** aligned with requirements (Business, World, Sports, plus verticals such as Cooking, Fashion — **not** an open-ended enum dump) |
| `st.button("Get News")` | `get_news_clicked` | submit trigger | rendered only when a chip is selected; return value read directly — no `session_state` intermediary to prevent double-submission |
| `st.session_state.session_id` | `session_id` | `UserQuery.session_id` | UUID4 auto-generated on first load |

#### Session & conversation history management

`st.session_state.messages` stores a list of turn dicts. User turns: `{"role": "user", "content": str}`. Assistant turns: `{"role": "assistant", "response": SupervisorResponse}`.

`_build_history(messages)` converts this list to the `AgentState.conversation_history` format required by `supervisor_node`:
- User turns → `{"role": "user", "content": message["content"]}`
- Assistant turns → `{"role": "assistant", "content": "[Covered: query1, query2]"}` — serialized from `response.sections[*].query`

This compact representation passes what the supervisor needs (which topics were already covered) without embedding full article text. `trim_history()` then caps the list at `MAX_HISTORY_TURNS` before the LLM call.

#### World chip — helper copy (#3)

The chip label is **"World"** in UI copy. The supervisor emits **"General News"** internally. To prevent users from thinking this is a bug or duplicate of free-text chat, render a one-line `st.caption` beneath the chip group:

> *"Major global and political headlines."*

Do **not** surface the internal enum name (`General News`) in the UI. One plain sentence is sufficient.

#### LLM provider + model — sidebar placement (#4)

Move provider and model controls to `st.sidebar` inside `st.expander("Advanced", expanded=False)`. These are power-user settings that compete for attention with the category chips on the main surface.

- Surface the active values from `settings` (`LLM_PROVIDER`, `SUPERVISOR_MODEL`) as the default display in the expander.
- **v1.0 constraint:** `SUPERVISOR_LLM_CONFIG` is a module-level singleton. Sidebar selectboxes are **display-only** in v1.0 — they show the active config but do not override it mid-session. Label them `st.caption("Active config (set via .env)")`. Session-level provider overrides are a Future Feature.
- Persist sidebar choices in `st.session_state` so Streamlit reruns don't reset the displayed values.

| Widget | Variable | Maps To | Constraints |
|---|---|---|---|
| `st.selectbox` (sidebar) | `llm_provider` | display only — `settings.llm_provider` | options from `LLMFactory.supported_providers()`; labeled "display only" |
| `st.selectbox` (sidebar) | `model_name` | display only — `settings.supervisor_model` | dynamically populated per provider |

### Response Rendering

v1.0 does **not** render LLM-written prose per section — only **sub-query labels** and **article cards** (see `IntentSection`: `query` + `articles`).

#### Source badge above expander (#5)

Before each expander, render a compact one-line header that shows which backend ran and how many articles were returned. This gives scan-level provenance without requiring the user to open the expander.

Backend label mapping (from `IntentSection.agent`):

| `AgentName` | UI label |
|-------------|----------|
| `business_agent` | `NewsAPI · Business` |
| `sports_agent` | `Guardian · Sports` |
| `general_agent` | `NewsAPI · World` |
| `web_search_agent` | `Web Search` |

Use `section.agent` (not individual article providers) so the label is stable even when articles come from mixed sources. For Sports, individual article cards show `provider` = `guardian` or `espn` in the caption; the section badge always reads `Guardian · Sports`.

```
SupervisorResponse
└── sections[0]                   ← IntentSection for plan step 1
│   ├── st.markdown(f"**{section.query}**")   ← sub-query from routing plan (not LLM summary)
│   ├── st.caption(f"{BACKEND_LABEL[section.agent]} · {len(section.articles)} articles")  ← source badge (#5)
│   └── st.expander("Sources (N)")
│       └── for each article:
│           ├── [col1] st.image(image_url) if article.image_url else st.caption("No image")
│           └── [col2] st.markdown(f"**[{title}]({url})**\n{summary}")
│                      st.caption(f"{source_name} · {published_at:%b %d, %H:%M}")
│
└── sections[1]                   ← IntentSection for plan step 2
    ├── st.divider()              ← only rendered when len(sections) > 1
    ├── st.markdown(f"**{section.query}**")
    ├── st.caption(f"{BACKEND_LABEL[section.agent]} · {len(section.articles)} articles")  ← source badge (#5)
    └── st.expander("Sources (N)")
        └── ...same pattern...
```

### Article Card Component (`src/ui/article_card.py`)

```python
def render_article_card(article: NormalizedArticle) -> None:
    """Render one NormalizedArticle as a two-column card inside an expander."""
    col_img, col_text = st.columns([1, 3])
    with col_img:
        if article.image_url:
            st.image(article.image_url, use_container_width=True)
        else:
            st.caption("No image")
    with col_text:
        st.markdown(f"**[{article.title}]({article.url})**")
        st.write(article.summary)
        st.caption(
            f"{article.source_name} · "
            f"{article.published_at.strftime('%b %d, %H:%M UTC')} · "
            f"`{article.provider}`"
        )
```

### Full Widget Map

| Widget | Variable | Maps To | Notes |
|---|---|---|---|
| `st.chat_message("user")` | — | `UserQuery.text` | role="user" |
| `st.chat_message("assistant")` | — | All `SupervisorResponse.sections` | single bubble, N sections inside |
| `st.markdown` | `section.query` | `IntentSection.query` | Sub-query heading — v1.0 has no LLM prose field on sections |
| `st.caption` | `BACKEND_LABEL[section.agent]` | `IntentSection.agent` | Source badge: backend name + article count; rendered above expander (#5) |
| `st.caption` | — | World chip group | "Major global and political headlines." — one line beneath World chip (#3) |
| `st.expander` | — | `IntentSection.articles` | label shows article count; collapsed by default |
| `st.image` | `article.image_url` | `NormalizedArticle.image_url` | only called when non-null |
| `st.divider` | — | between sections | only when `len(sections) > 1` |
| `st.button("Get News")` | `get_news_clicked` | chip-only submit | rendered only when `selected_chip is not None`; return value read directly |
| `st.spinner` | — | LangGraph execution | shown during graph invoke |
| `st.warning` | — | `SupervisorResponse.fallback_used` | shown above sections if True |
| `st.sidebar` + `st.expander` | `llm_provider`, `model_name` | display only — `settings` values | "Advanced" expander, collapsed by default; display-only in v1.0 (#4) |

---

## 7. Project Structure

One public class per file throughout `src/`. `tests/` mirrors `src/` exactly.

```
newsgenie/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_news_agent.py        # Abstract base: fetch() interface
│   │   ├── business_agent.py         # Business news agent (NewsAPI everything)
│   │   ├── sports_agent.py           # Guardian + ESPN Scoreboard agent (always parallel)
│   │   └── general_news_agent.py     # NewsAPI agent
│   ├── api/
│   │   ├── __init__.py
│   │   ├── guardian_client.py        # HTTP client for Guardian (Sports)
│   │   ├── newsapi_client.py         # HTTP client for NewsAPI (Business + General/World)
│   │   └── espn_client.py            # ESPN Scoreboard client (no API key; US league scores)
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── supervisor_node.py        # supervisor_node + SUPERVISOR_PROMPT + INTENT_LABEL_TO_AGENT + category filter
│   │   ├── agent_nodes.py            # business_node, sports_node, general_node wrappers
│   │   ├── web_search_node.py        # web_search_node — wires SerpAPI LangGraph tool
│   │   ├── assemble_node.py          # assemble node — programmatic SupervisorResponse builder
│   │   ├── edges.py                  # route_from_plan conditional edge function
│   │   └── builder.py                # Assembles and returns the compiled LangGraph graph
│   ├── cli/
│   │   ├── __init__.py
│   │   └── cli.py                    # Click CLI entry point
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py                    # Streamlit application
│   │   └── article_card.py           # render_article_card()
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py              # MAX_HISTORY_TURNS — zero imports; safe cross-layer import
│   │   ├── provider_builder.py       # LLMProviderBuilder ABC only
│   │   ├── openai_builder.py         # OpenAIBuilder only
│   │   ├── groq_builder.py           # GroqBuilder only
│   │   ├── llm_factory.py            # LLMFactory only (registry + cache)
│   │   ├── normalizer.py             # Raw → NormalizedArticle: normalize_newsapi, normalize_guardian, normalize_espn
│   │   ├── history.py                # trim_history()
│   │   └── config.py                 # Settings class + SUPERVISOR_LLM_CONFIG
│   └── data/
│       ├── __init__.py               # Re-exports all public symbols
│       ├── enums.py                  # NewsCategory, LLMProvider, AgentName
│       ├── llm_config.py             # LLMConfig
│       ├── user_query.py             # UserQuery
│       ├── plan.py                   # PlanStep, RoutingPlan
│       ├── raw_articles.py           # GuardianRawArticle, NewsAPIRawArticle (MarketAux not implemented)
│       ├── normalized_article.py     # NormalizedArticle
│       ├── agent_result.py           # NewsAgentResult
│       ├── response.py               # IntentSection, SupervisorResponse
│       └── state.py                  # AgentState (with reducer annotation)
├── tests/
│   ├── conftest.py
│   ├── agents/
│   │   ├── test_base_news_agent.py
│   │   ├── test_business_agent.py
│   │   ├── test_sports_agent.py
│   │   └── test_general_news_agent.py
│   ├── api/
│   │   ├── test_guardian_client.py
│   │   ├── test_newsapi_client.py
│   │   └── test_espn_client.py
│   ├── graph/
│   │   ├── test_supervisor_node.py
│   │   ├── test_agent_nodes.py
│   │   ├── test_web_search_node.py
│   │   ├── test_assemble_node.py
│   │   ├── test_edges.py
│   │   └── test_builder.py
│   ├── cli/
│   │   └── test_cli.py
│   ├── utils/
│   │   ├── test_constants.py
│   │   ├── test_config.py
│   │   ├── test_provider_builder.py
│   │   ├── test_openai_builder.py
│   │   ├── test_groq_builder.py
│   │   ├── test_llm_factory.py
│   │   ├── test_normalizer.py
│   │   └── test_history.py
│   └── data/
│       ├── test_enums.py
│       ├── test_user_query.py
│       ├── test_plan.py
│       ├── test_normalized_article.py
│       ├── test_agent_result.py
│       ├── test_response.py
│       └── test_state.py
├── README.md
├── pyproject.toml
├── .gitignore
└── .env.example
```

---

## 8. Poetry Dependencies

### `pyproject.toml`

```toml
[tool.poetry]
name = "newsgenie"
version = "0.1.0"
description = "Multi-agent real-time news assistant powered by LangGraph"
authors = ["Tim Hazed"]
readme = "README.md"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = "^3.13"
langchain = "^0.3"
langchain-core = "^0.3"
langchain-openai = "^0.3"
langchain-groq = "^0.2"
langchain-community = "^0.3"   # SerpAPI tool integration
langgraph = "^0.3"
pydantic = "^2.7"
pydantic-settings = "^2.3"
httpx = "^0.27"
tenacity = "^9.0"
streamlit = "^1.35"
click = "^8.1"
rich = "^13.7"
python-dateutil = "^2.9"
google-search-results = "^2.4"  # SerpAPI Python client

[tool.poetry.group.dev.dependencies]
pytest = "^8.2"
pytest-asyncio = "^0.23"
pytest-cov = "^5.0"
respx = "^0.21"
ruff = "^0.5"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src --cov-report=term-missing --cov-fail-under=80 -v"
asyncio_mode = "auto"

[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B"]
ignore = ["E501"]

[tool.coverage.report]
fail_under = 80
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:"]
```

### `.env.example`

```bash
# LLM Provider — "openai" or "groq" (both supported in v1.0)
LLM_PROVIDER=openai

# Model for supervisor_node (the only LLM call in the system)
SUPERVISOR_MODEL=gpt-4o-mini    # or: llama-3.1-70b-versatile (for Groq)

# LLM API keys — provide the key for whichever provider is selected above
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# News API Keys — missing key disables that agent gracefully
# Business & General/World → NewsAPI /v2/everything | Sports → Guardian
GUARDIAN_API_KEY=...
NEWSAPI_API_KEY=...

# Web search — one provider per run (like LLM_PROVIDER); only that provider’s key is required
WEB_SEARCH_PROVIDER=serpapi
# WEB_SEARCH_PROVIDER=serper
SERPAPI_API_KEY=...
SERPERDEV_API_KEY=...

# Tuning
MAX_ARTICLES_PER_AGENT=5
MAX_HISTORY_TURNS=10
LOG_LEVEL=INFO
```

---
