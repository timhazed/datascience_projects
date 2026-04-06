import re

from src.data import AgentName, AgentState, NewsCategory
from src.data.enums import LLMProvider
from src.data.plan import PlanStep, RoutingPlan
from src.utils.config import SUPERVISOR_LLM_CONFIG
from src.utils.groq_builder import GroqBuilder
from src.utils.history import trim_history
from src.utils.llm_factory import LLMFactory

# Register Groq builder at module load so both providers are available.
LLMFactory.register(LLMProvider.GROQ, GroqBuilder())

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

# Maps LLM intent labels to AgentName enum members (never use raw strings for PlanStep.agent).
INTENT_LABEL_TO_AGENT: dict[str, AgentName] = {
    "Business News": AgentName.BUSINESS,
    "Sports News": AgentName.SPORTS,
    "General News": AgentName.GENERAL,
    "Web Search": AgentName.WEB_SEARCH,
}


def _format_history(history: list[dict]) -> str:
    """Convert trimmed history list to a plain-text block for the supervisor prompt.

    Empty list returns "No prior conversation." so the {history} slot is never blank.
    Non-empty list renders one message per line as "ROLE: content".
    """
    if not history:
        return "No prior conversation."
    return "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history)


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
