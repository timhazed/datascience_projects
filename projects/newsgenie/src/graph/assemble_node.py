from src.data import AgentState
from src.data.response import IntentSection, SupervisorResponse


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
