from src.data import AgentName, AgentState


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
