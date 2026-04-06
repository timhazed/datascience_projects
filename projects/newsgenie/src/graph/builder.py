from langgraph.graph import END, StateGraph

from src.data import AgentName, AgentState
from src.graph.agent_nodes import business_node, general_node, sports_node
from src.graph.assemble_node import assemble
from src.graph.edges import route_from_plan
from src.graph.supervisor_node import supervisor_node
from src.graph.web_search_node import web_search_node


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
