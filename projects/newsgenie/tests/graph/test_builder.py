"""Tests for src/graph/builder.py — build_graph."""




class TestBuildGraph:
    def test_graph_compiles_without_error(self):
        """build_graph() returns a compiled graph without raising."""
        from src.graph.builder import build_graph

        graph = build_graph()
        assert graph is not None

    def test_all_four_agent_nodes_registered(self):
        """The compiled graph must contain all four agent nodes."""
        from src.graph.builder import build_graph

        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "business_agent" in node_names
        assert "sports_agent" in node_names
        assert "general_agent" in node_names
        assert "web_search_agent" in node_names

    def test_assemble_node_registered(self):
        """The assemble node must be present in the compiled graph."""
        from src.graph.builder import build_graph

        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "assemble" in node_names
