"""LangGraph node callables (intent, planner, tools, summarizer).

Each module exposes a ``make_*_node(...)`` factory that closes over chains and DB
handles. Import concrete factories from their modules, e.g.
``from src.agents.planner_node import make_planner_node`` — this package does not
re-export them to avoid a heavy import graph at ``import src.agents``.
"""
