"""chunk_dispatcher LangGraph node — Spec §2.1, §4.

Fan-out dispatcher: signals intent to the graph that parsed_chunks should be dispatched
to parallel intent_summarizer invocations.  The actual Send fan-out is performed by the
conditional edge routing function route_after_dispatch() in build_sync_graph() — LangGraph
1.x requires Send objects to come from edge routing functions, not from node return values.

Empty-chunks guard: if parsed_chunks is empty (empty repo or all files quarantined),
this node returns a trace entry so the conditional edge routes to chroma_upsert.
"""

import logging
from collections.abc import Callable

from src.models.trace_event import TraceEvent, TraceEventType

logger = logging.getLogger(__name__)


def make_chunk_dispatcher_node() -> Callable[[dict], dict]:
    """Return a chunk_dispatcher node (no injected dependencies).

    Returns:
        LangGraph node function that appends a trace entry. The Send fan-out to
        intent_summarizer is handled by the conditional edge routing function in
        build_sync_graph() which reads parsed_chunks from state and emits Send objects.
    """

    def chunk_dispatcher(state: dict) -> dict:
        """Log dispatch intent; actual Send fan-out is performed by the edge routing function.

        LangGraph 1.x requires Send objects to be returned from conditional edge routing
        functions, not from node functions.  This node simply records the chunk count in
        the trace so the pipeline log is complete.
        """
        chunks = state.get("parsed_chunks", [])

        if not chunks:
            logger.info("chunk_dispatcher: no chunks — routing to chroma_upsert")
            return {
                "trace": ["chunk_dispatcher: no chunks — routing to chroma_upsert"],
                "trace_events": [
                    TraceEvent(node="chunk_dispatcher", event=TraceEventType.SKIP, count=0)
                ],
            }

        logger.info("chunk_dispatcher: %d chunks ready for intent_summarizer", len(chunks))
        return {
            "trace": [f"chunk_dispatcher: dispatching {len(chunks)} chunks"],
            "trace_events": [
                TraceEvent(node="chunk_dispatcher", event=TraceEventType.DISPATCHED, count=len(chunks))
            ],
        }

    return chunk_dispatcher
