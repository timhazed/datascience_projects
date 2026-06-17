"""Shared aggregation utility for ChromaDB result dicts.

Extracts the common counter-based loop used by tendency_scanner, persona_loader,
and skills_aggregator. All three iterate ChromaDB result dicts to count semantic_type
labels, tech_stack items, and collect intent summaries — this is the canonical
implementation to prevent divergence when the metadata schema evolves.

Works with both plain result dicts (weight defaults to 1.0) and dicts enriched by
apply_temporal_weight() (weight key present). The round() call normalises float
weights into integer counts suitable for Counter arithmetic.
"""

from collections import Counter


def aggregate_chroma_docs(
    docs: list[dict],
) -> tuple[Counter[str], Counter[str], list[tuple[float, str]]]:
    """Extract weighted Counter objects and pattern list from ChromaDB result dicts.

    Iterates the metadata of each doc, accumulating:
    - semantic_type frequency (weighted)
    - tech_stack item frequency (weighted)
    - intent_summary strings paired with their weight for downstream sorting

    Compatible with plain docs (no "weight" key — defaults to 1.0) and docs
    enriched by apply_temporal_weight() (weight = 3.0 for recent docs).

    Args:
        docs: List of ChromaDB result dicts, each with at minimum a "metadata" key
            containing "semantic_type", "tech_stack", and "intent_summary" sub-keys.
            An optional "weight" key (float) controls the frequency contribution
            of each document to the counters.

    Returns:
        A 3-tuple:
        - semantic_counter: Counter[str] — weighted count per semantic_type label.
        - tech_counter: Counter[str] — weighted count per tech_stack item.
        - weighted_patterns: list[(weight, intent_summary)] sorted by the caller;
            empty intent_summary strings are excluded.
    """
    semantic_counter: Counter[str] = Counter()
    tech_counter: Counter[str] = Counter()
    patterns: list[tuple[float, str]] = []

    for doc in docs:
        weight = doc.get("weight", 1.0)
        # round() converts temporal weights (e.g. 3.0) to integer increments for Counter
        w = round(weight)
        meta = doc.get("metadata", {})

        stype = meta.get("semantic_type", "Logic")
        semantic_counter[stype] += w

        for tech in meta.get("tech_stack", []):
            tech_counter[tech] += w

        summary = meta.get("intent_summary", "")
        if summary:
            patterns.append((weight, summary))

    return semantic_counter, tech_counter, patterns
