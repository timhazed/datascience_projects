"""Centralised ``ChatPromptTemplate`` definitions for healthcare LangGraph chains.

Templates live here only; sibling modules (``history_chain``, ``planner_chain``, etc.)
compose them with LLMs and parsers. Do not add Runnable or LLM construction here.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

_HISTORY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clinical records assistant. Using only the retrieved patient records "
        "provided below, answer the query accurately. Do not invent information. "
        "If a field is unknown, say \"not recorded\".",
    ),
    # Patient record chunks are placed in the human turn (not the system prompt) to
    # limit the impact of adversarial content that may have been embedded in PDF/xlsx
    # data loaded into the vector store (prompt injection defense).
    ("human", "Retrieved records:\n{retrieved_chunks}\n\nQuery: {query}"),
])

_GUARD_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical assistant intake filter.\n"
        "Classify the following user message.\n"
        "Allowed topics: patient medical history, appointment scheduling, "
        "medical information queries, record updates, treatment summaries.\n"
        "Reply SAFE if on-topic. Reply UNSAFE if off-topic. No other output.",
    ),
    ("human", "{user_query}"),
])

_MEMORY_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clinical memory assistant. Compress the prior summary and recent "
        "conversation turns into a single concise summary of the patient's stated concerns, "
        "key facts discussed, and any actions taken. Be brief and clinically accurate. "
        "Maximum {max_chars} characters.",
    ),
    ("human", "Prior summary:\n{prior_summary}\n\nRecent turns:\n{recent_turns}"),
])

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical assistant planner. Decompose the user request into an ordered list "
        "of atomic sub-goals. Each sub-goal has a task name and a parameters dict.\n\n"
        "AVAILABLE TASKS — use ONLY these parameter field names:\n\n"
        "resolve_patient\n"
        '  {{"patient_name": "<full name or null if patient is anonymous>", "phone": null}}\n\n'
        "retrieve_history\n"
        '  {{"patient_id": "", "query": "<optional specific question about the history>"}}\n\n'
        "update_history\n"
        '  {{"patient_id": "", '
        '"field": "<one of: conditions, medications, allergies, notes, summary>", '
        '"value": "<new text to write>", "operation": "append"}}\n\n'
        "book_appointment\n"
        '  {{"patient_id": "", "specialty": "<medical specialty>", '
        '"preferred_date": null, "urgency": "routine", "reason": "<brief reason>"}}\n\n'
        "search_disease\n"
        '  {{"query": "<clinical search query — disease name + topic>", "max_results": 5}}\n\n'
        "RULES:\n"
        "- patient_id is always set to empty string '' in the plan; it is filled from state at "
        "runtime after resolve_patient runs.\n"
        "- If the user mentions a patient by name, always start with resolve_patient.\n"
        "- If no patient name is mentioned, omit resolve_patient AND also omit retrieve_history, "
        "update_history, and book_appointment — these tasks all require a resolved patient and "
        "will always fail without one.\n"
        "- Do not invent parameter fields not listed above.\n"
        "- Keep the reasoning field to one short sentence (~20 words). Output budget must "
        "fit the full sub_goals JSON.\n"
        "- CRITICAL: task names must be EXACTLY one of the five listed above "
        "(resolve_patient, retrieve_history, update_history, book_appointment, "
        "search_disease). Any other value is an error.",
    ),
    # patient_context is runtime-resolved data — placed in the human turn so a
    # crafted patient name cannot inject instructions into the privileged system role.
    (
        "human",
        "Patient context (if already resolved): {patient_context}\n\n"
        "Conversation memory (bounded, may be empty):\n{memory_context}\n\n"
        "Request: {user_query}",
    ),
])

_SEARCH_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical information specialist. Synthesize the web search results "
        "provided below into a clear, accurate clinical summary of 2-4 paragraphs. "
        "Cite sources inline as [1], [2], etc. "
        "Only include information from the provided sources — do not draw on prior knowledge.",
    ),
    # Search results are placed in the human turn (not the system prompt) to limit
    # the impact of adversarial content in web-fetched snippets (prompt injection defence).
    ("human", "Search results:\n{search_results}\n\nQuery: {query}"),
])

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a healthcare assistant. Synthesize the results of multiple completed tasks "
        "into a single cohesive response for the user. "
        "Be concise, accurate, and clinically appropriate. "
        "Include all relevant findings. If any tasks failed, note them clearly.\n\n"
        "Completed task results (JSON):\n{completed_tasks}\n\n"
        "Prior conversation with this patient (bounded excerpt from stored chat; may be empty):\n"
        "{conversation_memory}\n\n"
        "Rules:\n"
        "- If the user asks about prior conversation (what was said before, recap, "
        "your last answer), use the prior-conversation excerpt above; "
        "quote or paraphrase accurately.\n"
        "- For clinical or chart questions, prioritize findings from the completed task results, "
        "especially successful retrieve_history output.\n"
        "- If the excerpt and task results conflict, state both and attribute sources "
        "(e.g. from our earlier messages vs from the retrieved record).",
    ),
    ("human", "{user_query}"),
])

__all__ = [
    "_GUARD_PROMPT",
    "_HISTORY_PROMPT",
    "_MEMORY_SUMMARY_PROMPT",
    "_PLANNER_PROMPT",
    "_SEARCH_SUMMARY_PROMPT",
    "_SUMMARY_PROMPT",
]