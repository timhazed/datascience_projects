"""Chain for query result synthesis — Spec §5, §7 (snippet_summarizer node).

Produces a free-text narrative answer from ChromaDB query results using the
Extract→Contrast→Synthesize prompt structure validated in Appendix D. Returns a plain
string via StrOutputParser — no structured output required for conversational answers.
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a developer knowledge assistant with access to a semantic memory of indexed "
    "code repositories. Answer the developer's query by synthesizing the provided code "
    "snippets and their intent summaries.\n\n"
    "Use the Extract→Contrast→Synthesize method:\n"
    "1. Extract: identify the key patterns in each result.\n"
    "2. Contrast: note similarities and differences across results.\n"
    "3. Synthesize: produce a coherent, actionable answer referencing specific files.\n\n"
    "IMPORTANT — code output rule: if the query asks for code (e.g. 'show me', 'how do I', "
    "'give me an example', 'create', 'write'), reproduce the most relevant code snippet "
    "verbatim inside a fenced code block (``` language). Do NOT paraphrase or describe "
    "code when the user asked to see it.\n\n"
    "Cite file paths when relevant. Do not hallucinate results not present in the context."
)

_HUMAN = (
    "Query: {query}\n\n"
    "Results:\n{results}\n\n"
    "Answer the query. If the query asks for code, include a fenced code block with the "
    "most relevant snippet. Otherwise, synthesize a concise answer (3–6 sentences) "
    "grounded in the results above."
)

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def build_snippet_summary_chain(llm: ChatOllama) -> Runnable:
    """Build the snippet summarization chain — called once at graph construction time.

    The returned chain accepts {"query": str, "results": str} and returns a plain
    text answer synthesizing the ChromaDB results into a developer-facing response.

    Args:
        llm: ChatOllama instance constructed at server startup. Temperature 0.1 per §5.

    Returns:
        A LangChain Runnable: dict → str.
    """
    # StrOutputParser: free-text synthesis; no structured schema needed here.
    # External data (query, results) are in the human turn — never the system prompt.
    return _PROMPT | llm | StrOutputParser()
