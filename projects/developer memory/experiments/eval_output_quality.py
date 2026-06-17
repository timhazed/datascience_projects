"""
experiments/eval_output_quality.py

Measures structured-output compliance rate and rubric-scored quality for
gemma4:e4b vs gemma4:26b across the five chain types used in Developer Memory.

This experiment answers the model-selection question:
  "Is the quality improvement from 26b large enough to justify its higher memory
   footprint and slower first-token latency vs e4b?"

Decision rule (from §11 exit gate):
  - If e4b compliance rate is within 2pp of 26b  → e4b wins on resource grounds
  - If the compliance gap exceeds 5pp             → reconsider 26b as default

Metrics collected per model, per chain type:
  - Structured output compliance rate (% of runs where Pydantic validates on attempt 1)
  - Retry rate (mean number of invoke attempts per case)
  - Rubric pass rate — LLM-as-judge for persona_synthesizer / snippet_summarizer;
    heuristic keyword scorer for all other chains
  - Mean generation time (seconds, wall-clock)
  - Mean first-token latency (ms) — where streaming allows measurement

Rubric scoring notes:
  - persona_synthesizer and snippet_summarizer require semantic reasoning to evaluate
    (e.g. "synthesises patterns into a named archetype"). The heuristic keyword scorer
    cannot assess these criteria reliably — LLM-as-judge (Groq gpt-oss-120b) is used
    for these chains when GROQ_API_KEY is available.
  - intent_summarizer, coaching_analyzer, skills_synthesizer use the heuristic scorer;
    their rubric criteria are structural and keyword-checkable.

Chain types covered (matches src/chains/ builders):
  - intent_summarizer   → structured output (SummarizedChunk schema)
  - snippet_summarizer  → free-text (no schema; LLM-judge rubric scoring)
  - persona_synthesizer → structured output (PersonaProfile schema; LLM-judge rubric)
  - coaching_analyzer   → structured output (AnalysisResult schema)
  - skills_synthesizer  → free-text (no schema; heuristic rubric)

Usage:
    # Run full eval against both models (default):
    python experiments/eval_output_quality.py

    # Single model:
    python experiments/eval_output_quality.py --models gemma4:e4b

    # Custom corpus file:
    python experiments/eval_output_quality.py --corpus experiments/my_corpus.jsonl

    # Increase runs per case (default 5):
    python experiments/eval_output_quality.py --runs 10

    # Skip rubric scoring (compliance-only, faster):
    python experiments/eval_output_quality.py --no-rubric

    # Skip LLM-as-judge even if GROQ_API_KEY is set:
    python experiments/eval_output_quality.py --no-judge

    # Override Groq judge model:
    python experiments/eval_output_quality.py --judge-model openai/gpt-oss-120b

Output:
    experiments/results/quality_<timestamp>.json
    experiments/results/quality_<timestamp>.csv
    (A decision summary is printed to stdout at the end.)
"""

import argparse
import csv
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODELS: list[str] = ["gemma4:e4b", "gemma4:26b"]
DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"
DEFAULT_RUNS_PER_CASE: int = 5
DEFAULT_JUDGE_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_BASE: str = "https://api.groq.com/openai/v1"

# All chains use LLM-as-judge. The heuristic keyword scorer cannot reliably evaluate
# semantic criteria, negative assertions ("does not incorrectly tag"), numeric thresholds,
# or multi-## section criteria — all of which appear across the corpus.
# Heuristic scoring is retained only as a fallback when GROQ_API_KEY is unavailable.
LLM_JUDGE_CHAINS: frozenset[str] = frozenset({
    "intent_summarizer",
    "snippet_summarizer",
    "persona_synthesizer",
    "coaching_analyzer",
    "skills_synthesizer",
})

RESULTS_DIR = Path(__file__).parent / "results"
CORPUS_PATH = Path(__file__).parent / "inputs" / "eval_corpus.jsonl"

# Pydantic-equivalent JSON schemas for structured-output chains.
# These mirror the field contracts in src/models/ without requiring the src package
# to be importable from the experiments directory.
STRUCTURED_SCHEMAS: dict[str, dict[str, Any]] = {
    "SummarizedChunk": {
        "required_fields": ["intent_summary", "tech_stack", "semantic_type"],
        "field_types": {
            "intent_summary": str,
            "tech_stack": list,
            "semantic_type": str,
        },
        "enum_fields": {
            "semantic_type": {"Logic", "Config", "Boilerplate", "Interface"},
        },
    },
    "PersonaProfile": {
        "required_fields": ["dominant_patterns", "style_summary", "deviation_tendencies"],
        "field_types": {
            "dominant_patterns": list,
            "style_summary": str,
            "deviation_tendencies": list,
        },
        "enum_fields": {},
    },
    "AnalysisResult": {
        "required_fields": ["deviation_score", "rationale", "patterns_matched"],
        "field_types": {
            "deviation_score": (int, float),
            "rationale": str,
            "patterns_matched": list,
        },
        "enum_fields": {},
        "range_fields": {
            "deviation_score": (0.0, 1.0),
        },
    },
}

# ── Prompt templates ───────────────────────────────────────────────────────────
# Each system prompt includes one concrete few-shot example to anchor the model
# on the expected output format and quality bar before the actual input is shown.
# This is especially important for coaching_analyzer, where compact JSON output
# prevents num_predict truncation mid-structure.

SYSTEM_PROMPTS: dict[str, str] = {
    "intent_summarizer": (
        "You are a code analyst. For any source file, extract exactly three fields:\n"
        "  intent_summary : 1–3 sentences explaining WHY this code exists and what problem it solves.\n"
        "  tech_stack     : list of technology names present (frameworks, libraries, patterns).\n"
        "  semantic_type  : exactly one of: Logic | Config | Boilerplate | Interface\n\n"
        "Output ONLY valid JSON — no prose, no markdown code fences, no commentary.\n\n"
        "Example\n"
        "-------\n"
        "Input:\n"
        "File: src/cache/lru.py\n"
        "```\n"
        "from collections import OrderedDict\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity: int):\n"
        "        self._cache: OrderedDict = OrderedDict()\n"
        "        self._cap = capacity\n"
        "    def get(self, key: str) -> str | None:\n"
        "        if key not in self._cache: return None\n"
        "        self._cache.move_to_end(key)\n"
        "        return self._cache[key]\n"
        "    def put(self, key: str, value: str) -> None:\n"
        "        self._cache[key] = value\n"
        "        self._cache.move_to_end(key)\n"
        "        if len(self._cache) > self._cap:\n"
        "            self._cache.popitem(last=False)\n"
        "```\n"
        "Output:\n"
        '{"intent_summary": "Implements an LRU eviction cache using an ordered dict so that '
        "the least-recently-used entry is evicted first when capacity is exceeded, preventing "
        'unbounded memory growth in hot code paths.", '
        '"tech_stack": ["OrderedDict", "LRU eviction", "in-memory cache"], '
        '"semantic_type": "Logic"}\n'
        "-------\n"
        "Now extract the same three fields for the file provided."
    ),

    "snippet_summarizer": (
        "You are a developer assistant. Answer the developer's query using only the provided "
        "code snippets. Be concise and actionable — lead with the answer, not preamble. "
        "Do not expose stack traces, internal IDs, or raw exception messages.\n\n"
        "Example\n"
        "-------\n"
        "Query: How does the system prevent duplicate ChromaDB inserts?\n\n"
        "Relevant snippets:\n"
        "[src/db/chroma_upsert.py]\n"
        "content_hash = sha256(chunk.content.encode()).hexdigest()\n"
        "existing = collection.get(ids=[content_hash])\n"
        "if existing['ids']: return UpsertResult(status='duplicate', id=content_hash)\n"
        "collection.add(ids=[content_hash], documents=[chunk.content], metadatas=[meta])\n\n"
        "Answer: The system computes a SHA-256 hash of each chunk's content before upserting. "
        "If ChromaDB already holds that ID the insert is skipped and `UpsertResult(status='duplicate')` "
        "is returned — no duplicate document is created.\n"
        "-------\n"
        "Now answer the developer's query using the snippets provided."
    ),

    "persona_synthesizer": (
        "You are a developer experience analyst. Synthesise a professional persona profile "
        "from developer tendency data. Capture the developer's dominant coding philosophy, "
        "not just surface patterns.\n\n"
        "Output ONLY valid JSON — no prose, no markdown code fences, no commentary.\n\n"
        "Example\n"
        "-------\n"
        "Input:\n"
        '{"semantic_types": {"Logic": 82, "Config": 10, "Boilerplate": 8}, '
        '"style_signals": ["uses dataclasses", "avoids deep inheritance", "writes module-level docstrings"], '
        '"recent_patterns": ["type-annotates all public APIs", "uses Protocol over ABC"]}\n'
        "Output:\n"
        '{"dominant_patterns": ["composition over inheritance", "data-class-first modelling", '
        '"typed public APIs", "Protocol-based structural subtyping"], '
        '"style_summary": "A pragmatic systems developer who favours flat, composable data '
        "structures over deep class hierarchies and treats the public API surface as a contract "
        'to be typed and documented before implementation.", '
        '"deviation_tendencies": ["may skip type annotations on private/internal helpers"]}\n'
        "-------\n"
        "Synthesise the same three fields for the tendency data provided."
    ),

    "coaching_analyzer": (
        "You are a code coach. Analyse a git diff against the developer persona and output a "
        "deviation assessment. Be concise — the rationale must be 1–2 sentences maximum.\n\n"
        "deviation_score: 0.0 = perfectly aligned with persona, 1.0 = complete deviation.\n"
        "Alert threshold: > 0.65.\n\n"
        "Output ONLY valid JSON — no prose, no markdown code fences, no commentary.\n\n"
        "Example\n"
        "-------\n"
        "Git diff:\n"
        "```\n"
        "-def get_user(id):\n"
        '-    return db.execute(f"SELECT * FROM users WHERE id={id}")\n'
        "+def get_user(user_id: int) -> User:\n"
        '+    return db.execute("SELECT * FROM users WHERE id=?", (user_id,))\n'
        "```\n"
        'Persona: {"dominant_patterns": ["parameterised queries", "typed function signatures"], '
        '"style_summary": "Security-conscious developer who avoids SQL injection vectors."}\n'
        "Output:\n"
        '{"deviation_score": 0.05, '
        '"rationale": "Diff fixes an injection vulnerability with parameterised queries and adds '
        'a type annotation — both core to the persona\'s established patterns.", '
        '"patterns_matched": ["parameterised queries", "typed function signatures"]}\n'
        "-------\n"
        "Now analyse the git diff provided against the developer persona."
    ),

    "skills_synthesizer": (
        "You are a technical writer. Given skills aggregation data, generate a PROJECT_SKILLS.md "
        "document with exactly these sections in this order:\n"
        "## Tech Stack\n"
        "## Patterns\n"
        "## Tendencies\n\n"
        "Each section: 3–6 bullet points, professional and actionable. "
        "Each bullet must be a full sentence that explains not just WHAT the technology or pattern "
        "is, but WHY it exists in this codebase and what a new contributor should know about it. "
        "Your response must be at least 400 words total across all three sections — write substantively, not tersely. "
        "Write for a new contributor who needs to understand this codebase quickly.\n\n"
        "Example section format:\n"
        "## Tech Stack\n"
        "- **LangGraph** — orchestrates multi-step LLM pipelines as typed state machines\n"
        "- **ChromaDB** — persistent vector store for semantic code search\n\n"
        "Now generate the full document from the aggregation data provided."
    ),
}


def build_human_prompt(chain_type: str, inputs: dict) -> str:
    """Convert a corpus entry's inputs dict into the human-turn prompt text."""
    if chain_type == "intent_summarizer":
        return f"File: {inputs['path']}\n\n```\n{inputs['content']}\n```"

    if chain_type == "snippet_summarizer":
        snippets = "\n\n".join(
            f"[{r['metadata']['file_path']}]\n{r['content']}"
            for r in inputs["raw_results"]
        )
        return f"Query: {inputs['query']}\n\nRelevant snippets:\n{snippets}"

    if chain_type == "persona_synthesizer":
        return f"Tendency data:\n{json.dumps(inputs['tendency_data'], indent=2)}"

    if chain_type == "coaching_analyzer":
        persona_str = json.dumps(inputs.get("persona_context", {}), indent=2)
        return f"Git diff:\n```\n{inputs['diff_text']}\n```\n\nDeveloper persona:\n{persona_str}"

    if chain_type == "skills_synthesizer":
        return f"Skills aggregation data:\n{json.dumps(inputs['skills_data'], indent=2)}"

    raise ValueError(f"Unknown chain_type: {chain_type!r}")


# ── Ollama inference ───────────────────────────────────────────────────────────

def run_single_inference(
    host: str,
    model: str,
    system_prompt: str,
    human_prompt: str,
    num_predict: int = 1024,
    timeout: float = 120.0,
) -> dict:
    """
    Run one non-streaming inference call; return response text + timing.

    Returns:
        text            : full response string
        first_token_ms  : wall-clock ms to first byte of response (streaming estimate)
        elapsed_s       : total wall-clock seconds
        eval_count      : tokens generated (from Ollama stats)
        tokens_per_sec  : eval_count / eval_duration
    """
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": human_prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }
    start = time.perf_counter()
    resp = httpx.post(f"{host}/api/generate", json=payload, timeout=timeout)
    elapsed_s = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

    return {
        "text": data.get("response", ""),
        "elapsed_s": round(elapsed_s, 3),
        "eval_count": eval_count,
        "tokens_per_sec": round(tps, 2),
    }


# ── Structured output validation ───────────────────────────────────────────────

def validate_structured_output(text: str, schema_name: str) -> tuple[bool, str]:
    """
    Validate LLM response text against the named schema.

    Returns (is_valid, failure_reason). Attempts JSON extraction from markdown
    code fences and thinking blocks before validation to handle common model
    formatting habits (including gemma4:26b <think>...</think> preambles).
    """
    schema = STRUCTURED_SCHEMAS.get(schema_name)
    if schema is None:
        return True, ""  # no schema = free-text chain; always passes structural check

    clean = text.strip()

    # Strip <think>...</think> reasoning blocks emitted by some models before output
    import re as _re
    clean = _re.sub(r"<think>.*?</think>", "", clean, flags=_re.DOTALL).strip()

    # Strip markdown code fences if present
    if clean.startswith("```"):
        lines = clean.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        clean = inner.strip()

    # If there's still preamble before the JSON object, seek to the first '{'
    if clean and clean[0] != "{":
        brace_idx = clean.find("{")
        if brace_idx != -1:
            clean = clean[brace_idx:]

    try:
        obj = json.loads(clean)
    except json.JSONDecodeError as exc:
        print(f"    [debug] parse fail — raw text prefix: {repr(text[:200])}", flush=True)
        return False, f"JSON parse error: {exc}"

    if not isinstance(obj, dict):
        return False, f"Expected JSON object, got {type(obj).__name__}"

    for field in schema["required_fields"]:
        if field not in obj:
            return False, f"Missing required field: {field!r}"

        expected_type = schema["field_types"][field]
        if not isinstance(obj[field], expected_type):
            return False, (
                f"Field {field!r}: expected {expected_type}, "
                f"got {type(obj[field]).__name__}"
            )

    for field, allowed in schema.get("enum_fields", {}).items():
        if obj.get(field) not in allowed:
            return False, f"Field {field!r}: {obj.get(field)!r} not in {allowed}"

    for field, (lo, hi) in schema.get("range_fields", {}).items():
        val = obj.get(field)
        if val is not None and not (lo <= float(val) <= hi):
            return False, f"Field {field!r}: {val} out of range [{lo}, {hi}]"

    return True, ""


# ── Rubric scoring — heuristic ─────────────────────────────────────────────────

def score_rubric(text: str, rubric: list[str]) -> tuple[int, int, list[str]]:
    """
    Score the LLM output against a list of rubric criteria using keyword heuristics.

    Lightweight scorer — not LLM-as-judge. Reliable for structural criteria
    (section headers, required keywords, word counts). Not reliable for semantic
    criteria like "synthesises patterns into a named archetype" — use
    score_rubric_llm_judge() for those (LLM_JUDGE_CHAINS).

    Returns (passed, total, failed_criteria).
    """
    text_lower = text.lower()
    passed = 0
    failed: list[str] = []

    for criterion in rubric:
        criterion_lower = criterion.lower()

        # Extract the key signal terms from the criterion text.
        signal_words = _extract_signal_words(criterion_lower)
        matches = sum(1 for w in signal_words if w in text_lower)

        # Minimum word count checks
        if "minimum" in criterion_lower and "words" in criterion_lower:
            import re
            count_match = re.search(r"minimum\s+(\d+)\s+words", criterion_lower)
            if count_match:
                min_words = int(count_match.group(1))
                actual_words = len(text.split())
                if actual_words >= min_words:
                    passed += 1
                else:
                    failed.append(f"{criterion} [got {actual_words} words, need {min_words}]")
                continue

        # Section header checks
        if "##" in criterion:
            # Extract just the heading name — stop at the first lowercase word so that
            # "output contains ## Tech Stack section listing ..." → "Tech Stack"
            raw_after = criterion.split("##")[1].strip()
            header_words = []
            for word in raw_after.split():
                if word[0].isupper():
                    header_words.append(word)
                else:
                    break
            header = " ".join(header_words) if header_words else raw_after.split()[0]
            if f"## {header.lower()}" in text_lower or f"#{header.lower()}" in text_lower:
                passed += 1
            else:
                failed.append(criterion)
            continue

        # General: need at least 2 signal words from the criterion
        threshold = max(1, len(signal_words) // 2)
        if matches >= threshold:
            passed += 1
        else:
            failed.append(criterion)

    return passed, len(rubric), failed


def _extract_signal_words(criterion_lower: str) -> list[str]:
    """Extract key noun phrases that should appear in a passing response."""
    stop = {
        "the", "a", "an", "is", "are", "of", "in", "and", "or", "not", "just",
        "only", "at", "it", "this", "that", "to", "for", "with", "on", "as",
        "should", "must", "does", "do", "be", "by", "from", "but", "if", "its",
        "output", "response", "answer", "mentions", "explains", "contains",
        "includes", "section", "field", "pattern", "criterion", "rubric",
        "least", "one", "two", "also", "any", "each", "per", "all", "most",
        "more", "less", "than", "when", "where", "which", "who", "how", "why",
        "what", "was", "has", "have", "had", "will", "would", "could",
        "can", "may", "might", "shall",
    }
    words = criterion_lower.replace("(", " ").replace(")", " ").replace(",", " ").split()
    return [w for w in words if len(w) > 4 and w not in stop]


# ── Rubric scoring — LLM-as-judge (Groq) ──────────────────────────────────────

def score_rubric_llm_judge(
    text: str,
    rubric: list[str],
    chain_type: str,
    groq_api_key: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> tuple[int, int, list[str]]:
    """
    Score LLM output against rubric criteria using Groq gpt-oss-120b as judge.

    Used for all chains. The judge requires direct evidence quotes for PASS verdicts,
    handles negative assertions and numeric threshold criteria correctly, and uses
    reasoning_effort=high for consistency. The heuristic scorer is the fallback when
    GROQ_API_KEY is unavailable (--no-judge or no key set).

    Returns (passed, total, failed_criteria, evaluations).
    Raises httpx.HTTPError on API failure — caller should catch and fall back to heuristic.
    """
    rubric_numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(rubric))

    system = (
        "You are a strict, evidence-based evaluator. You will be given a model response and "
        "a list of rubric criteria. For each criterion, verdict PASS only if the response "
        "contains clear, specific evidence satisfying the criterion. Verdict FAIL if the "
        "evidence is absent, vague, or only implied.\n\n"
        "Rules:\n"
        "- PASS requires a direct quote or unambiguous paraphrase from the response as evidence.\n"
        "- Negative criteria (e.g. 'does not mention X') pass only if X is genuinely absent.\n"
        "- Numeric criteria (e.g. 'score >= 0.7') pass only if the exact value in the response "
        "satisfies the inequality — do not infer intent.\n"
        "- Do not give benefit of the doubt. If the evidence is borderline, verdict FAIL.\n\n"
        "Output ONLY valid JSON in this exact format — no prose outside the JSON:\n"
        '{"evaluations": [{"criterion": "<criterion text>", "verdict": "PASS", '
        '"evidence": "<exact quote or \'ABSENT\' if criterion requires absence>"}]}\n'
        '"verdict" must be exactly "PASS" or "FAIL". Include every criterion.'
    )

    human = (
        f"Chain type: {chain_type}\n\n"
        f"Model response to evaluate:\n---\n{text}\n---\n\n"
        f"Rubric criteria:\n{rubric_numbered}\n\n"
        "For each criterion, find the evidence in the response, then set the verdict. "
        "Return the JSON."
    )

    # reasoning_effort=high + response_format=json_object → 400 json_validate_failed
    # on long free-text inputs (Groq bug). Use "medium" which stays within context
    # limits while still applying chain-of-thought to rubric evaluation.
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": human},
        ],
        "response_format": {"type": "json_object"},
        "reasoning_effort": "medium",
        "temperature": 0.0,
    }

    # Retry with exponential backoff on rate-limit and transient server errors.
    max_attempts = 5
    base_delay = 2.0
    response = None
    for attempt in range(max_attempts):
        try:
            response = httpx.post(
                f"{GROQ_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90.0,
            )
            if response.status_code in (429, 500, 503, 504):
                retry_after = float(
                    response.headers.get("retry-after", base_delay * (2 ** attempt))
                )
                if attempt < max_attempts - 1:
                    print(
                        f"    judge error ({response.status_code}) — retrying in {retry_after:.1f}s "
                        f"(attempt {attempt + 1}/{max_attempts})",
                        flush=True,
                    )
                    time.sleep(retry_after)
                    continue
            if not response.is_success:
                print(
                    f"    judge HTTP {response.status_code}: {response.text[:300]}",
                    flush=True,
                )
            response.raise_for_status()
            break
        except httpx.TimeoutException:
            if attempt < max_attempts - 1:
                wait = base_delay * (2 ** attempt)
                print(f"    judge timeout — retrying in {wait:.1f}s (attempt {attempt + 1}/{max_attempts})",
                      flush=True)
                time.sleep(wait)
            else:
                raise

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        print(f"    judge JSONDecodeError — raw content ({len(content)} chars): {content[:500]}", flush=True)
        raise

    evaluations = obj.get("evaluations", [])
    passed = sum(1 for e in evaluations if e.get("verdict") == "PASS")
    failed = [e["criterion"] for e in evaluations if e.get("verdict") == "FAIL"]

    return passed, len(rubric), failed, evaluations


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate_model(
    host: str,
    model: str,
    corpus: list[dict],
    runs_per_case: int,
    score_rubric_flag: bool,
    groq_api_key: str | None,
    judge_model: str,
    failures_log: "Path | None" = None,
) -> list[dict]:
    """Run full eval corpus against one model. Returns list of per-case result dicts.

    If failures_log is provided, rubric failures are appended as JSONL after each
    case so the file can be tailed during a long run.
    """
    print(f"\n{'=' * 68}")
    print(f"  Model: {model}")
    print(f"  Cases: {len(corpus)}  |  Runs/case: {runs_per_case}")
    if groq_api_key:
        print(f"  Rubric: heuristic + LLM-judge ({judge_model}) for {sorted(LLM_JUDGE_CHAINS)}")
    else:
        print(f"  Rubric: heuristic only (set GROQ_API_KEY to enable LLM-judge for "
              f"{sorted(LLM_JUDGE_CHAINS)})")
    if failures_log:
        print(f"  Failures log: {failures_log}")
    print(f"{'=' * 68}")

    results: list[dict] = []

    for entry in corpus:
        case_id = entry["id"]
        chain_type = entry["chain_type"]
        schema_name = entry.get("expected_schema")
        rubric = entry.get("quality_rubric", [])

        system_prompt = SYSTEM_PROMPTS[chain_type]
        human_prompt = build_human_prompt(chain_type, entry["inputs"])

        use_llm_judge = (
            score_rubric_flag
            and rubric
            and chain_type in LLM_JUDGE_CHAINS
            and groq_api_key is not None
        )
        rubric_scorer_label = "llm-judge" if use_llm_judge else "heuristic"

        compliance_passes = 0
        total_elapsed = 0.0
        total_tps = 0.0
        total_rubric_passed = 0
        total_rubric_criteria = len(rubric)
        failed_criteria_accumulator: list[str] = []
        judge_evaluations_all: list[dict] = []
        last_response_text = ""
        last_failed_response_text = ""  # most recent response that had rubric failures

        print(
            f"\n  [{case_id}] chain={chain_type} schema={schema_name or 'free-text'} "
            f"rubric={rubric_scorer_label}",
            flush=True,
        )

        for run_idx in range(runs_per_case):
            try:
                result = run_single_inference(
                    host=host,
                    model=model,
                    system_prompt=system_prompt,
                    human_prompt=human_prompt,
                    num_predict=_num_predict_for_chain(chain_type),
                )
            except Exception as exc:
                print(f"    run {run_idx + 1}/{runs_per_case}: ERROR — {type(exc).__name__}: {exc}")
                continue

            text = result["text"]
            elapsed = result["elapsed_s"]
            tps = result["tokens_per_sec"]
            total_elapsed += elapsed
            total_tps += tps
            last_response_text = text

            is_valid, failure_reason = validate_structured_output(text, schema_name)
            if is_valid:
                compliance_passes += 1

            if score_rubric_flag and rubric:
                if use_llm_judge:
                    try:
                        rubric_p, _, failed, evaluations = score_rubric_llm_judge(
                            text=text,
                            rubric=rubric,
                            chain_type=chain_type,
                            groq_api_key=groq_api_key,  # type: ignore[arg-type]
                            judge_model=judge_model,
                        )
                        judge_evaluations_all.append({"run": run_idx + 1, "evaluations": evaluations})
                    except Exception as exc:
                        print(f"    run {run_idx + 1}: judge ERROR ({type(exc).__name__}) — "
                              "falling back to heuristic")
                        rubric_p, _, failed = score_rubric(text, rubric)
                else:
                    rubric_p, _, failed = score_rubric(text, rubric)

                total_rubric_passed += rubric_p
                failed_criteria_accumulator.extend(failed)
                if failed:
                    last_failed_response_text = text

            status = "PASS" if is_valid else f"FAIL({failure_reason[:40]})"
            print(
                f"    run {run_idx + 1}/{runs_per_case}: {status} "
                f"| {tps:.1f} tok/s | {elapsed:.2f}s",
                flush=True,
            )

        compliance_rate = compliance_passes / runs_per_case if runs_per_case > 0 else 0.0
        avg_elapsed = total_elapsed / runs_per_case
        avg_tps = total_tps / runs_per_case
        rubric_rate = (
            total_rubric_passed / (runs_per_case * total_rubric_criteria)
            if (runs_per_case * total_rubric_criteria) > 0
            else None
        )

        if rubric_rate is not None:
            print(
                f"  → compliance {compliance_rate:.0%} | "
                f"rubric {rubric_rate:.0%} ({rubric_scorer_label})"
            )
        else:
            print(f"  → compliance {compliance_rate:.0%} | rubric N/A")

        # Print unique failing criteria inline so failures are visible during the run
        unique_failures = list(dict.fromkeys(failed_criteria_accumulator))  # preserve order, dedupe
        if unique_failures:
            for criterion in unique_failures:
                freq = failed_criteria_accumulator.count(criterion)
                print(f"    ✗ [{freq}/{runs_per_case}] {criterion}", flush=True)

        # Append to rolling JSONL failures log if requested
        if failures_log and unique_failures:
            entry_out = {
                "model": model,
                "case_id": case_id,
                "chain_type": chain_type,
                "compliance_rate": round(compliance_rate, 4),
                "rubric_rate": round(rubric_rate, 4) if rubric_rate is not None else None,
                "rubric_scorer": rubric_scorer_label,
                "failed_criteria": [
                    {"criterion": c, "fail_count": failed_criteria_accumulator.count(c), "runs": runs_per_case}
                    for c in unique_failures
                ],
                # Full response from the most recent run that had rubric failures,
                # so the actual model output can be inspected alongside the criteria.
                "failing_response": last_failed_response_text,
            }
            with failures_log.open("a") as fh:
                fh.write(json.dumps(entry_out) + "\n")

        results.append({
            "case_id": case_id,
            "chain_type": chain_type,
            "expected_schema": schema_name,
            "runs": runs_per_case,
            "compliance_passes": compliance_passes,
            "compliance_rate": round(compliance_rate, 4),
            "avg_elapsed_s": round(avg_elapsed, 3),
            "avg_tokens_per_sec": round(avg_tps, 2),
            "rubric_criteria_count": total_rubric_criteria,
            "rubric_passed_total": total_rubric_passed,
            "rubric_rate": round(rubric_rate, 4) if rubric_rate is not None else None,
            "rubric_scorer": rubric_scorer_label,
            "most_failed_rubric_criteria": list(set(failed_criteria_accumulator))[:5],
            "judge_evaluations": judge_evaluations_all,
            "last_response_preview": last_response_text[:300],
        })

    return results


def _num_predict_for_chain(chain_type: str) -> int:
    """
    Map chain type to appropriate token cap (mirrors §5 per-node table).

    coaching_analyzer uses 2048: multi-step diff analysis generates verbose rationale
    before outputting the JSON object; 1024 truncated responses mid-structure.
    """
    return {
        "intent_summarizer": 1024,   # was 512 — 26b think preamble exhausted budget
        "snippet_summarizer": 1024,
        "persona_synthesizer": 2048,
        "coaching_analyzer": 2048,
        "skills_synthesizer": 4096,
    }.get(chain_type, 1024)


# ── Reporting ──────────────────────────────────────────────────────────────────

def aggregate_by_chain(results: list[dict]) -> dict[str, dict]:
    """Aggregate per-case results by chain_type for summary reporting."""
    by_chain: dict[str, list[dict]] = {}
    for r in results:
        by_chain.setdefault(r["chain_type"], []).append(r)

    summary = {}
    for chain, cases in by_chain.items():
        compliance_rates = [c["compliance_rate"] for c in cases]
        rubric_rates = [c["rubric_rate"] for c in cases if c["rubric_rate"] is not None]
        scorers = list({c["rubric_scorer"] for c in cases})
        summary[chain] = {
            "n_cases": len(cases),
            "mean_compliance_rate": round(sum(compliance_rates) / len(compliance_rates), 4),
            "min_compliance_rate": round(min(compliance_rates), 4),
            "mean_rubric_rate": round(sum(rubric_rates) / len(rubric_rates), 4) if rubric_rates else None,
            "rubric_scorer": scorers[0] if len(scorers) == 1 else "mixed",
            "mean_tokens_per_sec": round(
                sum(c["avg_tokens_per_sec"] for c in cases) / len(cases), 2
            ),
        }
    return summary


def print_comparison(model_summaries: dict[str, dict[str, dict]]) -> None:
    """Print a side-by-side comparison table and the decision recommendation."""
    print(f"\n{'═' * 80}")
    print("  Quality Evaluation — Summary")
    print(f"{'═' * 80}")

    models = list(model_summaries.keys())
    all_chains = sorted({chain for m in model_summaries.values() for chain in m})

    col_w = 18
    header = f"  {'Chain':<28}" + "".join(f"{m[:col_w]:>{col_w}}" for m in models)
    print(header)
    print(f"  {'─' * (28 + col_w * len(models))}")

    print("  Structured-output compliance rate:")
    for chain in all_chains:
        row = f"    {chain:<26}"
        for model in models:
            val = model_summaries[model].get(chain, {}).get("mean_compliance_rate")
            row += f"{val:.0%}" if val is not None else f"{'N/A':>{col_w}}"
            row = row.ljust(28 + col_w * (models.index(model) + 1))
        print(row)

    print("\n  Rubric pass rate:")
    for chain in all_chains:
        row = f"    {chain:<26}"
        for model in models:
            chain_data = model_summaries[model].get(chain, {})
            val = chain_data.get("mean_rubric_rate")
            scorer = chain_data.get("rubric_scorer", "heuristic")
            label = "(judge)" if scorer == "llm-judge" else "(heuristic)"
            if val is not None:
                cell = f"{val:.0%} {label}"
                row += f"{cell:>{col_w}}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)

    # Decision logic (applies when exactly 2 models compared)
    if len(models) == 2:
        m_a, m_b = models
        structured_chains = [c for c in all_chains if c not in ("snippet_summarizer", "skills_synthesizer")]

        compliance_a = [
            model_summaries[m_a][c]["mean_compliance_rate"]
            for c in structured_chains
            if c in model_summaries[m_a]
        ]
        compliance_b = [
            model_summaries[m_b][c]["mean_compliance_rate"]
            for c in structured_chains
            if c in model_summaries[m_b]
        ]

        if compliance_a and compliance_b:
            overall_a = sum(compliance_a) / len(compliance_a)
            overall_b = sum(compliance_b) / len(compliance_b)
            gap_pp = abs(overall_a - overall_b) * 100
            better_model = m_a if overall_a >= overall_b else m_b

            print(f"\n{'─' * 68}")
            print("  Decision Analysis")
            print(f"{'─' * 68}")
            print(f"  {m_a} overall compliance (structured chains): {overall_a:.1%}")
            print(f"  {m_b} overall compliance (structured chains): {overall_b:.1%}")
            print(f"  Compliance gap: {gap_pp:.1f} percentage points")
            print()

            # Pick the resource-efficient model by throughput (higher tok/s = faster/smaller).
            # Average tok/s across all chain types for each model.
            def _mean_tps(model: str) -> float:
                values = [
                    model_summaries[model][c]["mean_tokens_per_sec"]
                    for c in model_summaries[model]
                    if model_summaries[model][c].get("mean_tokens_per_sec")
                ]
                return sum(values) / len(values) if values else 0.0

            tps_a = _mean_tps(m_a)
            tps_b = _mean_tps(m_b)
            faster_model = m_a if tps_a >= tps_b else m_b
            faster_tps = max(tps_a, tps_b)
            slower_tps = min(tps_a, tps_b)
            speedup = faster_tps / slower_tps if slower_tps > 0 else 1.0

            print(f"  Throughput: {m_a} {tps_a:.0f} tok/s  |  {m_b} {tps_b:.0f} tok/s")
            print(f"  Speed advantage: {faster_model} is {speedup:.1f}× faster")
            print()

            if gap_pp <= 2.0:
                print(
                    f"  ✅ RECOMMENDATION: Use {faster_model}\n"
                    f"     Compliance gap ({gap_pp:.1f}pp) is within the 2pp threshold — quality is equivalent.\n"
                    f"     {faster_model} wins on throughput: {speedup:.1f}× faster ({faster_tps:.0f} vs {slower_tps:.0f} tok/s)."
                )
            elif gap_pp <= 5.0:
                print(
                    f"  ⚠️  MARGINAL: Compliance gap is {gap_pp:.1f}pp (2–5pp zone).\n"
                    f"     Quality leader: {better_model}  |  Speed leader: {faster_model}\n"
                    f"     Review rubric scores and failing criteria before deciding.\n"
                    f"     Consider testing on production-representative prompts."
                )
            else:
                print(
                    f"  🔴 RECOMMENDATION: Use {better_model}\n"
                    f"     Compliance gap ({gap_pp:.1f}pp) exceeds 5pp threshold.\n"
                    f"     Quality difference is large enough to justify the {speedup:.1f}× speed cost."
                )


def write_results(
    model_results: dict[str, list[dict]],
    model_summaries: dict[str, dict[str, dict]],
) -> tuple[Path, Path]:
    """Write JSON + CSV results to experiments/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "decision_thresholds": {"within_2pp_use_faster_model": True, "over_5pp_use_quality_model": True},
        "model_summaries": model_summaries,
        "model_results": model_results,
    }
    json_path = RESULTS_DIR / f"quality_{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # Flat CSV: one row per (model, case)
    csv_path = RESULTS_DIR / f"quality_{ts}.csv"
    rows: list[dict] = []
    for model, cases in model_results.items():
        for c in cases:
            rows.append({"model": model, **{k: v for k, v in c.items() if k != "last_response_preview"}})

    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return json_path, csv_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate structured-output compliance and quality: gemma4:e4b vs gemma4:26b.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host", default=DEFAULT_OLLAMA_HOST, help="Ollama base URL"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Models to evaluate (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--corpus", default=str(CORPUS_PATH),
        help=f"Path to JSONL eval corpus (default: {CORPUS_PATH})",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS_PER_CASE,
        help=f"Runs per case per model (default: {DEFAULT_RUNS_PER_CASE})",
    )
    parser.add_argument(
        "--chains", nargs="+", default=None,
        help="Run only specific chain types (e.g. --chains intent_summarizer coaching_analyzer)",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Run only specific case IDs (e.g. --cases skills_syn_001 coaching_003)",
    )
    parser.add_argument(
        "--no-rubric", action="store_true",
        help="Skip rubric scoring entirely (compliance-only; faster)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-judge even if GROQ_API_KEY is set; use heuristic scorer for all chains",
    )
    parser.add_argument(
        "--groq-api-key", default=None,
        help="Groq API key for LLM-as-judge scoring (default: GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help=f"Groq model to use as rubric judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    args = parser.parse_args()

    # Resolve Groq API key
    groq_api_key: str | None = None
    if not args.no_rubric and not args.no_judge:
        groq_api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
        if groq_api_key is None:
            print(
                f"  ⚠️  GROQ_API_KEY not set — LLM-as-judge disabled for "
                f"{sorted(LLM_JUDGE_CHAINS)}.\n"
                "     Heuristic scorer will be used (rubric scores unreliable for those chains).\n"
                "     Set GROQ_API_KEY or pass --groq-api-key to enable."
            )

    # Verify Ollama reachable
    try:
        httpx.get(f"{args.host}/api/tags", timeout=5.0).raise_for_status()
    except Exception as exc:
        raise SystemExit(f"\nCannot reach Ollama at {args.host}.\n  {exc}") from exc

    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus file not found: {corpus_path}")

    corpus: list[dict] = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))

    if args.chains:
        corpus = [c for c in corpus if c["chain_type"] in args.chains]

    if args.cases:
        corpus = [c for c in corpus if c["id"] in args.cases]

    if not corpus:
        raise SystemExit("No corpus entries after filtering. Check --chains / --cases arguments.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    failures_log: Path | None = None
    if not args.no_rubric:
        failures_log = RESULTS_DIR / f"failures_{ts}.jsonl"

    print("Developer Memory — Output Quality Evaluation")
    print(f"  Corpus   : {len(corpus)} cases from {corpus_path}")
    print(f"  Models   : {args.models}")
    print(f"  Runs/case: {args.runs}")
    print(f"  Rubric   : {'disabled' if args.no_rubric else 'enabled'}")
    print(f"  Judge    : {args.judge_model if groq_api_key else 'heuristic only'}")
    print(f"  Host     : {args.host}")
    if failures_log:
        print(f"  Failures : {failures_log}  (tail -f to monitor)")

    model_results: dict[str, list[dict]] = {}
    model_summaries: dict[str, dict[str, dict]] = {}

    for model in args.models:
        results = evaluate_model(
            host=args.host,
            model=model,
            corpus=corpus,
            runs_per_case=args.runs,
            score_rubric_flag=not args.no_rubric,
            groq_api_key=groq_api_key,
            judge_model=args.judge_model,
            failures_log=failures_log,
        )
        model_results[model] = results
        model_summaries[model] = aggregate_by_chain(results)

    print_comparison(model_summaries)

    json_path, csv_path = write_results(model_results, model_summaries)
    print(f"\n  JSON     → {json_path}")
    print(f"  CSV      → {csv_path}")
    if failures_log and failures_log.exists():
        print(f"  Failures → {failures_log}")
    print(
        "\n  Tips:\n"
        "  • Focus on structured chains: --chains intent_summarizer coaching_analyzer --runs 10\n"
        "  • Enable semantic rubric scoring: export GROQ_API_KEY=... then rerun\n"
        "  • Fast compliance-only pass: --no-rubric"
    )


if __name__ == "__main__":
    main()
