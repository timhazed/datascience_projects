import logging

from langchain_core.prompts import ChatPromptTemplate

from src.llm.llm_factory import get_llm_for_agent
from src.models.schemas import UserContext
from src.validators.red_flag_scanner import RedFlagScanner

logger = logging.getLogger(__name__)

_SAFETY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a safety screener for an exercise coaching app. "
        "Does this message describe a medical emergency or acute symptom "
        "requiring IMMEDIATE medical attention? "
        "Reply YES if so, NO if not. No other text.",
    ),
    ("human", "{message}"),
])

# Acute symptom patterns moved verbatim from StateRouter._check_safety() lines 242-262.
# Plain strings matched with `in message_lower` — not regex.
_ACUTE_PATTERNS: list[str] = [
    "chest pain",
    "can't breathe",
    "shortness of breath",
    "breathless",
    "dizzy",
    "lightheaded",
    "numbness",
    "shooting pain",
    "radiating pain",
    "severe pain",
    "emergency",
    "heart attack",
    "stroke",
    # Neurological red flags (nerve compression indicators)
    "electrical zap",
    "zapping",
    "zap when",
    "sharp zap",
    "tingling",
    "pins and needles",
]


class SafetyJudge:
    """Four-layer safety screener: regex fast-path (×3) + LLM semantic backstop.

    Layers L1, L1b, L2 run synchronously with no LLM cost. Layer L3 (LLM)
    is invoked only when all three regex layers return empty. assess() has a
    single exit point — the first non-empty concern list short-circuits.

    StateRouter constructs SafetyJudge with no arguments. The LLM is resolved
    internally via get_llm_for_agent("safety") — all StateRouter() call sites
    remain zero-argument.
    """

    def __init__(self) -> None:
        """Initialise scanners. The LLM chain is built lazily on first L3 call.

        Lazy init keeps StateRouter() construction free of I/O and API-key
        requirements, which makes unit tests cheaper and avoids startup errors
        when no API key is present.
        """
        self._red_flag_scanner = RedFlagScanner(strict_mode=True)
        self._chain = None

    @property
    def chain(self):
        """LangChain chain for L3 LLM assessment, built on first access."""
        if self._chain is None:
            self._chain = _SAFETY_PROMPT | get_llm_for_agent("safety")
        return self._chain

    def assess(self, message: str, context: UserContext) -> list[str]:
        """Return concern strings. Empty list means safe. Never raises.

        Single exit. Layers evaluated left to right; first non-empty result
        short-circuits. Python list truthiness: [] is falsy, non-empty is truthy.

        Args:
            message: Raw user message being routed.
            context: UserContext accumulated so far for this session.

        Returns:
            List of concern strings, or empty list if no concerns detected.
        """
        return (
            self._scan_context(context)
            or self._scan_message_deep(message, context)
            or self._scan_message_regex(message)
            or self._run_llm_assessment(message)
        )

    def _scan_context(self, context: UserContext) -> list[str]:
        """L1: RedFlagScanner on stored user context. Fast, no LLM.

        Catches red flags already stored in the user profile across prior turns.
        """
        result = self._red_flag_scanner.validate(context)
        return result.errors if not result.is_valid else []

    def _scan_message_deep(self, message: str, context: UserContext) -> list[str]:
        """L1b: RedFlagScanner on message text via temp context.

        Preserves the deep-scan logic at state_router.py:269-278. Catches
        RED_FLAG_PATTERNS regex matches on first-turn messages where no user
        context has been stored yet (e.g. "I have arrhythmia").

        A temporary context is used to avoid mutating the caller's state.
        """
        if not message:
            return []
        temp_context = UserContext(
            medical_history=context.medical_history.model_copy(deep=True),
        )
        notes = temp_context.medical_history.notes or ""
        temp_context.medical_history.notes = f"{notes}\n{message}".strip()
        result = self._red_flag_scanner.validate(temp_context)
        return result.errors if not result.is_valid else []

    def _scan_message_regex(self, message: str) -> list[str]:
        """L2: _ACUTE_PATTERNS string list on raw message. Fast, no LLM.

        Covers known acute signals: "electrical zap", "chest pain", "dizzy", etc.
        """
        msg_lower = message.lower()
        for pattern in _ACUTE_PATTERNS:
            if pattern in msg_lower:
                return [f"Acute symptom mentioned: {pattern}"]
        return []

    def _run_llm_assessment(self, message: str) -> list[str]:
        """L3: LLM semantic backstop. 2-attempt retry; fail-open on both failures.

        Catches novel phrasings not covered by L1/L1b/L2 (e.g. "fluttering in
        my chest and sweaty"). The LLM receives a YES/NO prompt — no structured
        output binding required for a binary decision.

        Token budget: 1 token (YES or NO) × 1.3 safety margin = 2 → max_tokens=5.
        temperature=0.0 enforces determinism (set in config.yaml agents.safety).

        On failure (both retry attempts exhausted), logs a warning and returns []
        (fail-open) so the routing pipeline is never crashed by a judge timeout.
        """
        for attempt in range(2):
            try:
                response = self.chain.invoke({"message": message})
                if response.content.strip().upper().startswith("YES"):
                    return [f"Potential medical emergency: {message[:120]}"]
                return []
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "SafetyJudge LLM assessment attempt %d failed: %s",
                    attempt,
                    type(exc).__name__,
                )
                if attempt == 1:
                    logger.warning(
                        "SafetyJudge LLM assessment failed after 2 attempts: %s",
                        type(exc).__name__,
                    )
        return []  # fail-open — do not crash routing pipeline
