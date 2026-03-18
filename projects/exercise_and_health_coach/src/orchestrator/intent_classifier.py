import re
from dataclasses import dataclass

from src.models.enums import IntentType


@dataclass
class ClassifiedIntent:
    """Result of intent classification."""

    intent: IntentType
    confidence: float
    keywords_matched: list[str]
    raw_message: str


# Keyword patterns for each intent type
INTENT_PATTERNS: dict[IntentType, list[str]] = {
    IntentType.EXERCISE_REQUEST: [
        r"\b(workout|exercise|training|lift|strength|muscle|hypertrophy)\b",
        r"\b(build|gain|grow)\s*(muscle|strength|mass)\b",
        r"\b(gym|weights|lifting|resistance)\b",
        r"\b(routine|program|plan)\b.*\b(exercise|workout|training)\b",
        r"\b(chest|back|legs|arms|shoulders)\s*(day|workout)\b",
        r"\b(push|pull|legs?)\s*(day|split)\b",
        r"\b(sets?|reps?|volume)\b",
        r"\b(stronger|fitter|bigger)\b",
        r"\b(run|running)\b",
        r"\b(miles?|km|kilometers?)\b.*\b(run|jog)\b",
        r"\b(want\s*to|trying\s*to|able\s*to)\s*(run|jog|swim|bike|cycle)\b",
        r"\b(create|make|give|build)\s*(me\s*)?(a\s*)?(workout|routine|plan)\b",
        # Common exercises
        r"\b(pull[- ]?ups?|push[- ]?ups?|squats?|deadlifts?|bench\s*press)\b",
        r"\b(dips?|rows?|curls?|lunges?|planks?)\b",
        # Goal-based requests
        r"\b(my\s*)?goal\s*(is\s*)?to\b",
        r"\b(want\s*to|trying\s*to|help\s*me)\s*(do|achieve|reach|get)\b",
        r"\b(improve|increase|more)\s*(my\s*)?(pull[- ]?ups?|push[- ]?ups?|reps?)\b",
        # Specific movement skills
        r"\b(handstand|muscle[- ]?up|pistol\s*squat|flag|planche|front\s*lever)\b",
        r"\b(snatch|clean\s*and\s*jerk|olympic\s*lift)\b",
        r"\b(vertical\s*jump|box\s*jump|plyometric)\b",
        r"\b(drills?|progressions?)\s*(for|to)\b",
        r"\b(wrist|shoulder)\s*(stability|prep)\b.*\b(drill|exercise)\b",
        r"\blearn\s*(how\s*to|to\s*do)?\s*(a\s*)?(handstand|muscle[- ]?up|pistol)\b",
        # Body part focus (boulder shoulders, etc.)
        r"\bboulder\s*shoulders\b",
        r"\b(arms?|biceps?|triceps?)\s*(day|workout)\b",
    ],
    IntentType.RECOVERY_REQUEST: [
        r"\b(stretch|stretching|mobility|flexibility)\b",
        r"\b(foam\s*roll|smr|myofascial)\b",
        r"\b(tight|stiff|soreness|doms)\b",
        r"\b(recovery|recover)\b(?!.*\b(workout|training|exercise)\b)",
        r"\b(yoga|pilates)\b",
        r"\b(relax|relaxation|cool\s*down)\b",
        r"\b(pain|ache|discomfort)\b(?!.*\b(chest|heart|cardiac)\b)",
        # Travel/sitting-related recovery
        r"\b(flight|travel|sitting|driving)\s*(for|all\s*day|hours?|\d+\s*(hour|hr))\b",
        r"\b(locked\s*up|seized\s*up|frozen)\b",
        r"\b(hip\s*flexor|lower\s*back|upper\s*back)\b.*\b(tight|stiff|locked)\b",
        r"\b(after|from)\s*(a\s*)?(flight|travel|long\s*drive|sitting)\b",
        # Body part specific recovery
        r"\b(it\s*band|itb)\b.*\b(tight|release|roll)\b",
        r"\b(tmj|jaw)\b.*\b(tight|tension|release)\b",
        r"\b(shin\s*splints?|shins?\s*(throbbing|hurt|pain))\b",
        # Movement prep and mobility
        r"\b(movement\s*prep|warm[- ]?up\s*routine)\b",
        r"\b(overhead\s*mobility|squat\s*mobility|hip\s*mobility)\b",
        r"\b(pinch|pinching)\b.*\b(hip|squat)\b",
        # Recovery flow/routine
        r"\b(recovery\s*flow|mobility\s*flow|release\s*routine)\b",
        r"\b(deload|cns\s*(fried|fatigue))\b",
        # Standing/activity-related fatigue and recovery
        r"\b(throbbing)\b",
        r"\b(standing|stood)\s*(all\s*)?(night|day|for\s*hours?)\b",
        r"\b(concert|event|shift)\b.*\b(feet|legs?|back)\b",
        r"\b(tension\s*headache|headache)\b",
        r"\b(eyes?\s*(tired|fatigue|strain))\b",
        r"\b(\d+[- ]?hour\s*shift)\b",
        r"\b(long\s*shift|after\s*(a\s*)?shift)\b",
        r"\b(relief\s*routine|quick\s*relief)\b",
        r"\b(\d+[- ]?minute\s*(relief|flow|routine))\b",
        r"\b(relaxation[- ]?focused)\b",
    ],
    IntentType.INTEGRATED_REQUEST: [
        r"\b(full|complete|comprehensive)\s*(workout|routine|session)\b",
        r"\b(workout|exercise).*\b(stretch|recovery|mobility)\b",
        r"\b(warm\s*up|cool\s*down).*\b(workout|exercise)\b",
        r"\b(training|program).*\b(recovery|mobility)\b",
        # Responses to "workout, recovery, or both?" question
        r"^both\.?$",
        r"\bboth\b(?!\s*(of|are|were|have|had))",  # "both" but not "both of us", etc.
        r"\b(workout|exercise)\s*(and|&|with|plus)\s*(recovery|mobility|stretch)",
    ],
    IntentType.INTAKE_UPDATE: [
        r"\b(i\s*am|i'm)\s*\d+\s*(years?|yrs?)\b",
        r"\b(my|i)\s*(age|weight|height)\s*(is|:)\s*\d+\b",
        r"\b(i\s*weigh|i\s*am)\s*\d+\s*(kg|lbs?|pounds?|kilos?)\b",
        r"\b(beginner|intermediate|advanced)\b",
        r"\b(i\s*have|i've\s*got)\s*(access\s*to|a|the)\s*(gym|equipment)\b",
        r"\b(injury|injuries|condition|medical)\b",
    ],
    IntentType.GENERAL_QUESTION: [
        r"\b(what|how|why|when|where|which|can|should|is|are|do|does)\b.*\?",
        r"\b(explain|tell\s*me|help\s*me\s*understand)\b",
        r"\b(difference\s*between|better|best|recommend)\b",
    ],
    IntentType.CLARIFICATION: [
        r"\b(yes|no|yeah|nope|correct|right|exactly)\b",
        r"\b(that's|that\s*is)\s*(right|correct|it)\b",
        r"\b(i\s*meant|i\s*mean|actually)\b",
        r"\b(instead|rather|change)\b",
    ],
}


class IntentClassifier:
    """
    Classifies user intent based on keyword patterns.

    Uses rule-based classification for speed and reliability.
    Falls back to GENERAL_QUESTION for ambiguous inputs.
    """

    def __init__(self):
        # Pre-compile regex patterns
        self._compiled_patterns: dict[IntentType, list[re.Pattern]] = {}
        for intent, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def classify(self, message: str) -> ClassifiedIntent:
        """
        Classify user message intent.

        Args:
            message: User's input message.

        Returns:
            ClassifiedIntent with intent type and confidence.
        """
        message_lower = message.lower().strip()

        # Check for definitional questions first - these are always GENERAL_QUESTION
        # Patterns like "What is X?", "What does X mean?", "Define X", "Explain X"
        definitional_patterns = [
            r"^what\s+(is|are|does)\s+\w+.*\?$",
            r"^(define|explain|describe)\s+\w+",
            r"^what\s+does\s+\w+\s+mean",
            r"^how\s+does\s+\w+\s+work",
        ]
        for pattern in definitional_patterns:
            if re.search(pattern, message_lower):
                return ClassifiedIntent(
                    intent=IntentType.GENERAL_QUESTION,
                    confidence=0.85,
                    keywords_matched=["definitional_question"],
                    raw_message=message,
                )

        # Score each intent
        scores: dict[IntentType, tuple[float, list[str]]] = {}

        for intent, patterns in self._compiled_patterns.items():
            matches = []
            for pattern in patterns:
                match = pattern.search(message_lower)
                if match:
                    matches.append(match.group())

            if matches:
                # Score based on number of matches with reasonable baseline
                # 1 match = 0.5, 2 matches = 0.7, 3+ matches = 0.85+
                match_count = len(matches)
                if match_count == 1:
                    base_score = 0.5
                elif match_count == 2:
                    base_score = 0.7
                else:
                    base_score = min(0.95, 0.7 + (match_count - 2) * 0.1)
                scores[intent] = (base_score, matches)

        # Highest-priority recovery override
        recovery_protocol_terms = [
            "recovery protocol",
            "recovery routine",
            "recovery plan",
        ]
        if any(term in message_lower for term in recovery_protocol_terms):
            return ClassifiedIntent(
                intent=IntentType.RECOVERY_REQUEST,
                confidence=0.92,
                keywords_matched=[t for t in recovery_protocol_terms if t in message_lower],
                raw_message=message,
            )

        # Recovery-first override: if strong recovery cues present, prefer recovery intent
        recovery_override_terms = [
            "recovery protocol",
            "mobility routine",
            "mobility flow",
            "recovery flow",
            "release routine",
            "relief routine",
            "tech neck",
            "foam roll",
            "stretch",
            "tight neck",
            "sore calves",
            "ankle mobility",
            "tight ankles",
            # Travel/sitting recovery
            "locked up",
            "seized up",
            "after flight",
            "after travel",
            "long flight",
            "hip flexors",
            "lower back tight",
            "it band",
            "tmj",
            "shin splints",
            "overhead mobility",
            "squat mobility",
            "movement prep",
            "deload",
            "cns fried",
            # Standing/fatigue recovery
            "throbbing",
            "tension headache",
            "eyes tired",
            "standing all",
            "after concert",
            "after shift",
            "long shift",
            "relaxation-focused",
            "relaxation focused",
            "quick relief",
        ]
        exercise_terms = ["workout", "training", "lift", "program"]
        if any(term in message_lower for term in recovery_override_terms) and not any(
            ex in message_lower for ex in exercise_terms
        ):
            return ClassifiedIntent(
                intent=IntentType.RECOVERY_REQUEST,
                confidence=0.9,
                keywords_matched=[t for t in recovery_override_terms if t in message_lower],
                raw_message=message,
            )

        # Determine winner
        if not scores:
            return ClassifiedIntent(
                intent=IntentType.GENERAL_QUESTION,
                confidence=0.3,
                keywords_matched=[],
                raw_message=message,
            )

        # Sort by score
        sorted_intents = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        winner_intent, (winner_score, winner_matches) = sorted_intents[0]

        # Handle special cases

        # If both exercise and recovery keywords present, it's likely integrated
        if IntentType.EXERCISE_REQUEST in scores and IntentType.RECOVERY_REQUEST in scores:
            ex_score = scores[IntentType.EXERCISE_REQUEST][0]
            rec_score = scores[IntentType.RECOVERY_REQUEST][0]

            # If scores are close, treat as integrated
            if abs(ex_score - rec_score) < 0.3:
                combined_matches = (
                    scores[IntentType.EXERCISE_REQUEST][1]
                    + scores[IntentType.RECOVERY_REQUEST][1]
                )
                return ClassifiedIntent(
                    intent=IntentType.INTEGRATED_REQUEST,
                    confidence=min(0.9, (ex_score + rec_score) / 2 + 0.2),
                    keywords_matched=combined_matches,
                    raw_message=message,
                )

        # Intake updates can coexist with other intents
        # If intake keywords present with exercise/recovery, prioritize the action
        if winner_intent == IntentType.INTAKE_UPDATE:
            for intent in [
                IntentType.EXERCISE_REQUEST,
                IntentType.RECOVERY_REQUEST,
                IntentType.INTEGRATED_REQUEST,
            ]:
                if intent in scores and scores[intent][0] > 0.2:
                    # There's a clear action intent alongside intake info
                    # Return that intent, intake will be processed automatically
                    action_score, action_matches = scores[intent]
                    return ClassifiedIntent(
                        intent=intent,
                        confidence=min(0.95, action_score + 0.1),
                        keywords_matched=action_matches,
                        raw_message=message,
                    )

        # Normalize confidence to 0-1 range
        confidence = min(0.95, winner_score)

        return ClassifiedIntent(
            intent=winner_intent,
            confidence=confidence,
            keywords_matched=winner_matches,
            raw_message=message,
        )

    def is_exercise_related(self, message: str) -> bool:
        """Quick check if message is exercise-related."""
        result = self.classify(message)
        return result.intent in (
            IntentType.EXERCISE_REQUEST,
            IntentType.INTEGRATED_REQUEST,
        )

    def is_recovery_related(self, message: str) -> bool:
        """Quick check if message is recovery-related."""
        result = self.classify(message)
        return result.intent in (
            IntentType.RECOVERY_REQUEST,
            IntentType.INTEGRATED_REQUEST,
        )

    def needs_action(self, message: str) -> bool:
        """Check if message requires generating a plan."""
        result = self.classify(message)
        return result.intent in (
            IntentType.EXERCISE_REQUEST,
            IntentType.RECOVERY_REQUEST,
            IntentType.INTEGRATED_REQUEST,
        )
