import re
from dataclasses import dataclass

from src.models.enums import RedFlagCategory
from src.models.schemas import RedFlag, UserContext
from src.validators.base import BaseValidator, ValidationResult


@dataclass
class RedFlagPattern:
    """Pattern for detecting red flags in user input."""

    category: RedFlagCategory
    patterns: list[str]
    severity: str
    recommendation: str


# Red flag detection patterns
RED_FLAG_PATTERNS: list[RedFlagPattern] = [
    # Neurological red flags
    RedFlagPattern(
        category=RedFlagCategory.NEUROLOGICAL,
        patterns=[
            r"numbness",
            r"tingling",
            r"dizzy\b",
            r"lighthead(ed)?",
            r"nerve\s*pain",
            r"radiating\s*pain",
            r"shooting\s*pain",
            r"radiating\s*down\s*(the\s*)?(leg|arm)",
            r"electric\s*(shock|pain)",
            r"burning\s*down\s*(leg|arm)",
            r"sciatica",
            r"pins\s*and\s*needles",
            r"weakness\s*in\s*(arm|leg|hand|foot)",
            r"loss\s*of\s*(sensation|feeling)",
            r"seizure",
            r"epilepsy",
            r"stroke",
            r"paralysis",
            # Electrical/zapping sensations (nerve compression indicators)
            r"\bzap\b",
            r"zapping",
            r"electrical\s*(zap|sensation|feeling)",
            r"(zap|shock|electrical)\b.*\b(grip|gripping|elbow|wrist)",
            r"(grip|gripping)\b.*\b(zap|shock|electrical)",
            r"sharp\s*electrical",
            # Peripheral nerve compression signs
            r"ulnar\s*nerve",
            r"median\s*nerve",
            r"carpal\s*tunnel",
            r"cubital\s*tunnel",
            r"thoracic\s*outlet",
            # Radiating symptoms with anatomy
            r"(tingling|numbness)\b.*\b(arm|hand|finger|leg|foot|toe)",
            r"(shooting|radiating)\s*(pain|sensation)\b.*\b(down|into)",
        ],
        severity="high",
        recommendation="Neurological symptoms require medical clearance before exercise.",
    ),
    # Cardiovascular red flags
    RedFlagPattern(
        category=RedFlagCategory.CARDIOVASCULAR,
        patterns=[
            r"chest\s*pain",
            r"heart\s*(attack|disease|failure|condition)",
            r"cardiac",
            r"arrhythmia",
            r"palpitation",
            r"shortness\s*of\s*breath\s*at\s*rest",
            r"short\s*of\s*breath\s*(while|when)\s*(resting|sitting)",
            r"breathless\s*(at\s*rest|while\s*sitting)",
            r"can't\s*breathe\s*(at\s*rest|sitting)",
            r"uncontrolled\s*(hypertension|blood\s*pressure)",
            r"angina",
            r"syncope",
            r"faint(ing)?",
            r"passed\s*out",
            r"dvt",
            r"deep\s*vein\s*thrombosis",
            r"blood\s*clot",
            r"pulmonary\s*embolism",
        ],
        severity="critical",
        recommendation="Cardiovascular conditions require physician clearance before exercise.",
    ),
    # Acute injury red flags
    RedFlagPattern(
        category=RedFlagCategory.ACUTE_INJURY,
        patterns=[
            r"acute\s*(injury|pain|tear)",
            r"recent\s*(surgery|operation)",
            r"fracture",
            r"broken\s*bone",
            r"dislocation",
            r"torn\s*(ligament|tendon|muscle)",
            r"acl\s*(tear|injury|surgery)",
            r"meniscus\s*tear",
            r"rotator\s*cuff\s*tear",
            r"herniated\s*disc",
            r"bulging\s*disc",
            r"severe\s*sprain",
            r"post.?op",
        ],
        severity="high",
        recommendation="Acute injuries require healing time and medical clearance.",
    ),
    # Inflammatory red flags
    RedFlagPattern(
        category=RedFlagCategory.INFLAMMATORY,
        patterns=[
            r"rheumatoid\s*arthritis",
            r"lupus",
            r"fibromyalgia\s*flare",
            r"autoimmune",
            r"active\s*inflammation",
            r"swelling\s*and\s*redness",
            r"fever\s*with\s*joint\s*pain",
            r"gout\s*flare",
        ],
        severity="medium",
        recommendation="Active inflammatory conditions may require modified exercise approach.",
    ),
    # Contraindicated conditions
    RedFlagPattern(
        category=RedFlagCategory.CONTRAINDICATED,
        patterns=[
            r"pregnant",
            r"pregnancy",
            r"uncontrolled\s*diabetes",
            r"dialysis",
            r"recent\s*chemotherapy",
            r"osteoporosis",
            r"aneurysm",
            r"marfan",
            r"ehlers.?danlos",
        ],
        severity="high",
        recommendation="This condition requires specialized exercise prescription.",
    ),
]

# Age-related concerns
AGE_CONCERNS = {
    "youth": {
        "min_age": 13,
        "max_age": 17,
        "concerns": ["growth plate safety", "appropriate loading", "supervision recommended"],
    },
    "senior": {
        "min_age": 65,
        "max_age": None,
        "concerns": ["fall risk assessment", "bone density consideration", "recovery time"],
    },
}


class RedFlagScanner(BaseValidator[UserContext]):
    """
    Scans user context for clinical red flags.

    Detects conditions that:
    - Require medical clearance before exercise (critical/high severity)
    - Require program modifications (medium severity)
    - Warrant additional caution (low severity)

    Critical flags result in rejection; others result in warnings/modifications.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize scanner.

        Args:
            strict_mode: If True, high severity flags are errors.
                        If False, only critical flags are errors.
        """
        self.strict_mode = strict_mode
        self._compiled_patterns: list[tuple[RedFlagPattern, list[re.Pattern]]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for flag_pattern in RED_FLAG_PATTERNS:
            compiled = [
                re.compile(p, re.IGNORECASE) for p in flag_pattern.patterns
            ]
            self._compiled_patterns.append((flag_pattern, compiled))

    @property
    def name(self) -> str:
        return "red_flag_scanner"

    def validate(self, data: UserContext) -> ValidationResult:
        """
        Scan user context for red flags.

        Args:
            data: UserContext to scan.

        Returns:
            ValidationResult with detected red flags.
        """
        errors: list[str] = []
        warnings: list[str] = []
        detected_flags: list[RedFlag] = []

        # Gather all text to scan
        text_sources = self._gather_text_sources(data)

        # Scan for pattern-based red flags
        for source_name, text in text_sources.items():
            if not text:
                continue

            for flag_pattern, compiled_patterns in self._compiled_patterns:
                for pattern in compiled_patterns:
                    if pattern.search(text):
                        # Detect historical/non-acute context to avoid false rejections
                        historical = self._is_historical_injury(text)
                        severity = flag_pattern.severity
                        if historical and flag_pattern.category == RedFlagCategory.ACUTE_INJURY:
                            severity = "medium"

                        red_flag = RedFlag(
                            category=flag_pattern.category,
                            description=f"Detected: {pattern.pattern}",
                            severity=severity,
                            source_field=source_name,
                            recommendation=flag_pattern.recommendation,
                        )
                        detected_flags.append(red_flag)

                        # Determine if error or warning
                        if severity == "critical":
                            errors.append(
                                f"CRITICAL: {flag_pattern.category.value} - "
                                f"{flag_pattern.recommendation}"
                            )
                        elif severity == "high" and self.strict_mode:
                            errors.append(
                                f"HIGH RISK: {flag_pattern.category.value} - "
                                f"{flag_pattern.recommendation}"
                            )
                        else:
                            warnings.append(
                                f"CAUTION: {flag_pattern.category.value} - "
                                f"{flag_pattern.recommendation}"
                            )
                        break  # Only flag once per category per source

        # Check age-related concerns
        age_warnings = self._check_age_concerns(data)
        warnings.extend(age_warnings)

        # Check for explicit contraindications
        if data.medical_history.contraindications:
            for contraindication in data.medical_history.contraindications:
                warnings.append(f"User-reported contraindication: {contraindication}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "red_flags": [f.model_dump() for f in detected_flags],
                "red_flag_count": len(detected_flags),
                "critical_count": sum(1 for f in detected_flags if f.severity == "critical"),
                "high_count": sum(1 for f in detected_flags if f.severity == "high"),
            },
        )

    def _gather_text_sources(self, data: UserContext) -> dict[str, str]:
        """Gather all text fields to scan for red flags."""
        sources = {}

        # Medical history
        sources["conditions"] = " ".join(data.medical_history.conditions)
        sources["injuries"] = " ".join(data.medical_history.injuries)
        sources["medications"] = " ".join(data.medical_history.medications)
        sources["contraindications"] = " ".join(data.medical_history.contraindications)
        sources["medical_notes"] = data.medical_history.notes or ""

        # Pain areas and requests
        sources["pain_areas"] = " ".join(data.pain_areas)
        sources["specific_requests"] = " ".join(data.specific_requests)

        return sources

    def _check_age_concerns(self, data: UserContext) -> list[str]:
        """Check for age-related concerns."""
        warnings = []

        if data.biometrics.age is None:
            return warnings

        age = data.biometrics.age

        # Youth concerns
        if 13 <= age <= 17:
            warnings.append(
                f"Youth athlete (age {age}): Ensure appropriate supervision, "
                "avoid maximal loading, and prioritize technique."
            )

        # Senior concerns
        if age >= 65:
            warnings.append(
                f"Senior athlete (age {age}): Consider fall risk, "
                "allow extra recovery time, and include balance work."
            )

        # Very young
        if age < 13:
            warnings.append(
                f"Age {age} is below recommended minimum. "
                "Focus on play-based movement and age-appropriate activities."
            )

        return warnings

    def get_detected_flags(self, data: UserContext) -> list[RedFlag]:
        """
        Get detailed list of detected red flags.

        Convenience method for getting structured flag data.
        """
        result = self.validate(data)
        return [
            RedFlag(**f) for f in result.metadata.get("red_flags", [])
        ]

    @staticmethod
    def _is_historical_injury(text: str) -> bool:
        """
        Heuristic: detect phrases indicating an injury is historical or repaired,
        so we can warn instead of hard-blocking.
        """
        historical_markers = [
            r"history\s+of",
            r"old\s+injury",
            r"previous\s+injury",
            r"past\s+injury",
            r"repaired",
            r"reconstructed",
            r"years?\s+ago",
            r"\d+\s+years?\s+ago",
            r"months?\s+ago",
            r"fully\s+healed",
            r"no\s+current\s+pain",
            r"no\s+pain",
            r"have\s+no\s+pain",
            r"pain[- ]?free",
            # Medical clearance markers (flexible word order)
            r"cleared\s+by",
            r"fully\s+cleared",
            r"been\s+cleared",
            r"cleared\s+by\s+(my\s+)?(doctor|pt|physio|physician|therapist)",
            r"(doctor|pt|physio|physician|therapist)\s+cleared",
            r"medical\s+clearance",
            r"got\s+clearance",
            # Past tense indicators
            r"had\s+a\s+\w+\s+(repair|surgery|operation)",
            r"recovered\s+from",
            r"healed",
            r"used\s+to\s+have",
            r"previously\s+had",
            # Explicitly resolved
            r"but\s+(it\s+)?(doesn't|does\s+not|never)\s+(cause|hurt|bother)",
        ]
        return any(re.search(pat, text, re.IGNORECASE) for pat in historical_markers)


class MedicationInteractionScanner(BaseValidator[UserContext]):
    """
    Scans for medications that affect exercise recommendations.

    Identifies medications that may require:
    - Heart rate monitoring modifications (beta blockers)
    - Hydration considerations (diuretics)
    - Bleeding risk awareness (blood thinners)
    - Energy/fatigue considerations (various)
    """

    # Medications and their exercise implications
    MEDICATION_CONCERNS = {
        "beta blocker": {
            "patterns": [r"metoprolol", r"atenolol", r"propranolol", r"beta.?blocker"],
            "concern": "Heart rate will be lower than expected; use RPE instead of HR zones",
        },
        "blood thinner": {
            "patterns": [r"warfarin", r"coumadin", r"eliquis", r"xarelto", r"blood\s*thinner"],
            "concern": "Increased bleeding risk; avoid contact sports and high-impact activities",
        },
        "diuretic": {
            "patterns": [r"furosemide", r"lasix", r"hydrochlorothiazide", r"diuretic"],
            "concern": "Increased dehydration risk; ensure adequate fluid intake",
        },
        "statin": {
            "patterns": [r"atorvastatin", r"simvastatin", r"lipitor", r"statin"],
            "concern": "May cause muscle pain/weakness; monitor for unusual soreness",
        },
        "insulin": {
            "patterns": [r"insulin", r"diabetes\s*medication"],
            "concern": "Blood sugar management required; have fast-acting carbs available",
        },
    }

    @property
    def name(self) -> str:
        return "medication_interaction_scanner"

    def validate(self, data: UserContext) -> ValidationResult:
        """Scan medications for exercise interactions."""
        warnings: list[str] = []

        medications_text = " ".join(data.medical_history.medications).lower()

        for med_type, info in self.MEDICATION_CONCERNS.items():
            for pattern in info["patterns"]:
                if re.search(pattern, medications_text, re.IGNORECASE):
                    warnings.append(f"Medication ({med_type}): {info['concern']}")
                    break

        return ValidationResult(
            is_valid=True,  # Medication concerns are warnings, not blocking
            errors=[],
            warnings=warnings,
            metadata={
                "medications_scanned": len(data.medical_history.medications),
                "concerns_found": len(warnings),
            },
        )
