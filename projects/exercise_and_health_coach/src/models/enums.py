from enum import Enum


class ExperienceLevel(str, Enum):
    """User training experience levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class FitnessGoal(str, Enum):
    """User fitness objectives."""

    HYPERTROPHY = "hypertrophy"
    STRENGTH = "strength"
    ENDURANCE = "endurance"
    GENERAL_FITNESS = "general_fitness"
    WEIGHT_LOSS = "weight_loss"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    REHABILITATION = "rehabilitation"


class Equipment(str, Enum):
    """Available training equipment."""

    BARBELL = "barbell"
    DUMBBELL = "dumbbell"
    KETTLEBELL = "kettlebell"
    CABLE = "cable"
    MACHINE = "machine"
    BODYWEIGHT = "bodyweight"
    BANDS = "bands"
    FOAM_ROLLER = "foam_roller"
    LACROSSE_BALL = "lacrosse_ball"
    TREADMILL = "treadmill"  # For running/cardio goals
    MEDICINE_BALL = "medicine_ball"


class MovementPattern(str, Enum):
    """Fundamental movement patterns for exercise classification."""

    HORIZONTAL_PUSH = "horizontal_push"
    HORIZONTAL_PULL = "horizontal_pull"
    VERTICAL_PUSH = "vertical_push"
    VERTICAL_PULL = "vertical_pull"
    HIP_HINGE = "hip_hinge"
    SQUAT = "squat"
    LUNGE = "lunge"
    CARRY = "carry"
    ROTATION = "rotation"
    ANTI_ROTATION = "anti_rotation"
    CARDIO = "cardio"  # Running, cycling, etc. for endurance goals


class MuscleGroup(str, Enum):
    """Target muscle groups for volume tracking."""

    CHEST = "chest"
    BACK = "back"
    SHOULDERS = "shoulders"
    BICEPS = "biceps"
    TRICEPS = "triceps"
    QUADRICEPS = "quadriceps"
    HAMSTRINGS = "hamstrings"
    GLUTES = "glutes"
    CALVES = "calves"
    CORE = "core"
    FOREARMS = "forearms"


class RecoveryModality(str, Enum):
    """Recovery intervention types in prescribed order."""

    SMR = "smr"  # Self-myofascial release (foam rolling)
    DYNAMIC_MOBILITY = "dynamic_mobility"
    STATIC_STRETCHING = "static_stretching"
    BREATHING = "breathing"
    ACTIVE_RECOVERY = "active_recovery"


class AuditStatus(str, Enum):
    """Clinical gatekeeper decision states."""

    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"


class RedFlagCategory(str, Enum):
    """Categories of clinical red flags requiring escalation."""

    NEUROLOGICAL = "neurological"
    CARDIOVASCULAR = "cardiovascular"
    ACUTE_INJURY = "acute_injury"
    INFLAMMATORY = "inflammatory"
    CONTRAINDICATED = "contraindicated"


class IntentType(str, Enum):
    """User intent categories for routing."""

    EXERCISE_REQUEST = "exercise_request"
    RECOVERY_REQUEST = "recovery_request"
    INTEGRATED_REQUEST = "integrated_request"
    INTAKE_UPDATE = "intake_update"
    GENERAL_QUESTION = "general_question"
    CLARIFICATION = "clarification"


class WorkflowType(str, Enum):
    """Available workflow types."""

    INTEGRATED = "integrated"  # Exercise + Recovery
    RECOVERY_ONLY = "recovery_only"
    RECOVERY_FOLLOWUP = "recovery_followup"  # Short follow-up Q after a recovery plan
    INTAKE_NEEDED = "intake_needed"  # More info required
    SAFETY_BLOCK = "safety_block"  # Red flags detected
    GENERAL_RESPONSE = "general_response"  # Q&A, not plan generation
