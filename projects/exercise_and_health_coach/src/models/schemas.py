from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.models.enums import (
    AuditStatus,
    Equipment,
    FitnessGoal,
    MovementPattern,
    MuscleGroup,
    RecoveryModality,
    RedFlagCategory,
)

# =============================================================================
# User Context Models
# =============================================================================


class Biometrics(BaseModel):
    """User biometric data extracted from intake."""

    age: int | None = Field(default=None, ge=13, le=100, description="User age in years")
    weight_kg: float | None = Field(default=None, gt=0, description="Weight in kilograms")
    height_cm: float | None = Field(default=None, gt=0, description="Height in centimeters")
    sex: str | None = Field(default=None, pattern="^(male|female|other)$")
    resting_heart_rate: int | None = Field(default=None, ge=30, le=200)

    def is_complete(self) -> bool:
        """Check if minimum required biometrics are present."""
        return self.age is not None and self.weight_kg is not None


class MedicalHistory(BaseModel):
    """User medical history and contraindications."""

    conditions: list[str] = Field(default_factory=list)
    injuries: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    notes: str | None = None
    no_concerns_reported: bool = False

    def has_red_flags(self) -> bool:
        """Check for potential red flag conditions."""
        red_flag_keywords = [
            "chest pain",
            "heart",
            "cardiac",
            "stroke",
            "seizure",
            "uncontrolled",
            "acute",
            "severe",
            "numbness",
            "tingling",
            "dizziness",
        ]
        all_text = " ".join(
            self.conditions + self.injuries + self.medications + (self.notes or "").split()
        ).lower()
        return any(keyword in all_text for keyword in red_flag_keywords)


class UserContext(BaseModel):
    """Complete user context accumulated from intake conversation."""

    biometrics: Biometrics = Field(default_factory=Biometrics)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    fitness_goals: list[FitnessGoal] = Field(default_factory=list)
    available_equipment: list[Equipment] = Field(default_factory=list)
    experience_level: str | None = Field(
        default=None, pattern="^(beginner|intermediate|advanced)$"
    )
    training_days_per_week: int | None = Field(default=None, ge=1, le=7)
    session_duration_minutes: int | None = Field(default=None, ge=15, le=180)
    specific_requests: list[str] = Field(default_factory=list)
    pain_areas: list[str] = Field(default_factory=list)

    def get_missing_required_fields(self) -> list[str]:
        """Return list of required fields that are missing."""
        missing = []
        if not self.biometrics.is_complete():
            if self.biometrics.age is None:
                missing.append("age")
            if self.biometrics.weight_kg is None:
                missing.append("weight")
        if not self.fitness_goals:
            missing.append("fitness_goals")
        if self.experience_level is None:
            missing.append("experience_level")
        if not self.available_equipment:
            missing.append("equipment")
        return missing

    def is_complete_for_exercise(self) -> bool:
        """Check if context is complete for exercise prescription."""
        return len(self.get_missing_required_fields()) == 0

    def is_complete_for_recovery(self) -> bool:
        """
        Check if context is complete for recovery prescription.

        Recovery is low risk; allow quick-start even with minimal info.
        """
        return True


# =============================================================================
# Exercise Models
# =============================================================================


class Exercise(BaseModel):
    """Single exercise prescription."""

    name: str = Field(..., min_length=1, description="Exercise name")
    sets: int = Field(..., ge=1, le=10, description="Number of sets")
    reps: str = Field(..., description="Rep range or scheme (e.g., '8-12' or '5x5')")
    rest_seconds: int = Field(..., ge=0, le=600, description="Rest between sets in seconds")
    rpe_target: int | None = Field(default=None, ge=1, le=10, description="Target RPE")
    tempo: str | None = Field(default=None, description="Tempo notation (e.g., '3-1-2-0')")
    movement_pattern: MovementPattern
    primary_muscles: list[MuscleGroup] = Field(..., min_length=1)
    secondary_muscles: list[MuscleGroup] = Field(default_factory=list)
    equipment: list[Equipment] = Field(default_factory=list)
    notes: str | None = None
    alternatives: list[str] = Field(default_factory=list, description="Alternative exercises")

    @field_validator("secondary_muscles", "equipment", "alternatives", mode="before")
    @classmethod
    def coerce_none_to_empty_list(cls, v: list | None) -> list:
        """Coerce null from LLM output to empty list."""
        return v if v is not None else []


class ExerciseBlock(BaseModel):
    """A block of exercises (e.g., warm-up, main work, finisher)."""

    block_name: str = Field(..., description="Block identifier (e.g., 'warm_up', 'main_work')")
    block_type: str = Field(..., pattern="^(warm_up|main_work|accessory|finisher)$")
    exercises: list[Exercise] = Field(..., min_length=1)
    estimated_duration_minutes: int = Field(..., ge=1)
    notes: str | None = None

    def get_total_sets(self) -> int:
        """Calculate total sets in this block."""
        return sum(ex.sets for ex in self.exercises)

    def get_sets_by_muscle(self) -> dict[MuscleGroup, int]:
        """Calculate sets per muscle group."""
        muscle_sets: dict[MuscleGroup, int] = {}
        for ex in self.exercises:
            for muscle in ex.primary_muscles:
                muscle_sets[muscle] = muscle_sets.get(muscle, 0) + ex.sets
            for muscle in ex.secondary_muscles:
                # Secondary muscles get half credit
                muscle_sets[muscle] = muscle_sets.get(muscle, 0) + (ex.sets // 2)
        return muscle_sets


class WorkoutPlan(BaseModel):
    """Complete workout plan with multiple blocks."""

    plan_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    blocks: list[ExerciseBlock] = Field(..., min_length=1)
    total_duration_minutes: int
    target_goals: list[FitnessGoal] = Field(default_factory=list)
    difficulty_level: str = Field(..., pattern="^(beginner|intermediate|advanced)$")
    rationale: str = Field(..., description="Explanation of program design")

    def get_total_sets(self) -> int:
        """Calculate total sets across all blocks."""
        return sum(block.get_total_sets() for block in self.blocks)

    def get_sets_by_muscle(self) -> dict[MuscleGroup, int]:
        """Calculate sets per muscle group across all blocks."""
        total: dict[MuscleGroup, int] = {}
        for block in self.blocks:
            for muscle, sets in block.get_sets_by_muscle().items():
                total[muscle] = total.get(muscle, 0) + sets
        return total


# =============================================================================
# Recovery Models
# =============================================================================


class RecoveryExercise(BaseModel):
    """Single recovery/mobility intervention."""

    name: str = Field(..., min_length=1)
    modality: RecoveryModality
    duration_seconds: int = Field(..., ge=10, le=300)
    sets: int = Field(default=1, ge=1, le=5)
    target_areas: list[str] = Field(..., min_length=1)
    equipment: list[Equipment] = Field(default_factory=list)
    intensity: str = Field(default="moderate", pattern="^(light|moderate|deep)$")
    breathing_cue: str | None = None
    notes: str | None = None


class RecoveryBlock(BaseModel):
    """Block of recovery interventions following modality order."""

    block_name: str = Field(default="recovery")
    modality: RecoveryModality
    exercises: list[RecoveryExercise] = Field(..., min_length=1)
    estimated_duration_minutes: int = Field(..., ge=1)
    notes: str | None = None

    @field_validator("exercises")
    @classmethod
    def validate_modality_consistency(
        cls, exercises: list[RecoveryExercise], info
    ) -> list[RecoveryExercise]:
        """Ensure all exercises match the block modality."""
        modality = info.data.get("modality")
        if modality:
            for ex in exercises:
                if ex.modality != modality:
                    raise ValueError(
                        f"Exercise '{ex.name}' modality {ex.modality} "
                        f"doesn't match block modality {modality}"
                    )
        return exercises

    @classmethod
    def _normalize_modality(
        cls,
        value: str | RecoveryModality | None,
        block_name: str | None,
        exercises: list[dict | RecoveryExercise] | None = None,
    ):
        """Normalize modality from block_name or first exercise when missing."""
        if value is None and block_name:
            value = block_name

        # When block_name is generic ("recovery"), try inferring from first exercise
        if value is None and exercises:
            first_ex = exercises[0] if exercises else None
            if isinstance(first_ex, dict) and first_ex.get("modality"):
                value = first_ex["modality"]
            elif hasattr(first_ex, "modality"):
                value = first_ex.modality

        if isinstance(value, RecoveryModality):
            return value

        if isinstance(value, str):
            val = value.strip().lower()
            synonyms = {
                "smr": RecoveryModality.SMR,
                "self-myofascial release": RecoveryModality.SMR,
                "foam_roll": RecoveryModality.SMR,
                "dynamic": RecoveryModality.DYNAMIC_MOBILITY,
                "dynamic_mobility": RecoveryModality.DYNAMIC_MOBILITY,
                "mobility": RecoveryModality.DYNAMIC_MOBILITY,
                "static": RecoveryModality.STATIC_STRETCHING,
                "static_stretch": RecoveryModality.STATIC_STRETCHING,
                "static_stretching": RecoveryModality.STATIC_STRETCHING,
                "static_holding": RecoveryModality.STATIC_STRETCHING,
                "breathing": RecoveryModality.BREATHING,
                "active_recovery": RecoveryModality.ACTIVE_RECOVERY,
                "active recovery": RecoveryModality.ACTIVE_RECOVERY,
            }
            if val in synonyms:
                return synonyms[val]

        return value

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Override model_validate to pre-normalize modality using block_name/synonyms.
        This helps tolerate LLM outputs missing the modality field but with a block_name
        or exercises that have modality.
        """
        if isinstance(obj, dict):
            normalized_modality = cls._normalize_modality(
                obj.get("modality"),
                obj.get("block_name"),
                obj.get("exercises"),
            )
            obj = {**obj, "modality": normalized_modality}
        return super().model_validate(obj, *args, **kwargs)


class RecoveryPlan(BaseModel):
    """Complete recovery plan with blocks in prescribed order."""

    plan_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    blocks: list[RecoveryBlock] = Field(..., min_length=1)
    total_duration_minutes: int
    target_areas: list[str] = Field(default_factory=list)
    rationale: str = Field(..., description="Explanation of recovery approach")

    @field_validator("blocks")
    @classmethod
    def validate_modality_order(cls, blocks: list[RecoveryBlock]) -> list[RecoveryBlock]:
        """Ensure blocks follow SMR → Dynamic → Static order."""
        modality_order = [
            RecoveryModality.SMR,
            RecoveryModality.DYNAMIC_MOBILITY,
            RecoveryModality.STATIC_STRETCHING,
            RecoveryModality.BREATHING,
            RecoveryModality.ACTIVE_RECOVERY,
        ]
        last_idx = -1
        for block in blocks:
            if block.modality in modality_order:
                current_idx = modality_order.index(block.modality)
                if current_idx < last_idx:
                    raise ValueError(
                        f"Recovery modalities out of order: {block.modality} "
                        f"should come before previous modality"
                    )
                last_idx = current_idx
        return blocks


# =============================================================================
# Audit Models
# =============================================================================


class RedFlag(BaseModel):
    """Detected red flag requiring clinical attention."""

    category: RedFlagCategory
    description: str
    severity: str = Field(..., pattern="^(low|medium|high|critical)$")
    source_field: str | None = Field(default=None, description="Field that triggered the flag")
    recommendation: str


class AuditLog(BaseModel):
    """Clinical gatekeeper audit record."""

    audit_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    timestamp: datetime = Field(default_factory=datetime.now)
    status: AuditStatus
    red_flags: list[RedFlag] = Field(default_factory=list)
    modifications: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    approval_notes: str | None = None
    rejection_reason: str | None = None

    def has_critical_flags(self) -> bool:
        """Check if any critical red flags exist."""
        return any(flag.severity == "critical" for flag in self.red_flags)


# =============================================================================
# Coach Output Models
# =============================================================================


class CoachOutput(BaseModel):
    """Final output from the coaching system."""

    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    workout_plan: WorkoutPlan | None = None
    recovery_plan: RecoveryPlan | None = None
    audit: AuditLog
    user_message: str = Field(..., description="Human-readable response to user")
    follow_up_questions: list[str] = Field(default_factory=list)

    def is_approved(self) -> bool:
        """Check if output was approved by gatekeeper."""
        return self.audit.status in (AuditStatus.APPROVED, AuditStatus.MODIFIED)


class IntakeResponse(BaseModel):
    """Response from intake agent."""

    extracted_context: UserContext
    clarification_needed: list[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    raw_extraction_notes: str | None = None
