from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.config.settings import Settings, get_settings
from src.models.enums import AuditStatus
from src.models.schemas import AuditLog, RecoveryPlan, UserContext, WorkoutPlan
from src.skills.loader import load_skill
from src.validators.red_flag_scanner import RedFlagScanner

# Additional operational instructions appended to SKILL.md
GATEKEEPER_INSTRUCTIONS = """
## Clinical Safety Audit Protocol

### 1. The Historical Cleared Rule
- If an injury is listed as 'repaired,' 'history of,' or 'fully cleared by PT,' do NOT reject.
- Instead, use MODIFIED status and add monitoring warnings (e.g., 'Monitor knee for swelling').

### 2. Neurological Zero-Tolerance
- Any 'zaps,' 'tingling,' or 'numbness' in UserContext MUST trigger REJECTED. No modified version.

### 3. Volume and Intensity Checks
- Cross-reference plan with ACSM standards. Reject if RPE 10 prescribed for managed hypertension.
"""


class ClinicalGatekeeper(BaseAgent[AuditLog]):
    """
    Clinical safety gatekeeper for auditing exercise and recovery plans.

    Acts as a safety checkpoint that:
    - Scans for red flag symptoms
    - Identifies contraindications
    - Approves, modifies, or rejects plans
    - Provides clinical rationale
    """

    def __init__(
        self,
        llm: BaseChatModel,
        settings: Settings | None = None,
        max_retries: int = 3,
    ):
        self._settings = settings or get_settings()

        # Load SKILL.md at runtime
        self._skill_content = load_skill("gatekeeper")

        self._red_flag_scanner = RedFlagScanner(strict_mode=True)
        super().__init__(llm=llm, max_retries=max_retries, validators=None)

    @property
    def name(self) -> str:
        return "clinical_gatekeeper"

    @property
    def output_schema(self) -> type[AuditLog]:
        return AuditLog

    @property
    def system_prompt(self) -> str:
        """Build system prompt from SKILL.md + operational instructions."""
        return f"{self._skill_content}\n{GATEKEEPER_INSTRUCTIONS}"

    def format_user_message(self, **kwargs: Any) -> str:
        """
        Format audit request.

        Args:
            user_context: UserContext with user information.
            workout_plan: Optional WorkoutPlan to audit.
            recovery_plan: Optional RecoveryPlan to audit.
            pre_scan_result: Optional pre-scan from RedFlagScanner.

        Returns:
            Formatted prompt for audit.
        """
        user_context: UserContext = kwargs.get("user_context")
        workout_plan: WorkoutPlan | None = kwargs.get("workout_plan")
        recovery_plan: RecoveryPlan | None = kwargs.get("recovery_plan")
        pre_scan_result = kwargs.get("pre_scan_result")

        if not user_context:
            raise ValueError("user_context is required")

        parts = ["## Audit Request", ""]

        # User context
        parts.append("### User Information")
        if user_context.biometrics.age:
            parts.append(f"- Age: {user_context.biometrics.age}")
        if user_context.experience_level:
            parts.append(f"- Experience: {user_context.experience_level}")

        # Medical history (critical for gatekeeper)
        parts.append("\n### Medical History (REVIEW CAREFULLY)")
        if user_context.medical_history.conditions:
            parts.append(f"- Conditions: {', '.join(user_context.medical_history.conditions)}")
        if user_context.medical_history.injuries:
            parts.append(f"- Injuries: {', '.join(user_context.medical_history.injuries)}")
        if user_context.medical_history.medications:
            parts.append(f"- Medications: {', '.join(user_context.medical_history.medications)}")
        if user_context.medical_history.contraindications:
            contras = ", ".join(user_context.medical_history.contraindications)
            parts.append(f"- Known Contraindications: {contras}")
        if user_context.pain_areas:
            parts.append(f"- Pain Areas: {', '.join(user_context.pain_areas)}")
        if user_context.medical_history.notes:
            parts.append(f"- Notes: {user_context.medical_history.notes}")

        if not any([
            user_context.medical_history.conditions,
            user_context.medical_history.injuries,
            user_context.medical_history.medications,
            user_context.pain_areas,
        ]):
            parts.append("- No medical concerns reported")

        # Pre-scan results
        if pre_scan_result:
            parts.append("\n### Automated Red Flag Scan Results")
            if pre_scan_result.errors:
                parts.append("⚠️ RED FLAGS DETECTED:")
                for error in pre_scan_result.errors:
                    parts.append(f"  - {error}")
            if pre_scan_result.warnings:
                parts.append("⚡ WARNINGS:")
                for warning in pre_scan_result.warnings:
                    parts.append(f"  - {warning}")
            if not pre_scan_result.errors and not pre_scan_result.warnings:
                parts.append("✓ No automated red flags detected")

        # Workout plan to audit
        if workout_plan:
            parts.append("\n### Workout Plan to Audit")
            parts.append(f"- Difficulty: {workout_plan.difficulty_level}")
            parts.append(f"- Total Sets: {workout_plan.get_total_sets()}")
            parts.append(f"- Duration: {workout_plan.total_duration_minutes} minutes")
            parts.append("\nExercises:")
            for block in workout_plan.blocks:
                parts.append(f"\n  {block.block_name} ({block.block_type}):")
                for ex in block.exercises:
                    muscles = ", ".join(m.value for m in ex.primary_muscles)
                    parts.append(f"    - {ex.name}: {ex.sets}x{ex.reps} [{muscles}]")

        # Recovery plan to audit
        if recovery_plan:
            parts.append("\n### Recovery Plan to Audit")
            parts.append(f"- Duration: {recovery_plan.total_duration_minutes} minutes")
            parts.append("\nRecovery Exercises:")
            for block in recovery_plan.blocks:
                parts.append(f"\n  {block.modality.value}:")
                for ex in block.exercises:
                    areas = ", ".join(ex.target_areas)
                    parts.append(f"    - {ex.name}: {ex.duration_seconds}s [{areas}]")

        # Task
        parts.append("\n### Audit Task")
        parts.append("Review the above information and plans for safety.")
        parts.append("Determine: APPROVED, MODIFIED, or REJECTED")
        parts.append("Provide clear clinical reasoning for your decision.")

        return "\n".join(parts)

    def audit(
        self,
        user_context: UserContext,
        workout_plan: WorkoutPlan | None = None,
        recovery_plan: RecoveryPlan | None = None,
    ) -> AuditLog:
        """
        Audit workout and/or recovery plans for safety.

        Runs automated red flag scan first, then LLM audit.

        Args:
            user_context: Complete user context.
            workout_plan: Optional workout plan to audit.
            recovery_plan: Optional recovery plan to audit.

        Returns:
            AuditLog with status and findings.
        """
        print("Processing started by Gatekeeper")
        # Run automated red flag scan first
        pre_scan = self._red_flag_scanner.validate(user_context)

        # If automated scan finds critical issues, we can fast-fail
        if pre_scan.metadata.get("critical_count", 0) > 0:
            # Critical red flags = immediate rejection
            return AuditLog(
                status=AuditStatus.REJECTED,
                red_flags=[],  # Will be populated by LLM for detailed analysis
                rejection_reason="Critical red flags detected requiring medical clearance",
                warnings=pre_scan.warnings,
            )

        # Run LLM audit for nuanced analysis
        return self.invoke(
            user_context=user_context,
            workout_plan=workout_plan,
            recovery_plan=recovery_plan,
            pre_scan_result=pre_scan,
        )

    async def aaudit(
        self,
        user_context: UserContext,
        workout_plan: WorkoutPlan | None = None,
        recovery_plan: RecoveryPlan | None = None,
    ) -> AuditLog:
        """Async version of audit."""
        print("Processing started by Gatekeeper")
        pre_scan = self._red_flag_scanner.validate(user_context)

        if pre_scan.metadata.get("critical_count", 0) > 0:
            return AuditLog(
                status=AuditStatus.REJECTED,
                red_flags=[],
                rejection_reason="Critical red flags detected requiring medical clearance",
                warnings=pre_scan.warnings,
            )

        return await self.ainvoke(
            user_context=user_context,
            workout_plan=workout_plan,
            recovery_plan=recovery_plan,
            pre_scan_result=pre_scan,
        )

    def quick_safety_check(self, user_context: UserContext) -> bool:
        """
        Quick safety check without full audit.

        Returns True if safe to proceed, False if red flags detected.
        Useful for early screening before generating plans.
        """
        scan_result = self._red_flag_scanner.validate(user_context)
        return scan_result.is_valid
