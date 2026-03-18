"""Sample inputs for integration testing."""

# =============================================================================
# Healthy User Scenarios
# =============================================================================

HEALTHY_HYPERTROPHY_REQUEST = {
    "messages": [
        "I want to build muscle. I'm 30 years old, intermediate lifter with access to a full gym.",
    ],
    "expected_workflow": "integrated",
    "expected_status": "approved",
    "expected_has_workout": True,
    "expected_has_recovery": True,
}

HEALTHY_STRENGTH_REQUEST = {
    "messages": [
        "I'm 28, advanced lifter. Create a strength-focused workout for me.",
    ],
    "expected_workflow": "integrated",
    "expected_status": "approved",
    "expected_has_workout": True,
    "expected_has_recovery": True,
}

# =============================================================================
# Recovery-Only Scenarios
# =============================================================================

RECOVERY_ONLY_REQUEST = {
    "messages": [
        "My hips are tight from sitting. I need a mobility routine.",
    ],
    "expected_workflow": "recovery_only",
    "expected_status": "approved",
    "expected_has_workout": False,
    "expected_has_recovery": True,
}

POST_WORKOUT_RECOVERY = {
    "messages": [
        "I just finished leg day. What should I do for recovery?",
    ],
    "expected_workflow": "recovery_only",
    "expected_status": "approved",
    "expected_has_workout": False,
    "expected_has_recovery": True,
}

# =============================================================================
# Red Flag / Rejection Scenarios
# =============================================================================

NEUROLOGICAL_RED_FLAG = {
    "messages": [
        "I have lower back pain with numbness down my leg. Can I still workout?",
    ],
    "expected_workflow": "safety_block",
    "expected_status": "rejected",
    "expected_flags": ["neurological"],
}

CARDIOVASCULAR_RED_FLAG = {
    "messages": [
        "I get chest pain when I exercise. Help me build muscle.",
    ],
    "expected_workflow": "safety_block",
    "expected_status": "rejected",
    "expected_flags": ["cardiovascular"],
}

ACUTE_INJURY_RED_FLAG = {
    "messages": [
        "I had ACL surgery last week. When can I start training?",
    ],
    "expected_workflow": "safety_block",
    "expected_status": "rejected",
    "expected_flags": ["acute_injury"],
}

# =============================================================================
# Multi-Turn Intake Scenarios
# =============================================================================

MULTI_TURN_INTAKE = {
    "turns": [
        {
            "user": "I want to build muscle",
            "expected_type": "intake_prompt",
            "expected_missing": ["age", "experience_level"],
        },
        {
            "user": "I'm 32 years old",
            "expected_type": "intake_prompt",
            "expected_missing": ["experience_level"],
        },
        {
            "user": "I'm an intermediate lifter",
            "expected_type": "plan",
            "expected_workflow": "integrated",
        },
    ],
}

QUICK_INTAKE = {
    "turns": [
        {
            "user": "I'm 30 years old, intermediate, looking to build muscle with a full gym",
            "expected_type": "plan",
            "expected_workflow": "integrated",
        },
    ],
}

# =============================================================================
# Edge Cases
# =============================================================================

GENERAL_QUESTION = {
    "messages": [
        "What is progressive overload?",
    ],
    "expected_workflow": "general_response",
    "expected_type": "general",
}

CLARIFICATION_RESPONSE = {
    "messages": [
        "Yes, that's correct",
    ],
    "expected_workflow": "general_response",
    "expected_type": "clarification",
}

EMPTY_MESSAGE = {
    "messages": [
        "",
    ],
    "expected_workflow": "general_response",
}
