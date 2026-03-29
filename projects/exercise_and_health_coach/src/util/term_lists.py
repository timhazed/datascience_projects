"""
Shared term lists used by both IntentClassifier and StateRouter.

Centralising these prevents the classifier and router from drifting apart
when the lists are extended.
"""

# Terms that indicate an exercise/training context.
# Used by:
#   - IntentClassifier: recovery-override guard (messages containing these terms
#     are NOT flipped to RECOVERY_REQUEST even if recovery cues are present).
#   - StateRouter: recovery follow-up guard (messages containing these terms
#     are NOT routed to RECOVERY_FOLLOWUP even if other follow-up conditions hold).
#
# Canonical list (6 terms). Update both consumers at the same time — never edit
# the classifier list or the router check independently.
EXERCISE_TERMS: list[str] = [
    "workout",
    "training",
    "lift",
    "program",
    "sets",
    "reps",
]
