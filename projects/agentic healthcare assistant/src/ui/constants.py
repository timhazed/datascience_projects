"""UI constants — pure data for Streamlit app (no Streamlit imports)."""

from __future__ import annotations

_APP_VERSION = "1.0.0"

# Built-in interactive test scenarios (Tab 5)
_SCENARIOS: dict[str, str] = {
    "CKD + Nephrologist": (
        "My 70-year-old father has chronic kidney disease. "
        "Book a nephrologist and summarize treatment options."
    ),
    "Hypertension Checkup": (
        "Retrieve Ramesh Kulkarni's history and update his medication to Telmisartan 80mg."
    ),
    "Diabetes Search": (
        "Find the latest diabetes management guidelines and book a follow-up for David Thompson."
    ),
}

# Example queries shown in the Chat onboarding banner (label, query)
_SAMPLE_QUERIES: list[tuple[str, str]] = [
    (
        "\U0001f50d CKD treatment options",
        "Summarize treatment options for chronic kidney disease.",
    ),
    (
        "\U0001f48a Hypertension guidelines",
        "What are the latest guidelines for hypertension management?",
    ),
    (
        "\U0001f4cb Diabetes care",
        "Find the latest diabetes management guidelines.",
    ),
]

_SPECIALTIES = ["Cardiology", "Endocrinology", "General Practice", "Nephrology", "Pulmonology"]
_HISTORY_FIELDS = ["conditions", "medications", "allergies", "notes", "summary"]

# Human-readable labels for completed_task.task values shown in the Metrics tab
_TASK_LABELS: dict[str, str] = {
    "resolve_patient": "Patient Lookup",
    "retrieve_history": "Retrieve History",
    "update_history": "Update History",
    "book_appointment": "Book Appointment",
    "search_disease": "Disease Search",
}

# Human-readable labels for ExperimentMetrics.experiment_name values
_EXP_LABELS: dict[str, str] = {
    "medical_history": "Medical History",
    "disease_search": "Disease Search",
    "appointment_booking": "Appointment Booking",
}

# Experiment label → (ExperimentMetrics.experiment_name, input paths or defaults)
_EXPERIMENT_LABELS = [
    "Appointment Booking (Exp 3)",
    "Disease Search (Exp 2)",
    "Medical History (Exp 1)",
]

_QUALITY_LEGEND = (
    "**Quality** measures how accurate and trustworthy the assistant's answers were.\n\n"
    "- **History & Patient tasks** — shows \u2713 Success or \u2717 Failure. "
    "These are deterministic: "
    "the record was either found and returned correctly or not.\n"
    "- **Disease Search** — shows a score from 0.0 to 1.0. "
    "It reflects how well the answer sticks to the retrieved sources. "
    "1.0 means every statement in the answer came from the sources. "
    "Below 0.80 means some statements may not be supported.\n"
    "- **Experiments** — formal offline benchmarks with fixed test cases. "
    "Run them manually to measure baseline performance."
)

_QUALITY_THRESHOLD = 0.80  # applies to scored operations (disease_search)
