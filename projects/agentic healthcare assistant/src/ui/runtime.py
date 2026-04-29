"""Import-time settings and session identity for the Streamlit UI."""

from __future__ import annotations

import uuid

from dotenv import load_dotenv

from src.config.settings import load_settings

load_dotenv()
settings = load_settings()

# One UUID per process — identifies this app startup in the session_log table.
# Streamlit reruns the script on every interaction but the module is imported once,
# so this value is stable for the lifetime of the server process.
_SESSION_ID = str(uuid.uuid4())
