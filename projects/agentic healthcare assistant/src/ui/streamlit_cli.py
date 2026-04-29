"""Console entry for ``poetry run streamlit-app``; delegates to ``streamlit run``.

Calling ``src.ui.app:main`` directly bypasses Streamlit's runtime and triggers
``missing ScriptRunContext`` warnings. This module only spawns the real CLI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_py = Path(__file__).resolve().parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_py), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))
