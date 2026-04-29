"""ExperimentRunner — abstract base class for all three experiment runners.

Concrete runners:
  - experiments/search/search_experiment.py  (Phase 0 standalone → refactored Phase 1)
  - experiments/appointment/appointment_experiment.py  (Phase 4)
  - experiments/history/history_experiment.py  (Phase 4)
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path

from src.models.evaluation import ExperimentMetrics


class ExperimentRunner(ABC):
    """Abstract base for all three experiment runners.

    Subclasses implement run() and call _save_results() to persist output.
    _time_invoke() provides consistent latency measurement.
    """

    @abstractmethod
    def run(self) -> list[ExperimentMetrics]:
        """Execute the experiment and return one ExperimentMetrics per test case."""
        ...

    def _time_invoke(self, tool: object, inputs: dict) -> tuple[object, float]:
        """Invoke a tool and return (result, latency_ms).

        Args:
            tool: Any object with an .invoke(inputs) method.
            inputs: Dict of inputs to pass to the tool.

        Returns:
            Tuple of (tool result, wall-clock latency in milliseconds).
        """
        start = time.perf_counter()
        result = tool.invoke(inputs)  # type: ignore[attr-defined]
        return result, (time.perf_counter() - start) * 1000

    def _save_results(
        self,
        results: list[ExperimentMetrics],
        output_dir: str,
    ) -> None:
        """Persist results as JSON to the experiment's data/ subdirectory.

        Each ExperimentMetrics is written to its own file named
        {experiment_name}_{run_id}.json. No SQLite dependency.

        Args:
            results: List of ExperimentMetrics produced by run().
            output_dir: Directory path for output files (created if absent).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for r in results:
            path = Path(output_dir) / f"{r.experiment_name}_{r.run_id}.json"
            path.write_text(r.model_dump_json(indent=2))
