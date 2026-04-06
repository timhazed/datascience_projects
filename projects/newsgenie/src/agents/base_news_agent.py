"""Abstract base class defining the fetch() interface for all domain news agents."""

from abc import ABC, abstractmethod

from src.data.agent_result import NewsAgentResult


class BaseNewsAgent(ABC):
    """Abstract interface for all domain news agents.

    Each concrete subclass fetches articles for a specific news category and
    returns a fully normalized NewsAgentResult. Agents are stateless per-call
    objects; their HTTP clients live as module-level singletons in src/api/.
    """

    @abstractmethod
    def fetch(self, query: str) -> NewsAgentResult:
        """Fetch and normalize articles for the given query string.

        Args:
            query: Sub-query string produced by the supervisor node.

        Returns:
            NewsAgentResult with success=True and populated articles on success,
            or success=False with error_message on any failure.
        """
