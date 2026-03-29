class AgentExecutionError(Exception):
    """Errors from agent execution."""

    def __init__(self, agent_name: str, message: str, original_error: Exception | None = None):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}' error: {message}")
