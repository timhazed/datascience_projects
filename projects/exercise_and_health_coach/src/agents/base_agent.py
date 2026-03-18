import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from src.exceptions import AgentExecutionError
from src.validators.base import ValidationResult, ValidatorChain

logger = logging.getLogger(__name__)

# Type variable for output schema
T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base agent with structured output and retry logic.

    Features:
    - Uses `with_structured_output()` for reliable JSON parsing
    - Configurable retry logic (default: 3 attempts)
    - Optional validator chain integration
    - Standardized error handling
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_retries: int = 3,
        validators: ValidatorChain | None = None,
    ):
        """
        Initialize the agent.

        Args:
            llm: LangChain chat model instance.
            max_retries: Maximum retry attempts on failure.
            validators: Optional validator chain to run on output.
        """
        self.llm = llm
        self.max_retries = max_retries
        self.validators = validators
        self._structured_llm: BaseChatModel | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> type[T]:
        """Pydantic model for structured output."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining agent behavior."""
        pass

    @property
    def structured_llm(self) -> BaseChatModel:
        """Get LLM configured for structured output."""
        if self._structured_llm is None:
            self._structured_llm = self.llm.with_structured_output(
                self.output_schema,
                method="json_schema",
            )
        return self._structured_llm

    def build_prompt(self, **kwargs: Any) -> list[SystemMessage | HumanMessage]:
        """
        Build the prompt messages.

        Override this method to customize prompt construction.

        Args:
            **kwargs: Variables to inject into prompts.

        Returns:
            List of messages for the LLM.
        """
        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.format_user_message(**kwargs)),
        ]

    @abstractmethod
    def format_user_message(self, **kwargs: Any) -> str:
        """
        Format the user message from input data.

        Args:
            **kwargs: Input data for the agent.

        Returns:
            Formatted user message string.
        """
        pass

    def invoke(self, **kwargs: Any) -> T:
        """
        Execute the agent with retry logic.

        Args:
            **kwargs: Input data for the agent.

        Returns:
            Validated output matching the output schema.

        Raises:
            AgentExecutionError: If all retries fail.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"[{self.name}] Attempt {attempt}/{self.max_retries}")

                # Build and execute prompt
                messages = self.build_prompt(**kwargs)

                result = self.structured_llm.invoke(messages)  

                # Validate output if validators are configured
                if self.validators is not None:
                    validation_result = self.validators.validate(result)
                    if not validation_result.is_valid:
                        logger.warning(
                            f"[{self.name}] Validation failed: {validation_result.errors}"
                        )
                        # Store validation errors for potential retry
                        last_error = ValueError(
                            f"Validation failed: {validation_result.errors}"
                        )
                        continue

                logger.info(f"[{self.name}] Successfully generated output")
                return result

            except ValidationError as e:
                logger.warning(f"[{self.name}] Pydantic validation error: {e}")
                last_error = e

            except Exception as e:
                logger.warning(f"[{self.name}] Attempt {attempt} failed: {e}")
                last_error = e

        # All retries exhausted
        raise AgentExecutionError(
            agent_name=self.name,
            message=f"Failed after {self.max_retries} attempts",
            original_error=last_error,
        )

    async def ainvoke(self, **kwargs: Any) -> T:
        """
        Async version of invoke.

        Args:
            **kwargs: Input data for the agent.

        Returns:
            Validated output matching the output schema.

        Raises:
            AgentExecutionError: If all retries fail.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"[{self.name}] Async attempt {attempt}/{self.max_retries}")

                messages = self.build_prompt(**kwargs)
                result = await self.structured_llm.ainvoke(messages)  

                if self.validators is not None:
                    validation_result = self.validators.validate(result)
                    if not validation_result.is_valid:
                        last_error = ValueError(
                            f"Validation failed: {validation_result.errors}"
                        )
                        continue

                logger.info(f"[{self.name}] Successfully generated output (async)")
                return result

            except ValidationError as e:
                logger.warning(f"[{self.name}] Pydantic validation error: {e}")
                last_error = e

            except Exception as e:
                logger.warning(f"[{self.name}] Async attempt {attempt} failed: {e}")
                last_error = e

        raise AgentExecutionError(
            agent_name=self.name,
            message=f"Failed after {self.max_retries} attempts (async)",
            original_error=last_error,
        )

    def validate_output(self, output: T) -> ValidationResult:
        """
        Validate output using configured validators.

        Args:
            output: The output to validate.

        Returns:
            ValidationResult with any errors/warnings.
        """
        if self.validators is None:
            return ValidationResult(is_valid=True)
        return self.validators.validate(output)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, retries={self.max_retries})"
