from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Success[T]:
    """Represents a successful result with a value."""

    value: T

    def is_success(self) -> bool:
        return True

    def is_error(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def map[U](self, func: Callable[[T], U]) -> Result[U, Exception]:
        try:
            return Success(func(self.value))
        except Exception as exc:  # pragma: no cover - defensive
            return Error(exc)


@dataclass
class Error[E: Exception]:
    """Represents an error result with an exception."""

    error: E

    def is_success(self) -> bool:
        return False

    def is_error(self) -> bool:
        return True

    def unwrap(self) -> None:
        raise self.error

    def unwrap_or[T](self, default: T) -> T:
        return default

    def map(self, func: Callable[[Any], Any]) -> Result[Any, E]:
        return self


type Result[T, E: Exception] = Success[T] | Error[E]


def ok[T](value: T) -> Success[T]:
    """Create a successful result."""
    return Success(value)


def err[E: Exception](error: E) -> Error[E]:
    """Create an error result."""
    return Error(error)


def try_result[T](func: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """Decorator to convert function exceptions into Result patterns."""

    def wrapper(*args: Any, **kwargs: Any) -> Result[T, Exception]:
        try:
            return Success(func(*args, **kwargs))
        except Exception as exc:
            return Error(exc)

    return wrapper
