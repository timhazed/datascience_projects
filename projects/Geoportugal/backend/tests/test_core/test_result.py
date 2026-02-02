import pytest

from app.core.result import Error, Success, err, ok, try_result


class TestSuccess:
    """Test cases for Success result type."""

    def test_success_creation(self):
        """Test creating a success result."""
        result = Success("test_value")
        assert result.value == "test_value"

    def test_success_is_success(self):
        """Test success identification."""
        result = Success(42)
        assert result.is_success() is True
        assert result.is_error() is False

    def test_success_unwrap(self):
        """Test unwrapping success value."""
        result = Success("unwrapped")
        assert result.unwrap() == "unwrapped"

    def test_success_unwrap_or(self):
        """Test unwrap_or returns value for success."""
        result = Success("original")
        assert result.unwrap_or("default") == "original"

    def test_success_map_success(self):
        """Test mapping success value with successful function."""
        result = Success(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_success()
        assert mapped.unwrap() == 10

    def test_success_map_error(self):
        """Test mapping success value with failing function."""
        result = Success(5)
        mapped = result.map(lambda x: 1 / 0)  # Division by zero
        assert mapped.is_error()
        assert isinstance(mapped.error, ZeroDivisionError)


class TestError:
    """Test cases for Error result type."""

    def test_error_creation(self):
        """Test creating an error result."""
        exception = ValueError("test error")
        result = Error(exception)
        assert result.error == exception

    def test_error_is_error(self):
        """Test error identification."""
        result = Error(ValueError("test"))
        assert result.is_success() is False
        assert result.is_error() is True

    def test_error_unwrap_raises(self):
        """Test unwrapping error raises the exception."""
        exception = RuntimeError("test error")
        result = Error(exception)
        with pytest.raises(RuntimeError, match="test error"):
            result.unwrap()

    def test_error_unwrap_or(self):
        """Test unwrap_or returns default for error."""
        result = Error(ValueError("failed"))
        assert result.unwrap_or("default_value") == "default_value"

    def test_error_map(self):
        """Test mapping error returns same error."""
        original_error = ValueError("original")
        result = Error(original_error)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_error()
        assert mapped.error == original_error


class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_ok_function(self):
        """Test ok() helper function."""
        result = ok("success_value")
        assert isinstance(result, Success)
        assert result.value == "success_value"

    def test_err_function(self):
        """Test err() helper function."""
        exception = ValueError("error_value")
        result = err(exception)
        assert isinstance(result, Error)
        assert result.error == exception

    def test_try_result_decorator_success(self):
        """Test try_result decorator with successful function."""

        @try_result
        def successful_function(x, y):
            return x + y

        result = successful_function(3, 4)
        assert result.is_success()
        assert result.unwrap() == 7

    def test_try_result_decorator_error(self):
        """Test try_result decorator with failing function."""

        @try_result
        def failing_function():
            raise ValueError("Function failed")

        result = failing_function()
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "Function failed"

    def test_try_result_decorator_with_args(self):
        """Test try_result decorator preserves function arguments."""

        @try_result
        def function_with_args(name, age=25):
            if age < 0:
                raise ValueError("Age cannot be negative")
            return f"{name} is {age} years old"

        # Test success case
        result = function_with_args("Alice", 30)
        assert result.is_success()
        assert result.unwrap() == "Alice is 30 years old"

        # Test error case
        result = function_with_args("Bob", -5)
        assert result.is_error()
        assert isinstance(result.error, ValueError)


class TestResultChaining:
    """Test cases for chaining operations."""

    def test_success_chain_map(self):
        """Test chaining map operations on success."""
        result = Success(10)
        final_result = result.map(lambda x: x * 2).map(lambda x: x + 5)
        assert final_result.is_success()
        assert final_result.unwrap() == 25

    def test_error_chain_map(self):
        """Test chaining map operations on error."""
        result = Error(ValueError("initial error"))
        final_result = result.map(lambda x: x * 2).map(lambda x: x + 5)
        assert final_result.is_error()
        assert isinstance(final_result.error, ValueError)
        assert str(final_result.error) == "initial error"

    def test_mixed_chain(self):
        """Test chain that starts successful but fails."""
        result = Success(5)
        final_result = result.map(lambda x: x * 2).map(lambda x: 1 / 0)  # Fails here
        assert final_result.is_error()
        assert isinstance(final_result.error, ZeroDivisionError)


class TestEdgeCases:
    """Test cases for edge cases and special values."""

    def test_success_with_none(self):
        """Test success with None value."""
        result = Success(None)
        assert result.is_success()
        assert result.unwrap() is None
        assert result.unwrap_or("default") is None

    def test_success_with_empty_string(self):
        """Test success with empty string."""
        result = Success("")
        assert result.is_success()
        assert result.unwrap() == ""

    def test_success_with_zero(self):
        """Test success with zero value."""
        result = Success(0)
        assert result.is_success()
        assert result.unwrap() == 0

    def test_error_with_custom_exception(self):
        """Test error with custom exception class."""

        class CustomError(Exception):
            pass

        custom_exception = CustomError("custom message")
        result = Error(custom_exception)
        assert result.is_error()
        assert isinstance(result.error, CustomError)

    def test_map_with_identity_function(self):
        """Test mapping with identity function."""
        result = Success(42)
        mapped = result.map(lambda x: x)
        assert mapped.is_success()
        assert mapped.unwrap() == 42

    def test_map_with_type_change(self):
        """Test mapping that changes the type."""
        result = Success(123)
        mapped = result.map(str)
        assert mapped.is_success()
        assert mapped.unwrap() == "123"
        assert isinstance(mapped.unwrap(), str)
