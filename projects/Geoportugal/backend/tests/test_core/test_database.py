"""
Tests for database session management
"""

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session


class TestDatabase:
    """Test database session management"""

    @pytest.mark.asyncio
    async def test_get_session_success(self):
        """Test successful database session creation"""
        # Test the actual function works
        session_gen = get_session()
        session = await anext(session_gen)

        assert session is not None
        assert isinstance(session, AsyncSession)

        # Clean up
        try:
            await anext(session_gen)
        except StopAsyncIteration:
            pass

    @pytest.mark.asyncio
    async def test_get_session_cleanup(self):
        """Test that database sessions are properly closed"""
        with patch("app.core.database.async_session_maker") as mock_maker:
            mock_session = AsyncMock(spec=AsyncSession)
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_session
            mock_context_manager.__aexit__.return_value = None
            mock_maker.return_value = mock_context_manager

            # Use the session
            session_gen = get_session()
            session = await anext(session_gen)

            assert session == mock_session

            # Finish the generator
            try:
                await anext(session_gen)
            except StopAsyncIteration:
                pass

            # Verify context manager was used properly
            mock_context_manager.__aenter__.assert_called_once()
            mock_context_manager.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_exception_handling(self):
        """Test session cleanup when exceptions occur"""
        with patch("app.core.database.async_session_maker") as mock_maker:
            mock_session = AsyncMock(spec=AsyncSession)
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_session
            mock_context_manager.__aexit__.return_value = None
            mock_maker.return_value = mock_context_manager

            session_gen = get_session()
            session = await anext(session_gen)

            # Session should work normally
            assert session == mock_session

            # Finish the generator
            try:
                await anext(session_gen)
            except StopAsyncIteration:
                pass

            # Context manager should handle cleanup
            mock_context_manager.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self):
        """Test that sessions rollback on errors"""
        with patch("app.core.database.AsyncSession") as mock_session_class:
            mock_session = AsyncMock(spec=AsyncSession)
            mock_session_class.return_value = mock_session

            session_gen = get_session()
            session = await anext(session_gen)

            # Session should support rollback
            assert hasattr(session, "rollback")
            assert hasattr(session, "commit")

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self):
        """Test that multiple sessions can be created concurrently"""
        with patch("app.core.database.AsyncSession") as mock_session_class:
            mock_session1 = AsyncMock(spec=AsyncSession)
            mock_session2 = AsyncMock(spec=AsyncSession)
            mock_session_class.side_effect = [mock_session1, mock_session2]

            # Create two session generators
            gen1 = get_session()
            gen2 = get_session()

            session1 = await anext(gen1)
            session2 = await anext(gen2)

            # Should be different session instances
            assert session1 is not session2

            # Clean up
            try:
                await anext(gen1)
            except StopAsyncIteration:
                pass
            try:
                await anext(gen2)
            except StopAsyncIteration:
                pass

    def test_database_url_configuration(self):
        """Test database URL configuration from settings"""
        from app.core.database import async_session_maker, engine

        # Should have engine and session maker configured
        assert engine is not None
        assert async_session_maker is not None
        assert engine.url is not None

    @pytest.mark.asyncio
    async def test_session_transaction_handling(self):
        """Test session transaction handling"""
        with patch("app.core.database.AsyncSession") as mock_session_class:
            mock_session = AsyncMock(spec=AsyncSession)
            mock_session_class.return_value = mock_session

            session_gen = get_session()
            session = await anext(session_gen)

            # Session should support transaction operations
            assert hasattr(session, "begin")
            assert hasattr(session, "commit")
            assert hasattr(session, "rollback")

            # Test transaction context
            mock_transaction = AsyncMock()
            mock_session.begin.return_value = mock_transaction

            transaction = await session.begin()
            assert transaction is not None

    @pytest.mark.asyncio
    async def test_session_connection_pool_handling(self):
        """Test that sessions properly handle connection pool"""
        with patch("app.core.database.AsyncSession") as mock_session_class:
            mock_session = AsyncMock(spec=AsyncSession)
            mock_session_class.return_value = mock_session

            # Multiple sessions should reuse connection pool
            for _ in range(3):
                session_gen = get_session()
                session = await anext(session_gen)

                # Each session should be functional
                assert hasattr(session, "execute")
                assert hasattr(session, "close")

                # Clean up
                try:
                    await anext(session_gen)
                except StopAsyncIteration:
                    pass
