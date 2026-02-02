import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import get_session
from app.db.models import Base
from app.main import app
from app.services.cache import cache_service

# Test database URL - using SQLite in memory for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session


@pytest_asyncio.fixture
async def test_app(test_session):
    """Create test FastAPI app with test database."""

    async def override_get_session():
        yield test_session

    app.dependency_overrides[get_session] = override_get_session

    # Disable cache for tests
    cache_service.redis = None

    yield app

    # Clean up
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def sample_data(test_session):
    """Create sample test data."""
    from app.db.models import District, Locality, Municipality

    # Create sample district
    district = District(name="Lisboa", code="11", population=2821697)
    test_session.add(district)
    await test_session.flush()

    # Create sample municipality
    municipality = Municipality(
        district_id=district.id, name="Lisboa", population=547631, area=100.05
    )
    test_session.add(municipality)
    await test_session.flush()

    # Create sample localities
    localities = [
        Locality(
            municipality_id=municipality.id,
            name="Alfama",
            feature_type="neighborhood",
            population=5000,
            latitude=38.7131,
            longitude=-9.1301,
        ),
        Locality(
            municipality_id=municipality.id,
            name="Belém",
            feature_type="neighborhood",
            population=8000,
            latitude=38.6979,
            longitude=-9.2064,
        ),
        Locality(
            municipality_id=municipality.id,
            name="Chiado",
            feature_type="neighborhood",
            population=3000,
            latitude=38.7106,
            longitude=-9.1423,
        ),
    ]

    for locality in localities:
        test_session.add(locality)

    await test_session.commit()

    return {
        "district": district,
        "municipality": municipality,
        "localities": localities,
    }
