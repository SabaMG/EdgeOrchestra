"""Shared test fixtures: in-memory SQLite, fake Redis, FastAPI test app."""

import uuid
from collections.abc import AsyncGenerator

import fakeredis
import httpx
import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from orchestrator.db.models import Base


@pytest.fixture
async def db_engine():
    """In-memory SQLite async engine with UUID→CHAR(36) remapping."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    # SQLite doesn't have a UUID type — store as CHAR(36)
    # Also handle server_default=func.now() by using CURRENT_TIMESTAMP
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Async session bound to the test engine, rolled back after each test."""
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest.fixture
async def fake_redis():
    """Fake Redis instance, flushed between tests."""
    r = fakeredis.aioredis.FakeRedis()
    yield r
    await r.flushall()
    await r.aclose()


@pytest.fixture
async def app(db_session: AsyncSession):
    """FastAPI app with DB session overridden to use test SQLite."""
    from orchestrator.api.app import create_app
    from orchestrator.db.engine import get_session

    test_app = create_app()

    async def _override_get_session():
        yield db_session

    test_app.dependency_overrides[get_session] = _override_get_session
    return test_app


@pytest.fixture
async def client(app) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for the test FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
