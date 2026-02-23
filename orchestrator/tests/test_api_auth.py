"""Tests for API key authentication middleware."""

import httpx
import pytest

from orchestrator.api.app import create_app
from orchestrator.api.middleware import ApiKeyMiddleware
from orchestrator.config import Settings

TEST_API_KEY = "test-secret-key"


@pytest.fixture
def auth_app():
    """FastAPI app with API key middleware enabled."""
    app = create_app()
    app.add_middleware(ApiKeyMiddleware, api_key=TEST_API_KEY)
    return app


@pytest.fixture
async def auth_client(auth_app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=auth_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestApiKeyMiddleware:
    async def test_api_key_required(self, auth_client: httpx.AsyncClient):
        resp = await auth_client.get("/api/v1/devices")
        assert resp.status_code == 401
        assert "API key" in resp.json()["detail"]

    async def test_api_key_valid(self, auth_client: httpx.AsyncClient):
        resp = await auth_client.get(
            "/api/v1/devices",
            headers={"x-api-key": TEST_API_KEY},
        )
        # 200 or 500 (no DB) — but NOT 401
        assert resp.status_code != 401

    async def test_api_key_wrong(self, auth_client: httpx.AsyncClient):
        resp = await auth_client.get(
            "/api/v1/devices",
            headers={"x-api-key": "wrong-key"},
        )
        assert resp.status_code == 401

    async def test_health_no_auth(self, auth_client: httpx.AsyncClient):
        resp = await auth_client.get("/health")
        assert resp.status_code != 401

    async def test_dashboard_no_auth(self, auth_client: httpx.AsyncClient):
        resp = await auth_client.get("/dashboard")
        assert resp.status_code == 200

    async def test_root_redirect_no_auth(self, auth_client: httpx.AsyncClient):
        resp = await auth_client.get("/", follow_redirects=False)
        assert resp.status_code == 307


class TestNoMiddlewareWhenKeyEmpty:
    async def test_no_auth_when_key_empty(self, client: httpx.AsyncClient):
        """When api_key is empty, the middleware is not added — no 401."""
        resp = await client.get("/api/v1/devices")
        assert resp.status_code != 401
