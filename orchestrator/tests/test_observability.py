"""Tests for observability: metrics endpoint, health check, request logging middleware."""

from unittest.mock import AsyncMock, patch

import pytest
from prometheus_client import REGISTRY


class TestMetricsEndpoint:
    async def test_metrics_returns_prometheus_format(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "eo_" in resp.text

    async def test_metrics_no_auth_required(self, client):
        """Metrics endpoint should be accessible without API key."""
        resp = await client.get("/metrics")
        assert resp.status_code == 200


class TestHealthCheck:
    async def test_health_ok(self, client):
        with (
            patch("orchestrator.api.routes.health._check_db", new_callable=AsyncMock) as mock_db,
            patch("orchestrator.api.routes.health._check_redis", new_callable=AsyncMock) as mock_redis,
            patch("orchestrator.api.routes.health._refresh_device_gauge", new_callable=AsyncMock),
        ):
            mock_db.return_value = {"status": "ok", "latency_ms": 1.0}
            mock_redis.return_value = {"status": "ok", "latency_ms": 1.0}

            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "database" in data["dependencies"]
            assert "redis" in data["dependencies"]

    async def test_health_degraded_when_redis_down(self, client):
        with (
            patch("orchestrator.api.routes.health._check_db", new_callable=AsyncMock) as mock_db,
            patch("orchestrator.api.routes.health._check_redis", new_callable=AsyncMock) as mock_redis,
            patch("orchestrator.api.routes.health._refresh_device_gauge", new_callable=AsyncMock),
        ):
            mock_db.return_value = {"status": "ok", "latency_ms": 1.0}
            mock_redis.return_value = {"status": "error", "error": "Connection refused"}

            resp = await client.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "degraded"

    async def test_health_backward_compat_status_key(self, client):
        """The 'status' key should always be at the root level."""
        with (
            patch("orchestrator.api.routes.health._check_db", new_callable=AsyncMock) as mock_db,
            patch("orchestrator.api.routes.health._check_redis", new_callable=AsyncMock) as mock_redis,
            patch("orchestrator.api.routes.health._refresh_device_gauge", new_callable=AsyncMock),
        ):
            mock_db.return_value = {"status": "ok", "latency_ms": 1.0}
            mock_redis.return_value = {"status": "ok", "latency_ms": 1.0}

            resp = await client.get("/health")
            data = resp.json()
            assert "status" in data


class TestRequestLoggingMiddleware:
    async def test_request_id_header_present(self, client):
        resp = await client.get("/metrics")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) == 8

    async def test_metrics_incremented_after_request(self, client):
        # Make a request to a non-skipped path
        await client.get("/docs")

        # Check that prometheus metrics are populated
        from orchestrator.observability.metrics import HTTP_REQUESTS_TOTAL

        # Verify the counter has been incremented (at least one sample should exist)
        metric_value = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/docs", status_code="200")
        assert metric_value._value.get() >= 1
