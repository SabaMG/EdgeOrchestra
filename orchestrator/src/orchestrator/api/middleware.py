import re
import time
from os import urandom

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from orchestrator.observability.metrics import HTTP_REQUEST_DURATION, HTTP_REQUESTS_TOTAL

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_SKIP_LOG_PREFIXES = ("/health", "/metrics", "/dashboard/static")


def _normalize_path(path: str) -> str:
    """Replace UUIDs in path segments with {id} for metric labels."""
    return _UUID_RE.sub("{id}", path)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = urandom(4).hex()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        method = request.method
        path = request.url.path
        status_code = str(response.status_code)
        normalized = _normalize_path(path)

        HTTP_REQUEST_DURATION.labels(method=method, path=normalized, status_code=status_code).observe(duration)
        HTTP_REQUESTS_TOTAL.labels(method=method, path=normalized, status_code=status_code).inc()

        if not any(path.startswith(p) for p in _SKIP_LOG_PREFIXES):
            logger = structlog.get_logger()
            logger.info(
                "http_request",
                method=method,
                path=path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 1),
            )

        response.headers["x-request-id"] = request_id
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/metrics"}
    PUBLIC_PREFIXES = ("/dashboard",)

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/" or path in self.PUBLIC_PATHS or any(
            path.startswith(p) for p in self.PUBLIC_PREFIXES
        ):
            return await call_next(request)
        if request.headers.get("x-api-key") != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)
