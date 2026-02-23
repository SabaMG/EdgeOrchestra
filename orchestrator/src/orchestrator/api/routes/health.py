import time

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text

from orchestrator.config import settings
from orchestrator.db.engine import engine
from orchestrator.observability.metrics import DEVICES_BY_STATUS

logger = structlog.get_logger()

router = APIRouter()


async def _check_db() -> dict:
    start = time.perf_counter()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok", "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def _check_redis() -> dict:
    start = time.perf_counter()
    try:
        r = Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        try:
            await r.ping()
        finally:
            await r.aclose()
        return {"status": "ok", "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


async def _refresh_device_gauge() -> None:
    """Best-effort update of device gauge from DB."""
    try:
        from orchestrator.db.engine import async_session
        from orchestrator.db.repositories import DeviceRepository

        async with async_session() as session:
            repo = DeviceRepository(session)
            for status in ("online", "offline", "training"):
                devices = await repo.list_all(status=status)
                DEVICES_BY_STATUS.labels(status=status).set(len(devices))
    except Exception:
        pass


@router.get("/health")
async def health():
    db = await _check_db()
    redis = await _check_redis()
    await _refresh_device_gauge()

    overall = "ok" if db["status"] == "ok" and redis["status"] == "ok" else "degraded"
    status_code = 200 if overall == "ok" else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall,
            "dependencies": {
                "database": db,
                "redis": redis,
            },
        },
    )
