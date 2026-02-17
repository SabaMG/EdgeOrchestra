import asyncio
import json
import uuid
from datetime import datetime, timezone

import structlog
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.config import settings
from orchestrator.db.engine import async_session
from orchestrator.db.repositories import DeviceRepository

logger = structlog.get_logger()


class HeartbeatMonitor:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis
        self.timeout_seconds = (
            settings.heartbeat_interval_seconds * settings.heartbeat_timeout_multiplier
        )

    async def process_heartbeat(
        self, session: AsyncSession, device_id: uuid.UUID, metrics: dict,
        battery_level: float | None = None, battery_state: str | None = None,
    ) -> None:
        key = f"heartbeat:{device_id}"
        await self.redis.set(key, datetime.now(timezone.utc).isoformat(), ex=self.timeout_seconds)

        repo = DeviceRepository(session)
        update_kwargs: dict = {"status": "online"}
        if battery_level is not None:
            update_kwargs["battery_level"] = battery_level
        if battery_state is not None:
            update_kwargs["battery_state"] = battery_state
        await repo.update(device_id, **update_kwargs)
        if metrics:
            await repo.update_metrics(device_id, metrics)

    async def get_pending_command(self, device_id: str) -> dict | None:
        key = f"command:{device_id}"
        raw = await self.redis.lpop(key)
        if raw:
            return json.loads(raw)
        return None

    async def queue_command(self, device_id: str, command: dict) -> None:
        key = f"command:{device_id}"
        await self.redis.rpush(key, json.dumps(command))

    async def run_stale_device_checker(self) -> None:
        logger.info("stale_device_checker_started", timeout=self.timeout_seconds)
        while True:
            try:
                await self._check_stale_devices()
            except Exception:
                logger.exception("stale_device_check_error")
            await asyncio.sleep(settings.heartbeat_interval_seconds)

    async def _check_stale_devices(self) -> None:
        async with async_session() as session:
            repo = DeviceRepository(session)
            devices = await repo.list_all(status="online")
            for device in devices:
                key = f"heartbeat:{device.id}"
                last_heartbeat = await self.redis.get(key)
                if last_heartbeat is None:
                    # No heartbeat key means device hasn't sent one or it expired
                    now = datetime.now(timezone.utc)
                    if device.last_seen_at:
                        last_seen = device.last_seen_at.replace(tzinfo=timezone.utc) if device.last_seen_at.tzinfo is None else device.last_seen_at
                        elapsed = (now - last_seen).total_seconds()
                        if elapsed > self.timeout_seconds:
                            await repo.update(device.id, status="offline")
                            logger.info(
                                "device_marked_offline",
                                device_id=str(device.id),
                                elapsed=elapsed,
                            )
