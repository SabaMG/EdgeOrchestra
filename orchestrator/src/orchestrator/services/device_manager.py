import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.models import Device
from orchestrator.db.repositories import DeviceRepository

logger = structlog.get_logger()


class DeviceManager:
    def __init__(self, session: AsyncSession) -> None:
        self.repo = DeviceRepository(session)

    async def register_device(
        self,
        name: str,
        device_model: str,
        os_version: str,
        chip: str | None = None,
        memory_bytes: int | None = None,
        cpu_cores: int | None = None,
        gpu_cores: int | None = None,
        neural_engine_cores: int | None = None,
        battery_level: float | None = None,
        battery_state: str | None = None,
        metrics: dict | None = None,
    ) -> Device:
        device = await self.repo.create(
            name=name,
            device_model=device_model,
            os_version=os_version,
            chip=chip,
            memory_bytes=memory_bytes,
            cpu_cores=cpu_cores,
            gpu_cores=gpu_cores,
            neural_engine_cores=neural_engine_cores,
            battery_level=battery_level,
            battery_state=battery_state,
            status="online",
            metrics=metrics,
        )
        logger.info("device_registered", device_id=str(device.id), name=name)
        return device

    async def unregister_device(self, device_id: uuid.UUID) -> bool:
        deleted = await self.repo.delete(device_id)
        if deleted:
            logger.info("device_unregistered", device_id=str(device_id))
        return deleted

    async def mark_offline(self, device_id: uuid.UUID) -> None:
        await self.repo.update(device_id, status="offline")
        logger.info("device_marked_offline", device_id=str(device_id))

    async def mark_online(self, device_id: uuid.UUID) -> None:
        await self.repo.update(device_id, status="online")

    async def update_metrics(self, device_id: uuid.UUID, metrics: dict) -> None:
        await self.repo.update_metrics(device_id, metrics)
