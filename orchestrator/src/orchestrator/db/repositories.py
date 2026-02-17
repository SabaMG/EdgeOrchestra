import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.models import Device


class DeviceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: object) -> Device:
        device = Device(**kwargs)
        self.session.add(device)
        await self.session.commit()
        await self.session.refresh(device)
        return device

    async def get(self, device_id: uuid.UUID) -> Device | None:
        return await self.session.get(Device, device_id)

    async def list_all(self, status: str | None = None) -> list[Device]:
        stmt = select(Device)
        if status:
            stmt = stmt.where(Device.status == status)
        stmt = stmt.order_by(Device.registered_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, device_id: uuid.UUID, **kwargs: object) -> Device | None:
        device = await self.get(device_id)
        if not device:
            return None
        for key, value in kwargs.items():
            setattr(device, key, value)
        device.last_seen_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(device)
        return device

    async def delete(self, device_id: uuid.UUID) -> bool:
        device = await self.get(device_id)
        if not device:
            return False
        await self.session.delete(device)
        await self.session.commit()
        return True

    async def update_last_seen(self, device_id: uuid.UUID) -> None:
        device = await self.get(device_id)
        if device:
            device.last_seen_at = datetime.now(timezone.utc)
            await self.session.commit()

    async def update_metrics(self, device_id: uuid.UUID, metrics: dict) -> None:
        device = await self.get(device_id)
        if device:
            device.metrics = metrics
            device.last_seen_at = datetime.now(timezone.utc)
            await self.session.commit()
