import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.models import Device, Model, TrainingJob


class ModelRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: object) -> Model:
        model = Model(**kwargs)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def get(self, model_id: uuid.UUID) -> Model | None:
        return await self.session.get(Model, model_id)

    async def list_all(self, architecture: str | None = None) -> list[Model]:
        stmt = select(Model)
        if architecture:
            stmt = stmt.where(Model.architecture == architecture)
        stmt = stmt.order_by(Model.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, model_id: uuid.UUID, **kwargs: object) -> Model | None:
        model = await self.get(model_id)
        if not model:
            return None
        for key, value in kwargs.items():
            setattr(model, key, value)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def delete(self, model_id: uuid.UUID) -> bool:
        model = await self.get(model_id)
        if not model:
            return False
        await self.session.delete(model)
        await self.session.commit()
        return True


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


class TrainingJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, **kwargs: object) -> TrainingJob:
        job = TrainingJob(**kwargs)
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def get(self, job_id: uuid.UUID) -> TrainingJob | None:
        return await self.session.get(TrainingJob, job_id)

    async def list_all(self, status: str | None = None) -> list[TrainingJob]:
        stmt = select(TrainingJob)
        if status:
            stmt = stmt.where(TrainingJob.status == status)
        stmt = stmt.order_by(TrainingJob.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, job_id: uuid.UUID, **kwargs: object) -> TrainingJob | None:
        job = await self.get(job_id)
        if not job:
            return None
        for key, value in kwargs.items():
            setattr(job, key, value)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def delete(self, job_id: uuid.UUID) -> bool:
        job = await self.get(job_id)
        if not job:
            return False
        await self.session.delete(job)
        await self.session.commit()
        return True
