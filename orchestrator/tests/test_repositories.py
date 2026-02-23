"""Tests for DeviceRepository and TrainingJobRepository CRUD operations."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.models import Device, TrainingJob
from orchestrator.db.repositories import DeviceRepository, TrainingJobRepository


def _device_kwargs(**overrides) -> dict:
    defaults = {
        "name": "Test iPhone",
        "device_model": "iPhone 15 Pro",
        "os_version": "17.0",
        "chip": "A17 Pro",
        "status": "online",
    }
    defaults.update(overrides)
    return defaults


class TestDeviceRepository:
    async def test_create_device(self, db_session: AsyncSession):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs())

        assert device.id is not None
        assert device.name == "Test iPhone"
        assert device.status == "online"

    async def test_get_device(self, db_session: AsyncSession):
        repo = DeviceRepository(db_session)
        created = await repo.create(**_device_kwargs())
        fetched = await repo.get(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == created.name

    async def test_get_device_not_found(self, db_session: AsyncSession):
        repo = DeviceRepository(db_session)
        result = await repo.get(uuid.uuid4())
        assert result is None

    async def test_list_all_devices(self, db_session: AsyncSession):
        repo = DeviceRepository(db_session)
        for i in range(3):
            await repo.create(**_device_kwargs(name=f"Device {i}"))

        devices = await repo.list_all()
        assert len(devices) == 3

    async def test_list_filter_by_status(self, db_session: AsyncSession):
        repo = DeviceRepository(db_session)
        await repo.create(**_device_kwargs(name="D1", status="online"))
        await repo.create(**_device_kwargs(name="D2", status="online"))
        await repo.create(**_device_kwargs(name="D3", status="offline"))

        online = await repo.list_all(status="online")
        assert len(online) == 2

        offline = await repo.list_all(status="offline")
        assert len(offline) == 1

    async def test_update_device(self, db_session: AsyncSession):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs())

        updated = await repo.update(device.id, status="offline")
        assert updated is not None
        assert updated.status == "offline"

    async def test_delete_device(self, db_session: AsyncSession):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs())

        result = await repo.delete(device.id)
        assert result is True

        fetched = await repo.get(device.id)
        assert fetched is None

    async def test_delete_device_not_found(self, db_session: AsyncSession):
        repo = DeviceRepository(db_session)
        result = await repo.delete(uuid.uuid4())
        assert result is False


class TestTrainingJobRepository:
    async def test_create_job(self, db_session: AsyncSession):
        repo = TrainingJobRepository(db_session)
        job = await repo.create(num_rounds=10, min_devices=2, learning_rate=0.01)

        assert job.id is not None
        assert job.status == "pending"
        assert job.current_round == 0
        assert job.num_rounds == 10

    async def test_list_filter_by_status(self, db_session: AsyncSession):
        repo = TrainingJobRepository(db_session)
        await repo.create(num_rounds=5)
        job2 = await repo.create(num_rounds=10)
        await repo.update(job2.id, status="completed")

        pending = await repo.list_all(status="pending")
        assert len(pending) == 1

        completed = await repo.list_all(status="completed")
        assert len(completed) == 1

    async def test_update_job(self, db_session: AsyncSession):
        repo = TrainingJobRepository(db_session)
        job = await repo.create(num_rounds=5)

        updated = await repo.update(
            job.id, status="running", round_metrics={"round_1": {"loss": 0.5}}
        )
        assert updated is not None
        assert updated.status == "running"
        assert updated.round_metrics == {"round_1": {"loss": 0.5}}
