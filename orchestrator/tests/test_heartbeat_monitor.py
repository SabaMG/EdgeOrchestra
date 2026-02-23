"""Tests for HeartbeatMonitor: Redis heartbeat, command queue, device status."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import patch

from orchestrator.db.repositories import DeviceRepository
from orchestrator.services.heartbeat_monitor import HeartbeatMonitor


def _device_kwargs(**overrides) -> dict:
    defaults = {
        "name": "Test iPhone",
        "device_model": "iPhone 15 Pro",
        "os_version": "17.0",
        "status": "offline",
    }
    defaults.update(overrides)
    return defaults


class TestHeartbeatMonitor:
    @pytest.fixture
    def monitor(self, fake_redis):
        with patch("orchestrator.services.heartbeat_monitor.settings") as mock_settings:
            mock_settings.heartbeat_interval_seconds = 5
            mock_settings.heartbeat_timeout_multiplier = 3
            yield HeartbeatMonitor(fake_redis)

    async def test_process_heartbeat_sets_redis_key(
        self, monitor: HeartbeatMonitor, fake_redis, db_session: AsyncSession
    ):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs())

        await monitor.process_heartbeat(db_session, device.id, metrics={})

        key = f"heartbeat:{device.id}"
        value = await fake_redis.get(key)
        assert value is not None

        ttl = await fake_redis.ttl(key)
        assert ttl > 0

    async def test_process_heartbeat_updates_device_status(
        self, monitor: HeartbeatMonitor, fake_redis, db_session: AsyncSession
    ):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs(status="offline"))
        assert device.status == "offline"

        await monitor.process_heartbeat(db_session, device.id, metrics={})

        updated = await repo.get(device.id)
        assert updated is not None
        assert updated.status == "online"

    async def test_queue_and_get_command(
        self, monitor: HeartbeatMonitor, fake_redis
    ):
        device_id = str(uuid.uuid4())
        command = {"action": "start_training", "job_id": "abc123"}

        await monitor.queue_command(device_id, command)
        result = await monitor.get_pending_command(device_id)

        assert result == command

    async def test_get_command_empty(self, monitor: HeartbeatMonitor, fake_redis):
        device_id = str(uuid.uuid4())
        result = await monitor.get_pending_command(device_id)
        assert result is None

    async def test_command_queue_fifo(self, monitor: HeartbeatMonitor, fake_redis):
        device_id = str(uuid.uuid4())
        cmd1 = {"action": "start_training"}
        cmd2 = {"action": "stop_training"}

        await monitor.queue_command(device_id, cmd1)
        await monitor.queue_command(device_id, cmd2)

        first = await monitor.get_pending_command(device_id)
        assert first == cmd1

        second = await monitor.get_pending_command(device_id)
        assert second == cmd2

    async def test_process_heartbeat_stores_is_low_power_mode(
        self, monitor: HeartbeatMonitor, fake_redis, db_session: AsyncSession
    ):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs())

        metrics = {"cpu_usage": 0.3, "memory_usage": 0.5, "thermal_pressure": 0.2}
        await monitor.process_heartbeat(
            db_session, device.id, metrics,
            battery_level=0.8, battery_state="discharging",
            is_low_power_mode=True,
        )

        updated = await repo.get(device.id)
        assert updated is not None
        assert updated.metrics["is_low_power_mode"] is True

    async def test_heartbeat_does_not_override_training_status(
        self, monitor: HeartbeatMonitor, fake_redis, db_session: AsyncSession
    ):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs(status="training"))

        await monitor.process_heartbeat(db_session, device.id, metrics={})

        updated = await repo.get(device.id)
        assert updated is not None
        assert updated.status == "training"
