"""Tests for the /api/v1/devices REST routes."""

import uuid

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.db.repositories import DeviceRepository


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


class TestDevicesAPI:
    async def test_list_devices_empty(self, client: httpx.AsyncClient):
        resp = await client.get("/api/v1/devices")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_devices_with_data(
        self, client: httpx.AsyncClient, db_session: AsyncSession
    ):
        repo = DeviceRepository(db_session)
        await repo.create(**_device_kwargs())

        resp = await client.get("/api/v1/devices")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test iPhone"

    async def test_get_device(
        self, client: httpx.AsyncClient, db_session: AsyncSession
    ):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs())

        resp = await client.get(f"/api/v1/devices/{device.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == str(device.id)

    async def test_get_device_not_found(self, client: httpx.AsyncClient):
        resp = await client.get(f"/api/v1/devices/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_delete_device(
        self, client: httpx.AsyncClient, db_session: AsyncSession
    ):
        repo = DeviceRepository(db_session)
        device = await repo.create(**_device_kwargs())

        resp = await client.delete(f"/api/v1/devices/{device.id}")
        assert resp.status_code == 204

        resp = await client.get(f"/api/v1/devices/{device.id}")
        assert resp.status_code == 404

    async def test_get_device_metrics(
        self, client: httpx.AsyncClient, db_session: AsyncSession
    ):
        repo = DeviceRepository(db_session)
        device = await repo.create(
            **_device_kwargs(battery_level=0.85, battery_state="charging")
        )

        resp = await client.get(f"/api/v1/devices/{device.id}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["device_id"] == str(device.id)
        assert data["battery_level"] == 0.85
        assert data["battery_state"] == "charging"
